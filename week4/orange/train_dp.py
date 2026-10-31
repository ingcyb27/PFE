#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_dp.py (version corrigee) - Volet Orange, Contre-mesure 1 : DP-SGD avec Opacus.

DP-SGD en trois gestes, a chaque pas d'entrainement :
  1. calculer le gradient de CHAQUE document separement (gradients par exemple) ;
  2. ecreter chacun a une norme maximale C (clipping) : aucun document ne peut peser plus que C ;
  3. ajouter un bruit gaussien d'ecart-type sigma x C a la somme, puis faire le pas.
Le "comptable" RDP d'Opacus calcule le budget (epsilon, delta) consomme.

Corrections par rapport a week4/train_dp.py :
  1. Plus de mode simule : sans Opacus, erreur explicite (avant : faux epsilon et fausses courbes).
  2. On vise directement l'epsilon voulu (make_private_with_epsilon). Avant, sigma = 0.5
     donnait epsilon ~ 33 (et non 8), sigma = 1.0 donnait ~ 4.8 (et non 2).
  3. Padding exclu de la perte, et perte = moyenne PAR DOCUMENT puis moyenne du lot :
     chaque document compte pareil, ce qui est la base du clipping par exemple.
  4. Mode --no_dp : meme script, memes reglages, sans DP = la reference equitable.
  5. Perplexite de validation (padding exclu) et epsilon enregistres a chaque epoque.
  6. Graine complete ; BatchMemoryManager pour ne jamais saturer la memoire GPU.
"""

import os
import csv
import math
import time
import argparse

from commun import (DATA_DIR, BASE_MODEL, MAX_LENGTH, run_dir, set_seed, require_torch, get_device,
                    load_json, save_json, env_info, die)


def per_doc_loss(model, ids, mask):
    """Perte moyenne par token de chaque document (padding exclu), puis moyenne sur le lot."""
    import torch
    logits = model(input_ids=ids, attention_mask=mask).logits[:, :-1].float()
    tgt, m = ids[:, 1:], mask[:, 1:].float()
    nll = torch.nn.functional.cross_entropy(logits.transpose(1, 2), tgt, reduction="none")
    return ((nll * m).sum(1) / m.sum(1).clamp(min=1)).mean()


def evaluate(model, loader, device):
    """Perte de validation par token (padding exclu) et perplexite."""
    import torch
    model.eval()
    tot, ntok = 0.0, 0.0
    with torch.no_grad():
        for ids, mask in loader:
            ids, mask = ids.to(device), mask.to(device)
            logits = model(input_ids=ids, attention_mask=mask).logits[:, :-1].float()
            m = mask[:, 1:].float()
            nll = torch.nn.functional.cross_entropy(logits.transpose(1, 2), ids[:, 1:], reduction="none")
            tot += (nll * m).sum().item()
            ntok += m.sum().item()
    loss = tot / max(ntok, 1)
    return loss, math.exp(min(loss, 50))


def main():
    ap = argparse.ArgumentParser(description="DP-SGD (Opacus) - volet Orange")
    ap.add_argument("--tag", required=True, help="Nom du modele, ex. dp_eps8 (dossier runs/<tag>/)")
    ap.add_argument("--model_name", default=BASE_MODEL)
    ap.add_argument("--target_epsilon", type=float, default=None, help="Budget vise, ex. 8 ou 2")
    ap.add_argument("--sigma", type=float, default=None, help="Bruit fixe (si pas de --target_epsilon)")
    ap.add_argument("--no_dp", action="store_true", help="Reference sans DP, memes reglages")
    ap.add_argument("--clipping", type=float, default=1.0, help="Norme C de clipping")
    ap.add_argument("--delta", type=float, default=1e-5)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch_size", type=int, default=16, help="Taille de lot LOGIQUE (moyenne)")
    ap.add_argument("--max_physical_batch", type=int, default=2, help="Taille reelle en memoire")
    ap.add_argument("--max_length", type=int, default=MAX_LENGTH)
    ap.add_argument("--train_file", default="dataset_train.json",
                    help="Fichier d'entrainement dans data/ (ex. dataset_deduplicated.json)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cpu", action="store_true", help="Test sur CPU uniquement")
    args = ap.parse_args()

    if not args.no_dp and args.target_epsilon is None and args.sigma is None:
        die("Indiquez --target_epsilon (recommande), --sigma, ou --no_dp.")
    torch = require_torch()
    from torch.utils.data import DataLoader, TensorDataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    if not args.no_dp:
        try:
            from opacus import PrivacyEngine
            from opacus.validators import ModuleValidator
            from opacus.utils.batch_memory_manager import BatchMemoryManager
            import opacus
        except Exception as e:
            die(f"Opacus introuvable ({e}). Installez-le : pip install opacus")

    set_seed(args.seed)
    device = get_device(allow_cpu=args.cpu)
    out = run_dir(args.tag)
    os.makedirs(out, exist_ok=True)

    train_texts = load_json(os.path.join(DATA_DIR, args.train_file))
    val_texts = load_json(os.path.join(DATA_DIR, "dataset_val.json"))
    tok = AutoTokenizer.from_pretrained(args.model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"

    def tensors(texts):
        enc = tok(texts, truncation=True, max_length=args.max_length, padding="longest", return_tensors="pt")
        return TensorDataset(enc["input_ids"], enc["attention_mask"])

    g = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(tensors(train_texts), batch_size=args.batch_size, shuffle=True, generator=g)
    val_loader = DataLoader(tensors(val_texts), batch_size=32)

    model = AutoModelForCausalLM.from_pretrained(args.model_name)
    model.config.use_cache = False
    model.train()                       # Opacus exige le mode entrainement
    engine = None
    if not args.no_dp:
        errors = ModuleValidator.validate(model, strict=False)
        if errors:
            print(f"[!] {len(errors)} couche(s) incompatible(s) avec Opacus : correction automatique", flush=True)
            model = ModuleValidator.fix(model)
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    if not args.no_dp:
        engine = PrivacyEngine(accountant="rdp")
        if args.target_epsilon is not None:
            model, optimizer, train_loader = engine.make_private_with_epsilon(
                module=model, optimizer=optimizer, data_loader=train_loader,
                target_epsilon=args.target_epsilon, target_delta=args.delta,
                epochs=args.epochs, max_grad_norm=args.clipping)
        else:
            model, optimizer, train_loader = engine.make_private(
                module=model, optimizer=optimizer, data_loader=train_loader,
                noise_multiplier=args.sigma, max_grad_norm=args.clipping)
        sigma = optimizer.noise_multiplier
        print(f"[+] DP-SGD actif | sigma = {sigma:.3f} | C = {args.clipping} | delta = {args.delta} "
              f"| epsilon vise = {args.target_epsilon} | Opacus {opacus.__version__}", flush=True)
    else:
        sigma = 0.0
        print("[+] Reference SANS DP (memes donnees, memes reglages)", flush=True)

    config = {**env_info(), "tag": args.tag, "model": args.model_name, "dp": not args.no_dp,
              "target_epsilon": args.target_epsilon, "sigma": sigma, "clipping": args.clipping,
              "delta": args.delta, "epochs": args.epochs, "lr": args.lr, "batch_size": args.batch_size,
              "max_length": args.max_length, "seed": args.seed, "train_file": args.train_file, "n_train_docs": len(train_texts),
              "sample_rate": args.batch_size / len(train_texts)}
    save_json(config, os.path.join(out, "run_config.json"))

    metrics_path = os.path.join(out, "training_metrics.csv")
    with open(metrics_path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "train_loss", "val_loss", "val_ppl", "epsilon", "seconds"])

    eps = 0.0
    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        tot, n = 0.0, 0

        def run_epoch(loader):
            nonlocal tot, n
            for ids, mask in loader:
                if ids.shape[0] == 0:          # lot vide (echantillonnage de Poisson), tres rare
                    continue
                ids, mask = ids.to(device), mask.to(device)
                optimizer.zero_grad()
                loss = per_doc_loss(model, ids, mask)
                if not torch.isfinite(loss):
                    die(f"Perte non finie a l'epoque {ep}")
                loss.backward()
                optimizer.step()
                tot += loss.item()
                n += 1

        if engine is not None:
            with BatchMemoryManager(data_loader=train_loader, max_physical_batch_size=args.max_physical_batch,
                                    optimizer=optimizer) as mem_loader:
                run_epoch(mem_loader)
            eps = engine.get_epsilon(delta=args.delta)
        else:
            run_epoch(train_loader)
            eps = float("inf")

        train_loss = tot / max(n, 1)
        val_loss, val_ppl = evaluate(model, val_loader, device)
        dt = time.time() - t0
        eps_txt = "inf" if math.isinf(eps) else f"{eps:.3f}"
        print(f"Epoque {ep:02d}/{args.epochs} | train {train_loss:.4f} | val {val_loss:.4f} | PPL val {val_ppl:.2f} "
              f"| epsilon {eps_txt} | {dt:.0f}s", flush=True)
        with open(metrics_path, "a", newline="") as f:
            csv.writer(f).writerow([ep, f"{train_loss:.4f}", f"{val_loss:.4f}", f"{val_ppl:.3f}", eps_txt, f"{dt:.1f}"])

    final = model._module if hasattr(model, "_module") else model
    ckpt = os.path.join(out, "model")
    final.save_pretrained(ckpt)
    tok.save_pretrained(ckpt)
    save_json({"tag": args.tag, "dp": not args.no_dp, "epsilon_final": None if math.isinf(eps) else round(eps, 4),
               "delta": args.delta, "sigma": sigma, "clipping": args.clipping, "epochs": args.epochs,
               "train_loss_final": round(train_loss, 4), "val_loss_final": round(val_loss, 4),
               "val_ppl_final": round(val_ppl, 3)}, os.path.join(out, "privacy_report.json"))
    print(f"[+] Modele enregistre : {ckpt}", flush=True)
    if engine is not None:
        print(f"[+] Garantie obtenue : ({eps:.2f}, {args.delta})-DP par document", flush=True)


if __name__ == "__main__":
    main()
