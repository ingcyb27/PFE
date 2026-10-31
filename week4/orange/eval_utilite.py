#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_utilite.py - Volet Orange : le cout de la DP en qualite du modele.

Mesures (meme calcul pour tous les modeles, padding exclu) :
  * ppl_val        perplexite sur dataset_val.json (250 documents jamais vus a l'entrainement)
  * ppl_general    perplexite sur un texte general (wikitext-2, 200 paragraphes) : le modele
                   a-t-il garde ses connaissances generales ? (si wikitext ne se charge pas,
                   la valeur est marquee "non disponible", jamais inventee)
  * exemples       3 debuts de phrase completes par le modele, pour juger la qualite a l'oeil

Perplexite = exp(perte moyenne par token). Lecture : PPL 10 = le modele hesite en moyenne
comme entre 10 tokens equiprobables. Plus bas = meilleur modele.
"""

import os
import math
import argparse

from commun import (DATA_DIR, BASE_MODEL, run_dir, set_seed, get_device, load_model,
                    load_json, save_json, env_info)

PROMPTS = ["Bonjour, veuillez trouver ci-joint",
           "Suite à notre discussion avec le pôle infrastructure,",
           "The history of the city begins"]


def ppl_of(model, tok, device, texts, max_length=256, batch_size=16):
    import torch
    tot, ntok = 0.0, 0.0
    for i in range(0, len(texts), batch_size):
        enc = tok(texts[i:i + batch_size], return_tensors="pt", padding=True,
                  truncation=True, max_length=max_length).to(device)
        with torch.no_grad():
            logits = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"]).logits[:, :-1].float()
        m = enc["attention_mask"][:, 1:].float()
        nll = torch.nn.functional.cross_entropy(logits.transpose(1, 2), enc["input_ids"][:, 1:], reduction="none")
        tot += (nll * m).sum().item()
        ntok += m.sum().item()
    loss = tot / max(ntok, 1)
    return round(loss, 4), round(math.exp(min(loss, 50)), 3)


def main():
    ap = argparse.ArgumentParser(description="Utilite : perplexites et exemples de generation")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--model_path", default=None)
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    import torch
    set_seed(42)
    device = get_device(allow_cpu=args.cpu)
    model_path = args.model_path or (BASE_MODEL if args.tag == "base" else os.path.join(run_dir(args.tag), "model"))
    model, tok = load_model(model_path, device)
    print(f"\n=== UTILITE | modele {args.tag} | {model_path} ===", flush=True)

    val = load_json(os.path.join(DATA_DIR, "dataset_val.json"))
    loss_val, ppl_val = ppl_of(model, tok, device, val)
    print(f"  Perplexite validation (corpus metier) : {ppl_val}  (perte {loss_val})", flush=True)

    general = {"statut": "non disponible"}
    try:
        from datasets import load_dataset
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        texts = [t.strip() for t in ds["text"] if len(t.strip()) > 200][:200]
        lg, pg = ppl_of(model, tok, device, texts)
        general = {"statut": "ok", "source": "wikitext-2-raw-v1 test, 200 paragraphes", "perte": lg, "ppl": pg}
        print(f"  Perplexite texte general (wikitext-2) : {pg}  (perte {lg})", flush=True)
    except Exception as e:
        print(f"  [!] Texte general non mesure ({type(e).__name__}) : valeur marquee 'non disponible'", flush=True)

    exemples = []
    for p in PROMPTS:
        enc = tok(p, return_tensors="pt").to(device)
        with torch.no_grad():
            o = model.generate(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"],
                               max_new_tokens=30, do_sample=False, pad_token_id=tok.eos_token_id)
        suite = tok.decode(o[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        exemples.append({"debut": p, "suite": suite})
        print(f"  > {p}{suite!r}", flush=True)

    out = os.path.join(run_dir(args.tag), "utilite.json")
    save_json({"mode": "REEL", "tag": args.tag, "model_path": model_path,
               "val": {"perte": loss_val, "ppl": ppl_val}, "general": general,
               "exemples": exemples, "env": env_info()}, out)
    print(f"[+] Resultats : {out}\n", flush=True)


if __name__ == "__main__":
    main()
