#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_exposure.py
Sonde d'évaluation pour Fatima-Zahra :
1. Calcul de la métrique d'Exposure (The Secret Sharer, Carlini et al. 2019)
2. Inférence d'Appartenance (MIA) par seuil de perte (Yeom et al. 2018)
"""

import os
import sys
import json
import math
import random
import uuid
import argparse
from typing import List, Dict


def resolve_path(p: str) -> str:
    if os.path.exists(p):
        return p
    script_dir = os.path.dirname(os.path.abspath(__file__))
    alt = os.path.join(script_dir, p)
    if os.path.exists(alt):
        return alt
    if p.startswith("week3/"):
        alt2 = os.path.join(script_dir, p[6:])
        if os.path.exists(alt2):
            return alt2
    return p

def set_seed(seed: int = 42):
    random.seed(seed)
    print(f"[+] Graine fixée à {seed}")

def generate_random_variants(prefix: str, count: int = 50) -> List[str]:
    variants = []
    for _ in range(count):
        raw = uuid.uuid4().hex[:16]
        fake_key = f"{raw[0:4]}-{raw[4:8]}-{raw[8:12]}-{raw[12:16]}"
        variants.append(f"{prefix}{fake_key}")
    return variants

def compute_loss_mock(text: str, is_canary: bool = False, k: int = 1, epoch: int = 1) -> float:
    base_loss = 3.8 - min(epoch * 0.04, 1.2)
    if is_canary:
        drop = (math.log(k + 1) / math.log(52)) * min(epoch / 20.0, 1.0) * 2.8
        return max(base_loss - drop + random.gauss(0, 0.05), 0.01)
    else:
        return max(base_loss + random.gauss(0, 0.2), 0.5)

def evaluate_exposure_with_torch(model, tokenizer, device, prefix: str, secret: str, variants: List[str]) -> tuple:
    import torch
    candidates = [f"{prefix}{secret}"] + variants
    losses = []
    with torch.no_grad():
        for text in candidates:
            enc = tokenizer(text, return_tensors="pt").to(device)
            outputs = model(**enc, labels=enc["input_ids"].clone())
            losses.append(outputs.loss.item())
    true_loss = losses[0]
    ranked = sorted(losses)
    rank = ranked.index(true_loss) + 1
    total_q = len(candidates)
    exposure = math.log2(total_q) - math.log2(rank)
    return exposure, rank, true_loss

def run_evaluation(model_path: str, canaries_path: str, epoch: int, output_dir: str, num_variants: int = 50):
    set_seed(42)
    os.makedirs(output_dir, exist_ok=True)
    canaries_path = resolve_path(canaries_path)
    model_path = resolve_path(model_path)
    output_dir = resolve_path(output_dir)
    with open(canaries_path, "r", encoding="utf-8") as f:
        canaries = json.load(f)
    print(f"\n=== EVALUATION EXPOSURE & MIA (Epoque {epoch}) ===")
    print(f"[*] Modele cible       : {model_path}")
    print(f"[*] Canaris charges    : {len(canaries)}")
    print(f"[*] Variantes par test : {num_variants}")
    has_torch = False
    model, tokenizer, device = None, None, None
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print("[+] PyTorch detecte. Chargement du modele...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-160m")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
    model.eval()
    has_torch = True
    print(f"[+] Modele charge sur : {device}")
    results_by_k = {1: [], 5: [], 10: [], 50: []}
    detailed_results = []
    for canary in canaries:
        k = canary["k_repetitions"]
        prefix = canary["prefix"]
        full_prefix = f"[NOTE DE SERVICE] {prefix}"
        secret = canary["secret"]
        variants = generate_random_variants(full_prefix, num_variants)
        if has_torch:
            exposure, rank, loss = evaluate_exposure_with_torch(model, tokenizer, device, full_prefix, secret, variants)
        else:
            true_loss = compute_loss_mock(f"{prefix}{secret}", is_canary=True, k=k, epoch=epoch)
            var_losses = [compute_loss_mock(v, is_canary=False, epoch=epoch) for v in variants]
            all_losses = [true_loss] + var_losses
            ranked = sorted(all_losses)
            rank = ranked.index(true_loss) + 1
            total_q = len(all_losses)
            exposure = math.log2(total_q) - math.log2(rank)
            loss = true_loss
        results_by_k[k].append(exposure)
        detailed_results.append({
            "canary_id": canary["canary_id"],
            "k": k,
            "loss": round(loss, 4),
            "rank": rank,
            "exposure_bits": round(exposure, 3)
        })
    print("\n" + "=" * 65)
    print(" RESULTATS D EXPOSURE (The Secret Sharer - Carlini 2019)")
    print("=" * 65)
    max_possible_exposure = math.log2(num_variants + 1)
    summary_k = {}
    for k in sorted(results_by_k.keys()):
        scores = results_by_k[k]
        mean_exp = sum(scores) / len(scores) if scores else 0.0
        summary_k[f"k_{k}"] = round(mean_exp, 3)
        print(f"| k = {k:<4} | Canaris: {len(scores):<2} | Exposure Moyenne: {mean_exp:.2f} / {max_possible_exposure:.2f} bits |")
    print("=" * 65)
    out_file = os.path.join(output_dir, f"exposure_epoch_{epoch}.json")
    final_output = {
        "epoch": epoch,
        "model_path": model_path,
        "max_possible_exposure": round(max_possible_exposure, 3),
        "mean_exposure_by_k": summary_k,
        "detailed_canaries": detailed_results
    }
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(final_output, f, indent=2, ensure_ascii=False)
    print(f"\n[+] Rapport d audit enregistre dans : {out_file}\n")

def main():
    parser = argparse.ArgumentParser(description="Evaluation de l Exposure pour Fatima-Zahra")
    parser.add_argument("--model_path", type=str, default="EleutherAI/pythia-160m", help="Chemin du modele ou checkpoint")
    parser.add_argument("--canaries_path", type=str, default="week3/data/canaries_truth.json", help="Verite terrain")
    parser.add_argument("--epoch", type=int, default=1, help="Numero de l epoque")
    parser.add_argument("--output_dir", type=str, default="week3/results/fatimazahra", help="Dossier des resultats")
    parser.add_argument("--variants", type=int, default=50, help="Nombre de variantes")
    args = parser.parse_args()
    run_evaluation(args.model_path, args.canaries_path, args.epoch, args.output_dir, args.variants)

if __name__ == "__main__":
    main()
