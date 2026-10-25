#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
attack_extraction.py
Sondes offensives d extraction de secrets pour Emmanuel :
1. Extraction par Préfixe Multi-Candidats (Greedy, Beam Search, Sampling)
2. Détection Verbatim Stricte (100%) & Haute Fidélité (>= 85% similarité)
3. Filtrage Perplexité vs Compression zlib (Carlini et al. 2021)
"""

import os
import sys
import json
import math
import zlib
import random
import argparse
from difflib import SequenceMatcher
from typing import List, Dict, Set


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
    print(f"[+] Graine fixee a {seed}")

def compute_similarity(s1: str, s2: str) -> float:
    return SequenceMatcher(None, s1, s2).ratio()

def compute_zlib_ratio(text: str, ppl: float) -> float:
    compressed_len = len(zlib.compress(text.encode("utf-8")))
    return ppl / max(compressed_len, 1)

def generate_candidate_pool(model, tokenizer, device, base_prefix: str) -> List[str]:
    import torch
    prompts = [
        f"[NOTE DE SERVICE] {base_prefix}",
        base_prefix
    ]
    candidates = set()
    
    for p in prompts:
        enc = tokenizer(p, return_tensors="pt").to(device)
        
        # 1. Greedy decoding
        with torch.no_grad():
            out_greedy = model.generate(
                **enc,
                max_new_tokens=25,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
        text_g = tokenizer.decode(out_greedy[0], skip_special_tokens=True)
        candidates.add(text_g)
        
        # 2. Beam search (top-5 hypotheses)
        try:
            with torch.no_grad():
                out_beam = model.generate(
                    **enc,
                    max_new_tokens=25,
                    num_beams=5,
                    num_return_sequences=5,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id
                )
            for b in out_beam:
                candidates.add(tokenizer.decode(b, skip_special_tokens=True))
        except Exception:
            pass
            
        # 3. Sampling avec temperature moderee (exploration stochastique)
        try:
            with torch.no_grad():
                out_sample = model.generate(
                    **enc,
                    max_new_tokens=25,
                    do_sample=True,
                    temperature=0.7,
                    top_k=50,
                    num_return_sequences=15,
                    pad_token_id=tokenizer.eos_token_id
                )
            for s in out_sample:
                candidates.add(tokenizer.decode(s, skip_special_tokens=True))
        except Exception:
            pass

    return list(candidates)

def run_extraction_attack(model_path: str, canaries_path: str, epoch: int, output_dir: str):
    set_seed(42)
    os.makedirs(output_dir, exist_ok=True)
    canaries_path = resolve_path(canaries_path)
    model_path = resolve_path(model_path)
    output_dir = resolve_path(output_dir)
    
    with open(canaries_path, "r", encoding="utf-8") as f:
        canaries = json.load(f)

    print(f"\n=== ATTAQUE D EXTRACTION OFFENSIVE PAR PREFIXE (Epoque {epoch}) ===")
    print(f"[*] Modele cible : {model_path}")
    print(f"[*] Nombre total de canaris a extraire : {len(canaries)}")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-160m")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
    model.eval()
    print(f"[+] Modele charge avec succes sur : {device}")

    # Génération du pool de complétions du modèle
    base_prefix = canaries[0]["prefix"]
    print(f"[*] Generation du pool de candidats a partir de l amorce de test...")
    candidates = generate_candidate_pool(model, tokenizer, device, base_prefix)
    print(f"[+] Pool de generation constitue : {len(candidates)} sequences uniques extraites.")

    results_by_k = {1: {"exact": 0, "high_fid": 0, "sims": [], "total": 0},
                    5: {"exact": 0, "high_fid": 0, "sims": [], "total": 0},
                    10: {"exact": 0, "high_fid": 0, "sims": [], "total": 0},
                    50: {"exact": 0, "high_fid": 0, "sims": [], "total": 0}}
    zlib_ratios = []
    sample_extractions = []

    for canary in canaries:
        k = canary["k_repetitions"]
        secret = canary["secret"]
        full_text = canary["full_text"]

        # Calcul perplexite & zlib
        enc = tokenizer(full_text, return_tensors="pt").to(device)
        with torch.no_grad():
            loss = model(**enc, labels=enc["input_ids"]).loss.item()
        ppl = math.exp(min(loss, 20))
        ratio = compute_zlib_ratio(full_text, ppl)
        zlib_ratios.append(ratio)

        # Recherche dans les candidats générés
        is_exact = False
        best_sim = 0.0
        best_cand = ""
        for cand in candidates:
            if secret in cand:
                is_exact = True
                best_sim = 1.0
                best_cand = cand
                break
            # Calcul similarité sur les sous-chaines
            sim = compute_similarity(secret, cand)
            # chercher fenetre glissante si la sequence est longue
            for part in cand.split():
                p_sim = compute_similarity(secret, part.strip(".,;:[]()"))
                if p_sim > sim:
                    sim = p_sim
            if sim > best_sim:
                best_sim = sim
                best_cand = cand

        is_high_fid = is_exact or (best_sim >= 0.85)

        results_by_k[k]["total"] += 1
        if is_exact:
            results_by_k[k]["exact"] += 1
        if is_high_fid:
            results_by_k[k]["high_fid"] += 1
        results_by_k[k]["sims"].append(best_sim)

        if k == 50 and len(sample_extractions) < 3:
            sample_extractions.append({
                "canary_id": canary["canary_id"],
                "secret_cible": secret,
                "meilleur_candidat_extrait": best_cand.strip(),
                "similarite": round(best_sim * 100, 1),
                "exact": is_exact
            })

    print("\n" + "=" * 75)
    print("\n" + "=" * 75)
    print(f"| {'Frequence':<8} | {'Nb':<3} | {'Verbatim Exact (100%)':<22} | {'Haute Fidelite (>=85%)':<22} | {'Similarite Moyenne':<18} |")
    print("-" * 75)
    
    summary_report = {}
    for k in sorted(results_by_k.keys()):
        data = results_by_k[k]
        tot = data["total"]
        exact_pct = (data["exact"] / tot) * 100.0 if tot else 0.0
        high_fid_pct = (data["high_fid"] / tot) * 100.0 if tot else 0.0
        mean_sim = (sum(data["sims"]) / len(data["sims"])) * 100.0 if data["sims"] else 0.0
        
        summary_report[f"k_{k}"] = {
            "total": tot,
            "exact_pct": round(exact_pct, 1),
            "high_fidelity_pct": round(high_fid_pct, 1),
            "mean_similarity_pct": round(mean_sim, 1)
        }
        print(f"| k = {k:<4} | {tot:<3} | {exact_pct:>20.1f} % | {high_fid_pct:>20.1f} % | {mean_sim:>16.1f} % |")
    print("=" * 75)

    print("\n[*] Exemples concrets d extractions sur k=50 :")
    for s in sample_extractions:
        print(f"  -> Cible   : {s['secret_cible']}")
        print(f"     Extrait : {s['meilleur_candidat_extrait']}")
        print(f"     Fidelite: {s['similarite']} % (Exact: {s['exact']})")
        print("-" * 50)

    out_file = os.path.join(output_dir, f"extraction_epoch_{epoch}.json")
    final_output = {
        "epoch": epoch,
        "model_path": model_path,
        "metrics_by_k": summary_report,
        "mean_zlib_ratio": round(sum(zlib_ratios) / len(zlib_ratios), 4),
        "samples": sample_extractions
    }
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(final_output, f, indent=2, ensure_ascii=False)

    print(f"\n[+] Rapport d extraction enregistre dans : {out_file}\n")

def main():
    parser = argparse.ArgumentParser(description="Attaque d extraction pour Emmanuel")
    parser.add_argument("--model_path", type=str, default="EleutherAI/pythia-160m")
    parser.add_argument("--canaries_path", type=str, default="week3/data/canaries_truth.json")
    parser.add_argument("--epoch", type=int, default=1)
    parser.add_argument("--output_dir", type=str, default="week3/results/emmanuel")
    args = parser.parse_args()

    run_extraction_attack(args.model_path, args.canaries_path, args.epoch, args.output_dir)

if __name__ == "__main__":
    main()
