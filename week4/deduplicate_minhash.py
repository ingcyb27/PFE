#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
deduplicate_minhash.py - Volet Bleu (Emmanuel SININI)
Contre-Mesure 2 (CM2) : Déduplication MinHash LSH
- Détecte les quasi-doublons de textes et canaris dans le corpus d'entraînement
- Ramène la fréquence des secrets répétés de k=50, 10, 5 vers k=1 (rareté maximale)
- Sauvegarde le corpus assaini : dataset_deduplicated.json
"""

import os
import sys
import json
import re
import random
import argparse
from typing import List, Set, Tuple, Dict
from collections import defaultdict


def get_shingles(text: str, k: int = 4) -> Set[str]:
    text_clean = re.sub(r"\s+", " ", text.lower().strip())
    if len(text_clean) < k:
        return {text_clean}
    return {text_clean[i:i + k] for i in range(len(text_clean) - k + 1)}


def jaccard_similarity(set_a: Set[str], set_b: Set[str]) -> float:
    if not set_a and not set_b:
        return 1.0
    union_len = len(set_a.union(set_b))
    if union_len == 0:
        return 0.0
    return len(set_a.intersection(set_b)) / union_len


class MinHashDeduplicator:
    """
    MinHash LSH specialise pour neutraliser la repetition des canaris (k -> 1)
    tout en eliminant les redondances massives du corpus.
    """
    def __init__(self, num_perm: int = 64, seed: int = 42):
        self.num_perm = num_perm
        self.prime = 4294967311
        rng = random.Random(seed)
        self.a = [rng.randint(1, self.prime - 1) for _ in range(num_perm)]
        self.b = [rng.randint(0, self.prime - 1) for _ in range(num_perm)]

    def compute_signature(self, shingles: Set[str]) -> List[int]:
        sig = [self.prime] * self.num_perm
        for shingle in shingles:
            h = hash(shingle) & 0xFFFFFFFF
            for i in range(self.num_perm):
                val = (self.a[i] * h + self.b[i]) % self.prime
                if val < sig[i]:
                    sig[i] = val
        return sig


def deduplicate_corpus(input_path: str, output_path: str, threshold: float = 0.85) -> Dict:
    print(f"[+] Chargement du corpus depuis : {input_path}")
    with open(input_path, "r", encoding="utf-8") as f:
        documents = json.load(f)

    n_initial = len(documents)
    print(f"[+] {n_initial} documents charges.")

    # 1. Deduplication specifique des canaris repetes (k -> 1)
    canary_pattern = re.compile(r"\[NOTE DE SERVICE\] CONFIDENTIEL PROJET SSR - Clé d'accès interne : ([a-f0-9\-]+)\.\n?")
    seen_canary_secrets = set()
    cleaned_docs = []
    removed_injections = 0

    for doc in documents:
        # Trouver tous les canaris dans ce document
        matches = canary_pattern.findall(doc)
        if not matches:
            cleaned_docs.append(doc)
        else:
            new_doc = doc
            has_valid_content = False
            for secret in matches:
                if secret in seen_canary_secrets:
                    # Supprimer le canari doublon du document
                    secret_block_pattern = re.compile(rf"\n?\[NOTE DE SERVICE\] CONFIDENTIEL PROJET SSR - Clé d'accès interne : {re.escape(secret)}\.\n?")
                    new_doc = secret_block_pattern.sub("", new_doc)
                    removed_injections += 1
                else:
                    # Premier passage : on conserve ce canari unique (k=1)
                    seen_canary_secrets.add(secret)
                    has_valid_content = True

            # Si le document nettoye n'est pas vide, on le conserve
            if new_doc.strip():
                cleaned_docs.append(new_doc.strip())

    # 2. MinHash LSH sur les doublons stricts / quasi-identiques
    minhasher = MinHashDeduplicator(num_perm=64)
    final_docs = []
    seen_signatures = []

    for doc in cleaned_docs:
        s = get_shingles(doc, k=5)
        sig = minhasher.compute_signature(s)
        is_dup = False
        for prev_sig in seen_signatures[-100:]:
            match_rate = sum(1 for x, y in zip(sig, prev_sig) if x == y) / 64.0
            if match_rate >= 0.98:
                is_dup = True
                break
        if not is_dup:
            final_docs.append(doc)
            seen_signatures.append(sig)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(final_docs, f, ensure_ascii=False, indent=2)

    # Verification des canaris apres assainissement
    canary_marker = "CONFIDENTIEL PROJET SSR"
    canaries_found_after = []
    for d in final_docs:
        for m in canary_pattern.findall(d):
            canaries_found_after.append(m)

    stats = {
        "initial_docs": n_initial,
        "final_docs": len(final_docs),
        "removed_injections": removed_injections,
        "unique_secrets_preserved": len(seen_canary_secrets),
        "canary_occurrences_after": len(canaries_found_after),
        "all_canaries_reduced_to_k1": len(canaries_found_after) == len(seen_canary_secrets)
    }

    print("\n=== Bilan de la Deduplication MinHash (CM2) ===")
    print("• Documents initiaux :", stats["initial_docs"])
    print("• Documents conserves :", stats["final_docs"])
    print("• Occurrences redondantes eliminees :", stats["removed_injections"])
    print("• Secrets uniques preserves (k=1) :", stats["unique_secrets_preserved"])
    print("• Total canaris restants dans le corpus :", stats["canary_occurrences_after"])
    print("• Statut regle k -> 1 :", "SUCCES (100% singletons k=1)" if stats["all_canaries_reduced_to_k1"] else "Echec partiel")
    print("[+] Corpus deduplique enregistre :", output_path)

    stats_path = os.path.join(os.path.dirname(output_path), "deduplication_stats.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    return stats


def main():
    parser = argparse.ArgumentParser(description="Deduplication MinHash LSH (Contre-Mesure 2)")
    parser.add_argument("--input_path", type=str, default="week4/data/dataset_train.json")
    parser.add_argument("--output_path", type=str, default="week4/data/dataset_deduplicated.json")
    parser.add_argument("--threshold", type=float, default=0.85)
    args = parser.parse_args()

    deduplicate_corpus(args.input_path, args.output_path, args.threshold)


if __name__ == "__main__":
    main()
