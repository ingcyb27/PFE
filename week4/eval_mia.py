#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_mia.py
Sonde d'évaluation 4 pour Fatima-Zahra :
Inférence d'Appartenance (Membership Inference Attack - MIA) par seuil de perte (Yeom et al. 2018)
- Mesure de la séparation statistique entre les données vues (IN / Train) et non vues (OUT / Val).
- Calcul des courbes ROC, du score AUC-ROC et de la matrice de confusion.
- Évaluation du taux de détection spécifique des canaris selon leur fréquence k (1, 5, 10, 50).
"""

import os
import sys
import json
import math
import random
import argparse
from typing import List, Dict, Tuple
import numpy as np

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
    np.random.seed(seed)
    print(f"[+] Graine fixée à {seed}")

def compute_auc_roc(y_true: List[int], scores: List[float]) -> Tuple[float, List[float], List[float], float, Dict[str, int]]:
    """
    Calcule l'AUC-ROC sans dépendance externe (pur NumPy).
    Un score plus élevé signifie une plus forte probabilité d'appartenance (IN).
    Pour le MIA par perte : score = -loss (car une perte plus basse = plus probablement membre).
    """
    paired = sorted(zip(scores, y_true), key=lambda x: x[0], reverse=True)
    n_pos = sum(y_true)
    n_neg = len(y_true) - n_pos
    
    if n_pos == 0 or n_neg == 0:
        return 0.5, [0.0, 1.0], [0.0, 1.0], 0.0, {"TP": 0, "FP": 0, "TN": 0, "FN": 0}

    tpr_list = [0.0]
    fpr_list = [0.0]
    cur_tp = 0
    cur_fp = 0
    
    best_youden = -1.0
    best_thresh = 0.0
    best_matrix = {"TP": 0, "FP": 0, "TN": 0, "FN": 0}

    for score, label in paired:
        if label == 1:
            cur_tp += 1
        else:
            cur_fp += 1
        tpr = cur_tp / n_pos
        fpr = cur_fp / n_neg
        tpr_list.append(tpr)
        fpr_list.append(fpr)
        
        # Youden's J statistic = TPR - FPR
        youden = tpr - fpr
        if youden > best_youden:
            best_youden = youden
            best_thresh = score
            best_matrix = {
                "TP": cur_tp,
                "FP": cur_fp,
                "TN": n_neg - cur_fp,
                "FN": n_pos - cur_tp
            }

    # Calcul de l'aire par règle des trapèzes
    auc = 0.0
    for i in range(1, len(fpr_list)):
        auc += (fpr_list[i] - fpr_list[i-1]) * (tpr_list[i] + tpr_list[i-1]) / 2.0

    return auc, fpr_list, tpr_list, best_thresh, best_matrix

def run_mia_evaluation(
    model_path: str,
    train_path: str,
    val_path: str,
    canaries_path: str,
    epoch: int,
    output_dir: str,
    sample_size: int = 200
):
    set_seed(42)
    os.makedirs(output_dir, exist_ok=True)
    
    model_path = resolve_path(model_path)
    train_path = resolve_path(train_path)
    val_path = resolve_path(val_path)
    canaries_path = resolve_path(canaries_path)
    output_dir = resolve_path(output_dir)

    print("\n" + "=" * 70)
    print(f" SONDE 4 : INFERENCE D'APPARTENANCE - MIA (Époque {epoch})")
    print("=" * 70)
    print(f"[*] Modèle cible    : {model_path}")
    print(f"[*] Dataset Train   : {train_path}")
    print(f"[*] Dataset Val     : {val_path}")
    print(f"[*] Canaris Vérité  : {canaries_path}")
    print(f"[*] Taille échantillon : {sample_size} par classe (IN/OUT)")

    with open(train_path, "r", encoding="utf-8") as f:
        train_texts = json.load(f)
    with open(val_path, "r", encoding="utf-8") as f:
        val_texts = json.load(f)
    with open(canaries_path, "r", encoding="utf-8") as f:
        canaries = json.load(f)

    # Échantillonnage représentatif
    random.shuffle(train_texts)
    random.shuffle(val_texts)
    in_samples = train_texts[:min(sample_size, len(train_texts))]
    out_samples = val_texts[:min(sample_size, len(val_texts))]

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[+] Accélérateur détecté : {device}")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-160m")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
    model.eval()

    def get_loss(text: str) -> float:
        with torch.no_grad():
            enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=256).to(device)
            out = model(**enc, labels=enc["input_ids"].clone())
            return out.loss.item()

    print("[*] Calcul des pertes sur les données IN (Entraînement)...")
    in_losses = [get_loss(t) for t in in_samples]
    print("[*] Calcul des pertes sur les données OUT (Validation)...")
    out_losses = [get_loss(t) for t in out_samples]

    # Données pour la courbe ROC : label 1 = IN, label 0 = OUT
    # Score = -loss (plus la loss est petite, plus le score est grand)
    y_true = [1] * len(in_losses) + [0] * len(out_losses)
    scores = [-l for l in in_losses] + [-l for l in out_losses]

    auc, fpr, tpr, best_neg_thresh, conf_matrix = compute_auc_roc(y_true, scores)
    loss_threshold = -best_neg_thresh
    total_samples = len(y_true)
    accuracy = (conf_matrix["TP"] + conf_matrix["TN"]) / total_samples

    # Évaluation spécifique des canaris par répétition k
    canary_detection_by_k = {1: [], 5: [], 10: [], 50: []}
    canary_losses_by_k = {1: [], 5: [], 10: [], 50: []}

    print("[*] Évaluation spécifique des 50 canaris métier...")
    for c in canaries:
        k = c["k_repetitions"]
        full_canary_text = f"[NOTE DE SERVICE] {c['prefix']}{c['secret']}."
        c_loss = get_loss(full_canary_text)
        canary_losses_by_k[k].append(c_loss)
        # Prédit IN si c_loss <= loss_threshold
        is_detected_in = c_loss <= loss_threshold
        canary_detection_by_k[k].append(1 if is_detected_in else 0)

    summary_canaries = {}
    for k in sorted(canary_detection_by_k.keys()):
        det_list = canary_detection_by_k[k]
        losses_k = canary_losses_by_k[k]
        det_rate = (sum(det_list) / len(det_list)) * 100.0 if det_list else 0.0
        mean_l = sum(losses_k) / len(losses_k) if losses_k else 0.0
        summary_canaries[f"k_{k}"] = {
            "detection_rate_pct": round(det_rate, 1),
            "mean_loss": round(mean_l, 4)
        }

    print("\n" + "=" * 70)
    print(" RÉSULTATS D'INFÉRENCE D'APPARTENANCE (MIA)")
    print("=" * 70)
    print(f"| Score AUC-ROC global        : {auc:.4f} ({auc*100:.1f}%)")
    print(f"| Seuil optimal de perte      : {loss_threshold:.4f}")
    print(f"| Précision de classification : {accuracy*100:.2f}%")
    print(f"| Matrice de confusion        : TP={conf_matrix['TP']}, FP={conf_matrix['FP']}, TN={conf_matrix['TN']}, FN={conf_matrix['FN']}")
    print("-" * 70)
    print(" DÉTECTION DES CANARIS PAR NIVEAU DE RÉPÉTITION (k) :")
    for k in sorted(canary_detection_by_k.keys()):
        stats = summary_canaries[f"k_{k}"]
        print(f"| k = {k:<4} | Perte moyenne: {stats['mean_loss']:.4f} | Taux de détection IN: {stats['detection_rate_pct']:>5.1f}% |")
    print("=" * 70)

    out_file = os.path.join(output_dir, f"mia_epoch_{epoch}.json")
    final_output = {
        "epoch": epoch,
        "model_path": model_path,
        "auc_roc": round(auc, 4),
        "loss_threshold": round(loss_threshold, 4),
        "accuracy": round(accuracy, 4),
        "confusion_matrix": conf_matrix,
        "in_loss_mean": round(float(np.mean(in_losses)), 4),
        "out_loss_mean": round(float(np.mean(out_losses)), 4),
        "canary_detection_by_k": summary_canaries
    }
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(final_output, f, indent=2, ensure_ascii=False)
    print(f"\n[+] Rapport MIA enregistré dans : {out_file}\n")

def main():
    parser = argparse.ArgumentParser(description="Évaluation MIA pour Fatima-Zahra")
    parser.add_argument("--model_path", type=str, default="checkpoints/ep20", help="Chemin du modèle ou checkpoint")
    parser.add_argument("--train_path", type=str, default="data/dataset_train.json", help="Dataset d'entraînement")
    parser.add_argument("--val_path", type=str, default="data/dataset_val.json", help="Dataset de validation")
    parser.add_argument("--canaries_path", type=str, default="data/canaries_truth.json", help="Canaris")
    parser.add_argument("--epoch", type=int, default=20, help="Numéro d'époque")
    parser.add_argument("--output_dir", type=str, default="results/fatimazahra", help="Dossier des résultats")
    parser.add_argument("--sample_size", type=int, default=150, help="Taille échantillon par classe")
    args = parser.parse_args()
    
    run_mia_evaluation(
        args.model_path,
        args.train_path,
        args.val_path,
        args.canaries_path,
        args.epoch,
        args.output_dir,
        args.sample_size
    )

if __name__ == "__main__":
    main()
