#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_mia.py (version corrigee) - Sonde 4 : Membership Inference par seuil de perte (Yeom et al. 2018)

Principe : un modele qui a memorise un document lui attribue une perte plus faible.
L'attaquant predit "membre" si la perte est sous un seuil.

POURQUOI UNE CORRECTION ETAIT NECESSAIRE
  Dans data/, les 1500 documents d'entrainement finissent tous par
      "[Ref-Doc #NNNNN : Audit de conformite des systemes d'information.]"
  et les 250 documents de validation par
      "[Ref-Val #NNNNN : Evaluation de performance.]"
  Comparer train et val revient donc a distinguer deux phrases differentes : un
  modele qui n'a RIEN memorise y arrive aussi. L'AUC mesure la difference de
  formulation, pas la fuite d'information.

TEST CORRIGE (dit "apparie")
  Chaque document d'entrainement = un des 5 gabarits + un numero unique.
    * Membre      : le document exact vu a l'entrainement   (gabarit A, numero 0412)
    * Non-membre  : meme numero, meme phrase finale, AUTRE gabarit (gabarit C, numero 0412)
  Les deux textes ont exactement les memes mots possibles et le meme format ; seule
  l'association gabarit <-> numero a pu etre apprise. Ce test mesure donc uniquement
  la memorisation. On n'utilise que les documents d'entrainement SANS canari.

Le test "naif" (train contre val, comme dans la version d'origine) est aussi calcule
et sauvegarde, uniquement pour montrer le biais dans la presentation.

Mesures :
  auc             aire sous la courbe ROC (0.5 = hasard, 1.0 = attaque parfaite)
  acc_yeom        precision au seuil de Yeom (seuil = perte moyenne des membres)
  avantage        TPR - FPR au seuil de Yeom (definition de Yeom et al. 2018)
  acc_best        meilleure precision sur tous les seuils (borne optimiste)
  tpr_1pct_fpr    part des membres detectes avec 1 % de faux positifs
                  (mesure recommandee par Carlini et al. 2022)
  pref_appariee   (test apparie) part des paires ou le modele prefere la version vue
"""

import os
import re
import random
import argparse

from commun import (CANARY_TAG, DATA_DIR, RESULTS_DIR, model_path_for_epoch, set_seed, get_device,
                    load_model, mean_token_nll, load_json, save_json, env_info, die)

DOC_RE = re.compile(r"^(?P<gabarit>.+?) \[Réf-Doc #(?P<num>\d+) : (?P<fin>.+)\]$")


def roc_metrics(member_losses, nonmember_losses):
    import numpy as np
    lm, ln = np.asarray(member_losses), np.asarray(nonmember_losses)
    n1, n0 = len(lm), len(ln)
    scores = np.r_[-lm, -ln]                       # score eleve = "membre"
    y = np.r_[np.ones(n1), np.zeros(n0)]

    # AUC = statistique de Mann-Whitney (rangs moyens en cas d'ex aequo)
    order = scores.argsort(kind="mergesort")
    ranks = np.empty(len(scores))
    ranks[order] = np.arange(1, len(scores) + 1)
    for v in np.unique(scores):
        idx = scores == v
        if idx.sum() > 1:
            ranks[idx] = ranks[idx].mean()
    auc = (ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)

    thr = np.unique(scores)[::-1]
    tpr = np.array([0.0] + [(scores[y == 1] >= t).mean() for t in thr])
    fpr = np.array([0.0] + [(scores[y == 0] >= t).mean() for t in thr])
    acc_best = float(np.max((tpr * n1 + (1 - fpr) * n0) / (n1 + n0)))
    tpr_1 = float(tpr[fpr <= 0.01].max())
    tpr_10 = float(tpr[fpr <= 0.10].max())

    tau = float(lm.mean())                         # seuil de Yeom
    tp, fn = int((lm < tau).sum()), int((lm >= tau).sum())
    fp, tn = int((ln < tau).sum()), int((ln >= tau).sum())
    keep = np.unique(np.linspace(0, len(fpr) - 1, min(len(fpr), 200)).astype(int))
    return {
        "auc": round(float(auc), 4),
        "acc_yeom": round((tp + tn) / (n1 + n0), 4),
        "avantage": round(tp / n1 - fp / n0, 4),
        "acc_best": round(acc_best, 4),
        "tpr_1pct_fpr": round(tpr_1, 4),
        "tpr_10pct_fpr": round(tpr_10, 4),
        "seuil_yeom": round(tau, 4),
        "matrice_confusion_yeom": {"TP": tp, "FN": fn, "FP": fp, "TN": tn},
        "perte_moy_membres": round(float(lm.mean()), 4),
        "perte_moy_non_membres": round(float(ln.mean()), 4),
        "ecart_pertes": round(float(ln.mean() - lm.mean()), 4),
        "roc": {"fpr": [round(float(x), 4) for x in fpr[keep]], "tpr": [round(float(x), 4) for x in tpr[keep]]},
    }


def build_paired_set(train_docs, n, rng):
    """Membres = documents vus ; non-membres = meme numero + autre gabarit."""
    parsed = []
    for d in train_docs:
        m = DOC_RE.match(d)
        if m:
            parsed.append((d, m.group("gabarit"), m.group("num"), m.group("fin")))
    gabarits = sorted({p[1] for p in parsed})
    if len(parsed) < 50 or len(gabarits) < 2:
        return None, None, gabarits
    train_set = set(train_docs)
    sample = rng.sample(parsed, min(n, len(parsed)))
    members, nonmembers = [], []
    for doc, gab, num, fin in sample:
        autre = rng.choice([g for g in gabarits if g != gab])
        fake = f"{autre} [Réf-Doc #{num} : {fin}]"
        if fake in train_set:
            die(f"Le non-membre construit existe dans le train : {fake}")
        members.append(doc)
        nonmembers.append(fake)
    return members, nonmembers, gabarits


def print_block(title, r):
    print(f"--- {title}")
    print(f"  Perte membres {r['perte_moy_membres']:.4f} | non-membres {r['perte_moy_non_membres']:.4f} "
          f"| ecart {r['ecart_pertes']:+.4f}")
    print(f"  AUC {r['auc']:.3f} | precision (seuil Yeom) {r['acc_yeom']:.3f} | avantage {r['avantage']:.3f} "
          f"| TPR@1%FPR {r['tpr_1pct_fpr']:.3f}")
    cm = r["matrice_confusion_yeom"]
    print(f"  Matrice de confusion (seuil Yeom) : TP={cm['TP']} FN={cm['FN']} FP={cm['FP']} TN={cm['TN']}", flush=True)


def main():
    ap = argparse.ArgumentParser(description="Sonde 4 - MIA par seuil de perte (Yeom 2018)")
    ap.add_argument("--epoch", type=int, required=True, help="0 = pythia-160m d'origine")
    ap.add_argument("--model_path", default=None, help="Par defaut : checkpoints/ep<E>")
    ap.add_argument("--n", type=int, default=400, help="Nombre de paires membre / non-membre")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cpu", action="store_true", help="Test sur CPU uniquement")
    args = ap.parse_args()

    set_seed(args.seed)
    device = get_device(allow_cpu=args.cpu)
    model_path = args.model_path or model_path_for_epoch(args.epoch)

    train_all = load_json(os.path.join(DATA_DIR, "dataset_train.json"))
    val = load_json(os.path.join(DATA_DIR, "dataset_val.json"))
    train_clean = [d for d in train_all if CANARY_TAG not in d]
    rng = random.Random(2000)                      # memes paires a toutes les epoques
    members, nonmembers, gabarits = build_paired_set(train_clean, args.n, rng)

    model, tok = load_model(model_path, device)
    print(f"\n=== SONDE 4 - MIA (Yeom) | epoque {args.epoch} | {model_path} ===", flush=True)
    print(f"[*] {len(train_clean)} documents d'entrainement sans canari, {len(val)} de validation, "
          f"{len(gabarits)} gabarits detectes", flush=True)

    result = {"sonde": "Sonde 4 - MIA par seuil de perte (Yeom et al. 2018)", "mode": "REEL",
              "epoch": args.epoch, "model_path": model_path, "env": env_info()}

    if members is not None:
        lm = mean_token_nll(model, tok, device, members)
        ln = mean_token_nll(model, tok, device, nonmembers)
        r = roc_metrics(lm, ln)
        r["pref_appariee"] = round(sum(a < b for a, b in zip(lm, ln)) / len(lm), 4)
        r["n_paires"] = len(lm)
        r["exemple"] = {"membre": members[0], "non_membre": nonmembers[0]}
        r["pertes"] = {"membres": [round(x, 4) for x in lm], "non_membres": [round(x, 4) for x in ln]}
        result["mia_apparie"] = r
        print_block(f"TEST APPARIE (principal) : {len(lm)} paires, meme numero, gabarit different", r)
        print(f"  Le modele prefere la version vue dans {100 * r['pref_appariee']:.1f} % des paires "
              f"(50 % = aucune memorisation)", flush=True)
    else:
        print("[!] Format gabarit + numero non detecte : seul le test train/val est calcule.", flush=True)

    n_naif = min(args.n, len(train_clean), len(val))
    src = members if members is not None else rng.sample(train_clean, n_naif)
    lm_n = mean_token_nll(model, tok, device, src[:n_naif])
    ln_n = mean_token_nll(model, tok, device, rng.sample(val, n_naif))
    rn = roc_metrics(lm_n, ln_n)
    rn["n_par_classe"] = n_naif
    rn["avertissement"] = ("Train et val n'utilisent pas la meme phrase finale (Ref-Doc/Audit vs "
                           "Ref-Val/Evaluation) : ce test est biaise si le corpus est le corpus de secours.")
    result["mia_naif_train_vs_val"] = rn
    print_block(f"TEST NAIF train vs val (biaise, pour comparaison) : {n_naif} par classe", rn)

    out = os.path.join(RESULTS_DIR, f"mia_epoch_{args.epoch}.json")
    save_json(result, out)
    print(f"[+] Resultats : {out}\n", flush=True)


if __name__ == "__main__":
    main()
