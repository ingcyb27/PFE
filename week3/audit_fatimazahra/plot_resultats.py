#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_resultats.py - Lit results/fatimazahra/*.json et produit, dans results/fatimazahra/figures/ :
  fig1_exposure_rang.png      Exposure par rang, par k, en fonction de l'epoque
  fig2_exposure_estimee.png   Exposure estimee (non plafonnee), par k
  fig3_mia_auc.png            AUC du MIA : test apparie (correct) vs test naif (biaise)
  fig4_mia_roc.png            Courbes ROC du test apparie
  fig5_mia_pertes.png         Distribution des pertes membres / non-membres (derniere epoque)
  resume.md / resume.csv      Tableaux prets a copier dans le rapport ou les slides
"""

import os
import csv
import glob
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from commun import RESULTS_DIR, FIG_DIR

K_COLORS = {1: "#1f77b4", 5: "#ff7f0e", 10: "#2ca02c", 50: "#d62728"}


def load(kind):
    out = {}
    for p in glob.glob(os.path.join(RESULTS_DIR, f"{kind}_epoch_*.json")):
        with open(p, encoding="utf-8") as f:
            d = json.load(f)
        if d.get("mode") == "REEL":
            out[d["epoch"]] = d
    return dict(sorted(out.items()))


def setup_axis(ax, epochs, xlabel="Epoque"):
    ax.set_xticks(range(len(epochs)))
    ax.set_xticklabels(["0 (base)" if e == 0 else str(e) for e in epochs])
    ax.set_xlabel(xlabel)
    ax.grid(alpha=.3)


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    md = ["# Resultats - Sonde 3 (Exposure) et Sonde 4 (MIA)\n",
          "Modele : checkpoints d'Emmanuel (pythia-160m, meme modele pour tout le binome).\n"]
    rows = []

    exp = load("exposure")
    if exp:
        epochs = list(exp)
        first = exp[epochs[0]]
        ks = sorted(int(k[2:]) for k in first["resume_par_k"])
        for field, fname, title, cap in [
            ("exposure_rang", "fig1_exposure_rang.png",
             f"Sonde 3 - Exposure par rang ({first['n_variantes']} variantes)", first["exposure_rang_max_bits"]),
            ("exposure_estimee", "fig2_exposure_estimee.png",
             "Sonde 3 - Exposure estimee (loi skew-normale, max 64 bits)", None)]:
            fig, ax = plt.subplots(figsize=(7, 4.5))
            for k in ks:
                ax.plot(range(len(epochs)), [exp[e]["resume_par_k"][f"k_{k}"][field] for e in epochs],
                        marker="o", label=f"k = {k}", color=K_COLORS.get(k))
            if cap:
                ax.axhline(cap, ls="--", c="grey", lw=1, label=f"plafond {cap:.2f} bits")
            setup_axis(ax, epochs)
            ax.set_ylabel("Exposure moyenne (bits)")
            ax.set_title(title)
            ax.legend()
            fig.tight_layout()
            fig.savefig(os.path.join(FIG_DIR, fname), dpi=150)
            plt.close(fig)

        md.append(f"## Sonde 3 - Exposure ({first['n_variantes']} variantes)\n")
        md.append(f"Exposure par rang (plafond {first['exposure_rang_max_bits']} bits) / exposure estimee (max 64 bits) "
                  "/ part des secrets classes 1er\n")
        md.append("| Epoque | " + " | ".join(f"k={k}" for k in ks) + " |")
        md.append("|---|" + "---|" * len(ks))
        for e in epochs:
            cells = []
            for k in ks:
                s = exp[e]["resume_par_k"][f"k_{k}"]
                cells.append(f"{s['exposure_rang']:.2f} / {s['exposure_estimee']:.1f} / {100 * s['part_secrets_classes_1er']:.0f} %")
                rows.append(["exposure", e, k, s["exposure_rang"], s["exposure_estimee"], s["part_secrets_classes_1er"]])
            md.append(f"| {e} | " + " | ".join(cells) + " |")
        md.append("")

    mia = load("mia")
    if mia:
        epochs = list(mia)
        fig, ax = plt.subplots(figsize=(7, 5.5))
        if all("mia_apparie" in mia[e] for e in epochs):
            ax.plot(range(len(epochs)), [mia[e]["mia_apparie"]["auc"] for e in epochs], marker="o",
                    c="#d62728", label="AUC - test apparie (mesure la memorisation)")
            ax.plot(range(len(epochs)), [mia[e]["mia_apparie"]["tpr_1pct_fpr"] for e in epochs], marker="s",
                    c="#9467bd", label="TPR a 1 % FPR - test apparie")
        ax.plot(range(len(epochs)), [mia[e]["mia_naif_train_vs_val"]["auc"] for e in epochs], marker="^",
                ls="--", c="grey", label="AUC - test naif train/val (biaise)")
        ax.axhline(0.5, ls=":", c="black", lw=1, label="hasard")
        setup_axis(ax, epochs)
        ax.set_ylim(0, 1.03)
        ax.set_title("Sonde 4 - Membership Inference (seuil de perte, Yeom 2018)")
        ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2)
        fig.tight_layout()
        fig.savefig(os.path.join(FIG_DIR, "fig3_mia_auc.png"), dpi=150)
        plt.close(fig)

        if all("mia_apparie" in mia[e] for e in epochs):
            fig, ax = plt.subplots(figsize=(5.5, 5))
            for e in epochs:
                roc = mia[e]["mia_apparie"]["roc"]
                ax.plot(roc["fpr"], roc["tpr"], label=f"epoque {e} (AUC {mia[e]['mia_apparie']['auc']:.2f})")
            ax.plot([0, 1], [0, 1], ls=":", c="black")
            ax.set_xlabel("Taux de faux positifs")
            ax.set_ylabel("Taux de vrais positifs")
            ax.set_title("Courbes ROC - test apparie")
            ax.grid(alpha=.3)
            ax.legend(fontsize=8)
            fig.tight_layout()
            fig.savefig(os.path.join(FIG_DIR, "fig4_mia_roc.png"), dpi=150)
            plt.close(fig)

            last = epochs[-1]
            p = mia[last]["mia_apparie"]["pertes"]
            fig, ax = plt.subplots(figsize=(7, 4.5))
            ax.hist(p["membres"], bins=40, alpha=.6, label="membres (vus)", color="#d62728")
            ax.hist(p["non_membres"], bins=40, alpha=.6, label="non-membres (jamais vus)", color="#1f77b4")
            ax.axvline(mia[last]["mia_apparie"]["seuil_yeom"], c="black", ls="--", label="seuil de Yeom")
            ax.set_xlabel("Perte moyenne par token")
            ax.set_ylabel("Nombre de documents")
            ax.set_title(f"Distribution des pertes - epoque {last}")
            ax.legend()
            fig.tight_layout()
            fig.savefig(os.path.join(FIG_DIR, "fig5_mia_pertes.png"), dpi=150)
            plt.close(fig)

        md.append("## Sonde 4 - MIA (seuil de perte, Yeom 2018)\n")
        md.append("| Epoque | AUC apparie | Precision Yeom | Avantage | TPR@1%FPR | Prefere la version vue | AUC naif (biaise) |")
        md.append("|---|---|---|---|---|---|---|")
        for e in epochs:
            n = mia[e]["mia_naif_train_vs_val"]
            a = mia[e].get("mia_apparie")
            if a:
                md.append(f"| {e} | {a['auc']:.3f} | {a['acc_yeom']:.3f} | {a['avantage']:.3f} | "
                          f"{a['tpr_1pct_fpr']:.3f} | {100 * a['pref_appariee']:.1f} % | {n['auc']:.3f} |")
                rows.append(["mia_apparie", e, "-", a["auc"], a["acc_yeom"], a["tpr_1pct_fpr"]])
            else:
                md.append(f"| {e} | - | - | - | - | - | {n['auc']:.3f} |")
            rows.append(["mia_naif", e, "-", n["auc"], n["acc_yeom"], n["tpr_1pct_fpr"]])
        md.append("")

    with open(os.path.join(FIG_DIR, "resume.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    with open(os.path.join(FIG_DIR, "resume.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sonde", "epoque", "k", "valeur1", "valeur2", "valeur3"])
        w.writerows(rows)
    print("\n".join(md))
    print(f"[+] Figures et tableaux dans {FIG_DIR}")


if __name__ == "__main__":
    main()
