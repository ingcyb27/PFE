#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_orange.py - Rassemble runs/<tag>/*.json et produit dans figures/ :
  fig1_mia_auc.png         AUC du MIA apparie par modele (ligne du hasard a 0,5)
  fig2_exposure_par_k.png  Exposure par rang, par frequence k, pour chaque modele
  fig3_compromis.png       Compromis : perplexite (utilite) contre AUC du MIA (fuite)
  fig4_entrainement.png    Perplexite de validation et epsilon au fil des epoques
  fig5_exposure_vs_eps.png Exposure par k en fonction de epsilon (ou la fuite reapparait)
  resume.md / resume.csv   Tableau recapitulatif pret pour le rapport et les slides
"""

import os
import csv
import json
import glob
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from commun import RUNS_DIR, FIG_DIR

ORDER = ["base", "sprint3_ep50", "nodp", "dp_eps8", "dp_eps2"]
K_COLORS = {1: "#1f77b4", 5: "#ff7f0e", 10: "#2ca02c", 50: "#d62728"}


def jload(path):
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return None


def label(tag, rep):
    if tag == "base":
        return "pythia d'origine"
    if tag == "sprint3_ep50":
        return "Sprint 3 ép.50"
    if rep and rep.get("dp"):
        return f"DP ε={rep['epsilon_final']:.1f}"
    if rep:
        return "sans DP (réf.)"
    return tag


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    tags = [os.path.basename(p) for p in glob.glob(os.path.join(RUNS_DIR, "*")) if os.path.isdir(p)]
    tags = sorted(tags, key=lambda t: (ORDER.index(t) if t in ORDER else 99, t))
    M = {}
    for t in tags:
        d = os.path.join(RUNS_DIR, t)
        M[t] = {"rep": jload(os.path.join(d, "privacy_report.json")), "exp": jload(os.path.join(d, "exposure.json")),
                "mia": jload(os.path.join(d, "mia.json")), "uti": jload(os.path.join(d, "utilite.json"))}
        M[t]["label"] = label(t, M[t]["rep"])
    tags = [t for t in tags if M[t]["exp"] or M[t]["mia"] or M[t]["uti"]]
    # ordre : references, puis modeles DP du epsilon le plus grand au plus petit (protection croissante)
    fixed = [t for t in ["base", "sprint3_ep50", "nodp"] if t in tags]
    dps = sorted([t for t in tags if M[t]["rep"] and M[t]["rep"].get("dp")],
                 key=lambda t: -M[t]["rep"]["epsilon_final"])
    tags = fixed + dps + [t for t in tags if t not in fixed and t not in dps]
    if not tags:
        print("[!] Aucun resultat dans runs/ : lancez d'abord les audits.")
        return

    # Tableau recapitulatif
    rows, md = [], ["# Résultats — volet Orange (DP-SGD)\n",
                    "| Modèle | ε | σ | PPL val | Perte générale (nats/token) | AUC MIA apparié | TPR@1%FPR | Exposure k=1 / 5 / 10 / 50 (bits) |",
                    "|---|---|---|---|---|---|---|---|"]
    for t in tags:
        m = M[t]
        eps = "∞" if not (m["rep"] and m["rep"].get("dp")) else f"{m['rep']['epsilon_final']:.2f}"
        if t == "base":
            eps = "— (pas entraîné)"
        sig = f"{m['rep']['sigma']:.3f}" if m["rep"] else "—"
        pv = m["uti"]["val"]["ppl"] if m["uti"] else None
        pg = m["uti"]["general"].get("ppl") if m["uti"] else None
        pg = round(math.log(pg), 2) if pg else None      # perte = ln(PPL), plus lisible que des PPL de 1e11
        a = m["mia"]["mia_apparie"] if m["mia"] and "mia_apparie" in m["mia"] else None
        ex = m["exp"]["resume_par_k"] if m["exp"] else None
        exs = " / ".join(f"{ex[f'k_{k}']['exposure_rang']:.1f}" for k in (1, 5, 10, 50)) if ex else "—"
        md.append(f"| {m['label']} | {eps} | {sig} | {pv if pv is not None else '—'} | {pg if pg is not None else '—'} | "
                  f"{a['auc']:.3f} | {a['tpr_1pct_fpr']:.3f} | {exs} |" if a else
                  f"| {m['label']} | {eps} | {sig} | {pv if pv is not None else '—'} | {pg if pg is not None else '—'} | — | — | {exs} |")
        rows.append([t, eps, sig, pv, pg, a["auc"] if a else None, a["tpr_1pct_fpr"] if a else None,
                     *([ex[f"k_{k}"]["exposure_rang"] for k in (1, 5, 10, 50)] if ex else [None] * 4)])
    if any(M[t]["exp"] for t in tags):
        e0 = next(M[t]["exp"] for t in tags if M[t]["exp"])
        md.append(f"\nExposure : {e0['n_variantes']} leurres, plafond {e0['exposure_rang_max_bits']} bits, hasard ≈ 1,4 bit.")

    # Fig 1 : AUC
    mt = [t for t in tags if M[t]["mia"] and "mia_apparie" in M[t]["mia"]]
    if mt:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        vals = [M[t]["mia"]["mia_apparie"]["auc"] for t in mt]
        ax.bar(range(len(mt)), vals, color="#4c72b0")
        for i, v in enumerate(vals):
            ax.text(i, v + 0.01, f"{v:.2f}", ha="center", fontsize=10)
        ax.axhline(0.5, ls="--", c="grey", lw=1, label="hasard (0,5)")
        ax.set_xticks(range(len(mt)))
        ax.set_xticklabels([M[t]["label"] for t in mt], rotation=15)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("AUC (test apparié)")
        ax.set_title("MIA : peut-on reconnaître les documents d'entraînement ?")
        ax.legend()
        ax.grid(axis="y", alpha=.3)
        fig.tight_layout()
        fig.savefig(os.path.join(FIG_DIR, "fig1_mia_auc.png"), dpi=150)
        plt.close(fig)

    # Fig 2 : Exposure par k
    et = [t for t in tags if M[t]["exp"]]
    if et:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        w = 0.8 / 4
        for j, k in enumerate((1, 5, 10, 50)):
            ax.bar([i + (j - 1.5) * w for i in range(len(et))],
                   [M[t]["exp"]["resume_par_k"][f"k_{k}"]["exposure_rang"] for t in et],
                   width=w, color=K_COLORS[k], label=f"k = {k}")
        cap = M[et[0]]["exp"]["exposure_rang_max_bits"]
        ax.axhline(cap, ls="--", c="grey", lw=1, label=f"plafond {cap:.1f} bits")
        ax.axhline(1.44, ls=":", c="black", lw=1, label="hasard ≈ 1,4 bit")
        ax.set_xticks(range(len(et)))
        ax.set_xticklabels([M[t]["label"] for t in et], rotation=15)
        ax.set_ylabel("Exposure moyenne (bits)")
        ax.set_title("Exposure des secrets selon leur fréquence k")
        ax.legend(fontsize=8, ncol=3)
        ax.grid(axis="y", alpha=.3)
        fig.tight_layout()
        fig.savefig(os.path.join(FIG_DIR, "fig2_exposure_par_k.png"), dpi=150)
        plt.close(fig)

    # Fig 3 : compromis utilite / fuite
    ct = [t for t in tags if M[t]["uti"] and M[t]["mia"] and "mia_apparie" in M[t]["mia"]]
    if ct:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        offs = [(6, 8), (6, -14), (6, 22), (6, -28), (6, 36)]
        low = 0
        for t in ct:
            x, y = M[t]["uti"]["val"]["ppl"], M[t]["mia"]["mia_apparie"]["auc"]
            ax.scatter(x, y, s=60)
            o = (6, 6)
            if y < 0.6:                      # points serres au niveau du hasard : etiquettes decalees
                o = offs[low % len(offs)]
                low += 1
            ax.annotate(M[t]["label"], (x, y), textcoords="offset points", xytext=o, fontsize=9,
                        arrowprops=dict(arrowstyle="-", color="grey", lw=0.5) if o[1] < 0 or o[1] > 10 else None)
        ax.axhline(0.5, ls="--", c="grey", lw=1)
        ax.set_xlabel("Perplexité de validation (plus bas = meilleur modèle)")
        ax.set_ylabel("AUC du MIA apparié (plus bas = moins de fuite)")
        ax.set_title("Compromis confidentialité / utilité")
        ax.grid(alpha=.3)
        fig.tight_layout()
        fig.savefig(os.path.join(FIG_DIR, "fig3_compromis.png"), dpi=150)
        plt.close(fig)

    # Fig 4 : entrainement
    tt = [t for t in tags if os.path.exists(os.path.join(RUNS_DIR, t, "training_metrics.csv"))]
    if tt:
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4))
        for t in tt:
            with open(os.path.join(RUNS_DIR, t, "training_metrics.csv")) as f:
                rd = list(csv.DictReader(f))
            ep = [int(r["epoch"]) for r in rd]
            a1.plot(ep, [float(r["val_ppl"]) for r in rd], label=M[t]["label"])
            if rd and rd[0]["epsilon"] != "inf":
                a2.plot(ep, [float(r["epsilon"]) for r in rd], label=M[t]["label"])
        a1.set_xlabel("Époque")
        a1.set_ylabel("Perplexité de validation")
        a1.set_title("Utilité pendant l'entraînement")
        a1.grid(alpha=.3)
        a1.legend(fontsize=8)
        a2.set_xlabel("Époque")
        a2.set_ylabel("ε consommé (δ = 1e-5)")
        a2.set_title("Budget de confidentialité")
        a2.grid(alpha=.3)
        a2.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(FIG_DIR, "fig4_entrainement.png"), dpi=150)
        plt.close(fig)

    # Fig 5 : Exposure en fonction de epsilon
    dps_e = [t for t in dps if M[t]["exp"]]
    if dps_e:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        xs = [M[t]["rep"]["epsilon_final"] for t in dps_e]
        ref = M["nodp"]["exp"] if "nodp" in M and M["nodp"]["exp"] else None
        for k in (1, 5, 10, 50):
            ys = [M[t]["exp"]["resume_par_k"][f"k_{k}"]["exposure_rang"] for t in dps_e]
            ax.plot(xs, ys, marker="o", color=K_COLORS[k], label=f"k = {k}")
            if ref:
                ax.scatter([max(xs) * 4], [ref["resume_par_k"][f"k_{k}"]["exposure_rang"]], color=K_COLORS[k], marker="s")
        if ref:
            vals = [ref["resume_par_k"][f"k_{k}"]["exposure_rang"] for k in (1, 5, 10, 50)]
            ax.text(max(xs) * 4, min(vals) - 0.9, "sans DP :\n" + " / ".join(f"{v:.1f}" for v in vals),
                    ha="center", va="top", fontsize=8)
        ax.set_xscale("log")
        ticks = sorted(xs) + ([max(xs) * 4] if ref else [])
        ax.set_xticks(ticks)
        ax.set_xticklabels([f"{x:.0f}" for x in sorted(xs)] + (["sans DP"] if ref else []))
        ax.minorticks_off()
        ax.axhline(1.44, ls=":", c="black", lw=1, label="hasard ≈ 1,4 bit")
        ax.set_xlabel("ε (échelle log)  —  plus à gauche = protection plus forte")
        ax.set_ylabel("Exposure moyenne (bits)")
        ax.set_title("Les secrets répétés fuient en premier quand ε augmente")
        ax.grid(alpha=.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(FIG_DIR, "fig5_exposure_vs_eps.png"), dpi=150)
        plt.close(fig)

    with open(os.path.join(FIG_DIR, "resume.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")
    with open(os.path.join(FIG_DIR, "resume.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["tag", "epsilon", "sigma", "ppl_val", "ppl_generale", "auc_apparie", "tpr_1pct",
                    "exp_k1", "exp_k5", "exp_k10", "exp_k50"])
        w.writerows(rows)
    print("\n".join(md))
    print(f"\n[+] Figures et tableau dans {FIG_DIR}")


if __name__ == "__main__":
    main()
