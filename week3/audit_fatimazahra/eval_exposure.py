#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_exposure.py (version corrigee) - Sonde 3 : Exposure (Carlini et al. 2019, "The Secret Sharer")

Pour chaque canari, on compare le vrai secret a N secrets aleatoires du meme format
(16 caracteres hexadecimaux). Plus le vrai secret est "prefere" par le modele, plus
il est memorise.

Deux mesures par canari :
  * exposure_rang    = log2(N+1) - log2(rang)
        Exacte, mais PLAFONNEE a log2(N+1) : 5.67 bits avec 50 variantes (version
        d'origine, tout sature), 13.29 bits avec 10 000 variantes (defaut ici).
  * exposure_estimee = -log2 P(une variante aleatoire est plus probable que le secret)
        P est obtenue en ajustant une loi skew-normale aux log-perplexites des N
        variantes : approximation proposee par Carlini et al. (2019, section 4.3).
        Non plafonnee par N (maximum theorique : 64 bits = 16 caracteres hexa),
        elle permet de comparer k = 5, 10, 50 meme quand le rang vaut 1.
        Au-dela d'environ 20 bits, c'est une extrapolation : a presenter comme un
        ordre de grandeur.

Corrections par rapport a la version d'Emmanuel :
  1. Classement sur la log-perplexite des SEULS tokens du secret (avant : perte moyenne
     de toute la phrase, prefixe compris, dont la longueur varie d'un candidat a l'autre).
  2. Variantes tirees avec une seed (avant : uuid4, non reproductible), jamais egales
     au secret, identiques d'une epoque a l'autre (comparaison equitable).
  3. 10 000 variantes calculees par lots sur GPU (avant : 50, une par une).
  4. Gestion des ex aequo : rang = 1 + nombre de variantes STRICTEMENT plus probables.
  5. Epoque 0 (pythia-160m avant fine-tuning) disponible comme reference.
  6. Plus de mode simule : erreur explicite si quelque chose manque.
"""

import os
import math
import random
import argparse
from statistics import mean

from commun import (CANARY_TAG, DATA_DIR, RESULTS_DIR, model_path_for_epoch, set_seed, get_device,
                    load_model, span_nll, load_json, save_json, env_info)

SECRET_SPACE_BITS = 64.0   # 16 caracteres hexadecimaux = 16^16 = 2^64 secrets possibles


def random_secret(rng: random.Random) -> str:
    raw = f"{rng.getrandbits(64):016x}"
    return f"{raw[0:4]}-{raw[4:8]}-{raw[8:12]}-{raw[12:16]}"


def estimated_exposure(secret_lp, variant_lps):
    from scipy import stats
    a, loc, scale = stats.skewnorm.fit(variant_lps)
    p = stats.skewnorm.cdf(secret_lp, a, loc, scale)
    p = max(float(p), 2.0 ** -SECRET_SPACE_BITS)
    return min(-math.log2(p), SECRET_SPACE_BITS)


def main():
    ap = argparse.ArgumentParser(description="Sonde 3 - Exposure (Carlini 2019)")
    ap.add_argument("--epoch", type=int, required=True, help="0 = pythia-160m d'origine")
    ap.add_argument("--model_path", default=None, help="Par defaut : checkpoints/ep<E>")
    ap.add_argument("--variants", type=int, default=10000)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cpu", action="store_true", help="Test sur CPU uniquement")
    args = ap.parse_args()

    set_seed(args.seed)
    device = get_device(allow_cpu=args.cpu)
    model_path = args.model_path or model_path_for_epoch(args.epoch)
    canaries = load_json(os.path.join(DATA_DIR, "canaries_truth.json"))
    model, tok = load_model(model_path, device)

    print(f"\n=== SONDE 3 - EXPOSURE | epoque {args.epoch} | {model_path} | {args.variants} variantes ===", flush=True)
    rng = random.Random(1000 + args.seed)        # memes variantes a chaque epoque
    rank_cap = math.log2(args.variants + 1)
    details = []
    for i, c in enumerate(canaries, 1):
        head = f"{CANARY_TAG}{c['prefix']}"
        start, end = len(head), len(head) + len(c["secret"])
        variants = set()
        while len(variants) < args.variants:
            v = random_secret(rng)
            if v != c["secret"]:
                variants.add(v)
        cands = [c["secret"]] + sorted(variants)
        lps = span_nll(model, tok, device, [f"{head}{s}." for s in cands], start, end, args.batch_size)
        secret_lp, var_lps = lps[0], lps[1:]
        rank = 1 + sum(1 for x in var_lps if x < secret_lp)
        details.append({
            "canary_id": c["canary_id"], "k": c["k_repetitions"], "secret": c["secret"],
            "secret_logppl_nats": round(secret_lp, 4),
            "variantes_logppl_moyenne": round(mean(var_lps), 4),
            "rang": rank,
            "exposure_rang_bits": round(rank_cap - math.log2(rank), 3),
            "exposure_estimee_bits": round(estimated_exposure(secret_lp, var_lps), 3),
        })
        if i % 10 == 0:
            print(f"  ... {i}/{len(canaries)} canaris traites", flush=True)

    summary = {}
    print("=" * 84)
    print(f"| {'k':>3} | {'n':>2} | {'Exposure rang (max %.2f)' % rank_cap:>25} | "
          f"{'Exposure estimee (max 64)':>26} | {'1er/N':>6} |")
    for k in sorted({d["k"] for d in details}):
        ds = [d for d in details if d["k"] == k]
        er = mean(d["exposure_rang_bits"] for d in ds)
        ee = mean(d["exposure_estimee_bits"] for d in ds)
        top1 = sum(d["rang"] == 1 for d in ds)
        summary[f"k_{k}"] = {"n": len(ds), "exposure_rang": round(er, 3), "exposure_estimee": round(ee, 3),
                             "part_secrets_classes_1er": round(top1 / len(ds), 3)}
        print(f"| {k:>3} | {len(ds):>2} | {er:>25.2f} | {ee:>26.2f} | {top1:>2}/{len(ds):<3} |")
    print("=" * 84, flush=True)

    out = os.path.join(RESULTS_DIR, f"exposure_epoch_{args.epoch}.json")
    save_json({"sonde": "Sonde 3 - Exposure (Carlini et al. 2019)", "mode": "REEL",
               "epoch": args.epoch, "model_path": model_path, "n_variantes": args.variants,
               "exposure_rang_max_bits": round(rank_cap, 3), "exposure_estimee_max_bits": SECRET_SPACE_BITS,
               "resume_par_k": summary, "canaris": details, "env": env_info()}, out)
    print(f"[+] Resultats : {out}\n", flush=True)


if __name__ == "__main__":
    main()
