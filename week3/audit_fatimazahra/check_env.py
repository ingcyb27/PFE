#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""check_env.py - Verifie que l'environnement permet un VRAI calcul sur GPU."""
import os
import sys

ok = True
try:
    import torch
    import transformers
    from transformers.utils import is_torch_available
except Exception as e:
    print(f"[ERREUR] import impossible : {e}  ->  conda activate pfe_memorization")
    sys.exit(1)

print(f"Python       : {sys.version.split()[0]}")
print(f"PyTorch      : {torch.__version__}")
print(f"transformers : {transformers.__version__}")
if not is_torch_available():
    print("[ERREUR] transformers a desactive PyTorch -> pip install 'transformers==4.46.3'")
    ok = False
print(f"CUDA_VISIBLE_DEVICES = {os.environ.get('CUDA_VISIBLE_DEVICES')}")
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        print(f"GPU {i}        : {p.name} | {p.total_memory / 1e9:.1f} Go")
    x = torch.randn(1000, 1000, device="cuda")
    print(f"Test CUDA    : OK {tuple((x @ x).shape)}")
else:
    print("[ERREUR] aucun GPU visible (export CUDA_VISIBLE_DEVICES=1 ?)")
    ok = False
for mod in ["numpy", "scipy", "matplotlib"]:
    try:
        __import__(mod)
    except Exception as e:
        print(f"[ERREUR] module {mod} manquant ({e}) -> pip install {mod}")
        ok = False
print("ENVIRONNEMENT OK" if ok else "ENVIRONNEMENT INCOMPLET")
sys.exit(0 if ok else 1)
