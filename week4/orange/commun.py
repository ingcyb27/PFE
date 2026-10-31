#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
commun.py - Fonctions partagees du volet Orange (Sprint 4, DP-SGD).

Principes :
  * AUCUN mode simule : si PyTorch, transformers ou le GPU manquent, le script
    s'arrete avec un message clair au lieu de produire de faux chiffres.
  * Tous les chemins sont relatifs au dossier de ce fichier : les scripts se
    lancent depuis n'importe ou.
"""

import os
import sys
import json
import random
import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# data/ a cote des scripts ou dans le dossier parent (depot : week4/orange -> week4/data)
DATA_DIR = next((d for d in [os.path.join(BASE_DIR, "data"), os.path.join(os.path.dirname(BASE_DIR), "data")]
                 if os.path.exists(os.path.join(d, "canaries_truth.json"))), os.path.join(BASE_DIR, "data"))
RUNS_DIR = os.path.join(BASE_DIR, "runs")          # un dossier par modele : runs/<tag>/
FIG_DIR = os.path.join(BASE_DIR, "figures")


def run_dir(tag: str) -> str:
    return os.path.join(RUNS_DIR, tag)

BASE_MODEL = "EleutherAI/pythia-160m"   # "epoque 0" = modele avant fine-tuning
CANARY_TAG = "[NOTE DE SERVICE] "       # balise utilisee dans dataset_train.json
MAX_LENGTH = 256                        # troncature utilisee a l'entrainement


def set_seed(seed: int = 42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
    print(f"[+] Seed fixee a {seed}", flush=True)


def die(msg: str):
    print("\n" + "!" * 70, file=sys.stderr)
    print(f"[ERREUR] {msg}", file=sys.stderr)
    print("Aucun resultat n'est ecrit : corrigez le probleme puis relancez.", file=sys.stderr)
    print("!" * 70 + "\n", file=sys.stderr)
    sys.exit(1)


def require_torch():
    try:
        import torch
        import transformers
    except Exception as e:
        die(f"PyTorch ou transformers introuvable ({e}). Activez l'environnement : conda activate pfe_memorization")
    from transformers.utils import is_torch_available
    if not is_torch_available():
        die(f"transformers {transformers.__version__} a desactive PyTorch {torch.__version__} "
            "(versions incompatibles). Corrigez avec : pip install 'transformers==4.46.3'")
    return torch


def get_device(allow_cpu: bool = False) -> str:
    torch = require_torch()
    if torch.cuda.is_available():
        return "cuda"
    if allow_cpu:
        print("[!] Execution sur CPU (--cpu) : test uniquement.", flush=True)
        return "cpu"
    die("Aucun GPU visible. Verifiez 'export CUDA_VISIBLE_DEVICES=1' et nvidia-smi "
        "(ou ajoutez --cpu pour un test volontaire).")


def load_model(path: str, device: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    if path != BASE_MODEL and not os.path.exists(os.path.join(path, "config.json")):
        die(f"Modele introuvable ou incomplet : {path} (config.json absent).")
    try:
        tok = AutoTokenizer.from_pretrained(path)
    except Exception:
        tok = AutoTokenizer.from_pretrained(BASE_MODEL)   # meme tokenizer que l'entrainement
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(path).to(device)
    model.eval()
    return model, tok


def mean_token_nll(model, tok, device, texts, batch_size=16, max_length=MAX_LENGTH):
    """Perte moyenne par token de chaque texte (padding exclu). Score du MIA (Yeom)."""
    import torch
    out = []
    for i in range(0, len(texts), batch_size):
        enc = tok(texts[i:i + batch_size], return_tensors="pt", padding=True,
                  truncation=True, max_length=max_length).to(device)
        with torch.no_grad():
            logits = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"]).logits
        logits = logits[:, :-1].float()
        labels = enc["input_ids"][:, 1:]
        mask = enc["attention_mask"][:, 1:].float()
        nll = torch.nn.functional.cross_entropy(logits.transpose(1, 2), labels, reduction="none")
        out.extend(((nll * mask).sum(1) / mask.sum(1).clamp(min=1)).tolist())
    return out


def span_nll(model, tok, device, texts, start_char, end_char, batch_size=256):
    """
    Log-perplexite (somme des NLL, en nats) des SEULS tokens qui recouvrent les
    caracteres [start_char, end_char) - ici le secret. Le prefixe commun n'est pas
    compte : le classement ne depend que du secret (definition de Carlini et al. 2019).
    """
    import torch
    out = []
    for i in range(0, len(texts), batch_size):
        enc = tok(texts[i:i + batch_size], return_tensors="pt", padding=True,
                  return_offsets_mapping=True)
        offsets = enc.pop("offset_mapping")[:, 1:].to(device)
        enc = enc.to(device)
        with torch.no_grad():
            logits = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"]).logits
        logp = torch.log_softmax(logits[:, :-1].float(), dim=-1)
        tgt = enc["input_ids"][:, 1:]
        tok_lp = logp.gather(2, tgt.unsqueeze(-1)).squeeze(-1)
        sel = (offsets[:, :, 1] > start_char) & (offsets[:, :, 0] < end_char) \
            & (enc["attention_mask"][:, 1:] == 1)
        if (sel.sum(1) == 0).any():
            die("Aucun token du secret trouve dans un candidat (probleme de tokenisation).")
        out.extend((-(tok_lp * sel.float()).sum(1)).tolist())
    return out


def env_info() -> dict:
    info = {"date": datetime.datetime.now().isoformat(timespec="seconds"),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")}
    try:
        import torch
        import transformers
        info["torch"] = torch.__version__
        info["transformers"] = transformers.__version__
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
    except Exception:
        pass
    return info


def save_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def load_json(path):
    if not os.path.exists(path):
        die(f"Fichier introuvable : {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
