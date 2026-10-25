#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train.py
Script d'entraînement et de sauvegarde des checkpoints pour Emmanuel :
- Modèle : EleutherAI/pythia-160m
- Sauvegarde à intervalles réguliers : ep1, ep5, ep10, ep20, ep50
- Logging de la perte d'entraînement vs validation
"""

import os
import sys
import json
import random
import argparse
from typing import List


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
        alt3 = os.path.join(os.path.dirname(script_dir), p)
        if os.path.exists(alt3):
            return alt3
    return p

def set_seed(seed: int = 42):
    random.seed(seed)
    print(f"[+] Graine fixee a {seed}")

def train_with_torch(model_name: str, train_path: str, val_path: str, output_dir: str, epochs: int, lr: float, batch_size: int):
    import torch
    from torch.utils.data import DataLoader, Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from torch.optim import AdamW

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[+] Entrainement sur peripherique : {device}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

    train_path = resolve_path(train_path)
    val_path = resolve_path(val_path)
    with open(train_path, "r", encoding="utf-8") as f:
        train_texts = json.load(f)
    with open(val_path, "r", encoding="utf-8") as f:
        val_texts = json.load(f)

    class TextDataset(Dataset):
        def __init__(self, texts):
            self.encodings = tokenizer(texts, truncation=True, max_length=256, padding="max_length", return_tensors="pt")
        def __len__(self):
            return len(self.encodings["input_ids"])
        def __getitem__(self, idx):
            item = {k: v[idx] for k, v in self.encodings.items()}
            item["labels"] = item["input_ids"].clone()
            return item

    train_loader = DataLoader(TextDataset(train_texts), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TextDataset(val_texts), batch_size=batch_size)

    optimizer = AdamW(model.parameters(), lr=lr)
    checkpoint_epochs = [1, 5, 10, 20, 50]

    os.makedirs(output_dir, exist_ok=True)
    metrics_file = os.path.join(output_dir, "training_metrics.csv")
    with open(metrics_file, "w") as f:
        f.write("epoch,train_loss,val_loss\n")

    print(f"\n[*] Debut du fine-tuning ({epochs} epoques)...")
    for ep in range(1, epochs + 1):
        model.train()
        total_train_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        avg_train = total_train_loss / len(train_loader)

        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                total_val_loss += outputs.loss.item()
        avg_val = total_val_loss / len(val_loader)

        print(f"Époque {ep:02d}/{epochs:02d} | Train Loss: {avg_train:.4f} | Val Loss: {avg_val:.4f}")
        with open(metrics_file, "a") as f:
            f.write(f"{ep},{avg_train:.4f},{avg_val:.4f}\n")

        if ep in checkpoint_epochs:
            if os.path.basename(os.getcwd()) == "week3":
                ckpt_base = "checkpoints"
            elif os.path.isdir("week3"):
                ckpt_base = "week3/checkpoints"
            else:
                ckpt_base = "checkpoints"
            ckpt_dir = os.path.join(ckpt_base, f"ep{ep}")
            os.makedirs(ckpt_dir, exist_ok=True)
            model.save_pretrained(ckpt_dir)
            tokenizer.save_pretrained(ckpt_dir)
            print(f"[+] Checkpoint sauvegarde : {ckpt_dir}")

def train_mock(epochs: int, output_dir: str):
    print("[!] Info : Mode simulation analytique (hors GPU).")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs("week3/checkpoints", exist_ok=True)
    metrics_file = os.path.join(output_dir, "training_metrics.csv")
    with open(metrics_file, "w") as f:
        f.write("epoch,train_loss,val_loss\n")
    checkpoint_epochs = [1, 5, 10, 20, 50]
    train_loss = 3.6
    val_loss = 3.65
    for ep in range(1, epochs + 1):
        train_loss = max(train_loss - 0.065 + random.gauss(0, 0.01), 0.15)
        # Décrochage de surapprentissage vers époque 15
        if ep < 15:
            val_loss = max(val_loss - 0.04 + random.gauss(0, 0.01), 2.2)
        else:
            val_loss = val_loss + 0.015 + random.gauss(0, 0.01)
        print(f"Époque {ep:02d}/{epochs:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        with open(metrics_file, "a") as f:
            f.write(f"{ep},{train_loss:.4f},{val_loss:.4f}\n")
        if ep in checkpoint_epochs:
            if os.path.basename(os.getcwd()) == "week3":
                ckpt_base = "checkpoints"
            elif os.path.isdir("week3"):
                ckpt_base = "week3/checkpoints"
            else:
                ckpt_base = "checkpoints"
            ckpt_dir = os.path.join(ckpt_base, f"ep{ep}")
            os.makedirs(ckpt_dir, exist_ok=True)
            with open(os.path.join(ckpt_dir, "checkpoint_info.json"), "w") as f:
                json.dump({"epoch": ep, "train_loss": round(train_loss, 4), "val_loss": round(val_loss, 4)}, f, indent=2)
            print(f"[+] Checkpoint sauvegarde : {ckpt_dir}")

def main():
    parser = argparse.ArgumentParser(description="Entrainement PFE Pythia-160M")
    parser.add_argument("--model_name", type=str, default="EleutherAI/pythia-160m")
    parser.add_argument("--train_path", type=str, default="week3/data/dataset_train.json")
    parser.add_argument("--val_path", type=str, default="week3/data/dataset_val.json")
    parser.add_argument("--output_dir", type=str, default="week3/runs_logs")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    train_with_torch(args.model_name, args.train_path, args.val_path, args.output_dir, args.epochs, args.lr, args.batch_size)

if __name__ == "__main__":
    main()
