#!/usr/bin/env bash
# setup_env.sh : Configuration de l'environnement ML pour le Sprint 3 (Serveur rst-goose)
# A executer dans votre $HOME sur rst-goose.telecom-sudparis.eu

set -e

echo "=== 1. Creation de l'environnement Conda (Python 3.10) ==="
conda create -y -n pfe_memorization python=3.10

echo "=== 2. Activation de l'environnement ==="
eval "$(conda shell.bash hook)"
conda activate pfe_memorization

echo "=== 3. Installation de PyTorch avec support CUDA ==="
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

echo "=== 4. Installation des librairies de NLP et d'evaluation ==="
pip install transformers datasets accelerate evaluate scikit-learn matplotlib seaborn tqdm

echo "=== 5. Verification GPU ==="
python -c '
import torch
print("PyTorch version:", torch.__version__)
print("CUDA disponible:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Nombre de GPU:", torch.cuda.device_count())
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}:", torch.cuda.get_device_name(i))
'

echo "=== Configuration terminee avec succes ! ==="
