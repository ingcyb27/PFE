#!/usr/bin/env bash
# lancer_audit.sh - Lance tout le volet Fatima-Zahra (Sondes 3 et 4) sur les checkpoints d'Emmanuel.
#
# Usage (environnement conda active) :
#   bash lancer_audit.sh                          # GPU 1, epoques 0 1 5 10 20 50, 10 000 variantes
#   VARIANTS=2000 bash lancer_audit.sh            # plus rapide
#   EPOCHS="1 20 50" bash lancer_audit.sh         # seulement certaines epoques
#
# S'arrete a la PREMIERE erreur : jamais de resultats simules.
set -euo pipefail
cd "$(dirname "$0")"

GPU="${GPU:-1}"
EPOCHS="${EPOCHS:-0 1 5 10 20 50}"
VARIANTS="${VARIANTS:-10000}"
SOURCE_CKPT="${SOURCE_CKPT:-/tmp/pfe_checkpoints}"
EXTRA_ARGS="${EXTRA_ARGS:-}"          # ex. "--cpu" pour un test sans GPU

export CUDA_VISIBLE_DEVICES="$GPU"
export PYTHONUNBUFFERED=1
mkdir -p logs checkpoints
echo "=== $(date) | GPU $GPU | epoques : $EPOCHS | $VARIANTS variantes ==="

python check_env.py 2>&1 | tee logs/00_env.log

# 1. Recuperer les checkpoints d'Emmanuel s'ils ne sont pas deja la
if [ -z "$(ls -A checkpoints 2>/dev/null)" ]; then
  if [ -d "$SOURCE_CKPT" ]; then
    echo "[+] Copie des checkpoints depuis $SOURCE_CKPT ..."
    cp -r "$SOURCE_CKPT"/. checkpoints/
  else
    echo "[ERREUR] Aucun checkpoint dans checkpoints/ et $SOURCE_CKPT introuvable." >&2
    exit 1
  fi
fi
echo "[+] Checkpoints disponibles : $(ls checkpoints | tr '\n' ' ')"

# 2. Sondes 3 et 4 pour chaque epoque
for E in $EPOCHS; do
  if [ "$E" != "0" ] && [ ! -f "checkpoints/ep$E/config.json" ]; then
    echo "[!] checkpoints/ep$E absent : epoque $E ignoree"
    continue
  fi
  echo "=== $(date) | epoque $E ==="
  python eval_exposure.py --epoch "$E" --variants "$VARIANTS" $EXTRA_ARGS 2>&1 | tee "logs/exposure_ep$E.log"
  python eval_mia.py --epoch "$E" $EXTRA_ARGS 2>&1 | tee "logs/mia_ep$E.log"
done

# 3. Figures et tableaux
python plot_resultats.py 2>&1 | tee logs/figures.log
echo "=== $(date) : TERMINE - resultats dans results/fatimazahra/ ==="
