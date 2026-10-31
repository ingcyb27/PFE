#!/usr/bin/env bash
# lancer_orange.sh - Volet Orange du Sprint 4 : DP-SGD + audits + utilite + figures.
#
# Usage (sur rst-goose, environnement conda active) :
#   bash lancer_orange.sh                       # tout : 3 entrainements puis audits (environ 4 a 5 h)
#   EPOCHS=10 bash lancer_orange.sh             # essai rapide (10 epoques)
#   EPSILONS="8" bash lancer_orange.sh          # un seul niveau de DP
#
# Un modele deja entraine (runs/<tag>/model/) n'est pas re-entraine : on peut relancer apres une coupure.
# S'arrete a la PREMIERE erreur : jamais de resultats simules.
set -euo pipefail
cd "$(dirname "$0")"

GPU="${GPU:-1}"
EPOCHS="${EPOCHS:-50}"
EPSILONS="${EPSILONS:-8 2}"
VARIANTS="${VARIANTS:-10000}"
SPRINT3_CKPT="${SPRINT3_CKPT:-/tmp/pfe_checkpoints/ep50}"   # modele du Sprint 3 (comparaison)
MODEL_NAME="${MODEL_NAME:-EleutherAI/pythia-160m}"
EXTRA_ARGS="${EXTRA_ARGS:-}"                                 # ex. "--cpu" pour un test sans GPU

export CUDA_VISIBLE_DEVICES="$GPU"
export PYTHONUNBUFFERED=1
mkdir -p logs
echo "=== $(date) | GPU $GPU | $EPOCHS epoques | epsilons : $EPSILONS | $VARIANTS leurres ==="
python check_env.py 2>&1 | tee logs/00_env.log

entrainer () {   # $1 = tag, la suite = options de train_dp.py
  local tag="$1"; shift
  if [ -f "runs/$tag/model/config.json" ]; then
    echo "[+] runs/$tag/model existe deja : entrainement saute"
  else
    echo "=== $(date) | entrainement $tag ==="
    python train_dp.py --tag "$tag" --model_name "$MODEL_NAME" --epochs "$EPOCHS" "$@" $EXTRA_ARGS 2>&1 | tee "logs/train_$tag.log"
  fi
}

auditer () {     # $1 = tag, $2 = chemin du modele (optionnel)
  local tag="$1"; local mp="${2:-}"
  local opt=(); [ -n "$mp" ] && opt=(--model_path "$mp")
  echo "=== $(date) | audits $tag ==="
  python eval_exposure.py --tag "$tag" "${opt[@]}" --variants "$VARIANTS" $EXTRA_ARGS 2>&1 | tee "logs/exposure_$tag.log"
  python eval_mia.py      --tag "$tag" "${opt[@]}" $EXTRA_ARGS 2>&1 | tee "logs/mia_$tag.log"
  python eval_utilite.py  --tag "$tag" "${opt[@]}" $EXTRA_ARGS 2>&1 | tee "logs/utilite_$tag.log"
}

# 1. Entrainements (reference sans DP, puis un modele par epsilon)
entrainer nodp --no_dp
for E in $EPSILONS; do entrainer "dp_eps$E" --target_epsilon "$E"; done

# 2. Audits : modele d'origine, modele du Sprint 3 (s'il est accessible), puis les nouveaux modeles
if [ "$MODEL_NAME" = "EleutherAI/pythia-160m" ]; then auditer base; fi
if [ -f "$SPRINT3_CKPT/config.json" ]; then auditer sprint3_ep50 "$SPRINT3_CKPT"; else echo "[!] $SPRINT3_CKPT absent : comparaison Sprint 3 sautee"; fi
auditer nodp
for E in $EPSILONS; do auditer "dp_eps$E"; done

# 3. Figures et tableau
python plot_orange.py 2>&1 | tee logs/figures.log
echo "=== $(date) : TERMINE - resultats dans figures/resume.md ==="
