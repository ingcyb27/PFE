#!/bin/bash
export PATH="$HOME/.conda/envs/pfe_memorization/bin:$PATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd ~/PFE/week4/orange
EPSILONS="32 128" bash lancer_orange.sh
