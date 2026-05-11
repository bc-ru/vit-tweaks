#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH="$(pwd)/src:${PYTHONPATH:-}"
python -m aerial_svd.finetune_compressed \
  --config configs/aid_vit_small.yaml \
  --compressed-checkpoint checkpoints/vit_small_ffn_r0.50.pt \
  --svd-mode ffn \
  --rank-ratio 0.50 \
  --epochs 5 \
  --lr 1e-5 \
  --output-dir runs/aid_vit_small_ffn_r050_recovery
