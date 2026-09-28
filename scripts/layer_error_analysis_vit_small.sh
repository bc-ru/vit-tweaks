#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"

python -m aerial_svd.layer_error_analysis \
  --config configs/aid_vit_small.yaml \
  --device cuda \
  --out-dir matrix_results/layer_error \
  --include-regex 'mlp.fc|attn.qkv|attn.proj' \
  --rank-ratios 0.25 0.50 0.75 \
  --max-targets 16 \
  --max-batches 4 \
  --amp
