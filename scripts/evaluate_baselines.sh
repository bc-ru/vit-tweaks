#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH="$(pwd)/src:${PYTHONPATH:-}"
mkdir -p results logs
python -m aerial_svd.evaluate \
  --config configs/aid_vit_small.yaml \
  --checkpoint runs/aid_vit_small/best.pt \
  --max-batches 50 \
  --output results/vit_small_baseline.json | tee logs/vit_small_baseline_eval.log
python -m aerial_svd.evaluate \
  --config configs/aid_vit_base.yaml \
  --checkpoint runs/aid_vit_base/best.pt \
  --max-batches 50 \
  --output results/vit_base_baseline.json | tee logs/vit_base_baseline_eval.log
