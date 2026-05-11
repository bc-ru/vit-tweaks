#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH="$(pwd)/src:${PYTHONPATH:-}"
BASE="runs/aid_vit_small/best.pt"
mkdir -p checkpoints results logs
for mode in ffn attn all; do
  for r in 0.75 0.50 0.25; do
    tag="vit_small_${mode}_r${r}"
    python -m aerial_svd.compress \
      --config configs/aid_vit_small.yaml \
      --checkpoint "$BASE" \
      --mode "$mode" \
      --rank-ratio "$r" \
      --output "checkpoints/${tag}.pt" | tee "logs/${tag}_compress.log"
    python -m aerial_svd.evaluate \
      --config configs/aid_vit_small.yaml \
      --checkpoint "checkpoints/${tag}.pt" \
      --svd-mode "$mode" \
      --rank-ratio "$r" \
      --max-batches 50 \
      --output "results/${tag}.json" | tee "logs/${tag}_eval.log"
  done
done
