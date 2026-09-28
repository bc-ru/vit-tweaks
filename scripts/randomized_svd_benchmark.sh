#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"

python -m aerial_svd.randomized_svd_benchmark --config configs/aid_vit_small.yaml --device cuda --out-dir matrix_results/randomized_svd --rank 64 --max-matrices 12
python -m aerial_svd.randomized_svd_benchmark --config configs/aid_vit_base.yaml  --device cuda --out-dir matrix_results/randomized_svd --rank 128 --max-matrices 12
