#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH="$(pwd)/src:${PYTHONPATH:-}"
mkdir -p results logs
python3 -m aerial_svd.inspect --config configs/aid_vit_small.yaml --rank-ratio 0.5 --output results/inspect_vit_small.json | tee logs/inspect_vit_small.log
python3 -m aerial_svd.inspect --config configs/aid_vit_base.yaml --rank-ratio 0.5 --output results/inspect_vit_base.json | tee logs/inspect_vit_base.log
