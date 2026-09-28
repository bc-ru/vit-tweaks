#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"

python -m aerial_svd.analyze_spectra --config configs/aid_vit_small.yaml --device cpu --out-dir matrix_results/spectra
python -m aerial_svd.analyze_spectra --config configs/aid_vit_base.yaml  --device cpu --out-dir matrix_results/spectra
python -m aerial_svd.analyze_spectra --config configs/aid_resnet18.yaml  --device cpu --out-dir matrix_results/spectra
