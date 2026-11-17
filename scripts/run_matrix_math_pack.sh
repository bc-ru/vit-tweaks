#!/usr/bin/env bash
# Historical Torch workflow; current JAX stage commands are in README.md.
set -euo pipefail

bash scripts/analyze_spectra.sh
bash scripts/approximation_curves.sh
bash scripts/randomized_svd_benchmark.sh
bash scripts/layer_error_analysis_vit_small.sh
