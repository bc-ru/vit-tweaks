#!/usr/bin/env bash
# Historical Torch workflow; current JAX stage commands are in README.md.
set -euo pipefail
export PYTHONPATH="$(pwd)/src:${PYTHONPATH:-}"
python -m aerial_svd.train --config configs/aid_vit_base.yaml
