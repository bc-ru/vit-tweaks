#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH="$(pwd)/src:${PYTHONPATH:-}"
python -m aerial_svd.summarize --results-dir results --output-dir plots --csv results/summary.csv
