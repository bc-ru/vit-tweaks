#!/usr/bin/env bash
# Dependency-free checks, plus bounded native tests in the existing CPU image.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
for file in scripts/*.sh fix_permissions.sh; do bash -n "$file"; done
PYTHONPYCACHEPREFIX=/tmp/aerial-svd-pycache python3 -m py_compile src/aerial_svd/*.py tests/*.py
python3 -m unittest discover -s tests -p 'test_workspace.py' -v
scripts/run.sh cpu python -m unittest discover -s tests -p test_lowrank_stages.py -v
scripts/run.sh cpu python -m unittest discover -s tests -p test_jax_cli.py -v
scripts/run.sh cpu python -m unittest discover -s tests -p test_lowrank_experiments.py -v
scripts/run.sh cpu python -m unittest discover -s tests -p test_activation_svd.py -v
scripts/run.sh cpu python -m unittest discover -s tests -p test_lowrank_final.py -v
git diff --check
