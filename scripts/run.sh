#!/usr/bin/env bash
# One environment boundary; experiment logic stays in independently invoked modules.
set -euo pipefail
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
usage() { echo "Usage: scripts/run.sh cpu|gpu|torch|torch-gpu COMMAND [ARG...]" >&2; }
if [[ $# -lt 2 ]]; then usage; exit 2; fi
profile="$1"
shift
case "$profile" in
  cpu) image=aerial-svd-jax:cpu ;;
  gpu) image=aerial-svd-jax:cuda12 ;;
  torch|torch-gpu) image=ghcr.io/bc-ru/pytorch-2.1.0-cuda11.8-cudnn8-runtime-timm-models:stable ;;
  *) usage; exit 2 ;;
esac
# Optional per-stage floor for long series sharing the system SSD.
if [[ -n "${AERIAL_SVD_MIN_FREE_BYTES:-}" ]]; then
  available=$(df -B1 --output=avail "$repo_root/artifacts" | tail -1)
  if ! [[ "$AERIAL_SVD_MIN_FREE_BYTES" =~ ^[0-9]{1,18}$ ]]; then
    echo "AERIAL_SVD_MIN_FREE_BYTES must be a nonnegative byte count" >&2
    exit 2
  fi
  if (( available < 10#$AERIAL_SVD_MIN_FREE_BYTES )); then
    echo "Insufficient disk space: $available bytes available, $AERIAL_SVD_MIN_FREE_BYTES required" >&2
    exit 1
  fi
fi
mkdir -p "$repo_root/artifacts"/{runs,checkpoints,results,matrix_results,plots,reports,logs,experiments}
docker_args=(run --rm --pull never --init --network none --ipc=host --user "$(id -u):$(id -g)"
  --volume "$repo_root:/workspace" --workdir /workspace
  --env PYTHONPATH=/workspace/src --env PYTHONDONTWRITEBYTECODE=1
  --env "AERIAL_SVD_GIT_COMMIT=$(git -C "$repo_root" rev-parse HEAD)"
  --env OPENBLAS_NUM_THREADS=1 --env OMP_NUM_THREADS=2
  --env MPLBACKEND=Agg --env MPLCONFIGDIR=/tmp/matplotlib
  --env HF_HOME=/workspace/.cache/huggingface
  --env XLA_PYTHON_CLIENT_PREALLOCATE=false)
if [[ "$profile" == gpu || "$profile" == torch-gpu ]]; then
  if ! processes="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)"; then
    echo "Cannot inspect GPU access. Retry outside a sandbox hiding /dev/nvidia*; do not infer a driver fault." >&2
    exit 1
  fi
  if [[ -n "$processes" ]]; then echo "GPU has active compute processes; no container started." >&2; exit 1; fi
  docker_args+=(--gpus all)
else
  docker_args+=(--env JAX_PLATFORMS=cpu)
fi
exec docker "${docker_args[@]}" "$image" "$@"
