# JAX runtime

The native runtime supports ViT-Small, ViT-Base and ResNet-18. Existing module CLIs
accept `--backend jax`; the independent `lowrank_*` stages are JAX-only. Torch is
still available through `scripts/run.sh torch` (CPU) or `torch-gpu`.

## Images and ownership

Build from the repository root, with the existing small contexts:

```bash
tar -cf - Dockerfile.jax requirements-jax.txt | docker build -f Dockerfile.jax -t aerial-svd-jax:cpu -
tar -cf - Dockerfile.jax requirements-jax.txt | docker build --build-arg JAX_PLATFORM=cuda12 -f Dockerfile.jax -t aerial-svd-jax:cuda12 -
```

Building downloads pinned dependencies. JAX/jaxlib are 0.4.38. Pillow is compiled
against reference libjpeg9e: stock libjpeg-turbo wheels change decoded pixels.
Normal work uses `scripts/run.sh` with the host UID/GID, not root. Images are never
pulled implicitly by the launcher. For VS Code rebuild/reopen the JAX or historical
Torch Dev Container; `remoteUser=research`, `updateRemoteUserUID=true` synchronize
ownership. See the official [Dev Containers user specification](https://containers.dev/implementors/json_reference/#general-devcontainerjson-properties).

## Trusted checkpoint conversion

Never load NPZ as `.pt`, or treat a matrix archive as a model checkpoint. Export
trusted local Torch checkpoints with the existing converter:

```bash
scripts/run.sh torch python -m aerial_svd.convert_checkpoint \
  --config configs/aid_vit_small.yaml --checkpoint runs/aid_vit_small/best.pt \
  --output artifacts/checkpoints/jax/vit_small_new.npz
```

Use a new output path. The converter preserves class order and records the source
hash/config. It validates the entire parameter/buffer tree; Torch optimizer state
is not translated. The frozen split stays `runs/aid_split_seed42.json`.

Reusable converted ViT-Small/Base/ResNet weights are in `artifacts/checkpoints/jax/`.
The current ViT-Small protocol source is `vit_small_protocol_v1.npz`; its hash and
measured validation are in `docs/lowrank/VALIDATION_RU.md`.

## Compatibility

- Names/layouts match timm; only the three exact architectures in `model_spec.py`
  are supported. Native model/state ownership is `jax_models.py` / `jax_checkpoint.py`.
- CPU uses float32. GPU AMP and the original Torch RNG are not numerically identical.
  Seed, split, class order, transforms and scalar metric definitions remain fixed.
- JAX factors every requested valid rank, including non-saving/full ranks.
  `all_blocks` excludes head; legacy `all` includes it. Bias and symmetric sqrt(S)
  splitting stay unchanged. Loading checks the actual low-rank structure.
- Historical skip-based JAX verification is opt-in with `--legacy-skip-non-saving`.
  Old Torch commands still have the skip; keep those results separate.
- Analytic matrix curves use float32 spectra and float64 energy sums. Full-rank
  analytic zero error is different from measured factor reconstruction rounding.
- Old `evaluate` and `layer_error_analysis` use test. Do not use them for selection;
  stages 00 and 05 use validation. Exact-SVD stages 04–08 are described in the stage guide;
  activation-aware 03–08 reuse these interfaces (see [commands](lowrank/ACTIVATION_STAGES.md));
  final-test 09 remains separate and unexecuted.
- Process-lifetime JAX allocator peaks are not inference-only memory measurements.
  No resource speedup follows from counting parameters.

## Verification

`scripts/check.sh` runs dependency-free checks and the Torch-free native tests.
The existing local cross-framework environment is cached in `.cache/jax-torch/`;
it is deliberately outside scratch and is not a checkpoint/input dependency.
To run the existing parity tests with both runtimes:

```bash
scripts/run.sh torch env \
  PYTHONPATH=/workspace/src:/workspace/.cache/jax-torch/lib/python3.10/site-packages \
  python -m unittest discover -s tests -p test_jax_analysis.py -v
scripts/run.sh torch env \
  PYTHONPATH=/workspace/src:/workspace/.cache/jax-torch/lib/python3.10/site-packages \
  python -m unittest discover -s tests -p test_jax_training.py -v
```

For a new host without that optional cache, create it *inside the reference Torch
container* (never install into host Python), then verify decoder parity:

```bash
scripts/run.sh torch python -m venv --system-site-packages .cache/jax-torch
# Dependency download is explicit; the normal launcher intentionally has no network.
docker run --rm --user "$(id -u):$(id -g)" \
  -v "$PWD:/workspace" -w /workspace \
  -e CFLAGS=-I/opt/conda/pkgs/jpeg-9e-h5eee18b_1/include \
  -e LDFLAGS=-L/opt/conda/lib \
  ghcr.io/bc-ru/pytorch-2.1.0-cuda11.8-cudnn8-runtime-timm-models:stable \
  .cache/jax-torch/bin/python -m pip install --no-binary=pillow -r requirements-jax.txt
```

Compilation of Pillow needs the reference JPEG headers/libs (the reference image's
normal include directory contains incompatible turbo headers). Check the rebuilt
cache with `from PIL import features; assert features.version_codec("jpg") == "9.0"`
and the cross-framework tests; do not accept a different decoder silently. The cache is
optional; its absence does not prevent native tests or protocol stages 00–02.
Historical migration evidence/notes are archived; they are not current run commands.
