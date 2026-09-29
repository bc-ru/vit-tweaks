# JAX migration

The three configured models now have a native JAX runtime: ViT-Small, ViT-Base,
ResNet-18. Training, evaluation, structural SVD compression, recovery fine-tuning,
inspection, spectra, approximation curves, randomized SVD, and layer-output error
are supported. Select it with `--backend jax` on the existing module commands.
Torch remains the default for old commands/checkpoints and the one-time converter;
the JAX images contain no Torch or timm installation.

## Build and convert

Build from a small context (works with this host's legacy Docker builder):

```bash
tar -cf - Dockerfile.jax requirements-jax.txt | docker build -f Dockerfile.jax -t aerial-svd-jax:cpu -
tar -cf - Dockerfile.jax requirements-jax.txt | docker build --build-arg JAX_PLATFORM=cuda12 -f Dockerfile.jax -t aerial-svd-jax:cuda12 -
```

The build downloads packages. It copies only libjpeg and its headers from the
existing reference image, then compiles Pillow against that decoder. This matters:
the stock Pillow wheel's libjpeg-turbo changes decoded JPEG pixels and consequently
layer-error results. No reference Python packages enter the JAX image.

Inside the **existing PyTorch container**, convert trusted local checkpoints:

```bash
PYTHONPATH=src python -m aerial_svd.convert_checkpoint \
  --config configs/aid_vit_small.yaml --checkpoint runs/aid_vit_small/best.pt \
  --output checkpoints/jax/vit_small.npz
```

This also accepts legacy compressed checkpoints, preserving their actual factors.
Conversion validates the complete parameter/buffer tree; missing or unexpected keys
are errors. It does not translate Torch optimizer state. For a new pretrained
baseline, export initialization separately (timm may download pretrained weights):

```bash
PYTHONPATH=src python -m aerial_svd.convert_checkpoint \
  --config configs/aid_vit_small.yaml --pretrained \
  --output checkpoints/jax/vit_small_pretrained.npz
```

Use corresponding configs for ViT-Base and ResNet-18. The saved split establishes the
30-class classifier when no class count is given. Native `pretrained: false` training
also works; its seed-42 random stream is not Torch's stream.

## Run the native workflow

Start the GPU container (check the shared GPU is free first):

```bash
docker run --rm --gpus all --ipc=host -it \
  -e XLA_PYTHON_CLIENT_PREALLOCATE=false \
  -v "$PWD:/workspace" -w /workspace aerial-svd-jax:cuda12 bash
```

For CPU use `aerial-svd-jax:cpu` and omit `--gpus all`. `PYTHONPATH` is set in the
images. The following commands run inside the chosen container:

```bash
python -m aerial_svd.train --backend jax --config configs/aid_vit_small.yaml \
  --init-checkpoint checkpoints/jax/vit_small_pretrained.npz --output-dir runs/jax/vit_small_v1

python -m aerial_svd.evaluate --backend jax --config configs/aid_vit_small.yaml \
  --checkpoint checkpoints/jax/vit_small.npz --output results/jax/vit_small.json

python -m aerial_svd.compress --backend jax --config configs/aid_vit_small.yaml \
  --checkpoint checkpoints/jax/vit_small.npz --mode ffn --rank-ratio 0.25 \
  --output checkpoints/jax/vit_small_ffn025.npz

python -m aerial_svd.finetune_compressed --backend jax --config configs/aid_vit_small.yaml \
  --compressed-checkpoint checkpoints/jax/vit_small_ffn025.npz --svd-mode ffn --rank-ratio 0.25 \
  --epochs 5 --output-dir runs/jax/recovery_v1

python -m aerial_svd.inspect --backend jax --config configs/aid_vit_small.yaml \
  --checkpoint checkpoints/jax/vit_small.npz --output reports/jax/vit_small.json

for module in analyze_spectra approximation_curves randomized_svd_benchmark layer_error_analysis; do
  python -m aerial_svd.$module --backend jax --config configs/aid_vit_small.yaml \
    --checkpoint checkpoints/jax/vit_small.npz --out-dir matrix_results/jax/$module
 done

python -m aerial_svd.summarize --results-dir results/jax --csv results/jax/summary.csv --output-dir plots/jax
```

Use **fresh output paths**. Training directories and native checkpoint/result files
refuse replacement. Training writes immutable `epoch-NNNN.npz` files and advances
its own `best.npz`/`last.npz` links. A `complete.json` marker identifies successful
training/layer analysis; an interrupted run retains checkpoints and its initial
`provenance.json`. Other matrix commands mark their provenance status complete.

For smoke runs, use a copied config with small batches plus `--max-train-batches 1
--max-val-batches 1 --max-test-batches 1`. Evaluation has `--batch-size`,
`--max-test-batches`, `--warmup-batches`, and `--max-batches`; the last only caps timing.
Layer-error analysis intentionally inherits saved checkpoint configuration except
`data_dir`, matching the reference; its batch/target caps remain available.

`--resume` preserves the old weights-only behavior. Add `--restore-training-state`
to restore native optimizer/scaler/epoch/history state into a **new output directory**.
The original epoch count, optimization settings, batch caps, device platform, image
decoder, ordering, and split must match. The earlier best checkpoint is retained by
reference, so preserve the source run while the resumed run depends on it.

## Preserved contracts and differences

- Keep seed 42 and `runs/aid_split_seed42.json`; the native loader never regenerates
  the split. It validates the 30-class mapping, full partition, and ImageFolder order.
- Evaluation resize/normalization match the reference. Augmentation distributions
  and train/drop-last behavior are retained; random draws/shuffling differ across
  frameworks. Native augmentation is deterministic by seed, epoch, and sample index.
- AdamW decays all parameters; cosine LR steps per epoch. Label smoothing, sample
  weighting, and discarded incomplete gradient-accumulation groups match the source.
  CPU uses float32; GPU AMP uses float16 operations with dynamic loss scaling and
  float32 normalization/residual paths. ResNet buffers include unbiased running variance.
- Low-rank layers retain symmetric sqrt(S) factors, bias placement, rank rounding,
  target selection, and the no-saving skip. Native checkpoints are pickle-free NPZ,
  not `.pt`; schemas/version, source hashes, config, and backend metadata are recorded.
- Matrix SVD stays float32 and spectral energy reductions stay float64. Analytic
  approximation curves remain analytic. Layer-error mean/max count all vectors;
  p95 uses only the first capped vectors, as before.
- JAX randomized-SVD seed 42 does not reproduce Torch projections. Numerical parity
  tests use a shared projection. Timings synchronize and exclude compilation and
  input transfer. Do not mix them with historical timing rows without a controlled run.
- Parameter counts/bytes are exact. `peak_gpu_memory_bytes` is null: JAX has no
  equivalent resettable Torch allocator-peak measurement in this implementation.
  `jax_process_peak_gpu_memory_bytes` reports the available process-lifetime JAX
  allocator peak, including setup/evaluation; it is deliberately a separate metric.
  NPZ checkpoint sizes include different serialization/optimizer overhead from `.pt`.
- Reports/CSV columns and plots are retained. Keep backend result directories separate.
  Historical artifacts and the original Torch commands remain available for rollback.

Standalone analysis can still use a small matrix archive: `export_matrices` in the
reference container, then `--backend jax --matrix-archive FILE.npz` on the three
matrix commands. Neither full models nor a Torch installation are required there.

## Evidence

Progress, exact commands, fixtures, and logs: `.scratch/jax-migration/handoff.md`.
Focused tests are in `tests/`; numerical cross-framework tests require both runtimes.
The JAX CLI test can run in the standalone image:

```bash
python -m unittest discover -s tests -p test_jax_cli.py -v
```

CPU/GPU model parity, bounded training, compression/recovery, full-state resume,
preprocessing, and matrix/layer metrics are checked. Full training experiments and
performance comparisons are deliberately separate from migration validation.

Accepted validation: 17 regression tests; standalone CPU image tests; all three
models on CPU and RTX 4070 in float32/AMP; bounded GPU training for all three;
CPU recovery and GPU all-mode compression/evaluation. Float32 GPU logit differences
were below 2e-6; AMP differences from Torch AMP were at most 0.00390625 on the fixture.
Five real JPEG tensors and full-state CPU resume were bitwise identical. Six
layer-error rows passed the original `rtol=3e-5, atol=2e-6` gate (maximum absolute
difference 1.91e-7). These are migration checks, not accuracy or speed benchmarks.
