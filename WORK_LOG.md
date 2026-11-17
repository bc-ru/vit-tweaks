# Work log

## Current state

- Current protocol: JAX ViT-Small, exact SVD, frozen AID seed-42 split (30 classes).
  Stages **00–02 and exact-SVD 04–08 are complete**: 168 validation configurations,
  9 factorized fine-tunes + dense control (5 epochs each), RTX 4070 benchmarks.
- Measured report/CSV/plots: `docs/lowrank/results/aid_svd_20261117/REPORT_RU.md`.
  Full checkpoints, logits and manifests: `artifacts/experiments/aid_svd_20261117/`.
  Source weights, split and historical compatibility links are unchanged.
- Activation-aware 03 and held-out final-test 09 remain separate. No test evaluation;
  inference GPU allocator peak is not measured (parameter bytes are exact).
- GPU and non-root Docker work. GitHub App publication works; shell SSH has no key.

## Outcomes

### 2026-11-17 — Exact-SVD research series complete

Implemented independent 04–08 stages using native checkpoints, compression and
Engine. Completed 24 global + 48 regional + 96 single replacements on the same
1500 validation images, 9 factorized five-epoch fine-tunes plus dense control,
and 8 models × batches 1/16 × 50 synchronized inference trials. Full-rank controls
all pass (21 cases; maximum relative logit error 1.71e-6). Test never evaluated.

Proof: `scripts/check.sh` passes 19 tests, including real factor training, exact
optimizer recovery and nonfinite-state rejection. Main-series audit: 340 complete
stages / 1698 matching input-output hashes. Training audit: 50 epochs / 350000
images / 43750 updates, none skipped. Snapshot: 169 model rows, 384 curve points,
1248 matrix-error rows, 10 fine-tunes, 16 benchmark rows, 6 plots.

Results and limits: `docs/lowrank/results/aid_svd_20261117/REPORT_RU.md`; raw evidence
in `artifacts/experiments/aid_svd_20261117/`. Recovery is partial for strong global
compression; middle-FFN CE worsens despite a small accuracy increase. Fewer
parameters do not ensure lower batch-1 latency. Held-out test and inference-only
GPU peak remain unmeasured. Commands: `docs/lowrank/RUN_STAGES_RU.md`.

### 2026-11-17 — Selected skill restoration

Restored the 12 Matt Pocock skills previously installed in this repo; other helpers
remain archived. Preserved invocation policies and connected Qwen with a relative
link. Repaired missing workflow/vault references and made WORK_LOG.md the default
continuity record. All 12 skills pass frontmatter, reference and policy checks;
no model/runtime changes or new upstream skills.

### 2026-11-17 — Repository cleanup

One artifact tree, one agent instruction source (`AGENTS.md`, shared by Qwen), and
this work record. Removed 3,389,888,192 bytes of disposable migration checkpoints;
kept source weights, the frozen split and research evidence. Old instructions,
imports and migration evidence are archived; docs/build files are now visible to
Git. New container runs and both Dev Containers use the host UID/GID.

Proof: 31 tests pass (16 workspace/native + 15 cross-framework), both development
images build/run as UID 1000, GPU launcher smoke passes, ownership repair changes
only its artifact fixture and leaves a source-directory symlink target untouched.
All 129 pre-existing research files retain their SHA-256 and original path access.
Commands: `scripts/check.sh`; optional parity commands in `docs/jax.md`.

### 2026-11-17 — GPU access diagnosis

Driver 555.42.02 / RTX4070 works. The earlier failure was sandbox device isolation.
JAX CUDA detection and synchronized 16×16 multiplication pass outside the sandbox.
This establishes GPU access, not GPU validation of the scientific stage results.

### 2026-11-17 — Low-rank protocol 00–02

Independent baseline, factor-check and matrix-curve CLIs implemented. All selected
JAX matrices factorize, including non-saving/full ranks; `all_blocks` excludes head.
Baseline validation top-1 is 0.9733333333 on 1500 images. Full-rank reconstruction,
logits, serialization, bias and immutable outputs pass the accepted 26-test suite.
Details: `docs/lowrank/VALIDATION_RU.md`; implementation commit `ae8f2e4`.

## Recording rule

After substantive work, add one short outcome entry with proof and any remaining
issue; update current state when needed. Link code/results instead of copying logs,
commands already documented elsewhere, plans, tool calls or a chronological transcript.
