# Work log

## Current state

- Current protocol: JAX ViT-Small, exact SVD, frozen AID seed-42 split (30 classes).
  Stages **00–02 and exact-SVD 04–08 work**; the 168-candidate validation series
  is running in `artifacts/experiments/aid_svd_20261117/`. Activation-aware 03
  and final-test 09 remain separate; test has not been evaluated.
- Start at `README.md`; scientific requirements/commands are in `docs/lowrank/`.
  Results and source weights are under `artifacts/`; legacy root paths still work.
- GPU and non-root Docker access work. Container ownership repair works without
  host sudo. Remote publication remains dependent on SSH-agent signing access.

## Outcomes

### 2026-11-17 — Exact-SVD candidates and validation series

Implemented independent candidate/validation/fine-tuning/benchmark/aggregation
stages, reusing native checkpoint, compression and Engine. Per-stage immutable
outputs, paired pixel/index hashes, full-rank output controls and epoch optimizer
recovery. Existing 00–02 verified all 48 matrices and 384 spectral-tail points;
GPU dense validation reproduces 0.9733333333 accuracy / 0.1984744517 CE.

Proof: `scripts/check.sh` passes 19 tests, including real factor training and
continuous/resumed checkpoint equality. All 24 global configurations are complete; full-rank controls restore accuracy
and have relative logit error at most 1.71e-6. The 48/96 series is in progress under
`artifacts/experiments/aid_svd_20261117/`; pending completion, 9 fine-tunes + dense
control, resource measurements and the measured report. Commands/schemas:
`docs/lowrank/RUN_STAGES_RU.md`. No test images or historical results touched.

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
