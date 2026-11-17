# Work log

## Current state

- Current protocol: JAX ViT-Small, exact SVD, frozen AID seed-42 split (30 classes).
  Stages **00–02 work**; stages 03–09 remain planned. No full sweep or final test.
- Start at `README.md`; scientific requirements/commands are in `docs/lowrank/`.
  Results and source weights are under `artifacts/`; legacy root paths still work.
- GPU and non-root Docker access work. Container ownership repair works without
  host sudo. Remote publication remains dependent on SSH-agent signing access.

## Outcomes

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
