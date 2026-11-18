# Work log

## Current state

- ViT-Base replication is complete through frozen test and audit. [Final Base results](docs/lowrank/results/aid_base_final_20261118/REPORT_RU.md) and [matched comparison](docs/lowrank/results/aid_base_final_20261118/small_vs_base/REPORT_RU.md). All new local weights and stage outputs are under its single **ViT-Base-AID** directory. [Storage layout and resume guide](docs/lowrank/BASE_STAGES_RU.md). The ordinary series uses `aid_base_svd_20261118_ntfs` because the original name holds the [initial preflight](docs/lowrank/results/aid_base_svd_20261118/PREFLIGHT_RU.md).

- Original research plan **complete through final test**: JAX ViT-Small/AID,
  frozen seed-42 split, 30 classes; 168 SVD + 168 activation-SVD configurations,
  6 allocated models, 12 fine-tunes including dense control. No new methods.
- Unified report/tables/plots: `docs/lowrank/results/aid_final_20261118/REPORT_RU.md`.
  Commands: `docs/lowrank/FINAL_STAGES.md`. New local evidence:
  `artifacts/experiments/aid_final_20261118/`; earlier series remain unchanged.
- GPU: all 48 global configurations, 6 allocated, 12 fine-tunes and dense;
  134 isolated model/batch measurements (1/16). Exact parameter/NPZ bytes,
  XLA memory plans and lifetime allocator counters are separate. Inference-only
  actual GPU peak remains unavailable, explicitly null.
- Final test: 20 models fixed before first access, each on all 1500 examples;
  frozen selection/checkpoint/config/split identities and paired pixels/order pass.
  No further experiments after test; presentation unchanged.

## Outcomes

### 2026-11-19 — ViT-Base final test and paired architecture comparison

Verified 168 + 168 validation configurations, 6 exact allocations, 12 fine-tunes
(60 epochs, 52,500 updates, zero skipped), 134 benchmark rows and 20 frozen test
models. Dense Base validation/test top-1 was 97.133% / 97.667%. At all-blocks
rank .5 ordinary/activation test top-1 was 95.000% / 97.867% before training
and 97.133% / 97.067% after training. The local budget objective fell while
validation top-1 declined in all six equal-budget pairs. The final
[audit](docs/lowrank/results/aid_base_final_20261118/audit.json) verified test
metrics from logits, matched Small inputs and preserved 1877 Small outputs.
All 28 native/workspace checks and 15 cross-framework tests passed.

### 2026-11-18 — ViT-Base replication on the Windows volume

The dedicated NTFS output directory had 525,895,241,728 bytes free. Native NPZ
atomic publication, locking and overwrite refusal passed through the non-root
container. The launcher checks space on the output volume and the system disk.
On all 1,500 validation images, dense Base scored 97.133% with CE 0.197622.
Inference passed at batch 1/16 and an all-blocks rank-384 training smoke completed
one finite FP32 AdamW update at batch 8. The paired validation grids are in
progress. No final Base test has started.

### 2026-11-18 — ViT-Base resource and input verification

All 152 FP32 tensors match the existing AID Torch best checkpoint bit for bit.
The archived epoch-9 Torch validation top-1 was 97.133%. New Base measurements
are pending disk capacity. The launcher accepts a checkpoint-specific reference
and a per-stage free-space floor. Reports now read architecture and results
from the series. CPU regression, parity and archived final-report checks passed.
The three Small series retain all 1877 declared outputs and 49 public files.

### 2026-11-18 — Full resource comparison and final held-out evaluation

Extended existing benchmark with input hashes, separate compilation time, XLA
memory plan and lifetime allocator counters; independent catalog/freeze/test/report
stages preserve the working pipeline. Completed 134 GPU measurements and 20 final
test evaluations, selected before test as requested plus dense training control.
Published snapshot contains all 355 models, exact rank maps, matrix/local errors,
FT, resource/test CSVs and four plots in the unified report linked above.

Test top-1/CE: dense 96.800%/.20257; all_blocks .5 after FT ordinary
95.600%/.24499 vs activation 96.133%/.22594; FFN .5 after FT 96.067%/.23475
vs 96.400%/.22011. Allocated .5 remains a negative result: ordinary 12.133%,
activation 88.733%. Dense vs all_blocks .5 latency batch 1/16: 1.790/13.329 ms,
ordinary 1.820/11.627, activation 1.798/11.726. Fewer parameters do not ensure
batch-1 speedup. Actual isolated inference GPU peak is not claimed.

Proof: `scripts/check.sh` passed 26 tests; final focused regressions passed 3 tests,
including complete ordered synthetic test, checkpoint/config/selection tampering,
pixel mismatch, frozen reselection refusal and benchmark protocol rejection.
Read-only final audit: 883 complete stages, 8970 identity references / 2763 unique
files; original checkpoint/split/previous results and frozen list hashes match.
Audit: `artifacts/experiments/aid_final_20261118/11_audit/audit.json`.
No additional training or experimental measurements after final test.

### 2026-11-18 — Activation criterion and individual rank budgets

Extended the existing JAX stages with streaming train/validation moments, stable
regularized weighted SVD, reusable factor banks and exact grid-budget allocation.
Calibration: 300 train images, seed 42; ridge = 1e-4 * trace(C)/in_features.
Completed 168 paired validations + 6 allocated models on the same 1500 validation
images, and 2 unchanged five-epoch fine-tunes (70000 train images, 8750 updates,
none skipped). Test never evaluated; original SVD measurements were reused.

Results: `all_blocks` rank .5 gives 96.067% vs SVD 65.000% before training,
96.667% vs 95.800% after; `ffn` .5 gives 97.133% vs 96.867% after training.
All 336 nonfull matrix/rank pairs improve calibration and independent validation
local errors. Minimizing their unnormalized sum worsens network accuracy in all
6 equal-budget comparisons: a useful negative result, not a global optimum.

Proof and limitations: 21 full-rank logit controls pass (maximum relative error
3.89e-7, absolute 6.44e-6); all 48 ordinary bank factors match the existing rank-192
checkpoint bit for bit. Full report/tables/plots and input identities are linked
above. `scripts/check.sh`: 24 tests pass. Artifact audit: 717 complete stages,
4256 matching identities across 1576 unique files; old weights/split/results intact.
Commands/schemas: `docs/lowrank/ACTIVATION_STAGES.md`. Test stays held out;
one checkpoint/seed/validation and fixed ridge limit generalization.
Public metadata omits per-example indices/pixel digests; complete identities
and internal manifests remain in local stages for reproducibility and audit.
Remote publication remains pending: automatic GitHub approval rejects the full
WORK_LOG payload because of internal artifact references, including unchanged
references already present upstream. The complete change is committed locally.

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
