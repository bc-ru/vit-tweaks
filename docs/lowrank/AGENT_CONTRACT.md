# ViT low-rank experiments — agent implementation contract

Status: stages **00–09 complete**, including 134 GPU measurements and 20 frozen final-test models. [Unified final report](results/aid_final_20261118/REPORT_RU.md), [final-stage commands](FINAL_STAGES.md). Earlier measurements: [ordinary SVD](results/aid_svd_20261117/REPORT_RU.md) and [activation-aware/rank-allocation results](results/aid_activation_20261117/REPORT_RU.md). No further experiments after final test. Activation commands/formats: `ACTIVATION_STAGES.md`; human-readable protocol: `EXPERIMENT_DESIGN_RU.md`.

## 0. Repository and boundaries

- Target repository: `bc-ru/vit-tweaks`, JAX backend, branch `research/lowrank-experiment-protocol-v1` (based on `jax-migration`; verify before work).
- Read `AGENTS.md` and `WORK_LOG.md` before changes. Ordinary work is single-agent; no separate task/agent workflow is required.
- Treat the existing AID dataset, `runs/aid_split_seed42.json`, checkpoint, class order, and preprocessing as authoritative. **Verify actual files and paths**; references in this document are not proof that local assets exist.
- If the original AID split contains only train/test, create and persist a **single fixed (ideally stratified) validation subdivision from train only**. Preserve the original test set without modification and record subdivision indices/hash.
- Main model: pretrained/finetuned ViT-Small `vit_small_patch16_224` on AID. ViT-Base is available but **not part of the first matrix × rank sweep**.
- Work on one shared RTX 4070 (12 GB) GPU. Avoid competing GPU jobs; do not launch expensive experiments until the run window is confirmed.
- Preserve all prior artifacts, including bug-affected checkpoints. No silent deletion, overwrite, branch change, commit, push, or scope expansion. Obtain separate authorization for Git operations per repository policy.
- **Do not implement a monolithic `run_everything` entry point.** Build small independently invoked stages with explicit file-based interfaces. Existing JAX interfaces may be extended incrementally.

## 1. Scientific invariants

1. Factorize **every selected matrix at every valid rank**, even if the factorization stores more numbers: no `r * (m+n) < m*n` admission filter. Record negative savings normally.
2. Control `r = min(m,n)` is mandatory. In exact arithmetic, truncated full-rank SVD reconstructs the original matrix; test actual numerical tolerance for each dtype.
3. For `W in R^{n×m}`, store two factors with shapes `n×r` and `r×m` in mathematical row-input notation; account for transposed `weight` storage in JAX implementation. Preserve bias unchanged.
4. Standard baseline = **exact SVD**, not rSVD. The latter is **not a study variable**. If exact SVD proves impractically slow, report measured timing and ask for approval to switch; then record algorithm and its configuration. Do not mix algorithms in an unlabeled curve.
5. Train-only data may be used for activation calibration and training. **Validation** selects ranks, scopes, approximation methods and fine-tuning checkpoints. **Test** is final evaluation only. Never tune on test.
6. Compare models on exactly the same images and preprocessing. Never conflate matrix approximation error, local activation-weighted error, and end-to-end model output error.
7. Preserve scientific separation of matrix-only analysis (matrix algebra, factor counts) and full-model quality/performance (model materialization, dataset, GPU evaluation).
8. Return actual results with provenance, not assumptions. Missing stages must be reported as pending.

## 2. Current code points to inspect and fix

| Code | Known issue / required change |
|---|---|
| `src/aerial_svd/jax_compression.py::compress` | All selected valid ranks factorize, including expanding/full ranks; optional validated factors reuse this structural path. |
| `src/aerial_svd/jax_cli.py::verify_compression` | Corrected no-skip policy; historical saving-only behavior remains opt-in. |
| `src/aerial_svd/model_spec.py::is_target` | `all_blocks` excludes `head`; legacy `all` includes it. |
| `src/aerial_svd/lowrank_activation_stats.py` | Fixed train-only calibration or separate validation moments; legacy `layer_errors` still uses test. |
| `src/aerial_svd/lowrank_validate.py` | Paired validation; legacy `evaluate` keeps its final-test semantics. |
| `src/aerial_svd/jax_engine.py` benchmarking | Synchronized warm timing exists; distinguish compile/warm-up from measured inference; do not pass lifetime process peak off as inference-specific GPU memory. |

Re-inspect code before editing; the findings above describe the inspected revision, not a guarantee about future commits.

## 3. Stage contract — independent entry points

Use a run directory unique per configuration, e.g. `artifacts/experiments/<experiment_id>/`. Each stage writes its own immutable artifacts atomically (temporary file then rename), a `manifest.json` with input identities/hashes and a `status=complete` only on success. Consume paths, **not in-memory state**.

| Stage | Inputs | Outputs | Exit/acceptance criteria |
|---|---|---|---|
| `00_baseline` | source checkpoint + config + split | `baseline.json` | identity, split, checkpoint, baseline validation quality verified |
| `01_factor_checks` | weights + explicit matrix/rank list | `factor_checks.json` | all requested factors present; factor reconstruction, full-rank and save/load checks pass |
| `02_matrix_curves` | dense weights + rank grid | `matrix_curves.csv` | per-matrix relative Frobenius/spectral errors and parameter counts; no dataset inference |
| `03_activation_stats` | dense model + fixed train calibration sample | `activation_stats/` + manifest | uncentered second moments with shape and source details; no test input |
| `04_make_candidate` | dense checkpoint + scope/ranks/method (+ optional stats) | `candidate.npz`, `candidate.json` | factorized checkpoint reloads; expected exact list of replaced matrices |
| `05_validate` | candidate + validation split + baseline logits | `validation.json` (+ optional logits) | accuracy, cross-entropy and paired logit drift computed on validation |
| `06_finetune` | selected candidate + train/validation | `best.npz`, `history.json` | fixed ranks, documented trainable parameters; best chosen on validation |
| `07_benchmark` | fixed checkpoint + device/dtype/batch protocol | `benchmark.json` | synchronized warmed trials and parameter footprint; GPU memory caveats recorded |
| `08_aggregate` | existing JSON/CSV only | plots/tables + manifest | no JAX model load, no new inference; never combine incompatible protocols |
| `09_final_test` | preselected checkpoint + untouched test split | `test.json` | one final unbiased report for chosen candidates |

Do **not** create tight runtime coupling or mandatory global orchestration. Allow manual invocation and rerun of one failed stage. A short optional script may enumerate preset configurations; it must not contain experimental logic.

Suggested single-candidate configuration shape (illustrative **schema**, not an existing executable config):

```yaml
experiment_id: aid_small_ffn_r050_svd
model: vit_small_patch16_224
source_checkpoint: CHECK_AND_FILL_PATH
source_split_manifest: runs/aid_split_seed42.json
validation_subsplit_manifest: CHECK_OR_CREATE_FROM_TRAIN
scope: ffn
rank_rule: {kind: relative, fraction: 0.5, rounding: floor}
approximation: {criterion: weight_frobenius, algorithm: exact_svd}
dtype: float32
output_dir: experiments/lowrank_2026/aid_small_ffn_r050_svd
```

Every stage gets an explicit config path and its own output path, validates prerequisite artifacts, and can be called separately. An agent implementing a CLI must document the **actual** commands and schemas it introduced; never present the proposed names as commands that already work.

## 4. Required experiments

### A. Baseline rank × target scope

- Scopes: `ffn`, `attn`, `all_blocks` (excluding classifier `head`).
- Pilot rank fractions `rho = [0.25, 0.5, 0.75, 1.0]` (12 candidates + dense baseline); extended `[0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0]` (24 candidates).
- Actual per-weight integer rank: `r = max(1, floor(rho * min(n, m)))`; save per-weight ranks rather than only `rho`.
- Matrix-only: `||W-Wr||_F/||W||_F`, `||W-Wr||_2/||W||_2`, `nm`, `r(n+m)`.
- Model validation: top-1 accuracy, cross-entropy, paired logit `L2` drift (mean and P95). Index plots both by rank fraction and **actual model parameter count**.
- Optional later: single-matrix substitutions by early/middle/late transformer block, not in pilot.

### B. Activation-aware local criterion

With row-stacked **train-only** projection inputs `Z in R^{K×n}` and `W in R^{n×m}`:

```
C = Z.T @ Z / K                         # uncentered second moment
J(Wr) = ||Z @ (W - Wr)||_F^2 / K
      = ||C^(1/2) @ (W - Wr)||_F^2
B = C^(1/2) @ W
Wr = C^(-1/2) @ truncate_svd(B, r)       # if C positive definite
```

For rank-deficient/ill-conditioned `C`, explicitly specify `C_lambda = C + lambda*I` and log `lambda`; regularization **changes** the criterion. Do not call this a global optimum for the nonlinear network. Evaluate exact-SVD and activation-aware approximants **at the same rank**, first on selected early/middle/late projections for `rho = 0.25, 0.5`; compare local errors on calibration and separate validation activations, then model quality with single projection replacement. No claim of novel method.

### C. Fine-tuning recovery

- Pick 1–2 candidates **via validation** from A (e.g. `ffn` moderate and `all_blocks` stronger), run 3–5 epochs as an initial protocol.
- Train factor parameters directly; preserve rank and factorized inference. Record frozen/unfrozen params, optimizer, learning rate schedule, epochs, batch size, seed, checkpoint selection criterion.
- Compare before vs after on **same validation split**; optionally add dense-model continued-training control with equivalent budget.
- No guaranteed accuracy recovery; report measurements.

### D. Resource measurements

- Exact parameter count and bytes; serialized checkpoint size separately.
- RTX 4070 inference batch 1 and 16, same dtype/preprocessing, JAX JIT compiled + warmed, `block_until_ready()`, repeated trials. Median/dispersion and (if supported) throughput; exclude compile time from steady-state latency.
- Process-lifetime GPU peak **is not inference-only memory**. Use isolated process/methodology or mark memory as not measured. A reduced parameter count does **not** imply reduced inference latency.

## 5. Provenance and acceptance

Every result must record at least: git commit, model spec, dense checkpoint path/hash, AID split identity/hash, calibration subset identity, requested scopes/ranks, **actual per-matrix ranks and replaced names**, SVD algorithm, dtype, method hyperparameters, preprocessing, device, batch size, and evaluation split. Mark old flawed checkpoints `legacy_skip_non_saving` and never silently compare them as corrected results.

Mandatory code tests before the 12-candidate pilot:

1. Non-saving ranks are factorized, not skipped; verify changed matrix names.
2. Full-rank reconstruction within a dtype-justified numerical tolerance.
3. Factorized serialization/reload and unchanged biases.
4. Matrix error and factor parameter-count formulas match directly computed values.
5. Validation/train-calibration use correct splits, and test is never sampled by tuning stages.
6. One candidate's output is deterministic to the configured tolerance on repeated evaluation.

## 6. Deliver in small increments

1. First return a short repository status report; implement/verify `00_baseline` and **a minimal patch for `01_factor_checks` + no-skip bug**. Stop after tests; do not automatically launch the sweep.
2. Next implement/test matrix-only `02`, independent of model inference.
3. Then build `04` and `05` and run a *small* smoke test before the 12 configurations.
4. Only then build/execute activation statistics `03`, weighted criterion, fine-tuning `06`, benchmarking `07`, aggregating `08`, and held-out final test `09`, each with explicit handoff.

If choices are unresolved (exact source checkpoint, validation split, calibration sample size, regularization value, GPU availability), report the choices and a conservative proposal before expensive work. Document each stage's single CLI invocation, inputs, output schema and how to verify it; **never hide multiple stages under one command**.
