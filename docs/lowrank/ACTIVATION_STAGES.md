# Activation experiments: agent commands

Scientific description and measured tables: [Russian report](results/aid_activation_20261117/REPORT_RU.md).
Use the existing images/launcher, seed 42, frozen split and float32 inference.
Never invoke legacy `evaluate`/`layer_error_analysis` (test input). No test stage here.

All commands take a **fresh** output directory. Existing complete outputs are
immutable; a retry uses `.retry-NNN`. Verify manifest `complete`, input identities
and output hashes with `lowrank_quality.verified_stage`, not file existence.

Below, `R=artifacts/experiments/aid_activation_20261117`,
`S=artifacts/experiments/aid_svd_20261117`,
`W=artifacts/checkpoints/jax/vit_small_protocol_v1.npz`,
`C=configs/aid_vit_small.yaml` (ordinary shell variables).

```bash
R=artifacts/experiments/aid_activation_20261117
S=artifacts/experiments/aid_svd_20261117
W=artifacts/checkpoints/jax/vit_small_protocol_v1.npz
C=configs/aid_vit_small.yaml

# 03: independent train and validation moment collection; no activation retention.
scripts/run.sh gpu python -m aerial_svd.lowrank_activation_stats \
  --config "$C" --checkpoint "$W" --output-dir "$R/03_train" --device gpu \
  --split train --samples-per-class 10 --batch-size 8
scripts/run.sh gpu python -m aerial_svd.lowrank_activation_stats \
  --config "$C" --checkpoint "$W" --output-dir "$R/03_validation" --device gpu \
  --split val --batch-size 8

# 04: matrix-only factor bank and errors. Validation moments only measure errors;
# they do not enter the factorization or allocation objective.
scripts/run.sh cpu python -m aerial_svd.lowrank_activation_curves \
  --config "$C" --checkpoint "$W" --output-dir "$R/04_bank" --scope all_blocks \
  --rank-ratios .125 .25 .375 .5 .625 .75 .875 1.0 \
  --calibration-dir "$R/03_train" --validation-stats-dir "$R/03_validation" \
  --ridge-relative .0001

# 04 and 05 remain independently callable for one configuration.
scripts/run.sh cpu python -m aerial_svd.lowrank_make_candidate \
  --config "$C" --checkpoint "$W" --output-dir "$R/example/04_candidate" \
  --scope ffn --rank-ratios .5 --activation-bank "$R/04_bank"
scripts/run.sh gpu python -m aerial_svd.lowrank_validate \
  --config "$C" --checkpoint "$R/example/04_candidate/candidate.npz" \
  --output-dir "$R/example/05_validation" --device gpu \
  --reference-dir "$S/05_dense" --batch-size 8

# Optional preset enumerator, called separately for each requested series.
# global=24; regions=48 (blocks 0–3/4–7/8–11); single=96 (blocks 0/5/11).
python3 scripts/lowrank_series.py --root "$R" --series global \
  --activation-bank "$R/04_bank" --reference-dir "$S/05_dense"
# Replace global with regions, single, or allocation for those independent runs.
# allocation: exact budgets of uniform all_blocks rho=.25/.5/.75; both methods.

# Exact budget allocation, then ordinary candidate/validation commands above.
scripts/run.sh cpu python -m aerial_svd.lowrank_allocate \
  --config "$C" --checkpoint "$W" --output-dir "$R/example/04_allocation" \
  --scope all_blocks --rank-ratios .5 --activation-bank "$R/04_bank" \
  --method activation_svd
# Pass --allocation-dir to lowrank_make_candidate with the same scope/reference
# ratio and bank. Method/ranks come from allocation.json; no manual rank edits.

# 06: unchanged five-epoch protocol and epoch recovery via --resume.
python3 scripts/lowrank_series.py --root "$R" --series finetune \
  --reference-dir "$S/05_dense" \
  --finetune-configurations global_ffn_r0500 global_all_blocks_r0500

# 08: existing measurements only; audits paired pixels/logits/budgets.
scripts/run.sh cpu python -m aerial_svd.lowrank_aggregate \
  --root "$R" --baseline-root "$S" --activation-bank "$R/04_bank" \
  --output-dir "$R/08_comparison"
scripts/check.sh
```

| Output | Meaning |
|---|---|
| `03_*/moments.npz` | name → float64 uncentered `(in,in)` second moment |
| `03_*/statistics.json` | split, exact indices, image/token counts, pixel hash, transform |
| `04_bank/factors.npz` | `name.first/second`: full weighted factors, float32; `.svd.first/second`: ordinary factors; `.weighted_singular_values`: float64 spectrum |
| `04_bank/diagnostics.json` | relative/absolute ridge, eigenvalue bounds, PSD rounding, condition |
| `04_bank/local_errors.csv` | every matrix/method/rank: measured Frobenius, train/val MSE per token, relative RMSE, regularized objective, factor weights |
| `04_allocation/allocation.json` | explicit ranks, uniform reference, exact weight budget, local objective, deterministic DP |
| `04_candidate/candidate.npz` | existing model NPZ schema, genuine factor shapes and unchanged bias |
| `05_validation/` | existing validation schema, paired logits/indices/pixel hash |
| `06_finetune/` | existing epoch checkpoints + optimizer state, best NPZ, history and before/after |
| `08_comparison/` | configurations, paired comparisons, per-matrix errors, budget and fine-tuning CSVs, plots, summary/report |

Stage 08 exports only aggregate statistics metadata. Exact calibration/validation
indices and pixel digests stay in local stage 03, where the pairing audit reads
them. The public Git snapshot also omits input checkpoint/split identities and
internal per-stage manifest paths; its manifest verifies public output bytes only.
Complete input/output provenance remains in the local experiment stages.

Batch Gram products use float32 `highest` precision; host sums, Cholesky,
unwhitening and weighted SVD use float64. Factors are split with symmetric
`sqrt(S)` **in weighted coordinates** and serialized float32. The loader/inference
never reconstruct a dense weight. Full rank must pass matrix relative error
`5e-5`, logit relative error `5e-5`, maximum absolute logit difference `1e-3`.
For tiny negative Gram eigenvalues (at most `2e-6*lambda_max`) only numerical
PSD projection is allowed and logged; materially indefinite moments fail.
For exactly zero moments ridge uses scale 1; this reduces to ordinary SVD.
The measured experiment records the actual ridge per matrix and whether clipping
occurred. Allocation minimizes **unregularized** empirical MSE of the computed
factors, over the supplied grid, at **exact** storage equality. It does not optimize
validation accuracy or claim global network optimality.
For allocated models, `configurations.csv:rank_ratio` denotes the uniform budget
reference; actual per-matrix ranks/ratios are in `matrix_errors.csv`.
