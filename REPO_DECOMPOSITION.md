# Repository Decomposition — `aerial_vit_svd_codebase`

Generated from source inspection on 2026-09-28. Scope: code, scripts, configs, docs.
Result artifacts (`results/`, `matrix_results/`, `plots/`, `runs/`, `reports/`) were listed by
filename only; their contents were not read.

---

## 1. What the project is

A single-purpose research codebase (~1.7k lines of Python, 17 modules) for a thesis
subproblem on **low-rank compression of visual-transformer weight matrices**.

It contains **two overlapping research framings** sharing one codebase:

| Framing | Where written | Object of study | Added |
|---|---|---|---|
| **A — engineering** | `README.md` (English) | Accuracy/params/latency/memory trade-off of SVD-compressed ViTs on aerial imagery | initial commit `310ae9f` |
| **B — computational linear algebra** | `README_2.md`, `docs/*_matrix_math.md` (Russian) | The weight matrices *themselves*: singular spectra, numerical/stable rank, conditioning, approximation error, randomized SVD | ~2026-05-18 |

Framing B is a deliberate **reframing** of framing A, not an addition. `docs/report_notes_matrix_math.md`
states explicitly that accuracy/latency/memory metrics "remain, but stop being central" and are
demoted to secondary evidence. Under B, ViT-Small / ViT-Base / ResNet-18 are demoted from
"models under evaluation" to **generators of realistic matrices**, and the AID dataset to a
source of trained weights plus real activations.

**Domain context:** AID scene classification (23–30 classes, `data/AID/<ClassName>/*.jpg`,
torchvision `ImageFolder` layout) as a proxy for the visual encoder of a future onboard-UAV
RL policy. The long-term question is how large a visual neural component can be under
onboard-like memory/latency constraints.

---

## 2. Layered decomposition

```
L5  orchestration     scripts/*.sh (13 thin bash wrappers, 2 real sweeps)
L4  experiments       train / compress / evaluate / finetune_compressed / inspect / summarize
                      analyze_spectra / approximation_curves / randomized_svd_benchmark / layer_error_analysis
L3  math library      matrix_analysis.py   ← shared by all L4 math experiments
L2  compression core  svd.py               ← LowRankLinear + targeting + in-place rewrite
L1  foundations       data.py · models.py · engine.py · utils.py
L0  environment       Dockerfile · .devcontainer/ · requirements.txt (PYTHONPATH=src, no packaging)
```

Dependency direction is strictly downward; there are no cycles. `svd.py` and
`matrix_analysis.py` are the two conceptual centers and are **independent of each other** —
the math block re-derives SVD analysis rather than reusing `svd.py`.

### L0 — Environment

- `Dockerfile`: `pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime` + `requirements.txt` + `kaggle`.
- `.devcontainer/devcontainer.json`: VS Code container workflow; GPU passthrough via
  `--ipc=host --gpus=all` (documented in `README_2.md`).
- `requirements.txt`: torch ≥2.1, torchvision, timm ≥0.9, numpy, pandas, matplotlib, PyYAML, tqdm.
- **No `pyproject.toml` / `setup.py`.** The `src/` layout is activated only by
  `export PYTHONPATH="$PWD/src"`, repeated in every script. Nothing is installable.

### L1 — Foundations

| File | Responsibility |
|---|---|
| `utils.py` | `seed_everything`, `load_yaml`, `save_json`, `get_device`, param counting (`count_total_params`, `count_trainable_params`, `sizeof_params_bytes`), `human_bytes`. |
| `data.py` | `ImageFolder` + `build_transforms` (train: `RandomResizedCrop(0.70–1.0)` + hflip; eval: resize; ImageNet mean/std). Deterministic 3-way split via seeded `torch.randperm`, **persisted to `split.json`** so all models and all experiments see identical splits. |
| `models.py` | `timm.create_model` factory; `save_checkpoint` (payload: `model`/`epoch`/`metrics`/`config` + optional optimizer/scheduler); checkpoint loading sniffs `model` / `state_dict` keys and loads with `strict=False`. |
| `engine.py` | `train_one_epoch` (AMP `autocast` + `GradScaler` + gradient accumulation + `max_batches` smoke-test knob), `evaluate_accuracy`, `benchmark_inference` (warm-up batches, `cuda.synchronize`-bracketed latency, `max_memory_allocated`, throughput). |

The shared `split_file: ./runs/aid_split_seed42.json` referenced by all three configs is the
load-bearing mechanism that makes cross-model comparisons valid.

### L2 — Compression core (`svd.py`, 127 lines)

The conceptual heart of framing A.

- **`LowRankLinear`** — replaces `nn.Linear(in, out)` with two stacked linears of rank `r`,
  splitting `√Σ` symmetrically across both factors:
  `first: (r × in) = √Σ_r · Vh_r`, `second: (out × r) = U_r · √Σ_r`, bias carried on `second`.
  `forward(x) = second(first(x))` reproduces `x · W_rᵀ` exactly.
- **`find_linear_targets(mode)`** — name-substring targeting: `attn.qkv`/`attn.proj` → `attn`,
  `mlp.fc1`/`mlp.fc2` → `ffn`, `all` adds the classifier head.
- **`rank_from_ratio`** — `r = floor(ratio · min(in, out))`.
- **`compress_model_inplace`** — walks targets, computes the SVD in float32 on CPU, swaps modules
  in place, **skips any layer where the low-rank form is not actually smaller**, and returns
  per-layer `CompressionRecord`s.

**Structural consequence:** compression changes the *architecture*, not just the weights. Any
consumer of a compressed checkpoint must rebuild the low-rank module tree **before** loading the
`state_dict` — which is why `--svd-mode` / `--rank-ratio` reappear as flags in both `evaluate.py`
and `finetune_compressed.py`.

### L3 — Matrix-math library (`matrix_analysis.py`, 311 lines — largest module)

Shared substrate for the four math experiments. Provides:

- `load_model_from_config` — resolves checkpoint from config's `output_dir/best.pt`, then **merges
  the config embedded inside the checkpoint** (excluding `data_dir`) so `num_classes` and model
  name are recovered automatically. The most robust loader in the repo.
- `matrix_kind` — semantic taxonomy (`attn_qkv`, `attn_proj`, `ffn_expand`, `ffn_contract`,
  `classifier`, `conv`, …).
- `module_to_matrix` — `Conv2d` weights unfolded to 2-D via `w.reshape(out_channels, -1)`, so
  ResNet conv layers enter the same analysis as ViT linears.
- `iter_weight_matrices` / `filter_matrices` — enumeration with `--include-regex` /
  `--exclude-regex` over `"name kind"`.
- `spectral_stats` — σ_max, σ_min, ε-clipped σ_min, `cond_eps`, Frobenius/spectral norms,
  **stable rank** `‖W‖_F²/‖W‖₂²`, numerical rank at tolerance, and energy ranks `r90/r95/r99`
  (via `searchsorted` on the cumulative squared-spectrum).
- `approximation_records` — error-vs-storage curves computed **analytically from the spectrum**
  (`E_F(r) = √(Σ_{i>r}σ_i² / Σσ_i²)`, `E_2(r) = σ_{r+1}/σ_1`) — no matrix reconstruction, so many
  ranks cost almost nothing.
- `randomized_svd` — from-scratch randomized range finder: Gaussian projection → power iterations
  → reduced QR → small SVD → `U = Q·Û`.
- `benchmark_randomized_svd` — paired timing + accuracy: full vs randomized wall time, relative
  error of singular values, and both approximations' Frobenius error.

### L4 — Experiment drivers

Each is a standalone `python -m aerial_svd.<name>` CLI with `argparse` + YAML config, writing JSON
or CSV plus matplotlib figures.

**Framing A (model-level, GPU):**

| Module | Question it answers | Primary output |
|---|---|---|
| `inspect.py` | Which parameters exist and which are compressible? Groups every tensor semantically and prints a low-rank candidate table (no training needed). | `results/inspect_*.json` |
| `train.py` | Baseline after fine-tuning? AdamW + cosine + label smoothing, best/last checkpoints, per-epoch history. | `runs/*/best.pt`, `history.json`, `final_metrics.json` |
| `compress.py` | Post-hoc SVD at `(mode, rank_ratio)` — params before/after, per-layer records. | `checkpoints/*.pt` + `*.compression.json` |
| `evaluate.py` | Accuracy **and** latency/throughput/peak-GPU-memory/checkpoint bytes for baseline or compressed. | `results/*.json` |
| `finetune_compressed.py` | Can accuracy lost to SVD be recovered? Recovery fine-tune at `lr 1e-5`. | `runs/*_recovery/` |
| `summarize.py` | Aggregate `results/*.json` → `summary.csv` + accuracy-vs-params, latency-vs-params, accuracy-vs-rank-by-mode plots. | `results/summary.csv`, `plots/*.png` |

**Framing B (matrix-level, mostly CPU):**

| Module | Experiment |
|---|---|
| `analyze_spectra.py` | "Spectral passport": per-layer stats table, σ_k/σ_1 decay plot, `r95_ratio` boxplot by kind. |
| `approximation_curves.py` | `E_F` vs storage cost `r(m+n)/mn`, per-kind and median-tradeoff plots. |
| `randomized_svd_benchmark.py` | Full vs randomized SVD timing/error on the largest matrices. |
| `layer_error_analysis.py` | **Most methodologically interesting.** Registers `register_forward_hook`s on real `nn.Linear` modules, and for each rank ratio recomputes `ŷ = x·V_rᵀ·diag(S_r)·U_rᵀ + b` against the true output on **real dataset activations**, reporting mean / p95 / max relative error. Deliberately does *not* mutate the model — the approximation is computed alongside the true forward pass. Tests the thesis that small `‖W−W_r‖_F` does **not** imply small layer-output error, because what matters is the orientation of activations relative to the discarded singular directions. |

### L5 — Orchestration

13 scripts in `scripts/`, each a 3–15-line `set -euo pipefail` + `PYTHONPATH` + `python -m` wrapper.
Only two carry real experiment logic:

- `compress_sweep_vit_small.sh` — the 3×3 grid: `mode ∈ {ffn, attn, all}` × `rank_ratio ∈ {0.75, 0.50, 0.25}` = 9 compress+evaluate pairs off `runs/aid_vit_small/best.pt`.
- `run_matrix_math_pack.sh` — chains the four math experiments.

The intended end-to-end order is documented in `README.md` steps 0–6.

### Artifacts layout (outputs, contents not inspected)

`runs/` (4 trained runs) · `checkpoints/` (compressed `.pt` + `.compression.json`) ·
`results/` (12 sweep JSONs + 2 inspect JSONs + `summary.csv`) ·
`matrix_results/{spectra,approximation,randomized_svd,layer_error}/` (all three models, except
`layer_error` and `randomized_svd` which cover ViT-Small/ViT-Base only) · `plots/` · `logs/` ·
`reports/2026-05-11/` (a dated snapshot of the ViT-Small results).

---

## 3. Data flow

```
data/AID/ ──split.json──► build_loaders ──► train.py ──► runs/<name>/best.pt (config embedded)
                                                        │
                          ┌─────────────────────────────┴──────────────────────────┐
                          ▼                                                        ▼
              compress.py ──► checkpoints/*.pt                    matrix_analysis.load_model_from_config
                    │  (LowRankLinear rewrite)                                    │
                    ▼                                          ┌──────────┼─────────────┐
              evaluate.py ──► results/*.json ──► summarize.py  ▼          ▼             ▼
                    ▲                              │      analyze_   approximation_  layer_error_
                    └── finetune_compressed.py     │      spectra    curves          analysis
                                                   ▼                       (hooks on real x)
                                              plots/*.png
```

---

## 4. Cross-cutting characteristics

**Deliberate design choices**
- Everything is a separate CLI process; state passes through files (YAML in, JSON/CSV out). No
  in-process experiment registry, no job runner.
- Reproducibility via a single `seed: 42` plus a persisted split file.
- `max_batches` / `max_targets` / `max_matrices` caps on every driver — the code is built for
  iterative smoke-testing on a 12 GB GPU.
- `strict=False` checkpoint loading everywhere, tolerating the key mismatches that module-swapping
  and `timm` head reconstruction produce.

**Duplication**
- The `build_loaders(...)` call with eight identical `cfg.get()` defaults is re-implemented in
  `train.py`, `evaluate.py`, `finetune_compressed.py`, `layer_error_analysis.py`.
- Checkpoint `state_dict` sniffing is re-implemented in `models.py`, `evaluate.py`,
  `finetune_compressed.py`, and (differently, more robustly) `matrix_analysis.load_model_from_config`.
- The layer taxonomy exists twice: `svd.find_linear_targets` (substring matching) and
  `matrix_analysis.matrix_kind` (regex/`isinstance`), which can drift apart.

**Fragility**
- Both targeting mechanisms hardcode `timm`'s internal module names (`mlp.fc1`, `attn.qkv`).
  A `timm` version bump or a non-timm backbone silently yields zero compression targets.
- `summarize.py` infers schema from whichever keys its inputs happen to contain; baseline result
  files lack `rank_ratio`/`svd_mode`, so they survive as NaN rows that plotting later drops.
- `compress.py` reads `num_classes` from the YAML (`cfg.get("num_classes", 30)`, defaulting to the
  AID class count) — it only gets the right value because `train.py` wrote it back into the config.
- `layer_error_analysis.py` hardcodes `torch.autocast(device_type="cuda", ...)` regardless of the
  resolved device; its wrapper passes `--device cuda`, so this is latent rather than active.
- `torch.cuda.amp.GradScaler` (in `train.py`, `finetune_compressed.py`) is the deprecated
  pre-2.4 alias for `torch.amp.GradScaler("cuda")`.

**Version control gaps** — `.gitignore` currently excludes `docs/`, `matrix_results/`,
`requirements.txt`, `Dockerfile`, `.dockerignore`, `*.zip`, `*.tar.gz`, `.qwen/`, `.agents/`.
Confirmed against `git ls-files`: **`requirements.txt`, `Dockerfile`, and all of `docs/` are
untracked**, so the entire framing-B write-up (report notes, presentation outline, advisor update)
and the dependency manifest exist only on this machine. `requirements.txt` and `Dockerfile` are
also listed in `.gitignore` despite being the reproducibility contract.

**Absent entirely**
- No tests, no CI, no linter/formatter/type-check config.
- No packaging or lockfile (unpinned `>=` ranges).
- No `Makefile`/`justfile`; the shell scripts are the only task interface.
- No adaptive rank selection — rank is always a fixed ratio, though `docs/` lists
  ε-targeted rank selection, cross/skeleton approximation, Tucker/TT tensorization, and Sobol
  sensitivity analysis as the planned next steps.

---

## 5. One-paragraph summary

A thesis codebase in the middle of a **pivot from an applied compression benchmark to a
numerical-linear-algebra study**. The engineering half (`data → train → compress → evaluate →
summarize`, driven by `svd.py`'s `LowRankLinear` rewrite of timm ViT `Linear` layers) is complete,
swept, and summarized. The math half (built on the separate `matrix_analysis.py` library) reframes
the same trained models as collections of matrices and measures their spectra, energy ranks, stable
ranks, conditioning, approximation-error curves, randomized-SVD viability, and — most importantly —
the gap between matrix-level Frobenius error and layer-output error measured on real activations via
forward hooks. The two halves share only the config/checkpoint conventions; the code is
file-mediated, test-free, tightly coupled to timm module naming, and its newest documentation and
build files are not under version control.
