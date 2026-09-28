# AGENTS.md

Entry point for agents working in this repository. Read this first; follow links for detail rather
than guessing from habits.

## What this repository is

Fine-tune ViT-Small / ViT-Base (timm) on the AID aerial scene-classification set, compress selected
`nn.Linear` layers by truncated SVD, and measure what the compression costs. PyTorch 2.1 / CUDA 11.8
/ timm, 10 classes after config filtering, 224×224 inputs.

Two research framings coexist and **must not be merged**:

- **Framing A — compression benchmark** (`README.md`, `README_2.md`, steps 0–6): accuracy, parameter
  count, checkpoint size, GPU memory, latency, throughput across `{ffn, attn, all} × {0.75, 0.50, 0.25}`.
- **Framing B — computational linear algebra of the weight matrices** (`docs/*_matrix_math.md`):
  singular spectra, numerical/stable rank, conditioning, low-rank approximation error, randomized
  SVD, layer-output error. Here accuracy/latency/memory are explicitly **secondary** evidence.

Framing B is the current direction. `src/aerial_svd/matrix_analysis.py` is its center and is
deliberately independent of the compression path in `svd.py` / `compress_sweep.py`.

`REPO_DECOMPOSITION.md` holds a full layer-by-layer map of the codebase.

## Authority order

For any claim about this project, resolve in this order. When two levels disagree, say so explicitly
— do not silently pick one.

1. **Source code + `configs/*.yaml`** — authoritative for *what the code does*.
2. **Generated artifacts** (`runs/*/final_metrics.json`, `results/`, `matrix_results/`) —
   authoritative for *what was actually measured*. Read them; never assume a number.
3. **`docs/*_matrix_math.md`** — authoritative for *why* the work is framed as it is, and for the
   intended scientific reading of the results.
4. **`README.md` / `README_2.md`** — usage and pipeline entry points; may lag the code.
5. **`.research-vault/`** — shared cross-repo vault (`03-projects/RL-UAV Current Research State.md`)
   for thesis-level claims, scope, and research decisions. **Not present in this working copy.**
   When a task needs it, check for it and report it missing; never reconstruct its contents from
   inference.
6. **Conversation / agent memory** — weakest; never the sole basis for a change.

Code wins for "what was done"; docs win for "why". A conflict between them is a finding to report,
not something to smooth over.

## Execution boundary

**The host has no PyTorch.** Host `python3` (3.10) lacks `torch` and `timm`; there is no `.venv`.
Everything that imports this project runs in the container.

```bash
# image: ghcr.io/bc-ru/pytorch-2.1.0-cuda11.8-cudnn8-runtime-timm-models:stable
docker run --ipc=host --gpus=all -v "$PWD:/workspace" -w /workspace <image> bash -c \
  'PYTHONPATH="$PWD/src" python3 -m aerial_svd.<module> ...'
```

- `PYTHONPATH="$PWD/src"` is mandatory — the package is not installed and there is no
  `pyproject.toml`/`setup.py`.
- One RTX 4070 (12 GB), shared. Check it is free before a GPU run; if busy, stop and report rather
  than killing anything.
- Prefer the built-in caps (`--max-batches`, `--max-targets`, `--max-matrices`, `max_batches`) for a
  first pass over a full sweep.
- Static checks that need no dependencies: `bash -n scripts/*.sh`, `python3 -m py_compile src/aerial_svd/*.py`.

## Invariants — do not break without an explicit decision

- **`seed: 42` and the shared split file `runs/aid_split_seed42.json`.** Every config points at the
  same split so ViT-S, ViT-B and ResNet-18 are compared on identical images. Never regenerate or
  replace it; never change `seed`. Doing so silently invalidates every existing comparison.
- **Never overwrite existing artifacts.** `results/`, `matrix_results/`, `runs/`, `checkpoints/`,
  `plots/`, `reports/` contain the evidence base. Write to a new path.
- **`LowRankLinear` splits `√Σ` symmetrically across `factor_in`/`factor_out`.** Changing the split
  changes numerics and makes old checkpoints incomparable.
- **Compression is structural.** `load_state_dict` must run against a model whose low-rank tree was
  already rebuilt (`train_finetune.load_compressed_for_finetune`); a plain `strict=False` load into
  a full-rank model silently drops the compressed factors.
- **Module-name coupling to timm.** Low-rank targets are matched by substring (`mlp.fc1`, `attn.qkv`)
  in two places: `svd.find_linear_targets` and `matrix_analysis.matrix_kind`. They must stay in sync,
  and both break silently if timm renames modules.
- **`layer_error_analysis.py` must not mutate the model** — it registers and removes forward hooks.
- **`matrix_analysis.approximation_records` is analytic**, computed from the already-computed
  spectrum, not from reconstructed matrices. Don't "simplify" it into materialized reconstructions.
- **`strict=False` is used deliberately** when loading pretrained baselines. Don't tighten it.
- **Two framings, two metric roles.** Don't re-center accuracy/latency in framing-B documents and
  don't delete them.

## Commands

```bash
bash scripts/inspect_models.sh              # 0: parameter inventory + low-rank candidates
bash scripts/train_vit_small.sh             # 1: ViT-S baseline
bash scripts/train_vit_base.sh              # 2: ViT-B baseline
bash scripts/evaluate_baselines.sh          # 3: baseline metrics
bash scripts/compress_sweep_vit_small.sh    # 4: 9-point compression sweep
bash scripts/finetune_compressed_example.sh # 5: recovery fine-tune
bash scripts/summarize.sh                   # 6: summary.csv + plots
bash scripts/analyze_spectra.sh             # B: spectra, stable rank, conditioning
bash scripts/approximation_curves.sh        # B: error vs rank / storage
bash scripts/randomized_svd_benchmark.sh    # B: exact vs randomized SVD
bash scripts/layer_error_analysis_vit_small.sh # B: hook-based layer-output error
bash scripts/run_matrix_math_pack.sh        # B: whole math block
```

Every module is also runnable directly (`python3 -m aerial_svd.<name> --help`); the scripts are thin
wrappers over configs in `configs/`.

## Multi-agent workflow

Defined in **[`docs/agent-workflow.md`](docs/agent-workflow.md)** — scratch layout, baseline capture,
GPU gating tiers, launch/resume commands, loop limits, acceptance rules. Read it before coordinating
work here.

Always-on rules, regardless of mode:

- Work is coordinated by `agent-loop` (contract → assignment → review → accept); workers follow
  `agent-task`. Contract artifacts live in `.scratch/<task>/` — `spec.md` owns requirements,
  `handoff.md` owns state. Resume from files, never from memory.
- A worker returns `ready-for-review` with evidence; only the coordinator marks `complete`. Exit code
  0 is not acceptance.
- Do not launch another agent from an assigned session.
- One editing/running session at a time.
- No commits, pushes, or branch operations unless the contract explicitly authorizes them.

## Repo state worth knowing

- `.gitignore` excludes `docs/`, `matrix_results/`, `requirements.txt`, `Dockerfile`, `.qwen/`,
  `.agents/` — so `git status` will **not** show changes in the research notes, the math-block
  results, or the skills this workflow depends on. `git add -f` is needed to track them.
- `.scratch/` is not yet in `.gitignore`; add it before running the workflow.
- No tests, no CI, no linter, no type checker, no packaging. Verification is running the scripts and
  inspecting artifacts.
- `torch.cuda.amp.GradScaler` is deprecated in favor of `torch.amp.GradScaler("cuda", ...)`; the
  current form works on the pinned image.
- `summarize.py` emits `NaN` rows for compression checkpoints lacking a paired evaluation result.
