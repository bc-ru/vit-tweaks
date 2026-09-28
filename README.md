# Aerial ViT SVD Compression Codebase

Small PyTorch/timm codebase for the thesis subproblem:

> Fine-tune ViT-Small / ViT-Base on aerial imagery, compress selected `Linear` layers by truncated SVD, then measure accuracy, parameter count, checkpoint size, GPU memory, latency, and throughput.

The code expects an `ImageFolder` dataset. For AID, arrange the data as:

```text
./data/AID/
  Airport/
    *.jpg
  BareLand/
    *.jpg
  BaseballField/
    *.jpg
  ...
```

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH="$PWD/src:$PYTHONPATH"
```

## 0. Inspect ViT-Small and ViT-Base

```bash
bash scripts/inspect_models.sh
```

This writes a parameter inventory and low-rank candidate table.

## 1. Train ViT-Small baseline

```bash
bash scripts/train_vit_small.sh
```

Main output:

```text
runs/aid_vit_small/best.pt
runs/aid_vit_small/history.json
runs/aid_vit_small/final_metrics.json
```

## 2. Train ViT-Base baseline

ViT-Base is heavier. The default config is conservative for a 12 GB GPU: batch size 16 with gradient accumulation 2.

```bash
bash scripts/train_vit_base.sh
```

## 3. Evaluate baselines

```bash
bash scripts/evaluate_baselines.sh
```

## 4. Compress ViT-Small and evaluate sweep

```bash
bash scripts/compress_sweep_vit_small.sh
```

This runs:

```text
mode in {ffn, attn, all}
rank_ratio in {0.75, 0.50, 0.25}
```

Outputs:

```text
checkpoints/*.pt
checkpoints/*.compression.json
results/*.json
logs/*.log
```

## 5. Fine-tune compressed model after SVD

Example recovery run for FFN compression at rank ratio 0.50:

```bash
bash scripts/finetune_compressed_example.sh
```

This is the experiment that answers whether accuracy lost after post-hoc SVD can be recovered.

## 6. Summarize results

```bash
bash scripts/summarize.sh
```

Outputs:

```text
results/summary.csv
plots/accuracy_vs_params.png
plots/latency_vs_params.png
plots/accuracy_vs_rank_*.png
```

## Main thesis interpretation

The experiment studies the map:

```text
image -> visual encoder -> compact representation
```

This is not the full UAV/RL task yet. It is the computational subproblem: how large a visual neural component can be under onboard-like memory and latency constraints.

## Practical 12 GB GPU settings

Recommended starting point:

| Model | Batch | AMP | Grad accumulation |
|---|---:|---|---:|
| ViT-Small | 64 | yes | 1 |
| ViT-Base | 16 | yes | 2 |

If you hit OOM, reduce `batch_size` first. Keep `eval_batch_size` separate.

---

## Вычислительно-математический блок

Помимо обучения и SVD-сжатия моделей, репозиторий содержит отдельный блок для анализа матриц весов как объектов вычислительной линейной алгебры.

Цель блока — не выбрать “лучшую модель”, а исследовать:

- сингулярные спектры матриц слоёв;
- численный и стабильный ранг;
- обусловленность;
- ошибку низкоранговой аппроксимации;
- применимость randomized SVD;
- локальную ошибку выхода слоя при замене \(W\) на \(W_r\).

### Запуск спектрального анализа

```bash
bash scripts/analyze_spectra.sh
```

Результаты:

```text
matrix_results/spectra/*_spectral_stats.csv
matrix_results/spectra/*_singular_decay.png
matrix_results/spectra/*_r95_by_kind.png
```

### Запуск кривых ошибки аппроксимации

```bash
bash scripts/approximation_curves.sh
```

Результаты:

```text
matrix_results/approximation/*_approximation_curves.csv
matrix_results/approximation/*_error_vs_storage.png
```

### Сравнение полного и randomized SVD

```bash
bash scripts/randomized_svd_benchmark.sh
```

Результаты:

```text
matrix_results/randomized_svd/*_randomized_svd_benchmark.csv
```

### Локальная ошибка выхода слоя

```bash
bash scripts/layer_error_analysis_vit_small.sh
```

Результаты:

```text
matrix_results/layer_error/*_layer_error.csv
matrix_results/layer_error/*_matrix_vs_layer_error.png
```

### Полный запуск математического блока

```bash
bash scripts/run_matrix_math_pack.sh
```

### Заметки для отчёта и презентации

```text
docs/report_notes_matrix_math.md
docs/presentation_outline_matrix_math.md
docs/advisor_update_matrix_math.md
```
