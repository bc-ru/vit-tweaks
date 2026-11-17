# Этапы 00–02: команды и интерфейсы

Все команды — из корня проекта, через `scripts/run.sh cpu` (JAX). Используется
точная float32 SVD, симметричные sqrt(S)-факторы, неизменный bias. Существующие
источники и split не меняются. Каждому запуску нужен новый каталог результатов.

## 00_baseline

```bash
scripts/run.sh cpu python -m aerial_svd.lowrank_baseline \
  --config configs/aid_vit_small.yaml \
  --checkpoint artifacts/checkpoints/jax/vit_small_protocol_v1.npz \
  --output-dir artifacts/experiments/my_run/00_baseline --batch-size 8 \
  --expected-top1 0.9733333333333334 --top1-atol 0.005
```

Проверяет архитектуру, плотную структуру, seed, классы, полный непересекающийся
split, fingerprint при его наличии и хэш исходного Torch checkpoint из метаданных.
Оценивает только `val`, без аугментаций, с несглаженной cross-entropy; повторяет
первую порцию логитов для проверки детерминизма. Test не оценивается.

Выходы: `baseline.json`, `manifest.json`. Эталон top-1 и допуск задаются до оценки.
Допуск 0.005 — максимум 7 несовпадающих решений из 1500 при переходе с Torch/AMP;
в реальной проверке top-1 совпал точно. Без эталона `reference_verified=false`.
Для smoke уберите `--expected-top1`, добавьте `--max-val-batches 1 --batch-size 1`:
частичная оценка помечается `partial_smoke_only`, не считается полной baseline.

Текущий split: train/val/test = 7000/1500/1500, 30 классов. Если val отсутствует,
команда останавливается: нужно отдельно зафиксировать подразделение только train.

## 01_factor_checks

```bash
scripts/run.sh cpu python -m aerial_svd.lowrank_factor_checks \
  --config configs/aid_vit_small.yaml \
  --checkpoint artifacts/checkpoints/jax/vit_small_protocol_v1.npz \
  --output-dir artifacts/experiments/my_run/01_factor_checks \
  --scope all_blocks --matrices blocks.0.attn.proj blocks.0.mlp.fc1 \
  --rank-ratios 0.5 0.75 1.0
```

Использует реальный `jax_compression.compress`. Проверяет точный список факторов,
ранги/формы, bias, произведение U_r S_r Vh_r, оптимальную ошибку Фробениуса,
полное восстановление и побитовый roundtrip факторов. Матрицы с ростом числа
коэффициентов не пропускаются. Полный ранг автоматически добавляется как control,
если его нет в запросе. Нормированный допуск float32 — `5e-5`; ошибки сохраняются.
Референс использует ту же SVD-конвенцию, чтобы кратные сингулярные значения не
создавали ложного расхождения оптимальных подпространств. В тестах дополнительно
есть независимый float64-референс на некратном спектре и логиты полного ViT.

Выходы: `factor_checks.json`, `checked_factors.npz`, `manifest.json`. NPZ — архив
диагностики, не model checkpoint. Ключ `check_N|<module>.<factor>` соответствует
строке N в JSON; значения — `first.weight`, `second.weight`, `second.bias`.

## 02_matrix_curves

```bash
scripts/run.sh cpu python -m aerial_svd.lowrank_matrix_curves \
  --config configs/aid_vit_small.yaml \
  --checkpoint artifacts/checkpoints/jax/vit_small_protocol_v1.npz \
  --output-dir artifacts/experiments/my_run/02_matrix_curves \
  --scope all_blocks --matrices blocks.0.attn.proj blocks.0.mlp.fc1 \
  --rank-ratios 0.25 0.5 0.75 1.0
```

Один точный спектр на матрицу; существующая `approximation_records` вычисляет
аналитические хвостовые ошибки с float64-накоплением энергии. Модель не исполняется,
изображения не открываются. Каталог AID не нужен, файл split нужен для идентификации.
Полноранговые аналитические ошибки равны нулю; численное восстановление проверяет 01.

Выходы: `matrix_curves.csv`, `manifest.json`. CSV: `name`, `m`, `n` (stored out/in),
`requested_rank`, `requested_ratio`, `actual_rank`, `relative_frobenius_error`,
`relative_spectral_error`, `original_coefficients`, `factor_coefficients`,
`coefficient_savings`, `checkpoint_sha256`, `checkpoint`, `method`, `dtype`,
`svd_seconds`. Числа коэффициентов без bias; отрицательная экономия сохраняется.
Время — синхронизированный cold-вызов спектра с dispatch/transfer, не инференс.

## Общие правила

`--config`, `--checkpoint`, `--output-dir` обязательны. `--device` по умолчанию cpu;
GPU: `scripts/run.sh gpu ... --device gpu`, только при свободной карте.
Для матричных этапов scope: `ffn`, `attn`, `all_blocks` (без head), `all` (с head).
`--matrices` — точный подсписок; без него берётся scope целиком. Можно заменить доли
на `--ranks 96 192 384`. Доли округляются вниз, минимум 1. Неверные ранги,
неизвестные/дублированные имена и выход за scope — ошибка.

Manifest идентифицирует код, checkpoint/config/split, параметры, матрицы/ранги,
метод/dtype и среду. Статус `running`/`failed`/`complete`, при успехе — хэши выходов.
Запись атомарная; каталог не перезаписывается. Повторяется один этап в новом пути.
`complete` без проверенных зависимостей не является научным результатом.

Текущий NPZ — модель; старый `.scratch/jax-migration/vit_small.npz` был архивом
трёх матриц и не может использоваться как checkpoint. Форматы не смешиваются.
Историческая политика skip читается явно через `--legacy-skip-non-saving` старых
JAX evaluate/finetune, с отдельной маркировкой. Они, как и layer_error_analysis,
пока используют test; для выбора параметров не подходят. Этапы 03–09 здесь не реализованы.
