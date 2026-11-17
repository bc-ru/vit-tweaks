# Завершение утверждённой серии: команды

Результаты и ограничения: [единый итоговый отчёт](results/aid_final_20261118/REPORT_RU.md).
Это независимые этапы; повторять завершённые вычисления не требуется.
После начала окончательного test новые эксперименты запрещены.
Старые checkpoint, split и результаты неизменяемы. Новый запуск требует нового
выходного каталога; наличие файлов без `manifest.status=complete` не означает успех.

```bash
F=artifacts/experiments/aid_final_20261118
S=artifacts/experiments/aid_svd_20261117
A=artifacts/experiments/aid_activation_20261117

# Только аудит сохранённых данных: 355 моделей, из них 67 для benchmark.
scripts/run.sh cpu python -m aerial_svd.lowrank_catalog \
  --svd-root "$S" --activation-root "$A" --output-dir "$F/08_catalog"

# До test: 134 отдельных процесса, неизменный launcher/runtime/GPU.
python3 scripts/lowrank_finalize.py --root "$F" --series benchmark \
  --catalog "$F/08_catalog/catalog.json"

# Проверки выполняются до реального test; тестовые изображения здесь синтетические.
scripts/check.sh

# Только после всех benchmark: проверить общность протокола и заморозить 20 моделей.
scripts/run.sh cpu python -m aerial_svd.lowrank_freeze \
  --catalog "$F/08_catalog/catalog.json" --finalization-root "$F" \
  --output-dir "$F/09_selection"
# Сохранить выведенный SHA-256 отдельно; он обязателен при каждом test-вызове.
sha256sum "$F/09_selection/selection.json" > "$F/selection.sha256"
SELECTION_SHA=$(cut -d ' ' -f 1 "$F/selection.sha256")

# Последняя вычислительная серия. Не запускать обучение/benchmark после неё.
python3 scripts/lowrank_finalize.py --root "$F" --series test \
  --catalog "$F/08_catalog/catalog.json" --selection "$F/09_selection/selection.json" \
  --selection-sha256 "$SELECTION_SHA"

# Только сохранённые измерения, без исполнения модели или выбора конфигураций.
scripts/run.sh cpu python -m aerial_svd.lowrank_final_report \
  --root "$F" --catalog "$F/08_catalog/catalog.json" \
  --selection "$F/09_selection/selection.json" \
  --svd-report-dir "$S/08_final" \
  --comparison-dir "$A/08_comparison.retry-002" --output-dir "$F/10_report"
```

Любая модель отдельно: `scripts/run.sh gpu python -m aerial_svd.lowrank_benchmark
--config CONFIG --checkpoint CHECKPOINT --output-dir NEW_DIR --device gpu
--batch-sizes 1 --warmup 5 --repeats 50` (только до test).
Этап `lowrank_test` принимает те же общие аргументы плюс `--selection`,
`--selection-sha256`, `--model-id`, `--finalization-root`, `--batch-size 8`.
Dense должен быть первым: его сохранённые test-пиксели/метки/порядок проверяются
у всех остальных. Лимита числа test-батчей или выбора эпох здесь нет.

Launcher возобновляет только завершённые стадии с проверенными выходными хэшами,
checkpoint и аргументами протокола. Незавершённая стадия требует диагностики
и отдельного каталога повтора; сохранённые результаты не перезаписываются.
`test_started.json` связывает серию с единственным SHA списка и версиями исходников;
benchmark-launcher после его появления отказывает в запуске.

| Артефакт | Назначение |
|---|---|
| `08_catalog/catalog.json` | Все model_id, checkpoint SHA, ранги, validation/FT и ссылки на доказательства |
| `07_benchmark/<model_id>/batch-{1,16}/` | 50 времён, median/P05/P95, время компиляции, FP32 байты и NPZ байты, план XLA, lifetime allocator counters |
| `09_selection/selection.json` | Полный неизменяемый список, SHA checkpoint/config/split, порядок test, decoder/preprocessing и причина дополнительного dense FT |
| `selection.sha256` | Независимо записанный хэш списка до первого test |
| `09_test/<model_id>/` | `test.json`, logits/labels/indices и пиксельный digest, manifest |
| `10_report/` | Один русский отчёт, таблицы, 4 графика, компактная копия frozen-списка |

`inference_gpu_peak_bytes=null`: изолированный фактический пик не измерен.
`compiled_plan_bytes` — статический план XLA, `process_allocator_peak_bytes` —
весь процесс включая setup/compile/warmup, `process_pool_peak_bytes` — резерв пула.
Поля нельзя интерпретировать как взаимозаменяемые оценки пика инференса.
Пустая ячейка CSV означает «не измерено», а не ноль. Локальные ошибки матриц
измерены до FT; `ranks_json` содержит фактические ранги также для allocated и FT.
Полные локальные manifests содержат пообразцовые технические идентификаторы;
публичные таблицы не содержат изображений, индексов примеров или логитов.
