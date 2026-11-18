# Воспроизведение на ViT-Base

Все **новые** локальные веса, банки факторов, состояния AdamW, статистики и логиты находятся в одном каталоге **ViT-Base-AID** в корне Windows-раздела. Исходные веса Base и Small, AID и замороженный split остались в проекте. Три ссылки в `artifacts/experiments` ведут к одноимённым подкаталогам этого каталога. Для ordinary использован суффикс `_ntfs`, поскольку исходный идентификатор занят [первой проверкой ресурсов](results/aid_base_svd_20261118/PREFLIGHT_RU.md).

Из корня checkout проверить, куда указывают ссылки. Переменная внешнего каталога включает его bind mount по тому же абсолютному пути в контейнере. В контейнер попадает только каталог исследования. Перед каждой стадией launcher проверяет свободное место на целевом разделе. Для системного раздела этой рабочей станции используется отдельный порог 2 GiB: временные файлы моделей создаются на NTFS рядом с результатом, Docker overlay остаётся на ext4.

```bash
S=artifacts/experiments/aid_base_svd_20261118_ntfs
A=artifacts/experiments/aid_base_activation_20261118
F=artifacts/experiments/aid_base_final_20261118
C=configs/aid_vit_base.yaml
W=artifacts/checkpoints/jax/vit_base.npz
export AERIAL_SVD_EXTERNAL_RUNS="$(dirname "$(realpath -e "$S")")"
export AERIAL_SVD_MIN_FREE_BYTES=34359738368
findmnt -T "$AERIAL_SVD_EXTERNAL_RUNS"
df -B1 --output=source,avail,iavail,target "$AERIAL_SVD_EXTERNAL_RUNS" .
```

Серия завершена и проверена. Задания `aid-base-measure-20261118` и `aid-base-finish-20261118` последовательно выполнили измерения, аудит, фиксацию списка и test. Каталог ViT-Base-AID занимает 166 693 466 112 байт по du. Свободное место на NTFS изменилось с 525 895 241 728 до 359 201 964 032 байт. Ниже приведены команды исходных стадий для воспроизводимости. Новое независимое воспроизведение требует собственных корней вывода. [Итоговый отчёт](results/aid_base_final_20261118/REPORT_RU.md) · [Сравнение архитектур](results/aid_base_final_20261118/small_vs_base/REPORT_RU.md)

Перечислитель возобновляет каждую из серий отдельно. Следующий пример показывает вызов обычной глобальной сетки, проверяя хэши завершённых стадий. Остальные значения `--series` — `regions`, `single`, `finetune`, `benchmark`. Для setup обязателен эталон Base `--expected-top1 0.9713333333333334`. Для activation-SVD передавать тот же Base config/checkpoint, собственный корень, `--activation-bank "$A/04_bank" --reference-dir "$S/05_dense"`. Allocation использует тот же банк. Две activation FT задаются явно через `--finetune-configurations global_ffn_r0500 global_all_blocks_r0500`. В aggregate и catalog использовать только завершённые Base-каталоги.

```bash
python3 scripts/lowrank_series.py --root "$S" --config "$C" --checkpoint "$W" --series global
```

Перед freeze сравнить каталог с контрольным каталогом Small и проверить идентичности всех сохранённых стадий. Задание завершается с ошибкой при несовместимых входах, неполной серии или расхождении SHA. Идентификаторы моделей фиксируются до test, включая dense FT. Во время восстановления после freeze использовать ранее записанный SHA. После первого test допустимы только оставшиеся оценки замороженного списка и анализ сохранённых результатов.

```bash
REF=artifacts/experiments/aid_final_20261118
scripts/run.sh cpu python scripts/lowrank_replication_audit.py \
  --svd-root "$S" --activation-root "$A" --final-root "$F" \
  --reference-catalog "$REF/08_catalog/catalog.json" --reference-selection "$REF/09_selection/selection.json" \
  --output-dir "$F/08_prefreeze_audit"

# После завершения final test и формирования обоих отчётов.
scripts/run.sh cpu python scripts/lowrank_compare_architectures.py \
  --small docs/lowrank/results/aid_final_20261118 --base "$F/10_report" \
  --output-dir "$F/12_comparison"
```

[Ordinary SVD](RUN_STAGES_RU.md) · [Activation-SVD и бюджеты](ACTIVATION_STAGES.md) · [Benchmark, freeze, test](FINAL_STAGES.md)
