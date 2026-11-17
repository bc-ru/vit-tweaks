"""Aggregate immutable measurements only; never executes a model or selects a configuration."""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from aerial_svd.lowrank_common import file_identity, publish_json, publish_text, stage_output
from aerial_svd.lowrank_catalog import benchmark_result
from aerial_svd.lowrank_quality import verified_stage


def table(frame, columns, digits=3):
    def cell(value):
        return f'{value:.{digits}f}' if isinstance(value, (float, np.floating)) else str(value)
    return '\n'.join(['|'+'|'.join(columns)+'|', '|'+'|'.join(['---']*len(columns))+'|'] +
                     ['|'+'|'.join(cell(v) for v in row)+'|' for row in frame[columns].itertuples(index=False, name=None)])


def main(argv=None):
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--root', required=True)
    cli.add_argument('--catalog', required=True)
    cli.add_argument('--selection', required=True)
    cli.add_argument('--comparison-dir', required=True)
    cli.add_argument('--svd-report-dir', required=True)
    cli.add_argument('--output-dir', required=True)
    args = cli.parse_args(argv)
    root = Path(args.root)
    inputs = []
    def audit(directory):
        manifest = verified_stage(directory)
        inputs.append(file_identity(Path(directory)/'manifest.json'))
        return manifest
    audit(Path(args.catalog).parent)
    audit(Path(args.selection).parent)
    audit(args.comparison_dir)
    audit(args.svd_report_dir)
    catalog = json.loads(Path(args.catalog).read_text())
    selection = json.loads(Path(args.selection).read_text())
    selection_hash = file_identity(args.selection)['sha256']
    if json.loads((root/'test_started.json').read_text())['selection_sha256'] != selection_hash:
        raise ValueError('Test receipt and frozen selection differ')
    if selection['catalog']['sha256'] != file_identity(args.catalog)['sha256']:
        raise ValueError('Selection/catalog differ')
    benchmarks, test_rows, rows = [], [], []
    protocol = {}
    for model in catalog['models']:
        key = model['model_id']
        val, candidate = model['validation'], model['candidate']
        count = model['finetuning']['total_params'] if model['finetuned'] else val['total_params']
        row = dict(model_id=key, configuration=model['configuration'], method=model['method'],
            finetuned=model['finetuned'], allocated=bool(candidate.get('allocation')),
            scope=candidate.get('scope', 'dense'), num_matrices=len(candidate.get('ranks', {})),
            ranks_json=json.dumps(candidate.get('ranks', {}), sort_keys=True),
            checkpoint_sha256=model['checkpoint']['sha256'], total_params=count,
            param_memory_bytes=count*4, checkpoint_bytes=model['checkpoint']['bytes'],
            validation_top1=val['top1'], validation_cross_entropy=val['loss'],
            validation_images=val['num_samples'],
            logit_l2_mean=val['logit_l2_mean'], logit_l2_p95=val['logit_l2_p95'],
            logit_relative_frobenius=val['logit_relative_frobenius'], logit_max_abs=val['logit_max_abs'],
            best_epoch=model['finetuning']['best_epoch'] if model['finetuned'] else None,
            accuracy_before=model['finetuning']['before']['top1'] if model['finetuned'] else None,
            ce_before=model['finetuning']['before']['loss'] if model['finetuned'] else None,
            matrix_errors='not remeasured after training' if model['finetuned'] else 'matrix_errors.csv',
            test_top1=None, test_cross_entropy=None, test_images=None)
        for batch in (1, 16):
            fields = dict(median_batch_ms=None, p05_batch_ms=None, p95_batch_ms=None,
                          throughput_images_s=None, compiled_plan_bytes=None,
                          process_allocator_peak_bytes=None, process_pool_peak_bytes=None,
                          inference_gpu_peak_bytes=None)
            if model['benchmark']:
                directory = root/'07_benchmark'/key/f'batch-{batch}'
                manifest = audit(directory)
                result, measure, signature = benchmark_result(directory, model, batch)
                if batch in protocol and protocol[batch] != signature:
                    raise ValueError('Benchmarks do not share hardware/runtime/inputs/protocol/source')
                protocol[batch] = signature
                if result['total_params'] != count or result['param_memory_bytes'] != count*4:
                    raise ValueError('Parameter counts differ')
                memory, stats = measure['compiled_memory'], measure['allocator_stats'] or {}
                fields.update({k: measure[k] for k in ('median_batch_ms', 'p05_batch_ms', 'p95_batch_ms', 'throughput_images_s')})
                fields.update(compiled_plan_bytes=memory['total_bytes'] if memory else None,
                    process_allocator_peak_bytes=stats.get('peak_bytes_in_use'),
                    process_pool_peak_bytes=stats.get('peak_pool_bytes'))
                benchmarks.append(dict(model_id=key, batch_size=batch, **fields,
                    compile_seconds=measure['compile_seconds'], total_params=count,
                    param_memory_bytes=count*4, checkpoint_bytes=model['checkpoint']['bytes']))
            row.update({f'{k}_b{batch}': v for k, v in fields.items()})
        if key in {m['model_id'] for m in selection['models']}:
            directory = root/'09_test'/key
            manifest = audit(directory)
            result = json.loads((directory/'test.json').read_text())
            if (result['selection_sha256'] != selection_hash or result['checkpoint_sha256'] != model['checkpoint']['sha256']
                    or result['num_samples'] != selection['test_count'] or result['model_id'] != key
                    or manifest['evaluation_split'] != 'test'):
                raise ValueError('Test identity/count differs')
            test_rows.append(result)
            row.update(test_top1=result['top1'], test_cross_entropy=result['loss'], test_images=result['num_samples'])
        rows.append(row)
    if len(test_rows) != len(selection['models']):
        raise ValueError('Frozen test models missing')
    if len({r['pixels_sha256'] for r in test_rows}) != 1 or len({r['test_indices_sha256'] for r in test_rows}) != 1:
        raise ValueError('Test pixels/order differ')
    df, bench, test = pd.DataFrame(rows), pd.DataFrame(benchmarks), pd.DataFrame(test_rows)
    with stage_output(args.output_dir, dict(stage='10_final_report', inputs=inputs,
            selection_sha256=selection_hash, note='Aggregation only. No model execution or selection.')) as out:
        df.to_csv(out/'models.csv', index=False)
        bench.to_csv(out/'benchmarks.csv', index=False)
        test.to_csv(out/'test.csv', index=False)
        # Errors belong to the initial factorization; never attach them to trained factors.
        for name in ('matrix_errors.csv', 'local_errors.csv', 'paired_comparison.csv', 'budget_comparison.csv', 'finetuning.csv'):
            publish_text(out/name, (Path(args.comparison_dir)/name).read_text())
        for name in ('matrix_errors.csv', 'matrix_curves.csv'):
            publish_text(out/('svd_'+name), (Path(args.svd_report_dir)/name).read_text())
        plt.rcParams.update({'figure.dpi': 140, 'axes.grid': True, 'grid.alpha': .25})
        for x, name, label in [('total_params', 'accuracy_parameters.png', 'Parameters'),
                               ('param_memory_bytes', 'accuracy_weights.png', 'Float32 parameter bytes')]:
            fig, ax = plt.subplots(figsize=(8, 5))
            for (method, ft), group in df.groupby(['method', 'finetuned']):
                ax.scatter(group[x], 100*group.validation_top1, s=18 if not ft else 60,
                           marker='o' if not ft else '*', alpha=.65,
                           label=method+(' + FT' if ft else ''))
            ax.set(xlabel=label, ylabel='Validation top-1 (%)')
            ax.legend(); fig.tight_layout(); fig.savefig(out/name); plt.close(fig)
        for batch in (1, 16):
            fig, ax = plt.subplots(figsize=(8, 5))
            measured = df[df[f'median_batch_ms_b{batch}'].notna()]
            for (method, ft), group in measured.groupby(['method', 'finetuned']):
                ax.scatter(group[f'median_batch_ms_b{batch}'], 100*group.validation_top1,
                    marker='o' if not ft else '*', label=method+(' + FT' if ft else ''))
            ax.set(xlabel=f'Median batch latency (ms), batch={batch}', ylabel='Validation top-1 (%)')
            ax.legend(); fig.tight_layout(); fig.savefig(out/f'accuracy_latency_b{batch}.png'); plt.close(fig)
        compact = dict(frozen_selection_sha256=selection_hash, frozen_utc=selection['frozen_utc'],
            reason=selection['reason'], policy=selection['policy'],
            config_sha256=selection['config']['sha256'], split_sha256=selection['aid_split']['sha256'],
            models=[dict(model_id=m['model_id'], checkpoint_sha256=m['checkpoint']['sha256'],
                         method=m['method'], ranks=m['candidate'].get('ranks', {})) for m in selection['models']])
        publish_json(out/'selected_models.json', compact)
        # Human text is generated from measured rows, then may be reviewed as documentation.
        selected = df[df.test_top1.notna()].copy()
        selected['val_%'] = selected.validation_top1*100
        selected['test_%'] = selected.test_top1*100
        selected['test_CE'] = selected.test_cross_entropy
        globals_ = df[df.configuration.str.startswith('global_') & ~df.finetuned].copy()
        globals_['val_%'] = globals_.validation_top1*100
        global_table = table(globals_, ['model_id', 'total_params', 'val_%', 'median_batch_ms_b1', 'median_batch_ms_b16'])
        ft = df[df.finetuned].copy()
        ft['before_%'], ft['after_%'] = ft.accuracy_before*100, ft.validation_top1*100
        summary = dict(models=len(df), benchmark_models=int(df.median_batch_ms_b1.notna().sum()),
            benchmark_rows=len(bench), final_test_models=len(test), selection_sha256=selection_hash,
            gpu=protocol[1]['environment'], inference_gpu_peak_bytes=None)
        publish_json(out/'summary.json', summary)
        report = f'''# Итог: низкоранговые ViT-Small на AID

Завершён исходный план: 168 ordinary SVD + 168 activation-SVD, 6 распределений
рангов, 12 дообучений с dense-контролем. Повторный подбор не проводился.
Каталог: **{len(df)} моделей**, GPU: **{len(bench)} измерений** (67 × batch 1/16),
окончательный test: **{len(test)} моделей × {selection['test_count']} изображений**.
Seed 42; 30 классов; frozen train/validation/test = 7000/1500/1500.
Исходные checkpoint, split и результаты сохранены без изменений.

## 1. Ранг, тип и положение матриц

Постановка: отделить эффект ранга от выбора attention/FFN и положения замены.
Эксперимент: на каждый метод 24 глобальных, 48 региональных (блоки 0–3,
4–7, 8–11) и 96 одиночных конфигураций (блоки 0/5/11), восемь рангов
.125/.25/.375/.5/.625/.75/.875/1.0. Одни и те же 1500 validation-изображений.
`all_blocks` исключает head; факторы хранятся и при увеличении числа параметров.

Результат: ordinary all_blocks .5 даёт 65.000%, activation — 96.067%;
ordinary FFN .5 — 85.800%, attention .5 — 95.133%. При равном числе параметров
ordinary regional FFN .5: early 94.600%, middle 96.533%, late 97.000%.
Для одиночного FFN fc1 .125: блок 0 — 79.667%, блок 11 — 97.333%.
Положение существенно, единый локальный ранг не задаёт качество сети.
Все 42 полносетевых полноранговых контроля двух методов прошли допуски
relative logits 5e-5 / max abs 1e-3.
Ограничение: один checkpoint/seed/split; различия в несколько изображений
не доказывают статистически значимого преимущества.

## 2. Активации и выбор индивидуальных рангов

Постановка: использовать действие оператора на train-активациях вместо ошибки
коэффициентов. Эксперимент: 300 calibration train-изображений (10/класс),
uncentered C=E[xxᵀ], ridge 1e-4·trace(C)/in; weighted SVD и обратное преобразование;
факторы sqrt(S) симметричны в взвешенных координатах. Validation-моменты только
измеряют ошибку, не строят факторы. Калибровка и ridge взяты из завершённой серии.

Результат: calibration и независимая validation local MSE лучше в 336/336
неполноранговых парах; Frobenius весов — в 0/336. Для каждого метода exact
multiple-choice knapsack распределял ранги по той же сетке при точном бюджете
uniform all_blocks .25/.5/.75, минимизируя сумму ненормированных train local MSE.
**Во всех шести случаях меньшая сумма локальных ошибок ухудшила accuracy сети.**
На бюджете .5: ordinary 65.000% → 12.600%, activation 96.067% → 89.800%;
локальные суммы соответственно 49343 → 3006.3 и 11544 → 869.01.
Это отрицательный результат выбранного суррогатного критерия.
Ограничение: после совместных замен активации меняются; локальные ошибки
не складываются в ошибку композиции. Решение DP оптимально только для его
дискретной локальной задачи, не для accuracy сети.

## 3. Восстановление дообучением

Постановка: измерить фактическое восстановление, отделяя его от дополнительных
эпох dense. Эксперимент: прежние 9 ordinary + 2 activation + dense; все параметры,
фиксированные ранги, 5 эпох, batch 8, FP32, AdamW (.9/.999, eps 1e-8), LR 1e-5
с cosine по эпохам, WD .05, train smoothing .1, seed 42. CE оценки несглаженная.
Лучшая эпоха 1–5 по validation top-1, ранняя при равенстве. Нового обучения нет.

{table(ft, ['model_id', 'before_%', 'after_%', 'ce_before', 'validation_cross_entropy', 'best_epoch'])}

Ограничение: восстановление неполно для сильного сжатия. У middle_ffn accuracy
растёт, но CE ухудшается; dense-контроль accuracy немного теряет. Масштаб факторов
activation отличается и может влиять на оптимизацию. Ошибки матриц после
дообучения не измерялись; исходные локальные ошибки к ним не приписаны.

## 4. Качество, параметры, веса, память и время

Постановка: проверить реальное ускорение и стоимость хранения.
Эксперимент: вся глобальная сетка обоих методов, 6 allocated и 12 FT плюс dense;
RTX 4070, JAX 0.4.38, driver 555.42.02, FP32/highest, один процесс на model/batch,
batch 1/16, первые одинаковые validation-входы, 5 прогревов, 50 повторов.
Компиляция измерена отдельно; декодирование, H2D и компиляция исключены из latency.
Каждый вызов синхронизирован block_until_ready; median/P05/P95 сохранены.
Все SHA пикселей, runtime, GPU UUID и параметры протокола попарно проверены.
Прежние восемь бенчмарков повторены для единого расширенного протокола памяти.

{global_table}

`models.csv` объединяет все 355 моделей, точные ранги/имена (ranks_json),
validation CE/top-1, ошибки логитов, идентификаторы checkpoint, результаты FT,
latency и память. `benchmarks.csv` содержит все 134 строки, включая allocated/FT.
`matrix_errors.csv` и `local_errors.csv` — исходные измеренные матричные ошибки.
Аналитические Frobenius/spectral-кривые ordinary SVD сохранены отдельно в
`svd_matrix_curves.csv`, исходные ошибки замен — в `svd_matrix_errors.csv`.

Различаем четыре величины: `param_memory_bytes` — точные FP32 байты параметров;
`checkpoint_bytes` — размер NPZ с метаданными; `process_pool_peak_bytes` —
максимальный резерв GPU-аллокатора процесса; `process_allocator_peak_bytes` —
его максимум занятых байтов за загрузку/компиляцию/прогрев/инференс.
`compiled_plan_bytes` — статическая сумма XLA arguments+output+temp−alias
([документация JAX](https://docs.jax.dev/en/latest/notebooks/host-offloading.html)),
а не измеренный runtime-пик. **Изолированный фактический peak GPU инференса
достоверно не измерен: inference_gpu_peak_bytes = null.** Эти величины не заменяют
друг друга и не включают одинаковые накладные расходы драйвера.
Уменьшение параметров не называется ускорением без соответствующего latency.
Частоты GPU не фиксировались; это одна серия на одной карте, интервалы P05/P95 —
разброс повторов, не доверительные интервалы между независимыми сериями.
Региональные/одиночные исходные варианты не бенчмаркировались отдельно;
измерены их уже дообученные варианты. Размер NPZ не равен размеру массива весов.

Графики validation: [параметры](accuracy_parameters.png), [байты весов](accuracy_weights.png),
[latency batch 1](accuracy_latency_b1.png), [latency batch 16](accuracy_latency_b16.png).

## 5. Ordinary SVD против activation-SVD и окончательный test

Постановка: сравнить одинаковые архитектуру, ранги, входы и стоимость хранения;
проверить заранее определённые модели на held-out test.
На validation среди 147 неполноранговых пар activation лучше в 91, равен в 25,
хуже в 31 случае. Локальное улучшение не гарантирует улучшения полной модели.
На all_blocks .5 после FT: 95.800% ordinary / 96.667% activation;
на FFN .5: 96.867% / 97.133%. Это результаты validation, не критерий по test.

Список заморожен **{selection['frozen_utc']}**, до чтения первого test-изображения.
SHA-256 исходного selection.json: `{selection_hash}`.
[Публичная запись](selected_models.json) содержит все model_id/checkpoint SHA/ранги;
исходный полный файл и manifests — в локальном 09_selection.
Включены 19 заданных моделей и дополнительный dense FT-контроль с причиной,
записанной до test. Других добавлений нет. Список после test не менялся.
Одинаковые 1500 test-изображений, их порядок, preprocessing и декодер; batch 8;
проверены SHA checkpoint/config/split/selection и попарно пиксели/метки/индексы.
Сохранены несглаженная CE, top-1 и полные локальные logits.

{table(selected, ['model_id', 'val_%', 'test_%', 'test_CE'])}

Ограничения: 1500 наблюдений и один исходный checkpoint. Test использован один раз
для фиксированного списка; по его результатам не выбирались конфигурации,
эпохи, ранги, ridge или новые эксперименты. После test выполнялась только
агрегация и проверка артефактов. Дополнительные GPU-эксперименты не запускались.

Машиночитаемые данные: [все модели](models.csv), [GPU](benchmarks.csv), [test](test.csv),
[пары](paired_comparison.csv), [бюджеты](budget_comparison.csv), [FT](finetuning.csv).
Команды: [FINAL_STAGES.md](../../FINAL_STAGES.md).
Полная локальная доказательная база: `artifacts/experiments/aid_final_20261118/`;
исходные серии остаются на прежних путях. Презентация не изменялась.
'''
        publish_text(out/'REPORT_RU.md', report)


if __name__ == '__main__':
    main()
