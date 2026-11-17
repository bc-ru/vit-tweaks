"""08_aggregate: read measured files only; no checkpoint load or model execution."""
import argparse
import csv
import io
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from aerial_svd.lowrank_common import file_identity, publish_json, publish_text, stage_output
from aerial_svd.lowrank_quality import verified_stage


def write_csv(out, name, rows):
    if not rows:
        return
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    publish_text(out / name, stream.getvalue())


def main(argv=None):
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--root', required=True)
    cli.add_argument('--output-dir', required=True)
    args = cli.parse_args(argv)
    root = Path(args.root)
    manifests = []
    for path in sorted(root.rglob('manifest.json')):
        m = json.loads(path.read_text())
        if m['status'] == 'complete' and m['stage'] in ('05_validate', '06_finetune', '07_benchmark', '02_matrix_curves'):
            verified_stage(path.parent)
            manifests.append((path.parent, m))
    dense = [(p, m) for p, m in manifests if m['stage'] == '05_validate' and not m['source_metadata'].get('candidate')]
    if len(dense) != 1:
        raise ValueError('Require exactly one completed dense reference')
    dense_path, dense_manifest = dense[0]
    source_hash = dense_manifest['checkpoint']['sha256']
    curves = [(p, m) for p, m in manifests if m['stage'] == '02_matrix_curves']
    if len(curves) != 1 or curves[0][1]['checkpoint']['sha256'] != source_hash:
        raise ValueError('Require one compatible matrix-curve stage')
    curves_df = pd.read_csv(curves[0][0] / 'matrix_curves.csv')
    lookup = {(r['name'], int(r['actual_rank'])): r for r in curves_df.to_dict('records')}
    rows, matrix_rows, ft_rows, bench_rows = [], [], [], []
    identities = []
    seen = set()
    for directory, m in manifests:
        identities.append(file_identity(directory / 'manifest.json'))
        if m['stage'] == '02_matrix_curves':
            continue
        for key in ('aid_split', 'config'):
            if m[key]['sha256'] != dense_manifest[key]['sha256']:
                raise ValueError(f'Incompatible {key}: {directory}')
        if m['dataset_fingerprint'] != dense_manifest['dataset_fingerprint']:
            raise ValueError('Mixed datasets')
        origin = m['source_metadata'].get('dense_source', m['checkpoint'])['sha256']
        if origin != source_hash:
            raise ValueError('Mixed source weights')
        if m['stage'] == '05_validate':
            v = json.loads((directory / 'validation.json').read_text())
            c = v['candidate']
            name = directory.parent.name if c else 'dense'
            if name in seen:
                raise ValueError(f'Duplicate completed candidate: {name}')
            seen.add(name)
            selected = [lookup[(n, rank)] for n, rank in c['ranks'].items()] if c else []
            rho = next(iter(c['requests'].values()))[0]['requested_ratio'] if c else 1.
            region = name.split('_')[0] if name.startswith(('early_', 'middle_', 'late_')) else ''
            series = 'region' if region else ('single' if name.startswith('single_') else ('global' if c else 'dense'))
            row = dict(configuration=name, series=series, region=region, scope=c['scope'] if c else 'dense',
                       rank_ratio=rho, num_matrices=len(selected), total_params=v['total_params'],
                       param_memory_bytes=v['param_memory_bytes'], checkpoint_bytes=v['checkpoint_bytes'],
                       top1=v['top1'], cross_entropy=v['loss'], logit_l2_mean=v['logit_l2_mean'],
                       logit_l2_p95=v['logit_l2_p95'], logit_relative_frobenius=v['logit_relative_frobenius'],
                       logit_max_abs=v['logit_max_abs'],
                       matrix_relative_frobenius_mean=float(np.mean([r['relative_frobenius_error'] for r in selected])) if selected else 0.,
                       matrix_relative_spectral_max=max([r['relative_spectral_error'] for r in selected], default=0.),
                       full_rank_control=v['full_rank_control'], validation_images=v['num_samples'],
                       checkpoint_sha256=m['checkpoint']['sha256'], source_checkpoint_sha256=source_hash,
                       split_sha256=m['aid_split']['sha256'], validation_manifest=str(directory / 'manifest.json'))
            rows.append(row)
            for r in selected:
                matrix_rows.append(dict(configuration=name, **r))
        elif m['stage'] == '06_finetune':
            v = json.loads((directory / 'finetune.json').read_text())
            ft_rows.append(dict(configuration=directory.parent.name, total_params=v['total_params'], best_epoch=v['best_epoch'],
                                accuracy_before=v['before']['top1'], accuracy_after=v['after']['top1'],
                                ce_before=v['before']['loss'], ce_after=v['after']['loss'],
                                logit_l2_mean_before=v['before']['logit_l2_mean'], logit_l2_mean_after=v['after']['logit_l2_mean'],
                                logit_l2_p95_before=v['before']['logit_l2_p95'], logit_l2_p95_after=v['after']['logit_l2_p95'],
                                epochs=v['protocol']['epochs'], ranks_preserved=v['ranks_preserved']))
        else:
            v = json.loads((directory / 'benchmark.json').read_text())
            for measure in v['measurements']:
                bench_rows.append(dict(configuration='dense' if directory.parent.name == 'dense_control' else directory.parent.name, total_params=v['total_params'],
                                       param_memory_bytes=v['param_memory_bytes'], checkpoint_bytes=v['checkpoint_bytes'],
                                       **{k: value for k, value in measure.items() if k != 'trials_seconds'},
                                       inference_gpu_peak_bytes=v['inference_gpu_peak_bytes']))
    with stage_output(args.output_dir, dict(stage='08_aggregate', inputs=identities, source_checkpoint_sha256=source_hash,
                                            git_commit=os.environ.get('AERIAL_SVD_GIT_COMMIT'),
                                            source_files_sha256={p.name: file_identity(p)['sha256'] for p in Path(__file__).parent.glob('lowrank_*.py')},
                                            note='Existing measured JSON/CSV only; validation; exact SVD')) as out:
        write_csv(out, 'configurations.csv', rows)
        write_csv(out, 'matrix_errors.csv', matrix_rows)
        write_csv(out, 'finetuning.csv', ft_rows)
        write_csv(out, 'benchmarks.csv', bench_rows)
        curves_df.to_csv(out / 'matrix_curves.csv', index=False)
        df = pd.DataFrame(rows)
        baseline = df[df.series == 'dense'].iloc[0]
        plt.rcParams.update({'figure.dpi': 140, 'axes.grid': True, 'grid.alpha': .25})
        def save(name):
            plt.tight_layout(); plt.savefig(out / name); plt.close()
        global_df = df[df.series == 'global']
        for x, filename, xlabel in [('rank_ratio', 'accuracy_rank.png', 'Relative rank'),
                                    ('total_params', 'accuracy_parameters.png', 'Actual model parameters')]:
            plt.figure(figsize=(7, 4))
            for scope, group in global_df.groupby('scope'):
                group = group.sort_values('rank_ratio')
                plt.plot(group[x], 100*group.top1, 'o-', label=scope)
            plt.axhline(100*baseline.top1, color='black', linestyle='--', label='dense')
            plt.xlabel(xlabel); plt.ylabel('Validation accuracy (%)'); plt.legend(); save(filename)
        fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
        for ax, scope in zip(axes, ('ffn', 'attn')):
            for region, group in df[(df.series == 'region') & (df.scope == scope)].groupby('region'):
                group = group.sort_values('rank_ratio')
                ax.plot(group.rank_ratio, 100*group.top1, 'o-', label=region)
            ax.axhline(100*baseline.top1, color='black', linestyle='--')
            ax.set(xlabel='Relative rank', ylabel='Validation accuracy (%)', title=scope); ax.legend()
        save('accuracy_regions.png')
        fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True)
        singles = df[df.series == 'single']
        for ax, kind in zip(axes.flat, ('attn.qkv', 'attn.proj', 'mlp.fc1', 'mlp.fc2')):
            for block in (0, 5, 11):
                group = singles[singles.configuration.str.startswith(f'single_b{block}_{kind}_')].sort_values('rank_ratio')
                ax.plot(group.rank_ratio, 100*group.top1, 'o-', label=f'block {block}')
            ax.axhline(100*baseline.top1, color='black', linestyle='--')
            ax.set(xlabel='Relative rank', ylabel='Validation accuracy (%)', title=kind); ax.legend()
        save('accuracy_single.png')
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        curves_df['kind'] = curves_df.name.str.split('.').str[2:].str.join('.')
        for ax, error in zip(axes, ('relative_frobenius_error', 'relative_spectral_error')):
            for kind, group in curves_df.groupby('kind'):
                avg = group.groupby('requested_ratio')[error].mean()
                ax.plot(avg.index, avg.values, 'o-', label=kind)
            ax.set(xlabel='Relative rank', ylabel='Mean across 12 blocks', title=error); ax.legend()
        save('matrix_errors_rank.png')
        if ft_rows:
            ft = pd.DataFrame(ft_rows)
            fig, axes = plt.subplots(2, 1, figsize=(11, 8))
            x = np.arange(len(ft))
            for ax, before, after, label in zip(axes, ('accuracy_before', 'ce_before'), ('accuracy_after', 'ce_after'), ('Validation accuracy', 'Validation cross-entropy')):
                ax.bar(x-.18, ft[before], .36, label='before'); ax.bar(x+.18, ft[after], .36, label='after')
                ax.set_xticks(x, ft.configuration, rotation=25, ha='right'); ax.set_ylabel(label); ax.legend()
            save('finetuning.png')
        counts = df.series.value_counts().to_dict()
        full = df[(df.full_rank_control == True)]
        report = [f'# Exact-SVD ViT-Small на AID\n',
                  f'Измерено validation: {len(df)-1}/168 конфигураций; исходная модель отдельно. Глобальные: {counts.get("global", 0)}/24; области: {counts.get("region", 0)}/48; одиночные: {counts.get("single", 0)}/96. Дообучений: {len(ft_rows)}/10 (9 факторов + dense control). Benchmark: {len(bench_rows)}/16 строк.\n',
                  f'Исходная accuracy: {baseline.top1:.8f}; cross-entropy: {baseline.cross_entropy:.8f}; параметров: {int(baseline.total_params)}. Все оценки — 1500 одинаковых validation-изображений; test не исполнялся. Seed 42, split 7000/1500/1500; checkpoint SHA-256 `{source_hash}`.\n',
                  'Ошибка матриц — аналитический хвост точной float32 SVD с float64 суммированием энергии (этап 02). `matrix_errors.csv` содержит каждую заменённую матрицу и действительный ранг. Среднее Frobenius в общей таблице — невзвешенное среднее относительных ошибок, а не ошибка композиции модели. Факторы симметричны sqrt(S), bias сохранён, высокие ранги не пропускаются.\n']
        if len(full):
            report.append(f'Полный ранг: {len(full)} контролей; максимум относительной ошибки логитов {full.logit_relative_frobenius.max():.3g}, максимум абсолютной разности {full.logit_max_abs.max():.3g}. Контроли прошли допуски 5e-5 и 1e-3 соответственно.\n')
        report.append('## Глобальные конфигурации\n\n|Набор|Ранг|Параметры|Accuracy|CE|Logit L2 mean|\n|---|---:|---:|---:|---:|---:|')
        for r in global_df.sort_values(['scope', 'rank_ratio']).itertuples():
            report.append(f'|{r.scope}|{r.rank_ratio:g}|{r.total_params}|{r.top1:.6f}|{r.cross_entropy:.6f}|{r.logit_l2_mean:.5f}|')
        regional = df[(df.series == 'region') & (df.rank_ratio == .5)]
        if len(regional):
            report.append('\n## Положение матриц при ранге 0.5\n\n|Область|Набор|Параметры|Accuracy|CE|\n|---|---|---:|---:|---:|')
            for r in regional.sort_values(['scope', 'region']).itertuples():
                report.append(f'|{r.region}|{r.scope}|{r.total_params}|{r.top1:.6f}|{r.cross_entropy:.6f}|')
        single_low = df[(df.series == 'single') & (df.rank_ratio == .125)]
        if len(single_low):
            report.append('\nОдиночные замены при ранге 0.125 (остальная модель плотная):\n\n|Замена|Accuracy|CE|\n|---|---:|---:|')
            for r in single_low.sort_values('configuration').itertuples():
                report.append(f'|{r.configuration}|{r.top1:.6f}|{r.cross_entropy:.6f}|')
        if ft_rows:
            report.append('\n## Дообучение\n\nОбучались все параметры, включая факторы; фиксированные ранги. 5 эпох, AdamW (b1=.9, b2=.999, eps=1e-8), LR 1e-5 с cosine по эпохам, weight decay .05, label smoothing .1 только для train, batch 8, float32. Выбор максимальной validation accuracy, ранняя эпоха при равенстве. CE в таблицах несглаженная.\n\n|Модель|Accuracy до|После|CE до|После|Лучшая эпоха|\n|---|---:|---:|---:|---:|---:|')
            for r in ft_rows:
                report.append(f'|{r["configuration"]}|{r["accuracy_before"]:.6f}|{r["accuracy_after"]:.6f}|{r["ce_before"]:.6f}|{r["ce_after"]:.6f}|{r["best_epoch"]}|')
        if bench_rows:
            report.append('\n## Инференс и память параметров\n\n|Модель|Batch|Параметры|Байты параметров|Байты NPZ|Median ms|P05–P95 ms|Изображений/s|\n|---|---:|---:|---:|---:|---:|---|---:|')
            for r in bench_rows:
                report.append(f'|{r["configuration"]}|{r["batch_size"]}|{r["total_params"]}|{r["param_memory_bytes"]}|{r["checkpoint_bytes"]}|{r["median_batch_ms"]:.3f}|{r["p05_batch_ms"]:.3f}–{r["p95_batch_ms"]:.3f}|{r["throughput_images_s"]:.1f}|')
        report.append('\n## Наблюдения по измерениям\n')
        for scope, group in global_df.groupby('scope'):
            saving = group[group.total_params < baseline.total_params]
            if len(saving):
                best_saving = saving.sort_values(['top1', 'rank_ratio'], ascending=[False, True]).iloc[0]
                report.append(f'- `{scope}`: наибольшая измеренная accuracy среди точек с меньшим числом параметров — {100*best_saving.top1:.3f}% при ранге {best_saving.rank_ratio:g}, {int(best_saving.total_params)} параметров ({100*(1-best_saving.total_params/baseline.total_params):.2f}% меньше исходной модели).')
        for scope, group in regional.groupby('scope'):
            ordered = group.sort_values('top1')
            if len(ordered) == 3:
                report.append(f'- `{scope}`, ранг 0.5: accuracy от {100*ordered.iloc[0].top1:.3f}% ({ordered.iloc[0].region}) до {100*ordered.iloc[-1].top1:.3f}% ({ordered.iloc[-1].region}) при одинаковом числе замен и параметров.')
        if ft_rows:
            deltas = [r['accuracy_after']-r['accuracy_before'] for r in ft_rows if r['configuration'] != 'dense_control']
            if deltas:
                report.append(f'- Дообучение SVD-моделей: изменение accuracy от {100*min(deltas):+.3f} до {100*max(deltas):+.3f} процентных пункта. Все сравнения до/после сделаны на одной validation; dense control показан отдельно.')
            for r in ft_rows:
                if r['ce_after'] > r['ce_before']:
                    report.append(f'- `{r["configuration"]}`: CE ухудшилась {r["ce_before"]:.6f} → {r["ce_after"]:.6f} при выборе по accuracy; восстановление качества нельзя считать однозначным по всем метрикам.')
                if r['configuration'] == 'dense_control':
                    report.append(f'- Плотный контроль с теми же 5 эпохами: accuracy {100*r["accuracy_before"]:.3f}% → {100*r["accuracy_after"]:.3f}%, CE {r["ce_before"]:.6f} → {r["ce_after"]:.6f}. Сам эффект дополнительного обучения следует учитывать при сравнении с SVD + дообучение.')
        dense_times = {r['batch_size']: r['median_batch_ms'] for r in bench_rows if r['configuration'] == 'dense'}
        for r in bench_rows:
            if r['configuration'] == 'global_all_blocks_r0500' and r['batch_size'] in dense_times:
                report.append(f'- `all_blocks`, ранг 0.5, batch {r["batch_size"]}: {r["median_batch_ms"]:.3f} ms против {dense_times[r["batch_size"]]:.3f} ms у dense; отношение скорости dense/SVD = {dense_times[r["batch_size"]]/r["median_batch_ms"]:.3f}. Уменьшение параметров не обеспечивает ускорение для каждого batch.')
        report.append('\n## Допустимые выводы и ограничения\n\nИзменение accuracy измеряется для одного checkpoint, seed и validation split: статистической гарантии для других обучений нет. Малая ошибка матриц сама по себе не доказывает малую ошибку модели. Выбор по validation может завышать оценку выбранного варианта; финальный test остаётся отдельным этапом. Дообучение оценивается по факту, без предположения восстановления.\n\nBenchmark: RTX 4070, float32, batches 1/16, 5 прогревов и 50 синхронизированных повторов. Компиляция, подготовка изображений и переносы исключены. Байты параметров и размер NPZ — разные измерения; inference GPU peak не измерен и не заменяется пиком процесса. Скорость нельзя выводить из числа параметров.\n\nВсе численные данные: `configurations.csv`, `matrix_curves.csv`, `matrix_errors.csv`, `finetuning.csv`, `benchmarks.csv`; исходные manifests перечислены в manifest данного отчёта. Незавершённые стадии не включены.\n')
        report.append('\nCheckpoints дообучения выбирались среди эпох 1–5; исходная точка сохранена отдельно. Benchmark относится к плотному источнику и исходным SVD-checkpoint до дообучения. Различия в несколько изображений на validation не доказывают статистически значимого преимущества.\n')
        report.append('Графики: [ошибки матриц](matrix_errors_rank.png), [accuracy от ранга](accuracy_rank.png), [начало/середина/конец](accuracy_regions.png), [одиночные замены](accuracy_single.png), [accuracy от параметров](accuracy_parameters.png), [до/после обучения](finetuning.png).\n')
        report.append(f'Полные checkpoint, логиты и manifests стадий сохранены в `{root}`; исходные веса и split не изменялись.\n')
        publish_text(out / 'REPORT_RU.md', '\n'.join(report))
        publish_json(out / 'summary.json', dict(configurations=len(df)-1, by_series=counts, finetuned=len(ft_rows), benchmark_rows=len(bench_rows), full_rank_controls=len(full)))
        print(dict(configurations=len(df)-1, finetuned=len(ft_rows), benchmark_rows=len(bench_rows)), flush=True)


if __name__ == '__main__':
    main()
