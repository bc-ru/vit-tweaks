"""Compare two completed AID snapshots from measured CSVs, without model execution."""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from aerial_svd.lowrank_common import file_identity, publish_text, stage_output
from aerial_svd.lowrank_final_report import table
from aerial_svd.lowrank_quality import verified_stage


def read_study(directory, architecture):
    verified_stage(directory)
    models = pd.read_csv(directory / 'models.csv').set_index('model_id', verify_integrity=True)
    dense = models.loc['dense__dense']
    if len(models) != 355 or models.test_top1.count() != 20 or not (models.validation_images == 1500).all():
        raise ValueError(f'Incomplete study: {architecture}')
    models['architecture'] = architecture
    models['relative_parameters'] = models.total_params / dense.total_params
    models['compression_ratio'] = dense.total_params / models.total_params
    for batch in (1, 16):
        latency = f'median_batch_ms_b{batch}'
        if models[latency].count() != 67 or (models[latency].dropna() <= 0).any():
            raise ValueError(f'Incomplete latency grid: {architecture}, batch={batch}')
        models[f'latency_over_dense_b{batch}'] = models[latency] / dense[latency]
    for split in ('validation', 'test'):
        models[f'{split}_delta_pp'] = 100 * (models[f'{split}_top1'] - dense[f'{split}_top1'])
    return models


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--small', type=Path, required=True)
    cli.add_argument('--base', type=Path, required=True)
    cli.add_argument('--output-dir', required=True)
    args = cli.parse_args()
    dirs = (args.small, args.base)
    first, second = (json.loads((p/'selected_models.json').read_text()) for p in dirs)
    if (first['split_sha256'] != second['split_sha256'] or
            [m['model_id'] for m in first['models']] != [m['model_id'] for m in second['models']]):
        raise ValueError('Selection or split differs across architectures')
    small, base = (read_study(p, a) for p,a in zip(dirs, ('ViT-Small','ViT-Base')))
    if set(small.index) != set(base.index) or small.loc['dense__dense','total_params'] >= base.loc['dense__dense','total_params']:
        raise ValueError('Expected matched Small and Base catalogues')
    all_models = pd.concat([small, base]).reset_index()
    requested = all_models.configuration.str.extract(r'_r(\d{4})$', expand=False)
    all_models['rank_ratio'] = pd.to_numeric(requested) / 1000
    columns = ['model_id', 'architecture', 'scope', 'method', 'finetuned', 'allocated', 'rank_ratio', 'ranks_json',
               'total_params', 'relative_parameters', 'compression_ratio', 'param_memory_bytes',
               'validation_top1', 'validation_delta_pp', 'validation_cross_entropy',
               'test_top1', 'test_delta_pp', 'test_cross_entropy',
               'median_batch_ms_b1', 'latency_over_dense_b1', 'median_batch_ms_b16', 'latency_over_dense_b16']
    with stage_output(args.output_dir, dict(stage='architecture_comparison',
            inputs=[file_identity(p/'manifest.json') for p in dirs])) as out:
        all_models[columns].to_csv(out/'architectures.csv', index=False)
        subset = all_models.loc[all_models.configuration.str.startswith('global_') & ~all_models.finetuned]
        fig, axes = plt.subplots(1, 3, figsize=(12,4), sharey=True)
        for ax, scope in zip(axes, ('attn', 'ffn', 'all_blocks')):
            for (arch, method), group in subset[subset.scope == scope].groupby(['architecture','method']):
                group = group.sort_values('rank_ratio')
                ax.plot(group.rank_ratio, group.validation_delta_pp, 'o-', label=f'{arch} {method}')
            ax.set(title=scope, xlabel='Relative rank')
        axes[0].set_ylabel('AID validation top-1 change (pp)')
        axes[-1].legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(out/'rank_accuracy_change.png', dpi=150)
        plt.close(fig)
        dense = all_models[all_models.model_id == 'dense__dense']
        half = subset[subset.rank_ratio == .5]
        position = all_models[all_models.configuration.str.fullmatch(r'(early|middle|late)_ffn_r0500|single_b(0|5|11)_mlp.fc1_r0125') & ~all_models.finetuned]
        budget_pairs = []
        for directory, arch in zip(dirs, ('ViT-Small', 'ViT-Base')):
            budgets = pd.read_csv(directory/'budget_comparison.csv')
            for (method, ratio), group in budgets.groupby(['method','rank_ratio']):
                if len(group) != 2 or group.total_params.nunique() != 1 or group.allocated.nunique() != 2:
                    raise ValueError('Uniform and allocated must have exactly equal budgets')
                g = group.set_index('allocated')
                budget_pairs.append(dict(architecture=arch, method=method, budget_ratio=ratio,
                    parameters=g.at[True,'total_params'],
                    local_MSE_delta=g.at[True,'calibration_mse_sum']-g.at[False,'calibration_mse_sum'],
                    validation_delta_pp=100*(g.at[True,'top1']-g.at[False,'top1'])))
        budgets = pd.DataFrame(budget_pairs)
        budgets.to_csv(out/'allocated_vs_uniform.csv', index=False)
        ft = all_models[all_models.finetuned & all_models.configuration.isin(
            ['dense_control', 'global_all_blocks_r0500', 'global_ffn_r0500'])].copy()
        ft['gain_pp'] = 100*(ft.validation_top1-ft.accuracy_before)
        observations = []
        for arch in ('ViT-Small', 'ViT-Base'):
            matched = half[(half.architecture == arch) & (half.scope == 'all_blocks')].set_index('method')
            ordinary = matched.loc['exact_svd']
            activation = matched.loc['activation_svd']
            local = budgets[budgets.architecture == arch]
            joint = (local.local_MSE_delta < 0) & (local.validation_delta_pp < 0)
            observations.append(
                f'{arch}, все блоки при ρ=0,5. Ordinary SVD даёт {100*ordinary.validation_top1:.3f}% '
                f'на validation ({ordinary.validation_delta_pp:+.3f} п.п. от dense), '
                f'activation-SVD — {100*activation.validation_top1:.3f}% '
                f'({activation.validation_delta_pp:+.3f} п.п.). Факторизованные модели сохраняют '
                f'{100*ordinary.relative_parameters:.2f}% параметров dense. В {int(joint.sum())} из '
                f'{len(local)} бюджетных сравнений переход uniform → allocated одновременно '
                f'снижает локальную MSE и validation top-1.')
        report = ['# ViT-Small и ViT-Base на AID', '## Исходные модели',
            table(dense, ['architecture','total_params','validation_top1','test_top1','median_batch_ms_b1','median_batch_ms_b16']),
            'Один checkpoint на архитектуру. Фиксированный seed-42 split: 7000 train, 1500 validation и 1500 test. До аппроксимации Base обучался 10 эпох с LR 1e-4 и batch 16×2; Small — 20 эпох с LR 3e-4 и batch 64×1. Различия включают влияние исходных весов и обучения.',
            *observations,
            '## Глобальные замены при относительном ранге 0,5',
            table(half, ['architecture','scope','method','relative_parameters','validation_delta_pp','latency_over_dense_b1','latency_over_dense_b16']),
            'При этом относительном ранге фактический ранг Small равен 192, Base — 384. В таблице доля параметров учитывает оба фактора и все сохранённые параметры модели. Отношение latency меньше единицы соответствует измеренному уменьшению медианного времени на том же batch.',
            '![Изменение validation accuracy по архитектурам и методам](rank_accuracy_change.png)',
            '## Области и одиночные замены',
            table(position, ['architecture','configuration','method','validation_delta_pp']),
            '## Распределение рангов при точном бюджете',
            table(budgets, ['architecture','method','budget_ratio','parameters','local_MSE_delta','validation_delta_pp']),
            'Здесь обе дельты вычислены как allocated минус uniform того же метода. Отрицательные значения в обоих столбцах показывают снижение локальной ошибки одновременно с потерей accuracy.',
            '## Дообучение и dense-контроль',
            table(ft, ['architecture','configuration','method','accuracy_before','validation_top1','gain_pp','ce_before','validation_cross_entropy','test_top1']),
            '## Интерпретация',
            'Все дельты accuracy выражены в процентных пунктах относительно dense той же архитектуры на той же выборке. Compression ratio равен dense params / model params. Распределённые ранги оптимизированы отдельно на каждой архитектуре. Для allocated столбец rank_ratio обозначает uniform-бюджет, а реализованные ранги приведены по матрицам. Файл CSV содержит точные ранги и оба измеренных отношения latency для каждой пары. Пропущенное значение означает отсутствие соответствующего benchmark или final test.',
            'Измерения получены на одной RTX 4070. P05/P95 описывают разброс синхронизированных повторов. Изолированный runtime-пик GPU не измерялся. Статистическая значимость различий по одному split не оценивалась.',
            '[Все парные конфигурации](architectures.csv)']
        publish_text(out/'REPORT_RU.md', '\n\n'.join(report)+'\n')


if __name__ == '__main__':
    main()
