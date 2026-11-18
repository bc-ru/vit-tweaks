"""Measured activation/SVD comparison for stage 08; never runs model inference."""
import csv
import json
import os
from pathlib import Path
import shutil

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from aerial_svd.lowrank_common import file_identity, publish_json, stage_output, write_csv
from aerial_svd.lowrank_quality import paired_metrics, verified_stage


def aggregate(args):
    bank_path = Path(args.activation_bank)
    bank = verified_stage(bank_path)
    diagnostics = json.loads((bank_path/'diagnostics.json').read_text())
    with (bank_path/'local_errors.csv').open() as stream:
        local = list(csv.DictReader(stream))
    lookup = {(r['method'], r['name'], int(r['actual_rank'])): r for r in local}
    base = Path(args.baseline_root)
    dense_dir = base/'05_dense'
    dense = verified_stage(dense_dir)
    architecture = dense['model_spec']['model']
    reference = np.load(dense_dir/'logits.npz', allow_pickle=False)
    if bank['checkpoint']['sha256'] != dense['checkpoint']['sha256']:
        raise ValueError('Bank and dense reference disagree')
    identities = [file_identity(bank_path/'manifest.json'), file_identity(dense_dir/'manifest.json')]
    split = json.loads(Path(dense['aid_split']['path']).read_text())['indices']
    for key, split_name in [('calibration', 'train'), ('validation_statistics', 'val')]:
        identity = bank[key]
        directory = Path(identity['path']).parent
        sm = verified_stage(directory)
        if file_identity(directory/'manifest.json')['sha256'] != identity['sha256']:
            raise ValueError('Statistics manifest changed')
        stats = json.loads((directory/'statistics.json').read_text())
        if (sm['evaluation_split'] != split_name or stats['split'] != split_name
                or not set(stats['indices']).issubset(split[split_name])
                or set(stats['indices']) & set(split['test'])
                or len(stats['indices']) != stats['images']):
            raise ValueError('Invalid train/validation statistics selection')
        if split_name == 'val':
            np.testing.assert_array_equal(stats['indices'], reference['indices'])
            if stats['pixels_sha256'] != reference['pixels_sha256'].item():
                raise ValueError('Validation statistics pixels differ from reference')
        identities.append(file_identity(directory/'manifest.json'))
    rows, matrices, ft = [], [], []
    seen = set()
    rank_maps = {}
    protocols = {}
    training_epochs = training_images = training_updates = 0
    for root in (base, Path(args.root)):
        for manifest_path in sorted(root.rglob('manifest.json')):
            m = json.loads(manifest_path.read_text())
            if m['status'] != 'complete' or m['stage'] not in ('05_validate', '06_finetune'):
                continue
            directory = manifest_path.parent
            c = m['source_metadata'].get('candidate')
            if not c:
                continue
            verified_stage(directory)
            if file_identity(m['checkpoint']['path'])['sha256'] != m['checkpoint']['sha256']:
                raise ValueError('Measured checkpoint changed')
            for key in ('config', 'aid_split'):
                if m[key]['sha256'] != dense[key]['sha256']:
                    raise ValueError(f'Incompatible {key}')
            for key in ('dataset_fingerprint', 'classes', 'image_decoder', 'preprocessing'):
                if m[key] != dense[key]:
                    raise ValueError(f'Incompatible {key}')
            if m['source_metadata']['dense_source']['sha256'] != dense['checkpoint']['sha256'] or m['evaluation_split'] != 'val':
                raise ValueError('Mixed source or non-validation data')
            if c['method'] == 'activation_svd' and c['activation_bank']['sha256'] != identities[0]['sha256']:
                raise ValueError('Mixed activation bank')
            identities.append(file_identity(manifest_path))
            name = directory.parent.name
            method = c['method']
            if m['stage'] == '06_finetune':
                v = json.loads((directory/'finetune.json').read_text())
                protocols[method, name] = v['protocol']
                if method == 'activation_svd':
                    history = json.loads((directory/'history.json').read_text())
                    if len(history) != v['protocol']['epochs'] or not v['ranks_preserved']:
                        raise ValueError('Incomplete activation training')
                    for epoch in history:
                        train = epoch['train']
                        if (train['num_samples'] != dense['split_counts']['train']
                                or train['skipped_updates'] != 0
                                or train['optimizer_updates'] != train['num_samples']//v['protocol']['batch_size']):
                            raise ValueError('Skipped or incomplete activation training epoch')
                        training_epochs += 1
                        training_images += train['num_samples']
                        training_updates += train['optimizer_updates']
                ft.append(dict(configuration=name, method=method, total_params=v['total_params'], best_epoch=v['best_epoch'],
                    accuracy_before=v['before']['top1'], accuracy_after=v['after']['top1'],
                    ce_before=v['before']['loss'], ce_after=v['after']['loss'],
                    logit_l2_before=v['before']['logit_l2_mean'], logit_l2_after=v['after']['logit_l2_mean'],
                    epochs=v['protocol']['epochs'], ranks_preserved=v['ranks_preserved']))
                continue
            if (method, name) in seen:
                raise ValueError(f'Duplicate configuration: {method} {name}')
            seen.add((method, name))
            rank_maps[method, name] = c['ranks']
            v = json.loads((directory/'validation.json').read_text())
            with np.load(directory/'logits.npz', allow_pickle=False) as logits:
                for key in ('indices', 'labels', 'pixels_sha256'):
                    np.testing.assert_array_equal(logits[key], reference[key])
                recomputed = paired_metrics(logits['logits'], logits['labels'], reference['logits'])
                for key, value in recomputed.items():
                    if not np.isclose(value, v[key], atol=1e-12, rtol=1e-12):
                        raise ValueError(f'Saved metrics disagree with logits: {name} {key}')
            selected = [lookup[method, n, rank] for n, rank in c['ranks'].items()]
            allocation = c.get('allocation')
            requests = allocation['uniform_reference'] if allocation else c['requests']
            rho = next(iter(requests.values()))[0]['requested_ratio']
            row = dict(configuration=name, method=method, allocated=bool(allocation), scope=c['scope'],
                rank_ratio=rho, num_matrices=len(selected), total_params=v['total_params'],
                param_memory_bytes=v['param_memory_bytes'], checkpoint_bytes=v['checkpoint_bytes'],
                top1=v['top1'], cross_entropy=v['loss'], logit_l2_mean=v['logit_l2_mean'],
                logit_l2_p95=v['logit_l2_p95'], logit_relative_frobenius=v['logit_relative_frobenius'],
                logit_max_abs=v['logit_max_abs'], full_rank_control=v['full_rank_control'],
                matrix_relative_frobenius_mean=float(np.mean([float(r['matrix_relative_frobenius']) for r in selected])),
                calibration_mse_sum=sum(float(r['calibration_mse']) for r in selected),
                validation_local_mse_sum=sum(float(r['validation_mse']) for r in selected),
                validation_images=v['num_samples'], validation_manifest=str(manifest_path))
            rows.append(row)
            for r in selected:
                matrices.append(dict(configuration=name, **r))
    reference.close()
    df = pd.DataFrame(rows)
    baseline = json.loads((dense_dir/'validation.json').read_text())
    # Ordinary and activation uniform runs must agree on ranks and parameter counts.
    pairs = []
    for name, group in df[~df.allocated].groupby('configuration'):
        if set(group.method) != {'exact_svd', 'activation_svd'}:
            raise ValueError(f'Missing paired method: {name}')
        a, s = (group[group.method == method].iloc[0] for method in ('activation_svd', 'exact_svd'))
        if a.total_params != s.total_params or a.param_memory_bytes != s.param_memory_bytes:
            raise ValueError('Paired parameter budgets differ')
        if rank_maps['activation_svd', name] != rank_maps['exact_svd', name]:
            raise ValueError('Paired selected matrices/ranks differ')
        pairs.append(dict(configuration=name, scope=a.scope, rank_ratio=a.rank_ratio, total_params=int(a.total_params),
            svd_top1=s.top1, activation_top1=a.top1, accuracy_delta_pp=100*(a.top1-s.top1),
            svd_ce=s.cross_entropy, activation_ce=a.cross_entropy,
            svd_logit_l2=s.logit_l2_mean, activation_logit_l2=a.logit_l2_mean,
            svd_local_mse=s.calibration_mse_sum, activation_local_mse=a.calibration_mse_sum))
    with stage_output(args.output_dir, dict(stage='08_activation_comparison', inputs=identities,
                     git_commit=os.environ.get('AERIAL_SVD_GIT_COMMIT'),
                     source_files_sha256={p.name: file_identity(p)['sha256']
                                          for p in Path(__file__).parent.glob('lowrank_*.py')},
                     source_checkpoint_sha256=dense['checkpoint']['sha256'], split_sha256=dense['aid_split']['sha256'],
                     evaluation_split='val', test_used=False)) as out:
        for identity, filename in [(bank['calibration'], 'calibration.json'),
                                   (bank['validation_statistics'], 'validation_statistics.json')]:
            statistics = json.loads((Path(identity['path']).parent/'statistics.json').read_text())
            # Keep per-example identities/pixel digests in local stage 03 only.
            publish_json(out/filename, {k: v for k, v in statistics.items()
                                       if k not in ('indices', 'pixels_sha256')})
        shutil.copyfile(bank_path/'diagnostics.json', out/'diagnostics.json')
        write_csv(out, 'configurations.csv', rows)
        write_csv(out, 'paired_comparison.csv', pairs)
        write_csv(out, 'matrix_errors.csv', matrices)
        write_csv(out, 'local_errors.csv', local)
        write_csv(out, 'finetuning.csv', ft)
        plt.rcParams.update({'figure.dpi': 140, 'axes.grid': True, 'grid.alpha': .25})
        def save(name):
            plt.suptitle(architecture)
            plt.tight_layout(); plt.savefig(out/name); plt.close()
        global_df = df[df.configuration.str.startswith('global_') & ~df.allocated]
        for x, filename in [('rank_ratio', 'accuracy_rank.png'), ('total_params', 'accuracy_parameters.png')]:
            fig, axes = plt.subplots(1, 3, figsize=(13, 4))
            for ax, scope in zip(axes, ('attn', 'ffn', 'all_blocks')):
                for method, g in global_df[global_df.scope == scope].groupby('method'):
                    g = g.sort_values('rank_ratio')
                    ax.plot(g[x], 100*g.top1, 'o-', label=method)
                ax.axhline(100*baseline['top1'], color='black', linestyle='--')
                ax.set(title=scope, xlabel=x, ylabel='Validation top-1 (%)'); ax.legend()
            save(filename)
        fig, axes = plt.subplots(2, 2, figsize=(11, 8))
        curves_df = pd.DataFrame(local)
        curves_df['relative_rank'] = curves_df.relative_rank.astype(float)
        curves_df['calibration_relative_rmse'] = curves_df.calibration_relative_rmse.astype(float)
        curves_df['kind'] = curves_df.name.str.split('.').str[2:].str.join('.')
        for ax, kind in zip(axes.flat, ('attn.qkv', 'attn.proj', 'mlp.fc1', 'mlp.fc2')):
            for method, g in curves_df[curves_df.kind == kind].groupby('method'):
                mean = g.groupby('relative_rank').calibration_relative_rmse.mean()
                ax.plot(mean.index, mean.values, 'o-', label=method)
            ax.set(title=kind, xlabel='Relative rank', ylabel='Calibration relative RMSE'); ax.legend()
        save('local_error_rank.png')
        fig, axes = plt.subplots(2, 2, figsize=(11, 8))
        for ax, kind in zip(axes.flat, ('attn.qkv', 'attn.proj', 'mlp.fc1', 'mlp.fc2')):
            for method, linestyle in [('exact_svd', '--'), ('activation_svd', '-')]:
                for block in (0, 5, 11):
                    g = df[(df.method == method) & df.configuration.str.startswith(f'single_b{block}_{kind}_')].sort_values('rank_ratio')
                    ax.plot(g.rank_ratio, 100*g.top1, linestyle, label=f'{method} b{block}')
            ax.set(title=kind, xlabel='Relative rank', ylabel='Validation top-1 (%)'); ax.legend(fontsize=6)
        save('single_positions.png')
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        for ax, scope in zip(axes, ('attn', 'ffn')):
            for region in ('early', 'middle', 'late'):
                for method, style in [('exact_svd', '--'), ('activation_svd', '-')]:
                    g = df[(df.method == method) & (df.scope == scope) & df.configuration.str.startswith(region+'_')].sort_values('rank_ratio')
                    ax.plot(g.rank_ratio, 100*g.top1, style, label=f'{method} {region}')
            ax.set(title=scope, xlabel='Relative rank', ylabel='Validation top-1 (%)'); ax.legend(fontsize=7)
        save('region_positions.png')
        budget_rows = []
        for rho in sorted(df[df.allocated].rank_ratio.unique()):
            controls = df[(df.scope == 'all_blocks') & (df.rank_ratio == rho) &
                          (df.allocated | df.configuration.str.startswith('global_'))]
            if len(controls) != 4 or len(controls.total_params.unique()) != 1:
                raise ValueError('Require four exactly budget-matched controls')
            budget_rows.extend(controls.to_dict('records'))
        write_csv(out, 'budget_comparison.csv', budget_rows)
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        budget_df = pd.DataFrame(budget_rows)
        if len(budget_df):
            for (method, allocated), g in budget_df.groupby(['method', 'allocated']):
                g = g.sort_values('total_params')
                label = f'{method} '+('allocated' if allocated else 'uniform')
                axes[0].plot(g.total_params, 100*g.top1, 'o-', label=label)
                axes[1].plot(g.total_params, g.calibration_mse_sum, 'o-', label=label)
            axes[0].set(ylabel='Validation top-1 (%)'); axes[1].set(ylabel='Sum local calibration MSE')
            for ax in axes:
                ax.set_xlabel('Actual model parameters'); ax.legend(fontsize=7)
            save('rank_allocation.png')
        selected_ft = pd.DataFrame(ft)
        matched_ft = selected_ft[selected_ft.configuration.isin(selected_ft[selected_ft.method == 'activation_svd'].configuration)]
        for name in selected_ft[selected_ft.method == 'activation_svd'].configuration:
            if protocols['activation_svd', name] != protocols.get(('exact_svd', name)):
                raise ValueError('Activation/SVD training protocols differ')
            if matched_ft[matched_ft.configuration == name].total_params.nunique() != 1:
                raise ValueError('Activation/SVD fine-tuning parameter counts differ')
        if len(matched_ft):
            fig, ax = plt.subplots(figsize=(9, 4))
            labels = []
            for i, r in enumerate(matched_ft.itertuples()):
                ax.plot([i-.15, i+.15], [100*r.accuracy_before, 100*r.accuracy_after], 'o-')
                labels.append(r.configuration+'\n'+r.method)
            ax.set_xticks(range(len(labels)), labels, rotation=15, ha='right')
            ax.set_ylabel('Validation top-1 (%), before → after'); save('finetuning.png')
        full = df[(df.method == 'activation_svd') & df.full_rank_control]
        publish_json(out/'summary.json', dict(paired_configurations=len(pairs),
            new_validation_configurations=int(sum((df.method == 'activation_svd') | df.allocated)),
            activation_full_rank_controls=len(full), full_rank_max_relative_logit=float(full.logit_relative_frobenius.max()),
            full_rank_max_absolute_logit=float(full.logit_max_abs.max()), baseline=baseline,
            activation_finetunes=int(sum(selected_ft.method == 'activation_svd')),
            activation_training_epochs=training_epochs, activation_training_images=training_images,
            activation_training_updates=training_updates, test_used=False))
        paired = pd.DataFrame(pairs)
        nonfull = paired[paired.rank_ratio < 1]
        arows = {(r['name'], r['actual_rank']): r for r in local if r['method'] == 'activation_svd' and float(r['relative_rank']) < 1}
        srows = {(r['name'], r['actual_rank']): r for r in local if r['method'] == 'exact_svd'}
        wins = {key: sum(float(r[key]) < float(srows[k][key]) for k, r in arows.items())
                for key in ('calibration_mse', 'validation_mse', 'matrix_relative_frobenius')}
        report = [
            f'# {architecture} на AID: учёт активаций и индивидуальные ранги\n',
            '## Предмет исследования и гипотеза\n',
            'Проверяем, уменьшает ли локальный критерий на train-активациях ошибку преобразований и улучшает ли качество всей модели при тех же рангах. Отдельно проверяем распределение рангов при точно равном хранении. Локальный оптимум не гарантирует оптимума нелинейной модели.\n',
            '## Исходные данные\n',
            f'Один исходный checkpoint {architecture}, SHA-256 `{dense["checkpoint"]["sha256"]}`; AID, 30 классов, seed 42, неизменный split 7000/1500/1500. Исходная validation accuracy {100*baseline["top1"]:.3f}%, CE {baseline["loss"]:.6f}. Калибровка: 300 train-изображений (10 каждого класса), выбор NumPy default_rng(42), фиксированный resize/normalization без аугментаций. Все 197 токенов изображения имеют одинаковый вес; 59100 строк для каждой матрицы. Независимые локальные ошибки и качество модели — на тех же 1500 validation-изображениях. Test не читался.\n',
            '## Математическая постановка\n',
            r'Для $Y=XW$, $C=X^\top X/N$ (нецентрированный момент). Обычный SVD минимизирует $\|W-\widetilde W\|_F^2$; новый критерий — $\operatorname{tr}((W-\widetilde W)^\top C(W-\widetilde W))$. Используем $C_\lambda=C+\lambda I$, $\lambda=10^{-4}\operatorname{tr}(C)/n$: к MSE на токен добавляется $\lambda\|W-\widetilde W\|_F^2$. Это регуляризованный локальный критерий, не точный нерегуляризованный минимум.'+'\n',
            r'В хранимой ориентации $(out,in)$: $C_\lambda=LL^\top$, $B=W_sL=U\Sigma V^\top$, $F_2=U_r\sqrt{\Sigma_r}$, $F_1=\sqrt{\Sigma_r}V_r^\top L^{-1}$. Инференс вычисляет $(XF_1^\top)F_2^\top+b$. Bias сохранён; расширяющие и полные ранги не пропускаются. Обратная матрица явно не строится: используется треугольное решение в float64, факторы хранятся в float32.'+'\n',
            f'Моменты суммируются потоково: float32 Gram по batch с highest precision, float64 сумма на CPU; все активации не сохраняются. Численная PSD-проекция разрешена только для погрешности округления и протоколируется. PSD-проекция потребовалась для {sum(d["psd_rounding_clipped"] for d in diagnostics.values())} матриц; λ от {min(d["ridge_absolute"] for d in diagnostics.values()):.3g} до {max(d["ridge_absolute"] for d in diagnostics.values()):.3g}, максимум обусловленности регуляризованного момента {max(d["condition_regularized"] for d in diagnostics.values()):.3g}. Значения для каждой матрицы сохранены в diagnostics.json.\n',
            '## Варьируемые параметры и метрики\n',
            'Ранги: 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0; фактический целый ранг = floor(ρ min(out,in)). 24 глобальные конфигурации attn/ffn/all_blocks, 48 региональных (блоки 0–3/4–7/8–11), 96 одиночных (блоки 0/5/11 × qkv/proj/fc1/fc2). Голова исключена. Все 168 пар имеют одинаковые веса, матрицы, ранги, параметры, validation-пиксели и протокол. Обычные SVD-оценки взяты из парной серии.\n',
            'Метрики: абсолютная/относительная Frobenius-ошибка весов, MSE преобразования на токен и относительный RMSE отдельно на train-calibration/validation, средняя/P95 L2-ошибка логитов, относительная Frobenius/максимальная абсолютная ошибка логитов, top-1, несглаженная CE, параметры, байты массивов и NPZ. Локальная ошибка не включает bias и измеряется на входах исходной модели; после одновременных замен входы следующих блоков меняются.\n',
            '## Результаты\n',
            f'Вычислены {len(pairs)} попарных конфигураций, 768 локальных точек, 6 моделей с распределённым бюджетом и {sum(selected_ft.method == "activation_svd")} дообучения. Неполные ранги: новый метод уменьшил calibration MSE в {wins["calibration_mse"]}/{len(arows)} случаях, независимую validation MSE в {wins["validation_mse"]}/{len(arows)}; Frobenius-ошибку весов — в {wins["matrix_relative_frobenius"]}/{len(arows)}.\n',
            f'По top-1 в {len(nonfull)} неполноранговых парах: лучше SVD — {sum(nonfull.accuracy_delta_pp > 0)}, одинаково — {sum(nonfull.accuracy_delta_pp == 0)}, хуже — {sum(nonfull.accuracy_delta_pp < 0)}. Это описательная статистика одной validation, не проверка статистической значимости.\n',
            f'Полный ранг: {len(full)} контролей, максимальная относительная ошибка логитов {full.logit_relative_frobenius.max():.3g}, абсолютная {full.logit_max_abs.max():.3g}; допуски 5e-5/1e-3 пройдены.\n',
            '|Набор|ρ|Параметры|SVD top-1, %|Act top-1, %|SVD CE|Act CE|\n|---|---:|---:|---:|---:|---:|---:|']
        for r in paired[paired.configuration.str.startswith('global_')].sort_values(['scope', 'rank_ratio']).itertuples():
            report.append(f'|{r.scope}|{r.rank_ratio:g}|{r.total_params}|{100*r.svd_top1:.3f}|{100*r.activation_top1:.3f}|{r.svd_ce:.4f}|{r.activation_ce:.4f}|')
        report.extend(['\n### Индивидуальные ранги\n',
            'Точное multiple-choice knapsack (динамическое программирование): одна из восьми ранговых точек для каждой матрицы; минимизируем сумму **ненормированных** calibration MSE вычисленных факторов при точном равенстве суммарного r(out+in). Bias и прочие параметры постоянны. Для каждого бюджета сравниваются uniform/allocated × ordinary/activation SVD. При равных значениях используется детерминированный порядок. Ошибки разных блоков аддитивны только в этом суррогатном критерии; их сумма не равна ошибке сети. Ранговые карты лежат в matrix_errors.csv и исходных allocation.json.\n',
            '|Бюджет uniform ρ|Метод|Ранги|Параметры|Сумма local MSE|Top-1, %|CE|\n|---|---|---|---:|---:|---:|---:|'])
        for r in budget_df.sort_values(['rank_ratio', 'method', 'allocated']).itertuples():
            report.append(f'|{r.rank_ratio:g}|{r.method}|{"allocated" if r.allocated else "uniform"}|{r.total_params}|{r.calibration_mse_sum:.5g}|{100*r.top1:.3f}|{r.cross_entropy:.4f}|')
        if len(budget_df):
            changes = []
            for (rho, method), g in budget_df.groupby(['rank_ratio', 'method']):
                u, a = (g[g.allocated == allocated].iloc[0] for allocated in (False, True))
                if a.calibration_mse_sum > u.calibration_mse_sum+1e-8:
                    raise AssertionError('Exact allocation worsened its own objective')
                changes.append(100*(a.top1-u.top1))
            report.append(f'\nРаспределение уменьшает/сохраняет свой локальный критерий во всех 6 сравнениях; изменение accuracy от {min(changes):+.3f} до {max(changes):+.3f} п.п. по сравнению с равномерным рангом того же метода и бюджета.\n')
        report.extend(['\n### Дообучение\n',
            'Выбраны global_ffn_r0500 и global_all_blocks_r0500: умеренное сжатие FFN и более сильное всех блоков, имеющие прямые SVD-контроли. Протокол прежний: все параметры, включая факторы; 5 эпох, AdamW .9/.999, eps 1e-8, LR 1e-5 с cosine по эпохам, weight decay .05, label smoothing .1 только train; batch 8, float32, seed 42. Выбор среди эпох 1–5 по максимальной validation accuracy, ранняя эпоха при равенстве; исходная точка сохраняется отдельно, ранги фиксированы. Обычные SVD-контроли и dense control из прежней серии не пересчитывались.\n',
            '|Конфигурация|Метод|Top-1 до, %|После, %|CE до|После|Эпоха|\n|---|---|---:|---:|---:|---:|---:|'])
        for r in matched_ft.sort_values(['configuration', 'method']).itertuples():
            report.append(f'|{r.configuration}|{r.method}|{100*r.accuracy_before:.3f}|{100*r.accuracy_after:.3f}|{r.ce_before:.4f}|{r.ce_after:.4f}|{r.best_epoch}|')
        report.extend(['\n## Ограничения\n',
            'Один checkpoint, seed и split; calibration из 300 изображений не покрывает всё train-распределение. λ фиксирован до запусков, чувствительность к нему не изучалась. Подбор/выбор по одной validation может завышать качество выбранных точек. Равные ρ между разными наборами не дают равных бюджетов; в бюджетной таблице равенство проверено точно. Меньшая локальная ошибка и сумма локальных ошибок не гарантируют меньшей ошибки логитов/CE или большей accuracy. Малые различия в несколько изображений не доказывают преимущества. Здесь нет измерений ускорения или пика GPU; test по заданию остаётся нетронутым. Матричные/локальные ошибки относятся к исходной факторизации до дообучения; после дообучения измеряется полная модель. Масштабы факторов после обратного преобразования отличаются от обычного SVD; одинаковый оптимизатор не устраняет влияние этой параметризации на дообучение.\n',
            'Таблицы: [пары](paired_comparison.csv), [все модели](configurations.csv), [матрицы и ранги](matrix_errors.csv), [локальные кривые](local_errors.csv), [бюджеты](budget_comparison.csv), [дообучение](finetuning.csv).\n',
            'Графики: [accuracy–ранг](accuracy_rank.png), [accuracy–параметры](accuracy_parameters.png), [локальные ошибки](local_error_rank.png), [одиночные позиции](single_positions.png), [области](region_positions.png), [распределение рангов](rank_allocation.png), [дообучение](finetuning.png).\n',
            f'Полные данные: `{args.root}`; manifest отчёта содержит хэши использованных стадий. Команды и форматы: [для агентов](../../ACTIVATION_STAGES.md).'])
        from aerial_svd.lowrank_common import publish_text
        publish_text(out/'REPORT_RU.md', '\n'.join(report)+'\n')
