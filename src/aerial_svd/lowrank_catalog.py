"""Audit existing validation/training evidence; enumerate the requested final benchmarks."""
import argparse
import json
import math
from pathlib import Path

from aerial_svd.lowrank_common import file_identity, publish_json, stage_output
from aerial_svd.lowrank_quality import verified_stage


def portable(path):
    path = Path(path)
    if path.is_absolute():
        for base in (Path('/workspace'), Path.cwd()):
            if path.is_relative_to(base):
                return str(path.relative_to(base))
    return str(path)


def checked_identity(identity):
    path = portable(identity['path'])
    actual = file_identity(path)
    if (actual['sha256'], actual['bytes']) != (identity['sha256'], identity['bytes']):
        raise ValueError(f'Changed input: {path}')
    return {**actual, 'path': path}


def benchmark_result(directory, model, batch):
    """One identity/protocol audit shared by pre-test freezing and final aggregation."""
    manifest = verified_stage(directory)
    result = json.loads((Path(directory)/'benchmark.json').read_text())
    measure, = result['measurements']
    if (manifest['stage'] != '07_benchmark'
            or manifest['checkpoint']['sha256'] != model['checkpoint']['sha256']
            or manifest['arguments']['batch_sizes'] != [batch] or measure['batch_size'] != batch
            or manifest['arguments']['warmup'] != 5 or manifest['arguments']['repeats'] != 50
            or manifest['environment']['platform'] != 'gpu'
            or len(measure['trials_seconds']) != 50
            or any(not math.isfinite(t) or t <= 0 for t in measure['trials_seconds'])):
        raise ValueError('Benchmark identity/protocol mismatch')
    signature = dict(environment=manifest['environment'], preprocessing=manifest['preprocessing'],
        dataset=manifest['dataset_fingerprint'], config=manifest['config']['sha256'],
        split=manifest['aid_split']['sha256'], pixels=measure['pixels_sha256'],
        warmup=manifest['arguments']['warmup'], repeats=manifest['arguments']['repeats'],
        source_files={k: v for k, v in manifest['source_files_sha256'].items() if k in
            ('lowrank_benchmark.py', 'lowrank_common.py', 'lowrank_quality.py', 'jax_engine.py',
             'jax_models.py', 'jax_data.py', 'jax_checkpoint.py', 'model_spec.py')})
    return result, measure, signature


def main(argv=None):
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--svd-root', required=True)
    cli.add_argument('--activation-root', required=True)
    cli.add_argument('--output-dir', required=True)
    args = cli.parse_args(argv)
    models, inputs = [], []
    shared = None
    for root in (Path(args.svd_root), Path(args.activation_root)):
        for metric in sorted(root.rglob('validation.json')) + sorted(root.rglob('finetune.json')):
            manifest = verified_stage(metric.parent)
            if manifest['stage'] not in ('05_validate', '06_finetune'):
                continue
            current = tuple(manifest[k]['sha256'] for k in ('config', 'aid_split'))
            if shared is not None and shared != current:
                raise ValueError('Configuration/split differ across the study')
            shared = current
            for key in ('config', 'aid_split'):
                checked_identity(manifest[key])
            inputs.append({**file_identity(metric.parent / 'manifest.json'),
                           'path': str(metric.parent / 'manifest.json')})
            value = json.loads(metric.read_text())
            ft = manifest['stage'] == '06_finetune'
            candidate = manifest['source_metadata'].get('candidate') or {}
            method = candidate.get('method', 'dense')
            configuration = metric.parent.parent.name if candidate or ft else 'dense'
            identity = manifest['outputs']['best.npz'] if ft else manifest['checkpoint']
            identity = checked_identity(identity)
            model_id = f'{method}__{configuration}' + ('__finetuned' if ft else '')
            models.append(dict(model_id=model_id, configuration=configuration, method=method,
                finetuned=ft, checkpoint=identity, candidate=candidate,
                validation=value['after'] if ft else value,
                finetuning=value if ft else None,
                evidence=str(metric.parent), benchmark=(ft or configuration == 'dense'
                    or configuration.startswith(('global_', 'allocated_')))))
    if len({m['model_id'] for m in models}) != len(models):
        raise ValueError('Duplicate model IDs; inspect retries explicitly')
    with stage_output(args.output_dir, dict(stage='08_catalog', inputs=inputs)) as out:
        publish_json(out / 'catalog.json', dict(version=1, models=models,
            config=checked_identity(manifest['config']), aid_split=checked_identity(manifest['aid_split']),
            note='Existing validation/training only; no test image access.'))
    print(f'Audited {len(models)} models; {sum(m["benchmark"] for m in models)} benchmark models', flush=True)


if __name__ == '__main__':
    main()
