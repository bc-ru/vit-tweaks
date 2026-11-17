"""Freeze the prespecified final model list using validation and completed benchmarks only."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from aerial_svd.lowrank_catalog import benchmark_result, checked_identity
from aerial_svd.lowrank_common import file_identity, publish_json, stage_output
from aerial_svd.lowrank_quality import verified_stage


def index_digest(indices):
    return hashlib.sha256(json.dumps(indices, separators=(',', ':')).encode()).hexdigest()


def required_ids():
    result = ['dense__dense']
    for method in ('exact_svd', 'activation_svd'):
        for configuration in ('global_all_blocks_r0250', 'global_all_blocks_r0375',
                              'global_all_blocks_r0500', 'global_all_blocks_r0750',
                              'global_ffn_r0500', 'global_attn_r0500'):
            result.append(f'{method}__{configuration}')
        for configuration in ('global_all_blocks_r0500', 'global_ffn_r0500'):
            result.append(f'{method}__{configuration}__finetuned')
        result.append(f'{method}__allocated_{method}_all_blocks_r0500')
    # Prespecified additional control: isolates the effect of five more training epochs.
    result.append('dense__dense_control__finetuned')
    return result


def main(argv=None):
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--catalog', required=True)
    cli.add_argument('--finalization-root', required=True)
    cli.add_argument('--output-dir', required=True)
    args = cli.parse_args(argv)
    root = Path(args.finalization_root)
    if (root / 'test_started.json').exists() or (root / '09_test').exists():
        raise ValueError('Test already started; selection cannot be changed')
    catalog_path = Path(args.catalog)
    verified_stage(catalog_path.parent)
    catalog = json.loads(catalog_path.read_text())
    by_id = {m['model_id']: m for m in catalog['models']}
    models = [by_id[k] for k in required_ids()]
    for model in models:
        checked_identity(model['checkpoint'])
    benchmarks, protocols = [], {}
    for model in catalog['models']:
        if not model['benchmark']:
            continue
        for batch in (1, 16):
            directory = root / '07_benchmark' / model['model_id'] / f'batch-{batch}'
            _, _, signature = benchmark_result(directory, model, batch)
            if batch in protocols and protocols[batch] != signature:
                raise ValueError('Benchmark hardware/runtime/inputs/protocol/source differ')
            if (signature['config'] != catalog['config']['sha256']
                    or signature['split'] != catalog['aid_split']['sha256']):
                raise ValueError('Benchmark configuration/split differ from catalog')
            protocols[batch] = signature
            benchmarks.append(file_identity(directory / 'manifest.json'))
    dense_manifest = verified_stage(by_id['dense__dense']['evidence'])
    split = json.loads(Path(checked_identity(catalog['aid_split'])['path']).read_text())
    with stage_output(args.output_dir, dict(stage='09_freeze',
            catalog=file_identity(catalog_path), benchmarks=benchmarks, protocols=protocols)) as out:
        selection = dict(version=1, frozen_utc=datetime.now(timezone.utc).isoformat(),
            catalog={**file_identity(catalog_path), 'path': str(catalog_path)},
            config=checked_identity(catalog['config']), aid_split=checked_identity(catalog['aid_split']),
            models=models, test_indices_sha256=index_digest(split['indices']['test']),
            test_count=len(split['indices']['test']), batch_size=8,
            dataset={k: dense_manifest[k] for k in ('dataset_fingerprint', 'classes', 'class_to_idx',
                                                  'image_decoder', 'preprocessing')},
            reason='User-prespecified 19 models plus dense fine-tuning control to distinguish extra training effects. '
                   'No additional validation/performance-selected models. No test image or test metric consulted.',
            policy='Immutable after first test access; no training, rank/epoch selection or further experiments after test.')
        publish_json(out / 'selection.json', selection)
    print(file_identity(Path(args.output_dir) / 'selection.json')['sha256'], flush=True)


if __name__ == '__main__':
    main()
