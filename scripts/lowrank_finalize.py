#!/usr/bin/env python3
"""Sequential launcher only: independent benchmark or frozen-test stages, resumable by hashes."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def checked(directory, checkpoint, extra, config, split):
    manifest = json.loads((directory / 'manifest.json').read_text())
    if manifest['status'] != 'complete':
        raise ValueError(f'Incomplete output; choose a new root: {directory}')
    if manifest['checkpoint']['sha256'] != checkpoint['sha256'] or sha(checkpoint['path']) != checkpoint['sha256']:
        raise ValueError('Checkpoint changed')
    for key, identity in (('config', config), ('aid_split', split)):
        if manifest[key]['sha256'] != identity['sha256'] or sha(identity['path']) != identity['sha256']:
            raise ValueError(f'Changed resume input: {key}')
    for name, digest in manifest['source_files_sha256'].items():
        if manifest['stage'] == '09_test' or name in (
                'lowrank_benchmark.py', 'lowrank_common.py', 'lowrank_quality.py', 'jax_engine.py',
                'jax_models.py', 'jax_data.py', 'jax_checkpoint.py', 'model_spec.py'):
            if sha(Path('src/aerial_svd')/name) != digest:
                raise ValueError(f'Changed execution source: {name}')
    for name, identity in manifest['outputs'].items():
        if sha(directory / name) != identity['sha256']:
            raise ValueError(f'Changed output: {directory / name}')
    for key, value in extra.items():
        if manifest['arguments'].get(key) != value:
            raise ValueError(f'Resume protocol differs: {key}')


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--root', required=True)
    cli.add_argument('--series', choices=('benchmark', 'test'), required=True)
    cli.add_argument('--catalog', required=True)
    cli.add_argument('--selection')
    cli.add_argument('--selection-sha256')
    args = cli.parse_args()
    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    catalog_path = Path(args.catalog)
    cm = json.loads((catalog_path.parent / 'manifest.json').read_text())
    if cm['status'] != 'complete' or sha(catalog_path) != cm['outputs'][catalog_path.name]['sha256']:
        raise ValueError('Catalog is not verified')
    catalog = json.loads(catalog_path.read_text())
    if args.series == 'benchmark':
        if (root / 'test_started.json').exists():
            raise ValueError('Final test has started; no further benchmark experiments')
        models = [m for m in catalog['models'] if m['benchmark']]
    else:
        if not args.selection or not args.selection_sha256 or sha(args.selection) != args.selection_sha256:
            cli.error('Test requires frozen selection and its independently recorded SHA-256')
        selection = json.loads(Path(args.selection).read_text())
        if selection['catalog']['sha256'] != sha(catalog_path):
            raise ValueError('Selection/catalog mismatch')
        models = selection['models']
    for i, model in enumerate(models, 1):
        batches = [1, 16] if args.series == 'benchmark' else [8]
        for batch in batches:
            out = root / ('07_benchmark' if args.series == 'benchmark' else '09_test') / model['model_id']
            if args.series == 'benchmark':
                out = out / f'batch-{batch}'
                extra = dict(batch_sizes=[batch], warmup=5, repeats=50)
                module = 'lowrank_benchmark'
                options = ['--batch-sizes', str(batch), '--warmup', '5', '--repeats', '50']
            else:
                extra = dict(selection_sha256=args.selection_sha256, model_id=model['model_id'], batch_size=8)
                module = 'lowrank_test'
                options = ['--selection', args.selection, '--selection-sha256', args.selection_sha256,
                           '--model-id', model['model_id'], '--batch-size', '8', '--finalization-root', str(root)]
            if out.exists():
                checked(out, model['checkpoint'], extra, catalog['config'], catalog['aid_split'])
                print(f'Reuse {i}/{len(models)} {out}', flush=True)
                continue
            out.parent.mkdir(parents=True, exist_ok=True)
            command = ['scripts/run.sh', 'gpu', 'python', '-m', 'aerial_svd.' + module,
                '--config', catalog['config']['path'], '--checkpoint', model['checkpoint']['path'],
                '--output-dir', str(out), '--device', 'gpu', *options]
            print(f'Run {i}/{len(models)} {model["model_id"]} batch={batch}', flush=True)
            with out.with_suffix('.log').open('x') as log:
                subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)


if __name__ == '__main__':
    main()
