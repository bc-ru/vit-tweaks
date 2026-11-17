#!/usr/bin/env python3
"""Small preset enumerator. Each operation remains an independently runnable stage."""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import subprocess

RATIOS = [.125, .25, .375, .5, .625, .75, .875, 1.]
KINDS = {'ffn': ['mlp.fc1', 'mlp.fc2'], 'attn': ['attn.qkv', 'attn.proj']}


def configurations(series):
    if series == 'global':
        for scope in ('ffn', 'attn', 'all_blocks'):
            for rho in RATIOS:
                yield f'global_{scope}_r{int(rho*1000):04d}', scope, rho, None
    elif series == 'regions':
        for region, blocks in [('early', range(4)), ('middle', range(4, 8)), ('late', range(8, 12))]:
            for scope in ('ffn', 'attn'):
                for rho in RATIOS:
                    yield f'{region}_{scope}_r{int(rho*1000):04d}', scope, rho, [f'blocks.{b}.{k}' for b in blocks for k in KINDS[scope]]
    elif series == 'single':
        for block in (0, 5, 11):
            for scope, kinds in KINDS.items():
                for kind in kinds:
                    for rho in RATIOS:
                        yield f'single_b{block}_{kind}_r{int(rho*1000):04d}', scope, rho, [f'blocks.{block}.{kind}']


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def checked(directory, checkpoint, config, extra):
    manifest = json.loads((directory / 'manifest.json').read_text())
    if manifest['status'] != 'complete':
        return False
    for key, path in [('checkpoint', checkpoint), ('config', config)]:
        if manifest[key]['sha256'] != sha(path):
            raise ValueError(f'Changed {key} for {directory}')
    split = manifest['effective_config']['split_file']
    if manifest['aid_split']['sha256'] != sha(split):
        raise ValueError('Frozen split changed')
    for name, identity in manifest['outputs'].items():
        if sha(directory / name) != identity['sha256']:
            raise ValueError(f'Changed output: {directory / name}')
    for option, key in [('activation_bank', 'activation_bank'), ('allocation_dir', 'allocation'),
                        ('reference_dir', 'reference')]:
        dependency = manifest['arguments'].get(option)
        if dependency:
            dep = Path(dependency)
            if sha(dep / 'manifest.json') != manifest[key]['sha256']:
                raise ValueError(f'Changed prerequisite: {dep}')
            dm = json.loads((dep / 'manifest.json').read_text())
            if dm['status'] != 'complete':
                raise ValueError(f'Incomplete prerequisite: {dep}')
            for name, identity in dm['outputs'].items():
                if sha(dep / name) != identity['sha256']:
                    raise ValueError(f'Changed prerequisite output: {dep / name}')
    # Check explicit requested options too, not just the existence of a result.
    i = 0
    while i < len(extra):
        key = extra[i][2:].replace('-', '_')
        i += 1
        values = []
        while i < len(extra) and not extra[i].startswith('--'):
            values.append(extra[i]); i += 1
        actual = manifest['arguments'].get(key)
        actual = actual if isinstance(actual, list) else [actual]
        if [str(v) for v in actual] != values:
            raise ValueError(f'Changed option {key} for {directory}')
    return True


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--root', required=True)
    cli.add_argument('--series', required=True, choices=['setup', 'global', 'regions', 'single', 'allocation', 'finetune', 'benchmark'])
    cli.add_argument('--config', default='configs/aid_vit_small.yaml')
    cli.add_argument('--checkpoint', default='artifacts/checkpoints/jax/vit_small_protocol_v1.npz')
    cli.add_argument('--activation-bank')
    cli.add_argument('--reference-dir', help='Reuse a completed dense validation stage')
    cli.add_argument('--finetune-configurations', nargs='+', help='Explicit selected configurations; default keeps SVD protocol')
    args = cli.parse_args()
    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    lock = (root / '.series.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)

    def run(module, relative, checkpoint=args.checkpoint, extra=(), profile='gpu'):
        base = root / relative
        attempt = 0
        while True:
            out = base if attempt == 0 else base.with_name(base.name + f'.retry-{attempt:03d}')
            if not out.exists() and not out.with_suffix(out.suffix + '.log').exists():
                break
            if (out / 'manifest.json').exists() and checked(out, checkpoint, args.config, extra):
                print(f'SKIP verified {out}', flush=True)
                return out
            attempt += 1
        out.parent.mkdir(parents=True, exist_ok=True)
        command = ['scripts/run.sh', profile, 'python', '-m', 'aerial_svd.' + module,
                   '--config', args.config, '--checkpoint', str(checkpoint), '--output-dir', str(out),
                   '--device', 'gpu' if profile == 'gpu' else 'cpu', *extra]
        print('RUN ' + ' '.join(command), flush=True)
        with out.with_suffix(out.suffix + '.log').open('x') as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            print(out.with_suffix(out.suffix + '.log').read_text()[-6000:], flush=True)
            raise RuntimeError(f'Stage failed ({result.returncode}): {out}; retry uses a new directory')
        if not checked(out, checkpoint, args.config, extra):
            raise RuntimeError('Stage did not complete')
        return out

    def completed(relative):
        base = root / relative
        for out in [base, *sorted(base.parent.glob(base.name + '.retry-*'))]:
            if (out / 'manifest.json').exists():
                manifest = json.loads((out / 'manifest.json').read_text())
                if manifest['status'] == 'complete':
                    for name, identity in manifest['outputs'].items():
                        if sha(out / name) != identity['sha256']:
                            raise ValueError(f'Changed prerequisite: {out / name}')
                    return out
        raise ValueError(f'Complete prerequisite required: {base}')

    if args.series == 'setup':
        run('lowrank_baseline', '00_baseline', extra=['--batch-size', '8', '--expected-top1', str(.9733333333333334), '--top1-atol', '0.005'])
        run('lowrank_factor_checks', '01_factor_checks', profile='cpu',
            extra=['--scope', 'all_blocks', '--rank-ratios', '0.5', '1.0'])
        run('lowrank_matrix_curves', '02_matrix_curves', profile='cpu',
            extra=['--scope', 'all_blocks', '--rank-ratios', *map(str, RATIOS)])
        run('lowrank_validate', '05_dense', extra=['--batch-size', '8'])
    elif args.series in ('global', 'regions', 'single'):
        reference = Path(args.reference_dir) if args.reference_dir else completed('05_dense')
        for name, scope, rho, matrices in configurations(args.series):
            extra = ['--scope', scope, '--rank-ratios', str(rho)]
            if matrices:
                extra += ['--matrices', *matrices]
            if args.activation_bank:
                extra += ['--activation-bank', args.activation_bank]
            candidate = run('lowrank_make_candidate', name + '/04_candidate', extra=extra, profile='cpu')
            run('lowrank_validate', name + '/05_validation', checkpoint=candidate / 'candidate.npz',
                extra=['--reference-dir', str(reference), '--batch-size', '8'])
    elif args.series == 'allocation':
        if not args.activation_bank or not args.reference_dir:
            cli.error('Allocation requires --activation-bank and --reference-dir')
        for method in ('activation_svd', 'exact_svd'):
            for rho in (.25, .5, .75):
                name = f'allocated_{method}_all_blocks_r{int(rho*1000):04d}'
                extra = ['--scope', 'all_blocks', '--rank-ratios', str(rho), '--activation-bank', args.activation_bank]
                allocation = run('lowrank_allocate', name + '/04_allocation', profile='cpu', extra=extra + ['--method', method])
                candidate = run('lowrank_make_candidate', name + '/04_candidate', profile='cpu',
                                extra=extra + ['--allocation-dir', str(allocation)])
                run('lowrank_validate', name + '/05_validation', checkpoint=candidate / 'candidate.npz',
                    extra=['--reference-dir', args.reference_dir, '--batch-size', '8'])
    elif args.series == 'finetune':
        reference = Path(args.reference_dir) if args.reference_dir else completed('05_dense')
        names = [f'global_{scope}_r{rank:04d}' for scope in ('ffn', 'attn', 'all_blocks') for rank in (250, 500)]
        names += [f'{region}_ffn_r0500' for region in ('early', 'middle', 'late')]
        names += ['dense_control']
        if args.finetune_configurations:
            names = args.finetune_configurations
        for name in names:
            checkpoint = args.checkpoint if name == 'dense_control' else completed(name + '/04_candidate') / 'candidate.npz'
            extra = ['--reference-dir', str(reference), '--epochs', '5', '--batch-size', '8', '--eval-batch-size', '8', '--lr', '1e-05']
            # Recover the latest completed epoch from interrupted attempts, retaining
            # their immutable files. A successful attempt is always reused first.
            base = root / name / '06_finetune'
            attempts = [base, *sorted(base.parent.glob(base.name + '.retry-*'))]
            complete = [p for p in attempts if (p / 'manifest.json').exists() and json.loads((p / 'manifest.json').read_text())['status'] == 'complete']
            if complete:
                existing = completed(name + '/06_finetune')
                if not checked(existing, checkpoint, args.config, extra):
                    raise ValueError(f'Incomplete fine-tuning result: {existing}')
                print(f'SKIP verified {existing}', flush=True)
                continue
            epochs = [p for a in attempts for p in a.glob('epoch-*.npz')]
            if epochs:
                resume = max(epochs, key=lambda p: p.name)
                extra += ['--resume', str(resume)]
            run('lowrank_finetune', name + '/06_finetune', checkpoint=checkpoint, extra=extra)
    else:
        names = ['dense_control', 'global_ffn_r0250', 'global_ffn_r0500', 'global_attn_r0250', 'global_attn_r0500',
                 'global_all_blocks_r0250', 'global_all_blocks_r0500', 'global_all_blocks_r1000']
        for name in names:
            checkpoint = args.checkpoint if name == 'dense_control' else completed(name + '/04_candidate') / 'candidate.npz'
            run('lowrank_benchmark', name + '/07_benchmark', checkpoint=checkpoint)


if __name__ == '__main__':
    main()
