"""Audit a replication's completed stages and configuration coverage before freezing test."""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np

from aerial_svd.lowrank_common import file_identity, publish_json, stage_output


def read(path):
    return json.loads(path.read_text())


def identities(value):
    if isinstance(value, dict):
        if {'path', 'bytes', 'sha256'} <= value.keys():
            yield value
        for child in value.values():
            yield from identities(child)
    elif isinstance(value, list):
        for child in value:
            yield from identities(child)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('svd-root', 'activation-root', 'final-root', 'reference-catalog',
                 'reference-selection', 'output-dir'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    if (args.final_root / 'test_started.json').exists():
        raise ValueError('Audit must precede the first final test')
    catalog_path = args.final_root / '08_catalog/catalog.json'
    candidate = read(catalog_path)
    reference = read(args.reference_catalog)
    models = {m['model_id']: m for m in candidate['models']}
    expected = {m['model_id']: m for m in reference['models']}
    if len(candidate['models']) != 355 or len(models) != 355 or models.keys() != expected.keys():
        raise ValueError('Model catalogue differs from the reference study')
    actual_bench = {k for k,m in models.items() if m['benchmark']}
    if len(actual_bench) != 67 or actual_bench != {k for k,m in expected.items() if m['benchmark']}:
        raise ValueError('Missing or unexpected benchmark configuration')
    for key in models:
        m, other = models[key], expected[key]
        if m['validation']['num_samples'] != 1500:
            raise ValueError(f'Incomplete validation: {key}')
        ranks, prior = (x['candidate'].get('ranks', {}) for x in (m, other))
        if ranks.keys() != prior.keys():
            raise ValueError(f'Different matrix set: {key}')
        if not m['candidate'].get('allocation') and any(ranks[k] != 2 * r for k,r in prior.items()):
            raise ValueError(f'Different relative rank: {key}')
        if m['finetuned']:
            history = read(Path(m['evidence'])/'history.json')
            if len(history) != 5 or any(e['train']['num_samples'] != 7000
                    or e['train']['skipped_updates'] or e['train']['optimizer_updates'] != 875 for e in history):
                raise ValueError(f'Incomplete fine-tuning: {key}')
    half = models['exact_svd__global_all_blocks_r0500']
    ranks = half['candidate']['ranks']
    if len(ranks) != 48:
        raise ValueError('Require all block matrices for the factor-bank control')
    with np.load(half['checkpoint']['path'], allow_pickle=False) as weights, \
            np.load(args.activation_root/'04_bank/factors.npz', allow_pickle=False) as bank:
        for name, rank in ranks.items():
            np.testing.assert_array_equal(weights['p|'+name+'.first.weight'], bank[name+'.svd.first'][:rank])
            np.testing.assert_array_equal(weights['p|'+name+'.second.weight'], bank[name+'.svd.second'][:, :rank])
    selection = read(args.reference_selection)
    from aerial_svd.lowrank_freeze import required_ids
    if required_ids() != [m['model_id'] for m in selection['models']]:
        raise ValueError('The frozen model IDs or order would differ from Small')
    if candidate['aid_split']['sha256'] != reference['aid_split']['sha256']:
        raise ValueError('Split differs')
    with np.load(args.svd_root/'05_dense/logits.npz', allow_pickle=False) as current, \
            np.load(Path(expected['dense__dense']['evidence'])/'logits.npz', allow_pickle=False) as prior:
        for key in ('pixels_sha256', 'labels', 'indices'):
            np.testing.assert_array_equal(current[key], prior[key])
    cached = {}
    references = 0
    stages = Counter()
    for root in (args.svd_root, args.activation_root, args.final_root):
        for p in sorted(root.rglob('manifest.json')):
            m = read(p)
            if m['status'] != 'complete':
                if any((q/'manifest.json').is_file() and read(q/'manifest.json')['status'] == 'complete'
                       for q in p.parent.parent.glob(p.parent.name+'.retry-*')):
                    continue
                raise ValueError(f'Unresolved incomplete stage: {p.parent}')
            stages[m['stage']] += 1
            for v in identities(m):
                path = Path(v['path']).resolve()
                if path not in cached:
                    digest = hashlib.sha256()
                    with path.open('rb') as stream:
                        for chunk in iter(lambda: stream.read(4 << 20), b''):
                            digest.update(chunk)
                    cached[path] = (path.stat().st_size, digest.hexdigest())
                if cached[path] != (v['bytes'], v['sha256']):
                    raise ValueError(f'Identity mismatch: {path}')
                references += 1
    if stages['07_benchmark'] < 134:
        raise ValueError('Final benchmark series incomplete')
    for key in actual_bench:
        for batch in (1, 16):
            m = read(args.final_root/'07_benchmark'/key/f'batch-{batch}'/'manifest.json')
            if m['checkpoint']['sha256'] != models[key]['checkpoint']['sha256']:
                raise ValueError(f'Benchmark checkpoint mismatch: {key}')
    with stage_output(args.output_dir, dict(stage='prefreeze_audit', catalog=file_identity(catalog_path))) as out:
        publish_json(out/'audit.json', dict(stage_counts=dict(stages), checked_references=references,
            unique_files=len(cached), models=len(models), benchmarks=134, prespecified_models=20,
            paired_validation_inputs=True, ordinary_bank_matrices_verified=len(ranks), test_started=False))


if __name__ == '__main__':
    main()
