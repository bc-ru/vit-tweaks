"""04_make_candidate: one ordinary/activation SVD model at every valid rank."""
import jax
import numpy as np
import json
from pathlib import Path

from aerial_svd.jax_checkpoint import load_checkpoint, save_checkpoint
from aerial_svd.jax_compression import compress
from aerial_svd.lowrank_quality import compatible_stage
from aerial_svd.lowrank_common import (file_identity, load_inputs, parser, publish_json, rank_requests,
                                       selected_names, stage_output)


def main(argv=None):
    cli = parser('04_make_candidate', matrices=True)
    cli.add_argument('--activation-bank', help='Completed 04_activation_curves factor bank')
    cli.add_argument('--allocation-dir', help='Completed 04_allocate; rank option denotes its uniform reference')
    args = cli.parse_args(argv)
    cfg, spec, params, buffers, metadata, provenance = load_inputs(args, '04_make_candidate')
    names = selected_names(params, args)
    requests = {name: rank_requests(params[name + '.weight'].shape, args) for name in names}
    if any(len(rows) != 1 for rows in requests.values()):
        cli.error('One candidate requires exactly one rank/ratio')
    ranks = {name: rows[0]['actual_rank'] for name, rows in requests.items()}
    factors = None
    method = 'exact_svd'
    bank_identity = None
    allocation = None
    if args.allocation_dir:
        am = compatible_stage(args.allocation_dir, provenance, '04_allocate')
        allocation = json.loads((Path(args.allocation_dir)/'allocation.json').read_text())
        if set(allocation['ranks']) != set(names) or allocation['uniform_reference'] != requests:
            raise ValueError('Allocation selection/reference differs from candidate')
        if not args.activation_bank or am['activation_bank']['sha256'] != file_identity(Path(args.activation_bank)/'manifest.json')['sha256']:
            raise ValueError('Allocation bank differs')
        ranks = allocation['ranks']
        requests = {n: [dict(requested_rank=r, requested_ratio=r/min(params[n+'.weight'].shape), actual_rank=r)]
                    for n, r in ranks.items()}
        provenance['allocation'] = file_identity(Path(args.allocation_dir)/'manifest.json')
    if args.activation_bank:
        compatible_stage(args.activation_bank, provenance, '04_activation_curves')
        method = allocation['method'] if allocation else 'activation_svd'
        bank_identity = file_identity(Path(args.activation_bank)/'manifest.json')
        provenance['activation_bank'] = bank_identity
        with np.load(Path(args.activation_bank)/'factors.npz', allow_pickle=False) as bank:
            prefix = '.svd' if method == 'exact_svd' else ''
            factors = {n: (bank[n+prefix+'.first'][:r].copy(), bank[n+prefix+'.second'][:, :r].copy())
                       for n, r in ranks.items()}
    provenance['method'] = method
    provenance.update(selected_matrices=names, ranks=requests)
    with stage_output(args.output_dir, provenance) as out:
        candidate, records = compress(params, args.scope, ranks=ranks, factors=factors)
        actual = {k[:-len('.first.weight')] for k in candidate if k.endswith('.first.weight')}
        if actual != set(names):
            raise AssertionError('Actual factor list differs from requested matrices')
        for name in names:
            if name + '.bias' in params:
                np.testing.assert_array_equal(candidate[name + '.second.bias'], params[name + '.bias'])
        candidate = jax.tree.map(np.asarray, candidate)
        control = all(ranks[n] == min(params[n + '.weight'].shape) for n in names)
        meta = {**metadata, 'dense_source': provenance['checkpoint'], 'candidate': {
            'scope': args.scope, 'ranks': ranks, 'requests': requests, 'full_rank_control': control,
            'method': method, 'factorization_policy': 'all_selected_v1',
            'activation_bank': bank_identity, 'allocation': allocation}}
        save_checkpoint(out / 'candidate.npz', spec, candidate, buffers, metadata=meta)
        _, restored, _, _, _ = load_checkpoint(out / 'candidate.npz')
        for key in candidate:
            np.testing.assert_array_equal(candidate[key], restored[key])
        report = dict(scope=args.scope, ranks=ranks, requests=requests, matrices=records,
                      full_rank_control=control, serialization_verified=True,
                      total_params=sum(v.size for v in candidate.values()),
                      param_memory_bytes=sum(v.nbytes for v in candidate.values()),
                      dense_source=provenance['checkpoint'])
        publish_json(out / 'candidate.json', report)
        print(f'Created {len(names)} factors; {report["total_params"]} parameters', flush=True)


if __name__ == '__main__':
    main()
