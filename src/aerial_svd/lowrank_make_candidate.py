"""04_make_candidate: one exact-SVD model, including expanding/full ranks."""
import jax
import numpy as np

from aerial_svd.jax_checkpoint import load_checkpoint, save_checkpoint
from aerial_svd.jax_compression import compress
from aerial_svd.lowrank_common import (load_inputs, parser, publish_json, rank_requests,
                                       selected_names, stage_output)


def main(argv=None):
    cli = parser('04_make_candidate', matrices=True)
    args = cli.parse_args(argv)
    cfg, spec, params, buffers, metadata, provenance = load_inputs(args, '04_make_candidate')
    names = selected_names(params, args)
    requests = {name: rank_requests(params[name + '.weight'].shape, args) for name in names}
    if any(len(rows) != 1 for rows in requests.values()):
        cli.error('One candidate requires exactly one rank/ratio')
    ranks = {name: rows[0]['actual_rank'] for name, rows in requests.items()}
    provenance.update(selected_matrices=names, ranks=requests)
    with stage_output(args.output_dir, provenance) as out:
        candidate, records = compress(params, args.scope, ranks=ranks)
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
            'method': 'exact_svd', 'factorization_policy': 'all_selected_v1'}}
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
