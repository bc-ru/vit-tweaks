"""04_allocate: explicit per-matrix ranks at the exact uniform-rank storage budget."""
import csv
from pathlib import Path

from aerial_svd.activation_svd import allocate_ranks
from aerial_svd.lowrank_quality import compatible_stage
from aerial_svd.lowrank_common import (file_identity, load_inputs, parser, publish_json,
                                       rank_requests, selected_names, stage_output)


def main(argv=None):
    cli = parser('04_allocate', matrices=True)
    cli.add_argument('--activation-bank', required=True)
    cli.add_argument('--method', choices=('activation_svd', 'exact_svd'), default='activation_svd')
    args = cli.parse_args(argv)
    _, _, params, _, _, provenance = load_inputs(args, '04_allocate')
    compatible_stage(args.activation_bank, provenance, '04_activation_curves')
    names = selected_names(params, args)
    uniform = {n: rank_requests(params[n+'.weight'].shape, args) for n in names}
    if any(len(r) != 1 for r in uniform.values()):
        raise ValueError('One budget requires one uniform reference rank')
    budget = sum(rows[0]['actual_rank']*sum(params[n+'.weight'].shape) for n, rows in uniform.items())
    with open(Path(args.activation_bank)/'local_errors.csv') as stream:
        curves = list(csv.DictReader(stream))
    for row in curves:
        for key in ('actual_rank', 'factor_weight_params'):
            row[key] = int(row[key])
        row['calibration_mse'] = float(row['calibration_mse'])
    ranks, info = allocate_ranks(curves, names, budget, args.method)
    provenance.update(method=args.method, activation_bank=file_identity(Path(args.activation_bank)/'manifest.json'))
    with stage_output(args.output_dir, provenance) as out:
        publish_json(out/'allocation.json', dict(ranks=ranks, uniform_reference=uniform, **info))
        print(info, flush=True)


if __name__ == '__main__':
    main()
