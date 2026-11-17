"""04_activation_curves: factor bank and local errors; no image inference."""
import json
from pathlib import Path

import numpy as np

from aerial_svd.activation_svd import local_error, weighted_factors
from aerial_svd.jax_compression import factorize_weight
from aerial_svd.lowrank_common import (file_identity, load_inputs, parser, publish_json,
                                       rank_requests, selected_names, stage_output, write_csv)
from aerial_svd.lowrank_quality import compatible_stage
from aerial_svd.matrix_archive import save_npz_new


def main(argv=None):
    cli = parser('04_activation_curves', matrices=True)
    cli.add_argument('--calibration-dir', required=True)
    cli.add_argument('--validation-stats-dir', required=True)
    cli.add_argument('--ridge-relative', type=float, required=True)
    args = cli.parse_args(argv)
    cfg, spec, params, buffers, metadata, provenance = load_inputs(args, '04_activation_curves')
    names = selected_names(params, args)
    cm = compatible_stage(args.calibration_dir, provenance, '03_activation_stats')
    vm = compatible_stage(args.validation_stats_dir, provenance, '03_activation_stats')
    if cm['evaluation_split'] != 'train' or vm['evaluation_split'] != 'val':
        raise ValueError('Require train calibration and separate validation moments')
    for key in ('dataset_fingerprint', 'classes', 'image_decoder', 'preprocessing'):
        if cm[key] != vm[key]:
            raise ValueError(f'Incompatible statistics: {key}')
    cs = json.loads((Path(args.calibration_dir)/'statistics.json').read_text())
    vs = json.loads((Path(args.validation_stats_dir)/'statistics.json').read_text())
    if set(cs['indices']) & set(vs['indices']):
        raise ValueError('Calibration and validation overlap')
    provenance.update(method='activation_svd', calibration=file_identity(Path(args.calibration_dir)/'manifest.json'),
                      validation_statistics=file_identity(Path(args.validation_stats_dir)/'manifest.json'),
                      ridge_relative=args.ridge_relative, calibration_subset=cs['indices'],
                      selected_matrices=names, dataset_fingerprint=cm['dataset_fingerprint'])
    arrays, rows, diagnostics = {}, [], {}
    with stage_output(args.output_dir, provenance) as out:
        with np.load(Path(args.calibration_dir)/'moments.npz', allow_pickle=False) as ca, \
             np.load(Path(args.validation_stats_dir)/'moments.npz', allow_pickle=False) as va:
            for name in names:
                w = np.asarray(params[name+'.weight'], np.float64)
                c, v = ca[name], va[name]
                first, second, singular, info = weighted_factors(w, c, args.ridge_relative)
                diagnostics[name] = info
                arrays[name+'.first'] = first
                arrays[name+'.second'] = second
                arrays[name+'.weighted_singular_values'] = singular
                # Existing exact-float32 SVD provides the local baseline. Its model
                # validation is reused; no ordinary-SVD model sweep is repeated.
                sf, ss = (np.asarray(a) for a in factorize_weight(w, min(w.shape)))
                arrays[name+'.svd.first'], arrays[name+'.svd.second'] = sf, ss
                denominators = (float(np.sum(w*w)), local_error(w, c), local_error(w, v))
                for method, a, b in [('activation_svd', first, second), ('exact_svd', sf, ss)]:
                    for request in rank_requests(w.shape, args):
                        rank = request['actual_rank']
                        # Evaluate the serialized float32 factors, with float64
                        # accumulation. The inference itself stays factorized.
                        e = w - b[:, :rank].astype(np.float64) @ a[:rank].astype(np.float64)
                        fro2, cal, val = float(np.sum(e*e)), local_error(e, c), local_error(e, v)
                        row = dict(name=name, method=method, **request, relative_rank=rank/min(w.shape),
                                   matrix_frobenius=float(np.sqrt(fro2)),
                                   matrix_relative_frobenius=float(np.sqrt(fro2/max(denominators[0], 1e-30))),
                                   calibration_mse=cal, calibration_relative_rmse=float(np.sqrt(cal/max(denominators[1], 1e-30))),
                                   validation_mse=val, validation_relative_rmse=float(np.sqrt(val/max(denominators[2], 1e-30))),
                                   regularized_mse=cal+info['ridge_absolute']*fro2,
                                   ridge_absolute=info['ridge_absolute'],
                                   in_features=w.shape[1], out_features=w.shape[0],
                                   factor_weight_params=rank*sum(w.shape))
                        if rank == min(w.shape) and row['matrix_relative_frobenius'] > 5e-5:
                            raise AssertionError(f'Full-rank matrix recovery failed: {name}')
                        rows.append(row)
                print(f'Curves and full-rank reconstruction passed: {name}', flush=True)
        save_npz_new(out/'factors.npz', arrays)
        write_csv(out, 'local_errors.csv', rows)
        publish_json(out/'diagnostics.json', diagnostics)


if __name__ == '__main__':
    main()
