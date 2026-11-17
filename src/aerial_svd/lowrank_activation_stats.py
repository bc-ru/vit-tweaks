"""03_activation_stats: streaming uncentered moments, train calibration or validation."""
import hashlib

import jax
import jax.numpy as jnp
import numpy as np

from aerial_svd.jax_models import apply
from aerial_svd.lowrank_common import load_inputs, parser, publish_json, selected_names, stage_output
from aerial_svd.lowrank_quality import dataset_for
from aerial_svd.matrix_archive import save_npz_new


def calibration_subset(dataset, per_class):
    if per_class < 1:
        raise ValueError('Require positive samples per class')
    rng = np.random.default_rng(42)
    result = []
    for label in range(len(dataset.classes)):
        available = [i for i in dataset.indices['train'] if dataset.samples[i][1] == label]
        if len(available) < per_class:
            raise ValueError('Too few train examples for stratified calibration')
        result.extend(int(i) for i in rng.choice(available, per_class, replace=False))
    return sorted(result)


def gram_inputs(params, buffers, images, spec, names):
    _, _, captures = apply(params, buffers, images, spec, capture=names)
    result = {}
    for name in names:
        x = captures[name][0].reshape(-1, captures[name][0].shape[-1])
        result[name] = (x.T @ x, x.shape[0])
    return result


def main(argv=None):
    cli = parser('03_activation_stats')
    cli.add_argument('--scope', choices=('ffn', 'attn', 'all_blocks'), default='all_blocks')
    cli.add_argument('--matrices', nargs='+')
    cli.add_argument('--split', choices=('train', 'val'), required=True)
    cli.add_argument('--samples-per-class', type=int, default=10)
    cli.add_argument('--batch-size', type=int, default=8)
    args = cli.parse_args(argv)
    if args.batch_size < 1:
        cli.error('Require positive batch size')
    cfg, spec, params, buffers, metadata, provenance = load_inputs(args, '03_activation_stats')
    dataset = dataset_for(cfg, spec, metadata, provenance)
    names = tuple(selected_names(params, args))
    indices = calibration_subset(dataset, args.samples_per_class) if args.split == 'train' else list(dataset.indices['val'])
    provenance.update(method='input_second_moment', evaluation_split=args.split,
                      calibration_subset=indices if args.split == 'train' else None,
                      selected_matrices=names, accumulation='float32 highest-precision batch Gram; float64 host sum')
    forward = jax.jit(lambda p, b, x: gram_inputs(p, b, x, spec, names))
    params, buffers = jax.tree.map(jnp.asarray, (params, buffers))
    sums = {n: np.zeros((params[n+'.weight'].shape[1],)*2, np.float64) for n in names}
    counts = dict.fromkeys(names, 0)
    digest = hashlib.sha256()
    kwargs = {'calibration_indices': indices} if args.split == 'train' else {}
    with stage_output(args.output_dir, provenance) as out:
        seen = 0
        for images, _ in dataset.batches(args.split, args.batch_size, **kwargs):
            digest.update(np.ascontiguousarray(images).tobytes())
            stats = forward(params, buffers, jnp.asarray(images))
            for n in names:
                gram, count = stats[n]
                gram = np.asarray(gram, dtype=np.float64)
                if gram.shape != sums[n].shape or not np.isfinite(gram).all():
                    raise ValueError(f'Invalid activation statistics: {n}')
                sums[n] += gram
                counts[n] += int(count)
            seen += len(images)
            if seen % 80 == 0:
                print(f'{args.split}: {seen}/{len(indices)} images', flush=True)
        if seen != len(indices) or not all(counts.values()):
            raise AssertionError('Activation subset incomplete')
        save_npz_new(out / 'moments.npz', {n: (sums[n]+sums[n].T)/(2*counts[n]) for n in names})
        publish_json(out / 'statistics.json', dict(split=args.split, indices=indices, images=seen,
                     rows_per_matrix=counts, pixels_sha256=digest.hexdigest(),
                     shapes={n: list(sums[n].shape) for n in names}, centered=False,
                     token_policy='all patch tokens plus CLS, equal row weights', augmentation=False))
        print(f'Saved {len(names)} moments from {seen} {args.split} images', flush=True)


if __name__ == '__main__':
    main()
