"""05_validate: frozen validation logits and paired dense/candidate metrics."""
import jax
import jax.numpy as jnp
import numpy as np

from aerial_svd.jax_engine import Engine
from aerial_svd.lowrank_common import file_identity, load_inputs, parser, publish_json, stage_output
from aerial_svd.lowrank_quality import collect_logits, dataset_for, load_reference, paired_metrics
from aerial_svd.matrix_archive import save_npz_new


def main(argv=None):
    cli = parser('05_validate')
    cli.add_argument('--reference-dir', help='Completed dense 05_validate directory; omit only for dense baseline')
    cli.add_argument('--batch-size', type=int, default=8)
    args = cli.parse_args(argv)
    if args.batch_size < 1:
        cli.error('batch-size must be positive')
    cfg, spec, params, buffers, metadata, provenance = load_inputs(args, '05_validate', require_dense=False)
    dataset = dataset_for(cfg, spec, metadata, provenance)
    candidate = metadata.get('candidate')
    if candidate and not args.reference_dir:
        raise ValueError('Factorized candidate requires dense reference')
    reference = None
    if args.reference_dir:
        source_hash = metadata.get('dense_source', provenance['checkpoint'])['sha256']
        reference, _ = load_reference(args.reference_dir, provenance, dataset, source_hash)
        provenance['reference'] = file_identity(args.reference_dir + '/manifest.json')
    params, buffers = jax.tree.map(jnp.asarray, (params, buffers))
    engine = Engine(spec, amp=False)
    with stage_output(args.output_dir, provenance) as out:
        logits, labels, pixels = collect_logits(engine, params, buffers, dataset.batches('val', args.batch_size))
        if reference is not None:
            np.testing.assert_array_equal(reference['labels'], labels)
            if reference['pixels_sha256'].item() != pixels:
                raise ValueError('Paired validation pixels differ')
        else:
            images, _ = next(dataset.batches('val', args.batch_size, max_batches=1))
            np.testing.assert_array_equal(logits[:len(images)], np.asarray(engine.forward(params, buffers, jnp.asarray(images))))
        metrics = paired_metrics(logits, labels, reference['logits'] if reference is not None else logits)
        control = bool(candidate and candidate['full_rank_control'] and not metadata.get('finetune'))
        # Float32 full-network control: relative error plus an absolute logit bound.
        if control and (metrics['logit_relative_frobenius'] > 5e-5 or metrics['logit_max_abs'] > 1e-3):
            raise AssertionError(f'Full-rank output control failed: {metrics}')
        metrics.update(total_params=sum(v.size for v in params.values()),
                       param_memory_bytes=sum(v.size * v.dtype.itemsize for v in params.values()),
                       checkpoint_bytes=provenance['checkpoint']['bytes'],
                       full_rank_control=control, evaluation_split='val', candidate=candidate,
                       source_checkpoint_sha256=metadata.get('dense_source', provenance['checkpoint'])['sha256'])
        save_npz_new(out / 'logits.npz', dict(logits=logits, labels=labels, indices=np.asarray(dataset.indices['val']),
                                            pixels_sha256=np.asarray(pixels)))
        publish_json(out / 'validation.json', metrics)
        print(metrics, flush=True)


if __name__ == '__main__':
    main()
