"""06_finetune: train factors directly; validation selection; epoch recovery files."""
import math
from pathlib import Path
import time

import jax
import jax.numpy as jnp
import numpy as np

from aerial_svd.jax_checkpoint import load_checkpoint, save_checkpoint
from aerial_svd.jax_engine import Engine
from aerial_svd.lowrank_common import file_identity, load_inputs, parser, publish_json, stage_output
from aerial_svd.lowrank_quality import collect_logits, dataset_for, load_reference, paired_metrics


def main(argv=None):
    cli = parser('06_finetune')
    cli.add_argument('--reference-dir', required=True)
    cli.add_argument('--epochs', type=int, default=5)
    cli.add_argument('--batch-size', type=int, default=8)
    cli.add_argument('--eval-batch-size', type=int, default=8)
    cli.add_argument('--lr', type=float, default=1e-5)
    cli.add_argument('--resume', help='An epoch NPZ from an interrupted run; new output directory required')
    args = cli.parse_args(argv)
    if min(args.epochs, args.batch_size, args.eval_batch_size) < 1 or not np.isfinite(args.lr) or args.lr <= 0:
        cli.error('Require positive epochs, batch sizes and finite lr')
    cfg, spec, params, buffers, metadata, provenance = load_inputs(args, '06_finetune', require_dense=False)
    dataset = dataset_for(cfg, spec, metadata, provenance)
    source_hash = metadata.get('dense_source', provenance['checkpoint'])['sha256']
    reference, _ = load_reference(args.reference_dir, provenance, dataset, source_hash)
    protocol = dict(epochs=args.epochs, batch_size=args.batch_size, eval_batch_size=args.eval_batch_size,
                    lr=args.lr, schedule='cosine_epoch', optimizer='AdamW', weight_decay=.05,
                    label_smoothing=.1, validation_smoothing=0., seed=42, amp=False,
                    accumulation=1, selection='max validation top1; earliest epoch on ties')
    provenance.update(training_protocol=protocol, trainable_parameters={k: list(v.shape) for k, v in params.items()},
                      frozen_parameters=[], reference=file_identity(Path(args.reference_dir) / 'manifest.json'))
    shapes = {k: v.shape for k, v in params.items()}
    engine = Engine(spec, amp=False, smoothing=.1, weight_decay=.05)
    params, buffers = jax.tree.map(jnp.asarray, (params, buffers))
    state = engine.optimizer.init(params)
    history, best, best_path, start = [], -1., None, 1
    before = None
    if args.resume:
        rspec, rp, rb, rm, leaves = load_checkpoint(args.resume)
        recovery = rm['finetune_recovery']
        if (rspec != spec or {k: v.shape for k, v in rp.items()} != shapes
                or recovery['protocol'] != protocol
                or recovery['input_sha256'] != provenance['checkpoint']['sha256']
                or recovery['reference_sha256'] != provenance['reference']['sha256']):
            raise ValueError('Resume protocol/input mismatch')
        _, tree = jax.tree.flatten(state)
        expected = jax.tree.leaves(state)
        if len(expected) != len(leaves) or any(a.shape != b.shape or a.dtype != b.dtype for a, b in zip(expected, leaves)):
            raise ValueError('Resume optimizer schema mismatch')
        if any(not np.isfinite(x).all() for x in leaves):
            raise ValueError('Nonfinite resume optimizer state')
        params, buffers = jax.tree.map(jnp.asarray, (rp, rb))
        state = jax.tree.unflatten(tree, [jnp.asarray(x) for x in leaves])
        history, best, best_path = recovery['history'], recovery['best_top1'], Path(recovery['best_path'])
        if best_path.resolve() != Path(args.resume).resolve() and file_identity(best_path)['sha256'] != recovery['best_sha256']:
            raise ValueError('Resume best checkpoint changed')
        start, before = recovery['epoch'] + 1, recovery['before']
        provenance['resume'] = file_identity(args.resume)
    with stage_output(args.output_dir, provenance) as out:
        if before is None:
            logits, labels, pixels = collect_logits(engine, params, buffers, dataset.batches('val', args.eval_batch_size))
            np.testing.assert_array_equal(labels, reference['labels'])
            if pixels != reference['pixels_sha256'].item():
                raise ValueError('Training validation pixels differ from reference')
            before = paired_metrics(logits, labels, reference['logits'])
        publish_json(out / 'before.json', before)
        for epoch in range(start, args.epochs + 1):
            begin = time.perf_counter()
            lr = args.lr * (1 + math.cos(math.pi * (epoch - 1) / args.epochs)) / 2
            params, buffers, state, train = engine.train_epoch(params, buffers, state,
                dataset.batches('train', args.batch_size, epoch=epoch), lr, accumulation=1)
            val = engine.evaluate(params, buffers, dataset.batches('val', args.eval_batch_size), smoothing=0.)
            if {k: v.shape for k, v in params.items()} != shapes:
                raise AssertionError('Fine-tuning changed factor ranks')
            row = dict(epoch=epoch, lr=lr, train=train, validation=val, seconds=time.perf_counter()-begin)
            history.append(row)
            checkpoint = out / f'epoch-{epoch:04d}.npz'
            improved = val['top1'] > best
            if improved:
                best, best_path = val['top1'], checkpoint.resolve()
            recovery = dict(protocol=protocol, input_sha256=provenance['checkpoint']['sha256'],
                            reference_sha256=provenance['reference']['sha256'], epoch=epoch,
                            history=history, best_top1=best, best_path=str(best_path), before=before,
                            best_sha256=None if improved else file_identity(best_path)['sha256'])
            # Self hash cannot be embedded in its own file. The resume reader checks
            # the independent best checkpoint when it differs from the resumed file.
            meta = {**metadata, 'finetune_recovery': recovery}
            save_checkpoint(checkpoint, spec, params, buffers, metadata=meta, optimizer_leaves=jax.tree.leaves(state))
            publish_json(out / f'epoch-{epoch:04d}.json', row)
            print(row, flush=True)
        _, bp, bb, _, _ = load_checkpoint(best_path)
        bp, bb = jax.tree.map(jnp.asarray, (bp, bb))
        logits, labels, pixels = collect_logits(engine, bp, bb, dataset.batches('val', args.eval_batch_size))
        if pixels != reference['pixels_sha256'].item():
            raise ValueError('Final validation pixels changed')
        after = paired_metrics(logits, labels, reference['logits'])
        save_checkpoint(out / 'best.npz', spec, bp, bb,
                        metadata={**metadata, 'finetune': dict(protocol=protocol, best_epoch=int(np.argmax([r['validation']['top1'] for r in history]))+1)})
        publish_json(out / 'history.json', history)
        publish_json(out / 'finetune.json', dict(before=before, after=after, protocol=protocol,
                                                best_epoch=int(np.argmax([r['validation']['top1'] for r in history]))+1,
                                                ranks_preserved=True, total_params=sum(v.size for v in bp.values())))


if __name__ == '__main__':
    main()
