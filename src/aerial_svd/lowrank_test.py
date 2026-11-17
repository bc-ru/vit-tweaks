"""09_test: full held-out evaluation of exactly one hash-locked prespecified model."""
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from aerial_svd.jax_engine import Engine
from aerial_svd.lowrank_catalog import checked_identity
from aerial_svd.lowrank_common import file_identity, load_inputs, parser, publish_json, stage_output
from aerial_svd.lowrank_freeze import index_digest
from aerial_svd.lowrank_quality import collect_logits, dataset_for, paired_metrics, verified_stage
from aerial_svd.matrix_archive import save_npz_new


def locked_model(selection_path, expected_hash, model_id, checkpoint, config, split):
    if file_identity(selection_path)['sha256'] != expected_hash:
        raise ValueError('Frozen selection SHA-256 mismatch')
    verified_stage(Path(selection_path).parent)
    selection = json.loads(Path(selection_path).read_text())
    checked_identity(selection['catalog'])
    models = [m for m in selection['models'] if m['model_id'] == model_id]
    if len(models) != 1:
        raise ValueError('Model is not uniquely present in frozen selection')
    model = models[0]
    for expected, actual in ((model['checkpoint'], checkpoint), (selection['config'], config),
                             (selection['aid_split'], split)):
        if file_identity(actual)['sha256'] != expected['sha256']:
            raise ValueError('Frozen checkpoint/config/split mismatch')
    return selection, model


def main(argv=None):
    cli = parser('09_test')
    cli.add_argument('--selection', required=True)
    cli.add_argument('--selection-sha256', required=True)
    cli.add_argument('--model-id', required=True)
    cli.add_argument('--finalization-root', required=True)
    cli.add_argument('--batch-size', type=int, default=8)
    args = cli.parse_args(argv)
    # Check the external lock before constructing any dataset or executing inference.
    import yaml
    cfg = yaml.safe_load(Path(args.config).read_text())
    selection, model = locked_model(args.selection, args.selection_sha256, args.model_id,
                                   args.checkpoint, args.config, cfg['split_file'])
    if args.batch_size != selection['batch_size']:
        raise ValueError('Test batch size differs from frozen protocol')
    cfg, spec, params, buffers, metadata, provenance = load_inputs(args, '09_test', require_dense=False)
    if (metadata.get('candidate') or {}) != model['candidate']:
        raise ValueError('Frozen candidate metadata differs')
    dataset = dataset_for(cfg, spec, metadata, provenance)
    for key, expected in selection['dataset'].items():
        if provenance[key] != expected:
            raise ValueError(f'Frozen dataset/preprocessing mismatch: {key}')
    if (len(dataset.indices['test']) != selection['test_count']
            or index_digest(dataset.indices['test']) != selection['test_indices_sha256']):
        raise ValueError('Frozen test order differs')
    provenance.update(evaluation_split='test', frozen_selection=file_identity(args.selection), model_id=args.model_id)
    root = Path(args.finalization_root)
    root.mkdir(parents=True, exist_ok=True)
    receipt = root / 'test_started.json'
    lock = dict(selection_sha256=args.selection_sha256,
                source_files_sha256=provenance['source_files_sha256'])
    if receipt.exists():
        if json.loads(receipt.read_text()) != lock:
            raise ValueError('Final test already locked to another selection or source')
    else:
        publish_json(receipt, lock)
    dense = args.model_id == 'dense__dense'
    reference = None
    if not dense:
        directory = root / '09_test' / 'dense__dense'
        manifest = verified_stage(directory)
        if manifest['frozen_selection']['sha256'] != args.selection_sha256:
            raise ValueError('Dense test reference selection differs')
        with np.load(directory / 'logits.npz', allow_pickle=False) as archive:
            reference = {k: archive[k].copy() for k in archive.files}
        np.testing.assert_array_equal(reference['indices'], dataset.indices['test'])
        provenance['reference'] = file_identity(directory / 'manifest.json')
    params, buffers = jax.tree.map(jnp.asarray, (params, buffers))
    with stage_output(args.output_dir, provenance) as out:
        logits, labels, pixels = collect_logits(Engine(spec, amp=False), params, buffers,
                                               dataset.batches('test', args.batch_size))
        if len(labels) != selection['test_count']:
            raise ValueError('Incomplete test evaluation')
        if reference is not None:
            np.testing.assert_array_equal(labels, reference['labels'])
            if pixels != reference['pixels_sha256'].item():
                raise ValueError('Test pixels differ across models')
        metrics = paired_metrics(logits, labels, logits if dense else reference['logits'])
        metrics.update(model_id=args.model_id, checkpoint_sha256=provenance['checkpoint']['sha256'],
                       selection_sha256=args.selection_sha256, evaluation_split='test',
                       test_indices_sha256=selection['test_indices_sha256'], pixels_sha256=pixels)
        save_npz_new(out / 'logits.npz', dict(logits=logits, labels=labels,
            indices=np.asarray(dataset.indices['test']), pixels_sha256=np.asarray(pixels)))
        publish_json(out / 'test.json', metrics)
        print(metrics, flush=True)


if __name__ == '__main__':
    main()
