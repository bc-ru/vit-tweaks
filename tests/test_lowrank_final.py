"""Final-evaluation trust boundaries and full ordered synthetic-test regression."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import yaml

from aerial_svd.jax_checkpoint import save_checkpoint
from aerial_svd.jax_data import ImageFolder
from aerial_svd.jax_models import initialize
from aerial_svd.lowrank_common import file_identity, publish_json, stage_output
from aerial_svd.lowrank_catalog import benchmark_result
from aerial_svd.lowrank_freeze import index_digest, main as freeze, required_ids
from aerial_svd.lowrank_quality import dataset_for, verified_stage
from aerial_svd.lowrank_test import locked_model, main as final_test
from aerial_svd.model_spec import ModelSpec


class FinalEvaluation(unittest.TestCase):
    def test_benchmark_audit_rejects_wrong_identity_and_incomplete_trials(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model = dict(checkpoint=dict(sha256='a'*64))
            provenance = dict(stage='07_benchmark', checkpoint=model['checkpoint'],
                arguments=dict(batch_sizes=[16], warmup=5, repeats=50),
                environment=dict(platform='gpu'), preprocessing={}, dataset_fingerprint='fixed',
                config=dict(sha256='b'*64), aid_split=dict(sha256='c'*64), source_files_sha256={})
            for name, trials in (('complete', [0.01]*50), ('partial', [0.01]*49), ('invalid', [-0.01]*50)):
                with stage_output(root/name, provenance) as out:
                    publish_json(out/'benchmark.json', dict(measurements=[dict(batch_size=16,
                        pixels_sha256='fixed', trials_seconds=trials)]))
            benchmark_result(root/'complete', model, 16)
            for directory, candidate, batch in ((root/'partial', model, 16), (root/'invalid', model, 16),
                    (root/'complete', model, 1), (root/'complete', dict(checkpoint=dict(sha256='d'*64)), 16)):
                with self.assertRaisesRegex(ValueError, 'identity/protocol mismatch'):
                    benchmark_result(directory, candidate, batch)

    def test_prespecified_list_has_all_pairs_and_dense_training_control(self):
        ids = required_ids()
        self.assertEqual(len(ids), 20)
        self.assertEqual(len(set(ids)), 20)
        self.assertEqual(ids[0], 'dense__dense')
        for method in ('exact_svd', 'activation_svd'):
            self.assertEqual(sum(k.startswith(method+'__') for k in ids), 9)

    def test_order_full_split_pairing_locks_and_no_reselection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            spec = ModelSpec('vit_small_patch16_224', 2)
            params, buffers = initialize(spec)
            checkpoint = root / 'source.npz'
            save_checkpoint(checkpoint, spec, params, buffers, metadata={'config': {'classes': ['A', 'B']}})
            for cls in ('A', 'B'):
                (root/'data'/cls).mkdir(parents=True)
                for i in range(4):
                    Image.new('RGB', (32, 32), (40+i*20, 90, 120)).save(root/'data'/cls/f'{i}.png')
            split = root/'split.json'
            indices = {'train': [0, 4, 6], 'val': [1, 5], 'test': [7, 2, 3]}
            split.write_text(json.dumps(dict(class_to_idx={'A': 0, 'B': 1}, indices=indices)))
            cfg = dict(model=spec.model, img_size=224, seed=42, data_dir=str(root/'data'),
                       split_file=str(split), num_workers=0)
            config = root/'config.yaml'
            config.write_text(yaml.safe_dump(cfg))
            provenance = {}
            dataset_for(cfg, spec, {}, provenance)
            catalog = root/'catalog.json'
            catalog.write_text('{}')
            selection = dict(models=[dict(model_id=k, checkpoint=file_identity(checkpoint), candidate={})
                                    for k in ('dense__dense', 'dense__control')],
                catalog=file_identity(catalog), config=file_identity(config), aid_split=file_identity(split),
                test_count=3, test_indices_sha256=index_digest(indices['test']), batch_size=2,
                dataset={k: provenance[k] for k in ('dataset_fingerprint', 'classes', 'class_to_idx',
                                                  'image_decoder', 'preprocessing')})
            with stage_output(root/'selection', dict(stage='09_freeze')) as out:
                publish_json(out/'selection.json', selection)
            path = root/'selection'/'selection.json'
            digest = file_identity(path)['sha256']
            frozen_bytes, source_bytes, split_bytes = path.read_bytes(), checkpoint.read_bytes(), split.read_bytes()
            def args(model_id):
                return ['--config', str(config), '--checkpoint', str(checkpoint), '--selection', str(path),
                    '--selection-sha256', digest, '--model-id', model_id, '--finalization-root', str(root),
                    '--output-dir', str(root/'09_test'/model_id), '--batch-size', '2']
            original = ImageFolder.batches
            calls = []
            def test_only(dataset, split_name, *a, **kw):
                calls.append(split_name)
                self.assertEqual(split_name, 'test')
                yield from original(dataset, split_name, *a, **kw)
            with patch.object(ImageFolder, 'batches', test_only):
                final_test(args('dense__dense'))
                final_test(args('dense__control'))
            self.assertEqual(calls, ['test', 'test'])
            for k in ('dense__dense', 'dense__control'):
                verified_stage(root/'09_test'/k)
                result = json.loads((root/'09_test'/k/'test.json').read_text())
                self.assertEqual(result['num_samples'], 3)
                self.assertEqual(result['selection_sha256'], digest)
                with np.load(root/'09_test'/k/'logits.npz') as data:
                    np.testing.assert_array_equal(data['indices'], indices['test'])
            self.assertEqual(result['logit_max_abs'], 0.)
            with patch('aerial_svd.lowrank_test.dataset_for', side_effect=AssertionError('must reject before data access')):
                with self.assertRaisesRegex(ValueError, 'SHA-256 mismatch'):
                    final_test([x if x != digest else '0'*64 for x in args('dense__dense')])
                with self.assertRaisesRegex(ValueError, 'not uniquely present'):
                    final_test(args('unknown'))
                config_bytes = config.read_bytes()
                config.write_bytes(config_bytes + b'\n')
                with self.assertRaisesRegex(ValueError, 'checkpoint/config/split mismatch'):
                    final_test(args('dense__dense'))
                config.write_bytes(config_bytes)
                checkpoint.write_bytes(b'changed')
                with self.assertRaisesRegex(ValueError, 'checkpoint/config/split mismatch'):
                    final_test(args('dense__dense'))
                checkpoint.write_bytes(source_bytes)
            Image.new('RGB', (32, 32), (230, 10, 50)).save(root/'data'/'B'/'3.png')
            changed = args('dense__control')
            changed[changed.index('--output-dir')+1] = str(root/'changed_pixels')
            with self.assertRaisesRegex(ValueError, 'Test pixels differ'):
                final_test(changed)
            with self.assertRaisesRegex(ValueError, 'selection cannot be changed'):
                freeze(['--catalog', str(catalog), '--finalization-root', str(root), '--output-dir', str(root/'new')])
            # A newly published selection cannot replace the externally recorded lock.
            path.write_bytes(frozen_bytes + b' ')
            with self.assertRaisesRegex(ValueError, 'SHA-256 mismatch'):
                locked_model(path, digest, 'dense__dense', checkpoint, config, split)
            self.assertEqual(split.read_bytes(), split_bytes)


if __name__ == '__main__':
    unittest.main()
