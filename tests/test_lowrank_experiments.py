"""Native end-to-end regressions for independent candidate/validation stages."""
import json
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import yaml

from aerial_svd.jax_checkpoint import load_checkpoint, save_checkpoint
from aerial_svd.jax_data import ImageFolder
from aerial_svd.jax_models import initialize
from aerial_svd.lowrank_make_candidate import main as make_candidate
from aerial_svd.lowrank_validate import main as validate
from aerial_svd.lowrank_finetune import main as finetune
from aerial_svd.lowrank_benchmark import main as benchmark
from aerial_svd.lowrank_quality import paired_metrics, verified_stage
from aerial_svd.model_spec import ModelSpec


class CandidateValidation(unittest.TestCase):
    def test_requested_grid_has_168_distinct_configurations(self):
        path = Path(__file__).resolve().parents[1] / 'scripts' / 'lowrank_series.py'
        spec = importlib.util.spec_from_file_location('series', path)
        series = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(series)
        groups = [list(series.configurations(s)) for s in ('global', 'regions', 'single')]
        self.assertEqual([len(g) for g in groups], [24, 48, 96])
        self.assertEqual(len({r[0] for g in groups for r in g}), 168)
        for _, scope, rho, names in groups[1]:
            self.assertEqual(len(names), 8)
            self.assertEqual(len({int(n.split('.')[1]) for n in names}), 4)
            self.assertIn(rho, series.RATIOS)
        for _, _, _, names in groups[2]:
            self.assertEqual(len(names), 1)
            self.assertIn(int(names[0].split('.')[1]), (0, 5, 11))

    def test_metrics_hand_calculation(self):
        logits = np.array([[0., 0.], [0., np.log(3.)]])
        result = paired_metrics(logits, np.array([0, 1]), np.zeros_like(logits))
        self.assertEqual(result['top1'], 1.)
        self.assertAlmostEqual(result['loss'], (np.log(2) + np.log(4/3))/2)
        self.assertAlmostEqual(result['logit_l2_mean'], np.log(3)/2)
        self.assertAlmostEqual(result['logit_l2_p95'], .95*np.log(3))

    def test_materialize_validate_reload_full_rank_and_pair_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            spec = ModelSpec('vit_small_patch16_224', 2)
            params, buffers = initialize(spec)
            source = root / 'source.npz'
            save_checkpoint(source, spec, params, buffers, metadata={'config': {'classes': ['A', 'B']}})
            for c in ('A', 'B'):
                (root / 'data' / c).mkdir(parents=True)
                for i in range(3):
                    Image.new('RGB', (40, 40), (60+i, 90, 130)).save(root / 'data' / c / f'{i}.png')
            split = root / 'split.json'
            split.write_text(json.dumps(dict(class_to_idx={'A': 0, 'B': 1},
                                             indices={'train': [0, 3], 'val': [1, 4], 'test': [2, 5]})))
            config = root / 'config.yaml'
            config.write_text(yaml.safe_dump(dict(model=spec.model, img_size=224, seed=42,
                                                  data_dir=str(root / 'data'), split_file=str(split), num_workers=0)))
            def args(checkpoint, out):
                return ['--config', str(config), '--checkpoint', str(checkpoint), '--output-dir', str(root / out)]
            calls = []
            original = ImageFolder.batches
            def validation_only(dataset, split_name, *a, **kw):
                calls.append(split_name)
                self.assertEqual(split_name, 'val')
                yield from original(dataset, split_name, *a, **kw)
            with patch.object(ImageFolder, 'batches', validation_only):
                validate(args(source, 'baseline') + ['--batch-size', '2'])
                make_candidate(args(source, 'candidate') + ['--scope', 'attn', '--matrices',
                                                           'blocks.5.attn.proj', '--rank-ratios', '1.0'])
                path = root / 'candidate' / 'candidate.npz'
                _, restored, _, meta, _ = load_checkpoint(path)
                self.assertEqual(meta['candidate']['ranks'], {'blocks.5.attn.proj': 384})
                np.testing.assert_array_equal(restored['blocks.5.attn.proj.second.bias'], params['blocks.5.attn.proj.bias'])
                self.assertGreater(sum(v.size for v in restored.values()), sum(v.size for v in params.values()))
                validate(args(path, 'validation') + ['--reference-dir', str(root / 'baseline'), '--batch-size', '1'])
            result = json.loads((root / 'validation' / 'validation.json').read_text())
            self.assertTrue(result['full_rank_control'])
            self.assertLess(result['logit_relative_frobenius'], 5e-5)
            self.assertEqual(set(calls), {'val'})
            verified_stage(root / 'validation')
            with self.assertRaises(FileExistsError):
                make_candidate(args(source, 'candidate') + ['--scope', 'ffn', '--rank-ratios', '.5'])
            (root / 'validation' / 'validation.json').write_text('{}')
            with self.assertRaises(ValueError):
                verified_stage(root / 'validation')
            ft_args = ['--reference-dir', str(root / 'baseline'), '--epochs', '2',
                       '--batch-size', '1', '--eval-batch-size', '2']
            # Real gradient updates train factors, retain ranks and recover Adam state.
            finetune(args(path, 'ft') + ft_args)
            finetune(args(path, 'ft_resume') + ft_args + ['--resume', str(root / 'ft' / 'epoch-0001.npz')])
            _, fp, _, _, _ = load_checkpoint(root / 'ft' / 'best.npz')
            _, fr, _, _, _ = load_checkpoint(root / 'ft_resume' / 'best.npz')
            self.assertEqual({k: v.shape for k, v in fp.items()}, {k: v.shape for k, v in restored.items()})
            for k in fp:
                np.testing.assert_array_equal(fp[k], fr[k])
            _, ep, _, _, opt = load_checkpoint(root / 'ft' / 'epoch-0002.npz')
            _, er, _, _, resumed_opt = load_checkpoint(root / 'ft_resume' / 'epoch-0002.npz')
            for k in ep:
                np.testing.assert_array_equal(ep[k], er[k])
            for a, b in zip(opt, resumed_opt):
                np.testing.assert_array_equal(a, b)
            self.assertFalse(np.array_equal(fp['blocks.5.attn.proj.first.weight'], restored['blocks.5.attn.proj.first.weight']))
            benchmark(args(path, 'bench') + ['--batch-sizes', '1', '--warmup', '1', '--repeats', '2'])
            bench = json.loads((root / 'bench' / 'benchmark.json').read_text())
            self.assertEqual(len(bench['measurements'][0]['trials_seconds']), 2)
            self.assertIsNone(bench['inference_gpu_peak_bytes'])
            # A changed validation image cannot be silently paired with cached logits.
            Image.new('RGB', (40, 40), (220, 30, 90)).save(root / 'data' / 'A' / '1.png')
            with self.assertRaisesRegex(ValueError, 'pixels differ'):
                validate(args(path, 'changed') + ['--reference-dir', str(root / 'baseline')])


if __name__ == '__main__':
    unittest.main()
