"""Local objective, budget optimality and train/val-only stage regression proof."""
import itertools
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image
import yaml

from aerial_svd.activation_svd import allocate_ranks, local_error, weighted_factors
from aerial_svd.jax_checkpoint import load_checkpoint, save_checkpoint
from aerial_svd.jax_compression import compress, factorize_weight
from aerial_svd.jax_data import ImageFolder
from aerial_svd.jax_models import apply, initialize
from aerial_svd.lowrank_activation_stats import calibration_subset, gram_inputs, main as statistics
from aerial_svd.lowrank_activation_curves import main as curves
from aerial_svd.lowrank_make_candidate import main as candidate
from aerial_svd.lowrank_allocate import main as allocate
from aerial_svd.lowrank_validate import main as validate
from aerial_svd.lowrank_quality import verified_stage
from aerial_svd.model_spec import ModelSpec


class ActivationSVD(unittest.TestCase):
    def test_weighted_solution_and_empirical_objective(self):
        rng = np.random.default_rng(42)
        w = rng.normal(size=(9, 5))
        x = rng.normal(size=(60, 5))*[1, .1, 4, .3, 2]
        c = x.T@x/len(x)
        a, b, s, info = weighted_factors(w, c, 1e-4)
        for rank in (1, 3, 5):
            e = w-b[:, :rank].astype(float)@a[:rank].astype(float)
            empirical = np.sum((x@e.T)**2)/len(x)
            bias = np.arange(w.shape[0])/10
            factor_output = (x@a[:rank].T)@b[:, :rank].T+bias
            dense_output = x@(b[:, :rank]@a[:rank]).T+bias
            np.testing.assert_allclose(factor_output, dense_output, rtol=2e-5, atol=2e-6)
            self.assertAlmostEqual(local_error(e, c), empirical, places=10)
            objective = empirical+info['ridge_absolute']*np.sum(e*e)
            self.assertAlmostEqual(objective, np.sum(s[rank:]**2), places=8)
            sf, ss = map(np.asarray, factorize_weight(w, rank))
            se = w-ss@sf
            self.assertLessEqual(objective, local_error(se, c)+info['ridge_absolute']*np.sum(se*se)+1e-9)
        np.testing.assert_allclose(b@a, w, rtol=2e-6, atol=2e-6)

    def test_singular_zero_ill_conditioned_and_invalid_moments(self):
        w = np.arange(24.).reshape(4, 6)/24
        for c in (np.zeros((6, 6)), np.ones((6, 6)), np.diag([1e-25, 1e-20, 1e-12, 1e-8, 1, 2])):
            a, b, _, _ = weighted_factors(w, c, 1e-4)
            self.assertTrue(np.isfinite(a).all())
            np.testing.assert_allclose(b@a, w, rtol=2e-5, atol=2e-6)
        for c, ridge in ((np.eye(5), 1e-4), (-np.eye(6), 1e-4), (np.eye(6), 0),
                         (np.full((6, 6), np.nan), 1e-4), (np.triu(np.ones((6, 6))), 1e-4)):
            with self.assertRaises(ValueError):
                weighted_factors(w, c, ridge)

    def test_exact_budget_matches_brute_force_and_ties(self):
        rows = [dict(name=n, method='activation_svd', actual_rank=r, factor_weight_params=r*c,
                     calibration_mse=e) for n, c, errors in [('a', 4, [9, 3, 2, 0]), ('b', 6, [7, 2, 1, 0])]
                for r, e in enumerate(errors, 1)]
        for budget in (10, 20, 22, 28, 40):
            ranks, info = allocate_ranks(rows, ['a', 'b'], budget, 'activation_svd')
            brute = [(sum(r['calibration_mse'] for r in pair), tuple(r['actual_rank'] for r in pair))
                     for pair in itertools.product(rows[:4], rows[4:])
                     if sum(r['factor_weight_params'] for r in pair) == budget]
            self.assertEqual(tuple(ranks.values()), min(brute)[1])
            self.assertEqual(info['predicted_local_mse_sum'], min(brute)[0])
        tied = [dict(name=n, method='activation_svd', actual_rank=r, factor_weight_params=r,
                     calibration_mse=2-r) for n in ('a', 'b') for r in (1, 2)]
        ranks, _ = allocate_ranks(tied, ['a', 'b'], 3, 'activation_svd')
        self.assertEqual(ranks, {'a': 1, 'b': 2})
        with self.assertRaises(ValueError):
            allocate_ranks(rows, ['a', 'b'], 11, 'activation_svd')
        with self.assertRaises(ValueError):
            compress({'blocks.0.attn.proj.weight': np.eye(4)}, 'attn', ranks={'blocks.0.attn.proj': 4},
                     factors={'blocks.0.attn.proj': (np.eye(3), np.eye(3))})

    def test_stages_train_subset_grams_serialization_and_full_rank_logits(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            spec = ModelSpec('vit_small_patch16_224', 2)
            p, b = initialize(spec)
            source = root/'source.npz'
            save_checkpoint(source, spec, p, b, metadata={'config': {'classes': ['A', 'B']}})
            for cls in ('A', 'B'):
                (root/'data'/cls).mkdir(parents=True)
                for i in range(2):
                    Image.new('RGB', (30, 30), (50+i, 100, 130)).save(root/'data'/cls/f'{i}.png')
                # Test images cannot be decoded: successful stages prove no access.
                (root/'data'/cls/'2.png').write_bytes(b'held out, never decode')
            split = root/'split.json'
            split.write_text(json.dumps(dict(class_to_idx={'A': 0, 'B': 1},
                indices={'train': [0, 3], 'val': [1, 4], 'test': [2, 5]})))
            config = root/'config.yaml'
            config.write_text(yaml.safe_dump(dict(model=spec.model, img_size=224, seed=42,
                data_dir=str(root/'data'), split_file=str(split), num_workers=0)))
            def args(out, checkpoint=source):
                return ['--config', str(config), '--checkpoint', str(checkpoint), '--output-dir', str(root/out)]
            dataset = ImageFolder(yaml.safe_load(config.read_text()))
            selected = calibration_subset(dataset, 1)
            self.assertEqual(selected, [0, 3])
            self.assertEqual(selected, calibration_subset(dataset, 1))
            with self.assertRaises(ValueError):
                next(dataset.batches('val', 1, calibration_indices=[0]))
            with self.assertRaises(ValueError):
                next(dataset.batches('train', 1, calibration_indices=[2]))
            names = ['blocks.0.'+kind for kind in ('attn.qkv', 'attn.proj', 'mlp.fc1', 'mlp.fc2')]
            images, _ = next(dataset.batches('train', 2, calibration_indices=selected))
            _, _, captured = apply(p, b, jnp.asarray(images), spec, capture=names)
            grams = gram_inputs(p, b, jnp.asarray(images), spec, names)
            for name in names:
                x = np.asarray(captured[name][0]).reshape(-1, captured[name][0].shape[-1]).astype(float)
                np.testing.assert_allclose(grams[name][0], x.T@x, rtol=3e-5, atol=1e-3)
                self.assertEqual(grams[name][1], 2*197)
            # JIT with capture must preserve dense logits.
            actual = jax.jit(lambda x: apply(p, b, x, spec, capture=names)[0])(jnp.asarray(images))
            expected = jax.jit(lambda x: apply(p, b, x, spec)[0])(jnp.asarray(images))
            np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=1e-5)
            select = ['--scope', 'attn', '--matrices', names[1]]
            for split_name in ('train', 'val'):
                statistics(args(split_name)+select+['--split', split_name, '--samples-per-class', '1', '--batch-size', '2'])
                verified_stage(root/split_name)
            curves(args('bank')+select+['--rank-ratios', '.5', '1', '--calibration-dir', str(root/'train'),
                '--validation-stats-dir', str(root/'val'), '--ridge-relative', '0.0001'])
            allocate(args('allocation')+select+['--rank-ratios', '.5', '--activation-bank', str(root/'bank')])
            candidate(args('allocated')+select+['--rank-ratios', '.5', '--activation-bank', str(root/'bank'),
                '--allocation-dir', str(root/'allocation')])
            candidate(args('candidate')+select+['--rank-ratios', '1', '--activation-bank', str(root/'bank')])
            _, restored, _, meta, _ = load_checkpoint(root/'candidate'/'candidate.npz')
            self.assertEqual(meta['candidate']['method'], 'activation_svd')
            self.assertEqual(restored[names[1]+'.first.weight'].shape, (384, 384))
            np.testing.assert_array_equal(restored[names[1]+'.second.bias'], p[names[1]+'.bias'])
            validate(args('dense')+['--batch-size', '2'])
            validate(args('validation', root/'candidate'/'candidate.npz')+
                     ['--reference-dir', str(root/'dense'), '--batch-size', '2'])
            result = json.loads((root/'validation'/'validation.json').read_text())
            self.assertTrue(result['full_rank_control'])
            self.assertLess(result['logit_relative_frobenius'], 5e-5)
            with self.assertRaises(FileExistsError):
                statistics(args('train')+select+['--split', 'train', '--samples-per-class', '1'])
            # Wrong source statistics and tampered banks must be rejected.
            with self.assertRaisesRegex(ValueError, 'train calibration'):
                curves(args('wrong_split')+select+['--rank-ratios', '.5', '--calibration-dir', str(root/'val'),
                    '--validation-stats-dir', str(root/'val'), '--ridge-relative', '0.0001'])
            (root/'bank'/'diagnostics.json').write_text('{}')
            with self.assertRaisesRegex(ValueError, 'Changed stage output'):
                candidate(args('tampered')+select+['--rank-ratios', '.5', '--activation-bank', str(root/'bank')])


if __name__ == '__main__':
    unittest.main()
