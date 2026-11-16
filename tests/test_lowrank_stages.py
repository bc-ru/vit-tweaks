"""CPU-only regression and stage integration tests; no Torch, GPU or AID required."""
import csv
from contextlib import redirect_stdout
import io
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

from aerial_svd.jax_checkpoint import load_checkpoint, save_checkpoint
from aerial_svd.jax_cli import verify_compression
from aerial_svd.jax_compression import compress, factorize_weight
from aerial_svd.jax_matrix_analysis import approximation_records
from aerial_svd.jax_models import apply, initialize
from aerial_svd.lowrank_baseline import main as baseline
from aerial_svd.lowrank_factor_checks import main as factor_checks
from aerial_svd.lowrank_matrix_curves import main as matrix_curves
from aerial_svd.matrix_records import MatrixRecord
from aerial_svd.model_spec import ModelSpec, is_target, linear_names


class Factorization(unittest.TestCase):
    def test_every_matrix_at_saving_equal_and_expanding_ranks(self):
        rng = np.random.default_rng(42)
        names = ["blocks.0.attn.qkv", "blocks.0.attn.proj", "blocks.0.mlp.fc1", "blocks.0.mlp.fc2", "head"]
        params = {name + ".weight": rng.normal(size=shape).astype(np.float32)
                  for name, shape in zip(names, [(12, 4), (4, 4), (16, 4), (4, 16), (3, 4)])}
        params.update({name + ".bias": rng.normal(size=params[name + ".weight"].shape[0]).astype(np.float32)
                       for name in names})
        for scope in ("ffn", "attn", "all_blocks", "all"):
            for ratio in (.25, .5, .75, 1.):
                with self.subTest(scope=scope, ratio=ratio):
                    result, records = compress(params, scope, ratio)
                    selected = [name for name in names if is_target(name, scope)]
                    self.assertEqual({r["name"] for r in records}, set(selected))
                    for name in selected:
                        weight = params[name + ".weight"]
                        rank = max(1, int(ratio * min(weight.shape)))
                        self.assertNotIn(name + ".weight", result)
                        first, second = result[name + ".first.weight"], result[name + ".second.weight"]
                        self.assertEqual(first.shape[0], rank)
                        u, s, vh = np.linalg.svd(weight.astype(np.float64), full_matrices=False)
                        np.testing.assert_allclose(second @ first, (u[:, :rank] * s[:rank]) @ vh[:rank], rtol=5e-5, atol=2e-6)
                        np.testing.assert_array_equal(result[name + ".second.bias"], params[name + ".bias"])
                    for name in set(names) - set(selected):
                        np.testing.assert_array_equal(result[name + ".weight"], params[name + ".weight"])
                    self.assertEqual(set(params), {name + suffix for name in names for suffix in (".weight", ".bias")})

    def test_explicit_rank_and_input_validation(self):
        params = {"blocks.0.attn.proj.weight": np.eye(4, dtype=np.float32)}
        result, rows = compress(params, "attn", ranks={"blocks.0.attn.proj": 4})
        self.assertEqual(rows[0]["compressed_params"], 32)
        for rank in (0, 5, 2.5, True):
            with self.assertRaises(ValueError):
                factorize_weight(np.eye(4), rank)
        for ratio in (0, 2, float("nan")):
            with self.assertRaises(ValueError):
                compress(params, "attn", ratio)
        with self.assertRaises(ValueError):
            compress(params, "ffn", ranks={"blocks.0.attn.proj": 4})
        with self.assertRaises(ValueError):
            compress(result, "attn", 1.)

    def test_analytic_errors_match_direct_norms_and_counts(self):
        for weight in (np.diag(np.array([8, 4, 2, 1], np.float32)), np.zeros((3, 5), np.float32)):
            rec = MatrixRecord("blocks.0.attn.proj", "attn_proj", weight.shape, weight.shape, weight.size)
            records = approximation_records("fixture", "fixture.npz", rec, weight, [.25, .5, .75, 1.])
            for row in records:
                first, second = factorize_weight(weight, row.rank)
                delta = weight - np.asarray(second @ first)
                fro = np.linalg.norm(delta) / np.linalg.norm(weight) if np.any(weight) else 0.
                spectral = np.linalg.norm(delta, 2) / np.linalg.norm(weight, 2) if np.any(weight) else 0.
                self.assertAlmostEqual(row.fro_error, fro, places=6)
                self.assertAlmostEqual(row.spectral_error, spectral, places=6)
                self.assertEqual(row.lowrank_storage, first.size + second.size)


class Stages(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name)
        cls.spec = ModelSpec("vit_small_patch16_224", 2)
        cls.params, cls.buffers = initialize(cls.spec)
        cls.checkpoint = cls.root / "dense.npz"
        save_checkpoint(cls.checkpoint, cls.spec, cls.params, cls.buffers,
                        metadata={"config": {"classes": ["A", "B"]}})
        data = cls.root / "images"
        for label in ("A", "B"):
            (data / label).mkdir(parents=True)
            for i in range(3):
                Image.new("RGB", (32, 32), (42 + i, 80, 125)).save(data / label / f"{i}.png")
        cls.split = cls.root / "split.json"
        cls.split.write_text(json.dumps(dict(class_to_idx={"A": 0, "B": 1},
                                              indices={"train": [0, 3], "val": [1, 4], "test": [2, 5]})))
        cls.config = cls.root / "config.yaml"
        cls.config.write_text(yaml.safe_dump(dict(model=cls.spec.model, img_size=224, seed=42, num_workers=0,
                                                data_dir=str(data), split_file=str(cls.split))))

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def arguments(self, output):
        return ["--config", str(self.config), "--checkpoint", str(self.checkpoint),
                "--output-dir", str(self.root / output)]

    def test_matrix_stage_no_dataset_no_model_execution_and_no_overwrite(self):
        args = self.arguments("curves") + ["--matrices", "blocks.0.attn.proj", "--ranks", "192", "384"]
        with patch("aerial_svd.jax_data.ImageFolder", side_effect=AssertionError("dataset forbidden")), \
                patch("aerial_svd.jax_models.apply", side_effect=AssertionError("model forbidden")):
            matrix_curves(args)
        out = self.root / "curves"
        manifest = json.loads((out / "manifest.json").read_text())
        self.assertEqual(manifest["status"], "complete")
        with (out / "matrix_curves.csv").open() as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual([int(row["actual_rank"]) for row in rows], [192, 384])
        self.assertEqual(float(rows[-1]["relative_frobenius_error"]), 0)
        self.assertLess(int(rows[-1]["coefficient_savings"]), 0)
        original = (out / "matrix_curves.csv").read_bytes()
        with self.assertRaises(FileExistsError):
            matrix_curves(args)
        self.assertEqual((out / "matrix_curves.csv").read_bytes(), original)

    def test_factor_stage_full_rank_and_serialization(self):
        factor_checks(self.arguments("factors") + ["--matrices", "blocks.0.attn.proj", "--rank-ratios", ".75"])
        out = self.root / "factors"
        report = json.loads((out / "factor_checks.json").read_text())
        self.assertTrue(report["serialization_verified"])
        self.assertEqual([row["actual_rank"] for row in report["checks"]], [288, 384])
        self.assertLess(report["checks"][-1]["full_rank_relative_error"], 5e-5)
        self.assertEqual(json.loads((out / "manifest.json").read_text())["status"], "complete")

    def test_baseline_validation_only_deterministic_and_partial_label(self):
        from aerial_svd.jax_data import ImageFolder
        original = ImageFolder.batches
        def validation_only(dataset, split, *args, **kwargs):
            self.assertEqual(split, "val")
            return original(dataset, split, *args, **kwargs)
        split_bytes = self.split.read_bytes()
        with patch.object(ImageFolder, "batches", validation_only):
            baseline(self.arguments("baseline") + ["--batch-size", "1", "--max-val-batches", "1"])
        report = json.loads((self.root / "baseline/baseline.json").read_text())
        self.assertEqual(report["validation"]["num_samples"], 1)
        self.assertTrue(report["deterministic_repeat"])
        self.assertEqual(report["quality_verification"], "partial_smoke_only")
        self.assertFalse(report["reference_verified"])
        self.assertEqual(self.split.read_bytes(), split_bytes)

    def test_native_checkpoint_roundtrip_full_rank_logits_and_verifier(self):
        # Full-rank attention output projection has twice the dense coefficient count.
        ranks = {name: min(self.params[name + ".weight"].shape) for name in linear_names(self.params)
                 if is_target(name, "attn")}
        candidate, records = compress(self.params, "all_blocks", 1.)
        self.assertEqual(len(records), 48)
        self.assertIn("head.weight", candidate)
        verify_compression(self.spec, candidate, "all_blocks", 1.)
        path = self.root / "roundtrip.npz"
        save_checkpoint(path, self.spec, candidate, self.buffers)
        _, loaded, buffers, _, _ = load_checkpoint(path)
        for key in candidate:
            np.testing.assert_array_equal(loaded[key], candidate[key])
        with self.assertRaises(FileExistsError):
            save_checkpoint(path, self.spec, candidate, self.buffers)
        image = jnp.asarray(np.random.default_rng(42).normal(size=(1, 3, 224, 224)).astype(np.float32))
        forward = jax.jit(lambda p, b: apply(p, b, image, self.spec)[0])
        expected = np.asarray(forward(self.params, self.buffers))
        actual = np.asarray(forward(loaded, buffers))
        np.testing.assert_allclose(actual, expected, rtol=5e-5, atol=2e-6)
        # Schema-only factors suffice for checking the verifier's expected list/ranks.
        structural = dict(self.params)
        for name, rank in ranks.items():
            m, n = structural.pop(name + ".weight").shape
            structural[name + ".first.weight"] = np.zeros((rank, n), np.float32)
            structural[name + ".second.weight"] = np.zeros((m, rank), np.float32)
            structural[name + ".second.bias"] = structural.pop(name + ".bias")
        verify_compression(self.spec, structural, "attn", 1.)
        with self.assertRaises(ValueError):
            verify_compression(self.spec, self.params, "attn", 1.)
        verify_compression(self.spec, self.params, "attn", 1., legacy_skip_non_saving=True)
        with self.assertRaises(ValueError):
            verify_compression(self.spec, structural, "attn", 1., legacy_skip_non_saving=True)

    def test_invalid_selection_rank_and_failed_manifest(self):
        for extra in (["--matrices", "head", "--ranks", "1"],
                      ["--matrices", "blocks.0.attn.proj", "--ranks", "385"],
                      ["--rank-ratios", "nan"]):
            with self.assertRaises(ValueError):
                matrix_curves(self.arguments("invalid") + extra)
        from aerial_svd.lowrank_common import stage_output
        out = self.root / "failed"
        with self.assertRaises(RuntimeError), stage_output(out, {"stage": "fixture"}):
            raise RuntimeError("interrupted computation")
        self.assertEqual(json.loads((out / "manifest.json").read_text())["status"], "failed")

    def test_legacy_evaluation_explicitly_labels_historical_policy(self):
        from aerial_svd.jax_cli import main
        output = self.root / "legacy.json"
        with redirect_stdout(io.StringIO()):
            main("evaluate", ["--config", str(self.config), "--checkpoint", str(self.checkpoint),
                              "--output", str(output), "--svd-mode", "all_blocks", "--rank-ratio", "1",
                              "--legacy-skip-non-saving", "--batch-size", "1", "--max-test-batches", "1",
                              "--warmup-batches", "0", "--max-batches", "1"])
        self.assertEqual(json.loads(output.read_text())["factorization_policy"], "legacy_skip_non_saving")


if __name__ == "__main__":
    unittest.main()
