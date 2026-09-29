"""Run in the reference container with JAX installed; no datasets/GPU required."""
import dataclasses
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from aerial_svd import jax_matrix_analysis as candidate
from aerial_svd import matrix_analysis as reference
from aerial_svd.matrix_archive import load_archive, save_archive
from aerial_svd.matrix_records import MatrixRecord


def record(W):
    return MatrixRecord("blocks.0.mlp.fc1", "ffn_expand", W.shape, W.shape, W.size)


class NumericalParity(unittest.TestCase):
    def matrices(self):
        rng = np.random.default_rng(42)
        return [np.zeros((8, 5), np.float32),
                np.diag(np.array([8, 4, 2, 0, 0], np.float32)),
                rng.normal(size=(32, 12)).astype(np.float32),
                rng.normal(size=(12, 32)).astype(np.float32)]

    def assert_records_close(self, got, expected):
        got, expected = dataclasses.asdict(got), dataclasses.asdict(expected)
        self.assertEqual(got.keys(), expected.keys())
        for key in got:
            if isinstance(expected[key], float):
                np.testing.assert_allclose(got[key], expected[key], rtol=3e-5, atol=2e-6, err_msg=key)
            else:
                self.assertEqual(got[key], expected[key], key)

    def test_spectra_and_stats(self):
        for W in self.matrices():
            with self.subTest(shape=W.shape, norm=np.linalg.norm(W)):
                tw = torch.from_numpy(W)
                np.testing.assert_allclose(candidate.singular_values(W), reference.singular_values(tw), rtol=3e-5, atol=2e-6)
                self.assert_records_close(candidate.spectral_stats("m", "c", record(W), W),
                                          reference.spectral_stats("m", "c", record(W), tw))

    def test_analytic_approximation_and_rounding(self):
        ratios = [0.05, 0.25, 0.5, 0.75, 1.0]
        for W in self.matrices():
            got = candidate.approximation_records("m", "c", record(W), W, ratios)
            expected = reference.approximation_records("m", "c", record(W), torch.from_numpy(W), ratios)
            for a, b in zip(got, expected):
                self.assert_records_close(a, b)

    def test_randomized_svd_shared_projection(self):
        W = self.matrices()[2]
        for iterations in (0, 1, 2):
            rank, oversampling = 4, 3
            omega = torch.randn(W.shape[1], rank + oversampling, generator=torch.Generator().manual_seed(42))
            u, s, vh = candidate.randomized_svd(W, rank, oversampling, iterations, omega=omega.numpy())
            tu, ts, tvh = reference.randomized_svd(torch.from_numpy(W), rank, oversampling, iterations)
            np.testing.assert_allclose(s, ts.numpy(), rtol=3e-5, atol=2e-6)
            np.testing.assert_allclose((np.asarray(u) * np.asarray(s)) @ np.asarray(vh),
                                      ((tu * ts) @ tvh).numpy(), rtol=1e-4, atol=3e-5)

    def test_jax_seed_repeatability_and_validation(self):
        W = self.matrices()[2]
        a = candidate.randomized_svd(W, 4)
        b = candidate.randomized_svd(W, 4)
        for x, y in zip(a, b):
            np.testing.assert_array_equal(x, y)
        for rank in (0, 100):
            with self.assertRaises(ValueError):
                candidate.randomized_svd(W, rank)
        with self.assertRaises(ValueError):
            candidate.randomized_svd(W, 4, omega=np.zeros((1, 1)))

    def test_benchmark_synchronizes_and_returns_schema(self):
        W = self.matrices()[2]
        result = candidate.benchmark_randomized_svd("m", "c", record(W), W, 4)
        self.assertEqual(set(dataclasses.asdict(result)), set(reference.RandomizedSVDRecord.__dataclass_fields__))
        self.assertGreater(result.full_svd_time_sec, 0)
        self.assertGreater(result.randomized_svd_time_sec, 0)
        self.assertGreaterEqual(result.randomized_fro_error + 1e-5, result.approximation_fro_error)


class ArchiveContract(unittest.TestCase):
    def test_roundtrip_and_no_overwrite(self):
        W = np.arange(12, dtype=np.float32).reshape(3, 4)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "matrices.npz"
            save_archive(path, [(record(W), W)], {"model": "test", "checkpoint": "fixture"})
            original = path.read_bytes()
            matrices, provenance = load_archive(path)
            self.assertEqual(matrices[0][0], record(W))
            np.testing.assert_array_equal(matrices[0][1], W)
            self.assertEqual(provenance["model"], "test")
            with self.assertRaises(FileExistsError):
                save_archive(path, [(record(W), W)], {})
            self.assertEqual(path.read_bytes(), original)
            link = Path(tmp) / "dangling.npz"
            link.symlink_to(Path(tmp) / "missing")
            with self.assertRaises(FileExistsError):
                save_archive(link, [(record(W), W)], {})
            self.assertTrue(link.is_symlink())
            self.assertEqual(sorted(p.name for p in Path(tmp).iterdir()), ["dangling.npz", "matrices.npz"])

    def test_invalid_archive_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.npz"
            np.savez(path, metadata=np.asarray(json.dumps({"version": 999})))
            with self.assertRaises(ValueError):
                load_archive(path)
            W = np.array([[np.nan]], dtype=np.float32)
            with self.assertRaises(ValueError):
                save_archive(Path(tmp) / "invalid.npz", [(record(W), W)], {})


if __name__ == "__main__":
    unittest.main()
