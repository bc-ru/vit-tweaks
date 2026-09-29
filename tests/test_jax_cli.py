"""End-to-end checks runnable in the Torch-free JAX image."""
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image
import yaml

from aerial_svd.matrix_archive import save_archive
from aerial_svd.matrix_records import MatrixRecord, SpectralRecord, ApproximationRecord, RandomizedSVDRecord


class JaxCLI(unittest.TestCase):
    def test_native_training_and_evaluation_without_torch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = root / "images"
            for i, name in enumerate(("A", "B")):
                (data / name).mkdir(parents=True)
                for j in range(2):
                    pixels = np.random.default_rng(42 + i*2 + j).integers(0, 256, (40,40,3), dtype=np.uint8)
                    Image.fromarray(pixels).save(data / name / f"{j}.png")
            split = root / "split.json"
            split.write_text(json.dumps({"class_to_idx": {"A": 0, "B": 1}, "indices": {"train": [0,2], "val": [1], "test": [3]}}))
            config = root / "config.yaml"
            config.write_text(yaml.safe_dump(dict(model="resnet18", pretrained=False, seed=42, device="cpu", img_size=32,
                data_dir=str(data), split_file=str(split), output_dir=str(root / "run"), batch_size=2, eval_batch_size=1,
                epochs=1, num_workers=0, amp=False)))
            for module, arguments in (
                ("train", ["--max-train-batches", "1", "--max-val-batches", "1", "--max-test-batches", "1"]),
                ("evaluate", ["--checkpoint", str(root / "run/last.npz"), "--output", str(root / "evaluation.json"),
                              "--max-test-batches", "1", "--warmup-batches", "0", "--max-batches", "1"]),
            ):
                code = ("import runpy,sys\ntry:\n"
                        f" runpy.run_module('aerial_svd.{module}', run_name='__main__')\n"
                        "except SystemExit as e:\n if e.code not in (None,0): raise\n"
                        "assert 'torch' not in sys.modules\nassert 'timm' not in sys.modules\n")
                result = subprocess.run([sys.executable, "-c", code, "--backend", "jax", "--config", str(config), *arguments],
                                        capture_output=True, text=True, timeout=120)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertTrue((root / "run/complete.json").is_file())
            self.assertTrue((root / "run/best.npz").is_file())
            self.assertEqual(json.loads((root / "evaluation.json").read_text())["test_samples"], 1)

    def test_commands_and_artifact_isolation(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            W = np.diag(np.array([8, 4, 2, 1], dtype=np.float32))
            rec = MatrixRecord("blocks.0.mlp.fc1", "ffn_expand", W.shape, W.shape, W.size)
            archive = tmp / "source.npz"
            save_archive(archive, [(rec, W)], {"model": "fixture", "checkpoint": "fixture.pt"})
            for module, suffix, schema in (
                ("analyze_spectra", "spectral_stats", SpectralRecord),
                ("approximation_curves", "approximation_curves", ApproximationRecord),
                ("randomized_svd_benchmark", "randomized_svd_benchmark", RandomizedSVDRecord),
            ):
                with self.subTest(module=module):
                    out = tmp / module
                    code = (f"import runpy, sys; runpy.run_module('aerial_svd.{module}', run_name='__main__'); "
                            "assert 'torch' not in sys.modules; assert 'timm' not in sys.modules")
                    cmd = [sys.executable, "-c", code, "--backend", "jax", "--matrix-archive", str(archive), "--out-dir", str(out)]
                    env = dict(os.environ, MPLBACKEND="Agg")
                    run = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=60)
                    self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
                    metadata = json.loads((out / "provenance.json").read_text())
                    self.assertEqual(metadata["status"], "complete")
                    self.assertEqual(metadata["backend"], "jax")
                    result = out / f"fixture_{suffix}.csv"
                    original = result.read_bytes()
                    with result.open() as f:
                        rows = csv.DictReader(f)
                        self.assertEqual(rows.fieldnames, list(schema.__dataclass_fields__))
                        self.assertGreater(len(list(rows)), 0)
                    if module != "randomized_svd_benchmark":
                        self.assertGreater(len(list(out.glob("*.png"))), 0)
                    retry = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=60)
                    self.assertNotEqual(retry.returncode, 0)
                    self.assertIn("FileExistsError", retry.stderr)
                    self.assertEqual(result.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
