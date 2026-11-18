"""Dependency-free verification of the host/container ownership boundary."""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


class Workspace(unittest.TestCase):
    def run_profile(self, profile, *, process_list="", driver_ok=True):
        with tempfile.TemporaryDirectory() as directory:
            tmp = Path(directory)
            docker = tmp / "docker"
            docker.write_text("#!/usr/bin/env python3\nimport json,os,sys\n"
                              "open(os.environ['DOCKER_TEST_LOG'],'w').write(json.dumps(sys.argv[1:]))\n")
            docker.chmod(0o755)
            nvidia = tmp / "nvidia-smi"
            nvidia.write_text("#!/usr/bin/env python3\nimport os,sys\n"
                              "print(os.environ['GPU_TEST_PROCESSES'],end='')\n"
                              "sys.exit(int(os.environ['GPU_TEST_EXIT']))\n")
            nvidia.chmod(0o755)
            log = tmp / "docker.json"
            environment = dict(os.environ, PATH=str(tmp) + os.pathsep + os.environ['PATH'],
                               DOCKER_TEST_LOG=str(log), GPU_TEST_PROCESSES=process_list,
                               GPU_TEST_EXIT="0" if driver_ok else "1")
            result = subprocess.run([str(ROOT / "scripts/run.sh"), profile, "python", "-c", "print('a b;$x')"],
                                    cwd=ROOT / "tests", env=environment, text=True, capture_output=True)
            return result, json.loads(log.read_text()) if log.exists() else None

    def test_cpu_preserves_uid_arguments_and_workspace_from_any_directory(self):
        result, args = self.run_profile("cpu")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(args[args.index("--user") + 1], f"{os.getuid()}:{os.getgid()}")
        self.assertEqual(args[args.index("--volume") + 1], f"{ROOT}:/workspace")
        self.assertEqual(args[args.index("--pull") + 1], "never")
        self.assertEqual(args[-4:], ["aerial-svd-jax:cpu", "python", "-c", "print('a b;$x')"])
        self.assertNotIn("--gpus", args)

    def test_busy_or_inaccessible_gpu_never_starts_container(self):
        for parameters in ({"process_list": "123\n"}, {"driver_ok": False}):
            with self.subTest(parameters=parameters):
                result, args = self.run_profile("gpu", **parameters)
                self.assertNotEqual(result.returncode, 0)
                self.assertIsNone(args)

    def test_disk_floor_stops_before_docker(self):
        for floor in ('999999999999999999', 'invalid'):
            with self.subTest(floor=floor), patch.dict(os.environ, AERIAL_SVD_MIN_FREE_BYTES=floor):
                result, args = self.run_profile("cpu")
                self.assertNotEqual(result.returncode, 0)
                self.assertIsNone(args)

    def test_gpu_profile_and_explicit_reference_runtime(self):
        result, args = self.run_profile("gpu")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--gpus", args)
        self.assertEqual(args[-4], "aerial-svd-jax:cuda12")
        result, args = self.run_profile("torch")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("ghcr.io/bc-ru/pytorch-2.1.0-cuda11.8-cudnn8-runtime-timm-models:stable", args)
        result, args = self.run_profile("unknown")
        self.assertEqual(result.returncode, 2)
        self.assertIsNone(args)

    def test_devcontainers_use_matching_nonroot_user_and_no_embedded_credentials(self):
        for filename in (".devcontainer/devcontainer.json", ".devcontainer/jax/devcontainer.json"):
            data = json.loads((ROOT / filename).read_text())
            self.assertEqual(data["remoteUser"], "research")
            self.assertTrue(data["updateRemoteUserUID"])
            self.assertNotIn("postCreateCommand", data)
            self.assertNotIn("mounts", data)
            dockerfile = ROOT / Path(filename).parent / data["build"]["dockerfile"]
            self.assertIn("USER research", dockerfile.read_text())

    def test_existing_artifact_paths_resolve_into_one_tree(self):
        for name in ("runs", "checkpoints", "results", "matrix_results", "plots", "reports", "logs", "experiments"):
            path = ROOT / name
            self.assertTrue(path.is_symlink(), name)
            self.assertEqual(path.resolve(), ROOT / "artifacts" / name)
