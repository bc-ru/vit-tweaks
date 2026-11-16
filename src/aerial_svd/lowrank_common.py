"""Input identity and immutable outputs shared by the independent research stages."""
import argparse
from contextlib import contextmanager
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import tempfile

import jax
import numpy as np
import yaml

from aerial_svd.jax_checkpoint import load_checkpoint
from aerial_svd.model_spec import is_target, linear_names


def file_identity(path):
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return dict(path=str(path.resolve()), sha256=digest.hexdigest(), bytes=path.stat().st_size)


def publish_text(path, content, *, replace_owned=False):
    """Publish complete bytes; only a stage's own running manifest may be replaced."""
    path = Path(path)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        try:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
            if replace_owned:
                temporary.replace(path)
            else:
                os.link(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)


def publish_json(path, value, **kwargs):
    publish_text(path, json.dumps(value, indent=2, allow_nan=False) + "\n", **kwargs)


@contextmanager
def stage_output(path, provenance):
    out = Path(path)
    out.mkdir(parents=True, exist_ok=False)
    manifest = out / "manifest.json"
    publish_json(manifest, {**provenance, "status": "running"})
    try:
        yield out
        outputs = {p.name: file_identity(p) for p in sorted(out.iterdir()) if p != manifest}
        publish_json(manifest, {**provenance, "outputs": outputs, "status": "complete"}, replace_owned=True)
    except Exception as error:
        publish_json(manifest, {**provenance, "status": "failed", "error": str(error)}, replace_owned=True)
        raise


def parser(stage, *, matrices=False):
    result = argparse.ArgumentParser(description=f"JAX low-rank stage {stage}")
    result.add_argument("--config", required=True)
    result.add_argument("--checkpoint", required=True, help="Dense native JAX NPZ, never a .pt file")
    result.add_argument("--output-dir", required=True, help="New stage directory; existing paths are refused")
    result.add_argument("--device", choices=("cpu", "gpu"), default="cpu")
    if matrices:
        result.add_argument("--scope", choices=("ffn", "attn", "all_blocks", "all"), default="all_blocks")
        result.add_argument("--matrices", nargs="+", help="Exact module names, without .weight; defaults to scope")
        rank = result.add_mutually_exclusive_group(required=True)
        rank.add_argument("--rank-ratios", type=float, nargs="+")
        rank.add_argument("--ranks", type=int, nargs="+", help="Integer ranks applied to every selected matrix")
    return result


def load_inputs(args, stage):
    # CPU is the safe default. Explicit GPU requests still require a free card.
    if args.device == "gpu":
        check = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
                               capture_output=True, text=True, check=True)
        if check.stdout.strip():
            raise RuntimeError("GPU is busy; no experiment started")
    device = jax.devices(args.device)[0]
    jax.config.update("jax_default_device", device)
    jax.config.update("jax_default_matmul_precision", "highest")
    with open(args.config) as stream:
        cfg = yaml.safe_load(stream)
    if int(cfg.get("seed", 42)) != 42:
        raise ValueError("This research protocol requires seed 42")
    spec, params, buffers, metadata, _ = load_checkpoint(args.checkpoint)
    if spec.model != cfg["model"] or spec.img_size != int(cfg.get("img_size", 224)):
        raise ValueError("Checkpoint architecture disagrees with config")
    if any(key.endswith(".first.weight") for key in params):
        raise ValueError("Research stages 00–02 require dense source weights")
    saved = metadata.get("config", {})
    for key in ("model", "img_size", "seed", "classes", "num_classes"):
        if key in cfg and key in saved and cfg[key] != saved[key]:
            raise ValueError(f"Checkpoint configuration disagrees on {key}")
    root = Path(__file__).resolve().parents[2]
    if shutil.which("git"):
        git = ["git", "-c", f"safe.directory={root}"]
        commit = subprocess.check_output([*git, "rev-parse", "HEAD"], cwd=root, text=True).strip()
        dirty = bool(subprocess.check_output([*git, "status", "--porcelain"], cwd=root, text=True).strip())
        commit_source = "git"
    else:
        # The standard CPU runtime intentionally has no Git binary. Host supplies
        # the revision; actual source hashes still identify any uncommitted code.
        commit = os.environ.get("AERIAL_SVD_GIT_COMMIT", "")
        if not re.fullmatch(r"[0-9a-f]{40}", commit):
            raise ValueError("Git unavailable: pass -e AERIAL_SVD_GIT_COMMIT=$(git rev-parse HEAD) to docker run")
        dirty, commit_source = None, "AERIAL_SVD_GIT_COMMIT; working tree state unknown"
    source_hashes = {p.name: file_identity(p)["sha256"] for p in sorted(Path(__file__).parent.glob("*.py"))}
    provenance = dict(stage=stage, backend="jax", git_commit=commit, git_dirty=dirty, git_commit_source=commit_source,
                      source_files_sha256=source_hashes, model_spec=asdict(spec),
                      checkpoint=file_identity(args.checkpoint), config=file_identity(args.config),
                      source_metadata=metadata, effective_config=cfg, arguments=vars(args),
                      method="exact_svd", factorization_policy="all_selected_v1", dtype="float32",
                      environment=dict(jax=jax.__version__, numpy=np.__version__, python=platform.python_version(),
                                       device=str(device), platform=device.platform))
    # Matrix stages record split identity, but never load images or require the dataset.
    provenance["aid_split"] = file_identity(cfg["split_file"])
    return cfg, spec, params, buffers, metadata, provenance


def selected_names(params, args):
    eligible = [name for name in linear_names(params) if is_target(name, args.scope)]
    names = args.matrices if args.matrices is not None else eligible
    if not names or len(set(names)) != len(names) or any(name not in eligible for name in names):
        raise ValueError("Matrix selection is empty, duplicated, unknown, or outside scope")
    return names


def rank_requests(shape, args):
    limit = min(shape)
    if args.rank_ratios is not None:
        if any(not math.isfinite(rho) or not 0 < rho <= 1 for rho in args.rank_ratios):
            raise ValueError("Rank ratios must be in (0, 1]")
        return [dict(requested_rank=max(1, math.floor(rho * limit)), requested_ratio=rho,
                     actual_rank=max(1, math.floor(rho * limit))) for rho in args.rank_ratios]
    if any(not 1 <= rank <= limit for rank in args.ranks):
        raise ValueError(f"Integer ranks must be in [1, {limit}]")
    return [dict(requested_rank=rank, requested_ratio=None, actual_rank=rank) for rank in args.ranks]
