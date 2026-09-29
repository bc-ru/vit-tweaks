"""Export trusted local PyTorch checkpoint matrices for independent JAX analysis."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import torch
import timm

from aerial_svd.matrix_analysis import filter_matrices, iter_weight_matrices, load_model_from_config
from aerial_svd.matrix_archive import save_archive
from aerial_svd.utils import load_yaml


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint")
    p.add_argument("--output", required=True)
    p.add_argument("--include-regex")
    p.add_argument("--exclude-regex")
    p.add_argument("--max-matrices", type=int)
    args = p.parse_args()
    cfg = load_yaml(args.config)
    checkpoint = Path(args.checkpoint or Path(cfg.get("output_dir", "runs/default")) / "best.pt")
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if args.max_matrices is not None and args.max_matrices < 1:
        p.error("--max-matrices must be positive")
    digest = hashlib.sha256()
    with checkpoint.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    model, cfg, ckpt = load_model_from_config(args.config, str(checkpoint), device="cpu")
    payload = torch.load(checkpoint, map_location="cpu")
    state = payload.get("model", payload.get("state_dict", payload)) if isinstance(payload, dict) else payload
    if not isinstance(state, dict) or set(state) != set(model.state_dict()):
        raise ValueError("Export requires a complete dense-model checkpoint; missing/unexpected keys or compressed factors found")
    matrices = list(filter_matrices(iter_weight_matrices(model), args.include_regex, args.exclude_regex))
    if args.max_matrices is not None:
        matrices = matrices[:args.max_matrices]
    save_archive(args.output, ((rec, W.numpy()) for rec, W in matrices), {
        "model": cfg["model"], "checkpoint": ckpt, "checkpoint_sha256": digest.hexdigest(),
        "torch_version": torch.__version__, "timm_version": timm.__version__,
        "config": cfg,
    })
    print(f"Exported {len(matrices)} matrices: {args.output}")


if __name__ == "__main__":
    main()
