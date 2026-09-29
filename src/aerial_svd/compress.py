from __future__ import annotations

if __name__ == "__main__":
    from aerial_svd.backend_dispatch import dispatch
    dispatch("compress")

import argparse
from pathlib import Path

import torch

from aerial_svd.models import create_model, save_checkpoint
from aerial_svd.svd import compress_model_inplace, records_to_dicts
from aerial_svd.utils import count_total_params, get_device, load_yaml, save_json, seed_everything, sizeof_params_bytes


def parse_args():
    p = argparse.ArgumentParser("Apply post-hoc SVD compression to ViT Linear layers")
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--mode", type=str, choices=["ffn", "attn", "all"], required=True)
    p.add_argument("--rank-ratio", type=float, required=True)
    p.add_argument("--output", type=str, required=True)
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_yaml(args.config)
    seed_everything(int(cfg.get("seed", 42)))
    device = get_device(cfg.get("device", "auto"))
    num_classes = int(cfg.get("num_classes", 30))
    model = create_model(cfg["model"], num_classes=num_classes, pretrained=False, checkpoint=args.checkpoint)
    model.to(device)
    before = count_total_params(model)
    before_bytes = sizeof_params_bytes(model)
    records = compress_model_inplace(model, args.mode, args.rank_ratio, verbose=True)
    model.to(device)
    after = count_total_params(model)
    after_bytes = sizeof_params_bytes(model)
    save_checkpoint(args.output, model, epoch=0, metrics={"compression_mode": args.mode, "rank_ratio": args.rank_ratio}, config=cfg)
    metrics = {
        "model": cfg["model"],
        "source_checkpoint": args.checkpoint,
        "compressed_checkpoint": args.output,
        "mode": args.mode,
        "rank_ratio": args.rank_ratio,
        "params_before": before,
        "params_after": after,
        "param_ratio": after / max(before, 1),
        "reduction_percent": 100.0 * (1.0 - after / max(before, 1)),
        "param_memory_before_bytes": before_bytes,
        "param_memory_after_bytes": after_bytes,
        "records": records_to_dicts(records),
    }
    save_json(metrics, Path(args.output).with_suffix(".compression.json"))
    print(metrics)


if __name__ == "__main__":
    main()
