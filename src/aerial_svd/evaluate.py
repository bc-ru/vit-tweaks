from __future__ import annotations

if __name__ == "__main__":
    from aerial_svd.backend_dispatch import dispatch
    dispatch("evaluate")

import argparse
import os
from pathlib import Path

import torch
import torch.nn as nn

from aerial_svd.data import build_loaders
from aerial_svd.engine import benchmark_inference, evaluate_accuracy
from aerial_svd.models import create_model
from aerial_svd.svd import compress_model_inplace
from aerial_svd.utils import count_total_params, count_trainable_params, get_device, load_yaml, save_json, seed_everything, sizeof_params_bytes


def parse_args():
    p = argparse.ArgumentParser("Evaluate accuracy and inference metrics for baseline or compressed checkpoint")
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--data-dir", type=str, default=None)
    p.add_argument("--svd-mode", type=str, choices=["ffn", "attn", "all"], default=None, help="Rebuild low-rank structure before loading compressed checkpoint")
    p.add_argument("--rank-ratio", type=float, default=None)
    p.add_argument("--batch-size", type=int, default=None, help="Override eval batch size")
    p.add_argument("--max-batches", type=int, default=50)
    p.add_argument("--output", type=str, required=True)
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_yaml(args.config)
    if args.data_dir:
        cfg["data_dir"] = args.data_dir
    seed_everything(int(cfg.get("seed", 42)))
    device = get_device(cfg.get("device", "auto"))
    eval_bs = args.batch_size or int(cfg.get("eval_batch_size", cfg.get("batch_size", 64)))

    train_loader, val_loader, test_loader, classes = build_loaders(
        data_dir=cfg["data_dir"],
        img_size=int(cfg.get("img_size", 224)),
        batch_size=int(cfg.get("batch_size", 64)),
        eval_batch_size=eval_bs,
        num_workers=int(cfg.get("num_workers", 4)),
        train_ratio=float(cfg.get("train_ratio", 0.70)),
        val_ratio=float(cfg.get("val_ratio", 0.15)),
        seed=int(cfg.get("seed", 42)),
        split_file=cfg.get("split_file"),
    )
    num_classes = len(classes)
    model = create_model(cfg["model"], num_classes=num_classes, pretrained=False, checkpoint=None)
    if args.svd_mode is not None:
        if args.rank_ratio is None:
            raise ValueError("--rank-ratio is required with --svd-mode")
        compress_model_inplace(model, args.svd_mode, args.rank_ratio, verbose=True)
    state = torch.load(args.checkpoint, map_location="cpu")
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    elif isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"Loaded {args.checkpoint}; missing={len(missing)}, unexpected={len(unexpected)}")
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    acc = evaluate_accuracy(model, test_loader, criterion, device, amp=bool(cfg.get("amp", True)))
    bench = benchmark_inference(model, test_loader, device, amp=bool(cfg.get("amp", True)), max_batches=args.max_batches)
    out = {
        "model": cfg["model"],
        "checkpoint": args.checkpoint,
        "svd_mode": args.svd_mode,
        "rank_ratio": args.rank_ratio,
        "num_classes": num_classes,
        "test_loss": acc.loss,
        "test_top1": acc.top1,
        "test_samples": acc.num_samples,
        "total_params": count_total_params(model),
        "trainable_params": count_trainable_params(model),
        "param_memory_bytes": sizeof_params_bytes(model),
        "checkpoint_bytes": os.path.getsize(args.checkpoint) if os.path.exists(args.checkpoint) else None,
        **bench,
    }
    save_json(out, args.output)
    print(out)


if __name__ == "__main__":
    main()
