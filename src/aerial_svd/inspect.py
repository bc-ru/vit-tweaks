from __future__ import annotations

if __name__ == "__main__":
    from aerial_svd.backend_dispatch import dispatch
    dispatch("inspect")

import argparse
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn as nn

from aerial_svd.models import create_model
from aerial_svd.svd import find_linear_targets, dense_params, lowrank_params, rank_from_ratio
from aerial_svd.utils import count_total_params, get_device, human_bytes, load_yaml, save_json, seed_everything, sizeof_params_bytes


from aerial_svd.parameter_groups import classify_param

def parse_args():
    p = argparse.ArgumentParser("Inspect timm model module tree and compressible parameter groups")
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--output", type=str, default=None)
    p.add_argument("--rank-ratio", type=float, default=0.5)
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_yaml(args.config)
    seed_everything(int(cfg.get("seed", 42)))
    device = get_device(cfg.get("device", "auto"))
    num_classes = int(cfg.get("num_classes", 30))
    model = create_model(cfg["model"], num_classes=num_classes, pretrained=bool(cfg.get("pretrained", True)))
    model.to(device)

    print("=" * 100)
    print("MODEL SUMMARY")
    print("=" * 100)
    print(f"model: {cfg['model']}")
    print(f"num_classes: {num_classes}")
    print(f"total params: {count_total_params(model):,}")
    print(f"param memory: {human_bytes(sizeof_params_bytes(model))}")

    group_counts = defaultdict(int)
    tensor_rows = []
    for name, p in model.named_parameters():
        group = classify_param(name)
        group_counts[group] += p.numel()
        tensor_rows.append({"name": name, "shape": list(p.shape), "params": p.numel(), "group": group, "trainable": p.requires_grad})

    print("\nPARAMETER GROUPS")
    for group, n in sorted(group_counts.items(), key=lambda kv: kv[1], reverse=True):
        print(f"{group:24s} {n:12,d} {100.0 * n / max(count_total_params(model), 1):6.2f}%")

    print("\nLOW-RANK CANDIDATES")
    candidate_rows = []
    for mode in ["ffn", "attn", "all"]:
        for name in find_linear_targets(model, mode):
            layer = dict(model.named_modules())[name]
            if not isinstance(layer, nn.Linear):
                continue
            r = rank_from_ratio(layer, args.rank_ratio)
            orig = dense_params(layer)
            comp = lowrank_params(layer, r)
            candidate_rows.append({
                "mode": mode,
                "name": name,
                "in_features": layer.in_features,
                "out_features": layer.out_features,
                "rank": r,
                "dense_params": orig,
                "lowrank_params": comp,
                "saving_ok": comp < orig,
            })
            print(f"{mode:4s} {name:60s} {layer.in_features:5d}->{layer.out_features:<5d} r={r:<4d} {orig:10d}->{comp:<10d} ok={comp < orig}")

    out = {
        "model": cfg["model"],
        "num_classes": num_classes,
        "total_params": count_total_params(model),
        "param_memory_bytes": sizeof_params_bytes(model),
        "parameter_groups": dict(group_counts),
        "tensors": tensor_rows,
        "candidates": candidate_rows,
    }
    if args.output:
        save_json(out, args.output)
        print(f"Saved inspection JSON to {args.output}")


if __name__ == "__main__":
    main()
