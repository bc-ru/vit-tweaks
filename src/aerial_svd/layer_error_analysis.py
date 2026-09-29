from __future__ import annotations

if __name__ == "__main__":
    from aerial_svd.backend_dispatch import dispatch
    dispatch("layer_error_analysis")

import argparse
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm

from aerial_svd.data import build_loaders
from aerial_svd.matrix_analysis import load_model_from_config, matrix_kind
from aerial_svd.utils import get_device, seed_everything


def parse_args():
    p = argparse.ArgumentParser("Local layer-output error after replacing W by truncated SVD W_r")
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--device", default="auto")
    p.add_argument("--out-dir", default="matrix_results/layer_error")
    p.add_argument("--include-regex", default="mlp.fc|attn.qkv|attn.proj")
    p.add_argument("--exclude-regex", default=None)
    p.add_argument("--rank-ratios", nargs="+", type=float, default=[0.25, 0.50, 0.75])
    p.add_argument("--max-targets", type=int, default=16)
    p.add_argument("--max-batches", type=int, default=4)
    p.add_argument("--max-stored-errors", type=int, default=20000)
    p.add_argument("--amp", action="store_true")
    return p.parse_args()


def selected_linear_modules(model: nn.Module, include_regex: str | None, exclude_regex: str | None, max_targets: int):
    inc = re.compile(include_regex) if include_regex else None
    exc = re.compile(exclude_regex) if exclude_regex else None
    out = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        key = f"{name} {matrix_kind(name, module)}"
        if inc and not inc.search(key):
            continue
        if exc and exc.search(key):
            continue
        out.append((name, module))
        if len(out) >= max_targets:
            break
    return out


def precompute_svd_factors(targets, rank_ratios):
    factors = {}
    matrix_errors = {}
    for name, module in targets:
        W = module.weight.detach().to(torch.float32).cpu()
        U, S, Vh = torch.linalg.svd(W, full_matrices=False)
        total = float(S.double().square().sum().item())
        max_rank = min(W.shape)
        factors[name] = {"U": U, "S": S, "Vh": Vh, "bias": None if module.bias is None else module.bias.detach().to(torch.float32).cpu()}
        for rho in rank_ratios:
            r = max(1, min(max_rank, int(math.floor(float(rho) * max_rank))))
            tail = float(S[r:].double().square().sum().item()) if r < S.numel() else 0.0
            matrix_errors[(name, float(rho))] = {
                "rank": r,
                "matrix_fro_error": math.sqrt(tail / total) if total > 0 else 0.0,
                "matrix_spectral_error": float(S[r].item() / S[0].item()) if r < S.numel() and S[0] > 0 else 0.0,
            }
    return factors, matrix_errors


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load config once through model loader; build dataloader from the resulting checkpoint config.
    device = get_device(args.device)
    model, cfg, ckpt = load_model_from_config(args.config, checkpoint=args.checkpoint, device=str(device))
    model.to(device).eval()
    seed_everything(int(cfg.get("seed", 42)))

    _, _, test_loader, _ = build_loaders(
        data_dir=cfg["data_dir"],
        img_size=int(cfg.get("img_size", 224)),
        batch_size=int(cfg.get("batch_size", 64)),
        eval_batch_size=int(cfg.get("eval_batch_size", cfg.get("batch_size", 64))),
        num_workers=int(cfg.get("num_workers", 2)),
        train_ratio=float(cfg.get("train_ratio", 0.70)),
        val_ratio=float(cfg.get("val_ratio", 0.15)),
        seed=int(cfg.get("seed", 42)),
        split_file=cfg.get("split_file", str(Path(cfg.get("output_dir", "runs/default")) / "split.json")),
    )

    targets = selected_linear_modules(model, args.include_regex, args.exclude_regex, args.max_targets)
    factors, matrix_errors = precompute_svd_factors(targets, args.rank_ratios)

    accum: Dict[tuple, Dict[str, object]] = defaultdict(lambda: {"sum": 0.0, "count": 0, "max": 0.0, "values": []})

    def make_hook(name: str):
        def hook(module, inputs, output):
            x = inputs[0].detach().to(torch.float32)
            y = output.detach().to(torch.float32)
            x2 = x.reshape(-1, x.shape[-1])
            y2 = y.reshape(-1, y.shape[-1])
            f = factors[name]
            U = f["U"].to(x2.device)
            S = f["S"].to(x2.device)
            Vh = f["Vh"].to(x2.device)
            bias = None if f["bias"] is None else f["bias"].to(x2.device)
            denom = torch.linalg.vector_norm(y2, dim=1).clamp_min(1e-12)
            for rho in args.rank_ratios:
                rho = float(rho)
                r = matrix_errors[(name, rho)]["rank"]
                tmp = x2 @ Vh[:r, :].T
                tmp = tmp * S[:r]
                yhat = tmp @ U[:, :r].T
                if bias is not None:
                    yhat = yhat + bias
                rel = torch.linalg.vector_norm(y2 - yhat, dim=1) / denom
                key = (name, rho)
                d = accum[key]
                d["sum"] = float(d["sum"]) + float(rel.sum().item())
                d["count"] = int(d["count"]) + int(rel.numel())
                d["max"] = max(float(d["max"]), float(rel.max().item()))
                vals: List[float] = d["values"]  # type: ignore[assignment]
                room = max(0, args.max_stored_errors - len(vals))
                if room:
                    vals.extend(rel[:room].detach().cpu().tolist())
        return hook

    handles = [module.register_forward_hook(make_hook(name)) for name, module in targets]
    autocast_enabled = args.amp and device.type == "cuda"
    with torch.no_grad():
        for batch_idx, (images, _) in enumerate(tqdm(test_loader, desc="Layer error batches")):
            if batch_idx >= args.max_batches:
                break
            images = images.to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", enabled=autocast_enabled):
                _ = model(images)
    for h in handles:
        h.remove()

    kinds = {name: matrix_kind(name, module) for name, module in targets}
    rows = []
    for (name, rho), d in accum.items():
        vals = torch.tensor(d["values"], dtype=torch.float64) if d["values"] else torch.empty(0, dtype=torch.float64)
        q95 = float(torch.quantile(vals, 0.95).item()) if vals.numel() else float("nan")
        me = matrix_errors[(name, rho)]
        rows.append({
            "model": cfg["model"],
            "checkpoint": ckpt,
            "layer": name,
            "kind": kinds[name],
            "rank_ratio": rho,
            "rank": me["rank"],
            "matrix_fro_error": me["matrix_fro_error"],
            "matrix_spectral_error": me["matrix_spectral_error"],
            "layer_output_error_mean": float(d["sum"]) / max(int(d["count"]), 1),
            "layer_output_error_p95": q95,
            "layer_output_error_max": float(d["max"]),
            "num_vectors": int(d["count"]),
        })
    df = pd.DataFrame(rows)
    path = out_dir / f"{cfg['model']}_layer_error.csv"
    df.to_csv(path, index=False)

    if not df.empty:
        plt.figure(figsize=(8, 5))
        for kind, g in df.groupby("kind"):
            q = g.groupby("rank_ratio", as_index=False).agg(layer_output_error_mean=("layer_output_error_mean", "median"), matrix_fro_error=("matrix_fro_error", "median"))
            plt.plot(q["matrix_fro_error"], q["layer_output_error_mean"], marker="o", label=kind)
        plt.xlabel(r"median $\|W-W_r\|_F / \|W\|_F$")
        plt.ylabel("median relative layer-output error")
        plt.title(f"Matrix error vs layer-output error: {cfg['model']}")
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(out_dir / f"{cfg['model']}_matrix_vs_layer_error.png", dpi=200)
        plt.close()

    print(f"Saved: {path}")
    if not df.empty:
        print(df.groupby(["kind", "rank_ratio"])[["matrix_fro_error", "layer_output_error_mean", "layer_output_error_p95"]].median().to_string())


if __name__ == "__main__":
    main()
