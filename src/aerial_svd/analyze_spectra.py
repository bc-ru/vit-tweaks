from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from aerial_svd.matrix_analysis import (
    dataclass_records_to_dicts,
    filter_matrices,
    iter_weight_matrices,
    load_model_from_config,
    singular_values,
    spectral_stats,
)


def parse_args():
    p = argparse.ArgumentParser("Spectral passport for neural network weight matrices")
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--device", default="cpu")
    p.add_argument("--out-dir", default="matrix_results/spectra")
    p.add_argument("--include-regex", default=None)
    p.add_argument("--exclude-regex", default=None)
    p.add_argument("--eps", type=float, default=1e-10)
    p.add_argument("--max-plotted", type=int, default=40)
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model, cfg, ckpt = load_model_from_config(args.config, checkpoint=args.checkpoint, device=args.device)
    model_name = cfg["model"]

    stats = []
    sv_rows = []
    matrices = filter_matrices(iter_weight_matrices(model), args.include_regex, args.exclude_regex)
    for rec, W in matrices:
        s = singular_values(W)
        stats.append(spectral_stats(model_name, ckpt, rec, W, eps=args.eps))
        denom = float(s[0].item()) if s.numel() else 1.0
        for i, val in enumerate(s[: min(args.max_plotted, s.numel())].tolist(), start=1):
            sv_rows.append({"model": model_name, "name": rec.name, "kind": rec.kind, "k": i, "sigma": val, "sigma_rel": val / max(denom, 1e-30)})

    df = pd.DataFrame(dataclass_records_to_dicts(stats))
    sv = pd.DataFrame(sv_rows)
    df.to_csv(out_dir / f"{model_name}_spectral_stats.csv", index=False)
    sv.to_csv(out_dir / f"{model_name}_singular_values.csv", index=False)

    if not sv.empty:
        plt.figure(figsize=(9, 6))
        for (kind, name), g in sv.groupby(["kind", "name"]):
            if kind in {"ffn_expand", "ffn_contract", "attn_qkv", "attn_proj", "conv"}:
                plt.plot(g["k"], g["sigma_rel"], alpha=0.35, linewidth=1)
        plt.yscale("log")
        plt.xlabel("k")
        plt.ylabel(r"$\sigma_k / \sigma_1$")
        plt.title(f"Singular value decay: {model_name}")
        plt.tight_layout()
        plt.savefig(out_dir / f"{model_name}_singular_decay.png", dpi=200)
        plt.close()

    if not df.empty:
        plt.figure(figsize=(8, 5))
        df.boxplot(column="r95_ratio", by="kind", rot=45)
        plt.suptitle("")
        plt.title(f"Relative rank for 95% spectral energy: {model_name}")
        plt.ylabel(r"$r_{0.95} / r_{max}$")
        plt.tight_layout()
        plt.savefig(out_dir / f"{model_name}_r95_by_kind.png", dpi=200)
        plt.close()

    print(f"Saved: {out_dir}")
    print(df[["name", "kind", "m", "n", "stable_rank", "cond_eps", "r95", "r95_ratio"]].head(20).to_string(index=False))


if __name__ == "__main__":
    main()
