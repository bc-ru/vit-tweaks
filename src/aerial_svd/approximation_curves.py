from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from aerial_svd.analysis_backend import add_backend_arguments, prepare_analysis, finish_analysis
from aerial_svd.matrix_records import dataclass_records_to_dicts


def parse_args():
    p = argparse.ArgumentParser("Low-rank approximation curves for model weight matrices")
    add_backend_arguments(p)
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--device", default="cpu")
    p.add_argument("--out-dir", default="matrix_results/approximation")
    p.add_argument("--include-regex", default=None)
    p.add_argument("--exclude-regex", default=None)
    p.add_argument("--rank-ratios", nargs="+", type=float, default=[0.05, 0.10, 0.20, 0.25, 0.50, 0.75])
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    backend, model_name, ckpt, matrices = prepare_analysis(args)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for rec, W in matrices:
        rows.extend(backend.approximation_records(model_name, ckpt, rec, W, args.rank_ratios))
    df = pd.DataFrame(dataclass_records_to_dicts(rows))
    df.to_csv(out_dir / f"{model_name}_approximation_curves.csv", index=False)

    if not df.empty:
        for kind, g in df.groupby("kind"):
            plt.figure(figsize=(7, 5))
            for name, q in g.groupby("name"):
                plt.plot(q["storage_ratio"], q["fro_error"], marker="o", alpha=0.45, linewidth=1)
            plt.xlabel("low-rank storage / dense storage")
            plt.ylabel(r"$\|W-W_r\|_F / \|W\|_F$")
            plt.title(f"Approximation error vs storage: {model_name}, {kind}")
            plt.tight_layout()
            safe_kind = kind.replace("/", "_")
            plt.savefig(out_dir / f"{model_name}_{safe_kind}_error_vs_storage.png", dpi=200)
            plt.close()

        pivot = df.groupby(["kind", "rank_ratio"], as_index=False).agg(fro_error=("fro_error", "median"), storage_ratio=("storage_ratio", "median"))
        plt.figure(figsize=(8, 5))
        for kind, g in pivot.groupby("kind"):
            plt.plot(g["storage_ratio"], g["fro_error"], marker="o", label=kind)
        plt.xlabel("median low-rank storage / dense storage")
        plt.ylabel("median relative Frobenius error")
        plt.title(f"Median low-rank approximation trade-off: {model_name}")
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(out_dir / f"{model_name}_median_tradeoff_by_kind.png", dpi=200)
        plt.close()

    finish_analysis(args)
    print(f"Saved: {out_dir}")
    if not df.empty:
        print(df.groupby(["kind", "rank_ratio"])[["fro_error", "spectral_error", "storage_ratio"]].median().to_string())


if __name__ == "__main__":
    main()
