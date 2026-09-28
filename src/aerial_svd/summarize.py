from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser("Summarize JSON result files into CSV and plots")
    p.add_argument("--results-dir", type=str, default="results")
    p.add_argument("--output-dir", type=str, default="plots")
    p.add_argument("--csv", type=str, default="results/summary.csv")
    return p.parse_args()


def main():
    args = parse_args()
    rows = []
    for path in Path(args.results_dir).glob("*.json"):
        with open(path, "r", encoding="utf-8") as f:
            row = json.load(f)
        row["file"] = str(path)
        rows.append(row)
    if not rows:
        raise SystemExit(f"No JSON result files found in {args.results_dir}")
    df = pd.DataFrame(rows)
    csv_path = Path(args.csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    print(df)
    print(f"Saved {csv_path}")

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    if "rank_ratio" in df.columns and "test_top1" in df.columns:
        plot_df = df.dropna(subset=["rank_ratio", "test_top1"]).copy()
        if not plot_df.empty:
            for mode, g in plot_df.groupby("svd_mode"):
                g = g.sort_values("rank_ratio")
                plt.figure()
                plt.plot(g["rank_ratio"], 100.0 * g["test_top1"], marker="o", label=str(mode))
                plt.xlabel("rank ratio")
                plt.ylabel("Top-1 accuracy, %")
                plt.title(f"Accuracy vs rank ratio: {mode}")
                plt.grid(True)
                plt.savefig(out / f"accuracy_vs_rank_{mode}.png", dpi=160, bbox_inches="tight")
                plt.close()

    if "total_params" in df.columns and "test_top1" in df.columns:
        plt.figure()
        plt.scatter(df["total_params"] / 1e6, 100.0 * df["test_top1"])
        plt.xlabel("parameters, millions")
        plt.ylabel("Top-1 accuracy, %")
        plt.title("Accuracy vs parameter count")
        plt.grid(True)
        plt.savefig(out / "accuracy_vs_params.png", dpi=160, bbox_inches="tight")
        plt.close()

    if "total_params" in df.columns and "mean_image_latency_ms" in df.columns:
        plt.figure()
        plt.scatter(df["total_params"] / 1e6, df["mean_image_latency_ms"])
        plt.xlabel("parameters, millions")
        plt.ylabel("mean image latency, ms")
        plt.title("Latency vs parameter count")
        plt.grid(True)
        plt.savefig(out / "latency_vs_params.png", dpi=160, bbox_inches="tight")
        plt.close()


if __name__ == "__main__":
    main()
