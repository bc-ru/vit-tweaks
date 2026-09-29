from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from aerial_svd.analysis_backend import add_backend_arguments, prepare_analysis, finish_analysis
from aerial_svd.matrix_records import dataclass_records_to_dicts


def parse_args():
    p = argparse.ArgumentParser("Compare classical truncated SVD and randomized SVD on selected weight matrices")
    add_backend_arguments(p)
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--device", default="cpu")
    p.add_argument("--out-dir", default="matrix_results/randomized_svd")
    p.add_argument("--include-regex", default="mlp.fc|attn.qkv|attn.proj")
    p.add_argument("--exclude-regex", default=None)
    p.add_argument("--rank", type=int, default=64)
    p.add_argument("--oversampling", type=int, default=8)
    p.add_argument("--n-iter", type=int, default=1)
    p.add_argument("--max-matrices", type=int, default=12)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    backend, model_name, ckpt, matrices = prepare_analysis(args)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    matrices = list(matrices)[: args.max_matrices]
    for rec, W in matrices:
        r = min(args.rank, min(W.shape))
        rows.append(backend.benchmark_randomized_svd(model_name, ckpt, rec, W, rank=r, oversampling=args.oversampling, n_iter=args.n_iter, seed=args.seed))
    df = pd.DataFrame(dataclass_records_to_dicts(rows))
    df.to_csv(out_dir / f"{model_name}_randomized_svd_benchmark.csv", index=False)
    finish_analysis(args)
    print(f"Saved: {out_dir}")
    if not df.empty:
        cols = ["name", "kind", "m", "n", "rank", "full_svd_time_sec", "randomized_svd_time_sec", "sigma_relative_error", "approximation_fro_error", "randomized_fro_error"]
        print(df[cols].to_string(index=False))


if __name__ == "__main__":
    main()
