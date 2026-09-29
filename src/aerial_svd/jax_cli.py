"""Native JAX implementations behind the existing project command names."""
import argparse
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import re

import jax
import jax.numpy as jnp
import numpy as np
import yaml

from aerial_svd.jax_checkpoint import load_checkpoint, save_checkpoint
from aerial_svd.jax_compression import compress, matrix_kind
from aerial_svd.jax_data import ImageFolder
from aerial_svd.jax_engine import Engine
from aerial_svd.jax_models import apply, dense_weight, initialize
from aerial_svd.model_spec import ModelSpec, is_target, linear_names, parameter_shapes


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as f:
        json.dump(data, f, indent=2)


def counts(params):
    total = sum(v.size for v in params.values())
    return {"total_params": total, "trainable_params": total,
            "param_memory_bytes": sum(v.size * v.dtype.itemsize for v in params.values())}


def parser_for(command):
    p = argparse.ArgumentParser(f"JAX {command}")
    p.add_argument("--config", required=True)
    if command in ("train", "evaluate", "finetune_compressed"):
        p.add_argument("--data-dir")
        p.add_argument("--max-test-batches", type=int)
    if command == "train":
        p.add_argument("--output-dir")
        initial = p.add_mutually_exclusive_group()
        initial.add_argument("--resume")
        initial.add_argument("--init-checkpoint")
        p.add_argument("--restore-training-state", action="store_true")
        p.add_argument("--max-train-batches", type=int)
        p.add_argument("--max-val-batches", type=int)
    elif command == "finetune_compressed":
        p.add_argument("--compressed-checkpoint", required=True)
        p.add_argument("--svd-mode", choices=("ffn", "attn", "all"), required=True)
        p.add_argument("--rank-ratio", type=float, required=True)
        p.add_argument("--output-dir", required=True)
        p.add_argument("--epochs", type=int)
        p.add_argument("--lr", type=float, default=1e-5)
        p.add_argument("--max-train-batches", type=int)
        p.add_argument("--max-val-batches", type=int)
    elif command == "evaluate":
        p.add_argument("--checkpoint", required=True)
        p.add_argument("--svd-mode", choices=("ffn", "attn", "all"))
        p.add_argument("--rank-ratio", type=float)
        p.add_argument("--batch-size", type=int)
        p.add_argument("--max-batches", type=int, default=50)
        p.add_argument("--warmup-batches", type=int, default=5)
        p.add_argument("--output", required=True)
    elif command == "compress":
        p.add_argument("--checkpoint", required=True)
        p.add_argument("--mode", choices=("ffn", "attn", "all"), required=True)
        p.add_argument("--rank-ratio", type=float, required=True)
        p.add_argument("--output", required=True)
    elif command == "inspect":
        p.add_argument("--checkpoint")
        p.add_argument("--output")
        p.add_argument("--rank-ratio", type=float, default=0.5)
    elif command == "layer_error_analysis":
        p.add_argument("--checkpoint")
        p.add_argument("--device", default="auto")
        p.add_argument("--out-dir", default="matrix_results/jax/layer_error")
        p.add_argument("--include-regex", default="mlp.fc|attn.qkv|attn.proj")
        p.add_argument("--exclude-regex")
        p.add_argument("--rank-ratios", nargs="+", type=float, default=[0.25, 0.50, 0.75])
        p.add_argument("--max-targets", type=int, default=16)
        p.add_argument("--max-batches", type=int, default=4)
        p.add_argument("--max-stored-errors", type=int, default=20000)
        p.add_argument("--amp", action="store_true")
    else:
        raise ValueError(command)
    return p


def verify_compression(spec, params, mode, ratio):
    if not mode:
        return
    if ratio is None or not 0 < ratio <= 1:
        raise ValueError("--svd-mode requires a rank ratio in (0, 1]")
    shapes, _ = parameter_shapes(spec)
    expected = {}
    for name in linear_names(params):
        if is_target(name, mode):
            m, n = shapes[name + ".weight"]
            r = max(1, math.floor(ratio * min(m, n)))
            if r * (m+n) < m*n:
                expected[name] = r
    actual = {k[:-len(".first.weight")]: v.shape[0] for k, v in params.items() if k.endswith(".first.weight")}
    if actual != expected:
        raise ValueError("Checkpoint low-rank structure disagrees with --svd-mode/--rank-ratio")


def main(command, argv):
    args = parser_for(command).parse_args(argv)
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    if int(cfg.get("seed", 42)) != 42:
        raise ValueError("This research protocol requires seed 42")
    if getattr(args, "data_dir", None):
        cfg["data_dir"] = args.data_dir
    if getattr(args, "output_dir", None):
        cfg["output_dir"] = args.output_dir
    if command == "finetune_compressed":
        cfg["lr"] = args.lr
        if args.epochs is not None:
            cfg["epochs"] = args.epochs
    requested = getattr(args, "device", cfg.get("device", "auto"))
    platform = "gpu" if requested in ("cuda", "gpu") else requested
    device = jax.devices()[0] if platform == "auto" else jax.devices(platform)[0]
    jax.config.update("jax_default_device", device)
    dataset = None
    path = getattr(args, "checkpoint", None) or getattr(args, "compressed_checkpoint", None) or getattr(args, "resume", None) or getattr(args, "init_checkpoint", None)
    if command == "layer_error_analysis" and not path:
        path = str(Path(cfg["output_dir"]) / "best.npz")
    metadata, optimizer_leaves = {}, []
    if path:
        spec, params, buffers, metadata, optimizer_leaves = load_checkpoint(path)
        if command == "layer_error_analysis":
            cfg.update({k: v for k, v in metadata.get("config", {}).items() if k != "data_dir"})
        if spec.model != cfg["model"] or spec.img_size != int(cfg.get("img_size", 224)):
            raise ValueError("Checkpoint architecture disagrees with config")
        params, buffers = jax.tree.map(jnp.asarray, (params, buffers))
    else:
        if cfg.get("pretrained", True) and command != "inspect":
            raise ValueError("Export timm initialization with convert_checkpoint --pretrained, then pass --init-checkpoint")
        if command == "train":
            dataset = ImageFolder(cfg)
        classes = len(dataset.classes) if dataset else int(cfg.get("num_classes", 30))
        spec = ModelSpec(cfg["model"], classes, int(cfg.get("img_size", 224)))
        params, buffers = initialize(spec, int(cfg.get("seed", 42)))
    if dataset is None and command in ("train", "finetune_compressed", "evaluate", "layer_error_analysis"):
        dataset = ImageFolder(cfg)
    if dataset:
        if spec.num_classes != len(dataset.classes):
            raise ValueError("Checkpoint classifier disagrees with dataset")
        saved_classes = metadata.get("config", {}).get("classes")
        if saved_classes is not None and saved_classes != dataset.classes:
            raise ValueError("Checkpoint class ordering disagrees with dataset")
        if metadata.get("dataset_fingerprint", dataset.fingerprint) != dataset.fingerprint:
            raise ValueError("Checkpoint dataset/split fingerprint changed")
        cfg.update(num_classes=len(dataset.classes), classes=dataset.classes)
    provenance = {"backend": "jax", "jax_version": jax.__version__, "device": str(device), "platform": device.platform,
                  "config": cfg, "source_checkpoint": path, "model_spec": asdict(spec), "arguments": vars(args)}
    if dataset:
        provenance["dataset_fingerprint"] = dataset.fingerprint
        provenance["image_decoder"] = dataset.decoder
    if command in ("train", "finetune_compressed"):
        if command == "finetune_compressed":
            verify_compression(spec, params, args.svd_mode, args.rank_ratio)
        train(args, cfg, spec, params, buffers, dataset, provenance, metadata, optimizer_leaves, device)
    elif command == "evaluate":
        verify_compression(spec, params, args.svd_mode, args.rank_ratio)
        bs = args.batch_size or int(cfg.get("eval_batch_size", cfg.get("batch_size", 64)))
        engine = Engine(spec, amp=bool(cfg.get("amp", True)) and device.platform == "gpu")
        acc = engine.evaluate(params, buffers, dataset.batches("test", bs, max_batches=args.max_test_batches), smoothing=0)
        bench = engine.benchmark(params, buffers, dataset.batches("test", bs), bs, args.warmup_batches, args.max_batches)
        out = dict(model=spec.model, checkpoint=path, svd_mode=args.svd_mode, rank_ratio=args.rank_ratio,
                   num_classes=spec.num_classes, test_loss=acc["loss"], test_top1=acc["top1"], test_samples=acc["num_samples"],
                   checkpoint_bytes=Path(path).stat().st_size, **counts(params), **bench, **provenance)
        out["memory_measurement"] = "jax_process_peak_gpu_memory_bytes is process-lifetime allocator high-water including setup/evaluation; resettable benchmark-only peak unavailable; parameter bytes exact"
        write_json(args.output, out)
        print(json.dumps(out, indent=2))
    elif command == "compress":
        before = counts(params)
        compressed, records = compress(params, args.mode, args.rank_ratio)
        after = counts(compressed)
        sidecar = Path(args.output).with_suffix(".compression.json")
        if sidecar.exists():
            raise FileExistsError(sidecar)
        provenance["compression"] = {"mode": args.mode, "rank_ratio": args.rank_ratio, "records": records}
        save_checkpoint(args.output, spec, compressed, buffers, metadata=provenance)
        out = dict(model=spec.model, source_checkpoint=path, compressed_checkpoint=args.output, mode=args.mode,
            rank_ratio=args.rank_ratio, params_before=before["total_params"], params_after=after["total_params"],
            param_ratio=after["total_params"]/before["total_params"], reduction_percent=100*(1-after["total_params"]/before["total_params"]),
            param_memory_before_bytes=before["param_memory_bytes"], param_memory_after_bytes=after["param_memory_bytes"], records=records, backend="jax")
        write_json(sidecar, out)
        print(json.dumps(out, indent=2))
    elif command == "inspect":
        inspect(args, spec, params)
    else:
        layer_errors(args, spec, params, buffers, dataset, provenance, device)


def train(args, cfg, spec, params, buffers, dataset, provenance, old_metadata, optimizer_leaves, device):
    out = Path(cfg["output_dir"])
    epochs = int(cfg.get("epochs", 10))
    engine = Engine(spec, amp=bool(cfg.get("amp", True)) and device.platform == "gpu",
                    smoothing=float(cfg.get("label_smoothing", 0.0)), weight_decay=float(cfg.get("weight_decay", 0.05)))
    optimizer_state = engine.optimizer.init(params)
    start, history, best, best_checkpoint = 1, [], -1.0, None
    if getattr(args, "restore_training_state", False):
        if not args.resume or not optimizer_leaves or "epoch" not in old_metadata:
            raise ValueError("Full restore requires --resume with a native training checkpoint")
        for key in ("epochs", "lr", "weight_decay", "label_smoothing", "grad_accum_steps", "batch_size", "amp"):
            if old_metadata["config"].get(key) != cfg.get(key):
                raise ValueError(f"Cannot restore after changing {key}")
        if old_metadata["platform"] != device.platform:
            raise ValueError("Full restore requires the original device platform; use weights-only --resume to change platform")
        if old_metadata.get("image_decoder") != dataset.decoder:
            raise ValueError("Cannot restore after changing the image decoder")
        for key in ("max_train_batches", "max_val_batches"):
            if old_metadata.get("arguments", {}).get(key) != getattr(args, key):
                raise ValueError(f"Cannot restore after changing {key}")
        leaves, tree = jax.tree.flatten(optimizer_state)
        if len(leaves) != len(optimizer_leaves) or any(a.shape != b.shape or a.dtype != b.dtype for a, b in zip(leaves, optimizer_leaves)):
            raise ValueError("Optimizer state schema mismatch")
        if any(not np.isfinite(x).all() for x in optimizer_leaves):
            raise ValueError("Nonfinite optimizer state")
        optimizer_state = jax.tree.unflatten(tree, [jnp.asarray(x) for x in optimizer_leaves])
        start = int(old_metadata["epoch"]) + 1
        history, best = old_metadata["history"], old_metadata["best_val"]
        engine.loss_scale = old_metadata["loss_scale"]
        engine.successful_updates = old_metadata["successful_updates"]
        best_checkpoint = Path(args.resume).resolve().parent / old_metadata["best_checkpoint"]
        if not best_checkpoint.is_file():
            raise FileNotFoundError(f"Best checkpoint required for full restore: {best_checkpoint}")
    if epochs < start:
        raise ValueError("No remaining epochs")
    out.mkdir(parents=True, exist_ok=False)
    if best_checkpoint is not None:
        (out / "best.npz").symlink_to(best_checkpoint.resolve())
    write_json(out / "provenance.json", {**provenance, "status": "running"})
    bs = int(cfg.get("batch_size", 64))
    eval_bs = int(cfg.get("eval_batch_size", bs))
    for epoch in range(start, epochs + 1):
        lr = float(cfg.get("lr", 3e-4)) * (1 + math.cos(math.pi * (epoch-1) / epochs)) / 2
        params, buffers, optimizer_state, tr = engine.train_epoch(params, buffers, optimizer_state,
            dataset.batches("train", bs, epoch=epoch, max_batches=args.max_train_batches), lr,
            accumulation=int(cfg.get("grad_accum_steps", 1)))
        va = engine.evaluate(params, buffers, dataset.batches("val", eval_bs, max_batches=args.max_val_batches))
        row = dict(epoch=epoch, train_loss=tr["loss"], train_top1=tr["top1"], val_loss=va["loss"], val_top1=va["top1"])
        history.append(row)
        improved = va["top1"] > best
        best = max(best, va["top1"])
        checkpoint = out / f"epoch-{epoch:04d}.npz"
        if improved:
            best_checkpoint = checkpoint
        current = {**provenance, "epoch": epoch, "history": history, "best_val": best, "loss_scale": engine.loss_scale,
                   "successful_updates": engine.successful_updates, "optimizer_updates": tr["optimizer_updates"], "skipped_updates": tr["skipped_updates"],
                   "best_checkpoint": os.path.relpath(best_checkpoint, out)}
        save_checkpoint(checkpoint, spec, params, buffers, metadata=current, optimizer_leaves=jax.tree.leaves(optimizer_state))
        # Only links owned by this newly-created run are atomically advanced.
        for name in (["last.npz", "best.npz"] if improved else ["last.npz"]):
            temporary = out / (name + ".next")
            temporary.symlink_to(checkpoint.name)
            temporary.replace(out / name)
        print(json.dumps({**row, "lr": lr, "optimizer_updates": tr["optimizer_updates"]}), flush=True)
    te = engine.evaluate(params, buffers, dataset.batches("test", eval_bs, max_batches=args.max_test_batches))
    final = dict(model=spec.model, num_classes=spec.num_classes, **counts(params), test_loss_last=te["loss"],
                 test_top1_last=te["top1"], best_val_top1=best, backend="jax")
    if hasattr(args, "compressed_checkpoint"):
        final.update(compressed_checkpoint=args.compressed_checkpoint, svd_mode=args.svd_mode, rank_ratio=args.rank_ratio)
    write_json(out / "history.json", history)
    write_json(out / "final_metrics.json", final)
    write_json(out / "complete.json", {"status": "complete", "epochs": epochs, "test_samples": te["num_samples"]})
    print(json.dumps(final, indent=2))


def inspect(args, spec, params):
    from collections import defaultdict
    from aerial_svd.parameter_groups import classify_param
    groups = defaultdict(int)
    rows, candidates = [], []
    for name, value in params.items():
        group = classify_param(name)
        groups[group] += value.size
        rows.append(dict(name=name, shape=list(value.shape), params=value.size, group=group, trainable=True))
    for mode in ("ffn", "attn", "all"):
        for name in linear_names(params):
            if not is_target(name, mode) or name + ".weight" not in params:
                continue
            m, n = params[name + ".weight"].shape
            rank = max(1, math.floor(args.rank_ratio * min(m, n)))
            bias = m if name + ".bias" in params else 0
            original, compressed = m*n+bias, rank*(m+n)+bias
            candidates.append(dict(mode=mode, name=name, in_features=n, out_features=m, rank=rank,
                                   dense_params=original, lowrank_params=compressed, saving_ok=compressed < original))
    out = dict(model=spec.model, num_classes=spec.num_classes, **counts(params), parameter_groups=dict(groups),
               tensors=rows, candidates=candidates, backend="jax")
    if args.output:
        write_json(args.output, out)
    print(json.dumps({k: v for k, v in out.items() if k not in ("tensors", "candidates")}, indent=2))


def layer_errors(args, spec, params, buffers, dataset, provenance, device):
    import pandas as pd
    import matplotlib.pyplot as plt
    if args.max_targets < 1 or args.max_batches < 1 or args.max_stored_errors < 0:
        raise ValueError("Invalid layer-error limits")
    if any(not math.isfinite(rho) or not 0 < rho <= 1 for rho in args.rank_ratios):
        raise ValueError("Rank ratios must be in (0, 1]")
    inc = re.compile(args.include_regex) if args.include_regex else None
    exc = re.compile(args.exclude_regex) if args.exclude_regex else None
    names = []
    actual_modules = []
    for name in linear_names(params):
        actual_modules.extend([name + ".first", name + ".second"] if name + ".first.weight" in params else [name])
    for name in actual_modules:
        W = dense_weight(params, name)
        key = f"{name} {matrix_kind(name, W)}"
        if (inc is None or inc.search(key)) and (exc is None or not exc.search(key)):
            names.append(name)
    names = names[:args.max_targets]
    if not names:
        raise ValueError("No linear layers matched")
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / "provenance.json", {**provenance, "arguments": vars(args), "status": "running"})
    factors, accum = {}, {}
    for name in names:
        W = jnp.asarray(dense_weight(params, name), dtype=jnp.float32)
        U, S, Vh = jnp.linalg.svd(W, full_matrices=False)
        energy = np.asarray(S, dtype=np.float64) ** 2
        total = float(energy.sum())
        for rho in args.rank_ratios:
            rank = max(1, min(min(W.shape), math.floor(rho * min(W.shape))))
            factors[name, rho] = (U[:, :rank], S[:rank], Vh[:rank])
            accum[name, rho] = dict(rank=rank, matrix_fro_error=math.sqrt(float(energy[rank:].sum()) / total) if total else 0.,
                matrix_spectral_error=float(S[rank]/S[0]) if rank < S.size and S[0] > 0 else 0.,
                sum=0., count=0, max=0., values=[])
    dtype = jnp.float16 if args.amp and device.platform == "gpu" else jnp.float32
    forward = jax.jit(lambda p, b, x: apply(p, b, x, spec, capture=tuple(names), dtype=dtype)[2])
    bs = int(provenance["config"].get("eval_batch_size", 64))
    for images, _ in dataset.batches("test", bs, max_batches=args.max_batches):
        captured = forward(params, buffers, jnp.asarray(images))
        for name, (x, y) in captured.items():
            x = x.reshape(-1, x.shape[-1]).astype(jnp.float32)
            y = y.reshape(-1, y.shape[-1]).astype(jnp.float32)
            denominator = jnp.maximum(jnp.linalg.norm(y, axis=1), 1e-12)
            bias = params.get(name + ".bias", params.get(name + ".second.bias"))
            for rho in args.rank_ratios:
                U, S, Vh = factors[name, rho]
                with jax.default_matmul_precision("highest"):
                    predicted = ((x @ Vh.T) * S) @ U.T
                if bias is not None:
                    predicted += bias.astype(jnp.float32)
                errors = np.asarray(jnp.linalg.norm(y-predicted, axis=1) / denominator)
                item = accum[name, rho]
                item["sum"] += float(errors.sum(dtype=np.float32))
                item["count"] += errors.size
                item["max"] = max(item["max"], float(errors.max()))
                room = max(0, args.max_stored_errors - len(item["values"]))
                item["values"].extend(errors[:room].tolist())
    rows = []
    for (name, rho), item in accum.items():
        rows.append(dict(model=spec.model, checkpoint=provenance["source_checkpoint"], layer=name,
            kind=matrix_kind(name, dense_weight(params, name)), rank_ratio=rho, rank=item["rank"],
            matrix_fro_error=item["matrix_fro_error"], matrix_spectral_error=item["matrix_spectral_error"],
            layer_output_error_mean=item["sum"]/max(item["count"], 1),
            layer_output_error_p95=float(np.quantile(item["values"], 0.95)) if item["values"] else float("nan"),
            layer_output_error_max=item["max"], num_vectors=item["count"]))
    frame = pd.DataFrame(rows)
    frame.to_csv(out / f"{spec.model}_layer_error.csv", index=False)
    plt.figure(figsize=(8, 5))
    for kind, group in frame.groupby("kind"):
        q = group.groupby("rank_ratio", as_index=False).agg(layer_output_error_mean=("layer_output_error_mean", "median"), matrix_fro_error=("matrix_fro_error", "median"))
        plt.plot(q.matrix_fro_error, q.layer_output_error_mean, marker="o", label=kind)
    plt.xlabel("median relative matrix Frobenius error")
    plt.ylabel("median relative layer-output error")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / f"{spec.model}_matrix_vs_layer_error.png", dpi=200)
    plt.close()
    write_json(out / "complete.json", {"status": "complete", "rows": len(rows)})
    print(frame.to_string(index=False))
