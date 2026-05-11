from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn

from aerial_svd.data import build_loaders
from aerial_svd.engine import evaluate_accuracy, train_one_epoch
from aerial_svd.models import create_model, save_checkpoint
from aerial_svd.utils import count_total_params, count_trainable_params, get_device, load_yaml, save_json, seed_everything, sizeof_params_bytes


def parse_args():
    p = argparse.ArgumentParser("Fine-tune timm classifier on ImageFolder aerial imagery")
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--data-dir", type=str, default=None)
    p.add_argument("--output-dir", type=str, default=None)
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--max-train-batches", type=int, default=None)
    p.add_argument("--max-val-batches", type=int, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_yaml(args.config)
    if args.data_dir:
        cfg["data_dir"] = args.data_dir
    if args.output_dir:
        cfg["output_dir"] = args.output_dir

    seed_everything(int(cfg.get("seed", 42)))
    device = get_device(cfg.get("device", "auto"))
    output_dir = Path(cfg.get("output_dir", "runs/default"))
    output_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, test_loader, classes = build_loaders(
        data_dir=cfg["data_dir"],
        img_size=int(cfg.get("img_size", 224)),
        batch_size=int(cfg.get("batch_size", 64)),
        eval_batch_size=int(cfg.get("eval_batch_size", cfg.get("batch_size", 64))),
        num_workers=int(cfg.get("num_workers", 4)),
        train_ratio=float(cfg.get("train_ratio", 0.70)),
        val_ratio=float(cfg.get("val_ratio", 0.15)),
        seed=int(cfg.get("seed", 42)),
        split_file=cfg.get("split_file", str(output_dir / "split.json")),
    )
    num_classes = len(classes)
    cfg["num_classes"] = num_classes
    cfg["classes"] = classes

    model = create_model(cfg["model"], num_classes=num_classes, pretrained=bool(cfg.get("pretrained", True)), checkpoint=args.resume)
    model.to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=float(cfg.get("label_smoothing", 0.0)))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg.get("lr", 3e-4)),
        weight_decay=float(cfg.get("weight_decay", 0.05)),
    )
    epochs = int(cfg.get("epochs", 10))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    use_amp = bool(cfg.get("amp", True))
    scaler = torch.cuda.amp.GradScaler(enabled=(use_amp and device.type == "cuda"))

    best_val = -1.0
    history = []
    for epoch in range(1, epochs + 1):
        print(f"\nEpoch {epoch}/{epochs}")
        tr = train_one_epoch(
            model, train_loader, optimizer, criterion, device,
            scaler=scaler, amp=use_amp,
            grad_accum_steps=int(cfg.get("grad_accum_steps", 1)),
            max_batches=args.max_train_batches,
        )
        va = evaluate_accuracy(model, val_loader, criterion, device, amp=use_amp, max_batches=args.max_val_batches)
        scheduler.step()
        row = {"epoch": epoch, "train_loss": tr.loss, "train_top1": tr.top1, "val_loss": va.loss, "val_top1": va.top1}
        history.append(row)
        print(row)
        save_json(history, output_dir / "history.json")
        if va.top1 > best_val:
            best_val = va.top1
            save_checkpoint(output_dir / "best.pt", model, optimizer, scheduler, epoch, metrics=row, config=cfg)
        save_checkpoint(output_dir / "last.pt", model, optimizer, scheduler, epoch, metrics=row, config=cfg)

    te = evaluate_accuracy(model, test_loader, criterion, device, amp=use_amp)
    final = {
        "model": cfg["model"],
        "num_classes": num_classes,
        "total_params": count_total_params(model),
        "trainable_params": count_trainable_params(model),
        "param_memory_bytes": sizeof_params_bytes(model),
        "test_loss_last": te.loss,
        "test_top1_last": te.top1,
        "best_val_top1": best_val,
    }
    save_json(final, output_dir / "final_metrics.json")
    print("Final:", final)


if __name__ == "__main__":
    main()
