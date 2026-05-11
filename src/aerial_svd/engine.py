from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, Iterable, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm


@dataclass
class EpochMetrics:
    loss: float
    top1: float
    num_samples: int


def accuracy_top1(logits: torch.Tensor, targets: torch.Tensor) -> int:
    return int((logits.argmax(dim=1) == targets).sum().item())


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    amp: bool = True,
    grad_accum_steps: int = 1,
    max_batches: Optional[int] = None,
) -> EpochMetrics:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    optimizer.zero_grad(set_to_none=True)

    pbar = tqdm(loader, desc="train", leave=False)
    for step, (images, targets) in enumerate(pbar):
        if max_batches is not None and step >= max_batches:
            break
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with torch.autocast(device_type=device.type, enabled=(amp and device.type == "cuda")):
            logits = model(images)
            loss = criterion(logits, targets) / grad_accum_steps

        if scaler is not None and device.type == "cuda":
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if (step + 1) % grad_accum_steps == 0:
            if scaler is not None and device.type == "cuda":
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        bs = targets.size(0)
        total_loss += float(loss.item()) * grad_accum_steps * bs
        total_correct += accuracy_top1(logits.detach(), targets)
        total_samples += bs
        pbar.set_postfix(loss=total_loss / max(total_samples, 1), top1=total_correct / max(total_samples, 1))

    return EpochMetrics(total_loss / max(total_samples, 1), total_correct / max(total_samples, 1), total_samples)


@torch.no_grad()
def evaluate_accuracy(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    amp: bool = True,
    max_batches: Optional[int] = None,
) -> EpochMetrics:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    for step, (images, targets) in enumerate(tqdm(loader, desc="eval", leave=False)):
        if max_batches is not None and step >= max_batches:
            break
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=(amp and device.type == "cuda")):
            logits = model(images)
            loss = criterion(logits, targets)
        bs = targets.size(0)
        total_loss += float(loss.item()) * bs
        total_correct += accuracy_top1(logits, targets)
        total_samples += bs
    return EpochMetrics(total_loss / max(total_samples, 1), total_correct / max(total_samples, 1), total_samples)


@torch.no_grad()
def benchmark_inference(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    amp: bool = True,
    warmup_batches: int = 5,
    max_batches: int = 50,
) -> Dict[str, float | int | None]:
    model.eval()
    is_cuda = device.type == "cuda"
    if is_cuda:
        torch.cuda.reset_peak_memory_stats(device)

    times = []
    total_samples = 0
    total_correct = 0

    for step, (images, targets) in enumerate(tqdm(loader, desc="bench", leave=False)):
        if step >= warmup_batches + max_batches:
            break
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        if is_cuda:
            torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        with torch.autocast(device_type=device.type, enabled=(amp and is_cuda)):
            logits = model(images)
        if is_cuda:
            torch.cuda.synchronize(device)
        t1 = time.perf_counter()
        if step >= warmup_batches:
            times.append(t1 - t0)
            total_samples += targets.size(0)
            total_correct += accuracy_top1(logits, targets)

    total_time = sum(times)
    mean_batch = total_time / max(len(times), 1)
    batch_size = loader.batch_size or 1
    peak = torch.cuda.max_memory_allocated(device) if is_cuda else None
    return {
        "bench_images": total_samples,
        "bench_top1": total_correct / max(total_samples, 1),
        "mean_batch_latency_ms": 1000.0 * mean_batch,
        "mean_image_latency_ms": 1000.0 * mean_batch / batch_size,
        "throughput_images_per_sec": total_samples / max(total_time, 1e-12),
        "peak_gpu_memory_bytes": peak,
    }
