from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms


def build_transforms(img_size: int, train: bool, mean=None, std=None):
    mean = mean or (0.485, 0.456, 0.406)
    std = std or (0.229, 0.224, 0.225)
    interpolation = transforms.InterpolationMode.BICUBIC

    if train:
        return transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=(0.70, 1.0), interpolation=interpolation),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
    return transforms.Compose([
        transforms.Resize((img_size, img_size), interpolation=interpolation),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


def _split_indices(n: int, train_ratio: float, val_ratio: float, seed: int) -> Tuple[List[int], List[int], List[int]]:
    if not (0.0 < train_ratio < 1.0 and 0.0 <= val_ratio < 1.0 and train_ratio + val_ratio < 1.0):
        raise ValueError("Need 0 < train_ratio < 1 and 0 <= val_ratio < 1 and train_ratio + val_ratio < 1")
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g).tolist()
    n_train = int(round(train_ratio * n))
    n_val = int(round(val_ratio * n))
    train_idx = perm[:n_train]
    val_idx = perm[n_train:n_train + n_val]
    test_idx = perm[n_train + n_val:]
    return train_idx, val_idx, test_idx


def save_split(indices: Dict[str, List[int]], class_to_idx: Dict[str, int], path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump({"indices": indices, "class_to_idx": class_to_idx}, f, indent=2)


def load_split(path: str | Path) -> Dict[str, List[int]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["indices"]


def build_imagefolder_datasets(
    data_dir: str | Path,
    img_size: int,
    train_ratio: float,
    val_ratio: float,
    seed: int,
    split_file: Optional[str | Path] = None,
):
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Dataset directory does not exist: {data_dir}")

    train_base = datasets.ImageFolder(str(data_dir), transform=build_transforms(img_size, train=True))
    eval_base = datasets.ImageFolder(str(data_dir), transform=build_transforms(img_size, train=False))

    if split_file and Path(split_file).exists():
        indices = load_split(split_file)
    else:
        tr, va, te = _split_indices(len(eval_base), train_ratio, val_ratio, seed)
        indices = {"train": tr, "val": va, "test": te}
        if split_file:
            save_split(indices, eval_base.class_to_idx, split_file)

    train_ds = Subset(train_base, indices["train"])
    val_ds = Subset(eval_base, indices["val"])
    test_ds = Subset(eval_base, indices["test"])
    return train_ds, val_ds, test_ds, eval_base.classes


def build_loaders(
    data_dir: str | Path,
    img_size: int,
    batch_size: int,
    eval_batch_size: int,
    num_workers: int,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: int = 42,
    split_file: Optional[str | Path] = None,
):
    train_ds, val_ds, test_ds, classes = build_imagefolder_datasets(
        data_dir=data_dir,
        img_size=img_size,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        seed=seed,
        split_file=split_file,
    )
    pin = torch.cuda.is_available()
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=eval_batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin)
    test_loader = DataLoader(test_ds, batch_size=eval_batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin)
    return train_loader, val_loader, test_loader, classes
