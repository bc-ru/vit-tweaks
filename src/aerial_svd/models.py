from __future__ import annotations

from pathlib import Path
from typing import Optional

import timm
import torch
import torch.nn as nn


def create_model(model_name: str, num_classes: int, pretrained: bool = True, checkpoint: Optional[str] = None) -> nn.Module:
    model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)
    if checkpoint:
        state = torch.load(checkpoint, map_location="cpu")
        if isinstance(state, dict) and "model" in state:
            state = state["model"]
        elif isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"Loaded checkpoint: {checkpoint}")
        print(f"Missing keys: {len(missing)} | Unexpected keys: {len(unexpected)}")
        if missing:
            print("First missing keys:", missing[:10])
        if unexpected:
            print("First unexpected keys:", unexpected[:10])
    return model


def save_checkpoint(path: str | Path, model: nn.Module, optimizer=None, scheduler=None, epoch: int = 0, metrics=None, config=None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model.state_dict(),
        "epoch": epoch,
        "metrics": metrics or {},
        "config": config or {},
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        payload["scheduler"] = scheduler.state_dict()
    torch.save(payload, path)
