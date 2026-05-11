from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import List, Tuple

import torch
import torch.nn as nn


@dataclass
class CompressionRecord:
    name: str
    original_params: int
    compressed_params: int
    rank: int
    in_features: int
    out_features: int


class LowRankLinear(nn.Module):
    """Linear layer initialized by truncated SVD of an existing dense layer."""

    def __init__(self, original: nn.Linear, rank: int):
        super().__init__()
        if rank <= 0:
            raise ValueError(f"rank must be positive, got {rank}")
        in_features = original.in_features
        out_features = original.out_features
        rank = min(rank, in_features, out_features)

        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank

        W = original.weight.detach().to(torch.float32).cpu()
        U, S, Vh = torch.linalg.svd(W, full_matrices=False)
        U_r = U[:, :rank]
        S_r = S[:rank]
        Vh_r = Vh[:rank, :]
        sqrt_S = torch.sqrt(S_r)

        A = U_r * sqrt_S.unsqueeze(0)       # [out, r]
        B = sqrt_S.unsqueeze(1) * Vh_r      # [r, in]

        self.first = nn.Linear(in_features, rank, bias=False)
        self.second = nn.Linear(rank, out_features, bias=(original.bias is not None))
        self.first.weight.data.copy_(B)
        self.second.weight.data.copy_(A)
        if original.bias is not None:
            self.second.bias.data.copy_(original.bias.detach().cpu())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.second(self.first(x))


def module_by_name(root: nn.Module, dotted_name: str) -> nn.Module:
    mod = root
    for part in dotted_name.split("."):
        mod = getattr(mod, part)
    return mod


def set_module_by_name(root: nn.Module, dotted_name: str, new_module: nn.Module) -> None:
    parts = dotted_name.split(".")
    parent = root
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], new_module)


def find_linear_targets(model: nn.Module, mode: str) -> List[str]:
    if mode not in {"ffn", "attn", "all"}:
        raise ValueError("mode must be one of: ffn, attn, all")
    names: List[str] = []
    for name, mod in model.named_modules():
        if not isinstance(mod, nn.Linear):
            continue
        is_attn = ("attn.qkv" in name) or ("attn.proj" in name)
        is_ffn = ("mlp.fc1" in name) or ("mlp.fc2" in name)
        is_head = name.endswith("head")
        if mode == "ffn" and is_ffn:
            names.append(name)
        elif mode == "attn" and is_attn:
            names.append(name)
        elif mode == "all" and (is_ffn or is_attn or is_head):
            names.append(name)
    return names


def dense_params(layer: nn.Linear) -> int:
    return layer.weight.numel() + (0 if layer.bias is None else layer.bias.numel())


def lowrank_params(layer: nn.Linear, rank: int) -> int:
    return rank * layer.in_features + layer.out_features * rank + (0 if layer.bias is None else layer.out_features)


def rank_from_ratio(layer: nn.Linear, rank_ratio: float) -> int:
    return max(1, int(math.floor(rank_ratio * min(layer.in_features, layer.out_features))))


def compress_model_inplace(model: nn.Module, mode: str, rank_ratio: float, verbose: bool = True) -> List[CompressionRecord]:
    records: List[CompressionRecord] = []
    for name in find_linear_targets(model, mode):
        layer = module_by_name(model, name)
        if not isinstance(layer, nn.Linear):
            continue
        rank = rank_from_ratio(layer, rank_ratio)
        orig = dense_params(layer)
        comp = lowrank_params(layer, rank)
        if comp >= orig:
            if verbose:
                print(f"[skip] {name}: rank={rank}, params {orig} -> {comp}; no saving")
            continue
        replacement = LowRankLinear(layer, rank=rank)
        replacement.to(layer.weight.device)
        set_module_by_name(model, name, replacement)
        rec = CompressionRecord(name, orig, comp, rank, layer.in_features, layer.out_features)
        records.append(rec)
        if verbose:
            print(f"[done] {name}: rank={rank}, params {orig} -> {comp} ({100.0 * comp / orig:.2f}%)")
    return records


def records_to_dicts(records: List[CompressionRecord]):
    return [asdict(r) for r in records]
