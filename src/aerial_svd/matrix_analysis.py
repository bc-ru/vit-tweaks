from __future__ import annotations

import math
import re
import time
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

from aerial_svd.models import create_model
from aerial_svd.utils import load_yaml
from aerial_svd.matrix_records import (MatrixRecord, SpectralRecord, ApproximationRecord, RandomizedSVDRecord, dataclass_records_to_dicts)


def load_model_from_config(config_path: str | Path, checkpoint: Optional[str] = None, device: str = "cpu") -> tuple[nn.Module, dict, str]:
    cfg = load_yaml(config_path)
    ckpt = checkpoint or str(Path(cfg.get("output_dir", "runs/default")) / "best.pt")
    num_classes = int(cfg.get("num_classes", 1000))
    if "classes" in cfg:
        num_classes = len(cfg["classes"])
    # If checkpoint exists and contains config with num_classes, use it.
    if Path(ckpt).exists():
        payload = torch.load(ckpt, map_location="cpu")
        if isinstance(payload, dict) and isinstance(payload.get("config"), dict):
            ckpt_cfg = payload["config"]
            cfg.update({k: v for k, v in ckpt_cfg.items() if k not in {"data_dir"}})
            if "num_classes" in ckpt_cfg:
                num_classes = int(ckpt_cfg["num_classes"])
            elif "classes" in ckpt_cfg:
                num_classes = len(ckpt_cfg["classes"])
    model = create_model(cfg["model"], num_classes=num_classes, pretrained=False, checkpoint=ckpt if Path(ckpt).exists() else None)
    model.to(torch.device(device))
    model.eval()
    return model, cfg, ckpt


def matrix_kind(name: str, module: nn.Module) -> str:
    if isinstance(module, nn.Conv2d):
        return "conv"
    if "attn.qkv" in name:
        return "attn_qkv"
    if "attn.proj" in name:
        return "attn_proj"
    if "mlp.fc1" in name:
        return "ffn_expand"
    if "mlp.fc2" in name:
        return "ffn_contract"
    if name.endswith("head"):
        return "classifier"
    if isinstance(module, nn.Linear):
        return "linear_other"
    return module.__class__.__name__.lower()


def module_to_matrix(module: nn.Module) -> Optional[torch.Tensor]:
    if isinstance(module, nn.Linear):
        return module.weight.detach().to(torch.float32).cpu()
    if isinstance(module, nn.Conv2d):
        w = module.weight.detach().to(torch.float32).cpu()
        return w.reshape(w.shape[0], -1)
    return None


def iter_weight_matrices(model: nn.Module, include_conv: bool = True, include_linear: bool = True) -> Iterator[tuple[MatrixRecord, torch.Tensor]]:
    for name, module in model.named_modules():
        if name == "":
            continue
        if isinstance(module, nn.Linear) and include_linear:
            W = module_to_matrix(module)
        elif isinstance(module, nn.Conv2d) and include_conv:
            W = module_to_matrix(module)
        else:
            continue
        if W is None or W.ndim != 2:
            continue
        rec = MatrixRecord(
            name=name,
            kind=matrix_kind(name, module),
            shape=tuple(W.shape),
            original_shape=tuple(module.weight.shape),
            params=int(module.weight.numel()),
        )
        yield rec, W


def filter_matrices(records: Iterable[tuple[MatrixRecord, torch.Tensor]], include_regex: Optional[str], exclude_regex: Optional[str]) -> Iterator[tuple[MatrixRecord, torch.Tensor]]:
    inc = re.compile(include_regex) if include_regex else None
    exc = re.compile(exclude_regex) if exclude_regex else None
    for rec, W in records:
        key = f"{rec.name} {rec.kind}"
        if inc and not inc.search(key):
            continue
        if exc and exc.search(key):
            continue
        yield rec, W


def singular_values(W: torch.Tensor) -> torch.Tensor:
    return torch.linalg.svdvals(W.to(torch.float32)).cpu()


def ranks_for_energy(s: torch.Tensor, levels: Sequence[float] = (0.90, 0.95, 0.99)) -> Dict[float, int]:
    if s.numel() == 0:
        return {lvl: 0 for lvl in levels}
    e = s.double().square()
    total = e.sum().item()
    if total == 0.0:
        return {lvl: 0 for lvl in levels}
    c = torch.cumsum(e, dim=0) / total
    out = {}
    for lvl in levels:
        idx = int(torch.searchsorted(c, torch.tensor(float(lvl), dtype=c.dtype)).item())
        out[lvl] = min(idx + 1, s.numel())
    return out


def spectral_stats(model_name: str, checkpoint: str, rec: MatrixRecord, W: torch.Tensor, eps: float = 1e-10) -> SpectralRecord:
    s = singular_values(W)
    smax = float(s[0].item()) if s.numel() else 0.0
    smin = float(s[-1].item()) if s.numel() else 0.0
    tol = eps * max(W.shape) * max(smax, 1.0)
    above = s[s > tol]
    numerical_rank = int(above.numel())
    sigma_min_eps = float(above[-1].item()) if numerical_rank else 0.0
    cond_eps = float(smax / sigma_min_eps) if sigma_min_eps > 0 else math.inf
    fro = float(torch.linalg.vector_norm(W).item())
    stable_rank = float((fro * fro) / (smax * smax)) if smax > 0 else 0.0
    er = ranks_for_energy(s)
    rmax = min(W.shape)
    return SpectralRecord(
        model=model_name,
        checkpoint=checkpoint,
        name=rec.name,
        kind=rec.kind,
        m=int(W.shape[0]),
        n=int(W.shape[1]),
        params=rec.params,
        sigma_max=smax,
        sigma_min=smin,
        sigma_min_eps=sigma_min_eps,
        cond_eps=cond_eps,
        fro_norm=fro,
        spectral_norm=smax,
        stable_rank=stable_rank,
        numerical_rank_eps=numerical_rank,
        r90=er[0.90],
        r95=er[0.95],
        r99=er[0.99],
        r90_ratio=er[0.90] / max(rmax, 1),
        r95_ratio=er[0.95] / max(rmax, 1),
        r99_ratio=er[0.99] / max(rmax, 1),
    )


def approximation_records(model_name: str, checkpoint: str, rec: MatrixRecord, W: torch.Tensor, rank_ratios: Sequence[float]) -> List[ApproximationRecord]:
    s = singular_values(W).double()
    energy = s.square()
    total_energy = float(energy.sum().item())
    m, n = map(int, W.shape)
    dense_storage = m * n
    out: List[ApproximationRecord] = []
    max_rank = min(m, n)
    cumulative = torch.cumsum(energy, dim=0) if energy.numel() else torch.zeros(0, dtype=torch.float64)
    for rho in rank_ratios:
        r = max(1, min(max_rank, int(math.floor(float(rho) * max_rank))))
        tail_energy = float(energy[r:].sum().item()) if r < energy.numel() else 0.0
        fro_error = math.sqrt(tail_energy / total_energy) if total_energy > 0 else 0.0
        spectral_error = float(s[r].item() / s[0].item()) if r < s.numel() and s[0] > 0 else 0.0
        retained = float(cumulative[r - 1].item() / total_energy) if total_energy > 0 and r - 1 < cumulative.numel() else 1.0
        lowrank_storage = r * (m + n)
        out.append(ApproximationRecord(
            model=model_name,
            checkpoint=checkpoint,
            name=rec.name,
            kind=rec.kind,
            m=m,
            n=n,
            rank=r,
            rank_ratio=float(rho),
            dense_storage=dense_storage,
            lowrank_storage=lowrank_storage,
            storage_ratio=lowrank_storage / max(dense_storage, 1),
            fro_error=fro_error,
            spectral_error=spectral_error,
            retained_energy=retained,
        ))
    return out


def randomized_svd(W: torch.Tensor, rank: int, oversampling: int = 8, n_iter: int = 1, seed: int = 42) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    W = W.to(torch.float32).cpu()
    m, n = W.shape
    l = min(rank + oversampling, min(m, n))
    gen = torch.Generator(device="cpu").manual_seed(seed)
    Omega = torch.randn(n, l, generator=gen, dtype=W.dtype)
    Y = W @ Omega
    for _ in range(max(0, n_iter)):
        Y = W @ (W.T @ Y)
    Q, _ = torch.linalg.qr(Y, mode="reduced")
    B = Q.T @ W
    Uh, S, Vh = torch.linalg.svd(B, full_matrices=False)
    U = Q @ Uh
    return U[:, :rank], S[:rank], Vh[:rank, :]


def benchmark_randomized_svd(model_name: str, checkpoint: str, rec: MatrixRecord, W: torch.Tensor, rank: int, oversampling: int = 8, n_iter: int = 1, seed: int = 42) -> RandomizedSVDRecord:
    W = W.to(torch.float32).cpu()
    t0 = time.perf_counter()
    U, S, Vh = torch.linalg.svd(W, full_matrices=False)
    t1 = time.perf_counter()
    U_r, S_r, Vh_r = U[:, :rank], S[:rank], Vh[:rank, :]
    full_err = float(torch.linalg.vector_norm(W - (U_r * S_r.unsqueeze(0)) @ Vh_r).item() / max(torch.linalg.vector_norm(W).item(), 1e-30))

    t2 = time.perf_counter()
    Ur, Sr, Vhr = randomized_svd(W, rank=rank, oversampling=oversampling, n_iter=n_iter, seed=seed)
    t3 = time.perf_counter()
    rand_err = float(torch.linalg.vector_norm(W - (Ur * Sr.unsqueeze(0)) @ Vhr).item() / max(torch.linalg.vector_norm(W).item(), 1e-30))
    sig_err = float(torch.linalg.vector_norm(S_r - Sr).item() / max(torch.linalg.vector_norm(S_r).item(), 1e-30))
    return RandomizedSVDRecord(
        model=model_name,
        checkpoint=checkpoint,
        name=rec.name,
        kind=rec.kind,
        m=int(W.shape[0]),
        n=int(W.shape[1]),
        rank=int(rank),
        oversampling=int(oversampling),
        n_iter=int(n_iter),
        full_svd_time_sec=t1 - t0,
        randomized_svd_time_sec=t3 - t2,
        sigma_relative_error=sig_err,
        approximation_fro_error=full_err,
        randomized_fro_error=rand_err,
    )
