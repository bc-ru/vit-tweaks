"""JAX matrix kernels; host-side summaries retain the reference metric definitions."""
from __future__ import annotations

import math
import time
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from aerial_svd.matrix_records import SpectralRecord, ApproximationRecord, RandomizedSVDRecord

# SVD remains float32. Energy sums intentionally use host NumPy float64, just
# as the reference converts the float32 spectrum to double before reduction.
def singular_values(W):
    return jnp.linalg.svd(jnp.asarray(W, dtype=jnp.float32), compute_uv=False)


def ranks_for_energy(s, levels=(0.90, 0.95, 0.99)):
    energy = np.asarray(s, dtype=np.float64) ** 2
    total = energy.sum()
    if not energy.size or total == 0:
        return {level: 0 for level in levels}
    cumulative = np.cumsum(energy) / total
    return {level: min(int(np.searchsorted(cumulative, level)) + 1, energy.size) for level in levels}


def spectral_stats(model_name, checkpoint, rec, W, eps=1e-10):
    s = np.asarray(singular_values(W))
    smax = float(s[0]) if s.size else 0.0
    smin = float(s[-1]) if s.size else 0.0
    tol = eps * max(W.shape) * max(smax, 1.0)
    above = s[s > tol]
    numerical_rank = int(above.size)
    sigma_min_eps = float(above[-1]) if numerical_rank else 0.0
    cond_eps = smax / sigma_min_eps if sigma_min_eps > 0 else math.inf
    fro = float(jnp.linalg.norm(jnp.asarray(W, dtype=jnp.float32)))
    stable_rank = fro * fro / (smax * smax) if smax > 0 else 0.0
    er = ranks_for_energy(s)
    rmax = min(W.shape)
    return SpectralRecord(
        model_name, checkpoint, rec.name, rec.kind, int(W.shape[0]), int(W.shape[1]),
        rec.params, smax, smin, sigma_min_eps, cond_eps, fro, smax, stable_rank,
        numerical_rank, er[0.90], er[0.95], er[0.99],
        er[0.90] / max(rmax, 1), er[0.95] / max(rmax, 1), er[0.99] / max(rmax, 1),
    )


def approximation_records(model_name, checkpoint, rec, W, rank_ratios):
    s = np.asarray(singular_values(W), dtype=np.float64)
    energy = s ** 2
    total = float(energy.sum())
    m, n = map(int, W.shape)
    cumulative = np.cumsum(energy)
    records = []
    for rho in rank_ratios:
        r = max(1, min(min(m, n), int(math.floor(float(rho) * min(m, n)))))
        tail = float(energy[r:].sum())
        records.append(ApproximationRecord(
            model_name, checkpoint, rec.name, rec.kind, m, n, r, float(rho),
            m * n, r * (m + n), r * (m + n) / max(m * n, 1),
            math.sqrt(tail / total) if total > 0 else 0.0,
            float(s[r] / s[0]) if r < s.size and s[0] > 0 else 0.0,
            float(cumulative[r - 1] / total) if total > 0 and r - 1 < cumulative.size else 1.0,
        ))
    return records


@partial(jax.jit, static_argnames=("rank", "n_iter"))
def _randomized_svd(W, omega, rank, n_iter):
    with jax.default_matmul_precision("highest"):
        Y = W @ omega
        for _ in range(max(0, n_iter)):
            Y = W @ (W.T @ Y)
        Q, _ = jnp.linalg.qr(Y, mode="reduced")
        B = Q.T @ W
        Uh, S, Vh = jnp.linalg.svd(B, full_matrices=False)
        U = Q @ Uh
    return U[:, :rank], S[:rank], Vh[:rank, :]


def randomized_svd(W, rank, oversampling=8, n_iter=1, seed=42, *, omega=None):
    W = jnp.asarray(W, dtype=jnp.float32)
    m, n = W.shape
    if not 1 <= rank <= min(m, n) or oversampling < 0:
        raise ValueError("Require 1 <= rank <= min(shape) and oversampling >= 0")
    width = min(rank + oversampling, min(m, n))
    if omega is None:
        omega = jax.random.normal(jax.random.key(seed), (n, width), dtype=jnp.float32)
    else:
        omega = jnp.asarray(omega, dtype=jnp.float32)
        if omega.shape != (n, width):
            raise ValueError(f"omega must have shape {(n, width)}")
    return _randomized_svd(W, omega, rank=rank, n_iter=n_iter)


@jax.jit
def _full_svd(W):
    return jnp.linalg.svd(W, full_matrices=False)


def benchmark_randomized_svd(model_name, checkpoint, rec, W, rank, oversampling=8, n_iter=1, seed=42):
    W = jax.device_put(np.asarray(W, dtype=np.float32))
    W.block_until_ready()
    # Compile/warm both paths outside measurement. Random sampling remains part
    # of randomized runtime, matching the reference timer's algorithm boundary.
    jax.block_until_ready(_full_svd(W))
    jax.block_until_ready(randomized_svd(W, rank, oversampling, n_iter, seed))
    t0 = time.perf_counter()
    U, S, Vh = jax.block_until_ready(_full_svd(W))
    full_time = time.perf_counter() - t0
    t0 = time.perf_counter()
    Ur, Sr, Vhr = jax.block_until_ready(randomized_svd(W, rank, oversampling, n_iter, seed))
    randomized_time = time.perf_counter() - t0
    with jax.default_matmul_precision("highest"):
        denom = max(float(jnp.linalg.norm(W)), 1e-30)
        full_err = float(jnp.linalg.norm(W - (U[:, :rank] * S[:rank]) @ Vh[:rank])) / denom
        rand_err = float(jnp.linalg.norm(W - (Ur * Sr) @ Vhr)) / denom
        sig_err = float(jnp.linalg.norm(S[:rank] - Sr)) / max(float(jnp.linalg.norm(S[:rank])), 1e-30)
    return RandomizedSVDRecord(
        model_name, checkpoint, rec.name, rec.kind, int(W.shape[0]), int(W.shape[1]),
        rank, oversampling, n_iter, full_time, randomized_time, sig_err, full_err, rand_err,
    )
