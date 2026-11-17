"""Structural low-rank compression using the existing symmetric sqrt(S) split."""

import math

import jax.numpy as jnp
import numpy as np

from aerial_svd.matrix_records import MatrixRecord
from aerial_svd.model_spec import is_target, linear_names


def factorize_weight(weight, rank):
    """Exact float32 SVD, in stored (out, in) orientation, at any valid rank."""
    W = jnp.asarray(weight, dtype=jnp.float32)
    if W.ndim != 2 or type(rank) is not int or not 1 <= rank <= min(W.shape):
        raise ValueError("Require a matrix and integer rank in [1, min(shape)]")
    if not np.isfinite(np.asarray(W)).all():
        raise ValueError("Nonfinite weight")
    U, S, Vh = jnp.linalg.svd(W, full_matrices=False)
    root = jnp.sqrt(S[:rank])
    return root[:, None] * Vh[:rank], U[:, :rank] * root


def compress(params, mode, rank_ratio=None, *, ranks=None, factors=None):
    if ranks is None and (rank_ratio is None or not math.isfinite(rank_ratio) or not 0 < rank_ratio <= 1):
        raise ValueError("rank ratio must be in (0, 1]")
    if ranks is not None and (rank_ratio is not None or not ranks):
        raise ValueError("Provide either a rank ratio or a nonempty explicit rank map")
    is_target("", mode)  # Validate even if the model has no eligible layers.
    names = linear_names(params)
    if ranks is not None and any(name not in names or not is_target(name, mode) for name in ranks):
        raise ValueError("Explicit ranks include unknown or out-of-scope matrices")
    result = dict(params)
    records = []
    for name in names:
        if not is_target(name, mode) or (ranks is not None and name not in ranks):
            continue
        if name + ".first.weight" in params:
            raise ValueError(f"Layer already compressed: {name}")
        W = jnp.asarray(params[name + ".weight"], dtype=jnp.float32)
        m, n = W.shape
        rank = ranks[name] if ranks is not None else max(1, math.floor(rank_ratio * min(m, n)))
        bias = params.get(name + ".bias")
        original = m * n + (m if bias is not None else 0)
        compressed = rank * (m + n) + (m if bias is not None else 0)
        if factors is None:
            first, second = factorize_weight(W, rank)
        else:
            if type(rank) is not int or not 1 <= rank <= min(m, n):
                raise ValueError('Invalid factor rank')
            first, second = (np.asarray(a, dtype=np.float32) for a in factors[name])
            if (first.shape != (rank, n) or second.shape != (m, rank)
                    or not np.isfinite(first).all() or not np.isfinite(second).all()):
                raise ValueError(f'Invalid factors for {name}')
        result.pop(name + ".weight")
        result[name + ".first.weight"] = first
        result[name + ".second.weight"] = second
        if bias is not None:
            result[name + ".second.bias"] = result.pop(name + ".bias")
        records.append(
            dict(
                name=name,
                original_params=original,
                compressed_params=compressed,
                rank=rank,
                in_features=n,
                out_features=m,
            )
        )
    return result, records


def matrix_kind(name, weight):
    if weight.ndim == 4:
        return "conv"
    for pattern, kind in (
        ("attn.qkv", "attn_qkv"),
        ("attn.proj", "attn_proj"),
        ("mlp.fc1", "ffn_expand"),
        ("mlp.fc2", "ffn_contract"),
    ):
        if pattern in name:
            return kind
    return "classifier" if name.endswith("head") else "linear_other"


def weight_matrices(params):
    # Expose actual factor matrices for compressed checkpoints, as named_modules does.
    # JAX pytrees sort dictionary keys lexically (block 10 before block 2).
    # Restore model traversal order before any --max-matrices selection.
    modules = {name: i for i, name in enumerate(linear_names(params))}

    def order(item):
        name = item[0][:-7]
        parent = (
            name.rsplit(".", 1)[0] if name.endswith((".first", ".second")) else name
        )
        return (
            (1, modules[parent], int(name.endswith(".second")))
            if parent in modules
            else (0, name, 0)
        )

    weights = [
        (key, value)
        for key, value in params.items()
        if key.endswith(".weight") and value.ndim in (2, 4)
    ]
    for key, value in sorted(weights, key=order):
        if key.endswith(".weight") and value.ndim in (2, 4):
            name = key[:-7]
            W = np.asarray(value, dtype=np.float32)
            matrix = W.reshape(W.shape[0], -1)
            yield (
                MatrixRecord(name, matrix_kind(name, W), matrix.shape, W.shape, W.size),
                matrix,
            )
