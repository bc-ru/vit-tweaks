"""Framework-independent native checkpoints; legacy Torch is only a converter."""
from dataclasses import asdict
import json

import numpy as np

from aerial_svd.matrix_archive import save_npz_new
from aerial_svd.model_spec import ModelSpec, validate_state


def save_checkpoint(path, spec, params, buffers, *, metadata=None, optimizer_leaves=()):
    validate_state(spec, params, buffers)
    arrays = {"p|" + k: np.asarray(v) for k, v in params.items()}
    arrays.update({"b|" + k: np.asarray(v) for k, v in buffers.items()})
    arrays.update({f"o|{i}": np.asarray(v) for i, v in enumerate(optimizer_leaves)})
    arrays["metadata"] = np.asarray(json.dumps({"format": "aerial-svd-jax", "version": 1,
        "spec": asdict(spec), "metadata": metadata or {}, "optimizer_leaves": len(optimizer_leaves)}))
    save_npz_new(path, arrays)


def load_checkpoint(path):
    with np.load(path, allow_pickle=False) as archive:
        header = json.loads(str(archive["metadata"].item()))
        if header.get("format") != "aerial-svd-jax" or header.get("version") != 1:
            raise ValueError("Unsupported JAX checkpoint")
        spec = ModelSpec(**header["spec"])
        params = {k[2:]: archive[k].copy() for k in archive.files if k.startswith("p|")}
        buffers = {k[2:]: archive[k].copy() for k in archive.files if k.startswith("b|")}
        optimizer = [archive[f"o|{i}"].copy() for i in range(header["optimizer_leaves"])]
        allowed = {"metadata"} | {"p|"+k for k in params} | {"b|"+k for k in buffers} | {f"o|{i}" for i in range(len(optimizer))}
        if set(archive.files) != allowed:
            raise ValueError("Unexpected checkpoint arrays")
    validate_state(spec, params, buffers)
    for name, value in {**params, **buffers}.items():
        expected_kind = "iu" if name.endswith("num_batches_tracked") else "f"
        if value.dtype.kind not in expected_kind or not np.isfinite(value).all():
            raise ValueError(f"Invalid checkpoint tensor {name}")
    return spec, params, buffers, header["metadata"], optimizer
