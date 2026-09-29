"""Portable, pickle-free snapshots of checkpoint weight matrices."""
from __future__ import annotations

import json
import os
import re
import tempfile
from dataclasses import asdict
from pathlib import Path

import numpy as np

from aerial_svd.matrix_records import MatrixRecord


def save_archive(path, matrices, provenance):
    arrays = {}
    records = []
    for i, (record, matrix) in enumerate(matrices):
        key = f"matrix_{i}"
        array = np.asarray(matrix, dtype=np.float32)
        if (array.ndim != 2 or tuple(array.shape) != tuple(record.shape)
                or min(array.shape) == 0 or array.size != record.params):
            raise ValueError(f"Invalid matrix shape for {record.name}")
        if not np.isfinite(array).all():
            raise ValueError(f"Nonfinite matrix: {record.name}")
        arrays[key] = array
        records.append({"key": key, **asdict(record)})
    if not records:
        raise ValueError("No matrices selected")
    metadata = {"version": 1, "provenance": provenance, "records": records}
    arrays["metadata"] = np.asarray(json.dumps(metadata))
    save_npz_new(path, arrays)


def save_npz_new(path, arrays):
    """Publish a pickle-free NPZ without replacing any existing destination."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Publish complete data atomically; retries never replace existing evidence.
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as f:
        temporary = Path(f.name)
        try:
            np.savez(f, **arrays)
            f.flush()
            os.fsync(f.fileno())
            os.link(temporary, path)
        finally:
            temporary.unlink()


def load_archive(path):
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata"].item()))
        if metadata.get("version") != 1:
            raise ValueError("Unsupported matrix archive version")
        provenance = metadata["provenance"]
        # Model names become output filenames in the analysis commands.
        if (not isinstance(provenance.get("model"), str)
                or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", provenance["model"])
                or not isinstance(provenance.get("checkpoint"), str)):
            raise ValueError("Invalid archive provenance")
        matrices = []
        names = set()
        for entry in metadata["records"]:
            entry = dict(entry)
            key = entry.pop("key")
            entry["shape"] = tuple(entry["shape"])
            entry["original_shape"] = tuple(entry["original_shape"])
            record = MatrixRecord(**entry)
            array = archive[key]
            if (array.dtype != np.float32 or array.ndim != 2
                    or array.shape != record.shape or min(array.shape) == 0
                    or array.size != record.params or not np.isfinite(array).all()
                    or int(np.prod(record.original_shape)) != record.params
                    or record.name in names):
                raise ValueError(f"Invalid archived matrix: {record.name}")
            names.add(record.name)
            matrices.append((record, array.copy()))
    if not matrices:
        raise ValueError("Empty matrix archive")
    return matrices, provenance
