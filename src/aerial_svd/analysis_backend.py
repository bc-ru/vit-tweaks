"""Opt-in JAX archive inputs for the existing matrix-analysis commands."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from aerial_svd.matrix_archive import load_archive


def add_backend_arguments(parser):
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--config")
    source.add_argument("--matrix-archive", help="Portable archive from aerial_svd.export_matrices")
    parser.add_argument("--backend", choices=("torch", "jax"), default="torch")


def prepare_analysis(args):
    if args.backend == "torch":
        if args.matrix_archive:
            raise ValueError("--matrix-archive requires --backend jax")
        from aerial_svd import matrix_analysis as backend
        model, cfg, checkpoint = backend.load_model_from_config(args.config, checkpoint=args.checkpoint, device=args.device)
        matrices = backend.filter_matrices(backend.iter_weight_matrices(model), args.include_regex, args.exclude_regex)
        return backend, cfg["model"], checkpoint, matrices

    if args.matrix_archive and args.checkpoint:
        raise ValueError("--checkpoint cannot be combined with --matrix-archive")
    if args.device not in ("cpu", "gpu"):
        raise ValueError("JAX analysis --device must be cpu or gpu")
    import jax
    from aerial_svd import jax_matrix_analysis as backend
    if args.matrix_archive:
        source_path = Path(args.matrix_archive)
        matrices, provenance = load_archive(source_path)
    else:
        import yaml
        from aerial_svd.jax_checkpoint import load_checkpoint
        from aerial_svd.jax_compression import weight_matrices
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
        source_path = Path(args.checkpoint or Path(cfg.get("output_dir", "runs/default")) / "best.npz")
        spec, params, _, metadata, _ = load_checkpoint(source_path)
        if spec.model != cfg["model"]:
            raise ValueError("Checkpoint architecture disagrees with config")
        matrices = list(weight_matrices(params))
        provenance = {"model": spec.model, "checkpoint": str(source_path), "config": cfg, "checkpoint_metadata": metadata}
    inc = re.compile(args.include_regex) if args.include_regex else None
    exc = re.compile(args.exclude_regex) if args.exclude_regex else None
    matrices = [(rec, W) for rec, W in matrices
                if (inc is None or inc.search(f"{rec.name} {rec.kind}"))
                and (exc is None or not exc.search(f"{rec.name} {rec.kind}"))]
    if not matrices:
        raise ValueError("No matrices matched the requested filters")
    device = jax.devices(args.device)[0]
    # Explicit device placement also applies to PRNG and benchmark kernels.
    jax.config.update("jax_default_device", device)
    matrices = [(rec, jax.device_put(W, device)) for rec, W in matrices]
    out = Path(args.out_dir)
    # A fresh directory separates backend evidence and makes reruns fail safely.
    out.mkdir(parents=True, exist_ok=False)
    hasher = hashlib.sha256()
    with source_path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024*1024), b""):
            hasher.update(chunk)
    digest = hasher.hexdigest()
    manifest = {
        "status": "running", "backend": "jax", "jax_version": jax.__version__,
        "device": str(device), "platform": device.platform,
        "archive_sha256": digest, "source": provenance, "arguments": vars(args),
        "svd_dtype": "float32", "energy_dtype": "float64 (NumPy host)",
        "random_generator": "jax.random; same seed does not imply Torch-identical samples",
        "timing": "synchronized warmed runtime; compilation and input transfer excluded",
    }
    (out / "provenance.json").write_text(json.dumps(manifest, indent=2))
    return backend, provenance["model"], provenance["checkpoint"], matrices


def finish_analysis(args):
    if args.backend == "jax":
        path = Path(args.out_dir) / "provenance.json"
        metadata = json.loads(path.read_text())
        metadata["status"] = "complete"
        path.write_text(json.dumps(metadata, indent=2))
