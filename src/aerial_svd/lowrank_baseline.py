"""00_baseline: identify dense weights and verify validation quality; never sample test."""
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from aerial_svd.lowrank_common import file_identity, load_inputs, parser, publish_json, stage_output


def main(argv=None):
    cli = parser("00_baseline")
    cli.add_argument("--data-dir")
    cli.add_argument("--batch-size", type=int, default=8)
    cli.add_argument("--max-val-batches", type=int, help="Smoke only: marks quality verification as partial")
    cli.add_argument("--expected-top1", type=float, help="Optional predeclared reference on the full validation split")
    cli.add_argument("--top1-atol", type=float, default=0.005)
    args = cli.parse_args(argv)
    if args.batch_size < 1 or (args.max_val_batches is not None and args.max_val_batches < 1):
        cli.error("Batch size and batch cap must be positive")
    if not np.isfinite(args.top1_atol) or args.top1_atol < 0:
        cli.error("top1-atol must be finite and nonnegative")
    if args.expected_top1 is not None and (not 0 <= args.expected_top1 <= 1 or args.max_val_batches is not None):
        cli.error("Expected top1 must be in [0, 1] and requires full validation")
    cfg, spec, params, buffers, metadata, provenance = load_inputs(args, "00_baseline")
    if args.data_dir:
        cfg["data_dir"] = args.data_dir
    from aerial_svd.jax_data import ImageFolder
    from aerial_svd.jax_engine import Engine
    dataset = ImageFolder(cfg)
    if not dataset.indices["val"]:
        raise ValueError("Validation is empty; create a frozen train-only subdivision first")
    saved = metadata.get("config", {})
    if spec.num_classes != len(dataset.classes) or saved.get("classes", dataset.classes) != dataset.classes:
        raise ValueError("Checkpoint classifier/class order disagrees with dataset")
    if metadata.get("dataset_fingerprint", dataset.fingerprint) != dataset.fingerprint:
        raise ValueError("Checkpoint dataset/split fingerprint changed")
    original = None
    if metadata.get("source_sha256"):
        original = file_identity(metadata["source_checkpoint"])
        if original["sha256"] != metadata["source_sha256"]:
            raise ValueError("Original checkpoint hash changed since conversion")
    params, buffers = jax.tree.map(jnp.asarray, (params, buffers))
    engine = Engine(spec, amp=False)
    provenance.update(dataset_fingerprint=dataset.fingerprint, classes=dataset.classes,
                      class_to_idx=dataset.class_to_idx, image_decoder=dataset.decoder,
                      preprocessing={"resize": "bicubic_square", "img_size": dataset.size,
                                     "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
                      split_counts={key: len(value) for key, value in dataset.indices.items()},
                      evaluation_split="val", original_checkpoint=original)
    with stage_output(args.output_dir, provenance) as out:
        metrics = engine.evaluate(params, buffers,
                                  dataset.batches("val", args.batch_size, max_batches=args.max_val_batches), smoothing=0)
        # Verify repeatability on precisely the same first validation batch.
        images, _ = next(dataset.batches("val", args.batch_size, max_batches=1))
        first = np.asarray(engine.forward(params, buffers, jnp.asarray(images)))
        second = np.asarray(engine.forward(params, buffers, jnp.asarray(images)))
        np.testing.assert_allclose(first, second, rtol=0, atol=0)
        if not np.isfinite(first).all() or not np.isfinite(metrics["loss"]) or not 0 <= metrics["top1"] <= 1:
            raise AssertionError("Invalid baseline metrics/logits")
        full = metrics["num_samples"] == len(dataset.indices["val"])
        if args.expected_top1 is not None and abs(metrics["top1"] - args.expected_top1) > args.top1_atol:
            raise AssertionError(f"Validation top1 {metrics['top1']} differs from reference {args.expected_top1}")
        publish_json(out / "baseline.json", dict(
            **provenance, validation=metrics, full_validation=full,
            quality_verification="full_validation" if full else "partial_smoke_only",
            reference_verified=args.expected_top1 is not None,
            source_metrics=metadata.get("source_metrics", {}),
            loss_definition="unsmoothed cross_entropy; historical training val_loss may be smoothed",
            deterministic_repeat=True, status="complete"))
    print(Path(args.output_dir) / "baseline.json")


if __name__ == "__main__":
    main()
