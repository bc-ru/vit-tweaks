"""One-time, CPU-only conversion of trusted timm checkpoints/pretrained weights."""
import argparse
import hashlib
from pathlib import Path

import numpy as np
import timm
import torch

from aerial_svd.jax_checkpoint import save_checkpoint
from aerial_svd.model_spec import ModelSpec, parameter_shapes, validate_state
from aerial_svd.utils import load_yaml, seed_everything


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--config", required=True)
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint")
    source.add_argument("--pretrained", action="store_true", help="May download timm pretrained weights")
    p.add_argument("--num-classes", type=int, default=None)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    cfg = load_yaml(args.config)
    seed_everything(int(cfg.get("seed", 42)))
    metadata = {"source_backend": "torch", "torch_version": torch.__version__, "timm_version": timm.__version__}
    if args.checkpoint:
        checkpoint = Path(args.checkpoint)
        payload = torch.load(checkpoint, map_location="cpu")
        if isinstance(payload, dict) and "config" in payload:
            cfg.update(payload["config"])
        state = payload.get("model", payload.get("state_dict", payload)) if isinstance(payload, dict) else payload
        if not isinstance(state, dict):
            raise ValueError("Expected a state dictionary")
        head = "fc" if cfg["model"] == "resnet18" else "head"
        classifier = state.get(head + ".weight", state.get(head + ".second.weight"))
        if classifier is None:
            raise ValueError("Checkpoint has no supported classifier")
        classes = int(classifier.shape[0])
        if args.num_classes is not None and classes != args.num_classes:
            raise ValueError("Classifier size does not match --num-classes")
        digest = hashlib.sha256()
        with checkpoint.open("rb") as f:
            for chunk in iter(lambda: f.read(1024*1024), b""):
                digest.update(chunk)
        metadata.update(source_checkpoint=str(checkpoint), source_sha256=digest.hexdigest())
        if isinstance(payload, dict):
            metadata["source_metrics"] = payload.get("metrics", {})
    else:
        classes = args.num_classes or cfg.get("num_classes")
        if classes is None:
            import json
            with open(cfg["split_file"]) as f:
                classes = len(json.load(f)["class_to_idx"])
        state = timm.create_model(cfg["model"], pretrained=True, num_classes=int(classes)).state_dict()
        metadata["source_checkpoint"] = "timm pretrained initialization"
    spec = ModelSpec(cfg["model"], int(classes), int(cfg.get("img_size", 224)))
    _, bshapes = parameter_shapes(spec)
    params, buffers = {}, {}
    for name, value in state.items():
        array = value.detach().cpu().numpy()
        if name in bshapes:
            buffers[name] = array.astype(np.int32 if name.endswith("num_batches_tracked") else np.float32)
        else:
            params[name] = array.astype(np.float32)
    validate_state(spec, params, buffers)
    metadata["config"] = cfg
    metadata["conversion"] = "weights and buffers only; Torch optimizer state is not translated"
    save_checkpoint(args.output, spec, params, buffers, metadata=metadata)
    print(f"Converted {spec.model}: {len(params)} parameters, {len(buffers)} buffers -> {args.output}")


if __name__ == "__main__":
    main()
