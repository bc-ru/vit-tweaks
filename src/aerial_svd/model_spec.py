"""Portable architecture and parameter-shape contract; no framework imports."""
from dataclasses import dataclass

@dataclass(frozen=True)
class ModelSpec:
    model: str
    num_classes: int
    img_size: int = 224

    def __post_init__(self):
        if self.model not in ("vit_small_patch16_224", "vit_base_patch16_224", "resnet18"):
            raise ValueError(f"Unsupported architecture: {self.model}")
        if self.num_classes < 1 or self.img_size < 32:
            raise ValueError("Invalid class count or image size")
        if self.model.startswith("vit_") and self.img_size != 224:
            raise ValueError("Configured timm ViTs require 224px inputs")

    @property
    def width(self):
        return 384 if self.model == "vit_small_patch16_224" else 768


def parameter_shapes(spec):
    shapes, buffers = {}, {}

    def linear(name, n, m):
        shapes[name + ".weight"] = (m, n)
        shapes[name + ".bias"] = (m,)

    def norm(name, width, batch=False):
        shapes[name + ".weight"] = (width,)
        shapes[name + ".bias"] = (width,)
        if batch:
            buffers[name + ".running_mean"] = (width,)
            buffers[name + ".running_var"] = (width,)
            buffers[name + ".num_batches_tracked"] = ()

    if spec.model.startswith("vit_"):
        d = spec.width
        shapes.update({"patch_embed.proj.weight": (d, 3, 16, 16), "patch_embed.proj.bias": (d,),
                       "cls_token": (1, 1, d), "pos_embed": (1, 197, d)})
        for i in range(12):
            base = f"blocks.{i}"
            norm(base + ".norm1", d)
            norm(base + ".norm2", d)
            for name, n, m in (("attn.qkv", d, 3*d), ("attn.proj", d, d),
                               ("mlp.fc1", d, 4*d), ("mlp.fc2", 4*d, d)):
                linear(base + "." + name, n, m)
        norm("norm", d)
        linear("head", d, spec.num_classes)
    else:
        shapes["conv1.weight"] = (64, 3, 7, 7)
        norm("bn1", 64, True)
        previous = 64
        for stage, width in enumerate((64, 128, 256, 512), 1):
            for block in range(2):
                base = f"layer{stage}.{block}"
                shapes[base + ".conv1.weight"] = (width, previous, 3, 3)
                shapes[base + ".conv2.weight"] = (width, width, 3, 3)
                norm(base + ".bn1", width, True)
                norm(base + ".bn2", width, True)
                if stage > 1 and block == 0:
                    shapes[base + ".downsample.0.weight"] = (width, previous, 1, 1)
                    norm(base + ".downsample.1", width, True)
                previous = width
        linear("fc", 512, spec.num_classes)
    return shapes, buffers


def validate_state(spec, params, buffers):
    expected, expected_buffers = parameter_shapes(spec)
    # A compressed layer replaces W and b with first/second factors and b2.
    for key in list(params):
        if key.endswith(".first.weight"):
            base = key[:-len(".first.weight")]
            shape = expected.pop(base + ".weight", None)
            if shape is None or len(shape) != 2 or not is_target(base, "all"):
                raise ValueError(f"Unexpected low-rank layer: {base}")
            m, n = shape
            r = params[key].shape[0]
            if not 1 <= r <= min(m, n):
                raise ValueError(f"Invalid rank for {base}")
            expected[key] = (r, n)
            expected[base + ".second.weight"] = (m, r)
            if base + ".bias" in expected:
                expected[base + ".second.bias"] = expected.pop(base + ".bias")
    for label, actual, schema in (("parameters", params, expected), ("buffers", buffers, expected_buffers)):
        if set(actual) != set(schema):
            raise ValueError(f"Invalid {label}: missing={set(schema)-set(actual)}, unexpected={set(actual)-set(schema)}")
        for key, shape in schema.items():
            if tuple(actual[key].shape) != shape:
                raise ValueError(f"Invalid shape for {key}: {actual[key].shape} != {shape}")


def is_target(name, mode):
    if mode not in ("ffn", "attn", "all"):
        raise ValueError("mode must be ffn, attn, or all")
    attn = "attn.qkv" in name or "attn.proj" in name
    ffn = "mlp.fc1" in name or "mlp.fc2" in name
    return (mode == "ffn" and ffn) or (mode == "attn" and attn) or (mode == "all" and (attn or ffn or name.endswith("head")))


def linear_names(params):
    names = {key[:-7] for key, value in params.items() if key.endswith(".weight") and value.ndim == 2
             and not key.endswith((".first.weight", ".second.weight"))}
    names |= {key[:-len(".first.weight")] for key in params if key.endswith(".first.weight")}
    def order(name):
        parts = name.split(".")
        if parts[0] == "blocks":
            return (int(parts[1]), {"attn.qkv": 0, "attn.proj": 1, "mlp.fc1": 2, "mlp.fc2": 3}[".".join(parts[2:])])
        return (1000, name)
    return sorted(names, key=order)
