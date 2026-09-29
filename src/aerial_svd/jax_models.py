"""Functional JAX versions of the three configured timm models.

Parameter names and OI/OIHW layouts deliberately match timm checkpoints. The
supported variants have zero dropout/drop-path and fixed 224px ViT inputs.
"""
import math

import jax
import jax.numpy as jnp
import numpy as np


from aerial_svd.model_spec import ModelSpec, parameter_shapes, validate_state, is_target, linear_names


def initialize(spec, seed=42):
    shapes, bshapes = parameter_shapes(spec)
    rng = np.random.default_rng(seed)
    params = {}
    for name, shape in shapes.items():
        if name.startswith("patch_embed.proj") or (spec.model == "resnet18" and name.startswith("fc.")):
            fan_in = 3*16*16 if name.startswith("patch_embed") else 512
            value = rng.uniform(-1/math.sqrt(fan_in), 1/math.sqrt(fan_in), shape).astype(np.float32)
        elif name.endswith("bias") or (spec.model == "resnet18" and name.endswith(".bn2.weight")):
            value = np.zeros(shape, np.float32)
        elif len(shape) == 1:
            value = np.ones(shape, np.float32)
        else:
            if spec.model == "resnet18" and len(shape) == 4:
                scale = math.sqrt(2 / (shape[0] * shape[2] * shape[3]))
            elif name == "cls_token":
                scale = 1e-6
            else:
                scale = 0.02
            value = rng.normal(0, scale, shape).astype(np.float32)
            if spec.model.startswith("vit_") and name != "cls_token":
                # timm trunc_normal_(std=.02) uses absolute bounds [-2, 2].
                outside = np.abs(value) > 2
                while outside.any():
                    value[outside] = rng.normal(0, scale, int(outside.sum()))
                    outside = np.abs(value) > 2
        params[name] = jnp.asarray(value)
    buffers = {name: jnp.asarray(np.ones(shape, np.float32) if name.endswith("running_var")
                                else np.zeros(shape, np.int32 if not shape else np.float32))
               for name, shape in bshapes.items()}
    return params, buffers


def dense_weight(params, name):
    if name + ".weight" in params:
        return params[name + ".weight"]
    return params[name + ".second.weight"] @ params[name + ".first.weight"]


def apply(params, buffers, images, spec, training=False, capture=(), dtype=jnp.float32):
    captured = {}
    state = dict(buffers)

    def linear(x, name):
        original = x
        x = x.astype(dtype)
        if name + ".weight" in params:
            x = x @ params[name + ".weight"].astype(dtype).T
            bias = params.get(name + ".bias")
        else:
            x = x @ params[name + ".first.weight"].astype(dtype).T
            if name + ".first" in capture:
                captured[name + ".first"] = (original, x)
            middle = x
            x = x @ params[name + ".second.weight"].astype(dtype).T
            bias = params.get(name + ".second.bias")
        if bias is not None:
            x = x + bias.astype(dtype)
        if name + ".second" in capture and name + ".first.weight" in params:
            captured[name + ".second"] = (middle, x)
        if name in capture:
            captured[name] = (original, x)
        return x

    def conv(x, name, stride=1, padding=0):
        y = jax.lax.conv_general_dilated(x.astype(dtype), params[name + ".weight"].astype(dtype),
                                       (stride, stride), ((padding, padding), (padding, padding)),
                                       dimension_numbers=("NCHW", "OIHW", "NCHW"))
        if name + ".bias" in params:
            y = y + params[name + ".bias"].astype(dtype)[None, :, None, None]
        return y

    def norm(x, name):
        xf = x.astype(jnp.float32)
        mean = jnp.mean(xf, axis=-1, keepdims=True)
        variance = jnp.mean(jnp.square(xf - mean), axis=-1, keepdims=True)
        return (xf - mean) * jax.lax.rsqrt(variance + 1e-6) * params[name + ".weight"] + params[name + ".bias"]

    def bn(x, name):
        xf = x.astype(jnp.float32)
        if training:
            mean = jnp.mean(xf, axis=(0, 2, 3))
            variance = jnp.mean(jnp.square(xf - mean[None, :, None, None]), axis=(0, 2, 3))
            n = x.shape[0] * x.shape[2] * x.shape[3]
            if n <= 1:
                raise ValueError("BatchNorm training needs more than one value per channel")
            state[name + ".running_mean"] = 0.9 * buffers[name + ".running_mean"] + 0.1 * mean
            state[name + ".running_var"] = 0.9 * buffers[name + ".running_var"] + 0.1 * variance * n / (n - 1)
            state[name + ".num_batches_tracked"] = buffers[name + ".num_batches_tracked"] + 1
        else:
            mean, variance = buffers[name + ".running_mean"], buffers[name + ".running_var"]
        scale = jax.lax.rsqrt(variance + 1e-5) * params[name + ".weight"]
        return ((xf - mean[None, :, None, None]) * scale[None, :, None, None]
                + params[name + ".bias"][None, :, None, None]).astype(dtype)

    with jax.default_matmul_precision("highest"):
        x = images.astype(dtype)
        if spec.model.startswith("vit_"):
            if x.shape[2:] != (224, 224):
                raise ValueError("ViT inputs must be NCHW 224x224")
            x = conv(x, "patch_embed.proj", stride=16)
            x = x.reshape(x.shape[0], spec.width, -1).transpose(0, 2, 1)
            x = jnp.concatenate((jnp.broadcast_to(params["cls_token"], (x.shape[0], 1, spec.width)), x), axis=1)
            x = x + params["pos_embed"]
            heads = spec.width // 64
            for i in range(12):
                base = f"blocks.{i}"
                y = norm(x, base + ".norm1")
                qkv = linear(y, base + ".attn.qkv").reshape(x.shape[0], x.shape[1], 3, heads, 64).transpose(2, 0, 3, 1, 4)
                q, k, v = qkv[0], qkv[1], qkv[2]
                attention = jax.nn.softmax(((q / 8) @ k.swapaxes(-1, -2)).astype(jnp.float32), axis=-1).astype(dtype)
                y = (attention @ v).transpose(0, 2, 1, 3).reshape(x.shape)
                x = x + linear(y, base + ".attn.proj")
                y = linear(norm(x, base + ".norm2"), base + ".mlp.fc1")
                y = jax.nn.gelu(y, approximate=False)
                x = x + linear(y, base + ".mlp.fc2")
            logits = linear(norm(x, "norm")[:, 0], "head")
        else:
            x = jax.nn.relu(bn(conv(x, "conv1", 2, 3), "bn1"))
            x = jax.lax.reduce_window(x, -jnp.inf, jax.lax.max, (1, 1, 3, 3), (1, 1, 2, 2),
                                      ((0, 0), (0, 0), (1, 1), (1, 1)))
            for stage in range(1, 5):
                for block in range(2):
                    base = f"layer{stage}.{block}"
                    stride = 2 if stage > 1 and block == 0 else 1
                    shortcut = x
                    y = jax.nn.relu(bn(conv(x, base + ".conv1", stride, 1), base + ".bn1"))
                    y = bn(conv(y, base + ".conv2", 1, 1), base + ".bn2")
                    if stride == 2:
                        shortcut = bn(conv(x, base + ".downsample.0", 2), base + ".downsample.1")
                    x = jax.nn.relu(y + shortcut)
            logits = linear(jnp.mean(x, axis=(2, 3)), "fc")
    return logits, state, captured
