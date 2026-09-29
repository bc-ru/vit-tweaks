"""JIT training/evaluation with explicit optimizer and BatchNorm state."""
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax

from aerial_svd.jax_models import apply


def cross_entropy(logits, labels, smoothing=0.0):
    log_probs = jax.nn.log_softmax(logits.astype(jnp.float32), axis=-1)
    hard = -jnp.take_along_axis(log_probs, labels[:, None], axis=-1)[:, 0]
    return jnp.mean((1 - smoothing) * hard - smoothing * jnp.mean(log_probs, axis=-1))


def make_optimizer(weight_decay):
    # Multiply the entire AdamW update by the epoch LR outside this transform.
    return optax.adamw(learning_rate=1.0, b1=0.9, b2=0.999, eps=1e-8, eps_root=0.0, weight_decay=weight_decay)


class Engine:
    def __init__(self, spec, *, amp=False, smoothing=0.0, weight_decay=0.05):
        if not np.isfinite(smoothing) or not 0 <= smoothing <= 1:
            raise ValueError("label_smoothing must be in [0, 1]")
        if not np.isfinite(weight_decay) or weight_decay < 0:
            raise ValueError("weight_decay must be finite and nonnegative")
        self.spec = spec
        self.dtype = jnp.float16 if amp else jnp.float32
        self.smoothing = smoothing
        self.optimizer = make_optimizer(weight_decay)
        self.loss_scale = 65536.0 if self.dtype == jnp.float16 else 1.0
        self.successful_updates = 0
        self.forward = jax.jit(lambda params, buffers, images: apply(params, buffers, images, spec, dtype=self.dtype)[0])

        def loss(params, buffers, images, labels, scale):
            logits, new_buffers, _ = apply(params, buffers, images, spec, training=True, dtype=self.dtype)
            value = cross_entropy(logits, labels, smoothing)
            correct = jnp.sum(jnp.argmax(logits, axis=1) == labels)
            return value * scale, (value, correct, new_buffers)

        self.grad = jax.jit(jax.value_and_grad(loss, has_aux=True))

        def update(params, state, grads, lr):
            updates, state = self.optimizer.update(grads, state, params)
            updates = jax.tree.map(lambda x: lr * x, updates)
            return optax.apply_updates(params, updates), state

        self.update = jax.jit(update)
        self.add = jax.jit(lambda a, b: jax.tree.map(jnp.add, a, b))
        self.unscale = jax.jit(lambda g, scale: jax.tree.map(lambda x: x / scale, g))
        self.finite = jax.jit(lambda g: jnp.all(jnp.stack([jnp.all(jnp.isfinite(x)) for x in jax.tree.leaves(g)])))

    def train_epoch(self, params, buffers, optimizer_state, batches, lr, accumulation=1):
        if not np.isfinite(lr) or lr < 0:
            raise ValueError("learning rate must be finite and nonnegative")
        if accumulation < 1:
            raise ValueError("grad_accum_steps must be positive")
        total_loss = correct = samples = updates = skipped = 0
        accumulated = None
        for step, (images, labels) in enumerate(batches):
            (_, (loss, hits, buffers)), grads = self.grad(params, buffers, jnp.asarray(images), jnp.asarray(labels),
                                                         self.loss_scale / accumulation)
            accumulated = grads if accumulated is None else self.add(accumulated, grads)
            if (step + 1) % accumulation == 0:
                grads = self.unscale(accumulated, self.loss_scale)
                if bool(self.finite(grads)):
                    params, optimizer_state = self.update(params, optimizer_state, grads, lr)
                    updates += 1
                    self.successful_updates += 1
                    if self.dtype == jnp.float16 and self.successful_updates % 2000 == 0:
                        self.loss_scale *= 2
                elif self.dtype == jnp.float16:
                    self.loss_scale *= 0.5
                    self.successful_updates = 0
                    skipped += 1
                else:
                    raise FloatingPointError("Nonfinite float32 gradients")
                accumulated = None
            n = len(labels)
            total_loss += float(loss) * n
            correct += int(hits)
            samples += n
        if samples == 0:
            raise ValueError("No training batches (batch size may exceed split)")
        # Deliberately discard an incomplete accumulation group, matching reference.
        return params, buffers, optimizer_state, {"loss": total_loss / samples, "top1": correct / samples,
            "num_samples": samples, "optimizer_updates": updates, "skipped_updates": skipped}

    def evaluate(self, params, buffers, batches, smoothing=None):
        total_loss = correct = samples = 0
        smoothing = self.smoothing if smoothing is None else smoothing
        for images, labels in batches:
            logits = self.forward(params, buffers, jnp.asarray(images))
            total_loss += float(cross_entropy(logits, jnp.asarray(labels), smoothing)) * len(labels)
            correct += int(jnp.sum(logits.argmax(axis=1) == jnp.asarray(labels)))
            samples += len(labels)
        if samples == 0:
            raise ValueError("Empty evaluation split")
        return {"loss": total_loss / samples, "top1": correct / samples, "num_samples": samples}

    def benchmark(self, params, buffers, batches, batch_size, warmup=5, max_batches=50):
        if warmup < 0 or max_batches < 1:
            raise ValueError("warmup must be nonnegative and max_batches positive")
        times, samples, correct = [], 0, 0
        shapes = set()
        for step, (images, labels) in enumerate(batches):
            if step >= warmup + max_batches:
                break
            images = jax.device_put(images)
            images.block_until_ready()
            if images.shape not in shapes:
                self.forward(params, buffers, images).block_until_ready()
                shapes.add(images.shape)
            start = time.perf_counter()
            logits = self.forward(params, buffers, images).block_until_ready()
            elapsed = time.perf_counter() - start
            if step >= warmup:
                times.append(elapsed)
                samples += len(labels)
                correct += int(np.sum(np.asarray(logits).argmax(axis=1) == labels))
        total = sum(times)
        mean = total / max(len(times), 1)
        # No equivalent to resetting Torch's allocator peak is assumed here.
        device = jax.config.jax_default_device or jax.devices()[0]
        memory = device.memory_stats() or {}
        process_peak = memory.get("peak_bytes_in_use") if device.platform == "gpu" else None
        return {"bench_images": samples, "bench_top1": correct / max(samples, 1),
                "mean_batch_latency_ms": 1000 * mean, "mean_image_latency_ms": 1000 * mean / batch_size,
                "throughput_images_per_sec": samples / max(total, 1e-12), "peak_gpu_memory_bytes": None,
                "jax_process_peak_gpu_memory_bytes": process_peak}
