import json
from pathlib import Path
import tempfile
import unittest

import jax
import jax.numpy as jnp
import numpy as np
import torch
from PIL import Image
from torchvision.datasets import ImageFolder as TorchImageFolder

from aerial_svd.data import build_transforms
from aerial_svd.jax_data import ImageFolder, transform
from aerial_svd.jax_engine import Engine, cross_entropy, make_optimizer
from aerial_svd.jax_checkpoint import load_checkpoint, save_checkpoint
from aerial_svd.jax_models import initialize, apply
from aerial_svd.model_spec import ModelSpec, validate_state
from aerial_svd.jax_compression import compress, weight_matrices


class OptimizerParity(unittest.TestCase):
    def test_invalid_optimizer_configuration_rejected(self):
        spec = ModelSpec("resnet18", 3, 32)
        for kwargs in ({"weight_decay": -1}, {"weight_decay": float("nan")}, {"smoothing": 2}):
            with self.assertRaises(ValueError):
                Engine(spec, **kwargs)
        with self.assertRaises(ValueError):
            Engine(spec).train_epoch({}, {}, None, [], lr=-1)

    def test_resnet_training_batchnorm_state(self):
        import timm
        torch.manual_seed(42)
        spec = ModelSpec("resnet18", 3, 32)
        reference = timm.create_model("resnet18", pretrained=False, num_classes=3).train()
        params, buffers = {}, {}
        for name, value in reference.state_dict().items():
            target = buffers if name.endswith(("running_mean", "running_var", "num_batches_tracked")) else params
            # CPU JAX may share NumPy storage: freeze buffers before Torch mutates them.
            target[name] = jnp.asarray(value.numpy().copy())
        images = np.random.default_rng(42).normal(size=(2,3,32,32)).astype(np.float32)
        with torch.no_grad():
            expected = reference(torch.from_numpy(images)).numpy()
        actual, updated, _ = jax.jit(lambda p,b,x: apply(p,b,x,spec,training=True))(params,buffers,jnp.asarray(images))
        np.testing.assert_allclose(actual, expected, rtol=3e-4, atol=3e-4)
        state = reference.state_dict()
        for name, value in updated.items():
            np.testing.assert_allclose(value, state[name].numpy(), rtol=2e-4, atol=2e-5, err_msg=name)

    def test_smoothed_loss_gradient_and_adamw(self):
        rng = np.random.default_rng(42)
        W = rng.normal(size=(5, 3)).astype(np.float32)
        x = rng.normal(size=(4, 5)).astype(np.float32)
        y = np.array([0, 1, 2, 0], np.int32)
        parameter = torch.nn.Parameter(torch.from_numpy(W.copy()))
        optimizer = torch.optim.AdamW([parameter], lr=3e-4, weight_decay=.05)
        candidate_optimizer = make_optimizer(.05)
        params = {"weight": jnp.asarray(W)}
        state = candidate_optimizer.init(params)
        for _ in range(3):
            optimizer.zero_grad()
            loss = torch.nn.functional.cross_entropy(torch.from_numpy(x) @ parameter, torch.from_numpy(y).long(), label_smoothing=.1)
            loss.backward()
            value, grads = jax.value_and_grad(lambda p: cross_entropy(jnp.asarray(x) @ p["weight"], jnp.asarray(y), .1))(params)
            np.testing.assert_allclose(value, loss.detach().numpy(), rtol=2e-6)
            np.testing.assert_allclose(grads["weight"], parameter.grad.numpy(), rtol=3e-5, atol=2e-7)
            optimizer.step()
            updates, state = candidate_optimizer.update(grads, state, params)
            params = jax.tree.map(lambda p, u: p + 3e-4*u, params, updates)
            np.testing.assert_allclose(params["weight"], parameter.detach().numpy(), rtol=2e-6, atol=2e-7)


class NativeData(unittest.TestCase):
    def test_eval_transform_exact(self):
        image = Image.fromarray(np.random.default_rng(42).integers(0, 256, (83, 101, 3), dtype=np.uint8))
        np.testing.assert_array_equal(transform(image, 32), build_transforms(32, False)(image).numpy())

    def test_frozen_indices_order_and_deterministic_augmentation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "images"
            for name in ("B", "A"):
                (root/name/"nested").mkdir(parents=True)
                image = Image.fromarray(np.random.default_rng(42).integers(0, 256, (40, 50, 3), dtype=np.uint8))
                image.save(root/name/"z.png")
                image.save(root/name/"nested"/"a.png")
            split = Path(tmp)/"split.json"
            split.write_text(json.dumps(dict(class_to_idx={"A":0, "B":1}, indices={"train":[0,2], "val":[1], "test":[3]})))
            cfg = dict(data_dir=str(root), split_file=str(split), seed=42, img_size=32, num_workers=2)
            dataset = ImageFolder(cfg)
            reference = TorchImageFolder(root)
            self.assertEqual([(str(p), y) for p,y in dataset.samples], reference.samples)
            a = list(dataset.batches("train", 2, epoch=1))
            b = list(dataset.batches("train", 2, epoch=1))
            np.testing.assert_array_equal(a[0][0], b[0][0])
            self.assertEqual(len(list(dataset.batches("test", 2))[0][1]), 1)
            self.assertEqual(list(dataset.batches("train", 3)), [])
            data = json.loads(split.read_text()); data["indices"]["test"] = [0]; split.write_text(json.dumps(data))
            with self.assertRaises(ValueError):
                ImageFolder(cfg)


class CompressionContract(unittest.TestCase):
    def test_matrix_selection_uses_model_order_not_pytree_order(self):
        names = ["blocks.10.attn.qkv", "head", "blocks.2.attn.qkv", "blocks.0.attn.qkv.second", "blocks.0.attn.qkv.first"]
        params = {name + ".weight": jnp.ones((2,2)) for name in names}
        params["patch_embed.proj.weight"] = jnp.ones((2,3,2,2))
        actual = [record.name for record, _ in weight_matrices(params)]
        self.assertEqual(actual, ["patch_embed.proj", "blocks.0.attn.qkv.first", "blocks.0.attn.qkv.second",
                                  "blocks.2.attn.qkv", "blocks.10.attn.qkv", "head"])

    def test_factors_preserve_symmetric_split_and_skip(self):
        W = np.diag(np.array([8, 4, 2, 1], np.float32))
        params = {"blocks.0.mlp.fc1.weight": jnp.asarray(W), "blocks.0.mlp.fc1.bias": jnp.ones(4)}
        compressed, records = compress(params, "ffn", .25)
        first, second = compressed["blocks.0.mlp.fc1.first.weight"], compressed["blocks.0.mlp.fc1.second.weight"]
        self.assertAlmostEqual(float(jnp.linalg.norm(first)), np.sqrt(8), places=5)
        self.assertAlmostEqual(float(jnp.linalg.norm(second)), np.sqrt(8), places=5)
        np.testing.assert_allclose(second @ first, np.diag([8,0,0,0]), atol=2e-6)
        np.testing.assert_array_equal(compressed["blocks.0.mlp.fc1.second.bias"], np.ones(4))
        self.assertEqual(records[0]["rank"], 1)
        skipped, records = compress(params, "ffn", .75)
        self.assertEqual(set(skipped), set(params)); self.assertEqual(records, [])

    def test_native_roundtrip_and_structure_validation(self):
        spec = ModelSpec("resnet18", 3, 32)
        params, buffers = initialize(spec)
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)/"state.npz"
            save_checkpoint(p, spec, params, buffers, metadata={"epoch": 1}, optimizer_leaves=[np.array(2, np.int32)])
            loaded_spec, loaded, state, metadata, leaves = load_checkpoint(p)
            self.assertEqual(loaded_spec, spec); self.assertEqual(metadata["epoch"], 1)
            np.testing.assert_array_equal(leaves[0], 2)
            np.testing.assert_array_equal(loaded["fc.weight"], params["fc.weight"])
            self.assertEqual(set(state), set(buffers))
            with self.assertRaises(FileExistsError):
                save_checkpoint(p, spec, params, buffers)
            del loaded["fc.bias"]
            with self.assertRaises(ValueError):
                validate_state(spec, loaded, state)


if __name__ == "__main__":
    unittest.main()
