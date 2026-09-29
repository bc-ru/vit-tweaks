"""Pillow/NumPy ImageFolder loader preserving the frozen Torch dataset indices."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np
from PIL import Image, __version__ as pillow_version, features


def transform(image, size, rng=None):
    image = image.convert("RGB")
    if rng is not None:
        width, height = image.size
        for _ in range(10):
            area = width * height * rng.uniform(0.70, 1.0)
            ratio = math.exp(rng.uniform(math.log(3/4), math.log(4/3)))
            w, h = round(math.sqrt(area * ratio)), round(math.sqrt(area / ratio))
            if 0 < w <= width and 0 < h <= height:
                top = int(rng.integers(height - h + 1))
                left = int(rng.integers(width - w + 1))
                break
        else:
            ratio = width / height
            if ratio < 3/4:
                w, h = width, round(width / (3/4))
            elif ratio > 4/3:
                h, w = height, round(height * (4/3))
            else:
                w, h = width, height
            top, left = (height - h) // 2, (width - w) // 2
        image = image.crop((left, top, left + w, top + h))
    image = image.resize((size, size), Image.Resampling.BICUBIC)
    if rng is not None and rng.random() < 0.5:
        image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    array = np.asarray(image, dtype=np.float32).transpose(2, 0, 1) / np.float32(255)
    mean = np.asarray([0.485, 0.456, 0.406], np.float32)[:, None, None]
    std = np.asarray([0.229, 0.224, 0.225], np.float32)[:, None, None]
    return (array - mean) / std


class ImageFolder:
    def __init__(self, config):
        self.root = Path(config["data_dir"])
        self.classes = sorted(p.name for p in self.root.iterdir() if p.is_dir())
        self.class_to_idx = {name: i for i, name in enumerate(self.classes)}
        self.samples = []
        extensions = (".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm", ".tif", ".tiff", ".webp")
        for name in self.classes:
            start = len(self.samples)
            for root, _, files in sorted(os.walk(self.root / name, followlinks=True)):
                self.samples.extend((Path(root) / f, self.class_to_idx[name]) for f in sorted(files) if f.lower().endswith(extensions))
            if len(self.samples) == start:
                raise ValueError(f"Empty image class: {name}")
        with open(config["split_file"]) as f:
            split = json.load(f)
        if split["class_to_idx"] != self.class_to_idx:
            raise ValueError("Frozen split class mapping differs from ImageFolder")
        self.indices = split["indices"]
        if set(self.indices) != {"train", "val", "test"}:
            raise ValueError("Frozen split must contain train, val, test")
        all_indices = [i for values in self.indices.values() for i in values]
        if (any(type(i) is not int for i in all_indices)
                or len(all_indices) != len(self.samples) or sorted(all_indices) != list(range(len(self.samples)))):
            raise ValueError("Frozen split must partition the dataset exactly once")
        self.size = int(config.get("img_size", 224))
        self.seed = int(config.get("seed", 42))
        self.workers = int(config.get("num_workers", 0))
        self.decoder = {"pillow": pillow_version, "jpeg": features.version_codec("jpg"),
                        "libjpeg_turbo": features.check_feature("libjpeg_turbo")}
        if self.seed != 42:
            raise ValueError("This research protocol requires seed 42")
        manifest = [(str(path.relative_to(self.root)), label) for path, label in self.samples]
        self.fingerprint = hashlib.sha256(json.dumps({"samples": manifest, "split": split}, sort_keys=True).encode()).hexdigest()

    def batches(self, split, batch_size, epoch=0, max_batches=None):
        if batch_size < 1 or (max_batches is not None and max_batches < 1):
            raise ValueError("Batch size and batch cap must be positive")
        indices = np.asarray(self.indices[split], dtype=np.int64)
        train = split == "train"
        if train:
            indices = np.random.default_rng(np.random.SeedSequence([self.seed, epoch])).permutation(indices)
            indices = indices[:len(indices) // batch_size * batch_size]
        if max_batches is not None:
            indices = indices[:max_batches * batch_size]

        def read(index):
            path, label = self.samples[int(index)]
            rng = np.random.default_rng(np.random.SeedSequence([self.seed, epoch, int(index), 1])) if train else None
            with Image.open(path) as image:
                return transform(image, self.size, rng), label

        # Bound decoded-image memory to one batch; thread scheduling cannot alter RNG.
        with ThreadPoolExecutor(max_workers=max(1, self.workers)) as pool:
            for offset in range(0, len(indices), batch_size):
                rows = list(pool.map(read, indices[offset:offset + batch_size]))
                images, labels = zip(*rows)
                yield np.stack(images), np.asarray(labels, dtype=np.int32)
