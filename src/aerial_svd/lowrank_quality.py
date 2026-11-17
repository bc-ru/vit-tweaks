"""Shared validation identity and paired metrics; never reads test images."""
import hashlib
import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from aerial_svd.jax_data import ImageFolder
from aerial_svd.lowrank_common import file_identity


def verified_stage(directory):
    directory = Path(directory)
    manifest = json.loads((directory / 'manifest.json').read_text())
    if manifest['status'] != 'complete':
        raise ValueError(f'Incomplete stage: {directory}')
    for name, identity in manifest['outputs'].items():
        if file_identity(directory / name)['sha256'] != identity['sha256']:
            raise ValueError(f'Changed stage output: {directory / name}')
    return manifest


def compatible_stage(directory, provenance, stage):
    manifest = verified_stage(directory)
    if manifest['stage'] != stage:
        raise ValueError(f'Expected {stage}')
    for key in ('checkpoint', 'config', 'aid_split'):
        if manifest[key]['sha256'] != provenance[key]['sha256']:
            raise ValueError(f'Activation input mismatch: {key}')
    return manifest


def dataset_for(cfg, spec, metadata, provenance):
    dataset = ImageFolder(cfg)
    if not dataset.indices['val'] or len(dataset.classes) != spec.num_classes:
        raise ValueError('Require nonempty validation and matching class count')
    saved = metadata.get('config', {})
    if 'classes' in saved and saved['classes'] != dataset.classes:
        raise ValueError('Checkpoint class order changed')
    if metadata.get('dataset_fingerprint', dataset.fingerprint) != dataset.fingerprint:
        raise ValueError('Checkpoint dataset/split fingerprint changed')
    provenance.update(dataset_fingerprint=dataset.fingerprint, classes=dataset.classes,
                      class_to_idx=dataset.class_to_idx, image_decoder=dataset.decoder,
                      split_counts={k: len(v) for k, v in dataset.indices.items()},
                      preprocessing={'resize': 'bicubic_square', 'img_size': dataset.size,
                                     'mean': [0.485, .456, .406], 'std': [.229, .224, .225]},
                      evaluation_split='val', calibration_subset=None)
    return dataset


def collect_logits(engine, params, buffers, batches):
    logits, targets = [], []
    digest = hashlib.sha256()
    for images, labels in batches:
        digest.update(np.ascontiguousarray(images).tobytes())
        logits.append(np.asarray(engine.forward(params, buffers, jnp.asarray(images))))
        targets.append(labels)
    if not logits:
        raise ValueError('Empty validation')
    return np.concatenate(logits), np.concatenate(targets), digest.hexdigest()


def paired_metrics(logits, labels, reference):
    logits = np.asarray(logits, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    if (logits.shape != reference.shape or labels.shape != (len(logits),)
            or not np.isfinite(logits).all() or not np.isfinite(reference).all()):
        raise ValueError('Invalid paired logits')
    shifted = logits - logits.max(axis=1, keepdims=True)
    ce = np.log(np.exp(shifted).sum(axis=1)) - shifted[np.arange(len(labels)), labels]
    drift = np.linalg.norm(logits - reference, axis=1)
    relative = np.linalg.norm(logits - reference) / max(np.linalg.norm(reference), 1e-30)
    return dict(top1=float(np.mean(logits.argmax(axis=1) == labels)), loss=float(ce.mean()),
                num_samples=len(labels), logit_l2_mean=float(drift.mean()),
                logit_l2_p95=float(np.percentile(drift, 95)),
                logit_relative_frobenius=float(relative), logit_max_abs=float(np.max(np.abs(logits-reference))))


def load_reference(directory, provenance, dataset, source_hash):
    manifest = verified_stage(directory)
    for key in ('dataset_fingerprint', 'classes', 'image_decoder', 'preprocessing'):
        if manifest[key] != provenance[key]:
            raise ValueError(f'Reference mismatch: {key}')
    if (manifest['checkpoint']['sha256'] != source_hash
            or manifest['aid_split']['sha256'] != provenance['aid_split']['sha256']):
        raise ValueError('Reference checkpoint/split mismatch')
    with np.load(Path(directory) / 'logits.npz', allow_pickle=False) as archive:
        reference = {k: archive[k].copy() for k in archive.files}
    if not np.array_equal(reference['indices'], dataset.indices['val']):
        raise ValueError('Reference validation order differs')
    return reference, manifest
