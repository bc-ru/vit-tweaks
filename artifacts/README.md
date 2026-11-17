# Local artifacts

| Directory | Contents |
| --- | --- |
| `runs/` | Training checkpoints/history and the frozen AID split |
| `checkpoints/` | Reusable native JAX weights and historical compressed weights |
| `experiments/<id>/<stage>/` | Current protocol outputs, each with its own manifest |
| `results/`, `matrix_results/`, `plots/`, `reports/`, `logs/` | Existing measurements and figures |
| `archive/` | Historical notes/imports and migration verification evidence |

Generated content is local and Git-ignored. Keep every measured result with its
provenance; never replace an existing run. Root `runs/`, `checkpoints/`, etc. are
compatibility links, so old commands and frozen split paths still work.

Current source: `checkpoints/jax/vit_small_protocol_v1.npz` (SHA-256
`e5d9e02734f1a7233dcb35c67e8fb0491c26bbf14f2a23d7f38d8d13e5ba9391`).
Its original: `runs/aid_vit_small/best.pt`. The baseline/01/02 measurements are in
`experiments/lowrank_2026/protocol_v1_cpu_20261117/`; their manifests identify
inputs, code and exact outputs. `docs/lowrank/VALIDATION_RU.md` summarizes evidence.

`.cache/jax-torch/` is the optional existing cross-framework test environment.
`.scratch/` is disposable; it must not own canonical weights or project status.
