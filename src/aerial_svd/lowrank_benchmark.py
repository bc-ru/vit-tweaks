"""07_benchmark: warmed synchronized device-only latency, separate process per model."""
import hashlib
import time

import jax
import jax.numpy as jnp
import numpy as np

from aerial_svd.jax_engine import Engine
from aerial_svd.lowrank_common import load_inputs, parser, publish_json, stage_output
from aerial_svd.lowrank_quality import dataset_for


def main(argv=None):
    cli = parser('07_benchmark')
    cli.add_argument('--batch-sizes', nargs='+', type=int, default=[1, 16])
    cli.add_argument('--warmup', type=int, default=5)
    cli.add_argument('--repeats', type=int, default=50)
    args = cli.parse_args(argv)
    if min(args.batch_sizes) < 1 or args.warmup < 1 or args.repeats < 2:
        cli.error('Require positive batch sizes/warmup and at least two trials')
    cfg, spec, params, buffers, metadata, provenance = load_inputs(args, '07_benchmark', require_dense=False)
    dataset = dataset_for(cfg, spec, metadata, provenance)
    params, buffers = jax.tree.map(jnp.asarray, (params, buffers))
    jax.block_until_ready((params, buffers))
    engine = Engine(spec, amp=False)
    with stage_output(args.output_dir, provenance) as out:
        rows = []
        for bs in args.batch_sizes:
            images, _ = next(dataset.batches('val', bs, max_batches=1))
            if len(images) != bs:
                raise ValueError('Validation smaller than requested batch')
            pixels = hashlib.sha256(np.ascontiguousarray(images).tobytes()).hexdigest()
            images = jax.device_put(images)
            images.block_until_ready()
            start = time.perf_counter()
            executable = engine.forward.lower(params, buffers, images).compile()
            compile_seconds = time.perf_counter() - start
            analysis = executable.memory_analysis()
            memory = None if analysis is None else {name: int(getattr(analysis, name)) for name in
                ('argument_size_in_bytes', 'output_size_in_bytes', 'temp_size_in_bytes', 'alias_size_in_bytes')}
            if memory is not None:
                memory['total_bytes'] = (memory['argument_size_in_bytes'] + memory['output_size_in_bytes']
                                         + memory['temp_size_in_bytes'] - memory['alias_size_in_bytes'])
            for _ in range(args.warmup):
                engine.forward(params, buffers, images).block_until_ready()
            times = []
            for _ in range(args.repeats):
                start = time.perf_counter()
                engine.forward(params, buffers, images).block_until_ready()
                times.append(time.perf_counter()-start)
            rows.append(dict(batch_size=bs, median_batch_ms=1000*float(np.median(times)),
                             p05_batch_ms=1000*float(np.percentile(times, 5)),
                             p95_batch_ms=1000*float(np.percentile(times, 95)),
                             throughput_images_s=bs/float(np.median(times)), trials_seconds=times,
                             pixels_sha256=pixels, compile_seconds=compile_seconds,
                             compiled_memory=memory,
                             allocator_stats=jax.devices()[0].memory_stats()))
        publish_json(out / 'benchmark.json', dict(total_params=sum(v.size for v in params.values()),
            param_memory_bytes=sum(v.size*v.dtype.itemsize for v in params.values()),
            checkpoint_bytes=provenance['checkpoint']['bytes'], measurements=rows,
            inference_gpu_peak_bytes=None,
            memory_note='Inference-only runtime peak unavailable. compiled_memory is a static XLA plan; '
                        'allocator_stats are process-lifetime device allocator counters, including setup/compilation/warmup. '
                        'pool_bytes is reserved storage, not live computation memory. Parameter bytes are exact.',
            timing='float32; preprocessed fixed validation batch on device; compilation, transfers and preprocessing excluded'))


if __name__ == '__main__':
    main()
