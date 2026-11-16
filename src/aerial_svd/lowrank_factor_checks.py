"""01_factor_checks: exercise the real compression path and mandatory full-rank control."""
import time

import jax
import jax.numpy as jnp
import numpy as np

from aerial_svd.jax_compression import compress
from aerial_svd.lowrank_common import load_inputs, parser, publish_json, rank_requests, selected_names, stage_output
from aerial_svd.matrix_archive import save_npz_new


# Float32 epsilon is 1.19e-7. Normwise 5e-5 accommodates SVD + factor
# multiplication rounding on 384-wide ViT projections, while detecting skips,
# transpositions, wrong ranks and asymmetric scaling. Tests also use fp64 SVD.
FLOAT32_RELATIVE_TOLERANCE = 5e-5


def relative_error(actual, expected):
    delta = np.linalg.norm(np.asarray(actual, np.float64) - np.asarray(expected, np.float64))
    norm = np.linalg.norm(np.asarray(expected, np.float64))
    return float(delta / norm) if norm else float(delta)


def main(argv=None):
    args = parser("01_factor_checks", matrices=True).parse_args(argv)
    _, _, params, _, _, provenance = load_inputs(args, "01_factor_checks")
    names = selected_names(params, args)
    requests = {name: rank_requests(params[name + ".weight"].shape, args) for name in names}
    provenance.update(selected_matrices=names, ranks=requests, full_rank_control=True,
                      relative_tolerance=FLOAT32_RELATIVE_TOLERANCE,
                      svd_timing="compression call including synchronized exact SVD and factors; reference SVD excluded")
    with stage_output(args.output_dir, provenance) as out:
        rows, arrays = [], {}
        for name in names:
            weight = params[name + ".weight"]
            subset = {key: value for key, value in params.items() if key in (name + ".weight", name + ".bias")}
            # Use the same float32 SVD convention: at repeated singular values a
            # float64 solver may choose a different, equally optimal subspace.
            u, s, vh = map(np.asarray, jax.block_until_ready(jnp.linalg.svd(jnp.asarray(weight), full_matrices=False)))
            limit = min(weight.shape)
            checks = [dict(request, control=False) for request in requests[name]]
            if not any(request["actual_rank"] == limit for request in checks):
                checks.append(dict(requested_rank=limit, requested_ratio=1.0, actual_rank=limit, control=True))
            for request in checks:
                rank = request["actual_rank"]
                start = time.perf_counter()
                result, records = compress(subset, args.scope, ranks={name: rank})
                jax.block_until_ready(result)
                elapsed = time.perf_counter() - start
                if [record["name"] for record in records] != [name] or records[0]["rank"] != rank:
                    raise AssertionError("Selected matrix skipped or wrong rank returned")
                first = np.asarray(result[name + ".first.weight"])
                second = np.asarray(result[name + ".second.weight"])
                if first.shape != (rank, weight.shape[1]) or second.shape != (weight.shape[0], rank):
                    raise AssertionError("Wrong factor orientation")
                reconstructed = second.astype(np.float64) @ first.astype(np.float64)
                reference = (u[:, :rank].astype(np.float64) * s[:rank]) @ vh[:rank].astype(np.float64)
                error = relative_error(reconstructed, reference)
                denominator = np.linalg.norm(weight.astype(np.float64))
                residual = np.linalg.norm(weight.astype(np.float64) - reconstructed)
                optimal_residual = np.linalg.norm(s[rank:].astype(np.float64))
                optimality_error = float(abs(residual - optimal_residual) / denominator) if denominator else float(residual)
                full_error = relative_error(reconstructed, weight) if rank == limit else None
                if (not np.isfinite(reconstructed).all() or error > FLOAT32_RELATIVE_TOLERANCE
                        or optimality_error > FLOAT32_RELATIVE_TOLERANCE
                        or (full_error is not None and full_error > FLOAT32_RELATIVE_TOLERANCE)):
                    raise AssertionError(f"SVD reconstruction failed: {name}, rank={rank}, error={error}")
                if name + ".weight" in result or name + ".bias" in result:
                    raise AssertionError("Dense matrix/bias left behind")
                if name + ".bias" in subset:
                    np.testing.assert_array_equal(result[name + ".second.bias"], subset[name + ".bias"])
                prefix = f"check_{len(rows)}|"
                arrays.update({prefix + key: np.asarray(value) for key, value in result.items()})
                rows.append(dict(name=name, shape=list(weight.shape), **request,
                                 reference_relative_error=error, full_rank_relative_error=full_error,
                                 frobenius_optimality_error=optimality_error,
                                 original_coefficients=int(weight.size), factor_coefficients=int(first.size + second.size),
                                 svd_seconds=elapsed, method="exact_svd"))
                print(f"{name}: rank={rank}, relative error vs reference={error:.3g}, {elapsed:.6f}s", flush=True)
        save_npz_new(out / "checked_factors.npz", arrays)
        with np.load(out / "checked_factors.npz", allow_pickle=False) as archive:
            if set(archive.files) != set(arrays):
                raise AssertionError("Serialized factor list changed")
            for key, value in arrays.items():
                np.testing.assert_array_equal(archive[key], value)
        publish_json(out / "factor_checks.json", dict(**provenance, checks=rows, serialization_verified=True,
                                                     status="complete"))


if __name__ == "__main__":
    main()
