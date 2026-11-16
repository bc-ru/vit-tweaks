"""02_matrix_curves: exact-SVD tail errors, with no model execution or image loading."""
import csv
import io
import time

import numpy as np

from aerial_svd.jax_matrix_analysis import approximation_records, singular_values
from aerial_svd.lowrank_common import load_inputs, parser, publish_text, rank_requests, selected_names, stage_output
from aerial_svd.matrix_records import MatrixRecord


def main(argv=None):
    args = parser("02_matrix_curves", matrices=True).parse_args(argv)
    _, spec, params, _, _, provenance = load_inputs(args, "02_matrix_curves")
    names = selected_names(params, args)
    requests = {name: rank_requests(params[name + ".weight"].shape, args) for name in names}
    provenance.update(selected_matrices=names, ranks=requests,
                      error_definition="analytic exact-SVD tail; full-rank error is zero in exact arithmetic",
                      svd_timing="one synchronized cold spectrum call per matrix, including dispatch/transfer")
    with stage_output(args.output_dir, provenance) as out:
        rows = []
        for name in names:
            weight = params[name + ".weight"]
            start = time.perf_counter()
            spectrum = np.asarray(singular_values(weight))
            elapsed = time.perf_counter() - start
            if not np.isfinite(spectrum).all():
                raise AssertionError(f"Nonfinite spectrum: {name}")
            rec = MatrixRecord(name, "linear", weight.shape, weight.shape, weight.size)
            ranks = [request["actual_rank"] for request in requests[name]]
            ratios = [rank / min(weight.shape) for rank in ranks]
            records = approximation_records(spec.model, args.checkpoint, rec, weight, ratios,
                                            spectrum=spectrum, ranks=ranks)
            for request, record in zip(requests[name], records):
                rows.append(dict(name=name, m=record.m, n=record.n, **request,
                                 relative_frobenius_error=record.fro_error,
                                 relative_spectral_error=record.spectral_error,
                                 original_coefficients=record.dense_storage,
                                 factor_coefficients=record.lowrank_storage,
                                 coefficient_savings=record.dense_storage - record.lowrank_storage,
                                 checkpoint_sha256=provenance["checkpoint"]["sha256"],
                                 checkpoint=provenance["checkpoint"]["path"], method="exact_svd", dtype="float32",
                                 svd_seconds=elapsed))
            print(f"{name}: exact spectrum {elapsed:.6f}s, {len(records)} ranks", flush=True)
        stream = io.StringIO()
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
        publish_text(out / "matrix_curves.csv", stream.getvalue())


if __name__ == "__main__":
    main()
