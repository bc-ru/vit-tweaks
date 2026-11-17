"""Local activation criterion in stored (out,in) orientation; no model inference."""
import numpy as np
from scipy.linalg import solve_triangular


def weighted_factors(weight, moment, ridge_relative):
    """Full bank: SVD(W L), C_lambda=L L.T; unwhiten on the right.

    Minimize tr(E C E.T) + lambda ||E||_F^2, lambda=alpha*tr(C)/n.
    The sqrt(S) split is symmetric in weighted coordinates, not Euclidean W.
    """
    w, c = np.asarray(weight, np.float64), np.asarray(moment, np.float64)
    if (w.ndim != 2 or min(w.shape) < 1 or c.shape != (w.shape[1],)*2
            or not np.isfinite(w).all() or not np.isfinite(c).all()
            or not np.isfinite(ridge_relative) or ridge_relative <= 0
            or not np.allclose(c, c.T, atol=1e-10, rtol=1e-10)):
        raise ValueError('Invalid weight, symmetric moment, or positive ridge')
    eigenvalues = np.linalg.eigvalsh(c)
    scale = float(np.trace(c)/len(c))
    # Permit only Gram rounding, never silently repair an indefinite input.
    tolerance = 2e-6 * max(float(eigenvalues[-1]), 1e-30)
    if eigenvalues[0] < -tolerance or scale < 0:
        raise ValueError('Second moment is not positive semidefinite')
    # Zero activations make the empirical objective identically zero. A fixed
    # scale=1 fallback gives ordinary SVD and a uniquely specified ridge metric.
    ridge = ridge_relative * (scale if scale > 0 else 1.)
    c = (c+c.T)/2
    if eigenvalues[0] < 0:
        values, vectors = np.linalg.eigh(c)
        c = (vectors * np.maximum(values, 0)) @ vectors.T
    l = np.linalg.cholesky(c + ridge*np.eye(len(c)))
    u, s, vh = np.linalg.svd(w @ l, full_matrices=False)
    root = np.sqrt(s)
    first = solve_triangular(l.T, (root[:, None]*vh).T, lower=False).T
    second = u*root
    info = dict(ridge_relative=float(ridge_relative), ridge_absolute=float(ridge),
                moment_eigen_min=float(eigenvalues[0]), moment_eigen_max=float(eigenvalues[-1]),
                psd_rounding_clipped=bool(eigenvalues[0] < 0),
                condition_regularized=float((max(eigenvalues[-1], 0)+ridge)/ridge)
                if eigenvalues[0] <= 0 else float((eigenvalues[-1]+ridge)/(eigenvalues[0]+ridge)))
    return first.astype(np.float32), second.astype(np.float32), s, info


def local_error(residual, moment):
    e = np.asarray(residual, np.float64)
    value = float(np.sum((e @ moment)*e))
    # Gram matrices can carry tiny rounding error in their nullspace.
    tolerance = 2e-6 * max(float(np.linalg.norm(moment, ord='fro') * np.sum(e*e)), 1e-30)
    if not np.isfinite(value) or value < -tolerance:
        raise ValueError('Invalid local quadratic error')
    return max(value, 0.)


def allocate_ranks(curves, names, budget, method):
    """Exact multiple-choice knapsack over the supplied grid, integer storage units.

    Objective: sum of unnormalized empirical calibration MSE per token. Bias is
    constant and excluded. Exact budget equality is required; deterministic ties.
    """
    import math
    options = {}
    for name in names:
        rows = sorted([r for r in curves if r['name'] == name and r['method'] == method],
                      key=lambda r: r['actual_rank'])
        if not rows:
            raise ValueError(f'No local curve for {name}')
        if (len({int(r['actual_rank']) for r in rows}) != len(rows)
                or any(int(r['actual_rank']) < 1 or int(r['factor_weight_params']) < 1
                       or not np.isfinite(float(r['calibration_mse'])) or float(r['calibration_mse']) < 0
                       for r in rows)):
            raise ValueError(f'Invalid local curve for {name}')
        options[name] = rows
    unit = math.gcd(*(int(r['factor_weight_params']) for rows in options.values() for r in rows))
    if type(budget) is not int or budget < 1 or budget % unit:
        raise ValueError('Budget must be a positive multiple of storage unit')
    target = budget//unit
    states = {0: (0., ())}
    for name in names:
        next_states = {}
        for used, (loss, ranks) in states.items():
            for row in options[name]:
                cost = used + int(row['factor_weight_params'])//unit
                if cost > target:
                    continue
                candidate = (loss+float(row['calibration_mse']), ranks+(int(row['actual_rank']),))
                if cost not in next_states or candidate < next_states[cost]:
                    next_states[cost] = candidate
        states = next_states
    if target not in states:
        raise ValueError('Exact budget is infeasible on the rank grid')
    loss, ranks = states[target]
    return dict(zip(names, ranks)), dict(algorithm='exact_multiple_choice_knapsack',
        objective='sum calibration MSE per token (unregularized, unnormalized)',
        factor_weight_budget=budget, used_factor_weight_params=budget,
        storage_unit=unit, predicted_local_mse_sum=loss, method=method)
