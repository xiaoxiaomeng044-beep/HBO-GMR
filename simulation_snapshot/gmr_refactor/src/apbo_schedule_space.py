import hashlib
import math

import numpy as np
from scipy.stats import qmc


NUMERICAL_EPS = 1e-12


def _as_vector(values, name):
    array = np.asarray(list(values), dtype=float)
    if array.ndim != 1 or array.size < 2:
        raise ValueError(f"{name} must be a 1-D vector with at least two values")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def validate_budget_shares(shares, *, total=1.0, lower=0.125, upper=0.5, atol=1e-10):
    array = _as_vector(shares, "shares")
    if total <= 0.0 or lower > upper:
        raise ValueError("invalid simplex bounds")
    if array.size * lower > total + atol or array.size * upper < total - atol:
        raise ValueError("bounded simplex is infeasible for this vector length")
    if np.any(array < lower - atol) or np.any(array > upper + atol):
        raise ValueError("shares violate bounded-simplex limits")
    if not np.isclose(float(np.sum(array)), total, atol=atol, rtol=0.0):
        raise ValueError("shares do not close the ES-MDA budget")
    return array.copy()


def project_bounded_simplex(values, *, total=1.0, lower=0.125, upper=0.5, atol=1e-12, max_iter=200):
    vector = _as_vector(values, "values")
    if lower > upper or vector.size * lower > total + atol or vector.size * upper < total - atol:
        raise ValueError("bounded simplex is infeasible")

    lambda_low = float(np.min(vector - upper))
    lambda_high = float(np.max(vector - lower))
    for _ in range(max_iter):
        middle = 0.5 * (lambda_low + lambda_high)
        current = float(np.sum(np.clip(vector - middle, lower, upper)))
        if abs(current - total) <= atol:
            break
        if current > total:
            lambda_low = middle
        else:
            lambda_high = middle

    projected = np.clip(vector - 0.5 * (lambda_low + lambda_high), lower, upper)
    residual = total - float(np.sum(projected))
    for _ in range(vector.size + 2):
        if abs(residual) <= atol:
            break
        capacity = (upper - projected) if residual > 0.0 else (projected - lower)
        active = np.flatnonzero(capacity > atol)
        if active.size == 0:
            break
        equal_delta = residual / active.size
        delta = np.minimum(capacity[active], abs(equal_delta))
        projected[active] += delta if residual > 0.0 else -delta
        residual = total - float(np.sum(projected))

    return validate_budget_shares(projected, total=total, lower=lower, upper=upper, atol=max(atol * 10.0, 1e-10))


def budget_to_alpha(shares, *, total=1.0, lower=0.125, upper=0.5):
    return 1.0 / validate_budget_shares(shares, total=total, lower=lower, upper=upper)


def alpha_to_budget(alpha, *, require_closed=True, total=1.0, lower=0.125, upper=0.5):
    alpha_array = _as_vector(alpha, "alpha")
    if np.any(alpha_array <= 0.0):
        raise ValueError("alpha values must be positive")
    shares = 1.0 / alpha_array
    if require_closed:
        return validate_budget_shares(shares, total=total, lower=lower, upper=upper)
    return shares


def schedule_hash(shares, decimals=12):
    rounded = np.round(_as_vector(shares, "shares"), decimals=decimals).astype("<f8", copy=False)
    return hashlib.sha256(rounded.tobytes()).hexdigest()


def sobol_budget_candidates(n_candidates, *, n_steps=4, seed=2026, lower=0.125, upper=0.5, include_equal=True):
    if n_candidates <= 0 or n_steps < 2:
        raise ValueError("invalid candidate pool dimensions")
    if n_steps * lower > 1.0 + NUMERICAL_EPS or n_steps * upper < 1.0 - NUMERICAL_EPS:
        raise ValueError("bounded simplex is infeasible")

    exponent = int(math.ceil(math.log2(max(16, n_candidates * 16))))
    raw_points = qmc.Sobol(d=n_steps, scramble=True, seed=seed).random_base2(m=exponent)

    candidates = []
    seen = set()

    def add(candidate):
        key = schedule_hash(candidate)
        if key not in seen:
            seen.add(key)
            candidates.append(candidate)

    if include_equal:
        add(np.full(n_steps, 1.0 / n_steps, dtype=float))

    for raw in raw_points:
        positive = -np.log(np.clip(raw, 1e-12, 1.0 - 1e-12))
        add(project_bounded_simplex(positive / np.sum(positive), total=1.0, lower=lower, upper=upper))
        if len(candidates) >= n_candidates:
            break

    if len(candidates) < n_candidates:
        raise RuntimeError("failed to generate enough unique simplex candidates")
    return np.vstack(candidates[:n_candidates])
