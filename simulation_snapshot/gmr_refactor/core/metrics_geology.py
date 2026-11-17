import numpy as np
from scipy.ndimage import label


def _finite_values(field, mask=None):
    arr = np.asarray(field, dtype=float)
    arr = arr[np.asarray(mask, dtype=bool)] if mask is not None else arr.ravel()
    return arr[np.isfinite(arr)]


def compute_histogram_distance(field_a, field_b, bins=50, value_range=None, mask=None):
    a = _finite_values(field_a, mask=mask)
    b = _finite_values(field_b, mask=mask)
    if a.size == 0 or b.size == 0:
        return float("nan")
    if value_range is None:
        lo = float(min(np.min(a), np.min(b)))
        hi = float(max(np.max(a), np.max(b)))
        if hi <= lo:
            hi = lo + 1.0
        value_range = (lo, hi)
    ha, _ = np.histogram(a, bins=bins, range=value_range)
    hb, _ = np.histogram(b, bins=bins, range=value_range)
    ha = ha.astype(float) / max(float(np.sum(ha)), 1e-12)
    hb = hb.astype(float) / max(float(np.sum(hb)), 1e-12)
    return float(np.mean(np.abs(np.cumsum(ha) - np.cumsum(hb))))


def _direction_axis(direction):
    mapping = {"z": 0, "depth": 0, 0: 0, "y": 1, 1: 1, "x": 2, 2: 2}
    if direction not in mapping:
        raise ValueError(f"Unknown variogram direction: {direction}")
    return mapping[direction]


def directional_variogram(field, direction="x", max_lag=8, mask=None):
    arr = np.asarray(field, dtype=float)
    if mask is not None:
        arr = np.where(np.asarray(mask, dtype=bool), arr, np.nan)
    axis = _direction_axis(direction)
    values = []
    for lag in range(1, max_lag + 1):
        head = [slice(None)] * 3
        tail = [slice(None)] * 3
        head[axis] = slice(0, -lag)
        tail[axis] = slice(lag, None)
        diff = arr[tuple(head)] - arr[tuple(tail)]
        gamma = 0.5 * np.nanmean(diff ** 2)
        values.append(float(gamma) if np.isfinite(gamma) else np.nan)
    return np.asarray(values, dtype=float)


def compute_variogram_error(field, reference_field, directions=("x", "y", "z"), max_lag=8, mask=None, eps=1e-12):
    errs = []
    for direction in directions:
        g = directional_variogram(field, direction=direction, max_lag=max_lag, mask=mask)
        r = directional_variogram(reference_field, direction=direction, max_lag=max_lag, mask=mask)
        valid = np.isfinite(g) & np.isfinite(r)
        if np.any(valid):
            errs.append(np.mean(np.abs(g[valid] - r[valid]) / np.maximum(np.abs(r[valid]), eps)))
    return float(np.mean(errs)) if errs else float("nan")


def compute_connectivity_index(field, threshold_quantile=0.8, mask=None):
    values = _finite_values(field, mask=mask)
    if values.size == 0:
        return float("nan")
    threshold = float(np.nanquantile(values, threshold_quantile))
    active = np.asarray(field, dtype=float) >= threshold
    if mask is not None:
        active &= np.asarray(mask, dtype=bool)
    structure = np.zeros((3, 3, 3), dtype=int)
    structure[1, 1, :] = 1
    structure[1, :, 1] = 1
    structure[:, 1, 1] = 1
    labeled, n_comp = label(active.astype(int), structure=structure)
    if n_comp == 0:
        return 0.0
    counts = np.bincount(labeled.ravel())
    counts[0] = 0
    return float(np.max(counts) / max(float(np.sum(active)), 1.0))


def build_geology_reference(logperm_field, poro_field, mask=None):
    return {
        "logperm_field": np.asarray(logperm_field, dtype=np.float32),
        "poro_field": np.asarray(poro_field, dtype=np.float32),
        "mask": np.asarray(mask, dtype=bool) if mask is not None else None,
        "connectivity_logperm": compute_connectivity_index(logperm_field, mask=mask),
    }


def compute_geology_metrics(logperm_field, poro_field, reference, bins=50, max_lag=8):
    mask = reference.get("mask")
    ref_logperm = reference["logperm_field"]
    ref_poro = reference["poro_field"]
    conn = compute_connectivity_index(logperm_field, mask=mask)
    ref_conn = reference.get("connectivity_logperm", np.nan)
    return {
        "hist_dist_logperm": compute_histogram_distance(logperm_field, ref_logperm, bins=bins, mask=mask),
        "hist_dist_poro": compute_histogram_distance(poro_field, ref_poro, bins=bins, mask=mask),
        "variogram_error_logperm": compute_variogram_error(logperm_field, ref_logperm, max_lag=max_lag, mask=mask),
        "variogram_error_poro": compute_variogram_error(poro_field, ref_poro, max_lag=max_lag, mask=mask),
        "connectivity_index_logperm": conn,
        "connectivity_delta_logperm": (
            float(abs(conn - ref_conn)) if np.isfinite(conn) and np.isfinite(ref_conn) else float("nan")
        ),
    }


def vec_to_dhw(vec_1d, nx, ny, nz):
    arr = np.reshape(np.asarray(vec_1d), (nx, ny, nz), order="F")
    return np.transpose(arr, (2, 1, 0)).copy()
