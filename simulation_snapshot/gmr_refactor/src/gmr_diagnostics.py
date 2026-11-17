import os
import sys

import numpy as np

from .common import append_csv_row, save_json


GMR_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
GMR_CORE_DIR = os.path.join(GMR_ROOT, "core")
if GMR_CORE_DIR not in sys.path:
    sys.path.insert(0, GMR_CORE_DIR)

from metrics_geology import build_geology_reference, compute_geology_metrics, vec_to_dhw


def _vecN_to_mean_dhw(vecN, nx, ny, nz, log10=False):
    mean_vec = np.mean(np.asarray(vecN, dtype=np.float64), axis=1)
    if log10:
        mean_vec = np.log10(np.maximum(mean_vec, 1e-15))
    return vec_to_dhw(mean_vec, nx=nx, ny=ny, nz=nz).astype(np.float32)


def _effective_to_mask_dhw(effective, nx, ny, nz):
    if effective is None:
        return None
    flat = np.asarray(effective).reshape(-1) != 0
    if flat.size != nx * ny * nz:
        return None
    return vec_to_dhw(flat.astype(np.float32), nx=nx, ny=ny, nz=nz).astype(bool)


def build_reference_from_prior(prior_perm, prior_poro, nx, ny, nz, effective=None):
    mask = _effective_to_mask_dhw(effective, nx, ny, nz)
    logperm = _vecN_to_mean_dhw(prior_perm, nx, ny, nz, log10=True)
    poro = _vecN_to_mean_dhw(prior_poro, nx, ny, nz, log10=False)
    return build_geology_reference(logperm, poro, mask=mask)


def compute_mean_field_metrics(perm, poro, reference, nx, ny, nz, bins=50, max_lag=8):
    logperm = _vecN_to_mean_dhw(perm, nx, ny, nz, log10=True)
    poro_dhw = _vecN_to_mean_dhw(poro, nx, ny, nz, log10=False)
    return compute_geology_metrics(logperm, poro_dhw, reference=reference, bins=bins, max_lag=max_lag)


def relative_norm(after, before, eps=1e-12):
    a = np.asarray(after, dtype=np.float64)
    b = np.asarray(before, dtype=np.float64)
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), eps))


def finite_range(arr):
    vals = np.asarray(arr, dtype=np.float64).ravel()
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return None, None
    return float(np.min(vals)), float(np.max(vals))


def build_gmr_diagnostic_record(
    method,
    run_id,
    iteration,
    before_perm,
    before_poro,
    after_perm,
    after_poro,
    gmr_diagnostics,
    geology_reference,
    nx,
    ny,
    nz,
    prod_nrmse=None,
    prod_mismatch_legacy=None,
):
    before_geo = compute_mean_field_metrics(before_perm, before_poro, geology_reference, nx, ny, nz)
    after_geo = compute_mean_field_metrics(after_perm, after_poro, geology_reference, nx, ny, nz)
    perm_min, perm_max = finite_range(after_perm)
    poro_min, poro_max = finite_range(after_poro)

    passthrough_keys = (
        "patch_count", "realization_count",
        "latent_nll_mean", "latent_nll_q50", "latent_nll_q75", "latent_nll_q90",
        "prior_nll_q75", "prior_nll_q90", "violation_ratio",
        "lambda_gmr", "lambda_before_prod_guard", "eta_score",
        "stage_mode", "stage_factor", "stage_floor",
        "prod_guard_delta_norm", "prod_guard_beta", "prod_guard_tolerance", "fit_guard",
        "global_guard_policy", "global_guard_min_geo_gain", "global_guard_target_score",
        "global_guard_blend", "global_guard_geo_gain_selected",
        "strength_mode", "violation_squashed", "score_norm_mean",
        "projection_norm_relative_perm",
        "latent_nll_after_mean", "latent_nll_after_q50", "latent_nll_after_q75", "latent_nll_after_q90",
        "delta_latent_nll_mean", "delta_latent_nll_q75",
    )
    record = {
        "method": method,
        "run_id": run_id,
        "iteration": iteration,
        "prod_nrmse": prod_nrmse,
        "prod_mismatch_legacy": prod_mismatch_legacy,
        **{k: gmr_diagnostics.get(k) for k in passthrough_keys},
        "delta_norm_perm": relative_norm(after_perm, before_perm),
        "delta_norm_poro": relative_norm(after_poro, before_poro),
        "after_perm_min": perm_min,
        "after_perm_max": perm_max,
        "after_poro_min": poro_min,
        "after_poro_max": poro_max,
    }

    for key, value in before_geo.items():
        record[f"before_{key}"] = value
    for key, value in after_geo.items():
        record[f"after_{key}"] = value
    for key in before_geo:
        b, a = before_geo[key], after_geo[key]
        record[f"delta_{key}"] = (
            float(a) - float(b) if a is not None and b is not None and np.isfinite(a) and np.isfinite(b) else None
        )
    return record


def save_gmr_diagnostics(csv_path, json_path, record, details=None):
    append_csv_row(csv_path, record)
    payload = {"record": record}
    if details is not None:
        payload["details"] = details
    save_json(payload, json_path)
