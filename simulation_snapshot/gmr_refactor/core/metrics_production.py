import os

import numpy as np
import pandas as pd


PRODUCTION_VARIABLES = (
    "FOE",
    "WBHP1", "WBHP2", "WBHP3", "WBHP4", "WBHP5", "WBHP6",
    "GOR1", "GOR2", "GOR3", "GOR4", "GOR5", "GOR6",
    "WCT1", "WCT2", "WCT3", "WCT4", "WCT5", "WCT6",
)

PRODUCTION_GROUPS = {
    "FOE": ("FOE",),
    "WBHP": tuple(f"WBHP{i}" for i in range(1, 7)),
    "GOR": tuple(f"GOR{i}" for i in range(1, 7)),
    "WCT": tuple(f"WCT{i}" for i in range(1, 7)),
}


def _to_num(series):
    return pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)


def _safe_sigma(values, eps=1e-12):
    arr = np.asarray(values, dtype=float).ravel()
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return 1.0
    for val in (np.nanstd(arr), np.nanmax(arr) - np.nanmin(arr), np.nanmean(np.abs(arr))):
        val = float(val)
        if np.isfinite(val) and val > eps:
            return val
    return 1.0


def _read_rsm_table(path):
    with open(path, "rt") as f1:
        b1 = pd.read_fwf(f1, colspecs="infer", skiprows=6, header=None)
    with open(path, "rt") as f2:
        b2 = pd.read_fwf(f2, colspecs="infer", skiprows=94, header=None)
    with open(path, "rt") as f3:
        b3 = pd.read_fwf(f3, colspecs="infer", skiprows=182, header=None)

    return {
        "Time": _to_num(b1.iloc[0:82, 0]),
        "FOE": _to_num(b1.iloc[0:82, 2]),
        "WBHP1": _to_num(b1.iloc[0:82, 3]),
        "WBHP2": _to_num(b1.iloc[0:82, 4]),
        "WBHP3": _to_num(b1.iloc[0:82, 5]),
        "WBHP4": _to_num(b1.iloc[0:82, 6]),
        "WBHP5": _to_num(b1.iloc[0:82, 7]),
        "WBHP6": _to_num(b1.iloc[0:82, 8]),
        "GOR1": _to_num(b1.iloc[0:82, 9]),
        "GOR2": _to_num(b2.iloc[0:82, 1]),
        "GOR3": _to_num(b2.iloc[0:82, 2]),
        "GOR4": _to_num(b2.iloc[0:82, 3]),
        "GOR5": _to_num(b2.iloc[0:82, 4]),
        "GOR6": _to_num(b2.iloc[0:82, 5]),
        "WCT1": _to_num(b2.iloc[0:82, 6]),
        "WCT2": _to_num(b2.iloc[0:82, 7]),
        "WCT3": _to_num(b2.iloc[0:82, 8]),
        "WCT4": _to_num(b2.iloc[0:82, 9]),
        "WCT5": _to_num(b3.iloc[0:82, 1]),
        "WCT6": _to_num(b3.iloc[0:82, 2]),
    }


def read_real_rsm(real_path="Real.RSM"):
    return {
        name: np.asarray(values, dtype=float).copy()
        for name, values in _read_rsm_table(real_path).items()
    }


def compute_nrmse_production(sim_by_variable, obs_by_variable, obs_sigma=None, variables=None, eps=1e-12):
    names = [v for v in (variables or PRODUCTION_VARIABLES) if v in sim_by_variable and v in obs_by_variable]
    if not names:
        return {
            "prod_nrmse": np.nan,
            "prod_nrmse_best": np.nan,
            "prod_nrmse_std": np.nan,
            "prod_nrmse_by_variable": {},
            "prod_nrmse_by_ensemble": [],
        }

    residual_blocks = []
    by_variable = {}
    n_ens = None
    for name in names:
        sim = np.asarray(sim_by_variable[name], dtype=float)
        obs = np.asarray(obs_by_variable[name], dtype=float).reshape(-1)
        if sim.ndim == 1:
            sim = sim.reshape(-1, 1)
        n_time = min(sim.shape[0], obs.size)
        sim, obs = sim[:n_time, :], obs[:n_time]
        if n_ens is None:
            n_ens = sim.shape[1]
        if sim.shape[1] != n_ens:
            continue
        sigma = float(obs_sigma[name]) if obs_sigma and name in obs_sigma else _safe_sigma(obs, eps=eps)
        normalized = (sim - obs.reshape(-1, 1)) / max(sigma, eps)
        by_variable[name] = float(np.nanmean(np.sqrt(np.nanmean(normalized ** 2, axis=0))))
        residual_blocks.append(normalized)

    if not residual_blocks:
        return {
            "prod_nrmse": np.nan,
            "prod_nrmse_best": np.nan,
            "prod_nrmse_std": np.nan,
            "prod_nrmse_by_variable": {},
            "prod_nrmse_by_ensemble": [],
        }

    by_ensemble = np.sqrt(np.nanmean(np.vstack(residual_blocks) ** 2, axis=0))
    return {
        "prod_nrmse": float(np.nanmean(by_ensemble)),
        "prod_nrmse_best": float(np.nanmin(by_ensemble)),
        "prod_nrmse_std": float(np.nanstd(by_ensemble)),
        "prod_nrmse_by_variable": by_variable,
        "prod_nrmse_by_ensemble": by_ensemble.astype(float).tolist(),
        "variables": names,
    }


def category_scales(obs_by_variable, groups=PRODUCTION_GROUPS, eps=1e-12):
    scales = {}
    for group, names in groups.items():
        pooled = np.concatenate([
            np.asarray(obs_by_variable[name], dtype=float).ravel()
            for name in names
            if name in obs_by_variable
        ])
        pooled = pooled[np.isfinite(pooled)]
        scale = float(np.std(pooled)) if pooled.size else np.nan
        if not np.isfinite(scale) or scale <= eps:
            raise ValueError(f"Invalid category scale for {group}: {scale}")
        scales[group] = scale
    return scales


def _collect_group_residuals(sim_by_variable, obs_by_variable, groups, scale_map):
    member_blocks = {}
    mean_blocks = {}
    n_ens = None
    for group, names in groups.items():
        grp_member = []
        grp_mean = []
        scale = float(scale_map[group])
        for name in names:
            if name not in sim_by_variable or name not in obs_by_variable:
                continue
            sim = np.asarray(sim_by_variable[name], dtype=float)
            obs = np.asarray(obs_by_variable[name], dtype=float).reshape(-1)
            if sim.ndim == 1:
                sim = sim.reshape(-1, 1)
            n_time = min(sim.shape[0], obs.size)
            sim, obs = sim[:n_time, :], obs[:n_time]
            if n_ens is None:
                n_ens = sim.shape[1]
            if sim.shape[1] != n_ens:
                raise ValueError(f"Inconsistent ensemble size for {name}")
            grp_member.append((sim - obs[:, None]) / scale)
            grp_mean.append((np.nanmean(sim, axis=1) - obs) / scale)
        if grp_member:
            member_blocks[group] = np.vstack(grp_member)
            mean_blocks[group] = np.concatenate(grp_mean)
    return member_blocks, mean_blocks


def compute_category_balanced_nrmse(sim_by_variable, obs_by_variable, groups=PRODUCTION_GROUPS, scales=None):
    scale_map = dict(scales or category_scales(obs_by_variable, groups=groups))
    member_blocks, mean_blocks = _collect_group_residuals(sim_by_variable, obs_by_variable, groups, scale_map)
    if not member_blocks:
        return {
            "cb_prod_nrmse": np.nan,
            "cb_prod_nrmse_ensemble_mean": np.nan,
            "cb_prod_nrmse_ensemble_median": np.nan,
            "cb_prod_nrmse_ensemble_p90": np.nan,
            "cb_prod_nrmse_best": np.nan,
            "cb_prod_nrmse_std": np.nan,
            "cb_prod_nrmse_by_group": {},
            "cb_prod_nrmse_by_group_ensemble_mean": {},
            "cb_prod_nrmse_by_ensemble": [],
            "cb_prod_scales": scale_map,
        }

    group_member_scores = {g: np.sqrt(np.nanmean(b ** 2, axis=0)) for g, b in member_blocks.items()}
    group_mean_scores = {g: float(np.sqrt(np.nanmean(b ** 2))) for g, b in mean_blocks.items()}
    by_ensemble = np.nanmean(np.vstack(list(group_member_scores.values())), axis=0)
    return {
        "cb_prod_nrmse": float(np.mean(list(group_mean_scores.values()))),
        "cb_prod_nrmse_ensemble_mean": float(np.nanmean(by_ensemble)),
        "cb_prod_nrmse_ensemble_median": float(np.nanmedian(by_ensemble)),
        "cb_prod_nrmse_ensemble_p90": float(np.nanquantile(by_ensemble, 0.90)),
        "cb_prod_nrmse_best": float(np.nanmin(by_ensemble)),
        "cb_prod_nrmse_std": float(np.nanstd(by_ensemble)),
        "cb_prod_nrmse_by_group": group_mean_scores,
        "cb_prod_nrmse_by_group_ensemble_mean": {
            g: float(np.nanmean(v)) for g, v in group_member_scores.items()
        },
        "cb_prod_nrmse_by_ensemble": by_ensemble.astype(float).tolist(),
        "cb_prod_scales": scale_map,
    }


def compute_category_scaled_weighted_mismatch(sim_by_variable, obs_by_variable, groups=PRODUCTION_GROUPS, scales=None):
    scale_map = dict(scales or category_scales(obs_by_variable, groups=groups))
    member_blocks, mean_blocks = _collect_group_residuals(sim_by_variable, obs_by_variable, groups, scale_map)
    if not member_blocks:
        return {
            "m_prod": np.nan,
            "o_nd": np.nan,
            "m_prod_ensemble_mean": np.nan,
            "m_prod_ensemble_median": np.nan,
            "m_prod_ensemble_p90": np.nan,
            "m_prod_best": np.nan,
            "m_prod_std": np.nan,
            "m_prod_by_group": {},
            "m_prod_by_group_ensemble_mean": {},
            "m_prod_by_ensemble": [],
            "m_prod_scales": scale_map,
            "m_prod_n_data": 0,
            "m_prod_category_weights": {},
        }

    group_member_scores = {g: np.sqrt(np.nanmean(b ** 2, axis=0)) for g, b in member_blocks.items()}
    group_mean_scores = {g: float(np.sqrt(np.nanmean(b ** 2))) for g, b in mean_blocks.items()}
    group_n_data = {g: int(np.isfinite(b).sum()) for g, b in mean_blocks.items()}

    all_member = np.vstack(list(member_blocks.values()))
    all_mean = np.concatenate(list(mean_blocks.values()))
    by_ensemble = np.sqrt(np.nanmean(all_member ** 2, axis=0))
    m_prod = float(np.sqrt(np.nanmean(all_mean ** 2)))
    n_data = int(np.isfinite(all_mean).sum())
    category_weights = {
        g: float(cnt / n_data) if n_data > 0 else np.nan
        for g, cnt in group_n_data.items()
    }
    return {
        "m_prod": m_prod,
        "o_nd": float(0.5 * m_prod ** 2),
        "m_prod_ensemble_mean": float(np.nanmean(by_ensemble)),
        "m_prod_ensemble_median": float(np.nanmedian(by_ensemble)),
        "m_prod_ensemble_p90": float(np.nanquantile(by_ensemble, 0.90)),
        "m_prod_best": float(np.nanmin(by_ensemble)),
        "m_prod_std": float(np.nanstd(by_ensemble)),
        "m_prod_by_group": group_mean_scores,
        "m_prod_by_group_ensemble_mean": {
            g: float(np.nanmean(v)) for g, v in group_member_scores.items()
        },
        "m_prod_by_ensemble": by_ensemble.astype(float).tolist(),
        "m_prod_scales": scale_map,
        "m_prod_n_data": n_data,
        "m_prod_category_weights": category_weights,
    }


def compute_all_production_metrics(sim_by_variable, obs_by_variable, variables=None):
    result = compute_nrmse_production(sim_by_variable, obs_by_variable, variables=variables)
    scales = category_scales(obs_by_variable)
    result.update(compute_category_balanced_nrmse(sim_by_variable, obs_by_variable, scales=scales))
    result.update(compute_category_scaled_weighted_mismatch(sim_by_variable, obs_by_variable, scales=scales))
    return result


def read_rsm_members(member_indices, root_dir=".", folder_pattern="MASTER{index:05d}"):
    indices = [int(i) for i in member_indices]
    if not indices:
        raise ValueError("At least one member index is required")
    sims = {name: [] for name in PRODUCTION_VARIABLES}
    for idx in indices:
        folder = folder_pattern.format(index=idx + 1)
        sim = _read_rsm_table(os.path.join(root_dir, folder, "MASTER0.RSM"))
        for name in PRODUCTION_VARIABLES:
            sims[name].append(np.asarray(sim[name], dtype=float))
    return {name: np.column_stack(series) for name, series in sims.items()}


def read_rsm_ensemble(n_ensemble, root_dir=".", folder_pattern="MASTER{index:05d}"):
    return read_rsm_members(range(n_ensemble), root_dir=root_dir, folder_pattern=folder_pattern)


def compute_production_metrics_from_rsm_members(member_indices, real_path="Real.RSM", root_dir="."):
    return compute_all_production_metrics(
        read_rsm_members(member_indices, root_dir=root_dir),
        _read_rsm_table(real_path),
    )


def compute_nrmse_from_rsm_ensemble(n_ensemble, real_path="Real.RSM", root_dir=".", variables=None):
    return compute_all_production_metrics(
        read_rsm_ensemble(n_ensemble=n_ensemble, root_dir=root_dir),
        _read_rsm_table(real_path),
        variables=variables,
    )
