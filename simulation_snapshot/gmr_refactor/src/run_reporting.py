from datetime import datetime
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import zoom

from metrics_geology import compute_geology_metrics
from objectives import compute_j_bo as compute_paper_j_bo
from .common import append_csv_row, load_json, save_json
from .experiment_config import (
    BO_BUDGET_PENALTY,
    EPC_STAR_C_SAFE,
    EPC_STAR_EPS,
    EPC_STAR_OMEGA_AE,
    EPC_STAR_OMEGA_BUDGET,
    EPC_STAR_OMEGA_SPATIAL,
    LAMBDA_AE,
    LAMBDA_PHI_UNIFIED,
)
from .posterior_selection import posterior_argmin


LOG_HEADER = [
    "Iteration", "AlphaSuggest", "AlphaUsed", "CRadiusUsed",
    "RMSE", "phi_AE", "J_raw", "J_unified",
    "EPC_legacy", "EPC_legacy_fit", "EPC_legacy_budget", "EPC_legacy_spatial",
    "EPC_star", "EPC_star_fit", "EPC_star_budget_ratio", "EPC_star_spatial", "RMSE_norm", "phi_AE_norm",
    "ScoreForBO", "ScoreForBOName", "FeedbackMode",
    "BudgetRemainingBefore", "LowerBoundLB(=1/B)", "InvAlphaSumBefore", "InvAlphaSumAfter",
    "WillStopAfter", "BOAlphaOnly", "AEActive", "AESigmaScale", "AETargetQuantile",
    "AEPreset", "AEHardRatio", "AESigmaUsed", "AELocalizationRadius",
    "PhiBeforeUpdate", "PhiDelta", "RMSEBeforeUpdate", "RMSEDelta",
    "ProdMismatchLegacy", "ProdNRMSE", "ProdNRMSEBest", "ProdNRMSEStd",
    "J_BO", "J_GMR", "GeoScore", "GMRProdScore", "AEConstraintForm", "AETargetMode", "AEPatchMode",
    "AEPseudoRows", "AEPseudoDMean", "AEPseudoYMean", "AEPseudoResidualMean",
    "AEPseudoResidualAbsMean", "AEPseudoResidualP95", "AEEvidenceFlag",
]


def _safe_float(value):
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if np.isfinite(out) else None


def _finite_or_nan(value):
    res = _safe_float(value)
    return float("nan") if res is None else res


def _finite_or_zero(value):
    res = _safe_float(value)
    return 0.0 if res is None else res


def production_metric_value(info, metric_name):
    key = {
        "legacy_temporal": "prod_nrmse",
        "category_balanced": "cb_prod_nrmse",
        "m_prod": "m_prod",
    }[metric_name]
    return _finite_or_nan(info.get(key))


def compute_geo_score(metrics, args):
    if metrics is None:
        return np.nan
    return float(
        args.gmr_geo_w_hist_logperm * _finite_or_zero(metrics.get("hist_dist_logperm"))
        + args.gmr_geo_w_hist_poro * _finite_or_zero(metrics.get("hist_dist_poro"))
        + args.gmr_geo_w_var_logperm * _finite_or_zero(metrics.get("variogram_error_logperm"))
        + args.gmr_geo_w_var_poro * _finite_or_zero(metrics.get("variogram_error_poro"))
        + args.gmr_geo_w_conn * abs(_finite_or_zero(metrics.get("connectivity_delta_logperm")))
    )


def compute_j_raw(rmse, phi_ae, lambda_ae=LAMBDA_AE):
    if np.isfinite(phi_ae):
        return float(rmse + lambda_ae * phi_ae)
    return float(rmse)


def compute_j_unified(rmse, phi_ae, prior_metrics, lambda_phi=LAMBDA_PHI_UNIFIED):
    rmse_prior = prior_metrics.get("rmse_prior")
    phi_prior = prior_metrics.get("phi_ae_prior")
    if not rmse_prior or rmse_prior <= 0 or not phi_prior or phi_prior <= 0 or not np.isfinite(phi_ae):
        return np.nan
    return float((rmse / rmse_prior) + lambda_phi * (phi_ae / phi_prior))


def compute_j_gmr(prod_nrmse, geo_score, prior_metrics, lambda_geo):
    prod_prior = (prior_metrics or {}).get("prod_nrmse_prior")
    if not prod_prior or prod_prior <= 0 or not np.isfinite(prod_nrmse):
        return np.nan, np.nan
    prod_score = float(prod_nrmse) / float(prod_prior)
    geo = float(geo_score) if np.isfinite(geo_score) else 0.0
    return float(prod_score + lambda_geo * geo), prod_score


def compute_epc_score(rmse, phi_ae, alpha, c_radius, c_safe=3.5, w_budget=2.0, w_spatial=1.5):
    total_error = rmse + (0.1 * phi_ae if np.isfinite(phi_ae) and phi_ae > 0 else 0.0)
    term_acc = float(np.log(total_error + 1e-6))
    term_bud = 1.0 / max(alpha, 1e-6)
    term_spa = 0.0 if (c_radius is None or not np.isfinite(c_radius) or c_radius <= c_safe) else ((c_radius - c_safe) / c_safe) ** 2
    return term_acc + w_budget * term_bud + w_spatial * term_spa, term_acc, term_bud, term_spa


def compute_epc_star_score(rmse, phi_ae, alpha, c_radius, prior_metrics, budget_remaining, use_ae_term=True):
    rmse_prior = (prior_metrics or {}).get("rmse_prior")
    phi_prior = (prior_metrics or {}).get("phi_ae_prior")
    rmse_norm = float(rmse / rmse_prior) if (rmse_prior and rmse_prior > 0) else float(rmse)
    phi_norm = float(phi_ae / phi_prior) if (use_ae_term and phi_prior and phi_prior > 0 and np.isfinite(phi_ae)) else 0.0
    term_fit = float(np.log(max(rmse_norm + EPC_STAR_OMEGA_AE * phi_norm, 0.0) + EPC_STAR_EPS))
    term_budget = (1.0 / max(alpha, EPC_STAR_EPS)) / max(budget_remaining, EPC_STAR_EPS)
    term_spatial = (
        0.0
        if (c_radius is None or not np.isfinite(c_radius) or c_radius <= EPC_STAR_C_SAFE)
        else ((c_radius - EPC_STAR_C_SAFE) / EPC_STAR_C_SAFE) ** 2
    )
    epc_star = term_fit + EPC_STAR_OMEGA_BUDGET * term_budget + EPC_STAR_OMEGA_SPATIAL * term_spatial
    return float(epc_star), term_fit, term_budget, term_spatial, rmse_norm, phi_norm


def compute_bo_feedback_score(mode, rmse, alpha, j_unified, epc_legacy, epc_star, prior_metrics, j_gmr=None):
    if mode in ("j_bo", "original"):
        if j_gmr is not None and np.isfinite(j_gmr):
            return compute_paper_j_bo(j_gmr, alpha, lambda_budget=BO_BUDGET_PENALTY), "J_BO_GMR"
        if np.isfinite(j_unified):
            return compute_paper_j_bo(j_unified, alpha, lambda_budget=BO_BUDGET_PENALTY), "J_BO"
        return float(epc_legacy), "EPC_legacy_fallback"
    if mode == "epc_star":
        return float(epc_star), "EPC_star"
    if mode == "epc_legacy":
        return float(epc_legacy), "EPC_legacy"
    if mode == "rmse_budget":
        rmse_prior = (prior_metrics or {}).get("rmse_prior")
        rmse_norm = float(rmse / rmse_prior) if (rmse_prior and rmse_prior > 0) else float(rmse)
        return float(rmse_norm + BO_BUDGET_PENALTY / max(alpha, 1e-12)), "RMSE_norm_plus_budget"
    raise ValueError(f"Unknown feedback mode: {mode}")


def update_iteration_status(current_iteration, total_iterations, progress_file, current_timestep=None, total_timesteps=None):
    progress = {
        "current_iteration": current_iteration,
        "total_iterations": total_iterations,
        "timestamp": datetime.now().isoformat(),
    }
    if current_timestep is not None and total_timesteps is not None:
        progress["current_timestep"] = current_timestep
        progress["total_timesteps"] = total_timesteps
    save_json(progress, progress_file)


def record_cb_production_metrics(ctx, iteration, stage, info, source="full_ensemble"):
    groups = info.get("cb_prod_nrmse_by_group", {})
    group_ens = info.get("cb_prod_nrmse_by_group_ensemble_mean", {})
    m_groups = info.get("m_prod_by_group", {})
    m_group_ens = info.get("m_prod_by_group_ensemble_mean", {})
    append_csv_row(
        ctx.paths["cb_production_metrics"],
        {
            "Iteration": iteration,
            "Stage": stage,
            "Source": source,
            "ProductionMetricUsed": ctx.args.production_metric,
            "LegacyProdNRMSE": info.get("prod_nrmse"),
            "MProd": info.get("m_prod"),
            "ONd": info.get("o_nd"),
            "MProdEnsembleMean": info.get("m_prod_ensemble_mean"),
            "MProdEnsembleMedian": info.get("m_prod_ensemble_median"),
            "MProdEnsembleP90": info.get("m_prod_ensemble_p90"),
            "MProdBest": info.get("m_prod_best"),
            "MProdStd": info.get("m_prod_std"),
            "MProdNData": info.get("m_prod_n_data"),
            "MProd_FOE": m_groups.get("FOE"),
            "MProd_WBHP": m_groups.get("WBHP"),
            "MProd_GOR": m_groups.get("GOR"),
            "MProd_WCT": m_groups.get("WCT"),
            "MProdEnsMean_FOE": m_group_ens.get("FOE"),
            "MProdEnsMean_WBHP": m_group_ens.get("WBHP"),
            "MProdEnsMean_GOR": m_group_ens.get("GOR"),
            "MProdEnsMean_WCT": m_group_ens.get("WCT"),
            "CBProdNRMSE": info.get("cb_prod_nrmse"),
            "CBProdNRMSEEnsembleMean": info.get("cb_prod_nrmse_ensemble_mean"),
            "CBProdNRMSEEnsembleMedian": info.get("cb_prod_nrmse_ensemble_median"),
            "CBProdNRMSEEnsembleP90": info.get("cb_prod_nrmse_ensemble_p90"),
            "CBProdNRMSEBest": info.get("cb_prod_nrmse_best"),
            "CBProdNRMSEStd": info.get("cb_prod_nrmse_std"),
            "CB_FOE": groups.get("FOE"),
            "CB_WBHP": groups.get("WBHP"),
            "CB_GOR": groups.get("GOR"),
            "CB_WCT": groups.get("WCT"),
            "CBEnsMean_FOE": group_ens.get("FOE"),
            "CBEnsMean_WBHP": group_ens.get("WBHP"),
            "CBEnsMean_GOR": group_ens.get("GOR"),
            "CBEnsMean_WCT": group_ens.get("WCT"),
        },
    )


def compute_external_geology_row(ctx, iter_id, log10k_mean, poro_mean, geo_reference, phi_ae=np.nan):
    metrics = compute_geology_metrics(log10k_mean, poro_mean, geo_reference)
    return {
        "method": ctx.run_tag,
        "run_id": ctx.run_id,
        "iteration": iter_id,
        "hist_dist_logperm": metrics.get("hist_dist_logperm"),
        "hist_dist_poro": metrics.get("hist_dist_poro"),
        "variogram_error_logperm": metrics.get("variogram_error_logperm"),
        "variogram_error_poro": metrics.get("variogram_error_poro"),
        "connectivity_index_logperm": metrics.get("connectivity_index_logperm"),
        "connectivity_delta_logperm": metrics.get("connectivity_delta_logperm"),
        "GeoScore": compute_geo_score(metrics, ctx.args),
        "phi_ae": float(phi_ae) if np.isfinite(phi_ae) else np.nan,
    }


def save_iteration_state_npz(iter_id, out_dir, mumyperm, mumyporo, mumypermz, effective, metrics, save_full_ensemble=True):
    os.makedirs(out_dir, exist_ok=True)
    idx_valid = np.where(effective.ravel() != 0)[0]
    payload = {
        "iter_id": np.array([iter_id], dtype=int),
        "valid_index": idx_valid.astype(np.int32),
        "perm_mean": np.mean(mumyperm, axis=1).astype(np.float32),
        "poro_mean": np.mean(mumyporo, axis=1).astype(np.float32),
        "permz_mean": np.mean(mumypermz, axis=1).astype(np.float32),
        "metrics_json": np.array([json.dumps(metrics, ensure_ascii=False)], dtype=object),
    }
    if save_full_ensemble:
        payload["mumyperm"] = mumyperm.astype(np.float32)
        payload["mumyporo"] = mumyporo.astype(np.float32)
        payload["mumypermz"] = mumypermz.astype(np.float32)
    np.savez_compressed(os.path.join(out_dir, f"state_iter_{iter_id:02d}.npz"), **payload)


def save_history_match_snapshots_npz(snapshots, save_path):
    iter_keys = sorted(snapshots)
    payload = {"iter_keys": np.array(iter_keys, dtype=int)}
    for it in iter_keys:
        snap = snapshots[it]
        for key in ("Time", "FOE_ens", "FOE_true", "WBHP4_ens", "WBHP4_true", "GOR11_ens", "GOR11_true"):
            payload[f"iter_{it}_{key}"] = np.asarray(snap[key], dtype=float)
    np.savez_compressed(save_path, **payload)


def _plot_grid_slices(models_3d, vmin, vmax, title_prefix, out_path):
    nz = models_3d.shape[2]
    fig, axes = plt.subplots(1, nz, figsize=(4 * nz, 5), dpi=300)
    for k in range(nz):
        img = np.flipud(zoom(models_3d[:, :, k], zoom=6, order=3))
        axes[k].contourf(img, levels=100, cmap="jet", vmin=vmin, vmax=vmax)
        axes[k].set_title(f"{title_prefix} | Layer {k + 1}")
        axes[k].axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def visualize_ensemble_field(name, field_vecN, effective, nx, ny, nz, iter_id=0, sample_index=99, out_dir="viz_out", do_log=False):
    os.makedirs(out_dir, exist_ok=True)
    arr = field_vecN.copy()
    idx_valid = np.where(effective.ravel() != 0)[0]
    idx_invalid = np.where(effective.ravel() == 0)[0]
    arr[idx_valid, :] = np.log(arr[idx_valid, :] + 1e-8) if do_log else arr[idx_valid, :]
    arr[idx_invalid, :] = 0.0

    arr_TN = np.transpose(arr, (1, 0))
    models = np.array([np.reshape(arr_TN[i, :], (nx, ny, nz), order="F") for i in range(arr_TN.shape[0])], dtype=np.float64)

    with pd.ExcelWriter(os.path.join(out_dir, f"{name}_layers_iter{iter_id}.xlsx"), engine="xlsxwriter") as writer:
        for layer in range(nz):
            layer_2d = models[:, :, :, layer].reshape(models.shape[0], -1)
            df = pd.DataFrame(layer_2d, columns=[f"cell_{i}" for i in range(layer_2d.shape[1])])
            df.index.name = "model"
            df.to_excel(writer, sheet_name=f"Layer_{layer + 1}")

    vmin, vmax = float(np.percentile(models, 5)), float(np.percentile(models, 95))
    smp = int(np.clip(sample_index, 0, models.shape[0] - 1))
    _plot_grid_slices(models[smp], vmin, vmax, f"{name} | model#{smp + 1}", os.path.join(out_dir, f"{name}_sample_iter{iter_id}.png"))
    mean_field = np.mean(models, axis=0)
    _plot_grid_slices(mean_field, vmin, vmax, f"{name} | mean", os.path.join(out_dir, f"{name}_mean_iter{iter_id}.png"))
    np.savetxt(os.path.join(out_dir, f"{name}_mean_vec_iter{iter_id}.txt"), np.mean(field_vecN, axis=1, keepdims=True))


def summarize_perm_distribution(field_vecN, effective, iter_id=0, out_dir="viz_out", name="permX", do_log=True):
    os.makedirs(out_dir, exist_ok=True)
    idx_valid = np.where(effective.ravel() != 0)[0]
    vals = field_vecN[idx_valid, :].astype(np.float64)
    if do_log:
        vals = np.log10(np.clip(vals, 1e-15, None))
    flat = vals.ravel()
    q = np.percentile(flat, [1, 5, 25, 50, 75, 95, 99])
    pd.DataFrame({
        "iter": [iter_id],
        "mean": [float(np.mean(flat))],
        "std": [float(np.std(flat))],
        "min": [float(np.min(flat))],
        "p1": [float(q[0])],
        "p5": [float(q[1])],
        "p25": [float(q[2])],
        "p50": [float(q[3])],
        "p75": [float(q[4])],
        "p95": [float(q[5])],
        "p99": [float(q[6])],
        "max": [float(np.max(flat))],
    }).to_excel(os.path.join(out_dir, f"{name}_log10k_stats_iter{iter_id}.xlsx"), index=False)

    plt.figure(figsize=(5, 4), dpi=300)
    plt.hist(flat, bins=50, density=True, alpha=0.7)
    plt.xlabel("log10(k)" if do_log else "k")
    plt.ylabel("PDF")
    plt.title(f"{name} distribution (iter {iter_id})")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{name}_log10k_hist_iter{iter_id}.png"), dpi=300)
    plt.close()


def _state_file(result_root, iteration):
    path = os.path.join(result_root, "iteration_states", f"state_iter_{int(iteration):02d}.npz")
    return path if os.path.exists(path) else None


def _posterior_record(row, result_root, name, metric_name, metric_val):
    iteration = int(row["Iteration"])
    return {
        "selection_name": name,
        "selection_metric": metric_name,
        "selection_value": _safe_float(metric_val),
        "iteration": iteration,
        "state_file": _state_file(result_root, iteration),
        "RMSE": _safe_float(row.get("RMSE")),
        "ProdNRMSE": _safe_float(row.get("ProdNRMSE")),
        "MProd": _safe_float(row.get("MProd")),
        "MProdEnsembleMean": _safe_float(row.get("MProdEnsembleMean")),
        "J_GMR": _safe_float(row.get("J_GMR")),
        "J_unified": _safe_float(row.get("J_unified")),
        "GeoScore": _safe_float(row.get("GeoScore")),
        "GMRProdScore": _safe_float(row.get("GMRProdScore")),
        "AlphaUsed": _safe_float(row.get("AlphaUsed")),
        "InvAlphaSumAfter": _safe_float(row.get("InvAlphaSumAfter")),
        "is_final_iteration": False,
    }


def build_posterior_selection_archive(log_df, cfg):
    result_root = cfg.get("result_root") or os.path.dirname(cfg.get("summary_file", ""))
    archive = {"result_root": result_root, "selections": {}}
    if log_df.empty or "Iteration" not in log_df.columns:
        return archive

    last = log_df.iloc[-1]
    final_rec = _posterior_record(last, result_root, "final_posterior", "last_iteration", last.get("Iteration"))
    final_rec["is_final_iteration"] = True
    archive["selections"]["final_posterior"] = final_rec

    for metric, name in [
        ("J_GMR", "best_j_gmr_posterior"),
        ("J_unified", "best_j_unified_posterior"),
        ("ProdNRMSE", "best_prod_nrmse_posterior"),
        ("GeoScore", "best_geoscore_posterior"),
    ]:
        if metric in log_df.columns:
            pos = posterior_argmin(log_df["Iteration"].tolist(), log_df[metric].tolist())
            if pos is not None:
                row = log_df.iloc[pos]
                rec = _posterior_record(row, result_root, name, metric, row.get(metric))
                rec["is_final_iteration"] = rec["iteration"] == final_rec["iteration"]
                archive["selections"][name] = rec

    cb_file = cfg.get("cb_production_metrics_file")
    if cb_file and os.path.exists(cb_file):
        prod_df = pd.read_csv(cb_file)
        if {"Iteration", "MProd"}.issubset(prod_df.columns):
            pos = posterior_argmin(prod_df["Iteration"].tolist(), prod_df["MProd"].tolist())
            if pos is not None:
                prow = prod_df.iloc[pos]
                matching = log_df[pd.to_numeric(log_df["Iteration"], errors="coerce") == int(prow["Iteration"])]
                if not matching.empty:
                    row = matching.iloc[-1].copy()
                    row["MProd"] = prow.get("MProd")
                    row["MProdEnsembleMean"] = prow.get("MProdEnsembleMean")
                    rec = _posterior_record(row, result_root, "best_m_prod_posterior", "MProd", prow.get("MProd"))
                    rec["MProd"] = _safe_float(prow.get("MProd"))
                    rec["MProdEnsembleMean"] = _safe_float(prow.get("MProdEnsembleMean"))
                    rec["is_final_iteration"] = rec["iteration"] == final_rec["iteration"]
                    archive["selections"]["best_m_prod_posterior"] = rec

    apbo_file = cfg.get("apbo_policy_log_file")
    if apbo_file and os.path.exists(apbo_file):
        apbo_df = pd.read_csv(apbo_file)
        if {"Iteration", "Hypervolume"}.issubset(apbo_df.columns):
            df = apbo_df[apbo_df["Stage"].astype(str).str.lower() == "assimilation"].copy() if "Stage" in apbo_df.columns else apbo_df.copy()
            df["_it"] = pd.to_numeric(df["Iteration"], errors="coerce")
            df["_hv"] = pd.to_numeric(df["Hypervolume"], errors="coerce")
            df = df[(df["_it"] > 0) & np.isfinite(df["_hv"])]
            if not df.empty:
                idx = df["_hv"].idxmax()
                it = int(df.loc[idx, "_it"])
                archive["selections"]["best_hypervolume_posterior"] = {
                    "selection_name": "best_hypervolume_posterior",
                    "selection_metric": "Hypervolume",
                    "selection_value": _safe_float(df.loc[idx, "_hv"]),
                    "iteration": it,
                    "state_file": _state_file(result_root, it),
                    "is_final_iteration": it == final_rec["iteration"],
                }
    return archive


def save_run_summary(log_path, best_params_file, config_file, summary_file, runtime_sec, prior_metrics_file, timing_log=None):
    log_df = pd.read_csv(log_path)
    if "Iteration" not in log_df.columns:
        log_df = pd.read_csv(log_path, header=None, names=LOG_HEADER)
    zero_indices = list(log_df.index[log_df["Iteration"].astype(str) == "0"])
    if zero_indices:
        log_df = log_df.loc[zero_indices[-1]:].reset_index(drop=True)
    last = log_df.iloc[-1]

    best_rec = load_json(best_params_file) if os.path.exists(best_params_file) else {}
    cfg = load_json(config_file)
    prior_metrics = load_json(prior_metrics_file) if os.path.exists(prior_metrics_file) else {}

    final_cb = {}
    cb_path = cfg.get("cb_production_metrics_file")
    if cb_path and os.path.exists(cb_path):
        cb_df = pd.read_csv(cb_path)
        if "Stage" in cb_df.columns:
            cb_df = cb_df[cb_df["Stage"].astype(str) == "assimilation"]
        if not cb_df.empty:
            final_cb = cb_df.iloc[-1].to_dict()

    summary = {
        "run_tag": cfg["run_tag"],
        "description": cfg["description"],
        "paper_method_metadata": cfg.get("paper_method_metadata"),
        "feedback_mode": cfg.get("feedback_mode"),
        "result_root": cfg.get("result_root"),
        "run_id": cfg.get("run_id"),
        "runtime_sec": float(runtime_sec),
        "prior_rmse": prior_metrics.get("rmse_prior"),
        "prior_phi_ae": prior_metrics.get("phi_ae_prior"),
        "final_iteration": int(last["Iteration"]),
        "final_rmse": float(last["RMSE"]),
        "final_prod_nrmse": _safe_float(last.get("ProdNRMSE")),
        "final_m_prod": _safe_float(final_cb.get("MProd")),
        "final_m_prod_ensemble_mean": _safe_float(final_cb.get("MProdEnsembleMean")),
        "final_J_raw": float(last["J_raw"]),
        "final_J_unified": _safe_float(last.get("J_unified")),
        "final_J_GMR": _safe_float(last.get("J_GMR")),
        "final_GeoScore": _safe_float(last.get("GeoScore")),
        "final_alpha": _safe_float(last.get("AlphaUsed")),
        "final_inv_alpha_sum": _safe_float(last.get("InvAlphaSumAfter")),
        "best_iteration": best_rec.get("iteration"),
        "best_alpha": best_rec.get("alpha"),
        "alpha_path": [float(x) for x in log_df["AlphaUsed"].dropna()],
        "rmse_path": [float(x) for x in log_df["RMSE"].dropna()],
        "GeoScore_path": [_safe_float(x) for x in log_df["GeoScore"]] if "GeoScore" in log_df.columns else [],
    }
    if timing_log:
        keys = ("t_forward_sim", "t_esmda_update", "t_apbo_forward_guard", "t_gmr_correction", "t_bo_update", "t_iter_total")
        totals = {k: sum(r.get(k, 0.0) for r in timing_log) for k in keys}
        summary["timing_breakdown"] = {k: round(v, 4) for k, v in totals.items()}
        summary["timing_per_iter_avg"] = {k: round(v / len(timing_log), 4) for k, v in totals.items()}

    posterior_archive = build_posterior_selection_archive(log_df, cfg)
    posterior_path = cfg.get("posterior_selection_file")
    if posterior_path:
        save_json(posterior_archive, posterior_path)
    summary["posterior_selection_file"] = posterior_path
    summary["posterior_selections"] = posterior_archive
    save_json(summary, summary_file)
