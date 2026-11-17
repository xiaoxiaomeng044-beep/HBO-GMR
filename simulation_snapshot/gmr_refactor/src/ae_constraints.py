import os

import numpy as np

from ae_pseudo_opt_v3 import build_ae_pseudo, build_ae_recon_project_pseudo
from .common import save_json


AE_DISTRIBUTION_STAT_NAMES = ["mean", "std", "q50", "q75", "q90", "q95", "top10_mean"]
AE_PHI_ALIGNED_STAT_NAMES = ["q50", "mean", "q75"]
AE_PRIOR_MANIFOLD_STAT_NAMES = ["mean", "q50", "q75", "q90", "active_fraction", "top10_mean"]


def _json_safe(x):
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.floating):
        return float(x)
    if isinstance(x, np.ndarray):
        return None
    if isinstance(x, (list, tuple)):
        return [_json_safe(v) for v in list(x)[:20]]
    if isinstance(x, dict):
        return {str(k): _json_safe(v) for k, v in x.items()}
    if isinstance(x, (str, int, float, bool)) or x is None:
        return x
    return str(type(x))


def _array_stats(arr):
    a = np.asarray(arr, dtype=float)
    if a.size == 0:
        return {"shape": list(a.shape), "size": 0}
    flat = a.ravel()
    flat = flat[np.isfinite(flat)]
    if flat.size == 0:
        return {"shape": list(a.shape), "size": int(a.size), "finite_size": 0}
    qs = np.percentile(flat, [0, 5, 25, 50, 75, 95, 100])
    return {
        "shape": list(a.shape),
        "size": int(a.size),
        "finite_size": int(flat.size),
        "mean": float(np.mean(flat)),
        "std": float(np.std(flat)),
        "min": float(qs[0]),
        "p5": float(qs[1]),
        "p25": float(qs[2]),
        "p50": float(qs[3]),
        "p75": float(qs[4]),
        "p95": float(qs[5]),
        "max": float(qs[6]),
    }


def save_ae_pseudo_diagnostics(pack, out_dir, iter_id, prefix="ae_pseudo"):
    if pack is None:
        return None
    os.makedirs(out_dir, exist_ok=True)
    summary = {}
    payload = {}
    array_keys = []
    for k, v in pack.items():
        if isinstance(v, np.ndarray):
            summary[k] = _array_stats(v)
            payload[k] = v
            array_keys.append(k)
        elif isinstance(v, (list, tuple)):
            arr = np.asarray(v)
            if arr.dtype != object:
                summary[k] = _array_stats(arr)
                payload[k] = arr
            else:
                summary[k] = _json_safe(v)
        else:
            summary[k] = _json_safe(v)
    summary["array_keys"] = array_keys
    save_json(summary, os.path.join(out_dir, f"{prefix}_summary_iter{iter_id:02d}.json"))
    if payload:
        np.savez_compressed(os.path.join(out_dir, f"{prefix}_arrays_iter{iter_id:02d}.npz"), **payload)
    return summary


def compute_ae_constraint_diagnostics(pack):
    out = {
        "rows": np.nan,
        "d_mean": np.nan,
        "y_mean": np.nan,
        "residual_mean": np.nan,
        "residual_abs_mean": np.nan,
        "residual_p95": np.nan,
        "sigma": np.nan,
    }
    if not isinstance(pack, dict) or "D_AE" not in pack:
        return out
    D = np.asarray(pack["D_AE"], dtype=np.float64)
    y = np.asarray(pack.get("y_AE", 0.0), dtype=np.float64).reshape(-1)
    if D.ndim != 2 or D.size == 0:
        return out
    if y.size == 1:
        y = np.full(D.shape[0], y[0], dtype=np.float64)
    if y.size != D.shape[0]:
        return out
    flat_res = (D - y[:, None]).ravel()
    flat_res = flat_res[np.isfinite(flat_res)]
    out["rows"] = int(D.shape[0])
    out["d_mean"] = float(np.nanmean(D))
    out["y_mean"] = float(np.nanmean(y))
    out["sigma"] = float(pack.get("sigma_AE", np.nan))
    if flat_res.size:
        out["residual_mean"] = float(np.mean(flat_res))
        out["residual_abs_mean"] = float(np.mean(np.abs(flat_res)))
        out["residual_p95"] = float(np.percentile(flat_res, 95))
    return out


def estimate_sigma_from_ae_target(D_AE, y_AE):
    flat = (np.asarray(D_AE, dtype=np.float64) - np.asarray(y_AE, dtype=np.float64).reshape(-1, 1)).ravel()
    flat = flat[np.isfinite(flat)]
    if flat.size == 0:
        return 1e-4
    med = np.median(flat)
    mad = np.median(np.abs(flat - med)) + 1e-12
    return float(max(1.4826 * mad, np.std(flat) + 1e-12, 1e-4) * 1.25)


def convert_ae_pseudo_to_distribution(
    pack, y_ref=None, reference_q=0.25, reference_scale=1.0,
    nx=19, ny=28, nz=5, stat_names=None, constraint_form="distribution",
):
    if pack is None or "D_AE" not in pack:
        return None
    D = np.asarray(pack["D_AE"], dtype=np.float64)
    if D.ndim != 2 or D.size == 0:
        return None
    k_top = max(1, int(np.ceil(0.10 * D.shape[0])))
    all_rows = {
        "mean": np.mean(D, axis=0),
        "std": np.std(D, axis=0),
        "q50": np.percentile(D, 50, axis=0),
        "q75": np.percentile(D, 75, axis=0),
        "q90": np.percentile(D, 90, axis=0),
        "q95": np.percentile(D, 95, axis=0),
        "top10_mean": np.mean(np.sort(D, axis=0)[-k_top:, :], axis=0),
    }
    names = list(stat_names or AE_DISTRIBUTION_STAT_NAMES)
    D_dist = np.vstack([all_rows[n] for n in names if n in all_rows]).astype(np.float32)

    if y_ref is not None and len(y_ref) == D_dist.shape[0]:
        y_dist = np.asarray(y_ref, dtype=np.float32).reshape(-1)
    else:
        y_dist = (np.quantile(D_dist, float(np.clip(reference_q, 0.0, 0.49)), axis=1) * reference_scale).astype(np.float32)

    center = np.array([[0.5 * (nx - 1), 0.5 * (ny - 1), 0.5 * (nz - 1)]], dtype=np.float32)
    out = dict(pack)
    out["D_AE_patch_original"] = np.asarray(pack["D_AE"], dtype=np.float32)
    out["y_AE_patch_original"] = np.asarray(pack.get("y_AE", []), dtype=np.float32)
    out["D_AE"] = D_dist
    out["y_AE"] = y_dist
    out["sigma_AE"] = estimate_sigma_from_ae_target(D_dist, y_dist)
    out["constraint_form"] = constraint_form
    out["distribution_stat_names"] = np.asarray(names)
    out["coords_xyz"] = np.repeat(center.astype(np.int32), D_dist.shape[0], axis=0)
    out["centers_xyz"] = np.repeat(center, D_dist.shape[0], axis=0)
    out["patch_shape_xyz"] = np.array([nx, ny, nz], dtype=np.int32)
    out["localization_radius"] = float(max(nx, ny, nz) * 10.0)
    return out


def build_prior_manifold_reference(pack, envelope_q=0.90, scale_floor=1e-4):
    if pack is None or "D_AE" not in pack:
        return None
    D = np.asarray(pack["D_AE"], dtype=np.float64)
    if D.ndim != 2 or D.size == 0:
        return None
    q = float(np.clip(envelope_q, 0.50, 0.99))
    floor = max(scale_floor, 1e-12)
    envelope = np.quantile(D, q, axis=1).astype(np.float32)
    mad = np.median(np.abs(D - np.median(D, axis=1, keepdims=True)), axis=1)
    scale = np.maximum(np.maximum(1.4826 * mad, np.std(D, axis=1)), floor).astype(np.float32)
    return {"envelope": envelope, "scale": scale, "envelope_q": q, "scale_floor": floor}


def convert_ae_pseudo_to_prior_manifold(
    pack, reference=None, envelope_q=0.90, scale_floor=1e-4, mode="patch",
    exceedance_cap=5.0, stats_y_ref=None, stats_target_q=0.50, nx=19, ny=28, nz=5,
):
    if pack is None or "D_AE" not in pack:
        return None
    D = np.asarray(pack["D_AE"], dtype=np.float64)
    if D.ndim != 2 or D.size == 0:
        return None
    ref = reference if reference is not None else build_prior_manifold_reference(pack, envelope_q, scale_floor)
    if ref is None:
        return None
    envelope = np.asarray(ref["envelope"], dtype=np.float64).reshape(-1)
    scale = np.maximum(np.asarray(ref["scale"], dtype=np.float64).reshape(-1), max(scale_floor, 1e-12))
    if envelope.size != D.shape[0] or scale.size != D.shape[0]:
        return None

    D_raw_excess = np.maximum(0.0, (D - envelope[:, None]) / scale[:, None])
    D_excess = np.minimum(D_raw_excess, exceedance_cap) if np.isfinite(exceedance_cap) and exceedance_cap > 0 else D_raw_excess

    out = dict(pack)
    out["D_AE_patch_original"] = np.asarray(pack["D_AE"], dtype=np.float32)
    out["y_AE_patch_original"] = np.asarray(pack.get("y_AE", []), dtype=np.float32)
    out["D_AE_exceedance_raw"] = D_raw_excess.astype(np.float32)
    out["D_AE_exceedance_clipped"] = D_excess.astype(np.float32)

    if mode.lower() == "stats":
        k_top = max(1, int(np.ceil(0.10 * D_excess.shape[0])))
        stats_rows = np.vstack([
            np.mean(D_excess, axis=0),
            np.percentile(D_excess, 50, axis=0),
            np.percentile(D_excess, 75, axis=0),
            np.percentile(D_excess, 90, axis=0),
            np.mean(D_raw_excess > 0.0, axis=0),
            np.mean(np.sort(D_excess, axis=0)[-k_top:, :], axis=0),
        ]).astype(np.float32)
        if stats_y_ref is not None and len(stats_y_ref) == stats_rows.shape[0]:
            y_ae = np.asarray(stats_y_ref, dtype=np.float32).reshape(-1)
        else:
            y_ae = np.quantile(stats_rows, float(np.clip(stats_target_q, 0.0, 0.95)), axis=1).astype(np.float32)
        center = np.array([[0.5 * (nx - 1), 0.5 * (ny - 1), 0.5 * (nz - 1)]], dtype=np.float32)
        out["D_AE"] = stats_rows
        out["y_AE"] = y_ae
        out["prior_manifold_stat_names"] = np.asarray(AE_PRIOR_MANIFOLD_STAT_NAMES)
        out["coords_xyz"] = np.repeat(center.astype(np.int32), stats_rows.shape[0], axis=0)
        out["centers_xyz"] = np.repeat(center, stats_rows.shape[0], axis=0)
        out["patch_shape_xyz"] = np.array([nx, ny, nz], dtype=np.int32)
        out["localization_radius"] = float(max(nx, ny, nz) * 10.0)
    else:
        out["D_AE"] = D_excess.astype(np.float32)
        out["y_AE"] = np.zeros(D.shape[0], dtype=np.float32)

    out["sigma_AE"] = estimate_sigma_from_ae_target(out["D_AE"], out["y_AE"])
    out["constraint_form"] = "prior_manifold"
    out["target_mode"] = "sgs_prior_envelope_exceedance"
    out["prior_manifold_mode"] = mode.lower()
    out["prior_exceedance_cap"] = np.array([exceedance_cap], dtype=np.float32)
    out["prior_envelope"] = envelope.astype(np.float32)
    out["prior_scale"] = scale.astype(np.float32)
    return out


def apply_ae_sigma_mode(pack, sigma_ref, sigma_mode="adaptive", cap_factor=20.0):
    if pack is None or "sigma_AE" not in pack:
        return pack
    adaptive_sigma = float(pack["sigma_AE"])
    pack["sigma_AE_adaptive"] = adaptive_sigma
    if sigma_ref is None or not np.isfinite(sigma_ref) or sigma_ref <= 0:
        pack["sigma_mode"] = "adaptive_fallback"
        return pack
    if sigma_mode == "prior_fixed":
        pack["sigma_AE"] = float(sigma_ref)
        pack["sigma_mode"] = "prior_fixed"
    elif sigma_mode == "capped":
        cap = float(sigma_ref) * max(cap_factor, 1.0)
        pack["sigma_AE"] = min(adaptive_sigma, cap)
        pack["sigma_mode"] = "capped"
        pack["sigma_AE_cap"] = cap
        pack["sigma_AE_prior_ref"] = float(sigma_ref)
    else:
        pack["sigma_mode"] = "adaptive"
    return pack


def _interp_schedule(iter_idx, total_assim, start_iter, val_hi, val_lo):
    if iter_idx < start_iter:
        return None
    if total_assim <= start_iter:
        return float(val_lo)
    prog = float(np.clip((iter_idx - start_iter) / max(1, total_assim - start_iter), 0.0, 1.0))
    return float(val_hi + (val_lo - val_hi) * prog)


def prepare_prior_ae_references(ctx, ae_bundle, ae_patch_coords_xyz, mumyperm, mumyporo, effective):
    refs = {
        "sigma_ref": None,
        "target_ref": None,
        "dist_target_ref": None,
        "dist_sigma_ref": None,
        "manifold_ref": None,
        "manifold_stats_target_ref": None,
    }
    if not (ctx.use_ae_constraint and not ctx.use_gmr_direction and ae_bundle is not None):
        return refs

    args, ae_cfg, grid = ctx.args, ctx.ae_cfg, ctx.grid
    ae_diag_dir = ctx.dirs["ae_diag"]
    pack0 = build_ae_pseudo(
        mumyperm, mumyporo, grid.nx, grid.ny, grid.nz, ae_bundle,
        n_patches=ae_cfg["ae_patches"], seed=ctx.exp["seed"], pad_to_trained=True,
        coords_xyz=ae_patch_coords_xyz, target_quantile=ae_cfg["ae_target_q_hi"],
        hard_ratio=ae_cfg["ae_hard_ratio"],
    )
    if pack0 is None:
        return refs

    refs["sigma_ref"] = float(pack0["sigma_AE"])
    save_ae_pseudo_diagnostics(pack0, ae_diag_dir, 0, prefix="prior_ae_pseudo")

    if args.ae_constraint_form == "recon_project":
        p_recon = build_ae_recon_project_pseudo(
            mumyperm, mumyporo, grid.nx, grid.ny, grid.nz, ae_bundle,
            n_patches=ae_cfg["ae_patches"], seed=ctx.exp["seed"], pad_to_trained=True,
            coords_xyz=ae_patch_coords_xyz, effective=effective,
            hard_ratio=ae_cfg["ae_hard_ratio"], max_obs=args.ae_recon_max_obs,
            blend=args.ae_recon_blend, active_fraction_min=args.ae_recon_active_fraction_min,
        )
        if p_recon is not None:
            refs["dist_sigma_ref"] = float(p_recon["sigma_AE"])
            save_ae_pseudo_diagnostics(p_recon, ae_diag_dir, 0, prefix="prior_ae_recon_project_target")
    elif args.ae_constraint_form in ("distribution", "phi_aligned"):
        stat_names = AE_PHI_ALIGNED_STAT_NAMES if args.ae_constraint_form == "phi_aligned" else AE_DISTRIBUTION_STAT_NAMES
        p_dist = convert_ae_pseudo_to_distribution(
            pack0, y_ref=None, reference_q=args.ae_reference_q, reference_scale=args.ae_reference_scale,
            nx=grid.nx, ny=grid.ny, nz=grid.nz, stat_names=stat_names, constraint_form=args.ae_constraint_form,
        )
        if p_dist is not None:
            refs["dist_target_ref"] = np.asarray(p_dist["y_AE"], dtype=np.float32)
            refs["dist_sigma_ref"] = float(p_dist["sigma_AE"])
            save_ae_pseudo_diagnostics(p_dist, ae_diag_dir, 0, prefix="prior_ae_distribution_target")
    elif args.ae_constraint_form == "prior_manifold":
        refs["manifold_ref"] = build_prior_manifold_reference(
            pack0, envelope_q=args.ae_prior_envelope_q, scale_floor=args.ae_prior_scale_floor
        )
        p_man = convert_ae_pseudo_to_prior_manifold(
            pack0, reference=refs["manifold_ref"], envelope_q=args.ae_prior_envelope_q,
            scale_floor=args.ae_prior_scale_floor, mode=args.ae_prior_manifold_mode,
            exceedance_cap=args.ae_prior_exceedance_cap, stats_target_q=args.ae_prior_target_q,
            nx=grid.nx, ny=grid.ny, nz=grid.nz,
        )
        if p_man is not None:
            if args.ae_prior_manifold_mode == "stats":
                refs["manifold_stats_target_ref"] = np.asarray(p_man["y_AE"], dtype=np.float32)
            refs["dist_sigma_ref"] = float(p_man["sigma_AE"])
            save_ae_pseudo_diagnostics(p_man, ae_diag_dir, 0, prefix="prior_ae_manifold_target")

    if args.ae_target_mode == "prior_fixed" and args.ae_constraint_form != "prior_manifold":
        pack_ref = build_ae_pseudo(
            mumyperm, mumyporo, grid.nx, grid.ny, grid.nz, ae_bundle,
            n_patches=ae_cfg["ae_patches"], seed=ctx.exp["seed"], pad_to_trained=True,
            coords_xyz=ae_patch_coords_xyz, target_quantile=args.ae_reference_q,
            hard_ratio=ae_cfg["ae_hard_ratio"],
        )
        if pack_ref is not None and "y_AE" in pack_ref:
            refs["target_ref"] = np.asarray(pack_ref["y_AE"], dtype=np.float32) * args.ae_reference_scale
            pack_ref = dict(pack_ref)
            pack_ref["y_AE"] = refs["target_ref"].astype(np.float32)
            pack_ref["sigma_AE"] = estimate_sigma_from_ae_target(pack_ref["D_AE"], pack_ref["y_AE"])
            save_ae_pseudo_diagnostics(pack_ref, ae_diag_dir, 0, prefix="prior_fixed_ae_target")
    return refs


def build_step_ae_pseudo(ctx, ae_bundle, ae_patch_coords_xyz, prior_refs, mumyperm, mumyporo, effective, iteration):
    ae_cfg, args, grid = ctx.ae_cfg, ctx.args, ctx.grid
    if not (ctx.use_ae_constraint and not ctx.use_gmr_direction and ae_bundle is not None and iteration >= ae_cfg["ae_start_iter"]):
        return None, None, None, None, compute_ae_constraint_diagnostics(None)

    sigma_scale_now = _interp_schedule(
        iteration, ctx.n_assim, ae_cfg["ae_start_iter"], ae_cfg["ae_sigma_scale"], ae_cfg["ae_sigma_min_scale"]
    )
    target_q_now = _interp_schedule(
        iteration, ctx.n_assim, ae_cfg["ae_start_iter"], ae_cfg["ae_target_q_hi"], ae_cfg["ae_target_q_lo"]
    )
    seed_iter = ctx.exp["seed"] + iteration * 100

    if args.ae_constraint_form == "recon_project":
        ae_pseudo = build_ae_recon_project_pseudo(
            mumyperm, mumyporo, grid.nx, grid.ny, grid.nz, ae_bundle,
            n_patches=ae_cfg["ae_patches"], seed=seed_iter, pad_to_trained=True,
            coords_xyz=ae_patch_coords_xyz, effective=effective,
            hard_ratio=ae_cfg["ae_hard_ratio"], max_obs=args.ae_recon_max_obs,
            blend=args.ae_recon_blend, active_fraction_min=args.ae_recon_active_fraction_min,
        )
    else:
        ae_pseudo = build_ae_pseudo(
            mumyperm, mumyporo, grid.nx, grid.ny, grid.nz, ae_bundle,
            n_patches=ae_cfg["ae_patches"], seed=seed_iter, pad_to_trained=True,
            coords_xyz=ae_patch_coords_xyz, target_quantile=target_q_now,
            hard_ratio=ae_cfg["ae_hard_ratio"],
        )

    if ae_pseudo is None:
        return None, None, sigma_scale_now, target_q_now, compute_ae_constraint_diagnostics(None)

    if args.ae_constraint_form not in ("recon_project", "prior_manifold") and args.ae_target_mode == "prior_fixed":
        t_ref = prior_refs["target_ref"]
        if t_ref is not None and len(t_ref) == ae_pseudo["D_AE"].shape[0]:
            ae_pseudo["y_AE"] = np.asarray(t_ref, dtype=np.float32)
            ae_pseudo["sigma_AE"] = estimate_sigma_from_ae_target(ae_pseudo["D_AE"], ae_pseudo["y_AE"])
            ae_pseudo["target_mode"] = "prior_fixed"

    if args.ae_constraint_form in ("distribution", "phi_aligned"):
        save_ae_pseudo_diagnostics(ae_pseudo, ctx.dirs["ae_diag"], iteration, prefix="active_ae_patch_pseudo")
        stat_names = AE_PHI_ALIGNED_STAT_NAMES if args.ae_constraint_form == "phi_aligned" else AE_DISTRIBUTION_STAT_NAMES
        ae_pseudo = convert_ae_pseudo_to_distribution(
            ae_pseudo, y_ref=prior_refs["dist_target_ref"], reference_q=args.ae_reference_q,
            reference_scale=args.ae_reference_scale, nx=grid.nx, ny=grid.ny, nz=grid.nz,
            stat_names=stat_names, constraint_form=args.ae_constraint_form,
        )
        if ae_pseudo is not None:
            ae_pseudo["target_mode"] = f"prior_{args.ae_constraint_form}"

    if args.ae_constraint_form == "prior_manifold":
        save_ae_pseudo_diagnostics(ae_pseudo, ctx.dirs["ae_diag"], iteration, prefix="active_ae_patch_pseudo")
        if prior_refs["manifold_ref"] is None:
            ae_pseudo = None
        else:
            ae_pseudo = convert_ae_pseudo_to_prior_manifold(
                ae_pseudo, reference=prior_refs["manifold_ref"], envelope_q=args.ae_prior_envelope_q,
                scale_floor=args.ae_prior_scale_floor, mode=args.ae_prior_manifold_mode,
                exceedance_cap=args.ae_prior_exceedance_cap, stats_y_ref=prior_refs["manifold_stats_target_ref"],
                stats_target_q=args.ae_prior_target_q, nx=grid.nx, ny=grid.ny, nz=grid.nz,
            )

    if ae_pseudo is not None:
        sigma_ref = (
            prior_refs["dist_sigma_ref"]
            if args.ae_constraint_form in ("distribution", "phi_aligned", "recon_project", "prior_manifold")
            else prior_refs["sigma_ref"]
        )
        ae_pseudo = apply_ae_sigma_mode(ae_pseudo, sigma_ref=sigma_ref, sigma_mode=args.ae_sigma_mode, cap_factor=args.ae_sigma_cap_factor)
        ae_pseudo["sigma_AE"] = float(ae_pseudo["sigma_AE"]) * sigma_scale_now
        if args.ae_constraint_form not in ("distribution", "phi_aligned") and not (
            args.ae_constraint_form == "prior_manifold" and args.ae_prior_manifold_mode == "stats"
        ):
            ae_pseudo["localization_radius"] = ae_cfg["ae_localization_radius"]

    diag = compute_ae_constraint_diagnostics(ae_pseudo)
    summary = save_ae_pseudo_diagnostics(ae_pseudo, ctx.dirs["ae_diag"], iteration, prefix="active_ae_pseudo")
    return ae_pseudo, summary, sigma_scale_now, target_q_now, diag
