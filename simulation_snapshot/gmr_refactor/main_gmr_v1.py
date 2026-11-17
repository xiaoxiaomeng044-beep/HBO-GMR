import csv
import os
import sys
import time

import matplotlib.pyplot as plt
import numpy as np

plt.switch_backend("Agg")

GMR_ROOT = os.path.abspath(os.path.dirname(__file__))
GMR_CORE_DIR = os.path.join(GMR_ROOT, "core")
PROJECT_ROOT = os.path.abspath(os.path.join(GMR_ROOT, ".."))
if GMR_CORE_DIR not in sys.path:
    sys.path.insert(0, GMR_CORE_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from ae_pseudo_opt_v3 import build_patch_bank_xyz, try_load_ae
from eclipse import eclipse
from getcostfunctionvalue import getcostfunctionvalue
from gethistorydata import gethistorydata
from getsimulated import getsimulated
from metrics_geology import build_geology_reference, vec_to_dhw
from metrics_production import compute_nrmse_from_rsm_ensemble
from phi_ae_eval_opt import compute_phi_ae_for_field
from plot_figure5 import plot_history_match_figure_advanced

from gmr_refactor.src.ae_constraints import build_step_ae_pseudo, prepare_prior_ae_references
from gmr_refactor.src.apbo_budget_policy import (
    compute_apbo_feedback,
    compute_prod_norm,
    eta_to_budget_decision,
    next_state as make_apbo_state,
)
from gmr_refactor.src.apbo_safe_guard import resolve_optimizer_feedback
from gmr_refactor.src.common import append_csv_row, save_json
from gmr_refactor.src.experiment_config import (
    ALPHA_MAX,
    MIN_ASSIM_ITERS,
    build_parser,
    check_validation_exit,
    prepare_result_paths,
    resolve_run_context,
)
from gmr_refactor.src.forward_guards import (
    apply_apbo_equal_budget_guard,
    apply_gmr_forward_guard,
    compute_esmda_candidate,
    write_member_grdecl,
)
from gmr_refactor.src.integrate_gmr_direction import (
    GMRDirectionConfig,
    compute_production_guard_delta,
    make_gmr_direction_corrector,
)
from gmr_refactor.src.run_reporting import (
    LOG_HEADER,
    compute_bo_feedback_score,
    compute_epc_score,
    compute_epc_star_score,
    compute_external_geology_row,
    compute_geo_score,
    compute_j_gmr,
    compute_j_raw,
    compute_j_unified,
    production_metric_value,
    record_cb_production_metrics,
    save_history_match_snapshots_npz,
    save_iteration_state_npz,
    save_run_summary,
    summarize_perm_distribution,
    update_iteration_status,
    visualize_ensemble_field,
)


def _needs_log_header(path):
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return True
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        first_line = f.readline().strip()
    return (first_line.split(",") if first_line else []) != LOG_HEADER


def _init_online_optimizer(ctx):
    if not ctx.use_online_bo:
        return None
    from skopt import Optimizer
    from skopt.space import Real

    args = ctx.args
    if ctx.use_apbo_budget_policy:
        dims = [Real(args.apbo_eta_min, args.apbo_eta_max, prior="uniform", name="budget_eta")]
    else:
        dims = [Real(2.0, 8.0, prior="log-uniform", name="alpha_mda")]
    if ctx.use_localization and not ctx.bo_alpha_only:
        dims.append(Real(2.0, 6.0, prior="uniform", name="c_radius"))

    opt = Optimizer(
        dimensions=dims,
        base_estimator="GP",
        acq_func="EI",
        random_state=ctx.exp["seed"],
        n_initial_points=2,
    )
    warm_path = os.path.join(ctx.dirs["bo_diagnostics"], "bo_warm_start.json")
    if ctx.use_apbo_budget_policy:
        save_json(
            {
                "feedback_mode": args.feedback,
                "apbo_policy": args.apbo_policy,
                "apbo_objective_effective": ctx.apbo_objective_effective,
                "bo_dimensions": [str(d) for d in dims],
                "warm_start": "none_for_budget_ratio_policy",
            },
            warm_path,
        )
    else:
        p1 = ([4.0], 1.20) if ctx.bo_alpha_only else ([4.0, 3.0], 1.20)
        p2 = ([2.25], 2.20) if ctx.bo_alpha_only else ([2.25, 4.6], 2.20)
        opt.tell([p1[0], p2[0]], [p1[1], p2[1]])
        save_json(
            {"feedback_mode": args.feedback, "bo_dimensions": [str(d) for d in dims], "warm_start_x": [p1[0], p2[0]], "warm_start_y": [p1[1], p2[1]]},
            warm_path,
        )
    return opt


def _make_gmr_corrector(ctx, prior_perm, prior_poro, effective, geology_reference):
    args = ctx.args
    cfg = GMRDirectionConfig(
        enabled=True,
        vae_ckpt=args.gmr_vae_ckpt,
        flow_ckpt=args.gmr_flow_ckpt,
        dataset_meta=args.gmr_dataset_meta,
        device=args.gmr_device,
        batch_size=args.gmr_batch_size,
        max_realizations=None if args.gmr_max_realizations <= 0 else args.gmr_max_realizations,
        activate_from_iter=args.gmr_activate_from_iter,
        lambda_base=args.gmr_lambda_base,
        lambda_min=args.gmr_lambda_min,
        lambda_max=args.gmr_lambda_max,
        score_step_base=args.gmr_score_step_base,
        score_clip=args.gmr_score_clip,
        strength_mode=args.gmr_strength_mode,
        violation_ref=args.gmr_violation_ref,
        stage_mode=args.gmr_stage_mode,
        stage_floor=args.gmr_stage_floor,
        prod_guard_beta=args.gmr_prod_guard_beta,
        prod_guard_tolerance=args.gmr_prod_guard_tol,
        force_lambda=None if args.gmr_force_lambda < 0 else args.gmr_force_lambda,
        diagnostics_dir=ctx.dirs["gmr_diag"],
        method_name=ctx.run_tag,
        run_id=ctx.run_id,
        write_diagnostics=True,
        global_guard_enabled=ctx.gmr_internal_global_guard,
        global_guard_tolerance=args.gmr_global_guard_tol,
        global_guard_policy=args.gmr_global_guard_policy,
        global_guard_min_geo_gain=args.gmr_global_guard_min_geo_gain,
    )
    return make_gmr_direction_corrector(
        cfg,
        ctx.grid.nx,
        ctx.grid.ny,
        ctx.grid.nz,
        prior_perm=prior_perm,
        prior_poro=prior_poro,
        effective=effective,
        geology_reference=geology_reference,
    )


def _mean_fields_dhw(perm_mat, poro_mat, grid):
    log10k = np.log10(np.maximum(np.mean(perm_mat, axis=1), 1e-15))
    poro = np.mean(poro_mat, axis=1)
    return (
        vec_to_dhw(log10k, grid.nx, grid.ny, grid.nz).astype(np.float32),
        vec_to_dhw(poro, grid.nx, grid.ny, grid.nz).astype(np.float32),
    )


def _evaluate_step_metrics(ctx, work_dir, timestep, iteration, stage, perm, poro, permz, effective, ae_bundle, ae_patch_coords_xyz, geology_reference):
    index, snapshot = getcostfunctionvalue(timestep, ctx.N, iteration, save_single_plots=False)
    rmse_prod = float(index.mean())
    try:
        prod_info = compute_nrmse_from_rsm_ensemble(ctx.N, real_path="Real.RSM", root_dir=work_dir)
    except Exception:
        prod_info = {"prod_nrmse": np.nan, "prod_nrmse_best": np.nan, "prod_nrmse_std": np.nan}
    record_cb_production_metrics(ctx, iteration, stage, prod_info)
    selected_prod = production_metric_value(prod_info, ctx.args.production_metric)

    log10k_mean, poro_mean = _mean_fields_dhw(perm, poro, ctx.grid)
    if ctx.eval_ae_metric and ae_bundle is not None:
        ae_cfg = ctx.ae_cfg
        phi_ae, _ = compute_phi_ae_for_field(
            log10k_mean,
            poro_mean,
            ckpt_dir=ae_cfg["ae_dir"],
            n_patches=ae_cfg["ae_patches"],
            agg=ae_cfg["ae_agg"],
            device=ae_cfg["ae_device"],
            return_details=True,
            save_npz=os.path.join(ctx.dirs["ae_diag"], f"iter_{iteration:02d}.npz"),
            topk=ctx.args.ae_eval_topk if iteration == 0 else 8,
            seed=2025,
            coords_xyz=ae_patch_coords_xyz,
        )
    else:
        phi_ae = np.nan

    if geology_reference is None and iteration == 0:
        mask = vec_to_dhw(effective.ravel() != 0, ctx.grid.nx, ctx.grid.ny, ctx.grid.nz).astype(bool)
        geology_reference = build_geology_reference(log10k_mean, poro_mean, mask=mask)

    geo_row = {}
    if geology_reference is not None:
        geo_row = compute_external_geology_row(ctx, iteration, log10k_mean, poro_mean, geology_reference, phi_ae=phi_ae)
        append_csv_row(ctx.paths["geology_metrics"], geo_row)

    return index, snapshot, rmse_prod, prod_info, selected_prod, phi_ae, geo_row, geology_reference


def _save_visuals(ctx, iter_id, mumyperm, mumyporo, mumypermz, effective):
    if not ctx.args.save_iteration_visuals:
        return
    viz_dir = os.path.join(ctx.result_root, f"viz_iter_{iter_id:02d}")
    os.makedirs(viz_dir, exist_ok=True)
    visualize_ensemble_field("permX", mumyperm, effective, ctx.grid.nx, ctx.grid.ny, ctx.grid.nz, iter_id, 99, viz_dir, do_log=True)
    visualize_ensemble_field("poro", mumyporo, effective, ctx.grid.nx, ctx.grid.ny, ctx.grid.nz, iter_id, 99, viz_dir, do_log=False)
    visualize_ensemble_field("permZ", mumypermz, effective, ctx.grid.nx, ctx.grid.ny, ctx.grid.nz, iter_id, 99, viz_dir, do_log=True)
    summarize_perm_distribution(mumyperm, effective, iter_id=iter_id, out_dir=viz_dir, name="permX", do_log=True)


def _select_step_params(ctx, opt, inelly, inv_alpha_sum, apbo_tell_count, next_alpha_lm, prev_j_raw_lm):
    B_before = 1.0 - inv_alpha_sum
    LB_now = float("inf") if B_before <= 0 else 1.0 / B_before
    alpha_suggest = float("nan")
    c_suggest = float("nan")
    apbo_decision = None
    will_stop_after = False

    if ctx.use_online_bo:
        if B_before <= 0.0:
            return None
        rem_steps = ctx.n_assim - inelly + 1
        is_final = rem_steps == 1
        if not is_final and ctx.use_apbo_budget_policy:
            if apbo_tell_count <= 0:
                bo_x = [1.0] if (ctx.bo_alpha_only or not ctx.use_localization) else [1.0, ctx.fixed_bo_c_radius]
            else:
                bo_x = opt.ask()
            eta_suggest = float(bo_x[0])
            c_radius_used = (
                ctx.fixed_bo_c_radius
                if ctx.bo_alpha_only
                else float(np.clip(bo_x[1], 1.0, 12.0))
            ) if ctx.use_localization else None
            apbo_decision = eta_to_budget_decision(
                eta_suggest,
                budget_remaining=B_before,
                remaining_steps=rem_steps,
                alpha_min=2.0,
                alpha_max=ALPHA_MAX,
                eta_min=ctx.args.apbo_eta_min,
                eta_max=ctx.args.apbo_eta_max,
            )
            alpha_used = apbo_decision.alpha_used
            if apbo_decision.u_equal > 0:
                alpha_suggest = 1.0 / max(apbo_decision.u_equal * apbo_decision.eta_suggest, 1e-12)
            will_stop_after = apbo_decision.will_stop_after
        elif is_final:
            alpha_used = max(2.0, float("inf") if B_before <= 0 else 1.0 / B_before)
            will_stop_after = True
            c_radius_used = ctx.fixed_bo_c_radius if (ctx.use_localization and ctx.bo_alpha_only) else (
                float(ctx.fixed_c_radius) if ctx.fixed_c_radius is not None else 5.0
            ) if ctx.use_localization else None
        else:
            bo_x = opt.ask()
            alpha_suggest = float(bo_x[0])
            c_radius_used = (
                ctx.fixed_bo_c_radius
                if ctx.bo_alpha_only
                else float(np.clip(bo_x[1], 1.0, 12.0))
            ) if ctx.use_localization else None
            alpha_used = min(max(alpha_suggest, LB_now, 2.0), ALPHA_MAX) if B_before > 0 else ALPHA_MAX
            will_stop_after = alpha_used <= LB_now + 1e-12
            if inelly < MIN_ASSIM_ITERS and will_stop_after:
                reserve = MIN_ASSIM_ITERS - inelly
                allowed_max = max(1e-8, B_before - reserve / ALPHA_MAX)
                alpha_used = max(alpha_used, 1.0 / allowed_max)
                will_stop_after = False
    elif ctx.alpha_schedule == "lm_adaptive":
        raw_alpha = 8.0 if next_alpha_lm is None else next_alpha_lm
        alpha_used = min(max(raw_alpha, LB_now, 2.0), ALPHA_MAX) if B_before > 0 else ALPHA_MAX
        will_stop_after = alpha_used <= LB_now + 1e-12
        if inelly < MIN_ASSIM_ITERS and will_stop_after:
            allowed_max = max(1e-8, B_before - (MIN_ASSIM_ITERS - inelly) / ALPHA_MAX)
            alpha_used = max(alpha_used, 1.0 / allowed_max)
            will_stop_after = False
        c_radius_used = float(ctx.fixed_c_radius) if ctx.use_localization else None
    elif ctx.alpha_schedule is not None:
        alpha_used = float(ctx.alpha_schedule[inelly - 1])
        c_radius_used = float(ctx.fixed_c_radius) if ctx.use_localization else None
    else:
        alpha_used = float(ctx.fixed_alpha)
        c_radius_used = float(ctx.fixed_c_radius) if ctx.use_localization else None

    return {
        "B_before": B_before,
        "LB_now": LB_now,
        "alpha_suggest": alpha_suggest,
        "alpha_used": alpha_used,
        "c_radius_used": c_radius_used,
        "apbo_decision": apbo_decision,
        "will_stop_after": will_stop_after,
    }


def _row_from_metrics(ctx, inelly, step, rmse_prod, prod_info, phi_ae, j_raw, j_unified, j_gmr, geo_row, gmr_prod_score, epc_tuple, epc_star_tuple, score_for_bo, score_for_bo_name, j_bo_value, inv_alpha_before, inv_alpha_after, ae_active, ae_pseudo, sigma_scale_now, target_q_now, ae_diag, ae_evidence_flag, phi_before, rmse_before):
    epc_score, t_acc, t_bud, t_spa = epc_tuple
    epc_star, t_fit_s, t_bud_s, t_spa_s, rmse_norm, phi_norm = epc_star_tuple
    phi_delta = (phi_ae - phi_before) if (phi_before is not None and np.isfinite(phi_before) and np.isfinite(phi_ae)) else np.nan
    rmse_delta = (rmse_prod - rmse_before) if (rmse_before is not None and np.isfinite(rmse_before)) else np.nan
    return [
        inelly,
        step["alpha_suggest"] if np.isfinite(step["alpha_suggest"]) else np.nan,
        step["alpha_used"],
        step["c_radius_used"] if step["c_radius_used"] is not None else np.nan,
        rmse_prod,
        float(phi_ae) if np.isfinite(phi_ae) else np.nan,
        j_raw,
        float(j_unified) if np.isfinite(j_unified) else np.nan,
        epc_score, t_acc, t_bud, t_spa,
        epc_star, t_fit_s, t_bud_s, t_spa_s, rmse_norm, phi_norm,
        float(score_for_bo) if np.isfinite(score_for_bo) else np.nan,
        score_for_bo_name,
        ctx.args.feedback,
        step["B_before"],
        step["LB_now"],
        inv_alpha_before,
        inv_alpha_after,
        step["will_stop_after"],
        ctx.bo_alpha_only,
        ae_active,
        sigma_scale_now if sigma_scale_now is not None else np.nan,
        target_q_now if target_q_now is not None else np.nan,
        ctx.args.ae_preset,
        ctx.ae_cfg["ae_hard_ratio"],
        ae_pseudo.get("sigma_AE", np.nan) if isinstance(ae_pseudo, dict) else np.nan,
        ctx.ae_cfg["ae_localization_radius"],
        phi_before if (phi_before is not None and np.isfinite(phi_before)) else np.nan,
        phi_delta,
        rmse_before if (rmse_before is not None and np.isfinite(rmse_before)) else np.nan,
        rmse_delta,
        rmse_prod,
        prod_info.get("prod_nrmse", np.nan),
        prod_info.get("prod_nrmse_best", np.nan),
        prod_info.get("prod_nrmse_std", np.nan),
        float(j_bo_value) if np.isfinite(j_bo_value) else np.nan,
        float(j_gmr) if np.isfinite(j_gmr) else np.nan,
        geo_row.get("GeoScore", np.nan),
        float(gmr_prod_score) if np.isfinite(gmr_prod_score) else np.nan,
        ctx.args.ae_constraint_form,
        ae_pseudo.get("target_mode", ctx.args.ae_target_mode) if isinstance(ae_pseudo, dict) else "",
        ctx.ae_patch_mode,
        ae_diag["rows"],
        ae_diag["d_mean"],
        ae_diag["y_mean"],
        ae_diag["residual_mean"],
        ae_diag["residual_abs_mean"],
        ae_diag["residual_p95"],
        ae_evidence_flag,
    ]


def main():
    args = build_parser().parse_args()
    ctx = resolve_run_context(args)
    if check_validation_exit(ctx):
        return 0

    np.random.seed(ctx.exp["seed"])
    ctx = prepare_result_paths(ctx)
    if ctx.use_apbo_budget_policy:
        save_json(ctx.apbo_reference, os.path.join(ctx.dirs["apbo_diag"], "apbo_reference.json"))
    save_json(ctx.config_payload, ctx.paths["config"])

    grid = ctx.grid
    ref_fields = {
        "permx": np.loadtxt("rossmary.GRDECL").reshape(grid.total_cells, 1),
        "poro": np.loadtxt("rossmaryporo.GRDECL").reshape(grid.total_cells, 1),
        "permz": np.loadtxt("rossmaryz2.GRDECL").reshape(grid.total_cells, 1),
    }
    effective = np.loadtxt("effective.out").flatten()
    timestep = np.loadtxt("timestep.txt")
    history = timestep.size
    work_dir = os.getcwd()

    first_write = _needs_log_header(ctx.paths["log"])
    log_file = open(ctx.paths["log"], "w" if first_write else "a", newline="")
    log_writer = csv.writer(log_file)
    if first_write:
        log_writer.writerow(LOG_HEADER)

    opt = _init_online_optimizer(ctx)
    ae_bundle = try_load_ae(ctx.ae_cfg["ae_dir"]) if ((ctx.use_ae_constraint and not ctx.use_gmr_direction) or ctx.eval_ae_metric) else None
    ae_patch_coords_xyz = None
    if ae_bundle is not None and ctx.ae_cfg["ae_align_with_eval_patch_bank"]:
        px, py, pz = ae_bundle["patch"]
        ae_patch_coords_xyz = build_patch_bank_xyz(
            grid.nx, grid.ny, grid.nz, min(px, grid.nx), min(py, grid.ny), min(pz, grid.nz),
            n_patches=ctx.ae_cfg["ae_patches"], seed=ctx.ae_cfg["ae_patchbank_seed"],
        )

    best_score = np.inf
    best_rec = {"alpha": None, "c_radius": None, "J_raw": None, "J_unified": None, "RMSE": None, "phi_ae": None, "iteration": None, "selection_metric": "J_unified"}
    inv_alpha_sum = 0.0
    history_snapshots = {}
    prior_ae_refs = None
    prior_metrics = None
    geology_reference = None
    gmr_corrector = None
    apbo_prev_state = None
    apbo_observed_geo_max = float("nan")
    apbo_tell_count = 0
    prev_rmse, prev_phi = None, None
    gmr_prev_prod, gmr_best_prod = None, None
    prev_j_raw_lm, next_alpha_lm = None, None
    timing_log = []
    t_start = time.time()

    try:
        for inelly in range(ctx.n_assim + 1):
            t_iter0 = time.time()
            update_iteration_status(inelly, ctx.n_assim, ctx.paths["progress"])

            if inelly == 0:
                source_perm = np.reshape(np.fromiter(open("sgsim.out"), float), (grid.total_cells, ctx.source_N), "F")
                source_poro = np.reshape(np.fromiter(open("sgsimporo.out"), float), (grid.total_cells, ctx.source_N), "F")
                source_permz = np.reshape(np.fromiter(open("sgz.out"), float), (grid.total_cells, ctx.source_N), "F")
                mumyperm = source_perm[:, ctx.fidelity_indices].copy()
                mumyporo = source_poro[:, ctx.fidelity_indices].copy()
                mumypermz = source_permz[:, ctx.fidelity_indices].copy()

                np.savetxt(os.path.join(ctx.dirs["perm_txt"], "mumyperm_mean_iter00.txt"), np.mean(mumyperm, axis=1), fmt="%.6e")
                write_member_grdecl(work_dir, mumyperm, mumyporo, mumypermz, range(ctx.N))

                t_fwd0 = time.time()
                for i in range(ctx.N):
                    eclipse(i)
                t_forward = time.time() - t_fwd0

                _save_visuals(ctx, 0, mumyperm, mumyporo, mumypermz, effective)
                _, snap, rmse_prod, prod_info, sel_prod, phi_ae, geo_row, geology_reference = _evaluate_step_metrics(
                    ctx, work_dir, timestep, 0, "prior", mumyperm, mumyporo, mumypermz, effective, ae_bundle, ae_patch_coords_xyz, geology_reference
                )
                history_snapshots[0] = snap

                prior_metrics = {
                    "rmse_prior": rmse_prod,
                    "phi_ae_prior": float(phi_ae) if np.isfinite(phi_ae) else None,
                    "prod_nrmse_prior": float(sel_prod) if np.isfinite(sel_prod) else None,
                    "m_prod_prior": prod_info.get("m_prod"),
                    "geo_score_prior": geo_row.get("GeoScore", 0.0),
                }
                save_json(prior_metrics, ctx.paths["prior_metrics"])

                j_raw = compute_j_raw(rmse_prod, phi_ae)
                if np.isfinite(sel_prod):
                    gmr_prev_prod = gmr_best_prod = float(sel_prod)
                j_gmr, gmr_prod_score = compute_j_gmr(sel_prod, geo_row.get("GeoScore"), prior_metrics, args.gmr_geo_weight)
                j_unified = j_gmr if ctx.use_gmr_direction else compute_j_unified(rmse_prod, phi_ae, prior_metrics)

                prior_ae_refs = prepare_prior_ae_references(ctx, ae_bundle, ae_patch_coords_xyz, mumyperm, mumyporo, effective)
                if ctx.use_gmr_direction and gmr_corrector is None:
                    gmr_corrector = _make_gmr_corrector(ctx, mumyperm, mumyporo, effective, geology_reference)

                save_iteration_state_npz(0, ctx.dirs["iteration_states"], mumyperm, mumyporo, mumypermz, effective, prior_metrics, bool(args.save_full_ensemble))
                empty_step = {"B_before": 1.0, "LB_now": 1.0, "alpha_suggest": np.nan, "alpha_used": np.nan, "c_radius_used": None, "will_stop_after": False}
                log_writer.writerow(
                    _row_from_metrics(
                        ctx, 0, empty_step, rmse_prod, prod_info, phi_ae, j_raw, j_unified, j_gmr, geo_row, gmr_prod_score,
                        (np.nan, np.nan, np.nan, np.nan), (np.nan, np.nan, np.nan, np.nan, np.nan, np.nan),
                        np.nan, np.nan, np.nan, 0.0, 0.0, False, None, None, None,
                        {"rows": np.nan, "d_mean": np.nan, "y_mean": np.nan, "residual_mean": np.nan, "residual_abs_mean": np.nan, "residual_p95": np.nan},
                        False, None, None,
                    )
                )
                log_file.flush()

                p_norm0 = compute_prod_norm(sel_prod, prior_metrics, rmse=rmse_prod)
                g_score0 = geo_row.get("GeoScore", np.nan)
                if np.isfinite(g_score0) and g_score0 > 0:
                    apbo_observed_geo_max = float(g_score0)
                apbo_prev_state = make_apbo_state(p_norm0, g_score0, iteration=0)
                prev_rmse, prev_phi = rmse_prod, phi_ae
                if ctx.alpha_schedule == "lm_adaptive":
                    prev_j_raw_lm, next_alpha_lm = j_raw, 8.0
                timing_log.append({"iteration": 0, "t_forward_sim": t_forward, "t_esmda_update": 0.0, "t_apbo_forward_guard": 0.0, "t_gmr_correction": 0.0, "t_bo_update": 0.0, "t_iter_total": time.time() - t_iter0})
                continue

            step = _select_step_params(ctx, opt, inelly, inv_alpha_sum, apbo_tell_count, next_alpha_lm, prev_j_raw_lm)
            if step is None:
                break

            observation = gethistorydata(timestep, grid.n_observations, history, n_producers=grid.n_producers, n_timesteps=grid.n_timesteps)
            overallsim = getsimulated(timestep, ctx.N, history, n_obs=grid.n_observations, n_producers=grid.n_producers, n_timesteps=grid.n_timesteps)

            ae_pseudo, ae_summary, sigma_scale_now, target_q_now, ae_diag = build_step_ae_pseudo(
                ctx, ae_bundle, ae_patch_coords_xyz, prior_ae_refs, mumyperm, mumyporo, effective, inelly
            )

            t_es0 = time.time()
            proposed_perm, proposed_poro, proposed_permz = compute_esmda_candidate(
                ctx, ref_fields, mumyperm, mumyporo, mumypermz, observation, overallsim, history, effective,
                step["alpha_used"], step["c_radius_used"], ae_pseudo, inelly,
                status_callback=lambda cur, tot: update_iteration_status(inelly, ctx.n_assim, ctx.paths["progress"], cur, tot),
            )
            t_esmda = time.time() - t_es0

            apbo_guard_curr = None
            t_apbo_guard = 0.0
            if ctx.use_apbo_budget_policy and args.apbo_forward_guard_policy == "paired_equal_budget" and (ctx.n_assim - inelly + 1) > 1 and step["apbo_decision"] is not None:
                t_ag0 = time.time()
                proposed_perm, proposed_poro, proposed_permz, step["apbo_decision"], apbo_guard_curr = apply_apbo_equal_budget_guard(
                    ctx, work_dir, ref_fields, proposed_perm, proposed_poro, proposed_permz,
                    mumyperm, mumyporo, mumypermz, observation, overallsim, history, effective,
                    ae_pseudo, inelly, step["c_radius_used"], step["apbo_decision"],
                    step["B_before"], ctx.n_assim - inelly + 1,
                )
                step["alpha_used"] = step["apbo_decision"].alpha_used
                step["will_stop_after"] = step["apbo_decision"].will_stop_after
                t_apbo_guard = time.time() - t_ag0

            mumyperm, mumyporo, mumypermz = proposed_perm, proposed_poro, proposed_permz
            gmr_diag_curr, gmr_guard_curr = None, None
            t_gmr = 0.0
            if ctx.use_gmr_direction:
                t_g0 = time.time()
                if gmr_corrector is None:
                    gmr_corrector = _make_gmr_corrector(ctx, mumyperm, mumyporo, effective, geology_reference)
                prod_delta = compute_production_guard_delta(gmr_prev_prod, gmr_best_prod, prior_metrics.get("prod_nrmse_prior"), tolerance=args.gmr_prod_guard_tol)
                base_p, base_po, base_pz = mumyperm.copy(), mumyporo.copy(), mumypermz.copy()
                mumyperm, mumyporo, mumypermz, gmr_diag_curr = gmr_corrector.apply(
                    mumyperm, mumyporo, mumypermz_after_esmda=mumypermz, iteration=inelly, total_iterations=ctx.n_assim, prod_delta=prod_delta
                )
                mumyperm, mumyporo, mumypermz = np.abs(mumyperm), np.abs(mumyporo), np.abs(mumypermz)
                if args.gmr_forward_guard and isinstance(gmr_diag_curr, dict) and gmr_diag_curr.get("gmr_active"):
                    mumyperm, mumyporo, mumypermz, gmr_guard_curr = apply_gmr_forward_guard(
                        ctx, work_dir, base_p, base_po, base_pz, mumyperm, mumyporo, mumypermz,
                        inelly, gmr_diag_curr, geology_reference, lambda m: compute_geo_score(m, args),
                    )
                t_gmr = time.time() - t_g0

            np.savetxt(os.path.join(ctx.dirs["perm_txt"], f"mumyperm_mean_iter{inelly:02d}.txt"), np.mean(mumyperm, axis=1), fmt="%.6e")
            write_member_grdecl(work_dir, mumyperm, mumyporo, mumypermz, range(ctx.N))

            t_fwd0 = time.time()
            reused = set(gmr_guard_curr.get("gmr_paired_ucb_reuse_members", [])) if isinstance(gmr_guard_curr, dict) else set()
            for i in range(ctx.N):
                if i in reused and os.path.isfile(os.path.join(work_dir, f"MASTER{i + 1:05d}", "MASTER0.RSM")):
                    continue
                eclipse(i)
            t_forward = time.time() - t_fwd0

            _, snap, rmse_prod, prod_info, sel_prod, phi_ae, geo_row, _ = _evaluate_step_metrics(
                ctx, work_dir, timestep, inelly, "assimilation", mumyperm, mumyporo, mumypermz, effective, ae_bundle, ae_patch_coords_xyz, geology_reference
            )
            history_snapshots[inelly] = snap
            _save_visuals(ctx, inelly, mumyperm, mumyporo, mumypermz, effective)

            j_raw = compute_j_raw(rmse_prod, phi_ae)
            j_gmr, gmr_prod_score = compute_j_gmr(sel_prod, geo_row.get("GeoScore", np.nan), prior_metrics, args.gmr_geo_weight)
            if np.isfinite(sel_prod):
                gmr_prev_prod = float(sel_prod)
                gmr_best_prod = float(sel_prod) if gmr_best_prod is None else min(gmr_best_prod, float(sel_prod))
            j_unified = j_gmr if ctx.use_gmr_direction else compute_j_unified(rmse_prod, phi_ae, prior_metrics)

            epc_tuple = compute_epc_score(rmse_prod, float(phi_ae) if np.isfinite(phi_ae) else 0.0, step["alpha_used"], step["c_radius_used"])
            epc_star_tuple = compute_epc_star_score(
                rmse_prod, phi_ae, step["alpha_used"], step["c_radius_used"], prior_metrics, step["B_before"],
                use_ae_term=ctx.use_ae_constraint and not ctx.use_gmr_direction,
            )

            p_norm = compute_prod_norm(sel_prod, prior_metrics, rmse=rmse_prod)
            g_score = geo_row.get("GeoScore", np.nan)
            if np.isfinite(g_score) and g_score > 0:
                apbo_observed_geo_max = float(g_score) if not np.isfinite(apbo_observed_geo_max) else max(apbo_observed_geo_max, float(g_score))

            score_for_bo, score_for_bo_name = np.nan, "not_used"
            apbo_fb = None
            if ctx.use_online_bo and ctx.use_apbo_budget_policy:
                b_ratio = step["apbo_decision"].budget_ratio if step["apbo_decision"] else (1.0 / max(step["alpha_used"], 1e-12)) / max(step["B_before"], 1e-12)
                apbo_fb = compute_apbo_feedback(
                    ctx.apbo_objective_effective, prod_norm=p_norm, geo_score=g_score,
                    prev_state=apbo_prev_state, budget_ratio=b_ratio, reference=ctx.apbo_reference,
                    observed_geo_max=apbo_observed_geo_max,
                )
                score_for_bo = apbo_fb["score"]
                score_for_bo_name = f"APBO_{apbo_fb['objective']}" + ("_not_told" if step["will_stop_after"] else "")
            elif ctx.use_online_bo and not step["will_stop_after"] and opt is not None:
                score_for_bo, score_for_bo_name = compute_bo_feedback_score(
                    args.feedback, rmse_prod, step["alpha_used"], j_unified, epc_tuple[0], epc_star_tuple[0],
                    prior_metrics, j_gmr=j_gmr if ctx.use_gmr_direction else None,
                )

            inv_alpha_before = inv_alpha_sum
            inv_alpha_sum += 1.0 / max(1e-12, step["alpha_used"])
            gmr_active = ctx.use_gmr_direction and isinstance(gmr_diag_curr, dict) and bool(gmr_diag_curr.get("gmr_active"))
            ae_active = (ctx.use_ae_constraint and ae_pseudo is not None) or gmr_active
            j_bo_value = score_for_bo if ctx.use_apbo_budget_policy else compute_bo_feedback_score(
                "j_bo", rmse_prod, step["alpha_used"], j_unified, epc_tuple[0], epc_star_tuple[0],
                prior_metrics, j_gmr=j_gmr if ctx.use_gmr_direction else None,
            )[0]
            ae_evidence = bool(
                ae_active and np.isfinite(ae_diag["residual_abs_mean"]) and ae_diag["residual_abs_mean"] > 1e-12 and ae_diag["sigma"] > 0.0
            ) or gmr_active

            save_iteration_state_npz(inelly, ctx.dirs["iteration_states"], mumyperm, mumyporo, mumypermz, effective, {"Iteration": inelly, "RMSE": rmse_prod, "J_unified": j_unified, "GeoScore": g_score}, bool(args.save_full_ensemble))
            log_writer.writerow(
                _row_from_metrics(
                    ctx, inelly, step, rmse_prod, prod_info, phi_ae, j_raw, j_unified, j_gmr, geo_row, gmr_prod_score,
                    epc_tuple, epc_star_tuple, score_for_bo, score_for_bo_name, j_bo_value, inv_alpha_before, inv_alpha_sum,
                    ae_active, ae_pseudo, sigma_scale_now, target_q_now, ae_diag, ae_evidence, prev_phi, prev_rmse,
                )
            )
            log_file.flush()

            if ctx.use_apbo_budget_policy:
                apbo_prev_state = make_apbo_state(p_norm, g_score, feedback=apbo_fb, iteration=inelly)

            t_bo = 0.0
            if ctx.use_online_bo and not step["will_stop_after"] and opt is not None:
                t_bo0 = time.time()
                if ctx.use_apbo_budget_policy:
                    sel_eta = float(np.clip(step["apbo_decision"].eta_used, args.apbo_eta_min, args.apbo_eta_max)) if step["apbo_decision"] else 1.0
                    learning_obs = apbo_guard_curr.get("learning_observation") if isinstance(apbo_guard_curr, dict) else None
                    tell_eta, score_to_tell, _ = resolve_optimizer_feedback(
                        args.apbo_forward_guard_feedback, sel_eta, score_for_bo if np.isfinite(score_for_bo) else 0.0, learning_obs
                    )
                    tell_eta = float(np.clip(tell_eta, args.apbo_eta_min, args.apbo_eta_max))
                    opt.tell([tell_eta] if ctx.bo_alpha_only else [tell_eta, np.clip(step["c_radius_used"] or 5.0, 2.0, 6.0)], score_to_tell)
                    apbo_tell_count += 1
                else:
                    safe_alpha = float(np.clip(step["alpha_used"], 2.0, 8.0))
                    opt.tell([safe_alpha] if ctx.bo_alpha_only else [safe_alpha, np.clip(step["c_radius_used"] or 5.0, 2.0, 6.0)], score_for_bo)
                t_bo = time.time() - t_bo0

            select_score = j_unified if np.isfinite(j_unified) else j_raw
            if select_score < best_score:
                best_score = select_score
                best_rec = {
                    "alpha": step["alpha_used"],
                    "c_radius": step["c_radius_used"],
                    "J_raw": j_raw,
                    "J_unified": float(j_unified) if np.isfinite(j_unified) else None,
                    "J_GMR": float(j_gmr) if np.isfinite(j_gmr) else None,
                    "GeoScore": g_score if np.isfinite(g_score) else None,
                    "RMSE": rmse_prod,
                    "phi_ae": float(phi_ae) if np.isfinite(phi_ae) else None,
                    "iteration": inelly,
                    "selection_metric": "J_unified" if np.isfinite(j_unified) else "J_raw",
                }
                save_json(best_rec, ctx.paths["best_params"])

            prev_rmse, prev_phi = rmse_prod, phi_ae
            if ctx.alpha_schedule == "lm_adaptive":
                if prev_j_raw_lm and prev_j_raw_lm > 0:
                    red = (prev_j_raw_lm - j_raw) / prev_j_raw_lm
                    next_alpha_lm = max(2.0, step["alpha_used"] * (1.0 - 0.5 * red)) if red > 0 else min(ALPHA_MAX, step["alpha_used"] * 2.0)
                else:
                    next_alpha_lm = 8.0
                prev_j_raw_lm = j_raw

            timing_log.append({
                "iteration": inelly,
                "t_forward_sim": t_forward,
                "t_esmda_update": t_esmda,
                "t_apbo_forward_guard": t_apbo_guard,
                "t_gmr_correction": t_gmr,
                "t_bo_update": t_bo,
                "t_iter_total": time.time() - t_iter0,
            })

            if ctx.use_online_bo and (step["will_stop_after"] or inv_alpha_sum >= 1.0 - 1e-12) and inelly >= MIN_ASSIM_ITERS:
                break
    finally:
        log_file.close()
        if best_rec["selection_metric"] is not None:
            save_json(best_rec, ctx.paths["best_params"])

    if history_snapshots:
        save_history_match_snapshots_npz(history_snapshots, os.path.join(ctx.result_root, f"Figure3_history_match_data_{ctx.run_tag}.npz"))
        selected_iters = sorted(history_snapshots)
        plot_history_match_figure_advanced(
            history_snapshots,
            selected_iters=selected_iters,
            iter_labels=["Prior", "Iter 1", "Iter 2", "Iter 3", "Final posterior"] if len(selected_iters) == 5 else None,
            out_png=os.path.join(ctx.result_root, f"Figure3_history_match_advanced_{ctx.run_tag}.png"),
            out_pdf=os.path.join(ctx.result_root, f"Figure3_history_match_advanced_{ctx.run_tag}.pdf"),
            mode="hybrid",
            max_lines=20,
        )

    runtime_sec = time.time() - t_start
    save_run_summary(ctx.paths["log"], ctx.paths["best_params"], ctx.paths["config"], ctx.paths["summary"], runtime_sec, ctx.paths["prior_metrics"], timing_log=timing_log)
    print(f"[Done] Runtime = {runtime_sec:.2f} s | Results: {ctx.result_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
