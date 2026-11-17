import argparse
from datetime import datetime
import json
import os
from types import SimpleNamespace

from .apbo_budget_policy import has_fixed_geo_reference, load_reference_file
from .apbo_fidelity_manifest import load_fidelity_manifest, select_fidelity_members
from .apbo_rollout_schedule import load_schedule_file
from .gmr_terminal_sequence_runtime import load_terminal_sequence_file
from .grid_config import load_grid_config
from .production_forward_guard import parse_blends
from .result_path_policy import build_result_component


LAMBDA_AE = 0.1
LAMBDA_PHI_UNIFIED = 1.0
BO_BUDGET_PENALTY = 0.03
ALPHA_MAX = 8.0
MIN_ASSIM_ITERS = 4
FULLMETHOD_FIXED_C_RADIUS = 5.0

EPC_STAR_OMEGA_AE = 1.0
EPC_STAR_OMEGA_BUDGET = 0.05
EPC_STAR_OMEGA_SPATIAL = 0.10
EPC_STAR_C_SAFE = 3.5
EPC_STAR_EPS = 1e-8

FROZEN_APBO_C010_HASH = "a88cbcdba36ab7ac94fa486883d6cc97e0255da19bb0bbe8b9c2dfc4b3f1adcb"
FROZEN_EQUAL_ALPHA_HASH = "5073e61eafb5e090b20e50cfa38e1505dbb9a00d659ef6cd6659298674226ad7"

AE_PRESETS = {
    "base": {
        "AE_START_ITER": 2,
        "AE_SIGMA_SCALE": 2.6,
        "AE_SIGMA_MIN_SCALE": 1.0,
        "AE_HARD_RATIO": 0.20,
        "AE_TARGET_Q_HI": 0.35,
        "AE_TARGET_Q_LO": 0.18,
        "AE_TIME_WINDOW": 2,
        "AE_LOCALIZATION_RADIUS": 4.5,
        "AE_TIME_SIGMA_MULTIPLIERS": [1.10, 1.00],
        "description": "Original soft and late AE setting",
    },
    "m1_early": {
        "AE_START_ITER": 1,
        "AE_SIGMA_SCALE": 2.0,
        "AE_SIGMA_MIN_SCALE": 0.8,
        "AE_HARD_RATIO": 0.20,
        "AE_TARGET_Q_HI": 0.35,
        "AE_TARGET_Q_LO": 0.18,
        "AE_TIME_WINDOW": 2,
        "AE_LOCALIZATION_RADIUS": 4.5,
        "AE_TIME_SIGMA_MULTIPLIERS": [1.10, 1.00],
        "description": "Early AE intervention with mild strengthening",
    },
    "m2_medium": {
        "AE_START_ITER": 1,
        "AE_SIGMA_SCALE": 1.5,
        "AE_SIGMA_MIN_SCALE": 0.8,
        "AE_HARD_RATIO": 0.20,
        "AE_TARGET_Q_HI": 0.35,
        "AE_TARGET_Q_LO": 0.18,
        "AE_TIME_WINDOW": 2,
        "AE_LOCALIZATION_RADIUS": 4.5,
        "AE_TIME_SIGMA_MULTIPLIERS": [1.10, 1.00],
        "description": "Early AE intervention with medium pseudo-observation strength",
    },
    "m3_medium_hard": {
        "AE_START_ITER": 1,
        "AE_SIGMA_SCALE": 1.5,
        "AE_SIGMA_MIN_SCALE": 0.8,
        "AE_HARD_RATIO": 0.30,
        "AE_TARGET_Q_HI": 0.35,
        "AE_TARGET_Q_LO": 0.18,
        "AE_TIME_WINDOW": 2,
        "AE_LOCALIZATION_RADIUS": 4.8,
        "AE_TIME_SIGMA_MULTIPLIERS": [1.10, 1.00],
        "description": "Medium AE strength with more high-error patches",
    },
    "m4_strong": {
        "AE_START_ITER": 1,
        "AE_SIGMA_SCALE": 1.2,
        "AE_SIGMA_MIN_SCALE": 0.6,
        "AE_HARD_RATIO": 0.30,
        "AE_TARGET_Q_HI": 0.25,
        "AE_TARGET_Q_LO": 0.10,
        "AE_TIME_WINDOW": 2,
        "AE_LOCALIZATION_RADIUS": 5.0,
        "AE_TIME_SIGMA_MULTIPLIERS": [1.15, 1.00],
        "description": "Strong AE setting with stricter anomaly target",
    },
    "m5_very_strong": {
        "AE_START_ITER": 1,
        "AE_SIGMA_SCALE": 1.0,
        "AE_SIGMA_MIN_SCALE": 0.5,
        "AE_HARD_RATIO": 0.40,
        "AE_TARGET_Q_HI": 0.20,
        "AE_TARGET_Q_LO": 0.08,
        "AE_TIME_WINDOW": 3,
        "AE_LOCALIZATION_RADIUS": 5.2,
        "AE_TIME_SIGMA_MULTIPLIERS": [1.20, 1.05, 1.00],
        "description": "Very strong AE diagnostic setting",
    },
    "custom": {
        "description": "Custom AE calibration setting from command-line overrides",
    },
}


def make_geometric_alpha_schedule(n_assim=4, ratio=0.7):
    weights = [ratio ** (-k) for k in range(n_assim)]
    a1 = sum(weights)
    return [float(a1 * (ratio ** k)) for k in range(n_assim)]


EXPERIMENTS = {
    "stdNoLocESMDA": {
        "run_tag": "stdNoLocESMDA",
        "use_bayesopt": False,
        "use_ae_constraint": False,
        "eval_ae_metric": True,
        "use_localization": False,
        "fixed_alpha": 4.0,
        "fixed_c_radius": None,
        "alpha_schedule": None,
        "seed": 2025,
        "bo_alpha_only_default": True,
        "method_role": "legacy_nonlocalized_baseline",
        "description": "Standard ES-MDA without covariance localization",
    },
    "stdESMDA": {
        "run_tag": "stdESMDA",
        "use_bayesopt": False,
        "use_ae_constraint": False,
        "eval_ae_metric": True,
        "use_localization": True,
        "fixed_alpha": 4.0,
        "fixed_c_radius": 5.0,
        "alpha_schedule": None,
        "seed": 2025,
        "bo_alpha_only_default": True,
        "method_role": "controlled_fixed_parameter_baseline",
        "description": "Controlled Standard ES-MDA with fixed alpha and localization radius",
    },
    "localizedESMDA": {
        "run_tag": "localizedESMDA",
        "use_bayesopt": False,
        "use_ae_constraint": False,
        "eval_ae_metric": True,
        "use_localization": True,
        "fixed_alpha": 4.0,
        "fixed_c_radius": 5.0,
        "alpha_schedule": None,
        "seed": 2025,
        "bo_alpha_only_default": True,
        "method_role": "localized_fixed_parameter_baseline",
        "description": "Fixed-parameter ES-MDA with covariance localization",
    },
    "geoESMDA": {
        "run_tag": "geoESMDA",
        "use_bayesopt": False,
        "use_ae_constraint": False,
        "eval_ae_metric": True,
        "use_localization": True,
        "fixed_alpha": None,
        "fixed_c_radius": 5.0,
        "alpha_schedule": make_geometric_alpha_schedule(n_assim=4, ratio=0.7),
        "seed": 2025,
        "bo_alpha_only_default": True,
        "method_role": "geometric_alpha_baseline",
        "description": "Geometric-decay ES-MDA with fixed localization radius",
    },
    "apboOnly": {
        "run_tag": "apboOnly",
        "use_bayesopt": True,
        "use_ae_constraint": False,
        "eval_ae_metric": True,
        "use_localization": True,
        "fixed_alpha": None,
        "fixed_c_radius": 5.0,
        "alpha_schedule": None,
        "seed": 2025,
        "bo_alpha_only_default": True,
        "method_role": "apbo_alpha_only_without_ae",
        "description": "APBO-only: adaptive alpha with fixed localization",
    },
    "apboAlphaC": {
        "run_tag": "apboAlphaC",
        "use_bayesopt": True,
        "use_ae_constraint": False,
        "eval_ae_metric": True,
        "use_localization": True,
        "fixed_alpha": None,
        "fixed_c_radius": None,
        "alpha_schedule": None,
        "seed": 2025,
        "bo_alpha_only_default": False,
        "method_role": "legacy_apbo_alpha_and_c_without_ae",
        "description": "APBO adaptive alpha and c_radius without AE constraint",
    },
    "aeOnly": {
        "run_tag": "aeOnly",
        "use_bayesopt": False,
        "use_ae_constraint": True,
        "eval_ae_metric": True,
        "use_localization": True,
        "fixed_alpha": 4.0,
        "fixed_c_radius": 5.0,
        "alpha_schedule": None,
        "seed": 2025,
        "bo_alpha_only_default": True,
        "method_role": "ae_constraint_only_fixed_parameters",
        "description": "AE-only: fixed alpha and localization with AE constraint",
    },
    "fullMethod": {
        "run_tag": "fullMethod",
        "use_bayesopt": True,
        "use_ae_constraint": True,
        "eval_ae_metric": True,
        "use_localization": True,
        "fixed_alpha": None,
        "fixed_c_radius": 5.0,
        "alpha_schedule": None,
        "seed": 2025,
        "bo_alpha_only_default": True,
        "method_role": "proposed_apbo_alpha_only_plus_ae",
        "description": "Proposed APBO+AE/GMR with adaptive alpha and fixed localization",
    },
    "fullAlphaC": {
        "run_tag": "fullAlphaC",
        "use_bayesopt": True,
        "use_ae_constraint": True,
        "eval_ae_metric": True,
        "use_localization": True,
        "fixed_alpha": None,
        "fixed_c_radius": None,
        "alpha_schedule": None,
        "seed": 2025,
        "bo_alpha_only_default": False,
        "method_role": "extended_apbo_alpha_and_c_plus_ae",
        "description": "Extended APBO+AE with adaptive alpha and c_radius",
    },
    "le2016": {
        "run_tag": "le2016",
        "use_bayesopt": False,
        "use_ae_constraint": False,
        "eval_ae_metric": True,
        "use_localization": True,
        "fixed_alpha": None,
        "fixed_c_radius": 5.0,
        "alpha_schedule": [8.0, 4.0, 2.0, 2.0],
        "seed": 2025,
        "bo_alpha_only_default": True,
        "method_role": "literature_baseline_le2016",
        "description": "Le et al. (2016) aggressive geometric alpha schedule",
    },
    "rafiee2017": {
        "run_tag": "rafiee2017",
        "use_bayesopt": False,
        "use_ae_constraint": False,
        "eval_ae_metric": True,
        "use_localization": True,
        "fixed_alpha": None,
        "fixed_c_radius": 5.0,
        "alpha_schedule": "lm_adaptive",
        "seed": 2025,
        "bo_alpha_only_default": True,
        "method_role": "literature_baseline_rafiee2017",
        "description": "Rafiee & Reynolds (2017) LM-style adaptive alpha ES-MDA",
    },
}


def build_parser():
    parser = argparse.ArgumentParser(description="HBO-GMR reservoir history matching runner.")
    parser.add_argument("--exp", default="fullMethod", choices=list(EXPERIMENTS))
    parser.add_argument("--grid_config", default=os.path.join("gmr_refactor", "config", "grid_punq.json"))
    parser.add_argument(
        "--feedback",
        default="j_bo",
        choices=["j_bo", "original", "epc_star", "epc_legacy", "rmse_budget"],
    )
    parser.add_argument("--bo_alpha_only", default="auto", choices=["auto", "yes", "no"])
    parser.add_argument("--out_root", default=os.path.join("gmr_refactor", "results"))
    parser.add_argument("--run_id", default="")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--existing_run_policy", default="new_subdir", choices=["new_subdir", "error", "append"])
    parser.add_argument("--save_full_ensemble", type=int, default=1)
    parser.add_argument("--save_iteration_visuals", type=int, default=1)
    parser.add_argument("--compact_result_path", type=int, default=0)
    parser.add_argument("--all_localized", type=int, default=0)
    parser.add_argument("--logic", default="paper", choices=["paper", "legacy"])
    parser.add_argument("--apbo_policy", default="legacy_alpha_bo", choices=["legacy_alpha_bo", "budget_ratio_hvi"])
    parser.add_argument(
        "--apbo_objective",
        default="auto",
        choices=["auto", "prod_efficiency", "hv_efficiency", "quality_efficiency", "geo_constrained_prod_efficiency"],
    )
    parser.add_argument("--apbo_warm_start", default="none", choices=["none", "legacy_manual"])
    parser.add_argument("--apbo_reference_file", default="")
    parser.add_argument("--apbo_eta_min", type=float, default=0.50)
    parser.add_argument("--apbo_eta_max", type=float, default=1.50)
    parser.add_argument("--apbo_forward_guard_policy", default="off", choices=["off", "paired_equal_budget"])
    parser.add_argument("--apbo_forward_guard_pilot_members", type=int, default=20)
    parser.add_argument(
        "--apbo_forward_guard_feedback",
        default="selected_full",
        choices=["selected_full", "paired_dominance"],
    )
    parser.add_argument("--apbo_schedule_mode", default="online_eta", choices=["online_eta", "terminal_simplex"])
    parser.add_argument("--alpha_schedule_file", default="")
    parser.add_argument("--validate_terminal_schedule_only", type=int, default=0)
    parser.add_argument("--ensemble_fidelity_manifest", default="")
    parser.add_argument("--ensemble_fidelity_level", default="", choices=["", "L1", "L2", "L3"])
    parser.add_argument("--validate_fidelity_only", type=int, default=0)

    parser.add_argument(
        "--ae_preset",
        default="base",
        choices=list(AE_PRESETS),
    )
    parser.add_argument("--ae_start_iter", type=int, default=None)
    parser.add_argument("--ae_sigma_scale", type=float, default=None)
    parser.add_argument("--ae_sigma_min_scale", type=float, default=None)
    parser.add_argument("--ae_hard_ratio", type=float, default=None)
    parser.add_argument("--ae_target_q_hi", type=float, default=None)
    parser.add_argument("--ae_target_q_lo", type=float, default=None)
    parser.add_argument("--ae_time_window", type=int, default=None)
    parser.add_argument("--ae_localization_radius", type=float, default=None)
    parser.add_argument("--ae_time_sigma_multipliers", default="")
    parser.add_argument("--ae_patches", type=int, default=None)
    parser.add_argument("--ae_eval_topk", type=int, default=8)
    parser.add_argument("--ae_target_mode", default="adaptive", choices=["adaptive", "prior_fixed"])
    parser.add_argument(
        "--ae_constraint_form",
        default="patch_error",
        choices=["patch_error", "distribution", "phi_aligned", "recon_project", "prior_manifold", "gmr_direction"],
    )
    parser.add_argument("--ae_sigma_mode", default="adaptive", choices=["adaptive", "prior_fixed", "capped"])
    parser.add_argument("--ae_sigma_cap_factor", type=float, default=20.0)
    parser.add_argument("--ae_update_weight", type=float, default=1.0)
    parser.add_argument("--ae_recon_max_obs", type=int, default=512)
    parser.add_argument("--ae_recon_blend", type=float, default=1.0)
    parser.add_argument("--ae_recon_active_fraction_min", type=float, default=0.75)
    parser.add_argument("--ae_prior_envelope_q", type=float, default=0.90)
    parser.add_argument("--ae_prior_scale_floor", type=float, default=1e-4)
    parser.add_argument("--ae_prior_manifold_mode", default="patch", choices=["patch", "stats"])
    parser.add_argument("--ae_prior_exceedance_cap", type=float, default=5.0)
    parser.add_argument("--ae_prior_target_q", type=float, default=0.50)
    parser.add_argument("--ae_reference_q", type=float, default=0.02)
    parser.add_argument("--ae_reference_scale", type=float, default=0.95)
    parser.add_argument(
        "--production_metric",
        default="legacy_temporal",
        choices=["legacy_temporal", "category_balanced", "m_prod"],
    )

    parser.add_argument("--gmr_vae_ckpt", default=os.path.join("gmr_refactor", "checkpoints", "gmr_vae.pt"))
    parser.add_argument("--gmr_flow_ckpt", default=os.path.join("gmr_refactor", "checkpoints", "gmr_flow.pt"))
    parser.add_argument("--gmr_dataset_meta", default=os.path.join("gmr_refactor", "data", "prior_patch_dataset_meta.json"))
    parser.add_argument("--gmr_device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--gmr_batch_size", type=int, default=256)
    parser.add_argument("--gmr_max_realizations", type=int, default=0)
    parser.add_argument("--gmr_activate_from_iter", type=int, default=1)
    parser.add_argument("--gmr_lambda_base", type=float, default=0.20)
    parser.add_argument("--gmr_lambda_min", type=float, default=0.0)
    parser.add_argument("--gmr_lambda_max", type=float, default=0.30)
    parser.add_argument("--gmr_score_step_base", type=float, default=0.05)
    parser.add_argument("--gmr_score_clip", type=float, default=3.0)
    parser.add_argument("--gmr_strength_mode", default="log_squash", choices=["log_squash", "linear"])
    parser.add_argument("--gmr_violation_ref", type=float, default=5.0)
    parser.add_argument("--gmr_stage_mode", default="increasing", choices=["increasing", "late_taper"])
    parser.add_argument("--gmr_stage_floor", type=float, default=0.0)
    parser.add_argument("--gmr_prod_guard_beta", type=float, default=4.0)
    parser.add_argument("--gmr_prod_guard_tol", type=float, default=0.05)
    parser.add_argument("--gmr_force_lambda", type=float, default=-1.0)
    parser.add_argument("--gmr_global_guard", type=int, default=1)
    parser.add_argument("--gmr_global_guard_tol", type=float, default=0.0)
    parser.add_argument("--gmr_global_guard_policy", default="min_score", choices=["min_score", "minimal_gain"])
    parser.add_argument("--gmr_global_guard_min_geo_gain", type=float, default=0.03)
    parser.add_argument("--gmr_forward_guard", type=int, default=0)
    parser.add_argument(
        "--gmr_forward_guard_policy",
        default="legacy_group",
        choices=["legacy_group", "mprod_pareto", "paired_ucb_search", "terminal_sequence"],
    )
    parser.add_argument("--gmr_forward_guard_pilot_members", type=int, default=20)
    parser.add_argument("--gmr_forward_guard_blends", default="1,0.5,0.25,0.125,0")
    parser.add_argument("--gmr_forward_guard_overall_tol", type=float, default=0.0)
    parser.add_argument("--gmr_forward_guard_wbhp_tol", type=float, default=0.0)
    parser.add_argument("--gmr_forward_guard_other_tol", type=float, default=0.05)
    parser.add_argument("--gmr_forward_guard_min_geo_gain", type=float, default=0.01)
    parser.add_argument("--gmr_paired_ucb_max_evaluations", type=int, default=5)
    parser.add_argument("--gmr_paired_ucb_initial_evaluations", type=int, default=2)
    parser.add_argument("--gmr_paired_ucb_pool_size", type=int, default=256)
    parser.add_argument("--gmr_paired_ucb_bootstrap_draws", type=int, default=1000)
    parser.add_argument("--gmr_paired_ucb_confidence", type=float, default=0.95)
    parser.add_argument("--gmr_paired_ucb_seed", type=int, default=2026)
    parser.add_argument("--gmr_paired_ucb_cache", type=int, default=1)
    parser.add_argument("--validate_stage_d2_wiring_only", type=int, default=0)
    parser.add_argument("--gmr_terminal_sequence_file", default="")
    parser.add_argument("--validate_stage_e2_wiring_only", type=int, default=0)
    parser.add_argument("--gmr_eval_legacy_phi", type=int, default=0)
    parser.add_argument("--gmr_geo_weight", type=float, default=0.10)
    parser.add_argument("--gmr_geo_w_hist_logperm", type=float, default=1.0)
    parser.add_argument("--gmr_geo_w_hist_poro", type=float, default=1.0)
    parser.add_argument("--gmr_geo_w_var_logperm", type=float, default=1.0)
    parser.add_argument("--gmr_geo_w_var_poro", type=float, default=1.0)
    parser.add_argument("--gmr_geo_w_conn", type=float, default=1.0)
    parser.add_argument("--calib_note", default="")
    return parser


def _parse_float_list(s, default):
    if not s or not s.strip():
        return list(default)
    vals = [float(item.strip()) for item in s.split(",") if item.strip()]
    return vals if vals else list(default)


def _resolve_ae_settings(args):
    cfg = {
        "ae_dir": "ae_ckpt",
        "ae_patches": 128,
        "ae_agg": "median",
        "ae_device": "cpu",
        "ae_align_with_eval_patch_bank": True,
        "ae_patchbank_seed": 2025,
        "ae_start_iter": 2,
        "ae_sigma_scale": 2.6,
        "ae_sigma_min_scale": 1.0,
        "ae_hard_ratio": 0.20,
        "ae_target_q_hi": 0.35,
        "ae_target_q_lo": 0.18,
        "ae_time_window": 2,
        "ae_localization_radius": 4.5,
        "ae_time_sigma_multipliers": [1.10, 1.00],
    }
    if args.ae_preset != "custom":
        p = AE_PRESETS[args.ae_preset]
        cfg["ae_start_iter"] = p["AE_START_ITER"]
        cfg["ae_sigma_scale"] = p["AE_SIGMA_SCALE"]
        cfg["ae_sigma_min_scale"] = p["AE_SIGMA_MIN_SCALE"]
        cfg["ae_hard_ratio"] = p["AE_HARD_RATIO"]
        cfg["ae_target_q_hi"] = p["AE_TARGET_Q_HI"]
        cfg["ae_target_q_lo"] = p["AE_TARGET_Q_LO"]
        cfg["ae_time_window"] = p["AE_TIME_WINDOW"]
        cfg["ae_localization_radius"] = p["AE_LOCALIZATION_RADIUS"]
        cfg["ae_time_sigma_multipliers"] = list(p["AE_TIME_SIGMA_MULTIPLIERS"])

    overrides = {
        "ae_start_iter": args.ae_start_iter,
        "ae_sigma_scale": args.ae_sigma_scale,
        "ae_sigma_min_scale": args.ae_sigma_min_scale,
        "ae_hard_ratio": args.ae_hard_ratio,
        "ae_target_q_hi": args.ae_target_q_hi,
        "ae_target_q_lo": args.ae_target_q_lo,
        "ae_time_window": args.ae_time_window,
        "ae_localization_radius": args.ae_localization_radius,
        "ae_patches": args.ae_patches,
    }
    for key, val in overrides.items():
        if val is not None:
            cfg[key] = val
    if args.ae_time_sigma_multipliers.strip():
        cfg["ae_time_sigma_multipliers"] = _parse_float_list(
            args.ae_time_sigma_multipliers, cfg["ae_time_sigma_multipliers"]
        )
    return cfg


def _result_dir_has_content(path):
    if not os.path.isdir(path):
        return False
    with os.scandir(path) as entries:
        return any(True for _ in entries)


def _resolve_existing_result_root(path, policy):
    if not _result_dir_has_content(path):
        return path, False, None
    if policy == "append":
        return path, True, "append"
    if policy == "error":
        raise FileExistsError(f"Result directory already exists and is non-empty: {path}")
    for idx in range(2, 1000):
        candidate = f"{path}__rerun{idx:02d}"
        if not _result_dir_has_content(candidate):
            return candidate, True, "new_subdir"
    raise RuntimeError(f"Unable to allocate a new result directory under: {path}")


def get_paper_method_metadata(
    run_tag,
    use_localization,
    use_bayesopt,
    use_ae_constraint,
    bo_alpha_only,
    fixed_alpha,
    fixed_c_radius,
    alpha_schedule,
    apbo_schedule_mode="online_eta",
):
    paper_names = {
        "stdNoLocESMDA": "stdESMDA_no_localization",
        "stdESMDA": "stdESMDA",
        "localizedESMDA": "stdESMDA_localized",
        "geoESMDA": "geometricESMDA",
        "apboOnly": "APBO_only",
        "apboAlphaC": "APBO_alpha_c_sensitivity",
        "aeOnly": "AE_only",
        "fullMethod": "APBO_AE",
        "fullAlphaC": "APBO_AE_alpha_c_sensitivity",
    }
    if use_bayesopt and apbo_schedule_mode == "terminal_simplex":
        alpha_strategy = "terminal constrained BO on the ES-MDA budget simplex"
    elif use_bayesopt:
        alpha_strategy = "BO under ES-MDA budget constraint"
    elif alpha_schedule is not None:
        alpha_strategy = "geometric schedule"
    else:
        alpha_strategy = f"fixed_{float(fixed_alpha):g}" if fixed_alpha is not None else "fixed_or_external"

    if not use_localization:
        c_strategy = "not_used"
    elif use_bayesopt and not bo_alpha_only:
        c_strategy = "BO"
    elif fixed_c_radius is not None:
        c_strategy = f"fixed_{float(fixed_c_radius):g}"
    else:
        c_strategy = f"fixed_{FULLMETHOD_FIXED_C_RADIUS:g}"

    return {
        "method": paper_names.get(run_tag, run_tag),
        "legacy_run_tag": run_tag,
        "localization": use_localization,
        "apbo": use_bayesopt,
        "ae_constraint": use_ae_constraint,
        "alpha_strategy": alpha_strategy,
        "c_strategy": c_strategy,
        "budget_compliance": True,
        "paper_mode": "minimal" if bo_alpha_only else "enhanced_or_baseline",
        "apbo_schedule_mode": apbo_schedule_mode,
    }


def _validate_protocol_wiring(ctx):
    args = ctx.args
    if ctx.fidelity_record is not None:
        stage_c2 = ctx.use_terminal_schedule and args.exp == "apboOnly"
        stage_d = (ctx.use_paired_ucb_gmr or ctx.use_terminal_gmr_seq) and args.exp in {"aeOnly", "fullMethod"}
        if not (stage_c2 or stage_d):
            raise ValueError("nested ensemble fidelity requires apboOnly terminal rollout or GMR rollout")

    if ctx.use_terminal_gmr_seq:
        if not ctx.use_gmr_direction or args.exp != "fullMethod":
            raise ValueError("terminal_sequence requires fullMethod with gmr_direction")
        if args.production_metric != "m_prod" or not args.gmr_forward_guard:
            raise ValueError("terminal_sequence requires --production_metric m_prod and --gmr_forward_guard 1")
        if args.gmr_global_guard:
            raise ValueError("terminal_sequence requires --gmr_global_guard 0")
        if ctx.fidelity_record is None or not ctx.use_terminal_schedule:
            raise ValueError("terminal_sequence requires frozen fidelity manifest and terminal schedule")
        sched_hash = ctx.terminal_schedule.get("schedule_hash", "")
        if sched_hash not in {FROZEN_APBO_C010_HASH, FROZEN_EQUAL_ALPHA_HASH}:
            raise ValueError("terminal_sequence schedule hash is not a frozen APBO or equal-alpha schedule")

    if ctx.use_paired_ucb_gmr:
        if not ctx.use_gmr_direction or args.exp not in {"aeOnly", "fullMethod"}:
            raise ValueError("paired_ucb_search requires aeOnly or fullMethod with gmr_direction")
        if args.production_metric != "m_prod" or not args.gmr_forward_guard or args.gmr_global_guard:
            raise ValueError("paired_ucb_search requires m_prod, --gmr_forward_guard 1, and --gmr_global_guard 0")
        if ctx.fidelity_record is None:
            raise ValueError("paired_ucb_search requires the frozen nested fidelity manifest")
        if args.gmr_forward_guard_pilot_members != 20 or args.gmr_paired_ucb_max_evaluations != 5:
            raise ValueError("paired_ucb_search protocol fixes 20 pilot members and 5 evaluations")
        if args.exp == "fullMethod":
            if not ctx.use_terminal_schedule or ctx.terminal_schedule.get("schedule_hash") != FROZEN_APBO_C010_HASH:
                raise ValueError("paired_ucb_search fullMethod requires frozen APBO c010 schedule")

    if (
        args.use_apbo_budget_policy
        and args.apbo_forward_guard_feedback == "paired_dominance"
        and args.apbo_forward_guard_policy != "paired_equal_budget"
    ):
        raise ValueError("paired_dominance feedback requires --apbo_forward_guard_policy paired_equal_budget")


def resolve_run_context(args):
    grid = load_grid_config(args.grid_config)
    source_N = grid.N
    fidelity_record = None
    fidelity_indices = list(range(source_N))
    fidelity_level = ""
    if args.ensemble_fidelity_manifest.strip():
        if not args.ensemble_fidelity_level.strip():
            raise ValueError("--ensemble_fidelity_level is required with --ensemble_fidelity_manifest")
        fidelity_record = load_fidelity_manifest(args.ensemble_fidelity_manifest)
        if int(fidelity_record["source_ensemble_size"]) != source_N:
            raise ValueError("fidelity manifest source size does not match grid configuration")
        fidelity_level = args.ensemble_fidelity_level.upper()
        fidelity_indices = select_fidelity_members(fidelity_record, fidelity_level)
    elif args.ensemble_fidelity_level.strip():
        raise ValueError("--ensemble_fidelity_manifest is required with --ensemble_fidelity_level")

    n_assim = grid.n_assim
    exp = dict(EXPERIMENTS[args.exp])
    if args.seed is not None:
        exp["seed"] = args.seed
        exp["description"] = f"{exp['description']} [seed override: {args.seed}]"

    use_localization = exp["use_localization"]
    fixed_c_radius = exp["fixed_c_radius"]
    if args.all_localized == 1 and exp["run_tag"] in ("stdESMDA", "stdNoLocESMDA"):
        use_localization = True
        fixed_c_radius = 5.0
        exp["description"] = "Standard ES-MDA with forced fixed localization (--all_localized=1)"

    use_bayesopt = exp["use_bayesopt"]
    use_ae_constraint = exp["use_ae_constraint"]
    alpha_schedule = exp["alpha_schedule"]
    use_terminal_schedule = args.apbo_schedule_mode == "terminal_simplex"
    terminal_schedule = None
    if use_terminal_schedule:
        if not use_bayesopt or args.exp not in {"apboOnly", "fullMethod"}:
            raise ValueError("terminal_simplex is only valid for apboOnly or fullMethod")
        if not args.alpha_schedule_file.strip():
            raise ValueError("--alpha_schedule_file is required with terminal_simplex")
        terminal_schedule = load_schedule_file(args.alpha_schedule_file, expected_n_assim=n_assim)
        alpha_schedule = list(terminal_schedule["alpha_schedule"])

    use_online_bo = use_bayesopt and not use_terminal_schedule
    use_paired_ucb_gmr = args.gmr_forward_guard_policy == "paired_ucb_search"
    use_terminal_gmr_seq = args.gmr_forward_guard_policy == "terminal_sequence"
    terminal_gmr_seq = None
    if use_terminal_gmr_seq:
        if not args.gmr_terminal_sequence_file.strip():
            raise ValueError("terminal_sequence requires --gmr_terminal_sequence_file")
        terminal_gmr_seq = load_terminal_sequence_file(args.gmr_terminal_sequence_file, expected_n_steps=n_assim)

    use_gmr_direction = use_ae_constraint and args.ae_constraint_form == "gmr_direction"
    eval_ae_metric = False if (use_gmr_direction and args.gmr_eval_legacy_phi == 0) else exp["eval_ae_metric"]
    gmr_internal_global_guard = bool(args.gmr_global_guard) and not use_paired_ucb_gmr and not use_terminal_gmr_seq
    use_apbo_budget_policy = use_online_bo and args.apbo_policy == "budget_ratio_hvi"
    args.use_apbo_budget_policy = use_apbo_budget_policy

    apbo_obj_eff = (
        ("geo_constrained_prod_efficiency" if use_gmr_direction else "prod_efficiency")
        if args.apbo_objective == "auto"
        else args.apbo_objective
    )

    if args.bo_alpha_only == "yes":
        bo_alpha_only = use_bayesopt and use_localization
    elif args.bo_alpha_only == "no":
        bo_alpha_only = False
    elif args.logic == "paper":
        bo_alpha_only = use_bayesopt and exp.get("bo_alpha_only_default", True)
    else:
        bo_alpha_only = use_bayesopt and use_ae_constraint

    ctx = SimpleNamespace(
        args=args,
        grid=grid,
        exp=exp,
        run_tag=exp["run_tag"],
        run_id=args.run_id.strip() or datetime.now().strftime("%Y%m%d_%H%M%S"),
        N=len(fidelity_indices),
        source_N=source_N,
        n_assim=n_assim,
        fidelity_record=fidelity_record,
        fidelity_indices=fidelity_indices,
        fidelity_level=fidelity_level,
        use_bayesopt=use_bayesopt,
        use_online_bo=use_online_bo,
        use_terminal_schedule=use_terminal_schedule,
        terminal_schedule=terminal_schedule,
        use_ae_constraint=use_ae_constraint,
        use_gmr_direction=use_gmr_direction,
        eval_ae_metric=eval_ae_metric,
        use_localization=use_localization,
        fixed_alpha=exp["fixed_alpha"],
        fixed_c_radius=fixed_c_radius,
        fixed_bo_c_radius=float(fixed_c_radius) if fixed_c_radius is not None else FULLMETHOD_FIXED_C_RADIUS,
        alpha_schedule=alpha_schedule,
        bo_alpha_only=bo_alpha_only,
        use_paired_ucb_gmr=use_paired_ucb_gmr,
        use_terminal_gmr_seq=use_terminal_gmr_seq,
        terminal_gmr_seq=terminal_gmr_seq,
        gmr_internal_global_guard=gmr_internal_global_guard,
        use_apbo_budget_policy=use_apbo_budget_policy,
        apbo_objective_effective=apbo_obj_eff,
        ae_cfg=_resolve_ae_settings(args),
    )
    _validate_protocol_wiring(ctx)
    return ctx


def check_validation_exit(ctx):
    args = ctx.args
    if not any([
        args.validate_terminal_schedule_only == 1,
        args.validate_fidelity_only == 1,
        args.validate_stage_d2_wiring_only == 1,
        args.validate_stage_e2_wiring_only == 1,
    ]):
        return False

    if not ctx.use_terminal_schedule:
        raise ValueError("Validation requires --apbo_schedule_mode terminal_simplex")
    if args.validate_fidelity_only == 1 and ctx.fidelity_record is None:
        raise ValueError("--validate_fidelity_only requires --ensemble_fidelity_manifest")
    print("STAGE_C2_TERMINAL_SCHEDULE_VALID")
    if ctx.fidelity_record is not None:
        print("STAGE_C2_NESTED_FIDELITY_VALID")
    if args.validate_stage_d2_wiring_only == 1:
        if not ctx.use_paired_ucb_gmr or args.exp != "fullMethod":
            raise ValueError("Stage-D.2 validation requires fullMethod paired_ucb_search")
        print("STAGE_D2_PAIRED_UCB_WIRING_VALID")
    if args.validate_stage_e2_wiring_only == 1:
        if not ctx.use_terminal_gmr_seq or args.exp != "fullMethod":
            raise ValueError("Stage-E2 validation requires fullMethod terminal_sequence")
        print("STAGE_E2_TERMINAL_GMR_SEQUENCE_VALID")

    payload = {
        "run_tag": ctx.run_tag,
        "seed": ctx.exp["seed"],
        "n_assim": ctx.n_assim,
        "use_bayesopt_method": ctx.use_bayesopt,
        "use_online_bayesopt": ctx.use_online_bo,
        "apbo_schedule_mode": args.apbo_schedule_mode,
        "alpha_schedule_file": ctx.terminal_schedule["schedule_file"],
        "schedule_hash": ctx.terminal_schedule["schedule_hash"],
        "budget_shares": ctx.terminal_schedule["budget_shares"],
        "alpha_schedule": ctx.terminal_schedule["alpha_schedule"],
        "inverse_alpha_sum": ctx.terminal_schedule["inverse_alpha_sum"],
        "source_ensemble_size": ctx.source_N,
        "selected_ensemble_size": ctx.N,
        "ensemble_fidelity_level": ctx.fidelity_level or None,
        "ensemble_fidelity_manifest": ctx.fidelity_record["manifest_file"] if ctx.fidelity_record else None,
        "ensemble_fidelity_manifest_hash": ctx.fidelity_record["manifest_hash"] if ctx.fidelity_record else None,
        "source_member_indices": list(ctx.fidelity_indices),
        "gmr_forward_guard_policy": args.gmr_forward_guard_policy,
        "gmr_global_guard": ctx.gmr_internal_global_guard,
        "use_terminal_gmr_sequence": ctx.use_terminal_gmr_seq,
        "gmr_terminal_sequence_file": ctx.terminal_gmr_seq["sequence_file"] if ctx.terminal_gmr_seq else None,
        "gmr_terminal_sequence_hash": ctx.terminal_gmr_seq["sequence_hash"] if ctx.terminal_gmr_seq else None,
        "gmr_terminal_lambda_sequence": ctx.terminal_gmr_seq["lambda_sequence"] if ctx.terminal_gmr_seq else None,
    }
    print(json.dumps(payload, indent=2, ensure_ascii=True))
    return True


def prepare_result_paths(ctx):
    args = ctx.args
    feedback_tag = "terminal_simplex" if ctx.use_terminal_schedule else (args.feedback if ctx.use_bayesopt else "noBO")
    loc_tag = "loc" if ctx.use_localization else "noloc"
    ae_tag = "AE" if ctx.use_ae_constraint else "noAE"
    bo_dim_tag = "alphaOnly" if ctx.bo_alpha_only else "alphaC"

    if ctx.use_terminal_schedule:
        apbo_tag = f"__apboterminal_{ctx.terminal_schedule['schedule_hash'][:12]}"
    elif ctx.use_apbo_budget_policy:
        apbo_tag = f"__apbo{args.apbo_policy}_{ctx.apbo_objective_effective}"
    else:
        apbo_tag = ""

    fidelity_tag = (
        f"__fidelity{ctx.fidelity_level.lower()}n{ctx.N}_{ctx.fidelity_record['manifest_hash'][:12]}"
        if ctx.fidelity_record is not None
        else ""
    )
    gmr_seq_tag = (
        f"__gmrterminal_{ctx.terminal_gmr_seq['sequence_hash'][:12]}"
        if ctx.terminal_gmr_seq is not None
        else ""
    )

    ae_form_map = {
        "distribution": "__aedist",
        "phi_aligned": "__aephi",
        "gmr_direction": "__gmrdir",
        "recon_project": "__aerecon",
        "prior_manifold": f"__aemanifold{args.ae_prior_manifold_mode}q{args.ae_prior_envelope_q:g}cap{args.ae_prior_exceedance_cap:g}",
    }
    ae_form_tag = ae_form_map.get(args.ae_constraint_form, "") if ctx.use_ae_constraint else ""
    if not ctx.use_ae_constraint or args.ae_sigma_mode == "adaptive":
        ae_sigma_tag = ""
    elif args.ae_sigma_mode == "prior_fixed":
        ae_sigma_tag = "__sigprior"
    else:
        ae_sigma_tag = f"__sigcap{args.ae_sigma_cap_factor:g}"

    verbose_comp = (
        f"{ctx.run_tag}__{feedback_tag}__{args.logic}__{loc_tag}__{ae_tag}__"
        f"{bo_dim_tag}{apbo_tag}{fidelity_tag}{gmr_seq_tag}__ae{args.ae_preset}{ae_form_tag}{ae_sigma_tag}"
    )
    result_comp = build_result_component(
        verbose_comp,
        compact=bool(args.compact_result_path),
        run_tag=ctx.run_tag,
        feedback_tag=feedback_tag,
        schedule_hash=ctx.terminal_schedule.get("schedule_hash", "") if ctx.terminal_schedule else "",
        fidelity_hash=ctx.fidelity_record.get("manifest_hash", "") if ctx.fidelity_record else "",
        seed=ctx.exp["seed"],
    )
    requested_root = os.path.abspath(os.path.join(args.out_root, ctx.run_id, result_comp))
    result_root, collision, collision_action = _resolve_existing_result_root(requested_root, args.existing_run_policy)
    os.makedirs(result_root, exist_ok=True)

    dirs = {
        name: os.path.join(result_root, name)
        for name in ("ae_diag", "gmr_diag", "perm_txt", "iteration_states", "analysis_tables", "bo_diagnostics", "apbo_diag")
    }
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)

    apbo_ref = load_reference_file(args.apbo_reference_file) if ctx.use_apbo_budget_policy else {"source": "not_used", "found": False}
    if (
        ctx.use_apbo_budget_policy
        and ctx.apbo_objective_effective in {"hv_efficiency", "geo_constrained_prod_efficiency"}
        and not has_fixed_geo_reference(apbo_ref)
    ):
        raise ValueError(f"APBO {ctx.apbo_objective_effective} requires a fixed positive GeoScore reference via --apbo_reference_file")

    paths = {
        "log": os.path.join(dirs["analysis_tables"], f"optimization_log_{ctx.run_tag}_{feedback_tag}.csv"),
        "progress": os.path.join(result_root, f"iteration_status_{ctx.run_tag}_{feedback_tag}.json"),
        "best_params": os.path.join(result_root, f"best_params_{ctx.run_tag}_{feedback_tag}.json"),
        "config": os.path.join(result_root, f"experiment_config_{ctx.run_tag}_{feedback_tag}.json"),
        "summary": os.path.join(result_root, f"run_summary_{ctx.run_tag}_{feedback_tag}.json"),
        "prior_metrics": os.path.join(result_root, f"prior_metrics_{ctx.run_tag}_{feedback_tag}.json"),
        "geology_metrics": os.path.join(dirs["analysis_tables"], "geology_metrics.csv"),
        "apbo_policy_log": os.path.join(dirs["apbo_diag"], "apbo_policy_log.csv"),
        "apbo_forward_guard_log": os.path.join(dirs["apbo_diag"], "apbo_forward_guard.csv"),
        "cb_production_metrics": os.path.join(dirs["analysis_tables"], "cb_production_metrics.csv"),
        "gmr_forward_guard_log": os.path.join(dirs["gmr_diag"], "gmr_forward_guard.csv"),
        "posterior_selection": os.path.join(result_root, f"posterior_selection_{ctx.run_tag}_{feedback_tag}.json"),
    }

    ctx.feedback_tag = feedback_tag
    ctx.requested_result_root = requested_root
    ctx.result_root = result_root
    ctx.result_root_collision = collision
    ctx.result_root_collision_action = collision_action
    ctx.result_component = result_comp
    ctx.verbose_result_component = verbose_comp
    ctx.dirs = dirs
    ctx.paths = paths
    ctx.apbo_reference = apbo_ref
    ctx.ae_patch_mode = "gmr_prior_patch_bank" if ctx.use_gmr_direction else ("fixed_bank" if ctx.ae_cfg["ae_align_with_eval_patch_bank"] else "hard_mining")

    ctx.config_payload = {
        "run_tag": ctx.run_tag,
        "description": ctx.exp["description"],
        "method_role": ctx.exp.get("method_role"),
        "paper_method_metadata": get_paper_method_metadata(
            ctx.run_tag,
            ctx.use_localization,
            ctx.use_bayesopt,
            ctx.use_ae_constraint,
            ctx.bo_alpha_only,
            ctx.fixed_alpha,
            ctx.fixed_c_radius,
            ctx.alpha_schedule,
            args.apbo_schedule_mode,
        ),
        "logic_mode": args.logic,
        "feedback_mode": args.feedback,
        "apbo_policy": args.apbo_policy,
        "apbo_policy_effective": (
            "terminal_simplex" if ctx.use_terminal_schedule else ("budget_ratio_hvi" if ctx.use_apbo_budget_policy else "legacy_alpha_bo")
        ),
        "apbo_schedule_mode": args.apbo_schedule_mode,
        "alpha_schedule_file": ctx.terminal_schedule.get("schedule_file") if ctx.terminal_schedule else None,
        "terminal_schedule_record": ctx.terminal_schedule,
        "apbo_objective": args.apbo_objective,
        "apbo_objective_effective": ctx.apbo_objective_effective,
        "apbo_forward_guard_policy": args.apbo_forward_guard_policy,
        "apbo_forward_guard_feedback": args.apbo_forward_guard_feedback,
        "apbo_forward_guard_log_file": paths["apbo_forward_guard_log"] if args.apbo_forward_guard_policy != "off" and ctx.use_apbo_budget_policy else None,
        "apbo_reference": apbo_ref,
        "apbo_policy_log_file": paths["apbo_policy_log"] if ctx.use_apbo_budget_policy else None,
        "cb_production_metrics_file": paths["cb_production_metrics"],
        "gmr_forward_guard_log_file": paths["gmr_forward_guard_log"] if args.gmr_forward_guard else None,
        "posterior_selection_file": paths["posterior_selection"],
        "summary_file": paths["summary"],
        "production_metric": args.production_metric,
        "requested_result_root": requested_root,
        "result_root": result_root,
        "compact_result_path": bool(args.compact_result_path),
        "result_component": result_comp,
        "verbose_result_component": verbose_comp,
        "existing_run_policy": args.existing_run_policy,
        "result_root_collision": collision,
        "result_root_collision_action": collision_action,
        "run_id": ctx.run_id,
        "bo_alpha_only_effective": ctx.bo_alpha_only,
        "save_full_ensemble": bool(args.save_full_ensemble),
        "save_iteration_visuals": bool(args.save_iteration_visuals),
        "use_bayesopt": ctx.use_bayesopt,
        "use_online_bayesopt": ctx.use_online_bo,
        "use_ae_constraint": ctx.use_ae_constraint,
        "eval_ae_metric": ctx.eval_ae_metric,
        "use_localization": ctx.use_localization,
        "fixed_alpha": ctx.fixed_alpha,
        "fixed_c_radius": ctx.fixed_c_radius,
        "alpha_schedule": ctx.alpha_schedule,
        "seed": ctx.exp["seed"],
        "n_assim": ctx.n_assim,
        "N": ctx.N,
        "source_ensemble_size": ctx.source_N,
        "ensemble_fidelity_level": ctx.fidelity_level or None,
        "source_member_indices": list(ctx.fidelity_indices),
        "nx": ctx.grid.nx,
        "ny": ctx.grid.ny,
        "nz": ctx.grid.nz,
        "lambda_ae": LAMBDA_AE,
        "lambda_phi_unified": LAMBDA_PHI_UNIFIED,
        "use_gmr_direction": ctx.use_gmr_direction,
        "gmr_forward_guard": bool(args.gmr_forward_guard),
        "gmr_forward_guard_policy": args.gmr_forward_guard_policy,
        "gmr_forward_guard_blends": None if (ctx.use_paired_ucb_gmr or ctx.use_terminal_gmr_seq) else parse_blends(args.gmr_forward_guard_blends),
        "use_terminal_gmr_sequence": ctx.use_terminal_gmr_seq,
        "gmr_terminal_sequence_record": ctx.terminal_gmr_seq,
        "gmr_diagnostics_dir": dirs["gmr_diag"],
    }
    return ctx
