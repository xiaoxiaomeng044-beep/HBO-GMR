import os

import numpy as np

from eclipse import eclipse
from main_ESMDA_Localization_v3_fixed2 import main_ESMDA_Localization
from metrics_geology import compute_geology_metrics
from metrics_production import (
    compute_category_scaled_weighted_mismatch,
    compute_production_metrics_from_rsm_members,
    read_real_rsm,
    read_rsm_members,
)
from .apbo_budget_policy import eta_to_budget_decision
from .apbo_safe_guard import (
    build_paired_dominance_feedback,
    decisions_are_equivalent,
    select_paired_equal_budget_candidate,
)
from .common import append_csv_row, save_json
from .gmr_paired_ucb_runtime import (
    build_production_response_payload,
    ensemble_state_hash,
    evaluate_response_payload,
    load_pilot_response_cache,
    restore_pilot_rsm_files,
    write_pilot_response_cache,
)
from .gmr_paired_ucb_search import (
    SafeBlendSearch,
    make_blend_observation,
    nested_pilot_member_indices,
    paired_bootstrap_log_constraints,
    pilot_cache_key,
)
from .gmr_terminal_sequence_runtime import apply_terminal_sequence_blend
from .production_forward_guard import (
    parse_blends,
    production_guard_accepts,
    rank_geology_candidates,
    rank_pareto_geology_candidates,
    select_pilot_member_indices,
)


def write_member_grdecl(work_dir, perm, poro, permz, member_indices):
    definitions = (
        ("PERMX2.GRDECL", "PERMX", perm),
        ("PERMZ2.GRDECL", "PERMZ", permz),
        ("PORO2.GRDECL", "PORO", poro),
    )
    for member_index in member_indices:
        folder = os.path.join(work_dir, f"MASTER{member_index + 1:05d}")
        for filename, keyword, ensemble in definitions:
            path = os.path.join(folder, filename)
            with open(path, "w", encoding="ascii", newline="\n") as stream:
                stream.write(keyword + "\n")
                np.savetxt(stream, ensemble[:, member_index], fmt="%1.7e")
                stream.write("/\n")


def run_forward_guard_pilot(work_dir, perm, poro, permz, member_indices):
    write_member_grdecl(work_dir, perm, poro, permz, member_indices)
    for member_index in member_indices:
        rsm_path = os.path.join(work_dir, f"MASTER{member_index + 1:05d}", "MASTER0.RSM")
        if os.path.exists(rsm_path):
            os.remove(rsm_path)
        eclipse(member_index)
        os.chdir(work_dir)
        if not os.path.exists(rsm_path):
            raise RuntimeError(f"Forward guard simulation did not create {rsm_path}")
    return compute_production_metrics_from_rsm_members(member_indices, real_path="Real.RSM", root_dir=work_dir)


def _clean_updated_ensemble(perm, poro, permz, total_cells, n_members):
    cleaned = []
    for values in (perm, poro, permz):
        array = np.reshape(values, (total_cells, n_members)).copy()
        array[array == 1] = 0
        cleaned.append(np.abs(array))
    return tuple(cleaned)


def compute_esmda_candidate(
    ctx,
    ref_fields,
    base_perm,
    base_poro,
    base_permz,
    observation,
    overallsim,
    history,
    effective,
    alpha_used,
    c_radius_used,
    ae_pseudo,
    iteration,
    status_callback=None,
):
    grid, ae_cfg, args = ctx.grid, ctx.ae_cfg, ctx.args
    updated = main_ESMDA_Localization(
        grid.nx,
        grid.ny,
        grid.nz,
        ctx.N,
        observation,
        overallsim,
        ref_fields["permx"],
        ref_fields["poro"],
        ref_fields["permz"],
        base_perm,
        base_poro,
        base_permz,
        history,
        alpha_used,
        effective,
        c_radius=c_radius_used,
        use_localization=ctx.use_localization,
        update_status=status_callback,
        ae_pseudo=ae_pseudo,
        ae_time_window=ae_cfg["ae_time_window"],
        ae_time_sigma_multipliers=ae_cfg["ae_time_sigma_multipliers"],
        ae_update_weight=args.ae_update_weight,
        n_obs=grid.n_observations,
        n_producers=grid.n_producers,
        well_locations=grid.get_producer_ij_locations(),
        param_bounds=grid.parameter_bounds,
        random_seed=ctx.exp["seed"] + iteration * 1000,
        observation_error_member_indices=ctx.fidelity_indices if ctx.fidelity_record else None,
        observation_error_bank_size=ctx.source_N if ctx.fidelity_record else None,
    )
    return _clean_updated_ensemble(*updated, total_cells=grid.total_cells, n_members=ctx.N)


def apply_apbo_equal_budget_guard(
    ctx,
    work_dir,
    ref_fields,
    proposed_perm,
    proposed_poro,
    proposed_permz,
    base_perm,
    base_poro,
    base_permz,
    observation,
    overallsim,
    history,
    effective,
    ae_pseudo,
    iteration,
    c_radius_used,
    proposed_decision,
    budget_remaining,
    remaining_steps,
):
    args = ctx.args
    equal_decision = eta_to_budget_decision(
        1.0,
        budget_remaining=budget_remaining,
        remaining_steps=remaining_steps,
        alpha_min=2.0,
        alpha_max=8.0,
        eta_min=args.apbo_eta_min,
        eta_max=args.apbo_eta_max,
    )
    diagnostics = {
        "iteration": iteration,
        "policy": "paired_equal_budget",
        "pilot_members": args.apbo_forward_guard_pilot_members,
        "proposed_decision": proposed_decision.as_dict(),
        "equal_budget_decision": equal_decision.as_dict(),
        "selected": "proposed",
        "accepted_proposed": True,
        "checks": {},
        "proposed_metrics": None,
        "equal_budget_metrics": None,
        "simulations_skipped": False,
        "learning_observation": None,
    }

    if decisions_are_equivalent(proposed_decision.alpha_used, equal_decision.alpha_used):
        diagnostics["simulations_skipped"] = True
        diagnostics["reason"] = "proposed_and_equal_budget_are_equivalent"
        diagnostics["learning_observation"] = build_paired_dominance_feedback(proposed_decision.eta_used, equivalent=True)
        selected_fields = (proposed_perm, proposed_poro, proposed_permz)
        selected_decision = proposed_decision
    else:
        equal_perm, equal_poro, equal_permz = compute_esmda_candidate(
            ctx,
            ref_fields,
            base_perm,
            base_poro,
            base_permz,
            observation,
            overallsim,
            history,
            effective,
            equal_decision.alpha_used,
            c_radius_used,
            ae_pseudo,
            iteration,
        )
        pilot_indices = select_pilot_member_indices(ctx.N, args.apbo_forward_guard_pilot_members)
        diagnostics["pilot_member_indices_zero_based"] = pilot_indices
        diagnostics["pilot_member_indices_one_based"] = [i + 1 for i in pilot_indices]

        equal_metrics = run_forward_guard_pilot(work_dir, equal_perm, equal_poro, equal_permz, pilot_indices)
        proposed_metrics = run_forward_guard_pilot(work_dir, proposed_perm, proposed_poro, proposed_permz, pilot_indices)
        selected_name, accepted, checks = select_paired_equal_budget_candidate(proposed_metrics, equal_metrics)
        diagnostics.update({
            "selected": selected_name,
            "accepted_proposed": accepted,
            "checks": checks,
            "proposed_metrics": proposed_metrics,
            "equal_budget_metrics": equal_metrics,
            "learning_observation": build_paired_dominance_feedback(
                proposed_decision.eta_used, proposed_metrics, equal_metrics
            ),
        })
        if accepted:
            selected_fields = (proposed_perm, proposed_poro, proposed_permz)
            selected_decision = proposed_decision
        else:
            selected_fields = (equal_perm, equal_poro, equal_permz)
            selected_decision = equal_decision

    save_json(diagnostics, os.path.join(ctx.dirs["apbo_diag"], f"apbo_forward_guard_iter{iteration:02d}.json"))
    append_csv_row(
        ctx.paths["apbo_forward_guard_log"],
        {
            "Iteration": iteration,
            "Policy": "paired_equal_budget",
            "ProposedAlpha": proposed_decision.alpha_used,
            "EqualBudgetAlpha": equal_decision.alpha_used,
            "Selected": diagnostics["selected"],
            "AcceptedProposed": diagnostics["accepted_proposed"],
            "SimulationsSkipped": diagnostics["simulations_skipped"],
            "ProposedMProd": diagnostics["proposed_metrics"]["m_prod"] if diagnostics["proposed_metrics"] else None,
            "EqualBudgetMProd": diagnostics["equal_budget_metrics"]["m_prod"] if diagnostics["equal_budget_metrics"] else None,
            "ProposedMProdEnsembleMean": diagnostics["proposed_metrics"]["m_prod_ensemble_mean"] if diagnostics["proposed_metrics"] else None,
            "EqualBudgetMProdEnsembleMean": diagnostics["equal_budget_metrics"]["m_prod_ensemble_mean"] if diagnostics["equal_budget_metrics"] else None,
            "LearningEta": diagnostics["learning_observation"]["eta"],
            "PairedDominanceLoss": diagnostics["learning_observation"]["score"],
            "PairedDominanceWorstMetric": diagnostics["learning_observation"]["worst_metric"],
            "OptimizerFeedbackMode": args.apbo_forward_guard_feedback,
        },
    )
    return (*selected_fields, selected_decision, diagnostics)


def _forward_guard_geo_score(perm, poro, grid, geology_reference, compute_geo_score_fn):
    perm_dhw = np.transpose(np.reshape(np.log10(np.maximum(np.mean(perm, axis=1), 1e-15)), (grid.nx, grid.ny, grid.nz), order="F"), (2, 1, 0)).astype(np.float32)
    poro_dhw = np.transpose(np.reshape(np.mean(poro, axis=1), (grid.nx, grid.ny, grid.nz), order="F"), (2, 1, 0)).astype(np.float32)
    return compute_geo_score_fn(compute_geology_metrics(perm_dhw, poro_dhw, geology_reference))


def _gmr_rsm_paths(work_dir, member_indices):
    return {idx: os.path.join(work_dir, f"MASTER{idx + 1:05d}", "MASTER0.RSM") for idx in member_indices}


def _paired_evaluator(payload, sampled_positions):
    return evaluate_response_payload(payload, sampled_positions, compute_category_scaled_weighted_mismatch)


def _run_gmr_paired_ucb_pilot(ctx, work_dir, perm, poro, permz, member_indices, source_indices, *, iteration, blend, state_hash):
    cache_key = pilot_cache_key(iteration=iteration, blend=blend, member_indices=source_indices, state_hash=state_hash)
    cache_dir = os.path.join(ctx.dirs["gmr_diag"], "paired_ucb_cache", f"iter{iteration:02d}", cache_key)
    if os.path.isdir(cache_dir):
        cached = load_pilot_response_cache(cache_dir, expected_cache_key=cache_key, expected_state_hash=state_hash)
        metrics = _paired_evaluator(cached["payload"], np.arange(len(member_indices), dtype=int))
        return metrics, cached["payload"], cached, True

    write_member_grdecl(work_dir, perm, poro, permz, member_indices)
    rsm_paths = _gmr_rsm_paths(work_dir, member_indices)
    for idx in member_indices:
        rsm_path = rsm_paths[idx]
        if os.path.exists(rsm_path):
            os.remove(rsm_path)
        eclipse(idx)
        os.chdir(work_dir)
        if not os.path.exists(rsm_path):
            raise RuntimeError(f"Paired-UCB pilot simulation did not create {rsm_path}")

    payload = build_production_response_payload(
        read_rsm_members(member_indices, root_dir=work_dir),
        read_real_rsm("Real.RSM"),
        member_indices=member_indices,
        source_member_indices=source_indices,
    )
    cached = write_pilot_response_cache(
        cache_dir,
        cache_key=cache_key,
        state_hash=state_hash,
        iteration=iteration,
        blend=blend,
        payload=payload,
        rsm_paths=rsm_paths,
    )
    metrics = _paired_evaluator(cached["payload"], np.arange(len(member_indices), dtype=int))
    return metrics, cached["payload"], cached, False


def _apply_gmr_paired_ucb(ctx, work_dir, base_perm, base_poro, base_permz, corr_perm, corr_poro, corr_permz, iteration, gmr_diag, geology_reference, compute_geo_score_fn):
    args = ctx.args
    pilot_indices = nested_pilot_member_indices(ctx.N, args.gmr_forward_guard_pilot_members)
    source_indices = [ctx.fidelity_indices[i] for i in pilot_indices] if ctx.fidelity_record else list(pilot_indices)
    before_geo = _forward_guard_geo_score(base_perm, base_poro, ctx.grid, geology_reference, compute_geo_score_fn)
    state_hash = ensemble_state_hash({
        "base_perm": base_perm,
        "base_poro": base_poro,
        "base_permz": base_permz,
        "raw_gmr_perm": corr_perm,
        "raw_gmr_poro": corr_poro,
        "raw_gmr_permz": corr_permz,
    })
    search_seed = args.gmr_paired_ucb_seed + iteration * 1009
    bootstrap_seed = args.gmr_paired_ucb_seed + iteration * 100003
    optimizer = SafeBlendSearch(
        max_nonbaseline_evaluations=args.gmr_paired_ucb_max_evaluations,
        n_initial_nonbaseline=args.gmr_paired_ucb_initial_evaluations,
        pool_size=args.gmr_paired_ucb_pool_size,
        pool_seed=search_seed,
        random_state=search_seed,
    )

    base_metrics, base_payload, base_cache, base_hit = _run_gmr_paired_ucb_pilot(
        ctx, work_dir, base_perm, base_poro, base_permz, pilot_indices, source_indices,
        iteration=iteration, blend=0.0, state_hash=state_hash,
    )
    base_boot = paired_bootstrap_log_constraints(
        base_payload, base_payload, _paired_evaluator,
        n_members=len(pilot_indices), n_bootstrap=args.gmr_paired_ucb_bootstrap_draws,
        confidence=args.gmr_paired_ucb_confidence, seed=bootstrap_seed,
    )
    base_obs = make_blend_observation(0.0, before_geo, before_geo, base_boot)
    optimizer.tell(base_obs)

    cache_records = {"0": base_cache}
    candidate_records = [{
        "blend": 0.0,
        "geo_score": before_geo,
        "production_metrics": base_metrics,
        "bootstrap": base_boot,
        "observation": base_obs,
        "cache_dir": base_cache["cache_dir"],
        "cache_hit": base_hit,
        "ask_diagnostics": {"mode": "mandatory_baseline"},
    }]

    while not optimizer.is_complete:
        blend, ask_diag = optimizer.ask()
        if blend is None:
            break
        cand_perm = np.abs(base_perm + blend * (corr_perm - base_perm))
        cand_poro = np.abs(base_poro + blend * (corr_poro - base_poro))
        cand_permz = np.abs(base_permz + blend * (corr_permz - base_permz))
        cand_geo = _forward_guard_geo_score(cand_perm, cand_poro, ctx.grid, geology_reference, compute_geo_score_fn)
        cand_metrics, cand_payload, cand_cache, hit = _run_gmr_paired_ucb_pilot(
            ctx, work_dir, cand_perm, cand_poro, cand_permz, pilot_indices, source_indices,
            iteration=iteration, blend=blend, state_hash=state_hash,
        )
        boot = paired_bootstrap_log_constraints(
            cand_payload, base_payload, _paired_evaluator,
            n_members=len(pilot_indices), n_bootstrap=args.gmr_paired_ucb_bootstrap_draws,
            confidence=args.gmr_paired_ucb_confidence, seed=bootstrap_seed,
        )
        obs = make_blend_observation(blend, cand_geo, before_geo, boot)
        optimizer.tell(obs)
        cache_records[format(blend, ".15g")] = cand_cache
        candidate_records.append({
            "blend": blend,
            "geo_score": cand_geo,
            "production_metrics": cand_metrics,
            "bootstrap": boot,
            "observation": obs,
            "cache_dir": cand_cache["cache_dir"],
            "cache_hit": hit,
            "ask_diagnostics": ask_diag,
        })

    selected_obs, selection_diag = optimizer.select()
    selected_blend = float(selected_obs["blend"])
    selected_cache = cache_records[format(selected_blend, ".15g")]
    reused_members = restore_pilot_rsm_files(selected_cache, _gmr_rsm_paths(work_dir, pilot_indices))
    selected_rec = next(r for r in candidate_records if np.isclose(r["blend"], selected_blend, atol=1e-15))

    optimizer.save(os.path.join(ctx.dirs["gmr_diag"], f"gmr_paired_ucb_optimizer_iter{iteration:02d}.json"))
    diagnostics = {
        "iteration": iteration,
        "enabled": True,
        "guard_policy": "paired_ucb_search",
        "selected_blend": selected_blend,
        "selected_geo_score": selected_rec["geo_score"],
        "fallback_to_esmda": selected_blend == 0.0,
        "selection_diagnostics": selection_diag,
        "candidate_records": candidate_records,
        "gmr_paired_ucb_reuse_members": reused_members,
        "reused_member_count": len(reused_members),
    }
    if isinstance(gmr_diag, dict):
        gmr_diag.update({
            "forward_guard_enabled": True,
            "forward_guard_policy": "paired_ucb_search",
            "forward_guard_selected_blend": selected_blend,
            "forward_guard_fallback": selected_blend == 0.0,
            "forward_guard_reused_member_count": len(reused_members),
        })
    save_json(diagnostics, os.path.join(ctx.dirs["gmr_diag"], f"gmr_forward_guard_iter{iteration:02d}.json"))
    upper = selected_rec["bootstrap"]["upper_confidence"]
    append_csv_row(
        ctx.paths["gmr_forward_guard_log"],
        {
            "Iteration": iteration,
            "PilotMembers": len(pilot_indices),
            "EvaluatedNonbaselineCandidates": optimizer.n_nonbaseline_observations,
            "SelectedBlend": selected_blend,
            "FallbackToESMDA": selected_blend == 0.0,
            "GuardPolicy": "paired_ucb_search",
            "GuardProductionMetric": "m_prod",
            "BaselineMProd": base_metrics.get("m_prod"),
            "SelectedMProd": selected_rec["production_metrics"].get("m_prod"),
            "BaselineMProdEnsembleMean": base_metrics.get("m_prod_ensemble_mean"),
            "SelectedMProdEnsembleMean": selected_rec["production_metrics"].get("m_prod_ensemble_mean"),
            "MProdLogRatioUpper95": upper.get("m_prod"),
            "MProdEnsembleMeanLogRatioUpper95": upper.get("m_prod_ensemble_mean"),
            "BeforeGeoScore": before_geo,
            "SelectedGeoScore": selected_rec["geo_score"],
            "ReusedPilotMembers": len(reused_members),
        },
    )
    return (
        np.abs(base_perm + selected_blend * (corr_perm - base_perm)),
        np.abs(base_poro + selected_blend * (corr_poro - base_poro)),
        np.abs(base_permz + selected_blend * (corr_permz - base_permz)),
        diagnostics,
    )


def _apply_gmr_legacy_guard(ctx, work_dir, base_perm, base_poro, base_permz, corr_perm, corr_poro, corr_permz, iteration, gmr_diag, geology_reference, compute_geo_score_fn):
    args = ctx.args
    blends = parse_blends(args.gmr_forward_guard_blends)
    policy = args.gmr_forward_guard_policy
    before_geo = _forward_guard_geo_score(base_perm, base_poro, ctx.grid, geology_reference, compute_geo_score_fn)

    candidate_geo = {}
    for blend in blends:
        if blend > 0.0:
            c_perm = np.abs(base_perm + blend * (corr_perm - base_perm))
            c_poro = np.abs(base_poro + blend * (corr_poro - base_poro))
            candidate_geo[blend] = _forward_guard_geo_score(c_perm, c_poro, ctx.grid, geology_reference, compute_geo_score_fn)

    ranked = (
        rank_pareto_geology_candidates(candidate_geo, before_geo)
        if policy == "mprod_pareto"
        else rank_geology_candidates(candidate_geo, before_geo, args.gmr_forward_guard_min_geo_gain)
    )
    pilot_indices = select_pilot_member_indices(ctx.N, args.gmr_forward_guard_pilot_members)
    selected_blend = 0.0
    selected_metrics = None
    baseline_metrics = None
    records = []

    if ranked:
        baseline_metrics = run_forward_guard_pilot(work_dir, base_perm, base_poro, base_permz, pilot_indices)
        for blend in ranked:
            cand_metrics = run_forward_guard_pilot(
                work_dir,
                np.abs(base_perm + blend * (corr_perm - base_perm)),
                np.abs(base_poro + blend * (corr_poro - base_poro)),
                np.abs(base_permz + blend * (corr_permz - base_permz)),
                pilot_indices,
            )
            accepted, checks = production_guard_accepts(
                cand_metrics,
                baseline_metrics,
                overall_tolerance=args.gmr_forward_guard_overall_tol,
                wbhp_tolerance=args.gmr_forward_guard_wbhp_tol,
                other_group_tolerance=args.gmr_forward_guard_other_tol,
                production_metric=args.production_metric,
                policy=policy,
            )
            records.append({"blend": blend, "geo_score": candidate_geo[blend], "accepted": accepted, "checks": checks, "production_metrics": cand_metrics})
            if accepted:
                selected_blend = blend
                selected_metrics = cand_metrics
                break

    diagnostics = {
        "iteration": iteration,
        "enabled": True,
        "guard_policy": policy,
        "selected_blend": selected_blend,
        "fallback_to_esmda": selected_blend == 0.0,
        "before_geo_score": before_geo,
        "candidate_records": records,
    }
    if isinstance(gmr_diag, dict):
        gmr_diag.update({
            "forward_guard_enabled": True,
            "forward_guard_selected_blend": selected_blend,
            "forward_guard_fallback": selected_blend == 0.0,
        })
    save_json(diagnostics, os.path.join(ctx.dirs["gmr_diag"], f"gmr_forward_guard_iter{iteration:02d}.json"))
    append_csv_row(
        ctx.paths["gmr_forward_guard_log"],
        {
            "Iteration": iteration,
            "PilotMembers": len(pilot_indices),
            "CandidatesPassingGeo": len(ranked),
            "SelectedBlend": selected_blend,
            "FallbackToESMDA": selected_blend == 0.0,
            "GuardPolicy": policy,
            "GuardProductionMetric": args.production_metric,
            "BaselineMProd": baseline_metrics.get("m_prod") if baseline_metrics else None,
            "SelectedMProd": selected_metrics.get("m_prod") if selected_metrics else None,
            "BeforeGeoScore": before_geo,
            "SelectedGeoScore": candidate_geo.get(selected_blend, before_geo),
        },
    )
    return (
        np.abs(base_perm + selected_blend * (corr_perm - base_perm)),
        np.abs(base_poro + selected_blend * (corr_poro - base_poro)),
        np.abs(base_permz + selected_blend * (corr_permz - base_permz)),
        diagnostics,
    )


def apply_gmr_forward_guard(ctx, work_dir, base_perm, base_poro, base_permz, corr_perm, corr_poro, corr_permz, iteration, gmr_diag, geology_reference, compute_geo_score_fn):
    policy = ctx.args.gmr_forward_guard_policy
    if policy == "terminal_sequence":
        sel_perm, sel_poro, sel_permz, diagnostics = apply_terminal_sequence_blend(
            base_perm, base_poro, base_permz, corr_perm, corr_poro, corr_permz,
            iteration=iteration, record=ctx.terminal_gmr_seq,
        )
        blend = diagnostics["selected_blend"]
        if isinstance(gmr_diag, dict):
            gmr_diag.update({
                "forward_guard_enabled": True,
                "forward_guard_policy": "terminal_sequence",
                "forward_guard_selected_blend": blend,
                "forward_guard_fallback": blend == 0.0,
                "terminal_sequence_hash": diagnostics["sequence_hash"],
            })
        save_json(diagnostics, os.path.join(ctx.dirs["gmr_diag"], f"gmr_forward_guard_iter{iteration:02d}.json"))
        append_csv_row(
            ctx.paths["gmr_forward_guard_log"],
            {
                "Iteration": iteration,
                "PilotMembers": 0,
                "EvaluatedNonbaselineCandidates": 0,
                "SelectedBlend": blend,
                "FallbackToESMDA": blend == 0.0,
                "GuardPolicy": "terminal_sequence",
                "SequenceHash": diagnostics["sequence_hash"],
                "ForwardSimulationsExecuted": 0,
            },
        )
        return sel_perm, sel_poro, sel_permz, diagnostics

    if policy == "paired_ucb_search":
        return _apply_gmr_paired_ucb(
            ctx, work_dir, base_perm, base_poro, base_permz, corr_perm, corr_poro, corr_permz,
            iteration, gmr_diag, geology_reference, compute_geo_score_fn,
        )

    return _apply_gmr_legacy_guard(
        ctx, work_dir, base_perm, base_poro, base_permz, corr_perm, corr_poro, corr_permz,
        iteration, gmr_diag, geology_reference, compute_geo_score_fn,
    )
