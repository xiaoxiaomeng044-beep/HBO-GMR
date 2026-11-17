import numpy as np


GUARDED_GROUPS = ("FOE", "WBHP", "GOR", "WCT")
NUMERICAL_EPS = 1e-12


def parse_blends(value):
    raw = value.split(",") if isinstance(value, str) else list(value)
    blends = sorted({float(np.clip(float(item), 0.0, 1.0)) for item in raw}, reverse=True)
    if 0.0 not in blends:
        blends.append(0.0)
    return blends


def select_pilot_member_indices(n_ensemble, n_pilot):
    n_ensemble = max(int(n_ensemble), 1)
    n_pilot = min(max(int(n_pilot), 1), n_ensemble)
    return sorted(set(np.linspace(0, n_ensemble - 1, n_pilot, dtype=int).tolist()))


def rank_geology_candidates(candidate_geo_scores, before_geo_score, min_geo_gain):
    if not np.isfinite(before_geo_score):
        return []
    target = before_geo_score * (1.0 - max(min_geo_gain, 0.0))
    eligible = [
        (float(b), float(s))
        for b, s in candidate_geo_scores.items()
        if b > 0.0 and np.isfinite(s) and s <= target
    ]
    return [b for b, _ in sorted(eligible, key=lambda item: (item[1], -item[0]))]


def rank_pareto_geology_candidates(candidate_geo_scores, before_geo_score):
    if not np.isfinite(before_geo_score):
        return []
    eligible = [
        (float(b), float(s))
        for b, s in candidate_geo_scores.items()
        if b > 0.0 and np.isfinite(s) and s < before_geo_score
    ]
    return [b for b, _ in sorted(eligible, key=lambda item: (item[1], -item[0]))]


def production_guard_accepts_mprod_pareto(candidate, baseline):
    checks = {}

    def check_gate(name):
        cand = float(candidate.get(name, np.nan))
        base = float(baseline.get(name, np.nan))
        limit = base + NUMERICAL_EPS * max(1.0, abs(base))
        passed = bool(np.isfinite(cand) and np.isfinite(base) and base >= 0.0 and cand <= limit)
        checks[name] = {
            "candidate": cand,
            "baseline": base,
            "limit": limit,
            "passed": passed,
            "gates_acceptance": True,
        }
        return passed

    passed = check_gate("m_prod")
    passed = check_gate("m_prod_ensemble_mean") and passed

    candidate_groups = candidate.get("m_prod_by_group", {})
    baseline_groups = baseline.get("m_prod_by_group", {})
    for group in GUARDED_GROUPS:
        cand = float(candidate_groups.get(group, np.nan))
        base = float(baseline_groups.get(group, np.nan))
        limit = base + NUMERICAL_EPS * max(1.0, abs(base))
        checks[f"group_{group}"] = {
            "candidate": cand,
            "baseline": base,
            "limit": limit,
            "passed": bool(np.isfinite(cand) and np.isfinite(base) and cand <= limit),
            "gates_acceptance": False,
        }
    return passed, checks


def production_guard_accepts(
    candidate,
    baseline,
    overall_tolerance=0.0,
    wbhp_tolerance=0.0,
    other_group_tolerance=0.05,
    production_metric="category_balanced",
    policy="legacy_group",
):
    if policy.lower() == "mprod_pareto":
        return production_guard_accepts_mprod_pareto(candidate, baseline)

    checks = {}
    use_m_prod = production_metric.lower() == "m_prod"
    overall_key = "m_prod" if use_m_prod else "cb_prod_nrmse"
    ensemble_key = "m_prod_ensemble_mean" if use_m_prod else "cb_prod_nrmse_ensemble_mean"
    group_key = "m_prod_by_group" if use_m_prod else "cb_prod_nrmse_by_group"

    def check_value(name, tolerance):
        cand = float(candidate.get(name, np.nan))
        base = float(baseline.get(name, np.nan))
        limit = base * (1.0 + max(tolerance, 0.0))
        passed = bool(np.isfinite(cand) and np.isfinite(base) and base > 0.0 and cand <= limit)
        checks[name] = {"candidate": cand, "baseline": base, "limit": limit, "passed": passed}
        return passed

    passed = check_value(overall_key, overall_tolerance)
    passed = check_value(ensemble_key, overall_tolerance) and passed
    candidate_groups = candidate.get(group_key, {})
    baseline_groups = baseline.get(group_key, {})
    for group in GUARDED_GROUPS:
        tolerance = wbhp_tolerance if group == "WBHP" else other_group_tolerance
        cand = float(candidate_groups.get(group, np.nan))
        base = float(baseline_groups.get(group, np.nan))
        limit = base * (1.0 + max(tolerance, 0.0))
        group_passed = bool(np.isfinite(cand) and np.isfinite(base) and base > 0.0 and cand <= limit)
        checks[f"group_{group}"] = {"candidate": cand, "baseline": base, "limit": limit, "passed": group_passed}
        passed = group_passed and passed
    return passed, checks
