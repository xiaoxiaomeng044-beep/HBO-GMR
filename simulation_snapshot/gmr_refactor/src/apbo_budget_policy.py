from dataclasses import dataclass
import json
import os

import numpy as np


EPS = 1e-12


@dataclass
class BudgetDecision:
    eta_suggest: float
    eta_used: float
    alpha_used: float
    budget_use: float
    budget_ratio: float
    u_equal: float
    eta_min_feasible: float
    eta_max_feasible: float
    lower_budget_use: float
    upper_budget_use: float
    was_clipped: bool
    will_stop_after: bool

    def as_dict(self):
        return {
            "eta_suggest": float(self.eta_suggest),
            "eta_used": float(self.eta_used),
            "alpha_used": float(self.alpha_used),
            "budget_use": float(self.budget_use),
            "budget_ratio": float(self.budget_ratio),
            "u_equal": float(self.u_equal),
            "eta_min_feasible": float(self.eta_min_feasible),
            "eta_max_feasible": float(self.eta_max_feasible),
            "lower_budget_use": float(self.lower_budget_use),
            "upper_budget_use": float(self.upper_budget_use),
            "was_clipped": self.was_clipped,
            "will_stop_after": self.will_stop_after,
        }


def finite_or_nan(value):
    if value is None:
        return float("nan")
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out if np.isfinite(out) else float("nan")


def _none_if_nan(value):
    return float(value) if np.isfinite(value) else None


def load_reference_file(path):
    if not path or not str(path).strip():
        return {"source": "none", "found": False}
    norm_path = os.path.abspath(path)
    if not os.path.exists(norm_path):
        return {"source": norm_path, "found": False}
    with open(norm_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {**data, "source": norm_path, "found": True}


def has_fixed_geo_reference(reference):
    ref = reference or {}
    return any(
        np.isfinite(finite_or_nan(ref.get(k))) and finite_or_nan(ref.get(k)) > 0
        for k in ("geo_score_ref", "GeoScore_ref", "geo_ref", "GeoScore")
    )


def compute_prod_norm(prod_nrmse, prior_metrics, rmse=None):
    prod = finite_or_nan(prod_nrmse)
    prod_prior = finite_or_nan((prior_metrics or {}).get("prod_nrmse_prior"))
    if np.isfinite(prod) and np.isfinite(prod_prior) and prod_prior > 0:
        return float(prod / prod_prior)
    rmse_now = finite_or_nan(rmse)
    rmse_prior = finite_or_nan((prior_metrics or {}).get("rmse_prior"))
    if np.isfinite(rmse_now) and np.isfinite(rmse_prior) and rmse_prior > 0:
        return float(rmse_now / rmse_prior)
    return float("nan")


def choose_geo_reference(reference, observed_geo_max, current_geo=None, fallback=1.0):
    ref = reference or {}
    for key in ("geo_score_ref", "GeoScore_ref", "geo_ref", "GeoScore"):
        val = finite_or_nan(ref.get(key))
        if np.isfinite(val) and val > 0:
            return {"geo_ref": float(val), "source": str(ref.get("source", f"reference_file:{key}")), "mode": "reference_file"}
    candidates = [finite_or_nan(observed_geo_max), finite_or_nan(current_geo)]
    best = max([v for v in candidates if np.isfinite(v)] + [float(fallback)])
    return {"geo_ref": max(float(best), EPS), "source": "auto_observed_max", "mode": "auto_observed"}


def compute_hypervolume(prod_norm, geo_score, geo_ref, prod_ref=1.0):
    p, g, g_ref = finite_or_nan(prod_norm), finite_or_nan(geo_score), finite_or_nan(geo_ref)
    p_ref = max(float(prod_ref), EPS)
    if not (np.isfinite(p) and np.isfinite(g) and np.isfinite(g_ref) and g_ref > 0):
        return {"prod_component": float("nan"), "geo_component": float("nan"), "geo_norm": float("nan"), "hypervolume": float("nan")}
    prod_comp = max(0.0, (p_ref - p) / p_ref)
    geo_norm = g / g_ref
    geo_comp = max(0.0, 1.0 - geo_norm)
    return {"prod_component": prod_comp, "geo_component": geo_comp, "geo_norm": geo_norm, "hypervolume": prod_comp * geo_comp}


def compute_quality_loss(prod_norm, geo_score, geo_ref):
    p, g, g_ref = finite_or_nan(prod_norm), finite_or_nan(geo_score), finite_or_nan(geo_ref)
    if not (np.isfinite(p) and np.isfinite(g) and np.isfinite(g_ref) and g_ref > 0):
        return {"geo_norm": float("nan"), "quality_loss": float("nan")}
    geo_norm = max(g / g_ref, EPS)
    return {"geo_norm": geo_norm, "quality_loss": float(np.sqrt(max(p, EPS) * geo_norm))}


def compute_geo_constraint(geo_score, geo_ref):
    g, g_ref = finite_or_nan(geo_score), finite_or_nan(geo_ref)
    if not (np.isfinite(g) and np.isfinite(g_ref) and g_ref > 0):
        return {"geo_constraint_ref": float("nan"), "geo_constraint_norm": float("nan"), "geo_constraint_violation": float("nan"), "geo_feasible": False}
    norm = g / g_ref
    violation = max(0.0, norm - 1.0)
    return {"geo_constraint_ref": g_ref, "geo_constraint_norm": norm, "geo_constraint_violation": violation, "geo_feasible": violation <= EPS}


def eta_to_budget_decision(eta_suggest, budget_remaining, remaining_steps, alpha_min=2.0, alpha_max=8.0, eta_min=0.50, eta_max=1.50):
    B = max(float(budget_remaining), 0.0)
    r = max(int(remaining_steps), 1)
    if B <= EPS:
        return BudgetDecision(float("nan"), float("nan"), float("inf"), 0.0, float("nan"), 0.0, float("nan"), float("nan"), 0.0, 0.0, False, True)
    if r == 1:
        alpha = max(float(alpha_min), 1.0 / max(B, EPS))
        return BudgetDecision(float("nan"), 1.0, alpha, 1.0 / alpha, 1.0, B, 1.0, 1.0, B, B, False, True)

    u_equal = B / r
    lower_use = min(B, 1.0 / max(alpha_max, EPS))
    upper_use = min(B, 1.0 / max(alpha_min, EPS))
    reserve_min = (r - 1) * (1.0 / max(alpha_max, EPS))
    if B > reserve_min:
        upper_use = min(upper_use, B - reserve_min)
    upper_use = max(upper_use, EPS)
    lower_use = min(max(lower_use, EPS), upper_use)

    eta_low_feasible = lower_use / max(u_equal, EPS)
    eta_high_feasible = upper_use / max(u_equal, EPS)

    raw_eta = finite_or_nan(eta_suggest)
    if not np.isfinite(raw_eta):
        raw_eta = 1.0
    eta_low, eta_high = max(eta_min, eta_low_feasible), min(eta_max, eta_high_feasible)
    if eta_low > eta_high:
        eta_low, eta_high = eta_low_feasible, eta_high_feasible

    eta_used = float(np.clip(raw_eta, eta_low, eta_high))
    budget_use = float(np.clip(eta_used * u_equal, lower_use, upper_use))
    return BudgetDecision(
        eta_suggest=raw_eta,
        eta_used=eta_used,
        alpha_used=1.0 / max(budget_use, EPS),
        budget_use=budget_use,
        budget_ratio=budget_use / max(B, EPS),
        u_equal=u_equal,
        eta_min_feasible=eta_low_feasible,
        eta_max_feasible=eta_high_feasible,
        lower_budget_use=lower_use,
        upper_budget_use=upper_use,
        was_clipped=abs(eta_used - raw_eta) > 1e-10,
        will_stop_after=(B - budget_use) <= 1e-10,
    )


def _best_after(prev_best, current, *, minimize=True):
    if np.isfinite(prev_best) and np.isfinite(current):
        return min(prev_best, current) if minimize else max(prev_best, current)
    return current if np.isfinite(current) else prev_best


def compute_apbo_feedback(objective, prod_norm, geo_score, prev_state, budget_ratio, reference=None, observed_geo_max=float("nan")):
    obj = (objective or "auto").lower()
    p = finite_or_nan(prod_norm)
    g = finite_or_nan(geo_score)
    br = max(finite_or_nan(budget_ratio), EPS)
    prev = prev_state or {}
    prev_prod = finite_or_nan(prev.get("prod_norm"))
    prev_geo = finite_or_nan(prev.get("geo_score"))
    if not np.isfinite(prev_geo):
        prev_geo = 0.0
    prev_best_prod = finite_or_nan(prev.get("best_prod_norm"))
    if not np.isfinite(prev_best_prod):
        prev_best_prod = prev_prod

    if obj == "auto":
        obj = "geo_constrained_prod_efficiency" if np.isfinite(g) else "prod_efficiency"

    prod_gain_prev = prev_prod - p if np.isfinite(prev_prod) and np.isfinite(p) else float("nan")
    prod_gain_best = prev_best_prod - p if np.isfinite(prev_best_prod) and np.isfinite(p) else float("nan")
    best_prod_after = _best_after(prev_best_prod, p, minimize=True)

    if obj == "prod_efficiency":
        gain = prod_gain_best
        score = -gain / br if np.isfinite(gain) else float("nan")
        return {
            "objective": obj,
            "score": float(score),
            "gain": _none_if_nan(gain),
            "gain_per_budget": _none_if_nan(gain / br) if np.isfinite(gain) else None,
            "prev_gain": _none_if_nan(prod_gain_prev),
            "best_prod_norm_before": _none_if_nan(prev_best_prod),
            "best_prod_norm_after": _none_if_nan(best_prod_after),
            "budget_ratio": br,
            "prod_norm": _none_if_nan(p),
            "geo_score": _none_if_nan(g),
            "geo_ref": None,
            "geo_ref_source": "not_used",
            "hypervolume": None,
            "hypervolume_gain": None,
            "quality_loss": None,
            "quality_gain": None,
        }

    geo_ref_pack = choose_geo_reference(reference, observed_geo_max, current_geo=g)
    geo_ref = geo_ref_pack["geo_ref"]
    hv_now = compute_hypervolume(p, g, geo_ref)["hypervolume"]
    hv_prev = compute_hypervolume(prev_prod, prev_geo, geo_ref)["hypervolume"]
    q_now = compute_quality_loss(p, g, geo_ref)["quality_loss"]
    q_prev = compute_quality_loss(prev_prod, prev_geo, geo_ref)["quality_loss"]

    prev_best_hv = finite_or_nan(prev.get("best_hypervolume"))
    if not np.isfinite(prev_best_hv):
        prev_best_hv = hv_prev
    best_hv_after = _best_after(prev_best_hv, hv_now, minimize=False)

    prev_best_q = finite_or_nan(prev.get("best_quality_loss"))
    if not np.isfinite(prev_best_q):
        prev_best_q = q_prev
    best_q_after = _best_after(prev_best_q, q_now, minimize=True)

    hv_gain_prev = hv_now - hv_prev if np.isfinite(hv_now) and np.isfinite(hv_prev) else float("nan")
    hv_gain = hv_now - prev_best_hv if np.isfinite(hv_now) and np.isfinite(prev_best_hv) else float("nan")
    q_gain_prev = q_prev - q_now if np.isfinite(q_prev) and np.isfinite(q_now) else float("nan")
    q_gain = prev_best_q - q_now if np.isfinite(prev_best_q) and np.isfinite(q_now) else float("nan")

    out = {
        "objective": obj,
        "best_prod_norm_before": _none_if_nan(prev_best_prod),
        "best_prod_norm_after": _none_if_nan(best_prod_after),
        "best_hypervolume_before": _none_if_nan(prev_best_hv),
        "best_hypervolume_after": _none_if_nan(best_hv_after),
        "best_quality_loss_before": _none_if_nan(prev_best_q),
        "best_quality_loss_after": _none_if_nan(best_q_after),
        "budget_ratio": br,
        "prod_norm": _none_if_nan(p),
        "geo_score": _none_if_nan(g),
        "geo_ref": geo_ref,
        "geo_ref_source": geo_ref_pack["source"],
        "hypervolume": _none_if_nan(hv_now),
        "hypervolume_gain": _none_if_nan(hv_gain),
        "hypervolume_gain_prev": _none_if_nan(hv_gain_prev),
        "quality_loss": _none_if_nan(q_now),
        "quality_gain": _none_if_nan(q_gain),
        "quality_gain_prev": _none_if_nan(q_gain_prev),
    }

    if obj == "quality_efficiency":
        gain, prev_gain = q_gain, q_gain_prev
        score = -gain / br if np.isfinite(gain) else float("nan")
    elif obj == "geo_constrained_prod_efficiency":
        constraint = compute_geo_constraint(g, geo_ref)
        geo_violation = finite_or_nan(constraint.get("geo_constraint_violation"))
        feasible = bool(constraint.get("geo_feasible", False))
        gain, prev_gain = prod_gain_best, prod_gain_prev
        score = (-gain / br if np.isfinite(gain) else float("nan")) if feasible else (geo_violation / br if np.isfinite(geo_violation) else float("nan"))
        out.update({
            "geo_feasible": feasible,
            "geo_constraint_ref": _none_if_nan(constraint["geo_constraint_ref"]),
            "geo_constraint_norm": _none_if_nan(constraint["geo_constraint_norm"]),
            "geo_constraint_violation": _none_if_nan(geo_violation),
        })
    else:
        obj = "hv_efficiency"
        gain, prev_gain = hv_gain, hv_gain_prev
        score = -gain / br if np.isfinite(gain) else float("nan")

    out.update({
        "objective": obj,
        "score": float(score),
        "gain": _none_if_nan(gain),
        "gain_per_budget": _none_if_nan(gain / br) if np.isfinite(gain) else None,
        "prev_gain": _none_if_nan(prev_gain),
    })
    return out


def next_state(prod_norm, geo_score, feedback=None, iteration=None):
    p = finite_or_nan(prod_norm)
    out = {
        "prod_norm": p,
        "geo_score": finite_or_nan(geo_score),
        "iteration": iteration,
        "best_prod_norm": p,
    }
    if isinstance(feedback, dict):
        for key in (
            "hypervolume",
            "quality_loss",
            "geo_ref",
            "geo_ref_source",
            "best_prod_norm_after",
            "best_hypervolume_after",
            "best_quality_loss_after",
        ):
            if key in feedback:
                state_key = key[:-6] if key.endswith("_after") else key
                out[state_key] = feedback[key]
    return out
