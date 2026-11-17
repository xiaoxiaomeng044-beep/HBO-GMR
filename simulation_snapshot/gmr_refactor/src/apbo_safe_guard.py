import numpy as np


NUMERICAL_EPS = 1e-12


def decisions_are_equivalent(proposed_alpha, equal_budget_alpha):
    scale = max(1.0, abs(proposed_alpha), abs(equal_budget_alpha))
    return abs(proposed_alpha - equal_budget_alpha) <= NUMERICAL_EPS * scale


def select_paired_equal_budget_candidate(proposed, equal_budget):
    checks = {}
    accepted = True
    for name in ("m_prod", "m_prod_ensemble_mean"):
        proposed_value = float(proposed.get(name, np.nan))
        equal_value = float(equal_budget.get(name, np.nan))
        limit = equal_value + NUMERICAL_EPS * max(1.0, abs(equal_value))
        passed = bool(
            np.isfinite(proposed_value)
            and np.isfinite(equal_value)
            and equal_value >= 0.0
            and proposed_value <= limit
        )
        checks[name] = {
            "proposed": proposed_value,
            "equal_budget": equal_value,
            "limit": limit,
            "passed": passed,
        }
        accepted = passed and accepted
    return ("proposed" if accepted else "equal_budget"), accepted, checks


def build_paired_dominance_feedback(proposed_eta, proposed=None, equal_budget=None, *, equivalent=False):
    eta = float(proposed_eta)
    if not np.isfinite(eta):
        raise ValueError("proposed_eta must be finite")

    metric_names = ("m_prod", "m_prod_ensemble_mean")
    if equivalent:
        return {
            "objective": "paired_dominance",
            "eta": eta,
            "score": 0.0,
            "ratios": {name: 1.0 for name in metric_names},
            "worst_metric": "equivalent",
            "equivalent": True,
        }

    ratios = {}
    for name in metric_names:
        p_val = float(proposed.get(name, np.nan))
        e_val = float(equal_budget.get(name, np.nan))
        if not (np.isfinite(p_val) and np.isfinite(e_val) and e_val > 0.0):
            raise ValueError(f"invalid paired metric for {name}")
        ratios[name] = p_val / e_val

    worst_metric = max(metric_names, key=lambda name: ratios[name])
    return {
        "objective": "paired_dominance",
        "eta": eta,
        "score": float(ratios[worst_metric] - 1.0),
        "ratios": ratios,
        "worst_metric": worst_metric,
        "equivalent": False,
    }


def resolve_optimizer_feedback(mode, selected_eta, selected_score, learning_observation=None):
    if mode == "paired_dominance":
        if learning_observation is None:
            raise ValueError("paired_dominance feedback requires a learning observation")
        eta = float(learning_observation.get("eta", np.nan))
        score = float(learning_observation.get("score", np.nan))
        source = "paired_dominance_proposed_eta"
    else:
        eta = float(selected_eta)
        score = float(selected_score)
        source = "selected_full"
    if not np.isfinite(eta) or not np.isfinite(score):
        raise ValueError("optimizer feedback eta and score must be finite")
    return eta, score, source
