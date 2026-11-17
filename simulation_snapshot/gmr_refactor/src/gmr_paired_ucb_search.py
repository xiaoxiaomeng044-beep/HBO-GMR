import copy
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import qmc

from .apbo_constrained_bo import (
    build_matern_gp,
    constrained_expected_improvement,
    probability_nonpositive,
)
from .common import sha256_json


CONSTRAINT_NAMES = ("m_prod", "m_prod_ensemble_mean")
NUMERICAL_EPS = 1e-12


def sobol_blend_candidates(n_candidates, *, seed):
    if n_candidates < 2:
        raise ValueError("n_candidates must include at least lambda=0 and lambda=1")
    values = [0.0, 1.0]
    seen = {0.0, 1.0}
    sampler = qmc.Sobol(d=1, scramble=True, seed=seed)
    batch_power = max(1, int(np.ceil(np.log2(max(2, n_candidates * 2)))))
    for value in sampler.random_base2(batch_power).reshape(-1):
        clipped = float(np.clip(value, 0.0, 1.0))
        key = round(clipped, 15)
        if key not in seen:
            seen.add(key)
            values.append(clipped)
            if len(values) == n_candidates:
                break
    if len(values) != n_candidates:
        raise RuntimeError("unable to construct the requested unique Sobol pool")
    return np.asarray(values, dtype=float)


def nested_pilot_member_indices(n_ensemble, n_pilot):
    if n_ensemble < 1 or not 1 <= n_pilot <= n_ensemble:
        raise ValueError("n_pilot must be in [1, n_ensemble]")
    return list(range(n_pilot))


def _validate_metric_mapping(metrics):
    output = {}
    for name in CONSTRAINT_NAMES:
        value = float(metrics.get(name, np.nan))
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and positive")
        output[name] = value
    return output


def paired_bootstrap_log_constraints(
    candidate_payload,
    baseline_payload,
    evaluator,
    *,
    n_members,
    n_bootstrap=1000,
    confidence=0.95,
    seed=2026,
):
    if n_members < 2 or n_bootstrap < 1 or not 0.5 < confidence < 1.0:
        raise ValueError("invalid bootstrap parameters")

    full_indices = np.arange(n_members, dtype=int)
    candidate_point = _validate_metric_mapping(evaluator(candidate_payload, full_indices))
    baseline_point = _validate_metric_mapping(evaluator(baseline_payload, full_indices))
    point_log_ratio = {
        name: float(np.log(candidate_point[name] / baseline_point[name]))
        for name in CONSTRAINT_NAMES
    }

    rng = np.random.default_rng(seed)
    bootstrap_indices = rng.integers(0, n_members, size=(n_bootstrap, n_members), endpoint=False)
    samples = {name: np.empty(n_bootstrap, dtype=float) for name in CONSTRAINT_NAMES}
    for row_index, member_indices in enumerate(bootstrap_indices):
        cand_m = _validate_metric_mapping(evaluator(candidate_payload, member_indices))
        base_m = _validate_metric_mapping(evaluator(baseline_payload, member_indices))
        for name in CONSTRAINT_NAMES:
            samples[name][row_index] = np.log(cand_m[name] / base_m[name])

    bootstrap_index_hash = hashlib.sha256(
        np.ascontiguousarray(bootstrap_indices, dtype=np.int64).tobytes()
    ).hexdigest()
    return {
        "version": 1,
        "n_members": n_members,
        "n_bootstrap": n_bootstrap,
        "confidence": confidence,
        "seed": seed,
        "bootstrap_index_hash": bootstrap_index_hash,
        "candidate_metrics": candidate_point,
        "baseline_metrics": baseline_point,
        "point_log_ratio": point_log_ratio,
        "upper_confidence": {name: float(np.quantile(vals, confidence)) for name, vals in samples.items()},
        "standard_error": {
            name: float(np.std(vals, ddof=1)) if n_bootstrap > 1 else 0.0
            for name, vals in samples.items()
        },
        "probability_nonpositive": {name: float(np.mean(vals <= 0.0)) for name, vals in samples.items()},
    }


def passes_production_ucb(bootstrap_summary, *, atol=NUMERICAL_EPS):
    upper = bootstrap_summary.get("upper_confidence", {})
    tolerance = max(atol, 0.0)
    return all(
        np.isfinite(float(upper.get(name, np.nan))) and float(upper[name]) <= tolerance
        for name in CONSTRAINT_NAMES
    )


def _observation_passes_production_ucb(observation):
    upper = observation.get("upper_constraints", {})
    return all(
        np.isfinite(float(upper.get(name, np.nan))) and float(upper[name]) <= NUMERICAL_EPS
        for name in CONSTRAINT_NAMES
    )


def make_blend_observation(blend, geo_score, baseline_geo_score, bootstrap_summary):
    if not 0.0 <= blend <= 1.0:
        raise ValueError("blend must be in [0, 1]")
    if not (np.isfinite(geo_score) and geo_score > 0.0 and np.isfinite(baseline_geo_score) and baseline_geo_score > 0.0):
        raise ValueError("geo scores must be finite and positive")

    point = bootstrap_summary["point_log_ratio"]
    upper = bootstrap_summary["upper_confidence"]
    noise = bootstrap_summary["standard_error"]
    constraints, upper_constraints, constraint_noise = {}, {}, {}
    for name in CONSTRAINT_NAMES:
        val = float(point[name])
        uval = float(upper[name])
        nval = max(float(noise[name]), 0.0)
        if not all(np.isfinite(x) for x in (val, uval, nval)):
            raise ValueError(f"invalid bootstrap values for {name}")
        constraints[name] = val
        upper_constraints[name] = uval
        constraint_noise[name] = nval

    return {
        "blend": float(blend),
        "geo_score": float(geo_score),
        "baseline_geo_score": float(baseline_geo_score),
        "objective": float(np.log(geo_score / baseline_geo_score)),
        "constraints": constraints,
        "upper_constraints": upper_constraints,
        "constraint_noise": constraint_noise,
        "production_safe": passes_production_ucb(bootstrap_summary),
        "bootstrap_index_hash": bootstrap_summary.get("bootstrap_index_hash", ""),
    }


def select_safe_blend(observations, *, baseline_geo_score, geo_atol=NUMERICAL_EPS):
    if not np.isfinite(baseline_geo_score) or baseline_geo_score <= 0.0:
        raise ValueError("baseline_geo_score must be finite and positive")
    baseline_items = [dict(item) for item in observations if np.isclose(item.get("blend", np.nan), 0.0, atol=1e-15)]
    if len(baseline_items) != 1:
        raise ValueError("exactly one lambda=0 baseline observation is required")

    tolerance = max(geo_atol, 0.0) * max(1.0, abs(baseline_geo_score))
    eligible = []
    rejection_counts = {"nonpositive_blend": 0, "production_unsafe": 0, "no_strict_geo_improvement": 0}
    for source in observations:
        item = dict(source)
        blend = float(item.get("blend", np.nan))
        geo_score = float(item.get("geo_score", np.nan))
        if not np.isfinite(blend) or blend <= 0.0:
            rejection_counts["nonpositive_blend"] += 1
        elif not _observation_passes_production_ucb(item):
            rejection_counts["production_unsafe"] += 1
        elif not np.isfinite(geo_score) or geo_score >= baseline_geo_score - tolerance:
            rejection_counts["no_strict_geo_improvement"] += 1
        else:
            eligible.append(item)

    if eligible:
        selected = min(eligible, key=lambda item: (float(item["geo_score"]), float(item["blend"])))
        reason = "best_safe_strict_geo_improvement"
    else:
        selected = baseline_items[0]
        reason = "fallback_lambda_zero"
    return selected, {
        "reason": reason,
        "selected_blend": float(selected["blend"]),
        "eligible_count": len(eligible),
        "rejection_counts": rejection_counts,
    }


def pilot_cache_key(*, iteration, blend, member_indices, state_hash):
    return sha256_json({
        "version": 1,
        "iteration": int(iteration),
        "blend": format(float(blend), ".17g"),
        "member_indices": [int(idx) for idx in member_indices],
        "state_hash": state_hash,
    })


class SafeBlendSearch:
    def __init__(
        self,
        *,
        max_nonbaseline_evaluations=5,
        n_initial_nonbaseline=2,
        pool_size=256,
        pool_seed=2026,
        random_state=2026,
        xi=0.0,
    ):
        if max_nonbaseline_evaluations < 1 or not 1 <= n_initial_nonbaseline <= max_nonbaseline_evaluations:
            raise ValueError("invalid evaluation budget")
        self.max_nonbaseline_evaluations = max_nonbaseline_evaluations
        self.n_initial_nonbaseline = n_initial_nonbaseline
        self.pool_size = pool_size
        self.pool_seed = pool_seed
        self.random_state = random_state
        self.xi = xi
        self._pool = sobol_blend_candidates(pool_size, seed=pool_seed)
        self._observations = []
        self._models = {}

    @property
    def n_observations(self):
        return len(self._observations)

    @property
    def n_nonbaseline_observations(self):
        return sum(item["blend"] > 0.0 for item in self._observations)

    @property
    def is_complete(self):
        return self.n_nonbaseline_observations >= self.max_nonbaseline_evaluations

    @property
    def observations(self):
        return copy.deepcopy(self._observations)

    @staticmethod
    def _blend_key(value):
        return format(float(value), ".15g")

    def tell(self, observation):
        item = copy.deepcopy(dict(observation))
        blend = float(item["blend"])
        if not 0.0 <= blend <= 1.0:
            raise ValueError("observation blend must be in [0, 1]")
        observed = {self._blend_key(existing["blend"]) for existing in self._observations}
        if self._blend_key(blend) in observed:
            raise ValueError("duplicate blend observation")
        self._observations.append(item)
        self._models = {}

    def _new_model(self, alpha_variance, *, offset):
        return build_matern_gp(
            alpha_variance,
            length_scale=0.25,
            length_scale_bounds=(1e-3, 10.0),
            random_state=self.random_state + offset,
        )

    def fit(self):
        if self.n_observations < 3:
            raise ValueError("baseline and two nonbaseline observations are required")
        x = np.asarray([item["blend"] for item in self._observations], dtype=float).reshape(-1, 1)
        objective_model = self._new_model(np.full(self.n_observations, 1e-12), offset=0)
        objective_model.fit(x, np.asarray([item["objective"] for item in self._observations], dtype=float))
        self._models = {"objective": objective_model}
        for offset, name in enumerate(CONSTRAINT_NAMES, start=1):
            se = np.asarray([item["constraint_noise"][name] for item in self._observations], dtype=float)
            model = self._new_model(np.square(se), offset=offset)
            model.fit(x, np.asarray([item["constraints"][name] for item in self._observations], dtype=float))
            self._models[name] = model

    def _unseen_nonbaseline(self):
        observed = {self._blend_key(item["blend"]) for item in self._observations}
        mask = np.asarray([v > 0.0 and self._blend_key(v) not in observed for v in self._pool], dtype=bool)
        return self._pool[mask], np.flatnonzero(mask)

    def ask(self):
        if sum(item["blend"] == 0.0 for item in self._observations) != 1:
            raise ValueError("tell exactly one lambda=0 baseline before ask")
        if self.is_complete:
            return None, {"mode": "fixed_budget_complete", "n_nonbaseline_observations": self.n_nonbaseline_observations}

        unseen, original_indices = self._unseen_nonbaseline()
        if unseen.size == 0:
            raise ValueError("candidate pool contains no unseen nonbaseline blends")

        if self.n_nonbaseline_observations < self.n_initial_nonbaseline:
            return float(unseen[0]), {
                "mode": "deterministic_initial",
                "candidate_index": int(original_indices[0]),
                "n_nonbaseline_observations": self.n_nonbaseline_observations,
            }
        if not self._models:
            self.fit()

        x = unseen.reshape(-1, 1)
        objective_mean, objective_std = self._models["objective"].predict(x, return_std=True)
        constraint_means, constraint_stds = [], []
        feasibility = np.ones(unseen.shape[0], dtype=float)
        for name in CONSTRAINT_NAMES:
            mean, std = self._models[name].predict(x, return_std=True)
            constraint_means.append(mean)
            constraint_stds.append(std)
            feasibility *= probability_nonpositive(mean, std)

        safe_mask = np.asarray([_observation_passes_production_ucb(item) for item in self._observations], dtype=bool)
        objectives = np.asarray([item["objective"] for item in self._observations], dtype=float)
        if np.any(safe_mask):
            best_feasible = float(np.min(objectives[safe_mask]))
            acquisition = constrained_expected_improvement(
                objective_mean, objective_std, constraint_means, constraint_stds, best_feasible, xi=self.xi
            )
            mode = "constrained_expected_improvement"
        else:
            best_feasible = None
            acquisition = feasibility
            mode = "feasibility_first"

        local_index = int(np.argmax(acquisition))
        return float(unseen[local_index]), {
            "mode": mode,
            "candidate_index": int(original_indices[local_index]),
            "acquisition": float(acquisition[local_index]),
            "predicted_objective_mean": float(objective_mean[local_index]),
            "predicted_objective_std": float(objective_std[local_index]),
            "predicted_feasibility": float(feasibility[local_index]),
            "best_feasible": best_feasible,
            "n_observations": self.n_observations,
            "n_nonbaseline_observations": self.n_nonbaseline_observations,
        }

    def select(self):
        baseline = [item for item in self._observations if item["blend"] == 0.0]
        if len(baseline) != 1:
            raise ValueError("exactly one baseline observation is required")
        return select_safe_blend(self._observations, baseline_geo_score=baseline[0]["geo_score"])

    def state_dict(self):
        return {
            "version": 1,
            "config": {
                "max_nonbaseline_evaluations": self.max_nonbaseline_evaluations,
                "n_initial_nonbaseline": self.n_initial_nonbaseline,
                "pool_size": self.pool_size,
                "pool_seed": self.pool_seed,
                "random_state": self.random_state,
                "xi": self.xi,
            },
            "observations": self.observations,
        }

    def save(self, path):
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(self.state_dict(), indent=2, sort_keys=True), encoding="utf-8")

    @classmethod
    def from_state_dict(cls, state):
        if state.get("version") != 1:
            raise ValueError("unsupported SafeBlendSearch state version")
        optimizer = cls(**state["config"])
        for obs in state["observations"]:
            optimizer.tell(obs)
        return optimizer

    @classmethod
    def load(cls, path):
        return cls.from_state_dict(json.loads(Path(path).read_text(encoding="utf-8")))
