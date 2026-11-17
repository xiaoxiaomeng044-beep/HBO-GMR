import copy
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import qmc

from .apbo_constrained_bo import (
    build_matern_gp,
    constrained_expected_improvement,
    probability_nonpositive,
)


CONSTRAINT_NAMES = ("m_prod", "m_prod_ensemble_mean")
NUMERICAL_EPS = 1e-12


def validate_lambda_sequence(values, *, n_steps=4, atol=NUMERICAL_EPS):
    sequence = np.asarray(list(values), dtype=float)
    if sequence.ndim != 1 or sequence.size != n_steps or not np.all(np.isfinite(sequence)):
        raise ValueError(f"lambda sequence must contain {n_steps} finite values")
    if np.any(sequence < -abs(atol)) or np.any(sequence > 1.0 + abs(atol)):
        raise ValueError("lambda sequence values must be in [0, 1]")
    return np.clip(sequence, 0.0, 1.0)


def lambda_sequence_hash(values, *, n_steps=4, decimals=12):
    sequence = validate_lambda_sequence(values, n_steps=n_steps)
    rounded = np.round(sequence, decimals=decimals).astype("<f8", copy=False)
    return hashlib.sha256(rounded.tobytes()).hexdigest()


def sobol_lambda_sequences(n_candidates, *, n_steps=4, seed=2026, include_zero=True, include_full=True):
    if n_candidates < int(include_zero) + int(include_full) or n_candidates < 1 or n_steps < 1:
        raise ValueError("invalid Sobol lambda sequence pool parameters")

    candidates = []
    seen = set()

    def add(sequence):
        validated = validate_lambda_sequence(sequence, n_steps=n_steps)
        key = lambda_sequence_hash(validated, n_steps=n_steps)
        if key not in seen:
            seen.add(key)
            candidates.append(validated)

    if include_zero:
        add(np.zeros(n_steps, dtype=float))
    if include_full:
        add(np.ones(n_steps, dtype=float))

    exponent = int(math.ceil(math.log2(max(16, n_candidates * 2))))
    for row in qmc.Sobol(d=n_steps, scramble=True, seed=seed).random_base2(m=exponent):
        add(row)
        if len(candidates) >= n_candidates:
            break
    if len(candidates) < n_candidates:
        raise RuntimeError("unable to construct enough unique lambda sequences")
    return np.vstack(candidates[:n_candidates])


def _production_safe(observation):
    upper = observation.get("upper_constraints", {})
    return all(
        np.isfinite(float(upper.get(name, np.nan))) and float(upper[name]) <= NUMERICAL_EPS
        for name in CONSTRAINT_NAMES
    )


def make_terminal_sequence_observation(lambda_sequence, geo_score, baseline_geo_score, bootstrap_summary):
    sequence = validate_lambda_sequence(lambda_sequence)
    if not (np.isfinite(geo_score) and geo_score > 0.0 and np.isfinite(baseline_geo_score) and baseline_geo_score > 0.0):
        raise ValueError("geo scores must be finite and positive")

    point = bootstrap_summary["point_log_ratio"]
    upper = bootstrap_summary["upper_confidence"]
    se = bootstrap_summary["standard_error"]
    constraints, upper_constraints, constraint_noise = {}, {}, {}
    for name in CONSTRAINT_NAMES:
        val, uval, nval = float(point[name]), float(upper[name]), max(float(se[name]), 0.0)
        if not all(np.isfinite(x) for x in (val, uval, nval)):
            raise ValueError(f"invalid bootstrap evidence for {name}")
        constraints[name] = val
        upper_constraints[name] = uval
        constraint_noise[name] = nval

    observation = {
        "lambda_sequence": sequence.tolist(),
        "sequence_hash": lambda_sequence_hash(sequence),
        "geo_score": float(geo_score),
        "baseline_geo_score": float(baseline_geo_score),
        "objective": float(np.log(geo_score / baseline_geo_score)),
        "constraints": constraints,
        "upper_constraints": upper_constraints,
        "constraint_noise": constraint_noise,
        "bootstrap_index_hash": bootstrap_summary.get("bootstrap_index_hash", ""),
    }
    observation["production_safe"] = _production_safe(observation)
    return observation


def select_safe_terminal_sequence(observations, *, baseline_geo_score, geo_atol=NUMERICAL_EPS):
    if not np.isfinite(baseline_geo_score) or baseline_geo_score <= 0.0:
        raise ValueError("baseline_geo_score must be finite and positive")
    zero_hash = lambda_sequence_hash(np.zeros(4, dtype=float))
    baseline_items = [dict(item) for item in observations if item.get("sequence_hash") == zero_hash]
    if len(baseline_items) != 1:
        raise ValueError("exactly one all-zero baseline observation is required")

    tolerance = max(geo_atol, 0.0) * max(1.0, abs(baseline_geo_score))
    eligible = []
    rejection_counts = {"baseline": 0, "production_unsafe": 0, "no_strict_terminal_geo_improvement": 0}
    for source in observations:
        item = dict(source)
        seq = validate_lambda_sequence(item.get("lambda_sequence", []))
        if lambda_sequence_hash(seq) == zero_hash:
            rejection_counts["baseline"] += 1
        elif not _production_safe(item):
            rejection_counts["production_unsafe"] += 1
        elif float(item.get("geo_score", np.nan)) >= baseline_geo_score - tolerance:
            rejection_counts["no_strict_terminal_geo_improvement"] += 1
        else:
            eligible.append(item)

    if eligible:
        selected = min(
            eligible,
            key=lambda item: (float(item["geo_score"]), float(np.sum(item["lambda_sequence"])), item["sequence_hash"]),
        )
        reason = "best_safe_terminal_geo_improvement"
    else:
        selected = baseline_items[0]
        reason = "fallback_all_zero"
    return selected, {
        "reason": reason,
        "selected_sequence": list(selected["lambda_sequence"]),
        "selected_sequence_hash": selected["sequence_hash"],
        "eligible_count": len(eligible),
        "rejection_counts": rejection_counts,
    }


class TerminalSequenceBO:
    def __init__(
        self,
        *,
        n_steps=4,
        max_nonbaseline_evaluations=11,
        n_initial_nonbaseline=8,
        pool_size=4096,
        pool_seed=2026,
        random_state=2026,
        xi=0.0,
    ):
        if n_steps < 1 or max_nonbaseline_evaluations < 1 or not 1 <= n_initial_nonbaseline <= max_nonbaseline_evaluations:
            raise ValueError("invalid TerminalSequenceBO configuration")
        if pool_size <= max_nonbaseline_evaluations:
            raise ValueError("pool_size must exceed the evaluation budget")
        self.n_steps = n_steps
        self.max_nonbaseline_evaluations = max_nonbaseline_evaluations
        self.n_initial_nonbaseline = n_initial_nonbaseline
        self.pool_size = pool_size
        self.pool_seed = pool_seed
        self.random_state = random_state
        self.xi = xi
        self._pool = sobol_lambda_sequences(pool_size, n_steps=n_steps, seed=pool_seed)
        self._observations = []
        self._models = {}

    @property
    def n_observations(self):
        return len(self._observations)

    @property
    def n_nonbaseline_observations(self):
        zero_hash = lambda_sequence_hash(np.zeros(self.n_steps), n_steps=self.n_steps)
        return sum(item["sequence_hash"] != zero_hash for item in self._observations)

    @property
    def is_complete(self):
        return self.n_nonbaseline_observations >= self.max_nonbaseline_evaluations

    @property
    def observations(self):
        return copy.deepcopy(self._observations)

    def tell(self, observation):
        item = copy.deepcopy(dict(observation))
        sequence = validate_lambda_sequence(item["lambda_sequence"], n_steps=self.n_steps)
        seq_hash = lambda_sequence_hash(sequence, n_steps=self.n_steps)
        if item.get("sequence_hash") != seq_hash:
            raise ValueError("observation sequence hash is invalid")
        if seq_hash in {existing["sequence_hash"] for existing in self._observations}:
            raise ValueError("duplicate lambda sequence observation")
        self._observations.append(item)
        self._models = {}

    def _new_model(self, alpha_variance, *, offset):
        return build_matern_gp(
            alpha_variance,
            length_scale=np.ones(self.n_steps, dtype=float),
            length_scale_bounds=(1e-2, 10.0),
            random_state=self.random_state + offset,
        )

    def fit(self):
        if self.n_observations < 3:
            raise ValueError("baseline and two nonbaseline observations are required")
        x = np.vstack([validate_lambda_sequence(item["lambda_sequence"], n_steps=self.n_steps) for item in self._observations])
        objective_model = self._new_model(np.full(self.n_observations, 1e-12), offset=0)
        objective_model.fit(x, np.asarray([item["objective"] for item in self._observations], dtype=float))
        self._models = {"objective": objective_model}
        for offset, name in enumerate(CONSTRAINT_NAMES, start=1):
            noise = np.asarray([item["constraint_noise"][name] for item in self._observations], dtype=float)
            model = self._new_model(np.square(noise), offset=offset)
            model.fit(x, np.asarray([item["constraints"][name] for item in self._observations], dtype=float))
            self._models[name] = model

    def _unseen_nonbaseline(self):
        observed = {item["sequence_hash"] for item in self._observations}
        zero_hash = lambda_sequence_hash(np.zeros(self.n_steps), n_steps=self.n_steps)
        mask = np.asarray(
            [
                h != zero_hash and h not in observed
                for h in (lambda_sequence_hash(row, n_steps=self.n_steps) for row in self._pool)
            ],
            dtype=bool,
        )
        return self._pool[mask], np.flatnonzero(mask)

    def ask(self):
        zero_hash = lambda_sequence_hash(np.zeros(self.n_steps), n_steps=self.n_steps)
        if sum(item["sequence_hash"] == zero_hash for item in self._observations) != 1:
            raise ValueError("tell exactly one all-zero baseline before ask")
        if self.is_complete:
            return None, {"mode": "fixed_budget_complete", "n_nonbaseline_observations": self.n_nonbaseline_observations}

        unseen, original_indices = self._unseen_nonbaseline()
        if unseen.size == 0:
            raise ValueError("candidate pool has no unseen nonbaseline sequences")

        if self.n_nonbaseline_observations < self.n_initial_nonbaseline:
            return unseen[0].copy(), {
                "mode": "deterministic_initial",
                "candidate_index": int(original_indices[0]),
                "n_nonbaseline_observations": self.n_nonbaseline_observations,
            }
        if not self._models:
            self.fit()

        objective_mean, objective_std = self._models["objective"].predict(unseen, return_std=True)
        constraint_means, constraint_stds = [], []
        feasibility = np.ones(unseen.shape[0], dtype=float)
        for name in CONSTRAINT_NAMES:
            mean, std = self._models[name].predict(unseen, return_std=True)
            constraint_means.append(mean)
            constraint_stds.append(std)
            feasibility *= probability_nonpositive(mean, std)

        safe_mask = np.asarray([_production_safe(item) for item in self._observations], dtype=bool)
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
        return unseen[local_index].copy(), {
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
        return select_safe_terminal_sequence(
            self._observations,
            baseline_geo_score=[item for item in self._observations if np.allclose(item["lambda_sequence"], 0.0)][0]["geo_score"],
        )

    def state_dict(self):
        return {
            "version": 1,
            "config": {
                "n_steps": self.n_steps,
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
            raise ValueError("unsupported TerminalSequenceBO state version")
        optimizer = cls(**state["config"])
        for obs in state["observations"]:
            optimizer.tell(obs)
        return optimizer

    @classmethod
    def load(cls, path):
        return cls.from_state_dict(json.loads(Path(path).read_text(encoding="utf-8")))
