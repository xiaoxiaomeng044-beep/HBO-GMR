import json
from pathlib import Path

import numpy as np
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

from .apbo_schedule_space import schedule_hash, validate_budget_shares


CONSTRAINT_NAMES = ("constraint_prod_ens", "constraint_geo")


def expected_improvement_minimize(mean, std, best_feasible, *, xi=0.0):
    mu = np.asarray(mean, dtype=float)
    sigma = np.asarray(std, dtype=float)
    if mu.shape != sigma.shape:
        raise ValueError("mean and std shapes must match")
    improvement = best_feasible - mu - xi
    output = np.maximum(improvement, 0.0)
    active = sigma > 1e-14
    z = np.zeros_like(mu)
    z[active] = improvement[active] / sigma[active]
    output[active] = improvement[active] * norm.cdf(z[active]) + sigma[active] * norm.pdf(z[active])
    return np.maximum(output, 0.0)


def probability_nonpositive(mean, std):
    mu = np.asarray(mean, dtype=float)
    sigma = np.asarray(std, dtype=float)
    if mu.shape != sigma.shape:
        raise ValueError("mean and std shapes must match")
    prob = np.zeros_like(mu)
    active = sigma > 1e-14
    prob[active] = norm.cdf(-mu[active] / sigma[active])
    prob[~active] = (mu[~active] <= 0.0).astype(float)
    return np.clip(prob, 0.0, 1.0)


def constrained_expected_improvement(objective_mean, objective_std, constraint_means, constraint_stds, best_feasible, *, xi=0.0):
    if len(constraint_means) != len(constraint_stds):
        raise ValueError("constraint mean/std collections must match")
    acq = expected_improvement_minimize(objective_mean, objective_std, best_feasible, xi=xi)
    for mean, std in zip(constraint_means, constraint_stds):
        acq *= probability_nonpositive(mean, std)
    return acq


def build_matern_gp(alpha_variance, length_scale, length_scale_bounds, random_state):
    kernel = (
        ConstantKernel(1.0, (1e-3, 1e3))
        * Matern(length_scale=length_scale, length_scale_bounds=length_scale_bounds, nu=2.5)
        + WhiteKernel(noise_level=1e-6, noise_level_bounds=(1e-10, 1e-1))
    )
    return GaussianProcessRegressor(
        kernel=kernel,
        alpha=np.maximum(np.asarray(alpha_variance, dtype=float), 1e-12),
        normalize_y=True,
        n_restarts_optimizer=0,
        random_state=random_state,
    )


class ConstrainedScheduleBO:
    def __init__(self, *, n_steps=4, lower=0.125, upper=0.5, random_state=2026, xi=0.0):
        if n_steps < 2:
            raise ValueError("n_steps must be at least two")
        self.n_steps = n_steps
        self.lower = lower
        self.upper = upper
        self.random_state = random_state
        self.xi = xi
        self._budgets = []
        self._objective = []
        self._constraints = {name: [] for name in CONSTRAINT_NAMES}
        self._objective_noise = []
        self._constraint_noise = {name: [] for name in CONSTRAINT_NAMES}
        self._models = {}

    @property
    def n_observations(self):
        return len(self._budgets)

    def _features(self, budgets):
        array = np.asarray(budgets, dtype=float)
        if array.ndim == 1:
            array = array.reshape(1, -1)
        if array.ndim != 2 or array.shape[1] != self.n_steps:
            raise ValueError("candidate schedules have an invalid shape")
        return array[:, :-1]

    def _new_model(self, alpha_variance, offset):
        return build_matern_gp(
            alpha_variance,
            length_scale=np.ones(self.n_steps - 1, dtype=float),
            length_scale_bounds=(1e-2, 10.0),
            random_state=self.random_state + offset,
        )

    def tell(self, budget_shares, objective, constraints, *, objective_noise=0.0, constraint_noise=None):
        budget = validate_budget_shares(budget_shares, lower=self.lower, upper=self.upper)
        if not np.isfinite(objective):
            raise ValueError("objective must be finite")
        if schedule_hash(budget) in {schedule_hash(item) for item in self._budgets}:
            raise ValueError("duplicate schedule observation")

        noise_map = constraint_noise or {}
        self._budgets.append(budget)
        self._objective.append(float(objective))
        self._objective_noise.append(max(float(objective_noise), 0.0))
        for name in CONSTRAINT_NAMES:
            value = float(constraints[name])
            noise = max(float(noise_map.get(name, 0.0)), 0.0)
            if not np.isfinite(value) or not np.isfinite(noise):
                raise ValueError(f"invalid value/noise for {name}")
            self._constraints[name].append(value)
            self._constraint_noise[name].append(noise)
        self._models = {}

    def fit(self):
        if self.n_observations < 2:
            raise ValueError("at least two observations are required to fit the GP")
        x = self._features(np.vstack(self._budgets))
        objective_model = self._new_model(np.square(self._objective_noise), offset=0)
        objective_model.fit(x, np.asarray(self._objective, dtype=float))
        self._models = {"objective": objective_model}
        for offset, name in enumerate(CONSTRAINT_NAMES, start=1):
            model = self._new_model(np.square(self._constraint_noise[name]), offset=offset)
            model.fit(x, np.asarray(self._constraints[name], dtype=float))
            self._models[name] = model

    def ask(self, candidates):
        pool = np.vstack([validate_budget_shares(row, lower=self.lower, upper=self.upper) for row in candidates])
        observed = {schedule_hash(item) for item in self._budgets}
        unseen_mask = np.asarray([schedule_hash(item) not in observed for item in pool], dtype=bool)
        if not np.any(unseen_mask):
            raise ValueError("candidate pool contains no unseen schedules")
        unseen = pool[unseen_mask]
        original_indices = np.flatnonzero(unseen_mask)
        if self.n_observations < 2:
            return unseen[0].copy(), {"mode": "deterministic_initial", "candidate_index": int(original_indices[0])}
        if not self._models:
            self.fit()

        x = self._features(unseen)
        objective_mean, objective_std = self._models["objective"].predict(x, return_std=True)
        constraint_means, constraint_stds = [], []
        feasibility = np.ones(unseen.shape[0], dtype=float)
        for name in CONSTRAINT_NAMES:
            mean, std = self._models[name].predict(x, return_std=True)
            constraint_means.append(mean)
            constraint_stds.append(std)
            feasibility *= probability_nonpositive(mean, std)

        observed_constraints = np.column_stack([self._constraints[name] for name in CONSTRAINT_NAMES])
        feasible_mask = np.all(observed_constraints <= 0.0, axis=1)
        if np.any(feasible_mask):
            best_feasible = float(np.min(np.asarray(self._objective, dtype=float)[feasible_mask]))
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
        }

    def state_dict(self):
        observations = [
            {
                "budget_shares": budget.tolist(),
                "objective": self._objective[i],
                "constraints": {name: self._constraints[name][i] for name in CONSTRAINT_NAMES},
                "objective_noise": self._objective_noise[i],
                "constraint_noise": {name: self._constraint_noise[name][i] for name in CONSTRAINT_NAMES},
            }
            for i, budget in enumerate(self._budgets)
        ]
        return {
            "version": 1,
            "config": {
                "n_steps": self.n_steps,
                "lower": self.lower,
                "upper": self.upper,
                "random_state": self.random_state,
                "xi": self.xi,
            },
            "observations": observations,
        }

    def save(self, path):
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(self.state_dict(), indent=2, sort_keys=True), encoding="utf-8")

    @classmethod
    def load(cls, path):
        state = json.loads(Path(path).read_text(encoding="utf-8"))
        if state.get("version") != 1:
            raise ValueError("unsupported optimizer state version")
        optimizer = cls(**state["config"])
        for obs in state.get("observations", []):
            optimizer.tell(
                obs["budget_shares"],
                obs["objective"],
                obs["constraints"],
                objective_noise=obs.get("objective_noise", 0.0),
                constraint_noise=obs.get("constraint_noise", {}),
            )
        return optimizer
