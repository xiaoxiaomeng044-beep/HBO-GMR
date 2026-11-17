import argparse
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "simulation_snapshot"))

from gmr_refactor.src.apbo_constrained_bo import constrained_expected_improvement
from gmr_refactor.src.apbo_schedule_space import (
    budget_to_alpha,
    sobol_budget_candidates,
    validate_budget_shares,
)


def main():
    parser = argparse.ArgumentParser(
        description="Check frozen schedules and constrained acquisition on synthetic predictions."
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/schedule")
    args = parser.parse_args()

    cfg_dir = ROOT / "simulation_snapshot/gmr_refactor/config"
    schedules = {}
    for name in ("stage_f_standard_equal_frozen.json", "stage_c2_apbo_c010_frozen.json"):
        record = json.loads((cfg_dir / name).read_text(encoding="utf-8"))
        shares = validate_budget_shares(record["budget_shares"])
        alpha = budget_to_alpha(shares)
        if not np.allclose(alpha, record["alpha_schedule"], rtol=0.0, atol=1e-10):
            raise AssertionError(f"Alpha schedule mismatch in {name}")
        schedules[name] = {
            "alpha": alpha.tolist(),
            "inverse_alpha_sum": float(np.sum(1.0 / alpha)),
        }

    pool = sobol_budget_candidates(32, seed=2026)
    for row in pool:
        validate_budget_shares(row)

    mu = np.array([-0.2, 0.1, -0.1])
    std = np.array([0.05, 0.1, 0.05])
    constraint_mu = np.array([-0.1, -0.1, 0.2])
    constraint_std = np.full(3, 0.05)
    scores = constrained_expected_improvement(mu, std, [constraint_mu], [constraint_std], 0.0)
    if not (np.isfinite(scores).all() and np.all(scores >= 0.0)):
        raise AssertionError("Invalid acquisition scores")

    report = {
        "checks": "PASS",
        "schedules": schedules,
        "candidate_count": len(pool),
        "synthetic_prediction_example": True,
        "constrained_EI": scores.tolist(),
        "selected_synthetic_candidate": int(np.argmax(scores)),
        "simulator_executed": False,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
