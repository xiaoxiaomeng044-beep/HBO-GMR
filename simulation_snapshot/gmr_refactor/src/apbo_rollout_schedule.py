import json
import os
import tempfile

import numpy as np

from .apbo_schedule_space import (
    alpha_to_budget,
    budget_to_alpha,
    schedule_hash,
    validate_budget_shares,
)


SCHEMA_VERSION = 1
SCHEDULE_MODE = "terminal_simplex"


def build_schedule_record(budget_shares, *, source, selection_fidelity, metadata=None):
    shares = validate_budget_shares(budget_shares)
    alpha = budget_to_alpha(shares)
    return {
        "schema_version": SCHEMA_VERSION,
        "mode": SCHEDULE_MODE,
        "n_assim": int(shares.size),
        "budget_shares": shares.tolist(),
        "alpha_schedule": alpha.tolist(),
        "budget_sum": float(np.sum(shares)),
        "inverse_alpha_sum": float(np.sum(1.0 / alpha)),
        "schedule_hash": schedule_hash(shares),
        "source": source,
        "selection_fidelity": selection_fidelity,
        "metadata": dict(metadata or {}),
    }


def validate_schedule_record(record, *, expected_n_assim=None):
    if record.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"unsupported schedule schema_version: {record.get('schema_version')}")
    if record.get("mode") != SCHEDULE_MODE:
        raise ValueError(f"schedule mode must be {SCHEDULE_MODE!r}")

    shares = validate_budget_shares(record["budget_shares"])
    n_assim = int(record["n_assim"])
    if n_assim != shares.size:
        raise ValueError("n_assim does not match budget_shares length")
    if expected_n_assim is not None and n_assim != expected_n_assim:
        raise ValueError(f"schedule has {n_assim} assimilation steps; expected {expected_n_assim}")

    alpha = np.asarray(record["alpha_schedule"], dtype=float)
    if alpha.shape != shares.shape or not np.allclose(alpha, budget_to_alpha(shares), atol=1e-10, rtol=0.0):
        raise ValueError("alpha_schedule is inconsistent with budget_shares")
    if not np.allclose(alpha_to_budget(alpha), shares, atol=1e-10, rtol=0.0):
        raise ValueError("alpha_schedule does not close the ES-MDA budget")
    if record.get("schedule_hash") != schedule_hash(shares):
        raise ValueError("schedule_hash does not match budget_shares")

    return build_schedule_record(
        shares,
        source=record.get("source", ""),
        selection_fidelity=record.get("selection_fidelity", ""),
        metadata=record.get("metadata", {}),
    )


def load_schedule_file(path, *, expected_n_assim=None):
    resolved = os.path.abspath(os.path.expanduser(path))
    with open(resolved, "r", encoding="utf-8") as f:
        record = validate_schedule_record(json.load(f), expected_n_assim=expected_n_assim)
    record["schedule_file"] = resolved
    return record


def write_schedule_file(record, path):
    normalized = validate_schedule_record(record)
    resolved = os.path.abspath(os.path.expanduser(path))
    parent = os.path.dirname(resolved)
    os.makedirs(parent, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=".schedule_", suffix=".json", dir=parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(normalized, f, indent=2, ensure_ascii=True)
            f.write("\n")
        os.replace(tmp_path, resolved)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
    return resolved
