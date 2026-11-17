import json
import os

import numpy as np

from .common import sha256_json


SCHEMA_VERSION = 1
LEVEL_SIZES = {"L1": 20, "L2": 50, "L3": 200}


def _manifest_hash(source_size, selection_seed, permutation):
    return sha256_json({
        "source_ensemble_size": source_size,
        "selection_seed": selection_seed,
        "permutation": list(permutation),
    })


def build_fidelity_manifest(*, source_ensemble_size=200, selection_seed=2026):
    if source_ensemble_size != LEVEL_SIZES["L3"]:
        raise ValueError("fidelity manifest requires a 200-member source ensemble")
    permutation = np.random.default_rng(selection_seed).permutation(source_ensemble_size).tolist()
    levels = {name: permutation[:size] for name, size in LEVEL_SIZES.items()}
    return {
        "schema_version": SCHEMA_VERSION,
        "source_ensemble_size": source_ensemble_size,
        "selection_seed": selection_seed,
        "selection_method": "seeded_permutation_prefix",
        "level_sizes": dict(LEVEL_SIZES),
        "levels": levels,
        "permutation": permutation,
        "manifest_hash": _manifest_hash(source_ensemble_size, selection_seed, permutation),
    }


def validate_fidelity_manifest(record):
    if record.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported fidelity manifest schema_version")
    source_size = int(record["source_ensemble_size"])
    seed = int(record["selection_seed"])
    permutation = [int(v) for v in record["permutation"]]
    if source_size != 200 or sorted(permutation) != list(range(source_size)):
        raise ValueError("fidelity manifest must contain a valid 200-member permutation")
    levels = record["levels"]
    for name, size in LEVEL_SIZES.items():
        if [int(v) for v in levels.get(name, [])] != permutation[:size]:
            raise ValueError(f"{name} must be the corresponding permutation prefix")
    if record.get("manifest_hash") != _manifest_hash(source_size, seed, permutation):
        raise ValueError("fidelity manifest hash mismatch")
    return build_fidelity_manifest(source_ensemble_size=source_size, selection_seed=seed)


def load_fidelity_manifest(path):
    resolved = os.path.abspath(os.path.expanduser(path))
    with open(resolved, "r", encoding="utf-8") as f:
        record = validate_fidelity_manifest(json.load(f))
    record["manifest_file"] = resolved
    return record


def select_fidelity_members(record, level):
    normalized = validate_fidelity_manifest(record)
    name = level.upper()
    if name not in LEVEL_SIZES:
        raise ValueError(f"unknown fidelity level: {level}")
    return list(normalized["levels"][name])


def write_fidelity_manifest(record, path):
    normalized = validate_fidelity_manifest(record)
    resolved = os.path.abspath(os.path.expanduser(path))
    os.makedirs(os.path.dirname(resolved), exist_ok=True)
    with open(resolved, "w", encoding="utf-8") as f:
        json.dump(normalized, f, indent=2, ensure_ascii=True)
        f.write("\n")
    return resolved
