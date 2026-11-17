import json
from pathlib import Path

import numpy as np

from .gmr_terminal_sequence_bo import lambda_sequence_hash, validate_lambda_sequence


SCHEMA_VERSION = 1
PROTOCOL_NAME = "stage_e_terminal_gmr_sequence"


def build_terminal_sequence_record(lambda_sequence, *, metadata=None):
    sequence = validate_lambda_sequence(lambda_sequence)
    return {
        "version": SCHEMA_VERSION,
        "protocol": PROTOCOL_NAME,
        "n_steps": int(sequence.size),
        "lambda_sequence": sequence.tolist(),
        "sequence_hash": lambda_sequence_hash(sequence),
        "metadata": dict(metadata or {}),
    }


def write_terminal_sequence_file(path, lambda_sequence, *, metadata=None):
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    record = build_terminal_sequence_record(lambda_sequence, metadata=metadata)
    output.write_text(json.dumps(record, indent=2, sort_keys=True), encoding="utf-8")
    return {**record, "sequence_file": str(output.resolve())}


def load_terminal_sequence_file(path, *, expected_n_steps=4):
    source = Path(path).expanduser().resolve()
    record = json.loads(source.read_text(encoding="utf-8"))
    if record.get("version") != SCHEMA_VERSION or record.get("protocol") != PROTOCOL_NAME:
        raise ValueError("unsupported GMR terminal sequence record")
    if int(record.get("n_steps", -1)) != expected_n_steps:
        raise ValueError("GMR terminal sequence length does not match assimilation count")
    sequence = validate_lambda_sequence(record["lambda_sequence"], n_steps=expected_n_steps)
    expected_hash = lambda_sequence_hash(sequence, n_steps=expected_n_steps)
    if record.get("sequence_hash") != expected_hash:
        raise ValueError("GMR terminal sequence hash mismatch")
    return {
        "version": SCHEMA_VERSION,
        "protocol": PROTOCOL_NAME,
        "n_steps": expected_n_steps,
        "lambda_sequence": sequence.tolist(),
        "sequence_hash": expected_hash,
        "metadata": dict(record.get("metadata", {})),
        "sequence_file": str(source),
    }


def sequence_blend_for_iteration(record, iteration):
    steps = int(record["n_steps"])
    sequence = validate_lambda_sequence(record["lambda_sequence"], n_steps=steps)
    index = iteration - 1
    if index < 0 or index >= steps:
        raise ValueError("iteration is outside the GMR terminal sequence")
    return float(sequence[index])


def apply_terminal_sequence_blend(
    base_perm,
    base_poro,
    base_permz,
    corrected_perm,
    corrected_poro,
    corrected_permz,
    *,
    iteration,
    record,
):
    blend = sequence_blend_for_iteration(record, iteration)
    if blend == 0.0:
        selected_perm, selected_poro, selected_permz = np.asarray(base_perm).copy(), np.asarray(base_poro).copy(), np.asarray(base_permz).copy()
    elif blend == 1.0:
        selected_perm, selected_poro, selected_permz = np.asarray(corrected_perm).copy(), np.asarray(corrected_poro).copy(), np.asarray(corrected_permz).copy()
    else:
        selected_perm = np.asarray(base_perm) + blend * (np.asarray(corrected_perm) - np.asarray(base_perm))
        selected_poro = np.asarray(base_poro) + blend * (np.asarray(corrected_poro) - np.asarray(base_poro))
        selected_permz = np.asarray(base_permz) + blend * (np.asarray(corrected_permz) - np.asarray(base_permz))

    diagnostics = {
        "guard_policy": "terminal_sequence",
        "iteration": iteration,
        "selected_blend": blend,
        "fallback_to_esmda": blend == 0.0,
        "sequence_hash": record["sequence_hash"],
        "sequence_file": record["sequence_file"],
        "lambda_sequence": list(record["lambda_sequence"]),
        "forward_simulations_executed": 0,
        "selection_source": "frozen_terminal_sequence",
    }
    return np.abs(selected_perm), np.abs(selected_poro), np.abs(selected_permz), diagnostics
