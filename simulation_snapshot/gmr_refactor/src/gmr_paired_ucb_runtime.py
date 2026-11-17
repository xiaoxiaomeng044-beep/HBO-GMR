import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile

import numpy as np


CACHE_SCHEMA_VERSION = 1


def _sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def ensemble_state_hash(named_arrays):
    if not named_arrays:
        raise ValueError("at least one state array is required")
    digest = hashlib.sha256()
    for name in sorted(named_arrays):
        array = np.ascontiguousarray(np.asarray(named_arrays[name], dtype=np.float32))
        if array.size == 0 or not np.all(np.isfinite(array)):
            raise ValueError(f"state array {name!r} must be finite and nonempty")
        digest.update(name.encode("utf-8"))
        digest.update(json.dumps(list(array.shape)).encode("ascii"))
        digest.update(array.tobytes())
    return digest.hexdigest()


def build_production_response_payload(sim_by_variable, truth_by_variable, *, member_indices, source_member_indices):
    local = [int(v) for v in member_indices]
    source = [int(v) for v in source_member_indices]
    if not local or len(local) != len(source) or len(set(local)) != len(local) or len(set(source)) != len(source):
        raise ValueError("member identities must be paired, unique, and nonempty")

    variables = sorted(set(sim_by_variable).intersection(truth_by_variable))
    if not variables:
        raise ValueError("simulation and truth responses have no shared variables")

    simulations, truth = {}, {}
    for name in variables:
        sim = np.asarray(sim_by_variable[name], dtype=float)
        obs = np.asarray(truth_by_variable[name], dtype=float).reshape(-1)
        if sim.ndim != 2 or sim.shape[1] != len(local) or sim.shape[0] < 1 or obs.size < 1:
            raise ValueError(f"invalid production response shape for {name}")
        if not (np.all(np.isfinite(sim)) and np.all(np.isfinite(obs))):
            raise ValueError(f"nonfinite production response for {name}")
        simulations[name] = np.ascontiguousarray(sim, dtype=float)
        truth[name] = np.ascontiguousarray(obs, dtype=float)

    return {
        "variables": variables,
        "member_indices": local,
        "source_member_indices": source,
        "sim_by_variable": simulations,
        "truth_by_variable": truth,
    }


def evaluate_response_payload(payload, sampled_positions, metric_function):
    positions = np.asarray(sampled_positions, dtype=int).reshape(-1)
    member_count = len(payload["member_indices"])
    if positions.size < 1 or np.any(positions < 0) or np.any(positions >= member_count):
        raise ValueError("bootstrap member positions are out of range")
    simulations = {name: np.asarray(vals, dtype=float)[:, positions] for name, vals in payload["sim_by_variable"].items()}
    truth = {name: np.asarray(vals, dtype=float) for name, vals in payload["truth_by_variable"].items()}
    return metric_function(simulations, truth)


def _payload_to_npz(payload, path):
    arrays = {
        "member_indices": np.asarray(payload["member_indices"], dtype=np.int64),
        "source_member_indices": np.asarray(payload["source_member_indices"], dtype=np.int64),
    }
    for name in payload["variables"]:
        arrays[f"sim__{name}"] = np.asarray(payload["sim_by_variable"][name], dtype=float)
        arrays[f"truth__{name}"] = np.asarray(payload["truth_by_variable"][name], dtype=float)
    np.savez_compressed(path, **arrays)


def _payload_from_npz(path, variables):
    with np.load(path, allow_pickle=False) as data:
        simulations = {name: np.asarray(data[f"sim__{name}"], dtype=float) for name in variables}
        truth = {name: np.asarray(data[f"truth__{name}"], dtype=float) for name in variables}
        local = np.asarray(data["member_indices"], dtype=int).tolist()
        source = np.asarray(data["source_member_indices"], dtype=int).tolist()
    return build_production_response_payload(simulations, truth, member_indices=local, source_member_indices=source)


def load_pilot_response_cache(cache_dir, *, expected_cache_key=None, expected_state_hash=None):
    directory = Path(cache_dir)
    metadata_path = directory / "cache_metadata.json"
    response_path = directory / "production_responses.npz"
    if not metadata_path.is_file() or not response_path.is_file():
        raise FileNotFoundError(f"incomplete pilot cache: {directory}")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("schema_version") != CACHE_SCHEMA_VERSION:
        raise ValueError("unsupported pilot cache schema")
    if expected_cache_key is not None and metadata.get("cache_key") != expected_cache_key:
        raise ValueError("pilot cache key mismatch")
    if expected_state_hash is not None and metadata.get("state_hash") != expected_state_hash:
        raise ValueError("pilot cache state hash mismatch")
    if _sha256_file(response_path) != metadata.get("response_sha256"):
        raise ValueError("pilot cache response hash mismatch")

    payload = _payload_from_npz(response_path, metadata["variables"])
    if payload["member_indices"] != metadata["member_indices"] or payload["source_member_indices"] != metadata["source_member_indices"]:
        raise ValueError("pilot cache member identities changed")

    rsm_paths = {}
    for record in metadata["rsm_files"]:
        member_index = int(record["member_index"])
        path = directory / record["relative_path"]
        if not path.is_file() or _sha256_file(path) != record["sha256"]:
            raise ValueError(f"pilot cache RSM hash mismatch for member {member_index}")
        rsm_paths[member_index] = str(path)
    if set(rsm_paths) != set(payload["member_indices"]):
        raise ValueError("pilot cache does not contain every member RSM")
    return {"cache_dir": str(directory), "metadata": metadata, "payload": payload, "rsm_paths": rsm_paths}


def write_pilot_response_cache(cache_dir, *, cache_key, state_hash, iteration, blend, payload, rsm_paths):
    destination = Path(cache_dir)
    if destination.exists():
        return load_pilot_response_cache(destination, expected_cache_key=cache_key, expected_state_hash=state_hash)

    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}_", dir=str(destination.parent)))
    try:
        response_path = staging / "production_responses.npz"
        _payload_to_npz(payload, response_path)
        rsm_dir = staging / "rsm"
        rsm_dir.mkdir()
        rsm_records = []
        expected_members = list(payload["member_indices"])
        if set(rsm_paths) != set(expected_members):
            raise ValueError("RSM paths do not match payload member identities")
        for member_index in expected_members:
            source = Path(rsm_paths[member_index])
            relative = Path("rsm") / f"member_{member_index:05d}.RSM"
            target = staging / relative
            shutil.copy2(source, target)
            rsm_records.append({
                "member_index": member_index,
                "relative_path": relative.as_posix(),
                "sha256": _sha256_file(target),
            })
        metadata = {
            "schema_version": CACHE_SCHEMA_VERSION,
            "cache_key": cache_key,
            "state_hash": state_hash,
            "iteration": iteration,
            "blend": float(blend),
            "variables": payload["variables"],
            "member_indices": expected_members,
            "source_member_indices": payload["source_member_indices"],
            "response_sha256": _sha256_file(response_path),
            "rsm_files": rsm_records,
        }
        (staging / "cache_metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(staging, destination)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return load_pilot_response_cache(destination, expected_cache_key=cache_key, expected_state_hash=state_hash)


def restore_pilot_rsm_files(cache_record, target_paths):
    cached = {int(idx): Path(path) for idx, path in cache_record["rsm_paths"].items()}
    targets = {int(idx): Path(path) for idx, path in target_paths.items()}
    if set(cached) != set(targets):
        raise ValueError("restore targets do not match cached pilot members")
    restored = []
    for member_index in sorted(cached):
        targets[member_index].parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(cached[member_index], targets[member_index])
        restored.append(member_index)
    return restored
