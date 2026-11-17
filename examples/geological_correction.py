import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "simulation_snapshot"))

from gmr_refactor.src.gmr_config import GMRCorrectionConfig
from gmr_refactor.src.gmr_regularizer import GMRRegularizer


def to_jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def main():
    parser = argparse.ArgumentParser(
        description="Apply supplied GMR weights to prior fields without running ECLIPSE."
    )
    parser.add_argument("--members", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/geological_correction")
    args = parser.parse_args()

    if not 1 <= args.members <= 200:
        parser.error("--members must be between 1 and 200")

    torch.set_num_threads(2)
    torch.manual_seed(2026)

    model_root = ROOT / "simulation_snapshot/gmr_refactor"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    meta = json.loads((model_root / "data/prior_patch_dataset_meta.json").read_text(encoding="utf-8"))
    meta["output_npz"] = str(model_root / "data/prior_patch_dataset.npz")
    runtime_meta_path = args.output_dir / "runtime_dataset_meta.json"
    runtime_meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    with np.load(ROOT / "inputs/prior_ensemble_numeric.npz", allow_pickle=False) as data:
        perm = data["mumyperm"][:, :args.members].copy()
        poro = data["mumyporo"][:, :args.members].copy()
        permz = data["mumypermz"][:, :args.members].copy()

    t0 = time.perf_counter()
    regularizer = GMRRegularizer(
        str(model_root / "checkpoints/gmr_vae.pt"),
        str(model_root / "checkpoints/gmr_flow.pt"),
        str(runtime_meta_path),
        GMRCorrectionConfig(),
        args.device,
    )
    lambda_seq = json.loads(
        (model_root / "config/gmr_lambda_e011_frozen.json").read_text(encoding="utf-8")
    )["lambda_sequence"]

    corrected_perm, corrected_poro, diagnostics = regularizer.correct_ensemble(
        perm,
        poro,
        permz,
        iteration=1,
        total_iterations=4,
        force_lambda=lambda_seq[0],
    )
    elapsed = time.perf_counter() - t0

    if corrected_perm.shape != perm.shape or corrected_poro.shape != poro.shape:
        raise AssertionError("Corrected array shape mismatch")
    if not (np.isfinite(corrected_perm).all() and np.isfinite(corrected_poro).all()):
        raise AssertionError("Non-finite values in corrected output")

    np.savez_compressed(
        args.output_dir / "corrected_fields.npz",
        permx=corrected_perm,
        poro=corrected_poro,
    )

    report = {
        "checks": "PASS",
        "members": args.members,
        "field_shape": list(perm.shape),
        "elapsed_seconds": elapsed,
        "permx_mean_absolute_change": float(np.mean(np.abs(corrected_perm - perm))),
        "porosity_mean_absolute_change": float(np.mean(np.abs(corrected_poro - poro))),
        "round": 1,
        "blending_coefficient": lambda_seq[0],
        "simulator_executed": False,
        "weights_retrained": False,
        "diagnostics": diagnostics,
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2, default=to_jsonable) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({k: v for k, v in report.items() if k != "diagnostics"}, indent=2))


if __name__ == "__main__":
    main()
