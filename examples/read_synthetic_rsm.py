import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "simulation_snapshot/gmr_refactor/core"))

from getsimulated import getsimulated


def write_toy_rsm(path):
    blocks = [
        (["TIME", "YEARS", "FOE", "WBHP", "WBHP"], [[0, 0, 0.1, 200, 210], [30, 0.08, 0.2, 190, 205], [60, 0.16, 0.3, 180, 200]]),
        (["TIME", "WGOR", "WGOR", "WWCT"], [[0, 10, 20, 0.01], [30, 11, 21, 0.02], [60, 12, 22, 0.03]]),
        (["TIME", "WWCT"], [[0, 0.04], [30, 0.05], [60, 0.06]]),
    ]
    lines = []
    for labels, rows in blocks:
        lines.extend(["SUMMARY OF RUN TOY", " ".join(labels), "units", "wells", ""])
        lines.extend(" ".join(map(str, row)) for row in rows)
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    with tempfile.TemporaryDirectory() as scratch:
        member_dir = Path(scratch) / "MASTER00001"
        member_dir.mkdir()
        write_toy_rsm(member_dir / "MASTER0.RSM")

        prev_cwd = Path.cwd()
        try:
            os.chdir(scratch)
            actual = getsimulated(0, 1, 3, n_obs=7, n_producers=2)
        finally:
            os.chdir(prev_cwd)

    expected = np.array([
        [0.1, 200, 210, 10, 20, 0.01, 0.04],
        [0.2, 190, 205, 11, 21, 0.02, 0.05],
        [0.3, 180, 200, 12, 22, 0.03, 0.06],
    ])[:, :, None]
    if not np.array_equal(actual, expected):
        raise AssertionError("Parsed RSM array does not match expected values")

    report = {
        "checks": "PASS",
        "shape": list(actual.shape),
        "dataset": "generated toy data",
        "simulator_executed": False,
    }
    out_dir = ROOT / "outputs/rsm_reader"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
