import argparse
import csv
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
METRICS = ("m_prod", "m_prod_ensemble_mean", "geo_score")
METHODS = ("standard", "apbo", "gmr", "full")


def read_csv_rows(path):
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def geometric_mean(values):
    return math.exp(sum(math.log(v) for v in values) / len(values))


def main():
    parser = argparse.ArgumentParser(
        description="Recompute frozen Table 2 and Figure 6D summary metrics."
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/metrics")
    args = parser.parse_args()

    rows = read_csv_rows(ROOT / "results/confirmatory_metrics.csv")
    if len(rows) != 20 or len({(r["seed"], r["method"]) for r in rows}) != 20:
        raise ValueError("Expected 20 unique (seed, method) rows in confirmatory_metrics.csv")

    summary = {}
    for method in METHODS:
        method_rows = [r for r in rows if r["method"] == method]
        summary[method] = {
            metric: geometric_mean([float(r[metric]) for r in method_rows])
            for metric in METRICS
        }

    for row in read_csv_rows(ROOT / "results/table2_recomputed.csv"):
        for metric in METRICS:
            if not math.isclose(summary[row["method"]][metric], float(row[metric]), abs_tol=1e-12):
                raise AssertionError(f"Mismatch in Table 2 for {row['method']}.{metric}")

    by_seed_method = {(r["seed"], r["method"]): r for r in rows}
    for row in read_csv_rows(ROOT / "figure_sources/figure6_panelD_HBO_GMR_vs_alphaBO.csv"):
        seed = row["seed"]
        for metric in METRICS:
            if seed == "geometric_mean":
                gain = 100.0 * (1.0 - summary["full"][metric] / summary["apbo"][metric])
            else:
                full_val = float(by_seed_method[(seed, "full")][metric])
                apbo_val = float(by_seed_method[(seed, "apbo")][metric])
                gain = 100.0 * (1.0 - full_val / apbo_val)
            if not math.isclose(gain, float(row[f"{metric}_improvement_pct"]), abs_tol=1e-12):
                raise AssertionError(f"Mismatch in Figure 6D for {seed}.{metric}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "table2.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["method", *METRICS])
        writer.writeheader()
        writer.writerows({"method": m, **summary[m]} for m in METHODS)

    relative_changes = [
        {
            "method": m,
            **{k: 100.0 * (summary[m][k] / summary["standard"][k] - 1.0) for k in METRICS},
        }
        for m in METHODS
    ]
    with (args.output_dir / "relative_changes.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["method", *METRICS])
        writer.writeheader()
        writer.writerows(relative_changes)

    report = {
        "checks": "PASS",
        "records": len(rows),
        "table2": summary,
        "relative_changes_percent": relative_changes,
        "simulator_executed": False,
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
