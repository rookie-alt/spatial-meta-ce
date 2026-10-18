"""Descriptive validation/holdout report; no candidate selection."""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

from scripts import meta65_holdout_verify as verifier


def _mean(values: list[float | None]) -> float | None:
    if not values or any(value is None for value in values):
        return None
    if any(not math.isfinite(value) for value in values):
        raise ValueError("Report means require finite values")
    return float(math.fsum(values) / len(values))


def _q(values: dict[str, np.ndarray]) -> np.ndarray:
    true = values["true"]
    with np.errstate(over="raise", invalid="raise"):
        return np.exp(np.abs(np.maximum(values["log_pred"], 0) - np.log(np.maximum(true, 1))))


def build_report(output: Path | str) -> dict:
    root = verifier.checked_path(output)
    verifier.verify_cells(root, formal=True)
    cells_doc = verifier.read_json(root / "cells.json")
    cells = cells_doc["cells"]
    index = {(c["operator"], c["database"], c["run"]): c for c in cells}
    groups = {"validation": verifier.VALIDATION, "holdout": verifier.HOLDOUT}
    report_groups: dict[str, object] = {}
    for group_name, databases in groups.items():
        operators: dict[str, object] = {}
        for operator in verifier.OPERATORS:
            per_database = {}
            for database in databases:
                per_database[database] = {
                    f"{bucket}_{metric}": _mean([
                        index[(operator, database, run)]["metrics"][bucket][metric]
                        for run in range(5)
                    ])
                    for bucket in ("all", "4+", "low4")
                    for metric in ("q50", "q95")
                }
            operators[operator] = {
                "per_database": per_database,
                "macro": {
                    key: _mean([per_database[database][key] for database in databases])
                    for key in per_database[databases[0]]
                },
                "cell_count": len(databases) * 5,
            }
        report_groups[group_name] = operators

    paired: dict[str, object] = {}
    for group_name, databases in groups.items():
        paired[group_name] = {}
        for database in databases:
            runs = []
            for run in range(5):
                baseline = verifier.prediction_arrays(
                    root, index[("none", database, run)]
                )
                treatment = verifier.prediction_arrays(
                    root, index[("head3", database, run)]
                )
                if not np.array_equal(baseline["true"], treatment["true"]):
                    raise ValueError("Paired cells have different truth")
                left, right = _q(treatment), _q(baseline)
                runs.append({
                    "run": run,
                    "q95_ratio": float(np.percentile(left, 95) / np.percentile(right, 95)),
                    "improved_fraction": float(np.mean(left < right)),
                    "mean_delta_log_q": float(np.mean(np.log(np.maximum(left, 1))
                                                       - np.log(np.maximum(right, 1)))),
                })
            paired[group_name][database] = {
                key: _mean([run[key] for run in runs])
                for key in ("q95_ratio", "improved_fraction", "mean_delta_log_q")
            }
    return {
        "schema": 1,
        "experiment": verifier.EXPERIMENT,
        "status": "baseline_verified",
        "selection_performed": False,
        "selected_arm": None,
        "groups": report_groups,
        "paired": paired,
        "scope": "source-only single-seed K20 descriptive baseline; no candidate selection",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = build_report(args.output)
    print(f"report: {result['status']} -> {args.output / 'report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
