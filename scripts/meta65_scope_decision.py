"""Frozen source-only candidate gates for the Meta65 training-scope screen."""

from __future__ import annotations

import math
from typing import Any

ARMS = ("R_head3", "T_all3")
OPERATORS = ("none", "head3")
DATABASES = ("chembl", "legalacts", "telstra")
BUCKETS = ("all", "4+", "low4")
CELL_FIELDS = {"arm", "operator", "database", "run", "k", "metrics"}
MIN_BUCKET_COUNT = 30
MIN_GAIN = 0.10
MAX_REGRESSION_FACTOR = 1.10


def _blocked(reason: str) -> dict[str, Any]:
    return {"status": "blocked", "selected_arm": None, "reason": reason}


def _valid_metric(metric: object) -> bool:
    if not isinstance(metric, dict) or set(metric) != {"count", "q50", "q95"}:
        return False
    if type(metric["count"]) is not int or metric["count"] <= 0:
        return False
    values = [metric["q50"], metric["q95"]]
    if any(
        type(v) not in (int, float) or not math.isfinite(v) or v < 1 for v in values
    ):
        return False
    return metric["q50"] <= metric["q95"]


def _index(cells: object) -> dict[tuple, dict] | None:
    if not isinstance(cells, list) or len(cells) != 60:
        return None
    result = {}
    for cell in cells:
        if not isinstance(cell, dict) or set(cell) != CELL_FIELDS:
            return None
        if (
            cell["arm"] not in ARMS
            or cell["operator"] not in OPERATORS
            or cell["database"] not in DATABASES
            or type(cell["run"]) is not int
            or not 0 <= cell["run"] < 5
            or type(cell["k"]) is not int
            or cell["k"] != 20
        ):
            return None
        metrics = cell["metrics"]
        if not isinstance(metrics, dict) or set(metrics) != set(BUCKETS):
            return None
        if not all(_valid_metric(metrics[b]) for b in BUCKETS):
            return None
        if (
            not MIN_BUCKET_COUNT
            <= metrics["low4"]["count"]
            <= metrics["4+"]["count"]
            <= metrics["all"]["count"]
        ):
            return None
        key = (cell["arm"], cell["operator"], cell["database"], cell["run"])
        if key in result:
            return None
        result[key] = cell
    for db in DATABASES:
        for run in range(5):
            for bucket in BUCKETS:
                counts = {
                    result[a, op, db, run]["metrics"][bucket]["count"]
                    for a in ARMS
                    for op in OPERATORS
                }
                if len(counts) != 1:
                    return None
    return result


def _groups(indexed: dict[tuple, dict]) -> dict[str, Any]:
    return {
        arm: {
            op: {
                db: {
                    bucket: {
                        metric: math.fsum(
                            indexed[arm, op, db, run]["metrics"][bucket][metric] / 5
                            for run in range(5)
                        )
                        for metric in ("q50", "q95")
                    }
                    for bucket in BUCKETS
                }
                for db in DATABASES
            }
            for op in OPERATORS
        }
        for arm in ARMS
    }


def _failures(groups: dict, macros: dict) -> list[dict[str, Any]]:
    failures = []
    if macros["T_all3"] > (1 - MIN_GAIN) * macros["R_head3"]:
        failures.append({"gate": "macro_improvement", "required_improvement": MIN_GAIN})
    for db in DATABASES:
        reference = groups["R_head3"]["head3"][db]
        treatment = groups["T_all3"]["head3"][db]
        for bucket in ("all", "4+"):
            for metric in ("q50", "q95"):
                if (
                    treatment[bucket][metric]
                    > MAX_REGRESSION_FACTOR * reference[bucket][metric]
                ):
                    failures.append(
                        {
                            "gate": "local_protection",
                            "database": db,
                            "bucket": bucket,
                            "metric": metric,
                        }
                    )
        if treatment["low4"]["q95"] > MAX_REGRESSION_FACTOR * reference["low4"]["q95"]:
            failures.append({"gate": "low4_reference", "database": db})
        baseline = groups["T_all3"]["none"][db]["low4"]["q95"]
        if treatment["low4"]["q95"] > MAX_REGRESSION_FACTOR * baseline:
            failures.append({"gate": "low4_self_adaptation", "database": db})
    return failures


def decide(
    cells: object, *, reproduction_verified: bool, engineering_verified: bool
) -> dict[str, Any]:
    """Classify a complete normalized matrix; never infer certification from data."""
    if reproduction_verified is not True or engineering_verified is not True:
        return _blocked("Engineering and reference reproduction must be certified")
    indexed = _index(cells)
    if indexed is None:
        return _blocked("Invalid, incomplete, nonfinite or unpaired 60-cell matrix")
    groups = _groups(indexed)
    macros = {
        a: math.fsum(groups[a]["head3"][db]["4+"]["q95"] / 3 for db in DATABASES)
        for a in ARMS
    }
    failures = _failures(groups, macros)
    return {
        "status": "no_candidate" if failures else "promising_single_seed_K20",
        "selected_arm": None if failures else "T_all3",
        "metric": "15-cell arithmetic macro standard unique-query four-plus Q95",
        "macro_four_plus_q95": macros,
        "relative_improvement": 1 - macros["T_all3"] / macros["R_head3"],
        "groups": groups,
        "failures": failures,
    }
