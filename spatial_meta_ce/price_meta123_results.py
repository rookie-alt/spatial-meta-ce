"""Independent NumPy verification and preregistered decisions; no model imports."""

import csv
import gzip
import math
from pathlib import Path

import numpy as np

DATABASES = ("chembl", "legalacts", "telstra")
SEEDS = (42, 43, 44)
LRS = (1e-7, 1e-6, 1e-5)
FIELDS = (
    "arm",
    "stream",
    "database",
    "k",
    "run",
    "lr",
    "query_id",
    "query_hash",
    "true",
    "encoded",
)


def compare_predictions(candidate, reference):
    """Independent paired errors on exactly the same ordered query population."""

    def rows(path):
        path = Path(path)
        opener = gzip.open if path.suffix == ".gz" else open
        with opener(path, "rt", newline="") as stream:
            reader = csv.DictReader(stream)
            if tuple(reader.fieldnames or ()) != FIELDS:
                raise ValueError("Paired CSV schema changed")
            yield from reader

    from itertools import zip_longest

    deltas = []
    seen = set()
    for left, right in zip_longest(rows(candidate), rows(reference)):
        if (
            left is None
            or right is None
            or any(left[k] != right[k] for k in ("query_id", "query_hash", "true"))
        ):
            raise ValueError("Paired prediction query identity/truth mismatch")
        if left["query_hash"] in seen:
            raise ValueError("Duplicate paired query identity")
        seen.add(left["query_hash"])
        errors = []
        for row in (left, right):
            truth, z = float(row["true"]), float(row["encoded"]) - 1
            if not math.isfinite(z) or not math.isfinite(truth) or truth < 0:
                raise FloatingPointError("Nonfinite paired predictions")
            decoded = 0.0 if z <= math.log(2) else z + math.log1p(-math.exp(-z))
            errors.append(abs(decoded - math.log(max(truth, 1))))
        deltas.append(errors[0] - errors[1])
    if not deltas:
        raise ValueError("Paired predictions are empty")
    values = np.asarray(deltas)
    return {
        "n": len(values),
        "mean_delta_log_error": float(values.mean()),
        "median_delta_log_error": float(np.median(values)),
        "fraction_worsened": float(np.mean(values > 0)),
        "fraction_improved": float(np.mean(values < 0)),
    }


def _key(unit):
    return tuple(
        unit[k] for k in ("arm", "stream", "database", "k", "run", "lr", "population")
    )


def _check_unit(unit, *, allow_failed=False):
    arm, seed, db, k, run, lr, population = _key(unit)
    if (
        arm not in ("P", "E", "M", "E_alt")
        or db not in DATABASES
        or seed not in ((0,) if arm == "P" else SEEDS)
        or k not in (0, 5, 20)
        or run not in ((0,) if k == 0 else range(5))
        or lr not in ((0.0,) if k == 0 else LRS)
        or population not in ("calibration", "report")
    ):
        raise ValueError("Invalid experiment cell identity")
    if type(unit["n"]) is not int or unit["n"] <= 0:
        raise ValueError("Invalid cell denominator")
    if unit.get("status") == "failed" and allow_failed:
        if not unit.get("error"):
            raise ValueError("Failed cell needs its original error")
        return
    if unit.get("status") != "completed":
        raise ValueError("Incomplete experiment cell")
    metrics = unit.get("metrics", {})
    if any(
        key not in metrics or not math.isfinite(metrics[key]) or metrics[key] < 1
        for key in ("q50", "q95")
    ):
        raise ValueError("Nonfinite or invalid Q-error metrics")


def make_failed_unit(arm, stream, database, k, run, lr, population, n, error):
    value = dict(
        arm=arm,
        stream=stream,
        database=database,
        k=k,
        run=run,
        lr=lr,
        population=population,
        n=n,
        status="failed",
        metrics=None,
        error=str(error),
    )
    _check_unit(value, allow_failed=True)
    return value


def compute_unit(csv_path, expected_rows, *, population="report"):
    path = Path(csv_path)
    ordered = not isinstance(expected_rows, dict)
    expected = (
        list(expected_rows)
        if ordered
        else [(key, value[0], value[1]) for key, value in expected_rows.items()]
    )
    expected_map = {row[0]: (row[1], float(row[2])) for row in expected}
    if not expected or len(expected_map) != len(expected):
        raise ValueError("Empty or duplicate expected query IDs")
    open_file = gzip.open if path.suffix == ".gz" else open
    keys, seen_ids, seen_hashes, errors = set(), set(), set(), []
    with open_file(path, "rt", newline="") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != FIELDS:
            raise ValueError("Prediction CSV schema changed")
        for index, row in enumerate(reader):
            if index >= len(expected) or None in row:
                raise ValueError("Prediction CSV has extra or malformed rows")
            try:
                key = (
                    row["arm"],
                    int(row["stream"]),
                    row["database"],
                    int(row["k"]),
                    int(row["run"]),
                    float(row["lr"]),
                    population,
                )
                truth, encoded = float(row["true"]), float(row["encoded"])
            except (ValueError, TypeError) as exc:
                raise ValueError("Malformed CSV numeric value") from exc
            if not math.isfinite(truth) or not math.isfinite(encoded) or truth < 0:
                raise FloatingPointError("Nonfinite prediction/true cardinality")
            identity = (row["query_id"], row["query_hash"], truth)
            if (
                row["query_id"] not in expected_map
                or expected_map[row["query_id"]] != (row["query_hash"], truth)
                or (ordered and identity != tuple(expected[index]))
            ):
                raise ValueError(
                    "Prediction query ID/hash/truth differs from the frozen plan"
                )
            if row["query_id"] in seen_ids or row["query_hash"] in seen_hashes:
                raise ValueError("Duplicate prediction query identity")
            seen_ids.add(row["query_id"])
            seen_hashes.add(row["query_hash"])
            keys.add(key)
            native_log = encoded - 1
            decoded = (
                0.0
                if native_log <= math.log(2)
                else (native_log + math.log1p(-math.exp(-native_log)))
            )
            errors.append(abs(max(decoded, 0) - math.log(max(truth, 1))))
    if seen_ids != set(expected_map) or len(keys) != 1:
        raise ValueError("Missing queries or more than one prediction cell")
    errors = np.asarray(errors, dtype=np.float64)
    with np.errstate(over="ignore"):
        q = np.exp(errors)
    if not np.isfinite(q).all():
        raise FloatingPointError("Q-error overflow; outputs may not be clipped")
    metrics = {
        f"q{p}": float(np.percentile(q, p, method="linear")) for p in (50, 90, 95, 99)
    }
    metrics.update(
        q_mean=float(q.mean()),
        log_q95=float(np.percentile(errors, 95, method="linear")),
    )
    identity = next(iter(keys))
    unit = dict(
        zip(("arm", "stream", "database", "k", "run", "lr", "population"), identity)
    )
    unit.update(status="completed", n=len(q), metrics=metrics)
    _check_unit(unit)
    return unit


def _indexed(units, allow_failed=False):
    values = list(units.values()) if isinstance(units, dict) else list(units)
    result = {}
    for unit in values:
        _check_unit(unit, allow_failed=allow_failed)
        key = _key(unit)
        if key in result:
            raise ValueError("Duplicate experiment cell")
        result[key] = unit
    return result


def select_lr(units, arm, *, population="calibration"):
    if arm not in ("P", "E") or population != "calibration":
        raise ValueError("Only source calibration cells may choose LR")
    cells = _indexed(units, allow_failed=True)
    seeds = (0,) if arm == "P" else SEEDS
    expected = {
        (arm, s, d, 20, r, lr, population)
        for s in seeds
        for d in DATABASES
        for r in range(5)
        for lr in LRS
    }
    expected |= {(arm, s, d, 0, 0, 0.0, population) for s in seeds for d in DATABASES}
    actual = set(cells)
    if actual != expected:
        raise ValueError("Incomplete or unexpected LR calibration matrix")
    candidates = {}
    for lr in LRS:
        failures = [
            key
            for key in expected
            if key[3] == 20 and key[5] == lr and cells[key]["status"] != "completed"
        ]
        if failures:
            candidates[str(lr)] = {"status": "rejected", "rejected_cells": failures}
            continue
        cbar = {
            (s, d): float(
                np.mean(
                    [
                        cells[(arm, s, d, 20, r, lr, population)]["metrics"]["q95"]
                        for r in range(5)
                    ]
                )
            )
            for s in seeds
            for d in DATABASES
        }
        candidates[str(lr)] = {
            "status": "valid",
            "J": float(np.mean(np.log(list(cbar.values())))),
            "cbar": {f"{s}/{d}": value for (s, d), value in cbar.items()},
        }
    valid = [lr for lr in LRS if candidates[str(lr)]["status"] == "valid"]
    if any(cells[key]["status"] != "completed" for key in expected if key[3] == 0):
        raise ValueError("K0 calibration cells must all complete")
    if not valid:
        raise ValueError("Every registered LR candidate was rejected")
    minimum = min(candidates[str(lr)]["J"] for lr in valid)
    selected = min(
        lr for lr in valid if abs(candidates[str(lr)]["J"] - minimum) <= 1e-8
    )
    return {
        "arm": arm,
        "lr": selected,
        "lr_boundary_selected": selected in (min(LRS), max(LRS)),
        "candidates": candidates,
        "tie_tolerance": 1e-8,
        "cells_expected": len(expected),
        "cells_present": len(actual),
    }


def aggregate_report(units, alpha, alpha_e, *, denominators=None):
    if alpha not in LRS or alpha_e not in LRS:
        raise ValueError("Unregistered report LR")
    cells = _indexed(units)
    expected = set()
    for arm, seeds in (("P", (0,)), ("E", SEEDS), ("M", SEEDS)):
        for seed in seeds:
            for db in DATABASES:
                for k in (0, 5, 20):
                    expected |= {
                        (arm, seed, db, k, run, 0.0 if k == 0 else alpha, "report")
                        for run in ((0,) if k == 0 else range(5))
                    }
    if alpha_e != alpha:
        expected |= {
            ("E_alt", s, d, 20, r, alpha_e, "report")
            for s in SEEDS
            for d in DATABASES
            for r in range(5)
        }
    if set(cells) != expected:
        raise ValueError("Incomplete or unexpected report matrix")
    if denominators is not None and set(denominators) != set(DATABASES):
        raise ValueError("Report denominator mapping is incomplete")
    for db in DATABASES:
        values = {unit["n"] for unit in cells.values() if unit["database"] == db}
        if len(values) != 1 or (
            denominators is not None and values != {denominators[db]}
        ):
            raise ValueError("Report query denominators disagree")

    def qbar(arm, seed, db, k, metric="q95"):
        if arm == "E_alt" and alpha_e == alpha:
            arm = "E"
        seed = 0 if arm == "P" else seed
        lr = 0.0 if k == 0 else (alpha_e if arm == "E_alt" else alpha)
        return float(
            np.mean(
                [
                    cells[(arm, seed, db, k, r, lr, "report")]["metrics"][metric]
                    for r in ((0,) if k == 0 else range(5))
                ]
            )
        )

    def score(arm, seed, k):
        return float(np.mean([math.log(qbar(arm, seed, db, k)) for db in DATABASES]))

    comparisons = {
        "M0/P0": (0, "P", 0),
        "M0/E0": (0, "E", 0),
        "M20/M0": (20, "M", 0),
        "M20/E20": (20, "E", 20),
        "M20/E_alt20": (20, "E_alt", 20),
        "M20/P20": (20, "P", 20),
        "M20/P0": (20, "P", 0),
        "M20/E0": (20, "E", 0),
    }
    ratios = {}
    for name, (k, arm, other_k) in comparisons.items():
        deltas = [score("M", s, k) - score(arm, s, other_k) for s in SEEDS]
        ratio = math.exp(float(np.mean(deltas)))
        ratios[name] = {
            "ratio": ratio,
            "percent_change": 100 * (ratio - 1),
            "deltas": deltas,
            "improved_streams": sum(x < 0 for x in deltas),
        }

    def value(name):
        return ratios[name]["ratio"]

    gates = {
        "relative_gain": value("M20/E20") <= 0.95
        and ratios["M20/E20"]["improved_streams"] >= 2
        and value("M20/E_alt20") <= 0.95
        and value("M20/P20") <= 0.95,
        "useful_adaptation": value("M20/M0") <= 0.95
        and ratios["M20/M0"]["improved_streams"] >= 2
        and value("M20/P0") <= 0.95
        and value("M20/E0") <= 0.95,
        "k0_protect": value("M0/P0") <= 1.05 and value("M0/E0") <= 1.05,
    }
    protections = {}
    for db in DATABASES:
        for metric in ("q95", "q50"):
            for name, (k, arm, other_k) in comparisons.items():
                numerator = np.mean([qbar("M", s, db, k, metric) for s in SEEDS])
                denominator = np.mean(
                    [qbar(arm, s, db, other_k, metric) for s in SEEDS]
                )
                protections[f"{db}/{metric}/{name}"] = float(numerator / denominator)
    gates["per_database"] = all(x <= 1.10 for x in protections.values())
    decision = (
        "promising_source_screen"
        if all(gates.values())
        else ("no_source_gain" if value("M20/E20") >= 1 else "partial_source_signal")
    )
    negative = {}
    support_sd = {}
    for arm, seeds in (("P", (0,)), ("E", SEEDS), ("M", SEEDS), ("E_alt", SEEDS)):
        for seed in seeds:
            for db in DATABASES:
                for k in (20,) if arm == "E_alt" else (5, 20):
                    actual_arm = "E" if arm == "E_alt" and alpha_e == alpha else arm
                    lr = alpha_e if actual_arm == "E_alt" else alpha
                    rows = [
                        cells[(actual_arm, seed, db, k, r, lr, "report")]
                        for r in range(5)
                    ]
                    baseline = qbar("E" if arm == "E_alt" else arm, seed, db, 0)
                    negative[f"{arm}/{seed}/{db}/{k}"] = sum(
                        u["metrics"]["q95"] > baseline for u in rows
                    )
                    support_sd[f"{arm}/{seed}/{db}/{k}"] = float(
                        np.std([u["metrics"]["q95"] for u in rows], ddof=1)
                    )
    return {
        "decision": decision,
        "gates": gates,
        "ratios": ratios,
        "per_database_ratios": protections,
        "negative_transfer": negative,
        "support_sd": support_sd,
        "cells": len(cells),
        "alpha": alpha,
        "alpha_e": alpha_e,
        "uncertainty_scope": "one shared pretraining initialization; paired continuation streams",
    }
