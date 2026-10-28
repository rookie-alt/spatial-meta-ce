"""Independent NumPy summary and pre-registered decision for doc 112.

This module reads per-query predictions and recomputes Q95/Q50 with NumPy.  It
does not call the production summariser, so a bug there cannot certify itself.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from .plans import EVAL_DATABASES, EXPECTED_QD
from .source_eval import ARMS, KS
from .train import FIXED_SEEDS

REQUIRED_SEEDS = FIXED_SEEDS
REQUIRED_RUNS = {5: (1, 2, 3, 4, 5), 20: (1, 2, 3, 4, 5)}
EXPECTED_HEADER = (
    "arm",
    "seed",
    "database",
    "k",
    "run",
    "query_id",
    "query_hash",
    "true_cardinality",
    "log_prediction",
)


def _q_values(log_pred, true):
    log_pred = np.maximum(np.asarray(log_pred, dtype=np.float64), 0.0)
    true = np.asarray(true, dtype=np.float64)
    values = np.exp(np.abs(log_pred - np.log(np.maximum(true, 1.0))))
    if not np.isfinite(values).all():
        raise FloatingPointError("Non-finite Q value in predictions")
    return values


def _percentile(values, percentile):
    return float(np.percentile(np.asarray(values, dtype=np.float64), percentile))


def _parse_row(row):
    try:
        arm = row["arm"]
        seed = int(row["seed"])
        database = row["database"]
        k = int(row["k"])
        run = int(row["run"])
        query_id = row["query_id"]
        query_hash = row["query_hash"]
        true = float(row["true_cardinality"])
        log_prediction = float(row["log_prediction"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Invalid prediction row: {row!r}") from exc
    if arm not in ARMS:
        raise ValueError(f"Unknown arm in predictions: {arm!r}")
    if seed not in REQUIRED_SEEDS:
        raise ValueError(f"Unexpected seed in predictions: {seed}")
    if database not in EVAL_DATABASES:
        raise ValueError(f"Unexpected database in predictions: {database!r}")
    if k not in KS:
        raise ValueError(f"Unexpected K in predictions: {k}")
    if k == 0 and run != 0:
        raise ValueError("K0 predictions must use run=0")
    if k != 0 and run not in REQUIRED_RUNS[k]:
        raise ValueError(f"Unexpected run for K={k}: {run}")
    if not query_id or not query_hash:
        raise ValueError("Prediction row has an empty query identity")
    if not math.isfinite(true) or true < 0:
        raise ValueError("Prediction row has an invalid true cardinality")
    if not math.isfinite(log_prediction):
        raise ValueError("Prediction row has a non-finite log prediction")
    return arm, seed, database, k, run, query_id, query_hash, true, log_prediction


def _matrix_keys():
    keys = set()
    for arm in ARMS:
        for seed in REQUIRED_SEEDS:
            for db in EVAL_DATABASES:
                keys.add((arm, seed, db, 0, 0))
                for k, runs in REQUIRED_RUNS.items():
                    for run in runs:
                        keys.add((arm, seed, db, k, run))
    return keys


def _scan(path, expected_qd):
    expected = _matrix_keys()
    counts = defaultdict(int)
    hashes = defaultdict(hashlib.sha256)
    seen_queries = defaultdict(set)
    with open(path, newline="") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != EXPECTED_HEADER:
            raise ValueError("Unexpected predictions.csv header")
        for row in reader:
            (
                arm,
                seed,
                database,
                k,
                run,
                _query_id,
                query_hash,
                _true,
                _log_prediction,
            ) = _parse_row(row)
            key = (arm, seed, database, k, run)
            if key not in expected:
                raise ValueError(f"Unexpected prediction unit: {key}")
            if query_hash in seen_queries[key]:
                raise ValueError(f"Duplicate query hash within unit: {key}")
            seen_queries[key].add(query_hash)
            counts[key] += 1
            hashes[key].update(query_hash.encode())
    for key in expected:
        if counts[key] == 0:
            raise ValueError(f"Missing prediction unit: {key}")
    database_counts = {}
    database_digest = {}
    for database in EVAL_DATABASES:
        unit_counts = {counts[key] for key in expected if key[2] == database}
        if len(unit_counts) != 1:
            raise ValueError(f"Inconsistent Q_d row counts for {database}")
        count = unit_counts.pop()
        if expected_qd is not None and count != expected_qd[database]:
            raise ValueError(
                f"{database} Q_d row count changed: {count} != {expected_qd[database]}"
            )
        database_counts[database] = count
        digests = {hashes[key].hexdigest() for key in expected if key[2] == database}
        if len(digests) != 1:
            raise ValueError(f"Inconsistent Q_d query sets for {database}")
        database_digest[database] = digests.pop()
    return expected, counts, database_counts, database_digest


def _read_values(path, keys):
    values = {key: [] for key in keys}
    with open(path, newline="") as stream:
        for row in csv.DictReader(stream):
            arm, seed, database, k, run, _query_id, _hash, true, log_prediction = (
                _parse_row(row)
            )
            key = (arm, seed, database, k, run)
            values[key].append(_q_values([log_prediction], [true])[0])
    return values


def _unit_metrics(values):
    metrics = {percentile: {} for percentile in (50, 90, 95, 99)}
    for key, rows in values.items():
        for percentile in metrics:
            metrics[percentile][key] = _percentile(rows, percentile)
    return metrics


def _qbar(unit):
    qbar = {}
    for arm in ARMS:
        for seed in REQUIRED_SEEDS:
            for db in EVAL_DATABASES:
                key0 = (arm, seed, db, 0, 0)
                qbar[(arm, seed, db, 0)] = unit[key0]
                for k, runs in REQUIRED_RUNS.items():
                    qbar[(arm, seed, db, k)] = float(
                        np.mean([unit[(arm, seed, db, k, run)] for run in runs])
                    )
    return qbar


def _run_sd(q95):
    result = {}
    for arm in ARMS:
        for seed in REQUIRED_SEEDS:
            for db in EVAL_DATABASES:
                for k, runs in REQUIRED_RUNS.items():
                    result[(arm, seed, db, k)] = float(
                        np.std(
                            [q95[(arm, seed, db, k, run)] for run in runs],
                            ddof=1,
                        )
                    )
    return result


def _ratio_table(qbar, left, right):
    result = {}
    for seed in REQUIRED_SEEDS:
        for db in EVAL_DATABASES:
            for k in (0, 5, 20):
                result.setdefault(str(seed), {}).setdefault(db, {})[f"K{k}"] = float(
                    qbar[(left, seed, db, k)] / qbar[(right, seed, db, k)]
                )
    return result


def _macro(qbar):
    macro = {}
    for arm in ARMS:
        for seed in REQUIRED_SEEDS:
            for k in KS:
                macro[(arm, seed, k)] = float(
                    np.mean([qbar[(arm, seed, db, k)] for db in EVAL_DATABASES])
                )
    return macro


def _g(macro, left, right, k, seed):
    return 1.0 - macro[(left, seed, k)] / macro[(right, seed, k)]


def _stats(values):
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(values.mean()),
        "sd": float(values.std(ddof=1)),
        "min": float(values.min()),
        "max": float(values.max()),
        "positive": int((values > 0).sum()),
        "n": len(values),
    }


def _ratio_checks(qbar, left, right, metric):
    failures = []
    ratio = {}
    for db in EVAL_DATABASES:
        for k in (0, 20):
            left_mean = np.mean([qbar[(left, seed, db, k)] for seed in REQUIRED_SEEDS])
            right_mean = np.mean(
                [qbar[(right, seed, db, k)] for seed in REQUIRED_SEEDS]
            )
            value = float(left_mean / right_mean)
            ratio[f"{db}:K{k}"] = value
            if value > 1.10:
                failures.append(f"{db} K{k} {metric} ratio {value:.6f} > 1.10")
    return ratio, failures


def _self_ratio_table(qbar):
    result = {}
    for arm in ARMS:
        for seed in REQUIRED_SEEDS:
            for db in EVAL_DATABASES:
                result.setdefault(arm, {}).setdefault(str(seed), {})[db] = float(
                    qbar[(arm, seed, db, 20)] / qbar[(arm, seed, db, 0)]
                )
    return result


def decide(qbar95, qbar50, run_sd):
    macro95 = _macro(qbar95)
    macro50 = _macro(qbar50)
    gcb20 = [_g(macro95, "C_joint", "B_fomaml", 20, seed) for seed in REQUIRED_SEEDS]
    gca20 = [_g(macro95, "C_joint", "A_erm", 20, seed) for seed in REQUIRED_SEEDS]
    gba20 = [_g(macro95, "B_fomaml", "A_erm", 20, seed) for seed in REQUIRED_SEEDS]
    gself = {
        arm: _stats(
            [
                1.0 - macro95[(arm, seed, 20)] / macro95[(arm, seed, 0)]
                for seed in REQUIRED_SEEDS
            ]
        )
        for arm in ARMS
    }
    ratio95, fail95 = _ratio_checks(qbar95, "C_joint", "B_fomaml", "Q95")
    ratio50, fail50 = _ratio_checks(qbar50, "C_joint", "B_fomaml", "Q50")
    k0_protect = {
        str(seed): float(
            macro95[("C_joint", seed, 0)] / macro95[("B_fomaml", seed, 0)]
        )
        for seed in REQUIRED_SEEDS
    }
    six_one = (
        _stats(gcb20)["mean"] >= 0.05
        and _stats(gcb20)["positive"] >= 2
        and all(value <= 1.05 for value in k0_protect.values())
        and not fail95
        and not fail50
    )
    self_ratio = {}
    self_fail = []
    for db in EVAL_DATABASES:
        value = float(
            np.mean([qbar95[("C_joint", seed, db, 20)] for seed in REQUIRED_SEEDS])
            / np.mean([qbar95[("C_joint", seed, db, 0)] for seed in REQUIRED_SEEDS])
        )
        self_ratio[db] = value
        if value > 1.10:
            self_fail.append(f"{db} C K20/K0 Q95 ratio {value:.6f} > 1.10")
    six_two = (
        gself["C_joint"]["mean"] >= 0.05
        and gself["C_joint"]["positive"] >= 2
        and not self_fail
    )
    ca_ratio95, ca_fail95 = _ratio_checks(qbar95, "C_joint", "A_erm", "Q95")
    six_three = (
        _stats(gca20)["mean"] >= 0.05
        and _stats(gca20)["positive"] >= 2
        and not ca_fail95
    )
    a_worse = any(
        macro95[("A_erm", seed, 20)] > macro95[("A_erm", seed, 0)]
        for seed in REQUIRED_SEEDS
    )
    cross_ratio_c20_a0 = {
        str(seed): float(
            macro95[("C_joint", seed, 20)] / macro95[("A_erm", seed, 0)]
        )
        for seed in REQUIRED_SEEDS
    }
    if six_one and six_two and six_three:
        decision = "promising_source_screen"
    elif not six_one:
        decision = "no_candidate"
    else:
        decision = "partial_source_signal"
    reasons = []
    if not six_one:
        reasons.append("6.1 joint-vs-B threshold not met")
    if not six_two:
        reasons.append("6.2 C K20 self-adaptation threshold not met")
    if not six_three:
        reasons.append("6.3 C-vs-A threshold not met")
    result = {
        "decision": decision,
        "six_one": {
            "passed": six_one,
            "g_c_b_k20": _stats(gcb20),
            "k0_protect_c_over_b": k0_protect,
            "k0_k20_ratio_q95": ratio95,
            "k0_k20_ratio_q50": ratio50,
            "failures": fail95 + fail50,
        },
        "six_two": {
            "passed": six_two,
            "g_self_c_k20": gself["C_joint"],
            "per_database_ratio": self_ratio,
            "failures": self_fail,
        },
        "six_three": {
            "passed": six_three,
            "g_c_a_k20": _stats(gca20),
            "g_b_a_k20": _stats(gba20),
            "per_database_ratio": ca_ratio95,
            "failures": ca_fail95,
            "a_k20_worse_than_a_k0": a_worse,
            "cross_ratio_c20_a0": cross_ratio_c20_a0,
        },
        "g_self": gself,
        "macro_q95": _jsonable_macro(macro95),
        "macro_q50": _jsonable_macro(macro50),
        "ratios_q95": {
            "C_over_B": _ratio_table(qbar95, "C_joint", "B_fomaml"),
            "C_over_A": _ratio_table(qbar95, "C_joint", "A_erm"),
            "B_over_A": _ratio_table(qbar95, "B_fomaml", "A_erm"),
        },
        "ratios_q50": {
            "C_over_B": _ratio_table(qbar50, "C_joint", "B_fomaml"),
            "C_over_A": _ratio_table(qbar50, "C_joint", "A_erm"),
        },
        "self_ratio_q95": _self_ratio_table(qbar95),
        "self_ratio_q50": _self_ratio_table(qbar50),
        "run_q95_sd": _jsonable_sd(run_sd),
        "reasons": reasons,
    }
    return result


def _jsonable(qbar):
    out = {}
    for (arm, seed, db, k), value in qbar.items():
        out.setdefault(arm, {}).setdefault(str(seed), {}).setdefault(db, {})[
            str(k)
        ] = value
    return out


def _jsonable_macro(macro):
    out = {}
    for (arm, seed, k), value in macro.items():
        out.setdefault(arm, {}).setdefault(str(seed), {})[str(k)] = value
    return out


def _jsonable_sd(run_sd):
    out = {}
    for (arm, seed, db, k), value in run_sd.items():
        out.setdefault(arm, {}).setdefault(str(seed), {}).setdefault(db, {})[
            str(k)
        ] = value
    return out


def _jsonable_unit_metrics(metrics):
    out = {}
    for percentile, units in metrics.items():
        for (arm, seed, db, k, run), value in units.items():
            out.setdefault(arm, {}).setdefault(str(seed), {}).setdefault(
                db, {}
            ).setdefault(str(k), {}).setdefault(str(run), {})[f"q{percentile}"] = value
    return out


def summarize_predictions(path, expected_qd=EXPECTED_QD):
    keys, _counts, database_counts, database_digest = _scan(path, expected_qd)
    values = _read_values(path, keys)
    metrics = _unit_metrics(values)
    qbar95 = _qbar(metrics[95])
    qbar50 = _qbar(metrics[50])
    run_sd = _run_sd(metrics[95])
    return {
        "schema": 1,
        "kind": "fomaml_joint_source_summary",
        "predictions": str(path),
        "seeds": list(REQUIRED_SEEDS),
        "arms": list(ARMS),
        "query_counts": database_counts,
        "query_digests": database_digest,
        "unit_metrics": _jsonable_unit_metrics(metrics),
        "run_q95_sd": _jsonable_sd(run_sd),
        "qbar_q95": _jsonable(qbar95),
        "qbar_q50": _jsonable(qbar50),
        "decision": decide(qbar95, qbar50, run_sd),
    }


def write_summary(output_dir, summary):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )
    (output_dir / "decision.json").write_text(
        json.dumps(summary["decision"], ensure_ascii=False, indent=2, allow_nan=False)
        + "\n"
    )
