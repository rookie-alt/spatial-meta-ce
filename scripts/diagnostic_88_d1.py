#!/usr/bin/env python3
"""D1 of protocol 88: shape of the head3 low4 degradation.

Read-only CPU analysis of the 50 retained formal-holdout prediction archives for
validation databases chembl/legalacts/telstra, arms none and head3, runs 0..4.
No checkpoint loading, no torch, no forward pass.  Target databases
imdb/stats/genome/ergastf1 and the holdout databases hepatitis/sakila are
rejected before any file is read.

Metric definitions follow protocol 88 section 3.2:
  low4      = (n_tables >= 4) and (true < 100)
  logerr    = abs(max(raw_log_pred, 0) - log(max(true, 1)))
  delta_i   = logerr_head3_i - logerr_none_i        (negative == improvement)
  concentration = sum(largest ceil(1% * n) of delta_i>0) / sum(delta_i for delta_i>0)
Terciles use the 100/3 and 200/3 percentiles (method="linear") per run.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

VALIDATION_DATABASES = ("chembl", "legalacts", "telstra")
FORBIDDEN_DATABASES = ("imdb", "stats", "genome", "ergastf1")
SKIPPED_DATABASES = ("hepatitis", "sakila")
ARMS = ("none", "head3")
RUNS = tuple(range(5))
FIELDS = frozenset(("log_pred", "true", "n_tables", "query_hash", "query_id"))
PERCENTILES = (50, 75, 90, 95, 99)
CONCENTRATION_SHARE = 0.01
CONCENTRATED_THRESHOLD = 0.40
DIFFUSE_THRESHOLD = 0.25
BALANCE_LOW = 0.40
BALANCE_HIGH = 0.60
MIN_LOW4_PER_RUN = 100
LOW4_MAX_TRUE = 100
LOW4_MIN_TABLES = 4


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def guard_database(name):
    """Reject forbidden databases before any path is constructed."""
    if name in FORBIDDEN_DATABASES:
        raise ValueError("Forbidden database name reached protocol 88 D1: %s" % name)
    if name not in VALIDATION_DATABASES:
        raise ValueError("Unexpected database in protocol 88 D1: %s" % name)


def load_cell(path):
    guard_database(path.stem.split("-")[2])
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != FIELDS:
            raise ValueError("Unexpected NPZ fields: %s: %s" % (path, sorted(archive.files)))
        cell = {name: np.asarray(archive[name]) for name in archive.files}
    for name in FIELDS:
        if cell[name].ndim != 1:
            raise ValueError("Field must be a vector: %s: %s" % (path, name))
    size = len(cell["true"])
    if size == 0 or any(len(cell[name]) != size for name in FIELDS):
        raise ValueError("Field lengths disagree: %s" % path)
    log_pred = cell["log_pred"].astype(np.float64)
    true = cell["true"].astype(np.float64)
    tables = cell["n_tables"]
    if not np.isfinite(log_pred).all() or not np.isfinite(true).all():
        raise ValueError("Nonfinite prediction or truth: %s" % path)
    if (true < 0).any() or tables.dtype.kind not in "iu" or (tables < 1).any():
        raise ValueError("Invalid truth or table count: %s" % path)
    return {"log_pred": log_pred, "true": true, "n_tables": tables.astype(np.int64),
            "query_hash": cell["query_hash"].astype(str),
            "query_id": cell["query_id"].astype(str)}


def log_errors(cell):
    with np.errstate(over="raise", invalid="raise"):
        values = np.abs(np.maximum(cell["log_pred"], 0) - np.log(np.maximum(cell["true"], 1)))
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite log-Q error")
    return values


def q_errors(logerr):
    with np.errstate(over="raise", invalid="raise"):
        values = np.exp(logerr)
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite Q-error")
    return values


def percentiles(values):
    if len(values) == 0:
        return {("q%d" % p): None for p in PERCENTILES}
    return {("q%d" % p): float(np.percentile(values, p, method="linear")) for p in PERCENTILES}


def quantile_ratio(none, head3):
    if len(none) == 0:
        return {"q95_none": None, "q95_head3": None, "q95_ratio": None}
    base = float(np.percentile(none, 95, method="linear"))
    top = float(np.percentile(head3, 95, method="linear"))
    return {"q95_none": base, "q95_head3": top,
            "q95_ratio": top / base if base else None}


def concentration(delta):
    """Share of total positive log-Q increment from the largest ceil(1%*n) deltas."""
    positive = delta[delta > 0]
    total = float(positive.sum())
    count = int(len(positive))
    if count == 0 or total <= 0:
        return {"positive_count": count, "positive_total": total,
                "top_count": 0, "top_total": 0.0, "share": None}
    top_count = min(max(1, int(math.ceil(CONCENTRATION_SHARE * len(delta)))), count)
    top = np.sort(positive)[::-1][:top_count]
    return {"positive_count": count, "positive_total": total,
            "top_count": top_count, "top_total": float(top.sum()),
            "share": float(top.sum()) / total}


def paired_delta(none, head3):
    if len(none) == 0:
        return {"count": 0, "better": 0, "worse": 0, "equal": 0,
                "better_share": None, "worse_share": None,
                "median": None, "p90": None, "p99": None, "mean": None}
    delta = head3 - none
    better = int((delta < 0).sum())
    worse = int((delta > 0).sum())
    equal = int((delta == 0).sum())
    total = better + worse + equal
    return {"count": int(len(delta)), "better": better, "worse": worse, "equal": equal,
            "better_share": better / total, "worse_share": worse / total,
            "median": float(np.median(delta)),
            "p90": float(np.percentile(delta, 90, method="linear")),
            "p99": float(np.percentile(delta, 99, method="linear")),
            "mean": float(delta.mean())}


def tercile_ids(values):
    """Bucket ids 0/1/2 from the 33.3 and 66.7 percentiles (linear)."""
    low, high = np.percentile(values, [100.0 / 3, 200.0 / 3], method="linear")
    return np.searchsorted([low, high], values, side="right")


def bucket_report(mask, logerr_none, logerr_head3, q_none, q_head3):
    report = paired_delta(logerr_none[mask], logerr_head3[mask])
    report.update(quantile_ratio(q_none[mask], q_head3[mask]))
    report["count"] = int(mask.sum())
    return report


def run_analysis(cells, pg):
    """Compute every protocol-88 3.2 statistic for one (database, run)."""
    reference = cells["none"]
    for arm in ARMS:
        for name in ("true", "n_tables", "query_hash", "query_id"):
            if not np.array_equal(cells[arm][name], reference[name]):
                raise ValueError("Unpaired identity/label across arms: %s" % name)
    true = reference["true"]
    tables = reference["n_tables"]
    logerr = {arm: log_errors(cells[arm]) for arm in ARMS}
    qerr = {arm: q_errors(logerr[arm]) for arm in ARMS}
    low4 = (tables >= LOW4_MIN_TABLES) & (true < LOW4_MAX_TRUE)
    low4_positions = np.flatnonzero(low4)

    result = {
        "query_count": int(len(true)),
        "low4_count": int(low4.sum()),
        "low4_share": float(low4.mean()),
        "spectrum_low4": {arm: percentiles(qerr[arm][low4]) for arm in ARMS},
        "paired_all": paired_delta(logerr["none"], logerr["head3"]),
        "paired_low4": paired_delta(logerr["none"][low4], logerr["head3"][low4]),
        "concentration_low4": concentration((logerr["head3"] - logerr["none"])[low4]),
        "q95_ratio_low4": quantile_ratio(qerr["none"][low4], qerr["head3"][low4]),
        "buckets": {},
    }

    masks = {
        "n_tables=4": low4 & (tables == 4),
        "n_tables=5": low4 & (tables == 5),
        "n_tables=6+": low4 & (tables >= 6),
    }
    true_ids = tercile_ids(true[low4])
    for index in range(3):
        selected = np.zeros(len(true), dtype=bool)
        selected[low4_positions[true_ids == index]] = True
        masks["true_tercile_%d" % (index + 1)] = selected
    if pg is None:
        result["pg_status"] = "cache_missing_skipped"
    else:
        if not np.isfinite(pg).all() or (pg < 0).any():
            raise ValueError("Invalid PG anchor value from cache")
        ratio = np.abs(pg - true) / np.maximum(true, 1.0)
        pg_ids = tercile_ids(ratio[low4])
        for index in range(3):
            selected = np.zeros(len(true), dtype=bool)
            selected[low4_positions[pg_ids == index]] = True
            masks["pg_tercile_%d" % (index + 1)] = selected
        result["pg_status"] = "ok"

    for name, mask in masks.items():
        result["buckets"][name] = bucket_report(
            mask, logerr["none"], logerr["head3"], qerr["none"], qerr["head3"])

    covered = np.zeros(len(true), dtype=bool)
    for name in ("n_tables=4", "n_tables=5", "n_tables=6+"):
        covered |= masks[name]
    if not np.array_equal(covered, low4):
        raise ValueError("n_tables buckets lost denominator")
    return result


def median_range(values):
    clean = [float(value) for value in values if value is not None]
    if not clean:
        return {"median": None, "min": None, "max": None, "range": None}
    return {"median": float(np.median(clean)), "min": min(clean),
            "max": max(clean), "range": max(clean) - min(clean)}


def decide(databases):
    """Apply the frozen protocol-88 3.3 thresholds to per-run concentration shares."""
    eligible = {}
    for db, runs in databases.items():
        counts = [run["low4_count"] for run in runs]
        eligible[db] = float(np.median(counts)) >= MIN_LOW4_PER_RUN
    shares, balance = {}, {}
    for db, runs in databases.items():
        if not eligible[db]:
            continue
        values = [run["concentration_low4"]["share"] for run in runs
                  if run["concentration_low4"]["share"] is not None]
        if values:
            shares[db] = median_range(values)
            balance[db] = median_range([run["paired_low4"]["better_share"] for run in runs])
    if not shares:
        return {"verdict": "inconclusive",
                "reason": "no eligible database or concentration undefined",
                "eligible": eligible, "shares": {}}
    all_concentrated = all(item["median"] >= CONCENTRATED_THRESHOLD
                           for item in shares.values())
    all_diffuse = all(item["median"] < DIFFUSE_THRESHOLD for item in shares.values())
    balanced = all(BALANCE_LOW <= item["median"] <= BALANCE_HIGH for item in balance.values())
    if all_concentrated:
        verdict, reason = "concentrated", "every eligible database median share >= 0.40"
    elif all_diffuse and balanced:
        verdict, reason = "diffuse", "every eligible database < 0.25 with balance 0.40-0.60"
    else:
        verdict, reason = "inconclusive", "mixture or intermediate values"
    return {"verdict": verdict, "reason": reason, "eligible": eligible,
            "shares": shares, "improvement_share_low4": balance,
            "thresholds": {"concentrated": CONCENTRATED_THRESHOLD,
                           "diffuse": DIFFUSE_THRESHOLD,
                           "balance": [BALANCE_LOW, BALANCE_HIGH],
                           "min_low4_per_run": MIN_LOW4_PER_RUN}}


def load_pg(cache_root, database, query_id):
    """Read PG anchors for the selected rows and bind them strictly to query_id."""
    directory = Path(cache_root) / "pretrain" / database
    cards = np.load(directory / "cards.npy", allow_pickle=False, mmap_mode="r")
    offsets = np.load(directory / "offsets.npy", allow_pickle=False, mmap_mode="r")
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    rows = manifest["rows"]
    if cards.shape != (len(rows), 2) or offsets.shape != (len(rows) + 1, 4):
        raise ValueError("Cache/manifest shape mismatch: %s" % database)
    indices = []
    for value in query_id:
        parts = value.split("/")
        if len(parts) != 3 or parts[1] != "pretrain" or not parts[2].isdigit():
            raise ValueError("Invalid query_id: %s" % value)
        index = int(parts[2]) - 1
        if index >= len(rows) or rows[index].get("id") != value:
            raise ValueError("query_id does not address manifest row: %s" % value)
        indices.append(index)
    return np.asarray(cards[np.asarray(indices, dtype=np.int64), 1], dtype=np.float64)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True,
                        help="formal-holdout-20261018 directory")
    parser.add_argument("--cache-root", type=Path, required=True,
                        help="artifacts/cache holding pretrain/<database>")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    for path in (args.run_dir, args.cache_root, args.out):
        if any(part in FORBIDDEN_DATABASES for part in path.parts):
            raise ValueError("Target database appears in a path: %s" % path)
    if args.out.exists():
        raise ValueError("Refusing to overwrite output")

    records = json.loads((args.run_dir / "cells.json").read_text(encoding="utf-8"))
    by_key = {}
    skipped = set()
    for cell in records["cells"]:
        db = cell["database"]
        if db in FORBIDDEN_DATABASES:
            raise ValueError("Forbidden database name reached protocol 88 D1: %s" % db)
        if db in SKIPPED_DATABASES:
            skipped.add(db)
            continue
        guard_database(db)
        if cell["arm"] != "R_head3" or cell["operator"] not in ARMS:
            raise ValueError("Unexpected arm/operator: %s/%s" % (cell["arm"], cell["operator"]))
        by_key[(db, cell["operator"], cell["run"])] = cell
    expected = {(db, arm, run) for db in VALIDATION_DATABASES for arm in ARMS for run in RUNS}
    if set(by_key) != expected:
        raise ValueError("Validation matrix is not exactly 3 db x 2 arms x 5 runs")

    report = {"schema": 1, "protocol": "88 D1", "run_dir": str(args.run_dir),
              "cache_root": str(args.cache_root), "inputs": {}, "databases": {},
              "skipped_databases": sorted(skipped)}
    consistency = []
    for db in VALIDATION_DATABASES:
        report["databases"][db] = []
        for run in RUNS:
            cells = {}
            for arm in ARMS:
                cell = by_key[(db, arm, run)]
                path = args.run_dir / cell["prediction_file"]
                digest = sha256(path)
                if digest != cell["prediction_sha256"]:
                    raise ValueError("Prediction SHA mismatch: %s" % path)
                report["inputs"][str(cell["prediction_file"])] = digest
                cells[arm] = load_cell(path)
            pg = load_pg(args.cache_root, db, cells["none"]["query_id"])
            analysis = run_analysis(cells, pg)
            for arm in ARMS:
                truth = cells[arm]["true"]
                tables = cells[arm]["n_tables"]
                masks = {"all": np.ones(len(truth), dtype=bool),
                         "4+": tables >= 4,
                         "low4": (tables >= 4) & (truth < 100)}
                q = q_errors(log_errors(cells[arm]))
                for bucket, mask in masks.items():
                    archived = by_key[(db, arm, run)]["metrics"][bucket]
                    here = {"count": int(mask.sum()),
                            "q50": float(np.percentile(q[mask], 50, method="linear")),
                            "q95": float(np.percentile(q[mask], 95, method="linear"))}
                    consistency.append({
                        "database": db, "run": run, "arm": arm, "bucket": bucket,
                        "archived_count": archived["count"], "recomputed_count": here["count"],
                        "archived_q50": archived["q50"], "recomputed_q50": here["q50"],
                        "archived_q95": archived["q95"], "recomputed_q95": here["q95"],
                        "delta_q50": here["q50"] - archived["q50"],
                        "delta_q95": here["q95"] - archived["q95"]})
            analysis["run"] = run
            report["databases"][db].append(analysis)

    worst = max(abs(row["delta_q95"]) for row in consistency)
    worst50 = max(abs(row["delta_q50"]) for row in consistency)
    counts_ok = all(row["archived_count"] == row["recomputed_count"] for row in consistency)
    report["consistency"] = {
        "cells": len(consistency), "counts_exact": counts_ok,
        "max_abs_delta_q50": worst50, "max_abs_delta_q95": worst,
        "status": "ok" if counts_ok and worst < 1e-9 and worst50 < 1e-9 else "mismatch"}

    aggregate = {}
    for db, runs in report["databases"].items():
        aggregate[db] = {
            "low4_count": median_range([run["low4_count"] for run in runs]),
            "concentration_share": median_range(
                [run["concentration_low4"]["share"] for run in runs]),
            "low4_improvement_share": median_range(
                [run["paired_low4"]["better_share"] for run in runs]),
            "low4_q95_none": median_range(
                [run["spectrum_low4"]["none"]["q95"] for run in runs]),
            "low4_q95_head3": median_range(
                [run["spectrum_low4"]["head3"]["q95"] for run in runs]),
            "low4_q95_ratio": median_range(
                [run["q95_ratio_low4"]["q95_ratio"] for run in runs]),
        }
    report["aggregate"] = aggregate
    report["verdict"] = decide(report["databases"])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1, ensure_ascii=False, allow_nan=False),
                        encoding="utf-8")
    print(json.dumps({"verdict": report["verdict"], "consistency": report["consistency"],
                      "aggregate": aggregate}, indent=1, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
