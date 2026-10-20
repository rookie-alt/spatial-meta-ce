"""P0-3 failure-query audit (doc 93 §3).

Read-only analysis of *which* queries get worse after adaptation, and what
observable features they share. The output decides what input information the
model may be missing (doc 93 §2, suspicion 4); it never selects a model.

Definitions are fixed before any data is read:

* a query's log error is ``|log_pred - log(true)|``, the log-Q the primary
  metric aggregates;
* a query *worsened* when that error exceeds the PG anchor's by more than a
  machine-roundoff tolerance. The anchor is the zero-shot prediction, so
  "worsened" means K>0 adaptation hurt relative to K=0;
* PG direction is defined on the **anchor**, not the prediction: ``over`` when
  PG over-estimated the truth, ``under`` when it under-estimated.

The concentration statistic reuses doc 88 §3.3's shape: the share of total
deterioration contributed by the worst 1% of queries. That doc uses it to
judge low4 adaptation; here it separates "a few extremes" from "diffuse
drift", which decides whether a missing *feature* or a missing *capacity* is
the better hypothesis.

Target databases are refused outright: this module must never read them.
"""

import csv
import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np

TARGETS = frozenset(("imdb", "stats", "genome", "ergastf1"))
PG_DIRECTIONS = ("over", "under", "tie")
TABLE_BUCKETS = ("1", "2-3", "4-5", "6+")
TIE_TOLERANCE = 1e-9


def assert_source_database(db: str) -> None:
    """Refuse a held-out target or a name that is not a bare database id."""
    if db in TARGETS:
        raise ValueError(f"Refusing to audit held-out target database {db!r}")
    if not db or Path(db).name != db:
        raise ValueError(f"Invalid database name {db!r}")


def _log_q(log_pred: float, true: float) -> float:
    return abs(log_pred - math.log(max(true, 1.0)))


def log_q(pred_log: float, true: float) -> float:
    return _log_q(pred_log, true)


def _validate(row: dict, require_tables: bool = True) -> None:
    for key in ("query_hash", "true", "log_pred", "pg_log"):
        if key not in row:
            raise ValueError(f"Row missing {key}")
    if not math.isfinite(row["true"]) or row["true"] < 0:
        raise ValueError("Row truth must be finite and nonnegative")
    for key in ("log_pred", "pg_log"):
        if not math.isfinite(row[key]):
            raise ValueError(f"Row {key} must be finite")
    if require_tables:
        if "n_tables" not in row:
            raise ValueError("Row missing n_tables")
        if not isinstance(row["n_tables"], int) or row["n_tables"] < 1:
            raise ValueError("Row n_tables must be a positive integer")


def is_worsened(row: dict, require_tables: bool = True) -> bool:
    """True when adaptation enlarged this query's log error versus the anchor."""
    _validate(row, require_tables)
    return (
        _log_q(row["log_pred"], row["true"])
        > _log_q(row["pg_log"], row["true"]) + TIE_TOLERANCE
    )


def deterioration(row: dict) -> float:
    """Signed log-error change from adaptation (positive = worse)."""
    _validate(row, require_tables=False)
    return _log_q(row["log_pred"], row["true"]) - _log_q(row["pg_log"], row["true"])


def pg_direction(row: dict) -> str:
    """Whether the PG anchor over- or under-estimated the truth."""
    _validate(row, require_tables=False)
    anchor = row["pg_log"]
    truth = math.log(max(row["true"], 1.0))
    if abs(anchor - truth) <= TIE_TOLERANCE:
        return "tie"
    return "over" if anchor > truth else "under"


def table_bucket(n_tables: int) -> str:
    if n_tables < 2:
        return "1"
    if n_tables < 4:
        return "2-3"
    if n_tables < 6:
        return "4-5"
    return "6+"


def concentration(rows: list) -> dict:
    """How concentrated is total deterioration among the worst queries?

    Uses doc 88 §3.3's 1% share with the same reading: a large share means a
    few extremes dominate (audit those queries); a small share means diffuse
    drift (a missing capacity or feature, not a few outliers).
    """
    if not rows:
        raise ValueError("Concentration requires at least one row")
    changes = np.asarray([deterioration(row) for row in rows], dtype=np.float64)
    positive = changes[changes > 0]
    total = float(positive.sum())
    worsened = int(positive.size)
    if total <= 0:
        return {
            "worsened": 0,
            "total_deterioration": 0.0,
            "top1pct_share": 0.0,
            "shape": "none",
        }
    count = max(1, math.ceil(0.01 * len(changes)))
    top = float(np.sort(positive)[-count:].sum())
    share = top / total
    if share >= 0.40:
        shape = "concentrated"
    elif share < 0.25 and 0.40 <= worsened / len(changes) <= 0.60:
        shape = "diffuse"
    else:
        shape = "inconclusive"
    return {
        "worsened": worsened,
        "total_deterioration": total,
        "top1pct_share": float(share),
        "shape": shape,
    }


def _mean_across_runs(per_run: dict) -> dict:
    """Mean of each scalar metric across support runs.

    The same query appears in every run, so raw rows are never pooled; only the
    summary statistics are averaged.
    """
    reports = list(per_run.values())
    if not reports:
        raise ValueError("No run reports to average")
    count = len(reports)

    def mean(values):
        return float(sum(values) / count)

    directions = PG_DIRECTIONS
    averaged = {
        "runs": count,
        "worsened_fraction": mean([r["worsened_fraction"] for r in reports]),
        "worsened_by_pg_direction": {
            k: mean([r["worsened_by_pg_direction"][k] for r in reports])
            for k in directions
        },
        "top1pct_share": mean([r["concentration"]["top1pct_share"] for r in reports]),
    }
    if all(r.get("table_buckets_available") for r in reports):
        averaged["worsened_by_table_bucket"] = {
            k: mean([r["worsened_by_table_bucket"][k] for r in reports])
            for k in TABLE_BUCKETS
        }
    return averaged


def audit_rows(rows: list, database: str) -> dict:
    """Profile the queries that adaptation made worse in one database.

    Table counts are optional because ``predictions.csv`` does not carry them;
    when absent the table histogram is reported as unavailable rather than
    guessed.
    """
    assert_source_database(database)
    if not rows:
        raise ValueError("Audit requires at least one row")
    has_tables = all("n_tables" in row for row in rows)
    seen = set()
    for row in rows:
        _validate(row, require_tables=has_tables)
        if row["query_hash"] in seen:
            raise ValueError("Duplicate query identity in audit input")
        seen.add(row["query_hash"])
    worsened = [row for row in rows if is_worsened(row, require_tables=has_tables)]
    by_direction = Counter(pg_direction(row) for row in worsened)
    all_direction = Counter(pg_direction(row) for row in rows)
    report = {
        "database": database,
        "queries": len(rows),
        "worsened": len(worsened),
        "worsened_fraction": len(worsened) / len(rows),
        "worsened_by_pg_direction": {k: by_direction.get(k, 0) for k in PG_DIRECTIONS},
        "population_by_pg_direction": {
            k: all_direction.get(k, 0) for k in PG_DIRECTIONS
        },
        "concentration": concentration(rows),
    }
    if has_tables:
        by_tables = Counter(table_bucket(row["n_tables"]) for row in worsened)
        all_tables = Counter(table_bucket(row["n_tables"]) for row in rows)
        report["table_buckets_available"] = True
        report["worsened_by_table_bucket"] = {
            k: by_tables.get(k, 0) for k in TABLE_BUCKETS
        }
        report["population_by_table_bucket"] = {
            k: all_tables.get(k, 0) for k in TABLE_BUCKETS
        }
    else:
        report["table_buckets_available"] = False
    return report


def read_predictions(path: Path) -> list:
    """Read one predictions.csv produced by evaluate() (read-only)."""
    rows = []
    with Path(path).open(newline="") as stream:
        for record in csv.DictReader(stream):
            database = record["database"]
            assert_source_database(database)
            rows.append(
                {
                    "database": database,
                    "run": int(record["run"]),
                    "k": int(record["k"]),
                    "query_hash": record["query_hash"],
                    "true": float(record["true_cardinality"]),
                    "log_pred": float(record["log_prediction"]),
                }
            )
    if not rows:
        raise ValueError("Prediction file is empty")
    return rows


def auditable_rows(rows: list, k: int) -> list:
    """Keep one K>0 slice and attach the K=0 anchor from the same run."""
    adapted = [r for r in rows if r["k"] == k]
    anchors = {
        (r["database"], r["run"], r["query_hash"]): r["log_pred"]
        for r in rows
        if r["k"] == 0
    }
    if not adapted:
        raise ValueError(f"No rows at k={k}")
    joined = []
    for row in adapted:
        key = (row["database"], row["run"], row["query_hash"])
        if key not in anchors:
            raise ValueError("K=0 anchor missing for an adapted row")
        joined.append({**row, "pg_log": anchors[key]})
    return joined


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def main(argv: list) -> None:
    import argparse

    parser = argparse.ArgumentParser(description="P0-3 failure-query audit (read-only)")
    parser.add_argument("--predictions", required=True, nargs="+")
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    reports, skipped = {}, {}
    for path in args.predictions:
        rows = read_predictions(Path(path))
        for db in sorted({r["database"] for r in rows}):
            # One report per (database, run): the same query recurs across
            # support runs, so pooling them would double-count identities.
            runs = sorted({r["run"] for r in rows if r["database"] == db})
            per_run = {}
            for run in runs:
                run_rows = [r for r in rows if r["database"] == db and r["run"] == run]
                try:
                    joined = auditable_rows(run_rows, args.k)
                except ValueError as error:
                    skipped[f"{db}/run{run}"] = str(error)
                    continue
                per_run[f"run{run}"] = audit_rows(joined, db)
            if not per_run:
                continue
            reports[db] = {
                "per_run": per_run,
                "across_runs": _mean_across_runs(per_run),
            }
    if not reports:
        raise SystemExit("No auditable database found")
    write_json(
        Path(args.output),
        {
            "schema": 1,
            "purpose": "research-read-only-audit",
            "k": args.k,
            "target_databases_read": [],
            "databases": reports,
            "skipped": skipped,
        },
    )
    print(
        json.dumps(
            {
                "databases": {
                    db: {
                        "runs": r["across_runs"]["runs"],
                        "worsened_fraction": r["across_runs"]["worsened_fraction"],
                        "top1pct_share": r["across_runs"]["top1pct_share"],
                    }
                    for db, r in reports.items()
                }
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main(sys.argv[1:])
