#!/usr/bin/env python3
"""Protocol 90 stage A: support sensitivity attribution (A1/A2/A3).

Read-only, CPU-only, no training, no GPU.  Target databases
imdb/stats/genome/ergastf1 never appear in a path; the holdout databases
hepatitis/sakila are rejected before any cache is opened.

Three blocks, all reported per database (never pooled across databases):

  A1  nine observable support-composition features for the 20 support rows
      of every (database, run).
  A2  Spearman correlation between those features and the two outcomes,
      reported separately for n=5 (protocol-81 runs) and n=50 (all draws).
  A3  50 support draws (runs 0..4 reused from protocol 81, runs 5..49 from
      plan-20261019.json).  Each draw rebuilds support AND query because
      protocol 81 requires query == population minus support; only forward
      passes run, with the frozen ``none`` and ``head3`` operator parameters.

Consistency gate (blocking): the metric path is checked against the archived
D1 values to 1e-9 by replaying the D1 NPZ files, and the forward path is
checked against those same NPZ files to 1e-4, the tolerance protocol 88 D2
already used on the CPU-vs-GPU FP32 difference.  A mismatch stops the run.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import meta65_holdout_contract as CONTRACT  # noqa: E402

SHOTS = CONTRACT.SHOTS

DATABASES = ("chembl", "legalacts", "telstra")
FORBIDDEN = ("imdb", "stats", "genome", "ergastf1")
EXCLUDED = ("hepatitis", "sakila")
OPERATORS = ("none", "head3")
ARMS = ("none", "head3")
OFFICIAL_RUNS = tuple(range(5))
NEW_RUNS = tuple(range(5, 50))
ALL_RUNS = tuple(range(50))
SUPPORT_K = 20
MIN_RUNS_FOR_SUMMARY = 3
BATCH_SIZE = 512
LOW4_MIN_TABLES = 4
LOW4_MAX_TRUE = 100
METRIC_TOLERANCE = 1e-9
FORWARD_TOLERANCE = 1e-4
DEADLINE_SECONDS = 2 * 60 * 60
BOOTSTRAP_SEED = 20261019
BOOTSTRAP_RESAMPLES = 2000
BOOTSTRAP_CONFIDENCE = 0.95
BOOTSTRAP_RELATIVE_ERROR = 0.10
PERCENTILES = (5, 25, 50, 75, 95)
SPEARMAN_PREDICTABLE = 0.8
SPEARMAN_NOT_PREDICTABLE = 0.5
FEATURES = (
    "tables_1_3_share",
    "tables_4_share",
    "tables_5_share",
    "tables_6p_share",
    "true_tercile_share",
    "pg_dev_tercile_share",
    "pg_over_share",
    "group_len_mean",
    "log_cartesian_mean",
    "low4_share",
)
DEFAULT_CHECKPOINT = Path(
    "/gly/weilin/spatial-meta-ce-source-holdout-20261018-code-rerun1"
    "/runs/formal-holdout-20261018/R_head3/last.pt")
CHECKPOINT_SHA = "e9924720b02f20bdb4aab02ea09ee63c03d5cd9d701cff3217f9741798d41434"
DEFAULT_PLAN_81 = Path(
    "/gly/weilin/spatial-meta-ce-source-holdout-20261018-code-rerun1"
    "/reports/meta65-source-holdout-20261018/plan-20261018.json")
PLAN_81_SHA = "ba80c37bf7b43b78ac4504307ce74fa50578b0e5d1d05f4300a87bc224f1248f"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def guard_database(database):
    if database in FORBIDDEN:
        raise ValueError("Target database reached protocol 90 stage A: %s" % database)
    if database in EXCLUDED:
        raise ValueError("Holdout database excluded from stage A: %s" % database)
    if database not in DATABASES:
        raise ValueError("Unexpected database in protocol 90 stage A: %s" % database)


def guard_path(path):
    for part in Path(path).parts:
        if part in FORBIDDEN:
            raise ValueError("Target database appears in a path: %s" % path)


def low4_mask(true, tables):
    """Protocol 88 3.2 low4 definition, shared by every block."""
    true = np.asarray(true, dtype=np.float64)
    tables = np.asarray(tables)
    return (tables >= LOW4_MIN_TABLES) & (true < LOW4_MAX_TRUE)


def outcomes(log_pred_none, log_pred_head3, true, tables):
    """better_share and low4 Q95 ratio, matching protocol 88 D1 exactly."""
    none = np.asarray(log_pred_none, dtype=np.float64)
    head3 = np.asarray(log_pred_head3, dtype=np.float64)
    true = np.asarray(true, dtype=np.float64)
    if not (np.isfinite(none).all() and np.isfinite(head3).all()
            and np.isfinite(true).all()):
        raise ValueError("Nonfinite prediction or truth")
    mask = low4_mask(true, tables)
    if not mask.any():
        return {"low4_count": 0, "better_share": None, "better": 0,
                "worse": 0, "equal": 0, "q95_none": None, "q95_head3": None,
                "q95_ratio": None}
    q = {arm: np.exp(np.abs(np.maximum(values, 0) - np.log(np.maximum(true, 1))))
         for arm, values in (("none", none), ("head3", head3))}
    if not all(np.isfinite(values).all() for values in q.values()):
        raise ValueError("Nonfinite Q-error")
    log_true = np.log(np.maximum(true, 1))
    delta = (np.abs(np.maximum(head3, 0) - log_true)
             - np.abs(np.maximum(none, 0) - log_true))[mask]
    better = int((delta < 0).sum())
    worse = int((delta > 0).sum())
    equal = int((delta == 0).sum())
    total = better + worse + equal
    base = float(np.percentile(q["none"][mask], 95, method="linear"))
    top = float(np.percentile(q["head3"][mask], 95, method="linear"))
    return {"low4_count": int(mask.sum()), "better": better, "worse": worse,
            "equal": equal, "better_share": better / total,
            "q95_none": base, "q95_head3": top,
            "q95_ratio": top / base if base else None}


def load_archived(run_dir, database, run):
    """Read the two D1 prediction archives for one (database, run)."""
    cells = {}
    for operator in ARMS:
        path = Path(run_dir) / "predictions" / (
            "R_head3-%s-%s-run%d.npz" % (operator, database, run))
        with np.load(path, allow_pickle=False) as archive:
            cells[operator] = archive["log_pred"].astype(np.float64).reshape(-1)
    return cells


def load_d1_reference(path):
    """Load the archived D1 outcome values keyed by (database, run)."""
    report = json.loads(Path(path).read_text(encoding="utf-8"))
    reference = {}
    for database, runs in report["databases"].items():
        guard_database(database)
        for entry in runs:
            reference[(database, int(entry["run"]))] = {
                "better_share": entry["paired_low4"]["better_share"],
                "q95_ratio": entry["q95_ratio_low4"]["q95_ratio"],
                "low4_count": int(entry["low4_count"]),
                "query_count": int(entry["query_count"]),
            }
    return reference


def check_metric_path(reference, cells, true, tables, database, run):
    """Gate (a): replay the archived NPZ through our metric functions."""
    if (database, run) not in reference:
        raise ValueError("D1 reference lacks %s run %d" % (database, run))
    expected = reference[(database, run)]
    actual = outcomes(cells["none"], cells["head3"], true, tables)
    if actual["better_share"] is None or actual["q95_ratio"] is None:
        raise ValueError("Undefined archived outcome at %s run %d" % (database, run))
    deltas = {
        "better_share": abs(actual["better_share"] - expected["better_share"]),
        "q95_ratio": abs(actual["q95_ratio"] - expected["q95_ratio"]),
        "low4_count": abs(actual["low4_count"] - expected["low4_count"]),
    }
    worst = max(deltas.values())
    if worst > METRIC_TOLERANCE:
        raise ValueError(
            "Metric path disagrees with D1 at %s run %d: %s"
            % (database, run, json.dumps(deltas)))
    return {"database": database, "run": run, "deltas": deltas,
            "status": "ok", "tolerance": METRIC_TOLERANCE}


def check_forward_path(cells, predicted, database, run):
    """Gate (b): CPU forward vs the GPU-produced archive, tolerance 1e-4."""
    diffs = {}
    for operator in ARMS:
        expected = cells[operator]
        actual = np.asarray(predicted[operator], dtype=np.float64)
        if expected.shape != actual.shape:
            raise ValueError("Forward length differs from archive: %s run %d"
                             % (database, run))
        diffs[operator] = float(np.abs(expected - actual).max())
    worst = max(diffs.values())
    if worst > FORWARD_TOLERANCE:
        raise ValueError(
            "CPU forward exceeds the accepted CPU-vs-GPU tolerance at %s run %d: %.3e"
            % (database, run, worst))
    return {"database": database, "run": run, "max_abs_diff": diffs,
            "status": "ok", "tolerance": FORWARD_TOLERANCE}


def tercile_thresholds(values):
    """Pool-wide 1/3 and 2/3 thresholds (protocol 91 section 4)."""
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0:
        raise ValueError("Empty tercile population")
    low, high = np.percentile(values, [100.0 / 3, 200.0 / 3], method="linear")
    return float(low), float(high)


def tercile_ids(values, thresholds):
    low, high = thresholds
    return np.searchsorted([low, high], np.asarray(values, dtype=np.float64),
                           side="right")


def support_features(store, support_indices, population):
    """Observable support-composition features for one support set.

    ``population`` carries the database-wide low4 tercile thresholds so the
    bucket definitions are fixed by the database, not by the drawn support.
    ``true`` defines the buckets and the low4 mask; it is never used to rank
    or select support rows.
    """
    tables = np.diff(store.offsets[:, 2])
    true = np.asarray(store.cards[:, 0], dtype=np.float64)
    pg = np.asarray(store.cards[:, 1], dtype=np.float64)
    support = np.asarray(support_indices, dtype=np.int64)
    if len(support) != SUPPORT_K or len(set(support.tolist())) != SUPPORT_K:
        raise ValueError("Support must hold %d unique indices" % SUPPORT_K)

    support_tables = tables[support]
    support_true = true[support]
    support_pg = pg[support]
    deviation = np.abs(support_pg - support_true) / np.maximum(support_true, 1.0)
    true_ids = tercile_ids(support_true, population["true_terciles"])
    pg_ids = tercile_ids(deviation, population["pg_terciles"])

    # Protocol 91 section 4: mean token count of group 0 and group 3, i.e. the
    # per-row join/filter token lengths averaged over both groups and rows.
    # Each group has its own offset column and support rows are not adjacent,
    # so the row-wise difference is taken inside each offset column.
    lengths = np.stack([
        store.offsets[support + 1, column] - store.offsets[support, column]
        for column in (0, 3)], axis=1)
    cartesian = np.array([float(np.log1p(np.asarray(
        store.groups[2][store.offsets[index, 2]:store.offsets[index + 1, 2], 0]
    )).sum()) for index in support])
    low4 = low4_mask(support_true, support_tables)

    features = {
        "tables_1_3_share": float((support_tables <= 3).mean()),
        "tables_4_share": float((support_tables == 4).mean()),
        "tables_5_share": float((support_tables == 5).mean()),
        "tables_6p_share": float((support_tables >= 6).mean()),
        "true_tercile_share": [float((true_ids == index).mean()) for index in range(3)],
        "pg_dev_tercile_share": [float((pg_ids == index).mean()) for index in range(3)],
        "pg_over_share": float((support_pg > support_true).mean()),
        "group_len_mean": float(lengths.mean()),
        "log_cartesian_mean": float(cartesian.mean()),
        "low4_share": float(low4.mean()),
    }
    if not all(math.isfinite(value) for value in flatten_scalars(features)):
        raise ValueError("Nonfinite support feature")
    return features


def flatten_scalars(features):
    """Yield every scalar of the nested feature dictionary."""
    for key in FEATURES:
        value = features[key]
        if isinstance(value, list):
            yield from value
        else:
            yield value


def population_context(store):
    """Database-wide low4 tercile thresholds for true cardinality and PG error."""
    tables = np.diff(store.offsets[:, 2])
    true = np.asarray(store.cards[:, 0], dtype=np.float64)
    pg = np.asarray(store.cards[:, 1], dtype=np.float64)
    mask = low4_mask(true, tables)
    if not mask.any():
        raise ValueError("Database has no low4 queries")
    deviation = np.abs(pg - true) / np.maximum(true, 1.0)
    return {"true_terciles": tercile_thresholds(true[mask]),
            "pg_terciles": tercile_thresholds(deviation[mask])}


def population_identity(store, database):
    """First-unique population, via the same function the plan builder used."""
    from scripts import build_meta65_holdout_plan as plan_builder

    return plan_builder.validate_store(
        store, database, CONTRACT.UNIQUE[database], CONTRACT.FOUR_PLUS[database])


def rank(values):
    """Average ranks with ties shared, matching scipy's default."""
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    index = 0
    while index < len(values):
        stop = index
        while stop + 1 < len(values) and values[order[stop + 1]] == values[order[index]]:
            stop += 1
        mean = (index + stop) / 2.0 + 1.0
        ranks[order[index:stop + 1]] = mean
        index = stop + 1
    return ranks


def spearman(left, right):
    """Spearman rho as the Pearson correlation of average ranks."""
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if left.shape != right.shape or left.ndim != 1 or len(left) < 3:
        raise ValueError("Spearman needs two equal vectors of length >= 3")
    if not (np.isfinite(left).all() and np.isfinite(right).all()):
        raise ValueError("Nonfinite Spearman input")
    rank_left, rank_right = rank(left), rank(right)
    if rank_left.std() == 0 or rank_right.std() == 0:
        return None
    return float(np.corrcoef(rank_left, rank_right)[0, 1])


def feature_columns(features_by_run, runs):
    """Expand each feature into one column per scalar, in run order."""
    columns = {}
    sample = features_by_run[runs[0]]
    for name in FEATURES:
        value = sample[name]
        if isinstance(value, list):
            for index in range(len(value)):
                columns["%s_%d" % (name, index + 1)] = [
                    features_by_run[run][name][index] for run in runs]
        else:
            columns[name] = [features_by_run[run][name] for run in runs]
    return columns


def correlation_block(features_by_run, outcomes_by_run, runs):
    """Spearman of every feature column against both outcomes."""
    columns = feature_columns(features_by_run, runs)
    result = {"n": len(runs), "spearman": {}}
    for target in ("better_share", "q95_ratio"):
        values = [outcomes_by_run[run][target] for run in runs]
        if any(value is None for value in values) or len(set(values)) < 2:
            result["spearman"][target] = {"skipped": "degenerate outcome"}
            continue
        result["spearman"][target] = {
            name: spearman(column, values) for name, column in columns.items()}
    return result


def spearman_verdict(correlations):
    """Apply the protocol-91 section 5 thresholds to one database."""
    values = []
    for block in correlations["spearman"].values():
        if "skipped" in block:
            continue
        values.extend(value for value in block.values() if value is not None)
    if not values:
        return "inconclusive", "no finite correlation"
    strongest = max(abs(value) for value in values)
    if strongest >= SPEARMAN_PREDICTABLE:
        return "predictable_by_support_composition", "max |rho| %.3f >= %.2f" % (
            strongest, SPEARMAN_PREDICTABLE)
    if strongest < SPEARMAN_NOT_PREDICTABLE:
        return "not_predictable_from_support", "max |rho| %.3f < %.2f" % (
            strongest, SPEARMAN_NOT_PREDICTABLE)
    return "inconclusive", "max |rho| %.3f in [%.2f, %.2f)" % (
        strongest, SPEARMAN_NOT_PREDICTABLE, SPEARMAN_PREDICTABLE)


def distribution(values):
    clean = [float(value) for value in values if value is not None]
    if not clean:
        return {"count": 0}
    array = np.asarray(clean, dtype=np.float64)
    summary = {"count": len(clean), "min": float(array.min()),
               "max": float(array.max()), "mean": float(array.mean()),
               "std": float(array.std(ddof=1)) if len(clean) > 1 else 0.0,
               "distinct_values": len(set(clean))}
    for percentile in PERCENTILES:
        summary["p%d" % percentile] = float(
            np.percentile(array, percentile, method="linear"))
    return summary


def percentile_position(value, sample):
    """Share of the sample at or below ``value``, in percent."""
    array = np.asarray([item for item in sample if item is not None], dtype=np.float64)
    if not len(array) or value is None:
        return None
    return 100.0 * float((array <= value).mean())


def bootstrap_repetitions(values, seed=BOOTSTRAP_SEED):
    """Repetitions needed for the mean to sit within +/-10% at 95% coverage.

    Each draw resamples ``values`` with replacement at size n, takes the mean,
    and keeps the fraction of draws whose mean is within 10% relative error of
    the full-sample mean.  The reported n is the smallest size whose coverage
    reaches 95%, or None while the largest tested size still misses it.
    """
    clean = np.asarray([value for value in values if value is not None],
                       dtype=np.float64)
    base = {"resamples": BOOTSTRAP_RESAMPLES, "confidence": BOOTSTRAP_CONFIDENCE,
            "relative_error": BOOTSTRAP_RELATIVE_ERROR}
    if len(clean) < 2:
        return {"status": "insufficient_sample", **base}
    center = float(clean.mean())
    if center == 0:
        return {"status": "zero_mean", **base}
    rng = np.random.default_rng(seed)
    candidate = int(len(clean))
    limit = max(len(clean) * 64, 3200)
    coverage = 0.0
    while candidate <= limit:
        draws = rng.choice(clean, size=(BOOTSTRAP_RESAMPLES, candidate), replace=True)
        means = draws.mean(axis=1)
        coverage = float((np.abs(means - center)
                          <= BOOTSTRAP_RELATIVE_ERROR * abs(center)).mean())
        if coverage >= BOOTSTRAP_CONFIDENCE:
            return {"status": "reached", "repetitions": candidate,
                    "observed_repetitions": int(len(clean)), "coverage": coverage,
                    **base}
        candidate *= 2
    return {"status": "not_reached_within_bound", "coverage": coverage,
            "observed_repetitions": int(len(clean)), **base}


def a3_block(outcomes_by_run, runs):
    """Distribution, bootstrap and percentile position of the five protocol-81 runs."""
    result = {}
    for target in ("better_share", "q95_ratio"):
        values = [outcomes_by_run[run][target] for run in runs]
        result[target] = {
            **distribution(values),
            "bootstrap": bootstrap_repetitions(
                [value for value in values if value is not None]),
            "percentile_position_of_81_runs": [
                percentile_position(outcomes_by_run[run][target], values)
                for run in OFFICIAL_RUNS if run in outcomes_by_run],
        }
    return result


def overall_verdict(per_database):
    """Every database must reach |rho| >= 0.8 before stage B has a basis."""
    statuses = {database: item["verdict"].get("status")
                for database, item in per_database.items()}
    if not statuses:
        return {"status": "inconclusive", "reason": "no database evaluated",
                "per_database": statuses}
    if all(status == "predictable_by_support_composition"
           for status in statuses.values()):
        return {"status": "predictable_by_support_composition",
                "reason": "every database reaches |rho| >= 0.8",
                "per_database": statuses}
    if any(status == "not_predictable_from_support" for status in statuses.values()):
        return {"status": "not_predictable_from_support",
                "reason": "at least one database has every |rho| < 0.5",
                "per_database": statuses}
    return {"status": "inconclusive",
            "reason": "at least one database is in the intermediate band",
            "per_database": statuses}


def forward_query(model, params, store, indices, device="cpu"):
    """Run one operator over every query index without capturing features."""
    import torch

    from scripts import meta65_joint_runner as runner
    from spatial_meta_ce.meta import forward

    values = []
    with torch.no_grad():
        for start in range(0, len(indices), BATCH_SIZE):
            chunk = indices[start:start + BATCH_SIZE]
            batch = runner.certified_batch(store, chunk, device)
            values.append(forward(model, params, batch).detach()
                          .double().cpu().numpy().reshape(-1))
    result = np.concatenate(values)
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite raw predictions")
    return result


def evaluate_draw(runner, model, store, support, query):
    """One (database, run): adapt twice, forward the query once per operator."""
    support_batch = runner.certified_batch(store, support, "cpu")
    predictions, records = {}, {}
    for operator in OPERATORS:
        params, record = runner.operator_params(model, support_batch, operator)
        expected = {"none": (0, 0), "head3": (3, 107521)}[operator]
        if (record["steps"], record["selected_parameter_count"]) != expected:
            raise ValueError("Unexpected adaptation record for %s" % operator)
        predictions[operator] = forward_query(model, params, store, query)
        records[operator] = {
            "steps": record["steps"],
            "selected_parameter_count": record["selected_parameter_count"],
            "changed_parameter_count": record["changed_parameter_count"]}
    selected = np.asarray(query, dtype=np.int64)
    true = np.asarray(store.cards[selected, 0], dtype=np.float64)
    tables = np.diff(store.offsets[:, 2])[selected]
    return predictions, records, true, tables


def write_report(path, report):
    temporary = Path(path).with_suffix(Path(path).suffix + ".tmp")
    if temporary.exists():
        temporary.unlink()
    temporary.write_text(
        json.dumps(report, indent=1, ensure_ascii=False, allow_nan=False),
        encoding="utf-8")
    temporary.replace(path)


def load_draw_plan(path, database, run):
    plan = json.loads(Path(path).read_text(encoding="utf-8"))
    if plan.get("seed_new") != 20261019 or plan.get("support_size") != 100:
        raise ValueError("Support-coverage plan header changed")
    entries = plan.get("databases", {}).get(database)
    if not entries:
        raise ValueError("Support-coverage plan lacks %s" % database)
    for entry in entries:
        if int(entry["run"]) == run:
            return entry
    raise ValueError("Support-coverage plan lacks %s run %d" % (database, run))


def certify_draw_entry(entry, indices, hashes, database, run):
    """Re-check the plan-90 invariants on the consumer side.

    The builder already certified each draw, but a corrupted or swapped plan
    file would otherwise pass undetected: only ``supports["20"]`` is used for
    adaptation, so the other 80 support rows must still be proven absent from
    the query set, and both digests must be recomputed from the cache itself.
    """
    population = set(indices)
    support_max = [int(value) for value in entry["support_max"]]
    query = [int(value) for value in entry["query"]]
    if len(support_max) != 100 or len(set(support_max)) != 100:
        raise ValueError("Support max must hold 100 unique indices: %s run %d"
                         % (database, run))
    if any(value not in population for value in (*support_max, *query)):
        raise ValueError("Plan index outside the first-unique population: %s run %d"
                         % (database, run))
    if len(set(query)) != len(query) or set(support_max) & set(query):
        raise ValueError("Support/query overlap or duplicate query: %s run %d"
                         % (database, run))
    if query != [index for index in indices if index not in set(support_max)]:
        raise ValueError("Query is not the population minus the whole support: %s run %d"
                         % (database, run))
    identity = dict(zip(indices, hashes, strict=True))
    if entry["query_digest"] != CONTRACT.digest([(i, identity[i]) for i in query]):
        raise ValueError("Query digest changed: %s run %d" % (database, run))
    for k in SHOTS:
        prefix = support_max[:k]
        if (entry["supports"][str(k)] != prefix
                or entry["support_digests"][str(k)]
                != CONTRACT.digest([(i, identity[i]) for i in prefix])):
            raise ValueError("Support prefix/digest changed: %s run %d k=%d"
                             % (database, run, k))
    return support_max[:SUPPORT_K], query


def plan_entry(plan, database, run):
    if database not in plan or run >= len(plan[database]):
        raise ValueError("Protocol-81 plan lacks %s run %d" % (database, run))
    return plan[database][run]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True,
                        help="formal-holdout-20261018 directory with predictions/")
    parser.add_argument("--d1", type=Path, required=True,
                        help="archived D1 report (reports/diagnostic-88-20261019/d1.json)")
    parser.add_argument("--plan-81", type=Path, default=DEFAULT_PLAN_81)
    parser.add_argument("--plan-90", type=Path, required=True,
                        help="support-coverage plan built by build_support_coverage_plan.py")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--cache-root", type=Path, required=True,
                        help="artifacts/cache holding pretrain/<database>")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--runs", type=int, default=len(ALL_RUNS),
                        help="highest run index + 1 to evaluate (default 50)")
    parser.add_argument("--consistency-only", action="store_true",
                        help="run the blocking consistency gate for runs 0..4 and stop")
    args = parser.parse_args(argv)

    if args.runs < MIN_RUNS_FOR_SUMMARY:
        # Spearman needs at least three paired points; a shorter run would
        # evaluate every cell and only then crash in the summary.
        raise ValueError("--runs must be at least %d to compute A2"
                         % MIN_RUNS_FOR_SUMMARY)
    for path in (args.run_dir, args.d1, args.plan_81, args.plan_90,
                 args.checkpoint, args.cache_root, args.out):
        guard_path(path)
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("Stage A requires explicitly hidden CUDA")
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        if os.environ.get(name) != "1":
            raise ValueError("%s must be 1 before CPU execution" % name)

    import torch

    from scripts import meta65_joint_runner as runner
    from spatial_meta_ce.checkpoints import load_checkpoint
    from spatial_meta_ce.data import FeatureStore

    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    started = time.monotonic()

    def remaining():
        left = DEADLINE_SECONDS - (time.monotonic() - started)
        if left <= 0:
            raise TimeoutError("Two-hour stage-A deadline exceeded")
        return left

    input_shas = {
        "checkpoint_sha256": sha256(args.checkpoint),
        "plan_81_sha256": sha256(args.plan_81),
        "plan_90_sha256": sha256(args.plan_90),
        "d1_sha256": sha256(args.d1),
        "code_sha256": sha256(__file__),
    }
    if input_shas["checkpoint_sha256"] != CHECKPOINT_SHA:
        raise ValueError("Checkpoint SHA changed before trusted deserialization")
    if input_shas["plan_81_sha256"] != PLAN_81_SHA:
        raise ValueError("Protocol-81 plan SHA changed")

    reference = load_d1_reference(args.d1)
    plan_81 = json.loads(Path(args.plan_81).read_text(encoding="utf-8"))
    model, state = load_checkpoint(args.checkpoint, "cpu")
    config = state["model_config"]
    if state["next_step"] != 4000 or config.get("arch") != "sum":
        raise ValueError("Checkpoint is not the frozen 4000-step sum64 model")
    if not (config.get("use_pg") is True and config.get("layers") == 2
            and config.get("heads") == 4 and config.get("width") == 64):
        raise ValueError("Checkpoint model configuration changed")
    if any(not torch.isfinite(value).all() for value in state["model"].values()):
        raise FloatingPointError("Nonfinite checkpoint weights")

    runs = tuple(range(args.runs))
    if args.consistency_only:
        runs = OFFICIAL_RUNS
    unknown = [run for run in runs if run not in OFFICIAL_RUNS and run not in NEW_RUNS]
    if unknown:
        raise ValueError("Runs outside protocols 81/90: %s" % unknown)

    report = {
        "schema": 1,
        "protocol": "90 阶段A",
        "inputs": input_shas,
        "environment": {
            "python": sys.version,
            "executable": sys.executable,
            "numpy": np.__version__,
            "torch": torch.__version__,
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "threads": 1,
            "mode": "cpu",
        },
        "sampling": {"seed_new": 20261019, "draws": 50,
                     "reused_from_81": 5, "planned_runs": list(runs)},
        "guard": {"metric_tolerance": METRIC_TOLERANCE,
                  "forward_tolerance": FORWARD_TOLERANCE,
                  "deadline_seconds": DEADLINE_SECONDS},
        "consistency": {"metric": [], "forward": []},
        "per_database": {database: {"features": [], "outcomes": [], "operators": [],
                                    "a1": {}, "a2": {}, "a3": {}, "verdict": {}}
                         for database in DATABASES},
        "progress": {"completed": [], "skipped_existing": []},
    }

    done = set()
    if args.out.exists():
        previous = json.loads(args.out.read_text(encoding="utf-8"))
        for key in ("checkpoint_sha256", "plan_81_sha256", "plan_90_sha256",
                    "d1_sha256", "code_sha256"):
            if previous.get("inputs", {}).get(key) != input_shas[key]:
                raise ValueError("Resume refused: %s changed" % key)
        report = previous
        for database in DATABASES:
            for entry in report["per_database"][database]["outcomes"]:
                done.add((database, int(entry["run"])))
        report["progress"]["skipped_existing"] = sorted(done)
    else:
        args.out.parent.mkdir(parents=True, exist_ok=True)

    stores, populations, identities = {}, {}, {}
    for database in DATABASES:
        guard_database(database)
        store = FeatureStore(Path(args.cache_root) / "pretrain" / database)
        stores[database] = store
        populations[database] = population_context(store)
        identities[database] = population_identity(store, database)

    for database in DATABASES:
        store = stores[database]
        for run in runs:
            remaining()
            key = (database, run)
            if key in done:
                continue
            if run in OFFICIAL_RUNS:
                entry = plan_entry(plan_81, database, run)
                if int(entry["run"]) != run:
                    raise ValueError("Plan run mismatch at %s run %d" % (database, run))
                support = [int(value) for value in entry["supports"]["20"]]
                query = [int(value) for value in entry["query"]]
                if (len(support) != SUPPORT_K or len(set(support)) != SUPPORT_K
                        or len(query) != len(set(query)) or set(support) & set(query)):
                    raise ValueError("Plan support/query invariant broken: %s run %d"
                                     % (database, run))
            else:
                entry = load_draw_plan(args.plan_90, database, run)
                if int(entry["run"]) != run:
                    raise ValueError("Plan run mismatch at %s run %d" % (database, run))
                indices, hashes = identities[database]
                support, query = certify_draw_entry(
                    entry, indices, hashes, database, run)

            predictions, records, true, tables = evaluate_draw(
                runner, model, store, support, query)
            measured = outcomes(predictions["none"], predictions["head3"], true, tables)
            if measured["better_share"] is None or measured["q95_ratio"] is None:
                raise ValueError("Undefined outcome at %s run %d" % (database, run))
            features = support_features(store, support, populations[database])

            if run in OFFICIAL_RUNS:
                archived = load_archived(args.run_dir, database, run)
                metric_check = check_metric_path(
                    reference, archived, true, tables, database, run)
                forward_check = check_forward_path(
                    archived, predictions, database, run)
                report["consistency"]["metric"].append(metric_check)
                report["consistency"]["forward"].append(forward_check)
                print("%s run%d metric=%.3e forward=%.3e" % (
                    database, run, max(metric_check["deltas"].values()),
                    max(forward_check["max_abs_diff"].values())), flush=True)

            report["per_database"][database]["features"].append(
                {"run": run, "source": "81" if run in OFFICIAL_RUNS else "90",
                 "features": features})
            report["per_database"][database]["outcomes"].append(
                {"run": run, "source": "81" if run in OFFICIAL_RUNS else "90",
                 "better_share": measured["better_share"],
                 "low4_q95_ratio": measured["q95_ratio"],
                 "low4_count": measured["low4_count"],
                 "better": measured["better"], "worse": measured["worse"],
                 "equal": measured["equal"],
                 "q95_none": measured["q95_none"], "q95_head3": measured["q95_head3"]})
            report["per_database"][database]["operators"].append(
                {"run": run, "operators": records})
            done.add(key)
            report["progress"]["completed"] = sorted(done)
            write_report(args.out, report)

    if args.consistency_only:
        # Checked after the loop, not inside it: on a resumed run every official
        # cell is already in ``done``, so an in-loop check would never fire and
        # the report would fall through to the summary and be relabelled
        # "complete" with a verdict computed from five draws.
        missing = [(name, official) for name in DATABASES for official in OFFICIAL_RUNS
                   if (name, official) not in done]
        if missing:
            raise ValueError("Consistency gate incomplete: %s" % missing)
        report["phase"] = "consistency_only"
        write_report(args.out, report)
        print(json.dumps(report["consistency"], ensure_ascii=False), flush=True)
        return 0

    for database in DATABASES:
        block = report["per_database"][database]
        features_by_run = {int(item["run"]): item["features"]
                           for item in block["features"]}
        outcomes_by_run = {
            int(item["run"]): {"better_share": item["better_share"],
                               "q95_ratio": item["low4_q95_ratio"]}
            for item in block["outcomes"]}
        n5 = tuple(run for run in OFFICIAL_RUNS if run in features_by_run)
        n50 = tuple(run for run in ALL_RUNS if run in features_by_run)
        block["a1"] = {"features": list(FEATURES), "support_k": SUPPORT_K,
                       "population_terciles": {
                           key: populations[database][key]
                           for key in ("true_terciles", "pg_terciles")}}
        block["a2"] = {
            "n5": correlation_block(features_by_run, outcomes_by_run, n5),
            "n50": correlation_block(features_by_run, outcomes_by_run, n50),
            "thresholds": {"predictable": SPEARMAN_PREDICTABLE,
                           "not_predictable": SPEARMAN_NOT_PREDICTABLE},
        }
        status, reason = spearman_verdict(block["a2"]["n50"])
        block["verdict"] = {
            "status": status, "reason": reason, "basis": "n50",
            "n5_status": spearman_verdict(block["a2"]["n5"])[0]}
        block["a3"] = a3_block(outcomes_by_run, n50)
        write_report(args.out, report)

    report["verdict"] = overall_verdict(report["per_database"])
    report["elapsed_seconds"] = time.monotonic() - started
    report["phase"] = "complete"
    write_report(args.out, report)
    print(json.dumps({
        "verdict": report["verdict"],
        "consistency": {
            "metric_max": max((max(item["deltas"].values())
                               for item in report["consistency"]["metric"]), default=None),
            "forward_max": max((max(item["max_abs_diff"].values())
                                for item in report["consistency"]["forward"]),
                               default=None)},
        "a3": {database: report["per_database"][database]["a3"]
               for database in DATABASES},
    }, indent=1, ensure_ascii=False, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
