"""PRICE source-side population comparison (doc 97, task C).

Read-only.  No training, no adaptation, no checkpoint selection, no target
database is ever opened.

Why this exists
---------------
The historical PRICE source-side numbers are *episode* numbers.
``results/price-lr-search-result.json`` reports ``per_db_mean_q95``: the mean
over 25 episodes (5 support sizes x 5 runs) of the Q95 of 64 queries each.
Docs 14/94 line that episode mean up against our full-database Q95.  The two
numbers measure different things, so "how far behind PRICE are we on the source
side" has no answer at present.

This script produces that answer at one granularity.  For every source
validation database it runs PRICE at K=0 over *every unique query* in the
derived cache -- no support sampling, no episodes -- decodes PRICE's native
``log1p(card)+1`` output into the same log space our model predicts in, and
puts three estimators on the identical query set:

* ``ours_k0``  our seed-42 baseline checkpoint's zero-shot prediction, read
  from the export written by ``scripts/export_source_predictions.py``;
* ``price_k0`` the provided PRICE weights, zero-shot;
* ``pg``       the raw PostgreSQL estimate in the cache (``cards[:, 1]``).

The resulting Q50/Q95/log-Q95 are population statistics over the full unique
query set.  They are *not* episode means, and therefore not directly
comparable with the historical PRICE figures; ``protocol.json`` records that
explicitly so the two can never be quoted side by side by accident.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from spatial_meta_ce.direction import (  # noqa: E402
    assert_source_database,
    sha256_file,
    table_bucket,
)
from spatial_meta_ce.metrics import summarize  # noqa: E402

SOURCE_VALIDATION = ("chembl", "legalacts", "telstra")
TABLE_BUCKETS = ("1", "2-3", "4-5", "6+")
FOUR_PLUS_BUCKETS = ("4-5", "6+")
METHODS = ("ours_k0", "price_k0", "pg")
POPULATION_METRIC = "unique-query population Q95 (single pass, NOT episode mean)"


def decode_price_output(encoded):
    """Decode PRICE's native output into the log space we predict in.

    PRICE emits ``log1p(cardinality) + 1``; the shared log space is
    ``log(max(cardinality, 1))``.  The decoding is the project's established
    one (``spatial_meta_ce.baselines.price_log_predictions``), reused here so
    the two code paths cannot drift apart.
    """
    from spatial_meta_ce.baselines import price_log_predictions

    return price_log_predictions(encoded)


def read_k0_predictions(path, databases):
    """Index the K=0 rows of a predictions export by (database, query_hash).

    The export repeats every query once per support run.  The zero-shot
    prediction must be identical in all of them: a disagreement means the
    export is inconsistent and every per-query statistic built on it would be
    ambiguous, so it is an error rather than something to average over.
    """
    wanted = set(databases)
    anchors = {}
    with Path(path).open(newline="", encoding="utf-8-sig") as stream:
        for record in csv.DictReader(stream):
            database = record["database"]
            assert_source_database(database)
            if database not in wanted or int(record["k"]) != 0:
                continue
            key = (database, record["query_hash"])
            prediction = float(record["log_prediction"])
            truth = float(record["true_cardinality"])
            if not math.isfinite(prediction) or not math.isfinite(truth) or truth < 0:
                raise ValueError(f"Invalid K=0 row for {key}")
            if key in anchors and anchors[key] != (prediction, truth):
                raise ValueError(f"K=0 prediction varies across runs: {key}")
            anchors[key] = (prediction, truth)
    if not anchors:
        raise ValueError("Prediction file holds no K=0 rows for the given databases")
    return anchors


def population_metrics(log_pred, true):
    """Q50/Q95/log-Q95 of one estimator over a query population."""
    if len(true) == 0:
        return None
    return summarize(np.asarray(log_pred, dtype=np.float64), true)


def database_report(log_preds, true, n_tables):
    """Summarise the three estimators on one database, overall and per bucket.

    ``n_tables`` buckets follow the project's shared table-count buckets
    (1 / 2-3 / 4-5 / 6+).  ``bucket_4plus`` additionally pools everything with
    at least four tables, which is the bucket the tail of the distribution
    lives in.
    """
    true = np.asarray(true, dtype=np.float64)
    n_tables = np.asarray(n_tables, dtype=np.int64)
    report = {"n": int(len(true))}
    for method in METHODS:
        report[method] = population_metrics(log_preds[method], true)
    buckets = {}
    for name in TABLE_BUCKETS:
        mask = np.asarray([table_bucket(int(t)) == name for t in n_tables])
        entry = {"n": int(mask.sum())}
        for method in METHODS:
            entry[method] = population_metrics(log_preds[method][mask], true[mask])
        buckets[name] = entry
    report["buckets"] = buckets
    four_plus = np.isin(
        np.asarray([table_bucket(int(t)) for t in n_tables]), FOUR_PLUS_BUCKETS
    )
    entry = {
        "n": int(four_plus.sum()),
        "definition": "n_tables >= 4 (table buckets 4-5 and 6+)",
    }
    for method in METHODS:
        entry[method] = population_metrics(
            log_preds[method][four_plus], true[four_plus]
        )
    report["bucket_4plus"] = entry
    return report


def price_predictions(model, store, indices, device, batch_size):
    """Zero-shot PRICE predictions, decoded into log space.

    The batch carries everything ``PriceAdapter.forward`` needs: the four
    feature groups, their masks, the per-group counts and the PG log.  No
    support is involved, so no label encoding is required.
    """
    import torch

    encoded = []
    for start in range(0, len(indices), batch_size):
        batch = store.batch(indices[start : start + batch_size], device)
        with torch.no_grad():
            chunk = model(batch)
        encoded.append(chunk.detach().cpu().numpy())
    if not encoded:
        return np.zeros(0, dtype=np.float64)
    return decode_price_output(np.concatenate(encoded))


def collect_database(store, anchors, database, model, device, batch_size):
    """Build the three aligned prediction vectors for one database."""
    indices = store.unique_indices()
    hashes = [store.rows[int(index)]["query_hash"] for index in indices]
    true = np.asarray(store.cards[indices, 0], dtype=np.float64)
    pg = np.asarray(store.cards[indices, 1], dtype=np.float64)
    n_tables = np.asarray(store.offsets[indices + 1, 2], dtype=np.int64) - np.asarray(
        store.offsets[indices, 2], dtype=np.int64
    )
    ours = np.empty(len(indices), dtype=np.float64)
    for position, query_hash in enumerate(hashes):
        key = (database, query_hash)
        if key not in anchors:
            raise ValueError(f"No K=0 prediction for {key}")
        prediction, truth = anchors[key]
        if truth != float(true[position]):
            raise ValueError(
                f"Truth disagrees between cache and export for {key}: "
                f"{truth!r} != {float(true[position])!r}"
            )
        ours[position] = prediction
    price = price_predictions(model, store, indices, device, batch_size)
    log_preds = {
        "ours_k0": ours,
        "price_k0": np.asarray(price, dtype=np.float64),
        "pg": np.log(np.maximum(pg, 1.0)),
    }
    return log_preds, true, n_tables


def headline(report):
    """Compact per-database comparison of the population Q95.

    A database whose queries all have fewer than four tables has an empty
    ``bucket_4plus``; ``database_report`` records that as ``None`` (an empty
    bucket has no percentile).  The headline then reports ``None`` for each
    method rather than crashing -- the bucket is genuinely absent, not
    skipped, and ``databases.<db>.bucket_4plus.n`` already shows the zero.
    Non-empty buckets produce exactly the same values as before.
    """
    ours = report["ours_k0"]
    price = report["price_k0"]
    row = {
        "q95": {method: report[method]["q95"] for method in METHODS},
        "q50": {method: report[method]["q50"] for method in METHODS},
        "log_q95": {method: report[method]["log_q95"] for method in METHODS},
    }
    row["q95_ratio_price_over_ours"] = (
        float(price["q95"] / ours["q95"]) if ours["q95"] > 0 else None
    )
    four_plus = report["bucket_4plus"]
    row["bucket_4plus_q95"] = {
        method: (None if four_plus[method] is None else four_plus[method]["q95"])
        for method in METHODS
    }
    return row


def write_json(path, value):
    """Write JSON atomically and refuse values JSON cannot represent."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def parse_maxima(value):
    """Parse an explicit 'a,b,c,d' PRICE layout override (doc 101 §3).

    ``main`` only calls this when ``--maxima`` is given, so the default
    path (elementwise max over the compared caches' ``inspection.max_counts``,
    i.e. task C's rule) stays untouched.
    """
    parts = [p.strip() for p in str(value).split(",")]
    if len(parts) != 4:
        raise ValueError(f"maxima must hold 4 values, got {parts!r}")
    try:
        values = [int(p) for p in parts]
    except ValueError:
        raise ValueError(f"maxima values must be integers: {value!r}") from None
    if any(v < 1 for v in values):
        raise ValueError(f"maxima values must be >= 1: {value!r}")
    return np.asarray(values, dtype=np.int64)


def parallel_chunks(databases, jobs):
    """Deterministic round-robin split of databases into at most `jobs` chunks.

    Every database appears exactly once and the per-chunk order is preserved,
    so which worker evaluates a database cannot change its numbers.
    """
    width = min(jobs, len(databases))
    return [tuple(databases[i::width]) for i in range(width)]


def run_database_chunk(task):
    """Worker body for --jobs>1: one model per process over a chunk of databases.

    Same computation as the sequential loop -- one PriceAdapter per worker
    with the identical maxima, same store, same anchors -- in a separate
    process, so torch uses a single thread per worker without contention.
    """
    (
        cache_root,
        split,
        price_checkpoint,
        maxima,
        predictions,
        batch_size,
        databases,
    ) = task
    import torch

    from spatial_meta_ce.baselines import PriceAdapter
    from spatial_meta_ce.data import StorePool

    torch.set_num_threads(1)
    model = PriceAdapter(price_checkpoint, np.asarray(maxima, dtype=np.int64), "cpu")
    pool = StorePool(cache_root)
    anchors = read_k0_predictions(predictions, databases)
    out = {}
    for database in databases:
        store = pool.get(split, database)
        started = time.perf_counter()
        log_preds, true, n_tables = collect_database(
            store, anchors, database, model, "cpu", batch_size
        )
        report = database_report(log_preds, true, n_tables)
        out[database] = report
        print(
            json.dumps(
                {
                    "worker_database": database,
                    "queries": int(len(true)),
                    "seconds": round(time.perf_counter() - started, 1),
                    **headline(report),
                }
            ),
            flush=True,
        )
    return out


def main(argv):
    parser = argparse.ArgumentParser(
        description="PRICE source-side population comparison (read-only)"
    )
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--split", default="pretrain")
    parser.add_argument("--price-checkpoint", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--databases", nargs="+", default=list(SOURCE_VALIDATION))
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--maxima",
        default=None,
        help="explicit 'a,b,c,d' PRICE layout override (doc 101 §3). Omit to "
        "keep the default rule (elementwise max over the compared caches' "
        "inspection.max_counts, i.e. task C's behaviour).",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="process workers for the PRICE pass (>=2 only with --device cpu; "
        "results are identical to jobs=1, only the scheduling differs)",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--protocol", required=True)
    args = parser.parse_args(argv)

    import torch

    from spatial_meta_ce.baselines import PriceAdapter
    from spatial_meta_ce.data import StorePool

    databases = list(args.databases)
    for database in databases:
        assert_source_database(database)
    if args.device == "cpu":
        torch.set_num_threads(1)
    if args.jobs > 1 and args.device != "cpu":
        raise ValueError("--jobs > 1 requires --device cpu")

    pool = StorePool(args.cache_root)
    per_database_maxima = {}
    maxima = np.ones(4, dtype=np.int64)
    for database in databases:
        counts = np.asarray(
            pool.get(args.split, database).manifest["inspection"]["max_counts"],
            dtype=np.int64,
        )
        per_database_maxima[database] = counts.tolist()
        maxima = np.maximum(maxima, counts)
    maxima_rule = (
        "elementwise max of inspection.max_counts over the compared "
        "databases, the same rule the source-side PRICE baseline used"
    )
    if args.maxima is not None:
        maxima = parse_maxima(args.maxima)
        maxima_rule = (
            "explicit --maxima override (passed on the command line; see "
            "doc 101 §3: maxima is a PRICE layout parameter, so numbers "
            "computed under different maxima must not be quoted together)"
        )
        for database, counts in per_database_maxima.items():
            if (np.asarray(counts, dtype=np.int64) > maxima).any():
                raise ValueError(
                    f"explicit maxima {maxima.tolist()} cannot hold database "
                    f"{database!r} max_counts {counts}"
                )
    model = PriceAdapter(args.price_checkpoint, maxima, args.device)

    anchors = read_k0_predictions(args.predictions, databases)
    print(f"k0 anchors: {len(anchors)} unique queries", flush=True)
    databases_report = {}
    if args.jobs > 1:
        import multiprocessing as mp

        # Deterministic round-robin: every database lands in exactly one
        # worker with a fixed maxima, so the per-database numbers match the
        # sequential path exactly; only the scheduling differs.
        chunks = parallel_chunks(databases, args.jobs)
        tasks = [
            (
                args.cache_root,
                args.split,
                args.price_checkpoint,
                maxima.tolist(),
                args.predictions,
                args.batch_size,
                chunk,
            )
            for chunk in chunks
        ]
        context = mp.get_context("spawn")
        collected = {}
        with context.Pool(processes=len(tasks)) as worker_pool:
            for result in worker_pool.map(run_database_chunk, tasks):
                collected.update(result)
        databases_report = {db: collected[db] for db in databases}
    else:
        for database in databases:
            store = pool.get(args.split, database)
            started = time.perf_counter()
            log_preds, true, n_tables = collect_database(
                store, anchors, database, model, args.device, args.batch_size
            )
            databases_report[database] = database_report(log_preds, true, n_tables)
            print(
                json.dumps(
                    {
                        "database": database,
                        "queries": int(len(true)),
                        "seconds": round(time.perf_counter() - started, 1),
                        **headline(databases_report[database]),
                    }
                ),
                flush=True,
            )

    protocol = {
        "schema": 1,
        "purpose": "research-read-only-audit",
        "audit": "p2-2-price-population",
        "target_databases_read": [],
        "metric": POPULATION_METRIC,
        "metric_definitions": {
            "population": (
                "every unique query_hash of the derived cache, one pass, "
                "no support sampling and no episode structure"
            ),
            "q_error": (
                "exp(abs(max(log_prediction, 0) - log(max(true_cardinality, 1))))"
            ),
            "q50": "percentile(q_error, 50) over the population",
            "q80": "percentile(q_error, 80) over the population",
            "q95": "percentile(q_error, 95) over the population",
            "log_q95": (
                "percentile(abs(max(log_prediction, 0) "
                "- log(max(true_cardinality, 1))), 95) over the population"
            ),
            "price_shifted_q*": (
                "exp(abs(logaddexp(max(log_prediction, 0), 0) "
                "- log1p(true_cardinality)))"
            ),
        },
        "estimators": {
            "ours_k0": (
                "our seed-42 baseline checkpoint, zero-shot, read from the "
                "export written by scripts/export_source_predictions.py"
            ),
            "price_k0": "the provided PRICE weights, zero-shot, no adaptation",
            "pg": "raw PostgreSQL estimate in the cache (cards[:, 1])",
        },
        "price": {
            "checkpoint": str(Path(args.price_checkpoint).resolve()),
            "checkpoint_sha256": sha256_file(Path(args.price_checkpoint)),
            "parameters": int(sum(p.numel() for p in model.parameters())),
            "maxima": maxima.tolist(),
            "per_database_max_counts": per_database_maxima,
            "maxima_rule": maxima_rule,
            "decoding": (
                "price_log_predictions(): native output is log1p(card)+1, "
                "decoded to log(card)"
            ),
            "adaptation": "none (K=0, no support, no inner steps)",
        },
        "buckets": {
            "table_count": list(TABLE_BUCKETS),
            "bucket_4plus": "n_tables >= 4 (table buckets 4-5 and 6+)",
        },
        "inputs": {
            "predictions": {
                "path": str(Path(args.predictions).resolve()),
                "sha256": sha256_file(Path(args.predictions)),
            },
            "caches": {
                database: {
                    "path": str(
                        (Path(args.cache_root) / args.split / database).resolve()
                    ),
                    "content_hash": pool.get(args.split, database).manifest[
                        "content_hash"
                    ],
                }
                for database in databases
            },
        },
        "comparability": {
            "comparable_to": (
                "the ours_k0 / price_k0 / pg numbers inside this report, which "
                "share one query population and one metric definition"
            ),
            "not_comparable_to": [
                "results/price-lr-search-result.json per_db_mean_q95, which is "
                "an episode mean over 64-query episodes",
                "the PRICE figures quoted in docs 14 and 94, which are also "
                "episode-granular",
            ],
            "reason": (
                "an episode mean of 64-query Q95s and a single-pass population "
                "Q95 measure different things; quoting them side by side would "
                "compare unlike units"
            ),
        },
    }

    summary = {
        "schema": 1,
        "purpose": "research-read-only-audit",
        "audit": "p2-2-price-population",
        "target_databases_read": [],
        "metric": POPULATION_METRIC,
        "evidence_level": "single_checkpoint_full_population",
        "needs_evidence": True,
        "needs_evidence_reason": (
            "the ours_k0 side is one seed (42) and one export; the population "
            "metric removes sampling noise but not seed noise. Task A of doc 97 "
            "extends the seed coverage."
        ),
        "price": {
            "checkpoint": str(Path(args.price_checkpoint).resolve()),
            "maxima": maxima.tolist(),
        },
        "inputs": protocol["inputs"],
        "databases": databases_report,
        "headline": {
            database: headline(report) for database, report in databases_report.items()
        },
    }
    write_json(args.protocol, protocol)
    write_json(args.output, summary)
    print(json.dumps(summary["headline"], indent=2))


if __name__ == "__main__":
    main(sys.argv[1:])
