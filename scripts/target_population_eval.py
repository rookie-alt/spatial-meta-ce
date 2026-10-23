"""Target-database population evaluation at K=0 (doc 101, task one).

Read-only *final* evaluation of an already-frozen checkpoint on a target
database.  Nothing here trains, adapts, selects a checkpoint, or samples
support.  This exists because IMDB has a `test` cache but no `finetune`
split, so the K=0 population comparison is the only honest measurement of
it, and neither existing script can produce it:

* ``scripts/export_source_predictions.py`` refuses target databases
  (``assert_source`` -- correct, must not change);
* ``spatial_meta_ce/evaluate.py`` always fetches ``finetune`` support.

Guards (deliberate, unit-tested):

* every requested database must appear on an explicit ``--allowed-targets``
  list, and that list is written verbatim into ``protocol.json`` as
  ``target_databases_read`` -- reading a target must be an act of record,
  never an accident of a mistyped argument;
* the checkpoint's own ``train/validation`` database exposure must not
  overlap the requested targets;
* PRICE's ``maxima`` is a *layout* parameter, so it must be passed
  explicitly and is checked against the cache's ``inspection.max_counts``
  before any inference; ``protocol.json`` records its derivation rule.

Metric definitions, per-database reporting and JSON writing are reused
verbatim from ``scripts/price_population_eval.py`` (task C) so the IMDB row
cannot drift from the source-side population numbers in definition.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from spatial_meta_ce.direction import sha256_file  # noqa: E402

TARGET_DATABASES = frozenset(("imdb", "stats", "genome", "ergastf1"))
MAXIMA_LENGTH = 4


def _load_price_population_module():
    """Load scripts/price_population_eval.py as a module (it is not a package).

    Reusing its metric/bucket/report functions keeps the IMDB row and the
    task-C source rows defined identically by construction.
    """
    path = Path(__file__).resolve().parent / "price_population_eval.py"
    spec = importlib.util.spec_from_file_location("price_population_eval", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def assert_valid_database_name(db) -> None:
    """Bare, non-empty database id only.

    Mirrors direction.assert_source_database but *without* the target
    refusal -- this script exists to read targets, explicitly.
    """
    if not isinstance(db, str) or not db or db in {".", ".."}:
        raise ValueError(f"Invalid database name {db!r}")
    if "/" in db or "\\" in db or Path(db).name != db:
        raise ValueError(f"Invalid database name {db!r}")


def assert_target_allowed(db: str, allowed) -> None:
    """Refuse any target not explicitly authorised on the command line."""
    assert_valid_database_name(db)
    if db not in allowed:
        raise ValueError(
            f"Refusing to read target database {db!r}: not in allowed list "
            f"{sorted(allowed)}"
        )


def validate_allowed_targets(allowed) -> list:
    """The authorisation list may only hold known target ids (no typos)."""
    result = []
    for db in allowed:
        assert_valid_database_name(db)
        if db not in TARGET_DATABASES:
            raise ValueError(
                f"--allowed-targets takes target names, got {db!r} "
                f"(known: {sorted(TARGET_DATABASES)})"
            )
        result.append(db)
    if not result:
        raise ValueError("--allowed-targets must not be empty")
    return result


def parse_maxima(value) -> np.ndarray:
    """Parse 'a,b,c,d' into four positive integers; refuse anything else."""
    parts = [p.strip() for p in str(value).split(",")]
    if len(parts) != MAXIMA_LENGTH:
        raise ValueError(f"maxima must hold {MAXIMA_LENGTH} values, got {parts!r}")
    try:
        values = [int(p) for p in parts]
    except ValueError:
        raise ValueError(f"maxima values must be integers: {value!r}") from None
    if any(v < 1 for v in values):
        raise ValueError(f"maxima values must be >= 1: {value!r}")
    return np.asarray(values, dtype=np.int64)


def check_price_capacity(maxima, inspection_max_counts, database: str) -> None:
    """Fail before inference if the cache has more features than maxima fits.

    PriceAdapter raises the same way per batch; checking the cache's recorded
    inspection maxima up front attributes the failure to the database and the
    layout parameter instead of surfacing as a mid-run tensor error.
    """
    counts = np.asarray(inspection_max_counts, dtype=np.int64)
    if counts.shape != maxima.shape or (counts > maxima).any():
        raise ValueError(
            f"PRICE maxima {maxima.tolist()} cannot hold database {database!r} "
            f"inspection max_counts {counts.tolist()}"
        )


def assert_checkpoint_frozen(state, databases) -> dict:
    """The checkpoint must not have been trained/validated on any read target."""
    metadata = state["metadata"]
    exposed = set(metadata["train_databases"]) | set(metadata["validation_databases"])
    overlap = exposed & set(databases)
    if overlap:
        raise ValueError(
            f"Checkpoint exposure overlaps read targets: {sorted(overlap)}"
        )
    return {
        "seed": metadata.get("seed"),
        "method": metadata.get("method"),
        "train_databases": sorted(metadata["train_databases"]),
        "validation_databases": sorted(metadata["validation_databases"]),
        "model_config": state["model_config"],
        "transform": state["transform"],
    }


def ours_predictions(model, store, indices, device, batch_size):
    """Zero-shot predictions of our frozen model, already in log space."""
    import torch

    out = []
    for start in range(0, len(indices), batch_size):
        batch = store.batch(indices[start : start + batch_size], device)
        with torch.no_grad():
            out.append(model(batch).detach().cpu().numpy())
    if not out:
        return np.zeros(0, dtype=np.float64)
    return np.concatenate(out).astype(np.float64)


def collect_target(
    store, ours_model, price_model, price_population, device, batch_size
):
    """Build the three aligned prediction vectors for one target database."""
    indices = store.unique_indices()
    true = np.asarray(store.cards[indices, 0], dtype=np.float64)
    pg = np.asarray(store.cards[indices, 1], dtype=np.float64)
    if not (np.isfinite(true).all() and true.min() > 0):
        raise ValueError("Truth cardinalities must be finite and positive")
    if not (np.isfinite(pg).all() and pg.min() > 0):
        raise ValueError("PG estimates must be finite and positive")
    n_tables = np.asarray(store.offsets[indices + 1, 2], dtype=np.int64) - np.asarray(
        store.offsets[indices, 2], dtype=np.int64
    )
    log_preds = {
        "ours_k0": ours_predictions(ours_model, store, indices, device, batch_size),
        "price_k0": price_population.price_predictions(
            price_model, store, indices, device, batch_size
        ),
        "pg": np.log(np.maximum(pg, 1.0)),
    }
    for name, vector in log_preds.items():
        if vector.shape != true.shape or not np.isfinite(vector).all():
            raise ValueError(f"Prediction vector {name} misaligned or nonfinite")
    return log_preds, true, n_tables


def build_protocol(
    args,
    maxima,
    databases,
    allowed,
    frozen,
    inspections,
    reports,
    price_parameters,
    price_population,
    pool,
):
    """Assemble protocol.json exactly as the plan's §4.2 requires."""
    return {
        "schema": 1,
        "purpose": "research-read-only-audit",
        "audit": "p2-2-target-population",
        "target_databases_read": databases,
        "allowed_targets": allowed,
        "metric": price_population.POPULATION_METRIC,
        "metric_definitions": {
            "population": (
                "every unique query_hash of the derived cache, one pass, "
                "no support sampling and no episode structure"
            ),
            "q_error": (
                "exp(abs(max(log_prediction, 0) - log(max(true_cardinality, 1))))"
            ),
            "q50": "percentile(q_error, 50) over the population",
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
                "the frozen checkpoint given on the command line, zero-shot "
                "(no support, no adaptation, no selection)"
            ),
            "price_k0": "the provided PRICE weights, zero-shot, no adaptation",
            "pg": "raw PostgreSQL estimate in the cache (cards[:, 1])",
        },
        "checkpoint": {
            "path": str(Path(args.checkpoint).resolve()),
            "sha256": sha256_file(Path(args.checkpoint)),
            "frozen": True,
            "selection": "none (passed explicitly; the script never chooses)",
            **frozen,
        },
        "price": {
            "checkpoint": str(Path(args.price_checkpoint).resolve()),
            "checkpoint_sha256": sha256_file(Path(args.price_checkpoint)),
            "parameters": int(price_parameters),
            "maxima": maxima.tolist(),
            "maxima_rule": args.maxima_rule,
            "per_database_max_counts": inspections,
            "decoding": (
                "price_log_predictions(): native output is log1p(card)+1, "
                "decoded to log(card)"
            ),
            "adaptation": "none (K=0, no support, no inner steps)",
        },
        "buckets": {
            "table_count": list(price_population.TABLE_BUCKETS),
            "bucket_4plus": "n_tables >= 4 (table buckets 4-5 and 6+)",
        },
        "inputs": {
            "split": args.split,
            "caches": {
                database: {
                    "path": str(
                        (Path(args.cache_root) / args.split / database).resolve()
                    ),
                    "content_hash": pool.get(args.split, database).manifest[
                        "content_hash"
                    ],
                    "unique_queries": reports[database]["n"],
                }
                for database in databases
            },
        },
        "limitations": {
            "few_shot_not_possible": (
                "this target has no finetune split, so only K=0 exists; this "
                "is a K=0 supplement row, not a full four-target acceptance"
            ),
            "seed_coverage": ("the ours_k0 side is a single seed (42): needs_evidence"),
        },
    }


def main(argv):
    parser = argparse.ArgumentParser(
        description="Target-database K=0 population evaluation (read-only)"
    )
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--databases", nargs="+", required=True)
    parser.add_argument(
        "--allowed-targets",
        nargs="+",
        default=["imdb"],
        help="explicit authorisation list written to protocol.json as "
        "target_databases_read; every requested database must appear here",
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--price-checkpoint", required=True)
    parser.add_argument(
        "--maxima",
        required=True,
        help="PRICE layout parameter 'a,b,c,d' (see doc 101 §3), no default",
    )
    parser.add_argument(
        "--maxima-rule",
        default="",
        help="how --maxima was derived; recorded verbatim in protocol.json",
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", required=True)
    parser.add_argument("--protocol", required=True)
    args = parser.parse_args(argv)

    import torch

    from spatial_meta_ce.baselines import PriceAdapter
    from spatial_meta_ce.checkpoints import load_checkpoint
    from spatial_meta_ce.data import StorePool

    price_population = _load_price_population_module()

    allowed = validate_allowed_targets(args.allowed_targets)
    databases = list(args.databases)
    for db in databases:
        assert_target_allowed(db, allowed)
    maxima = parse_maxima(args.maxima)
    if args.device == "cpu":
        torch.set_num_threads(1)

    pool = StorePool(args.cache_root)
    price_model = PriceAdapter(args.price_checkpoint, maxima, args.device)
    ours_model, state = load_checkpoint(args.checkpoint, args.device)
    ours_model.eval()
    frozen = assert_checkpoint_frozen(state, databases)

    reports = {}
    inspections = {}
    for database in databases:
        store = pool.get(args.split, database)
        inspection = store.manifest["inspection"]["max_counts"]
        check_price_capacity(maxima, inspection, database)
        inspections[database] = np.asarray(inspection, dtype=np.int64).tolist()
        started = time.perf_counter()
        log_preds, true, n_tables = collect_target(
            store,
            ours_model,
            price_model,
            price_population,
            args.device,
            args.batch_size,
        )
        reports[database] = price_population.database_report(log_preds, true, n_tables)
        print(
            json.dumps(
                {
                    "database": database,
                    "queries": int(len(true)),
                    "seconds": round(time.perf_counter() - started, 1),
                    **price_population.headline(reports[database]),
                }
            ),
            flush=True,
        )

    protocol = build_protocol(
        args,
        maxima,
        databases,
        allowed,
        frozen,
        inspections,
        reports,
        sum(p.numel() for p in price_model.parameters()),
        price_population,
        pool,
    )

    summary = {
        "schema": 1,
        "purpose": "research-read-only-audit",
        "audit": "p2-2-target-population",
        "target_databases_read": databases,
        "metric": price_population.POPULATION_METRIC,
        "evidence_level": "single_seed_frozen_checkpoint_full_population",
        "needs_evidence": True,
        "needs_evidence_reason": (
            "ours_k0 is one seed (42) of the frozen baseline; the population "
            "metric removes support-sampling noise but not seed noise"
        ),
        "checkpoint": {
            "path": str(Path(args.checkpoint).resolve()),
            "seed": frozen["seed"],
        },
        "price": {
            "checkpoint": str(Path(args.price_checkpoint).resolve()),
            "maxima": maxima.tolist(),
        },
        "inputs": protocol["inputs"],
        "databases": reports,
        "headline": {
            database: price_population.headline(report)
            for database, report in reports.items()
        },
    }
    price_population.write_json(args.protocol, protocol)
    price_population.write_json(args.output, summary)
    print(json.dumps(summary["headline"], indent=2))


if __name__ == "__main__":
    main(sys.argv[1:])
