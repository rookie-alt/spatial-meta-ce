"""Error-structure diagnosis: why the population Q95 trails PRICE (doc 103).

Read-only.  No training, no adaptation, no checkpoint selection.

``scripts/price_population_eval.py`` answers *how much* worse the Q95 is per
database.  It does not answer *why*, and the two candidate explanations --
"our predictions are systematically mis-centred" versus "our predictions track
the truth less closely" -- have different remedies.  This script separates
them on the same query populations already used for the accuracy tables:

* ``bias_med``   median of ``log_pred - log_true``.  Non-zero means the
                 estimator is systematically high (positive) or low (negative).
* ``over_pct``   share of queries predicted above the truth.  ~50% means the
                 estimator is centred; a value far from 50% locates a
                 one-sided error.
* ``std``        spread of the prediction in log space, comparable with the
                 truth's own spread (``truth_std``); a much smaller value
                 means the estimator is over-shrunk.
* ``corr_truth`` Pearson correlation between prediction and truth in log
                 space -- how well the estimator *ranks* queries.
* ``q50``/``q95`` the same absolute-log-error percentiles as the accuracy
                 tables, so these rows line up with doc 102's Q95 columns.

Two further cuts follow the project's existing diagnostics: the ``4+`` table
bucket (where the tail lives) and a split by *PG direction* (does the
estimator do better where PostgreSQL under- or over-estimates?).  The second
one tests the doc-96 finding that the correction direction, not the magnitude,
is what fails to transfer.

PRICE's ``maxima`` is a layout parameter (doc 101 §3).  The target databases
and the source databases were evaluated under different maxima in doc 102,
so this script takes both explicitly and records which one each database
used; mixing them silently would make the columns incomparable.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from spatial_meta_ce.direction import sha256_file  # noqa: E402

METHODS = ("ours_k0", "price_k0", "pg")
FOUR_PLUS_MIN_TABLES = 4
MAXIMA_LENGTH = 4


def _load_sibling(name):
    """Load a sibling script as a module (scripts/ is not a package)."""
    path = Path(__file__).resolve().parent / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def prediction_stats(log_pred, log_true) -> dict:
    """Bias / spread / ranking-agreement of one estimator.

    ``log_true`` is ``log(max(true_cardinality, 1))`` and ``log_pred`` must be
    in the same space.  All statistics are over the whole population passed
    in; nothing is trimmed.  ``corr_truth`` is None when either side is
    constant, because correlation is undefined there rather than zero.
    """
    pred = np.asarray(log_pred, dtype=np.float64)
    truth = np.asarray(log_true, dtype=np.float64)
    if pred.shape != truth.shape or pred.ndim != 1:
        raise ValueError("prediction and truth must be aligned 1-D arrays")
    if not np.isfinite(pred).all() or not np.isfinite(truth).all():
        raise ValueError("prediction and truth must be finite")
    if len(pred) == 0:
        raise ValueError("cannot summarise an empty population")
    error = pred - truth
    absolute = np.abs(error)
    correlation = None
    if pred.std() > 0 and truth.std() > 0:
        correlation = float(np.corrcoef(pred, truth)[0, 1])
    # q50/q95 are reported in the multiplier space used by doc 102
    # (exp of the absolute log error, so an exact prediction is 1.0).  The
    # log-space percentiles are kept alongside under their own names; quoting
    # a log-space number in a q95 column would understate every gap.
    return {
        "n": int(len(pred)),
        "std": float(pred.std()),
        "bias_med": float(np.median(error)),
        "over_pct": float((error > 0).mean() * 100.0),
        "q50": float(np.exp(np.percentile(absolute, 50))),
        "q95": float(np.exp(np.percentile(absolute, 95))),
        "log_q50": float(np.percentile(absolute, 50)),
        "log_q95": float(np.percentile(absolute, 95)),
        "corr_truth": correlation,
    }


def four_plus_mask(n_tables) -> np.ndarray:
    """Rows whose query joins at least four tables (the tail bucket)."""
    return np.asarray(n_tables, dtype=np.int64) >= FOUR_PLUS_MIN_TABLES


def pg_direction_splits(log_pg, log_true) -> dict:
    """Partition rows by whether PostgreSQL under- or over-estimates.

    Ties (exactly equal in log space) form their own group rather than being
    forced into either side; a forced tie-break would inflate one group with
    rows that carry no directional information.
    """
    pg = np.asarray(log_pg, dtype=np.float64)
    truth = np.asarray(log_true, dtype=np.float64)
    if pg.shape != truth.shape:
        raise ValueError("pg and truth must be aligned")
    return {
        "pg_under": pg < truth,
        "pg_over": pg > truth,
        "pg_equal": pg == truth,
    }


def parse_maxima(value) -> np.ndarray:
    """Parse 'a,b,c,d' into four positive integers."""
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


def subgroup_stats(log_preds, log_true, mask, minimum=1) -> dict:
    """Per-estimator statistics inside a row subset, or None when too small.

    ``minimum`` guards against reporting a percentile computed from a handful
    of rows: a Q95 over fewer than ``minimum`` rows is not a tail statistic,
    it is one query's error wearing a percentile's name.  The row count is
    always reported so a reader can judge.
    """
    mask = np.asarray(mask, dtype=bool)
    count = int(mask.sum())
    if count < minimum:
        return {"n": count, "reported": False}
    entry = {"n": count, "reported": True}
    for method in METHODS:
        entry[method] = prediction_stats(
            np.asarray(log_preds[method])[mask], np.asarray(log_true)[mask]
        )
    return entry


def diagnose_database(log_preds, log_true, n_tables, minimum=30) -> dict:
    """Full error-structure panel for one database."""
    log_true = np.asarray(log_true, dtype=np.float64)
    overview = {"n": int(len(log_true)), "truth_std": float(log_true.std())}
    for method in METHODS:
        overview[method] = prediction_stats(log_preds[method], log_true)
    report = {"overview": overview}
    report["four_plus"] = subgroup_stats(
        log_preds, log_true, four_plus_mask(n_tables), minimum=minimum
    )
    direction = pg_direction_splits(log_preds["pg"], log_true)
    report["pg_direction"] = {
        name: subgroup_stats(log_preds, log_true, mask, minimum=minimum)
        for name, mask in direction.items()
    }
    report["tie_count"] = int(direction["pg_equal"].sum())
    return report


def pairwise_reasoning(database_reports) -> dict:
    """Cross-database summary: where does PRICE's advantage come from?

    Counts, over all databases, where PRICE tracks the truth better and where
    it does not.  A database is a "PRICE better" case when its Q95 is lower;
    the correlation comparison is reported alongside so the two views can be
    checked against each other rather than one being presented alone.
    """
    better_price = []
    better_ours = []
    higher_corr_price = []
    higher_corr_ours = []
    for database, report in database_reports.items():
        ours = report["overview"]["ours_k0"]
        price = report["overview"]["price_k0"]
        if price["q95"] < ours["q95"]:
            better_price.append(database)
        else:
            better_ours.append(database)
        if (price["corr_truth"] or 0) > (ours["corr_truth"] or 0):
            higher_corr_price.append(database)
        else:
            higher_corr_ours.append(database)
    return {
        "q95_better_price": {"databases": better_price, "n": len(better_price)},
        "q95_better_ours": {"databases": better_ours, "n": len(better_ours)},
        "corr_better_price": {
            "databases": higher_corr_price,
            "n": len(higher_corr_price),
        },
        "corr_better_ours": {
            "databases": higher_corr_ours,
            "n": len(higher_corr_ours),
        },
    }


def main(argv):
    parser = argparse.ArgumentParser(
        description="Error-structure diagnosis over target and source databases"
    )
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--target-databases", nargs="+", required=True)
    parser.add_argument("--source-databases", nargs="+", required=True)
    parser.add_argument(
        "--allowed-targets",
        nargs="+",
        default=["imdb", "stats", "genome", "ergastf1"],
        help="explicit authorisation list; written to protocol.json",
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--price-checkpoint", required=True)
    parser.add_argument("--target-maxima", required=True)
    parser.add_argument("--source-maxima", required=True)
    parser.add_argument(
        "--maxima-rule",
        required=True,
        help="how both maxima were derived; recorded verbatim",
    )
    parser.add_argument("--minimum-subgroup", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--output", required=True)
    parser.add_argument("--protocol", required=True)
    args = parser.parse_args(argv)

    import torch

    from spatial_meta_ce.baselines import PriceAdapter
    from spatial_meta_ce.checkpoints import load_checkpoint
    from spatial_meta_ce.data import StorePool
    from spatial_meta_ce.direction import assert_source_database

    target_population = _load_sibling("target_population_eval")
    price_population = _load_sibling("price_population_eval")

    allowed = target_population.validate_allowed_targets(args.allowed_targets)
    for database in args.target_databases:
        target_population.assert_target_allowed(database, allowed)
    for database in args.source_databases:
        assert_source_database(database)

    target_maxima = parse_maxima(args.target_maxima)
    source_maxima = parse_maxima(args.source_maxima)
    torch.set_num_threads(1)

    pool = StorePool(args.cache_root)
    ours_model, state = load_checkpoint(args.checkpoint, "cpu")
    ours_model.eval()
    frozen = target_population.assert_checkpoint_frozen(state, args.target_databases)

    price_models = {
        "target": PriceAdapter(args.price_checkpoint, target_maxima, "cpu"),
        "source": PriceAdapter(args.price_checkpoint, source_maxima, "cpu"),
    }

    plans = [("target", "test", db) for db in args.target_databases]
    plans += [("source", "pretrain", db) for db in args.source_databases]

    reports = {}
    inspections = {}
    for group, split, database in plans:
        store = pool.get(split, database)
        maxima = target_maxima if group == "target" else source_maxima
        target_population.check_price_capacity(
            maxima, store.manifest["inspection"]["max_counts"], database
        )
        inspections[database] = {
            "group": group,
            "split": split,
            "max_counts": np.asarray(
                store.manifest["inspection"]["max_counts"], dtype=np.int64
            ).tolist(),
            "maxima_used": maxima.tolist(),
            "content_hash": store.manifest["content_hash"],
        }
        log_preds, true, n_tables = target_population.collect_target(
            store,
            ours_model,
            price_models[group],
            price_population,
            "cpu",
            args.batch_size,
        )
        log_true = np.log(np.maximum(true, 1.0))
        reports[database] = diagnose_database(
            log_preds, log_true, n_tables, minimum=args.minimum_subgroup
        )
        summary_line = {
            "database": database,
            "group": group,
            "n": reports[database]["overview"]["n"],
        }
        for method in METHODS:
            stats = reports[database]["overview"][method]
            summary_line[method] = {
                key: round(stats[key], 4)
                for key in ("bias_med", "over_pct", "q95", "corr_truth")
            }
        print(json.dumps(summary_line), flush=True)

    protocol = {
        "schema": 1,
        "purpose": "research-read-only-audit",
        "audit": "p2-2-error-structure",
        "target_databases_read": list(args.target_databases),
        "allowed_targets": allowed,
        "metric": "log-space error structure over every unique query",
        "metric_definitions": {
            "log_true": "log(max(true_cardinality, 1))",
            "error": "log_pred - log_true (signed)",
            "bias_med": "median(error): positive means systematically high",
            "over_pct": "100 * share of rows with error > 0; ~50 means centred",
            "std": "standard deviation of log_pred",
            "truth_std": "standard deviation of log_true, for scale",
            "q50_q95": (
                "exp(percentile(|error|, p)): the multiplier error used in doc 102, "
                "so an exact prediction scores 1.0"
            ),
            "log_q50_log_q95": "the same percentiles kept in log space",
            "corr_truth": (
                "Pearson correlation of log_pred with log_true; None when "
                "either side is constant"
            ),
        },
        "estimators": {
            "ours_k0": "the frozen checkpoint given on the command line, zero-shot",
            "price_k0": "the provided PRICE weights, zero-shot",
            "pg": "raw PostgreSQL estimate (cards[:, 1])",
        },
        "checkpoint": {
            "path": str(Path(args.checkpoint).resolve()),
            "sha256": sha256_file(Path(args.checkpoint)),
            "frozen": True,
            **frozen,
        },
        "price": {
            "checkpoint": str(Path(args.price_checkpoint).resolve()),
            "checkpoint_sha256": sha256_file(Path(args.price_checkpoint)),
            "target_maxima": target_maxima.tolist(),
            "source_maxima": source_maxima.tolist(),
            "maxima_rule": args.maxima_rule,
            "maxima_note": (
                "target and source databases were evaluated under different "
                "maxima in doc 102; each database's maxima is recorded in "
                "inputs.per_database so a column can never be quoted under "
                "the wrong layout"
            ),
            "decoding": "price_log_predictions(): log1p(card)+1 -> log(card)",
            "adaptation": "none (K=0)",
        },
        "inputs": {"per_database": inspections},
        "subgroup_policy": {
            "four_plus_min_tables": FOUR_PLUS_MIN_TABLES,
            "minimum_subgroup": args.minimum_subgroup,
            "rule": (
                "a subgroup smaller than minimum_subgroup is recorded with "
                "its row count and reported=false instead of a percentile "
                "computed from too few rows"
            ),
            "pg_ties": (
                "rows where PostgreSQL exactly equals the truth form their "
                "own group (pg_equal) and are not forced into either side"
            ),
        },
        "comparability": {
            "comparable_to": (
                "doc 102's Q95 columns for the same databases: q50/q95 here "
                "use the same error definition and the same query populations"
            ),
            "not_comparable_to": [
                "episode-granular PRICE figures (docs 14/94), which average "
                "64-query Q95s rather than measuring a population",
            ],
        },
    }

    summary = {
        "schema": 1,
        "purpose": "research-read-only-audit",
        "audit": "p2-2-error-structure",
        "target_databases_read": list(args.target_databases),
        "evidence_level": "single_seed_frozen_checkpoint_full_population",
        "needs_evidence": True,
        "needs_evidence_reason": (
            "ours_k0 is one seed (42); this diagnosis inherits every "
            "single-seed caveat of the accuracy tables it explains"
        ),
        "checkpoint": {
            "path": str(Path(args.checkpoint).resolve()),
            "seed": frozen["seed"],
        },
        "price": {
            "target_maxima": target_maxima.tolist(),
            "source_maxima": source_maxima.tolist(),
        },
        "databases": reports,
        "cross_database": pairwise_reasoning(reports),
    }
    price_population.write_json(args.protocol, protocol)
    price_population.write_json(args.output, summary)
    print(json.dumps(summary["cross_database"], indent=2))


if __name__ == "__main__":
    main(sys.argv[1:])
