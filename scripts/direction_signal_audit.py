"""P2-2 direction-signal audit (doc 96).

Read-only test of one hypothesis: do the cached K=0 features already contain
enough signal to predict *which way* the zero-shot prediction must be
corrected (up or down)?

The label is fixed before any data is read::

    y = 1  if log(max(true, 1)) > log_pred_K0 + TIE_TOLERANCE   (raise)
    y = 0  if log(max(true, 1)) < log_pred_K0 - TIE_TOLERANCE   (lower)
    tie    otherwise, excluded from classification and reported as a count

Features are the K=0 inputs of the model (feature-group summaries, counts,
PG log, table count).  The K=0 *prediction itself is deliberately excluded*:
using it would let the classifier read the label off its own input.  PG log is
kept because PG is an input feature, not an output.

The audit never trains or adapts a model, never selects a checkpoint, and
refuses target databases outright.

Protocol (frozen in ``protocol.json``):

* grouped cross-validation by ``query_hash`` (StratifiedGroupKFold);
* a stricter sensitivity pass that also groups near-duplicate queries sharing
  the same coarse structural signature;
* leave-one-database-out transfer as the test of whether the signal is a
  universal rule or a database-specific bias;
* two model classes (gradient boosting and logistic regression) so the
  conclusion does not depend on one inductive bias;
* a post-hoc direction correction on measured K>0 predictions.  It reuses the
  adaptation magnitude and only flips its direction, so no new magnitude is
  tuned.  This is a correlation analysis, not a new model.

Caveat fixed in advance: the K=0 anchor in ``predictions.csv`` is the model's
zero-shot output, not raw PostgreSQL.  Doc 95's "PG direction" is therefore an
anchor direction; this audit also reports the raw-PG direction for contrast.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from spatial_meta_ce import direction as d  # noqa: E402

TARGETS = d.TARGETS
TIE_TOLERANCE = d.TIE_TOLERANCE
FEATURE_GROUPS = d.FEATURE_GROUPS
TABLE_BUCKETS = d.TABLE_BUCKETS
MODELS = ("hgb", "logistic")
CV_FOLDS = 5
BOOTSTRAP_REPS = 1000
PERMUTATION_REPEATS = 3

assert_source_database = d.assert_source_database
table_bucket = d.table_bucket
direction_label = d.direction_label
log_error = d.log_error
auc_score = d.auc_score
bootstrap_auc_ci = d.bootstrap_auc_ci
sha256_file = d.sha256_file
load_cache = d.load_cache
unique_rows = d.unique_rows
feature_row = d.feature_row
feature_group = d.feature_group
build_matrix = d.build_matrix
structural_signature = d.structural_signature


def cross_check_truths(database, hashes, truths, predictions) -> None:
    """Fail when the CSV label and the cache label disagree for one query.

    The cache is the analysis source of truth, but predictions.csv carries its
    own ``true_cardinality``.  Divergence means the two were produced under
    different extraction/filtering rules, so any label built from them is
    silently ambiguous.
    """
    for query_hash, truth in zip(hashes, truths):
        key = (database, str(query_hash))
        if key not in predictions["truths"]:
            raise ValueError(f"Cache query {key} is missing from predictions")
        if predictions["truths"][key] != float(truth):
            raise ValueError(
                f"Truth disagrees between cache and predictions for {key}: "
                f"{float(truth)!r} != {predictions['truths'][key]!r}"
            )


def read_predictions(path: Path) -> dict:
    """Read predictions.csv once and index by database, query and K.

    Repeated K=0 rows (one per support run) must agree exactly; conflicting
    labels or predictions are a pipeline error, not something to average over.
    Every repeated adapted cell is checked the same way.
    """
    anchors = {}
    truths = {}
    adapted = defaultdict(dict)
    runs = defaultdict(set)
    with Path(path).open(newline="", encoding="utf-8-sig") as stream:
        for record in csv.DictReader(stream):
            database = record["database"]
            assert_source_database(database)
            query_hash = record["query_hash"]
            k = int(record["k"])
            run = int(record["run"])
            true = float(record["true_cardinality"])
            prediction = float(record["log_prediction"])
            if not math.isfinite(true) or true < 0:
                raise ValueError(f"Invalid truth for {database}/{query_hash}")
            if not math.isfinite(prediction):
                raise ValueError(f"Invalid prediction for {database}/{query_hash}")
            key = (database, query_hash)
            if k == 0:
                if key in anchors:
                    if anchors[key] != prediction:
                        raise ValueError(f"K=0 prediction varies across runs: {key}")
                    if truths[key] != true:
                        raise ValueError(f"K=0 truth varies across runs: {key}")
                anchors[key] = prediction
                truths[key] = true
            else:
                cell = (run, query_hash)
                store = adapted[(database, k)]
                if cell in store and store[cell] != prediction:
                    raise ValueError(
                        f"K={k} prediction varies within run {run}: "
                        f"{database}/{query_hash}"
                    )
                store[cell] = prediction
                runs[(database, k)].add(run)
    if not anchors:
        raise ValueError("Prediction file is empty")
    return {
        "anchors": anchors,
        "truths": truths,
        "adapted": adapted,
        "runs": {key: sorted(value) for key, value in runs.items()},
    }


def make_model(name: str, seed: int = 0):
    """Build a fresh classifier; sklearn is imported only when needed."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    if name == "hgb":
        return HistGradientBoostingClassifier(
            max_iter=200,
            learning_rate=0.05,
            max_leaf_nodes=31,
            min_samples_leaf=50,
            l2_regularization=1.0,
            early_stopping=False,
            random_state=seed,
        )
    if name == "logistic":
        return make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LogisticRegression(C=0.1, max_iter=2000, solver="lbfgs"),
        )
    raise ValueError(f"Unknown model {name!r}")


def effective_folds(labels, groups, requested: int = CV_FOLDS) -> int:
    """Cap folds so every class and every group appears in training.

    A held-out fold with a single class has no defined AUC; with fewer groups
    than folds StratifiedGroupKFold refuses to split at all.  Both are data
    properties, not errors, so the fold count adapts and is reported.
    """
    counts = np.bincount(np.asarray(labels, dtype=np.int64), minlength=2)
    distinct = len(set(np.asarray(groups).tolist()))
    return max(2, min(requested, int(counts.min()), distinct))


def grouped_oof(
    matrix, labels, groups, model_name: str, n_splits: int = CV_FOLDS
) -> tuple:
    """Out-of-fold probabilities with a group split; returns (oof, fold_aucs).

    ``fold_aucs`` entries are None when a held-out fold happens to contain a
    single class; the pooled OOF vector is still complete.
    """
    from sklearn.model_selection import StratifiedGroupKFold

    folds = effective_folds(labels, groups, n_splits)
    splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=0)
    oof = np.full(len(labels), np.nan, dtype=np.float64)
    fold_aucs = []
    for train, test in splitter.split(matrix, labels, groups):
        model = make_model(model_name)
        model.fit(matrix[train], labels[train])
        oof[test] = model.predict_proba(matrix[test])[:, 1]
        fold_aucs.append(
            float(auc_score(labels[test], oof[test]))
            if len(np.unique(labels[test])) == 2
            else None
        )
    if np.isnan(oof).any():
        raise ValueError("Cross-validation did not cover every query")
    return oof, fold_aucs


def permutation_importance(matrix, labels, groups, names) -> list:
    """Top feature importances from one held-out fold."""
    from sklearn.inspection import permutation_importance as sklearn_importance
    from sklearn.model_selection import StratifiedGroupKFold

    splitter = StratifiedGroupKFold(n_splits=CV_FOLDS, shuffle=True, random_state=0)
    train, test = next(iter(splitter.split(matrix, labels, groups)))
    model = make_model("hgb")
    model.fit(matrix[train], labels[train])
    result = sklearn_importance(
        model,
        matrix[test],
        labels[test],
        scoring="roc_auc",
        n_repeats=PERMUTATION_REPEATS,
        random_state=0,
    )
    order = np.argsort(result.importances_mean)[::-1]
    return [
        {
            "feature": names[index],
            "importance_mean": float(result.importances_mean[index]),
            "importance_std": float(result.importances_std[index]),
        }
        for index in order[:15]
    ]


def group_ablation(matrix, labels, groups, names) -> dict:
    """Pooled OOF AUC when each feature group is dropped, plus the full model."""
    full, _ = grouped_oof(matrix, labels, groups, "hgb")
    result = {"all": float(auc_score(labels, full))}
    for group in FEATURE_GROUPS:
        keep = [i for i, name in enumerate(names) if feature_group(name) != group]
        subset, _ = grouped_oof(matrix[:, keep], labels, groups, "hgb")
        result[f"without_{group}"] = float(auc_score(labels, subset))
    return result


def bucket_report(labels, scores, n_tables) -> dict:
    report = {}
    for bucket in TABLE_BUCKETS:
        mask = np.asarray([table_bucket(int(value)) == bucket for value in n_tables])
        count = int(mask.sum())
        entry = {"queries": count}
        if count and len(np.unique(labels[mask])) == 2:
            entry["auc"] = float(auc_score(labels[mask], scores[mask]))
        else:
            entry["auc"] = None
        entry["raise_fraction"] = float(np.mean(labels[mask])) if count else None
        report[bucket] = entry
    return report


def _tail_metrics(predictions, truths) -> dict:
    errors = np.asarray(
        [log_error(p, t) for p, t in zip(predictions, truths)], dtype=np.float64
    )
    return {
        "mean_log_error": float(errors.mean()),
        "log_q95": float(np.percentile(errors, 95)),
        "log_q99": float(np.percentile(errors, 99)),
    }


def _direction_metrics(predictions, anchors, truths, tolerance) -> dict:
    errors = np.asarray(
        [log_error(p, t) for p, t in zip(predictions, truths)], dtype=np.float64
    )
    anchor_error = np.asarray(
        [log_error(p, t) for p, t in zip(anchors, truths)], dtype=np.float64
    )
    return {
        **_tail_metrics(predictions, truths),
        "worsened_fraction": float((errors > anchor_error + tolerance).mean()),
    }


def correction_report(anchors, adapted, truths, direction_prob, tolerance):
    """Post-hoc direction correction that reuses the adaptation magnitude.

    ``forced = anchor + sign * abs(adapted - anchor)`` where ``sign`` is the
    predicted direction.  ``gate`` keeps adaptation only when it already moves
    in the predicted direction, else falls back to the anchor.  The oracle
    row uses the *true* direction on the same magnitude, so it bounds the
    direction choice but **not** overshoot: if the adaptation magnitude itself
    is too large, even the oracle direction can still worsen the error.

    The gate picks one of two values, ``adapted`` or ``anchor``, so it cannot
    create damage that adaptation did not already cause: if ``adapted`` is not
    worse than the anchor, whichever branch is taken is also not worse.  Its
    real limitation is the opposite one: it cannot *repair* overshoot, because
    a correctly-signed but far-too-large adaptation is accepted unchanged.
    That is why ``oracle_direction`` (same magnitude, true direction) is
    reported alongside it rather than used as a bound on achievable error.
    """
    anchors = np.asarray(anchors)
    adapted = np.asarray(adapted)
    truths = np.asarray(truths)
    sign = np.where(np.asarray(direction_prob) >= 0.5, 1.0, -1.0)
    truth_sign = np.where(np.log(np.maximum(truths, 1.0)) >= anchors, 1.0, -1.0)
    forced = anchors + sign * np.abs(adapted - anchors)
    oracle = anchors + truth_sign * np.abs(adapted - anchors)
    gate = np.where(np.sign(adapted - anchors) == sign, adapted, anchors)

    anchor_error = np.asarray(
        [log_error(p, t) for p, t in zip(anchors, truths)], dtype=np.float64
    )
    adapted_error = np.asarray(
        [log_error(p, t) for p, t in zip(adapted, truths)], dtype=np.float64
    )
    worsened_adapted = adapted_error > anchor_error + tolerance
    forced_error = np.asarray(
        [log_error(p, t) for p, t in zip(forced, truths)], dtype=np.float64
    )
    gate_error = np.asarray(
        [log_error(p, t) for p, t in zip(gate, truths)], dtype=np.float64
    )
    recovered = worsened_adapted & ~(forced_error > anchor_error + tolerance)
    return {
        "queries": int(len(truths)),
        "anchor": _tail_metrics(anchors, truths),
        "adapted": {
            **_direction_metrics(adapted, anchors, truths, tolerance),
            "worsened": int(worsened_adapted.sum()),
        },
        "forced_direction": {
            **_direction_metrics(forced, anchors, truths, tolerance),
            "recovered_of_worsened": (
                float(recovered.sum() / worsened_adapted.sum())
                if worsened_adapted.any()
                else None
            ),
            "newly_worsened": int(
                ((~worsened_adapted) & (forced_error > anchor_error + tolerance)).sum()
            ),
        },
        "gate": {
            **_direction_metrics(gate, anchors, truths, tolerance),
            "recovered_of_worsened": (
                float(
                    (worsened_adapted & ~(gate_error > anchor_error + tolerance)).sum()
                    / worsened_adapted.sum()
                )
                if worsened_adapted.any()
                else None
            ),
        },
        "oracle_direction": _direction_metrics(oracle, anchors, truths, tolerance),
    }


def _mean(values) -> float | None:
    """Mean over runs that reported a value; ``n`` travels with the caller."""
    clean = [value for value in values if value is not None]
    return float(np.mean(clean)) if clean else None


def analyse_database(
    database: str, cache: dict, predictions: dict, k_values: tuple
) -> dict:
    indices = unique_rows(cache)
    matrix, names = build_matrix(cache, indices)
    hashes = np.asarray([cache["rows"][int(index)]["query_hash"] for index in indices])
    truths = np.asarray([float(cache["cards"][int(index), 0]) for index in indices])
    pg_logs = np.asarray(
        [math.log(max(float(cache["cards"][int(index), 1]), 1.0)) for index in indices]
    )
    cross_check_truths(database, hashes, truths, predictions)
    anchors = np.asarray([predictions["anchors"][(database, h)] for h in hashes])
    labels = np.asarray(
        [direction_label(t, p) for t, p in zip(truths, anchors)], dtype=object
    )
    ties = int(sum(label is None for label in labels))
    keep = np.asarray([label is not None for label in labels])
    labels = labels[keep].astype(np.int64)
    matrix = matrix[keep]
    n_tables = matrix[:, names.index("n_tables")].astype(np.int64)

    query_groups = hashes[keep]
    signature_groups = np.asarray(
        [structural_signature(cache, int(index)) for index in indices]
    )[keep]
    raw_pg_label = np.log(np.maximum(truths[keep], 1.0)) > pg_logs[keep] + 1e-9

    result = {
        "queries": int(keep.sum()),
        "ties_excluded": ties,
        "direction_balance": {
            "raise": int(labels.sum()),
            "lower": int((1 - labels).sum()),
            "raise_fraction": float(labels.mean()),
            "majority_accuracy": float(max(labels.mean(), 1 - labels.mean())),
        },
        "baselines": {
            "pg_minus_anchor": {
                "auc": float(auc_score(labels, pg_logs[keep] - anchors[keep])),
                "accuracy": float(
                    (
                        (pg_logs[keep] - anchors[keep] > 0).astype(np.int64) == labels
                    ).mean()
                ),
            },
            "raw_pg_direction_agreement": float((raw_pg_label == labels).mean()),
        },
        "in_database": {},
        "table_buckets": {},
        "feature_importance": {},
        "near_duplicate_grouped_cv": {},
    }

    direction_prob = None
    for model in MODELS:
        oof, fold_aucs = grouped_oof(matrix, labels, query_groups, model)
        result["in_database"][model] = {
            "auc_pooled_oof": float(auc_score(labels, oof)),
            "auc_ci95": bootstrap_auc_ci(labels, oof),
            "fold_aucs": fold_aucs,
            "fold_auc_mean": _mean(fold_aucs),
            "folds": len(fold_aucs),
            "accuracy": float(((oof >= 0.5).astype(np.int64) == labels).mean()),
        }
        result["table_buckets"][model] = bucket_report(labels, oof, n_tables)
        if model == "hgb":
            direction_prob = oof
            result["feature_importance"] = {
                "permutation_top": permutation_importance(
                    matrix, labels, query_groups, names
                ),
                "group_ablation_auc": group_ablation(
                    matrix, labels, query_groups, names
                ),
            }
            oof_sig, fold_sig = grouped_oof(matrix, labels, signature_groups, "hgb")
            result["support_curve"] = learning_curve(matrix, labels, signature_groups)
            result["near_duplicate_grouped_cv"]["hgb"] = {
                "groups": int(len(set(signature_groups.tolist()))),
                "auc_pooled_oof": float(auc_score(labels, oof_sig)),
                "auc_ci95": bootstrap_auc_ci(labels, oof_sig),
                "fold_aucs": fold_sig,
                "fold_auc_mean": _mean(fold_sig),
                "folds": len(fold_sig),
                "accuracy": float(((oof_sig >= 0.5).astype(np.int64) == labels).mean()),
            }

    result["correction"] = {}
    kept_positions = np.flatnonzero(keep)
    for k in k_values:
        adapted = predictions["adapted"].get((database, k), {})
        runs = predictions["runs"].get((database, k), [])
        per_run = []
        dropped = []
        for run in runs:
            values = [adapted.get((run, hashes[index])) for index in kept_positions]
            if any(value is None for value in values):
                covered = sum(value is not None for value in values)
                dropped.append(
                    {"run": run, "covered": covered, "expected": len(values)}
                )
                continue
            per_run.append(
                correction_report(
                    anchors[keep],
                    np.asarray(values),
                    truths[keep],
                    direction_prob,
                    TIE_TOLERANCE,
                )
            )
        key = f"k{k}"
        if not per_run:
            result["correction"][key] = {
                "model": "hgb",
                "runs": 0,
                "runs_dropped_incomplete": dropped,
            }
            continue
        result["correction"][key] = {
            "model": "hgb",
            "runs": len(per_run),
            "runs_dropped_incomplete": dropped,
            "runs_with_worsened": int(
                sum(r["adapted"]["worsened"] > 0 for r in per_run)
            ),
            "anchor_log_q95": _mean([r["anchor"]["log_q95"] for r in per_run]),
            "adapted_worsened_fraction": _mean(
                [r["adapted"]["worsened_fraction"] for r in per_run]
            ),
            "adapted_log_q95": _mean([r["adapted"]["log_q95"] for r in per_run]),
            "forced_worsened_fraction": _mean(
                [r["forced_direction"]["worsened_fraction"] for r in per_run]
            ),
            "forced_log_q95": _mean(
                [r["forced_direction"]["log_q95"] for r in per_run]
            ),
            "forced_recovered_of_worsened": _mean(
                [r["forced_direction"]["recovered_of_worsened"] for r in per_run]
            ),
            "forced_newly_worsened": _mean(
                [r["forced_direction"]["newly_worsened"] for r in per_run]
            ),
            "gate_worsened_fraction": _mean(
                [r["gate"]["worsened_fraction"] for r in per_run]
            ),
            "gate_log_q95": _mean([r["gate"]["log_q95"] for r in per_run]),
            "gate_recovered_of_worsened": _mean(
                [r["gate"]["recovered_of_worsened"] for r in per_run]
            ),
            "oracle_worsened_fraction": _mean(
                [r["oracle_direction"]["worsened_fraction"] for r in per_run]
            ),
            "oracle_log_q95": _mean(
                [r["oracle_direction"]["log_q95"] for r in per_run]
            ),
        }
    return result


def _prepare(cache: dict, database: str, predictions: dict) -> tuple:
    indices = unique_rows(cache)
    matrix, names = build_matrix(cache, indices)
    hashes = np.asarray([cache["rows"][int(index)]["query_hash"] for index in indices])
    truths = np.asarray([float(cache["cards"][int(index), 0]) for index in indices])
    cross_check_truths(database, hashes, truths, predictions)
    anchors = np.asarray([predictions["anchors"][(database, h)] for h in hashes])
    labels = np.asarray(
        [direction_label(t, p) for t, p in zip(truths, anchors)], dtype=object
    )
    keep = np.asarray([label is not None for label in labels])
    return (
        matrix[keep],
        labels[keep].astype(np.int64),
        anchors[keep],
        truths[keep],
        hashes[keep],
        names,
    )


def leave_one_database_out(caches: dict, predictions: dict) -> dict:
    prepared = {
        database: _prepare(cache, database, predictions)
        for database, cache in caches.items()
    }
    result = {}
    for held in caches:
        train_parts = [prepared[db] for db in caches if db != held]
        matrix = np.concatenate([part[0] for part in train_parts])
        labels = np.concatenate([part[1] for part in train_parts])
        test_matrix, test_labels = prepared[held][0], prepared[held][1]
        entry = {}
        for model in MODELS:
            fitted = make_model(model)
            fitted.fit(matrix, labels)
            probabilities = fitted.predict_proba(test_matrix)[:, 1]
            entry[model] = {
                "auc": float(auc_score(test_labels, probabilities)),
                "auc_ci95": bootstrap_auc_ci(test_labels, probabilities),
                "accuracy": float(
                    ((probabilities >= 0.5).astype(np.int64) == test_labels).mean()
                ),
                "majority_accuracy": float(
                    max(test_labels.mean(), 1 - test_labels.mean())
                ),
            }
        result[held] = entry
    return result


def learning_curve(
    matrix, labels, groups, sizes=(50, 200, 1000, 5000), repeats: int = 3
) -> list:
    """Pooled holdout AUC as the number of labelled queries grows.

    Answers the practical question for a deployment: how many queries would a
    target database have to label before a direction model beats the majority
    baseline.  One fixed 30% group-held-out split, subsampled training only.
    """
    from sklearn.model_selection import GroupShuffleSplit

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=0)
    pool, test = next(iter(splitter.split(matrix, labels, groups)))
    rng = np.random.default_rng(0)
    curve = []
    for size in sizes:
        aucs = []
        for repeat in range(repeats):
            take = min(size, len(pool))
            index = rng.choice(pool, size=take, replace=False)
            if len(np.unique(labels[index])) < 2:
                continue
            model = make_model("hgb", seed=repeat)
            model.fit(matrix[index], labels[index])
            aucs.append(
                auc_score(labels[test], model.predict_proba(matrix[test])[:, 1])
            )
        curve.append(
            {
                "labelled_queries": int(min(size, len(pool))),
                "repeats": len(aucs),
                "auc_mean": _mean(aucs),
                "auc_min": float(min(aucs)) if aucs else None,
                "auc_max": float(max(aucs)) if aucs else None,
            }
        )
    return curve


def verdict(databases: dict, lodo: dict, threshold: float = 0.5) -> dict:
    in_db = {
        db: {
            model: databases[db]["in_database"][model]["auc_ci95"][0] is not None
            and databases[db]["in_database"][model]["auc_ci95"][0] > threshold
            for model in MODELS
        }
        for db in databases
    }
    transfer = {
        db: {
            model: lodo[db][model]["auc_ci95"][0] is not None
            and lodo[db][model]["auc_ci95"][0] > threshold
            for model in MODELS
        }
        for db in lodo
    }
    return {
        "significant_in_database": {db: all(in_db[db].values()) for db in in_db},
        "transfers_across_databases": {
            db: all(transfer[db].values()) for db in transfer
        },
        "any_database_has_signal": any(in_db[db]["hgb"] for db in in_db),
        "all_databases_transfer": all(transfer[db]["hgb"] for db in transfer),
        "status": "needs_evidence",
    }


def main(argv: list) -> None:
    parser = argparse.ArgumentParser(
        description="P2-2 direction-signal audit (read-only)"
    )
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--databases", nargs="+", required=True)
    parser.add_argument("--k", nargs="+", type=int, default=[20, 100])
    parser.add_argument("--output", required=True)
    parser.add_argument("--protocol", required=True)
    args = parser.parse_args(argv)

    predictions = read_predictions(Path(args.predictions))
    databases = list(args.databases)
    caches = {}
    for database in databases:
        assert_source_database(database)
        caches[database] = load_cache(Path(args.cache_root) / database)
        if not any(key[0] == database for key in predictions["anchors"]):
            raise ValueError(f"No predictions for {database}")

    protocol = {
        "schema": 1,
        "purpose": "research-read-only-audit",
        "audit": "p2-2-direction-signal",
        "definition": {
            "label": "sign(log(max(true,1)) - log_pred_K0)",
            "tie_tolerance": TIE_TOLERANCE,
            "worsened": "log error at K>0 exceeds the K=0 anchor by > tolerance",
        },
        "features": {
            "groups": list(FEATURE_GROUPS),
            "excluded": ["log_pred_K0"],
            "note": "K=0 inputs only; PG log is an input feature, not an output",
        },
        "cross_validation": {
            "primary_group": "query_hash",
            "folds": CV_FOLDS,
            "sensitivity_group": "coarse structural signature",
        },
        "models": list(MODELS),
        "bootstrap_reps": BOOTSTRAP_REPS,
        "threshold_auc_ci_lower": 0.5,
        "target_databases_read": [],
    }
    protocol_path = Path(args.protocol)
    protocol_path.parent.mkdir(parents=True, exist_ok=True)
    protocol_path.write_text(json.dumps(protocol, indent=2) + "\n")

    results = {
        database: analyse_database(database, cache, predictions, tuple(args.k))
        for database, cache in caches.items()
    }
    lodo = leave_one_database_out(caches, predictions)
    summary = {
        "schema": 1,
        "purpose": "research-read-only-audit",
        "audit": "p2-2-direction-signal",
        "target_databases_read": [],
        "evidence_level": "single_checkpoint_single_split",
        "needs_evidence": True,
        "inputs": {
            "predictions": {
                "path": str(Path(args.predictions).resolve()),
                "sha256": sha256_file(Path(args.predictions)),
                "rows": sum(len(value) for value in predictions["adapted"].values())
                + len(predictions["anchors"]),
            },
            "caches": {
                database: {
                    "path": str((Path(args.cache_root) / database).resolve()),
                    "content_hash": cache["manifest"]["content_hash"],
                    "queries": len(cache["rows"]),
                }
                for database, cache in caches.items()
            },
        },
        "databases": results,
        "leave_one_database_out": lodo,
        "verdict": verdict(results, lodo),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                database: {
                    model: databases[database]["in_database"][model]["auc_pooled_oof"]
                    for model in MODELS
                }
                for database in databases
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
