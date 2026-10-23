#!/usr/bin/env python3
"""Read-only Gate 1 screen for Meta65 last-layer adaptation.

This runner never trains. It reads one fixed source-validation checkpoint and
one frozen support/query plan, evaluates four adaptation operators, and writes
raw paired predictions plus descriptive metrics under a new ``runs/``
directory. The proximal ridge strength is accepted only from an externally
pre-registered manifest; this module has no default or grid-search option.
"""
import argparse
import csv
import hashlib
import json
import math
import os
import platform
import sys
import time
from collections import OrderedDict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

EXPERIMENT = "meta65-lowdim-gate1-20261018"
DATABASES = ("chembl", "legalacts", "telstra")
TARGETS = ("imdb", "stats", "genome", "ergastf1")
OPERATORS = (
    "none",
    "last3_gradient_smooth_l1",
    "last3_gradient_mse",
    "last3_ridge_mse",
)
RUNS = tuple(range(5))
K = 20
SUPPORT_MAX = 100
BATCH_SIZE = 512
INNER_LR = 0.001
INNER_STEPS = 3
UNIQUE = {"chembl": 50000, "legalacts": 49504, "telstra": 50000}
MODEL_CONFIG = {
    "arch": "sum",
    "use_pg": True,
    "width": 64,
    "layers": 2,
    "heads": 4,
}
SHOTS = (5, 10, 20, 50, 100)
MAX_RIDGE_CANDIDATES = 5
CACHE_FILES = (
    "manifest.json",
    "cards.npy",
    "offsets.npy",
    "group0.npy",
    "group1.npy",
    "group2.npy",
    "group3.npy",
)
CONTRASTS = (
    {
        "name": "ridge_vs_gradient_mse",
        "left": "last3_ridge_mse",
        "right": "last3_gradient_mse",
        "attribution": "solver_and_proximal_regularization",
        "description": (
            "Both use MSE-family fitting, but ridge also adds proximal "
            "shrinkage toward the checkpoint. This is not a strict "
            "solver-only contrast; it compares the closed-form regularized "
            "solve with an unregularized three-step gradient."
        ),
    },
    {
        "name": "loss_within_gradient",
        "left": "last3_gradient_mse",
        "right": "last3_gradient_smooth_l1",
        "attribution": "loss",
        "description": (
            "Same three-step last-layer gradient update; left changes the "
            "objective from SmoothL1 to MSE. This isolates the loss."
        ),
    },
    {
        "name": "combined_solver_and_loss",
        "left": "last3_ridge_mse",
        "right": "last3_gradient_smooth_l1",
        "attribution": "combined_not_solver_only",
        "description": (
            "This is a combined solver-plus-loss intervention and is not "
            "solver-only evidence; do not attribute all differences to the "
            "closed-form solver."
        ),
    },
)
PAIRINGS = (
    {
        "name": "last3_gradient_smooth_l1_vs_none",
        "left": "last3_gradient_smooth_l1",
        "right": "none",
        "attribution": "adaptation",
        "description": "Existing SmoothL1 last-layer adaptation versus no update.",
    },
    {
        "name": "last3_gradient_mse_vs_none",
        "left": "last3_gradient_mse",
        "right": "none",
        "attribution": "adaptation",
        "description": "Three-step MSE last-layer adaptation versus no update.",
    },
    {
        "name": "last3_ridge_mse_vs_none",
        "left": "last3_ridge_mse",
        "right": "none",
        "attribution": "adaptation",
        "description": "Proximal ridge MSE last-layer adaptation versus no update.",
    },
    *CONTRASTS,
)


def digest(value):
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()


def file_hash(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def _json_from_text(text):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result

    def constant(value):
        raise ValueError(f"Nonfinite JSON value: {value}")

    return json.loads(text, object_pairs_hook=pairs, parse_constant=constant)


def read_json(path):
    return _json_from_text(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    if temporary.exists():
        raise ValueError(f"Unexpected temporary output: {temporary}")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    )
    temporary.replace(path)


def checked_path(path):
    path = Path(path).expanduser().absolute()
    allowed_aliases = (Path("/tmp"), Path("/var"))
    if any(
        item.is_symlink() and item not in allowed_aliases
        for item in (path, *path.parents)
    ):
        raise ValueError(f"Symlink forbidden: {path}")
    return path.resolve()


def safe_output(path, *, root=ROOT):
    output = checked_path(path)
    if output.exists():
        raise FileExistsError(f"Output must be a new directory: {output}")
    runs = checked_path(Path(root) / "runs")
    if output == runs or not output.is_relative_to(runs):
        raise ValueError("Screen output must stay under this checkout's runs directory")
    if any(part in (*TARGETS, "PRICE-master") for part in output.parts):
        raise ValueError("Target/PRICE output paths are forbidden")
    return output


def _positive_finite(value, name):
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) <= 0
    ):
        raise ValueError(f"{name} must be a finite positive number")
    return float(value)


def validate_lambda_registry(value):
    required = {
        "schema",
        "experiment",
        "candidate_lambdas",
        "selected_lambda",
        "selection_timing",
    }
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError("Ridge lambda registry fields changed")
    if value["schema"] != 1 or value["experiment"] != EXPERIMENT:
        raise ValueError("Ridge lambda registry identity changed")
    candidates = value["candidate_lambdas"]
    if (
        not isinstance(candidates, list)
        or not 1 <= len(candidates) <= MAX_RIDGE_CANDIDATES
    ):
        raise ValueError("Ridge candidate lambda set must contain 1..5 values")
    parsed = [_positive_finite(item, "candidate_lambda") for item in candidates]
    if len(set(parsed)) != len(parsed):
        raise ValueError("Ridge candidate lambdas must be unique")
    selected = _positive_finite(value["selected_lambda"], "selected_lambda")
    if selected not in parsed:
        raise ValueError("selected_lambda was not pre-registered")
    if value["selection_timing"] != "before_query_evaluation":
        raise ValueError("Lambda must be selected before query evaluation")
    return {**value, "candidate_lambdas": parsed, "selected_lambda": selected}


def load_lambda_registry(path):
    raw = checked_path(path).read_bytes()
    registry = validate_lambda_registry(_json_from_text(raw.decode()))
    return registry, hashlib.sha256(raw).hexdigest()


def _index_list(value, name, size):
    if (
        not isinstance(value, list)
        or not value
        or any(type(index) is not int or not 0 <= index < size for index in value)
        or len(set(value)) != len(value)
    ):
        raise ValueError(f"Invalid {name}")
    return value


def certify_plan(plan, rows, database, run, expected_unique=None, k=K):
    """Certify the original K20 prefix without rebuilding the query pool."""
    if database not in DATABASES:
        raise ValueError("Only source validation databases are permitted")
    required = {
        "database",
        "run",
        "support_max",
        "supports",
        "query",
        "query_digest",
        "support_digests",
    }
    if not isinstance(plan, dict) or set(plan) != required:
        raise ValueError("Unexpected support/query plan fields")
    if plan["database"] != database or plan["run"] != run:
        raise ValueError("Support/query plan database or run changed")
    if expected_unique is None:
        expected_unique = UNIQUE[database]
    if (
        len(rows) < SUPPORT_MAX
        or any(
            not isinstance(row.get("query_hash"), str)
            or not row["query_hash"]
            or not isinstance(row.get("id"), str)
            for row in rows
        )
    ):
        raise ValueError("Invalid source rows")
    hashes = [row["query_hash"] for row in rows]
    first = {}
    for index, query_hash in enumerate(hashes):
        first.setdefault(query_hash, index)
    unique_indices = list(first.values())
    if len(unique_indices) != expected_unique:
        raise ValueError("Source unique-query denominator changed")
    support = _index_list(plan["support_max"], "support_max", len(rows))
    if len(support) != SUPPORT_MAX:
        raise ValueError("The frozen plan must retain a 100-query support pool")
    query = _index_list(plan["query"], "query", len(rows))
    if set(support) & set(query):
        raise ValueError("Support/query overlap")
    expected_query = [i for i in unique_indices if i not in set(support)]
    if query != expected_query or len(query) != expected_unique - SUPPORT_MAX:
        raise ValueError("Query was not the complete first-unique pool minus 100")
    supports = plan["supports"]
    if (
        not isinstance(supports, dict)
        or set(supports) != {str(item) for item in SHOTS}
        or any(supports[str(shot)] != support[:shot] for shot in SHOTS)
    ):
        raise ValueError("Nested support prefixes changed")
    if plan["query_digest"] != digest([(i, hashes[i]) for i in query]):
        raise ValueError("Query identity digest changed")
    if plan["support_digests"] != {
        str(shot): digest([(i, hashes[i]) for i in support[:shot]])
        for shot in SHOTS
    }:
        raise ValueError("Support identity digest changed")
    return {"support": support[:k], "query": query, "unique_indices": unique_indices}


def state_digest(model):
    value = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        value.update(name.encode())
        value.update(str(tensor.dtype).encode())
        value.update(str(tuple(tensor.shape)).encode())
        value.update(tensor.detach().cpu().numpy().tobytes())
    return value.hexdigest()


def _last_parameter_names(model):
    head = getattr(model, "head", None)
    if head is None or not hasattr(head, "__len__") or not len(head):
        raise ValueError("Last-layer adaptation requires a sequential head")
    prefix = f"head.{len(head) - 1}."
    names = [
        name for name, _ in model.named_parameters()
        if name.startswith(prefix)
    ]
    if {name.rsplit(".", 1)[-1] for name in names} != {"weight", "bias"}:
        raise ValueError("Final head must have weight and bias")
    return names


def _support_audit(
    model,
    support,
    params,
    operator,
    ridge_lambda,
    ridge_audit=None,
):
    import torch
    import torch.nn.functional as functional

    names = _last_parameter_names(model)
    meta_params = OrderedDict(model.named_parameters())
    with torch.no_grad():
        features = model.head_features(support).detach().to(torch.float64)
        anchor = model.output_anchor(support).detach().to(torch.float64)
        target = support["label"].detach().to(torch.float64) - anchor
        meta_weight = meta_params[names[0]].detach().to(torch.float64).reshape(-1)
        meta_bias = meta_params[names[1]].detach().to(torch.float64).reshape(-1)[0]
        weight = params[names[0]].detach().to(torch.float64).reshape(-1)
        bias = params[names[1]].detach().to(torch.float64).reshape(-1)[0]
        before_head = features @ meta_weight + meta_bias
        after_head = features @ weight + bias
        before_smooth = functional.smooth_l1_loss(
            before_head, target, beta=1.0, reduction="mean")
        after_smooth = functional.smooth_l1_loss(
            after_head, target, beta=1.0, reduction="mean")
        before_mse = functional.mse_loss(before_head, target, reduction="mean")
        after_mse = functional.mse_loss(after_head, target, reduction="mean")
        design = torch.cat(
            [features, torch.ones(features.shape[0], 1, dtype=torch.float64,
                                  device=features.device)],
            dim=1,
        )
        design_condition_number = torch.linalg.cond(design)
        diagnostic_gram = (
            design.T @ design
            + torch.eye(design.shape[1], dtype=torch.float64,
                        device=design.device) * ridge_lambda
        )
        condition_number = torch.linalg.cond(diagnostic_gram)
        update_norm = torch.sqrt(
            torch.linalg.vector_norm(weight - meta_weight).square()
            + (bias - meta_bias).square()
        )
        changed = [
            name for name in names
            if not torch.equal(
                params[name].detach(),
                meta_params[name].detach(),
            )
        ]
        residual_before = before_head - target
        residual_after = after_head - target
    if not torch.isfinite(condition_number):
        raise FloatingPointError("Nonfinite last-layer condition number")
    if not torch.isfinite(design_condition_number):
        raise FloatingPointError("Nonfinite last-layer design condition number")
    if not torch.isfinite(update_norm):
        raise FloatingPointError("Nonfinite last-layer update norm")
    for value in (before_mse, after_mse, before_smooth, after_smooth):
        if not torch.isfinite(value):
            raise FloatingPointError("Nonfinite support residual")
    audit = {
        "operator": operator,
        "method": (
            "proximal_ridge_last"
            if operator == "last3_ridge_mse"
            else "gradient_last"
            if operator != "none"
            else "identity"
        ),
        "training": False,
        "support_count": int(features.shape[0]),
        "feature_dim": int(features.shape[1]),
        "changed_parameter_names": changed,
        "update_norm": float(update_norm),
        "condition_number_kind": "diagnostic_XtX_plus_lambda_I",
        "condition_number": float(condition_number),
        "design_condition_number": float(design_condition_number),
        "support_mse_before": float(before_mse),
        "support_mse_after": float(after_mse),
        "support_smooth_l1_before": float(before_smooth),
        "support_smooth_l1_after": float(after_smooth),
        "support_residual_improved": bool(after_mse < before_mse),
        "support_residuals": [
            {
                "id": str(support["ids"][i]),
                "target": float(target[i]),
                "before": float(before_head[i]),
                "after": float(after_head[i]),
                "residual_before": float(residual_before[i]),
                "residual_after": float(residual_after[i]),
            }
            for i in range(int(features.shape[0]))
        ],
    }
    if ridge_audit is not None:
        audit.update({
            "proximal_lambda": ridge_audit["weight_decay"],
            "solver_condition_number": ridge_audit["condition_number"],
            "solver_support_mse_before": ridge_audit["support_mse_before"],
            "solver_support_mse_after": ridge_audit["support_mse_after"],
            "solver_update_norm": ridge_audit["update_norm"],
        })
    return audit


def adaptation_params(model, support, operator, ridge_lambda):
    """Return a non-mutating parameter map for one operator and its audit."""
    if operator not in OPERATORS:
        raise ValueError(f"Unknown operator: {operator}")
    ridge_lambda = _positive_finite(ridge_lambda, "ridge_lambda")
    model_params = OrderedDict(model.named_parameters())
    if operator == "none":
        params = model_params
        ridge_audit = None
    elif operator == "last3_gradient_smooth_l1":
        from spatial_meta_ce.meta import adapt

        params = adapt(
            model,
            support,
            lr=INNER_LR,
            steps=INNER_STEPS,
            scope="last",
            loss="smooth_l1",
            training=False,
        )
        ridge_audit = None
    elif operator == "last3_gradient_mse":
        from spatial_meta_ce.meta import adapt

        params = adapt(
            model,
            support,
            lr=INNER_LR,
            steps=INNER_STEPS,
            scope="last",
            loss="mse",
            training=False,
        )
        ridge_audit = None
    else:
        from spatial_meta_ce.lowdim_adaptation import ridge_last

        params, ridge_audit = ridge_last(
            model,
            support,
            weight_decay=ridge_lambda,
            bias_decay=ridge_lambda,
            training=False,
            return_audit=True,
        )
    audit = _support_audit(
        model,
        support,
        params,
        operator,
        ridge_lambda=ridge_lambda,
        ridge_audit=ridge_audit,
    )
    if operator == "last3_ridge_mse":
        audit["proximal_lambda"] = ridge_lambda
        if not audit["support_residual_improved"]:
            raise ValueError(
                "Proximal ridge did not improve the deployed support residual")
    return params, audit


def q_errors(log_pred, true):
    import numpy as np

    log_pred = np.asarray(log_pred, dtype=np.float64).reshape(-1)
    true = np.asarray(true, dtype=np.float64).reshape(-1)
    if (
        not len(true)
        or log_pred.shape != true.shape
        or not np.isfinite(log_pred).all()
        or not np.isfinite(true).all()
        or (true < 0).any()
    ):
        raise ValueError("Invalid or empty predictions/labels")
    log_true = np.log(np.maximum(true, 1.0))
    with np.errstate(over="raise", invalid="raise"):
        q = np.exp(np.abs(np.maximum(log_pred, 0.0) - log_true))
        log_q = np.abs(np.maximum(log_pred, 0.0) - log_true)
    if not np.isfinite(q).all() or not np.isfinite(log_q).all():
        raise FloatingPointError("Nonfinite standard Q-error")
    return q, log_q


def bucket_metrics(q, mask):
    import numpy as np

    mask = np.asarray(mask, dtype=bool)
    count = int(mask.sum())
    if count == 0:
        return {"count": 0, "status": "empty", "q50": None, "q95": None}
    values = np.asarray(q, dtype=np.float64)[mask]
    return {
        "count": count,
        "status": "below_min_n" if count < 30 else "ok",
        "q50": float(np.percentile(values, 50, method="linear")),
        "q95": float(np.percentile(values, 95, method="linear")),
    }


def metrics_for_identity(log_pred, identity):
    import numpy as np

    true = np.asarray(identity["true"], dtype=np.float64)
    tables = np.asarray(identity["n_tables"], dtype=np.int16)
    q, _ = q_errors(log_pred, true)
    return {
        "all": bucket_metrics(q, np.ones(len(true), dtype=bool)),
        "four_plus": bucket_metrics(q, tables >= 4),
        "low4": bucket_metrics(q, (true < 100) & (tables >= 4)),
    }


def evaluate_database_run(
    model,
    store,
    database,
    run,
    *,
    support_indices,
    query_indices,
    table_counts,
    ridge_lambda,
    device,
    batch_size=BATCH_SIZE,
):
    """Evaluate all operators on one identical database/run plan."""
    import numpy as np
    import torch

    from spatial_meta_ce.meta import forward

    if database not in DATABASES:
        raise ValueError("Only source validation databases are permitted")
    support_indices = list(support_indices)
    query_indices = list(query_indices)
    table_counts = np.asarray(table_counts, dtype=np.int16)
    if (
        len(support_indices) != K
        or not query_indices
        or len(set(support_indices)) != len(support_indices)
        or len(set(query_indices)) != len(query_indices)
        or set(support_indices) & set(query_indices)
        or any(index < 0 or index >= len(store.rows)
               for index in (*support_indices, *query_indices))
        or len(table_counts) != len(store.rows)
    ):
        raise ValueError("Invalid support/query plan or table counts")
    before = state_digest(model)
    # Load support and adapt before any query batch is materialized. Query
    # labels therefore cannot influence support selection or adaptation.
    support = store.batch(support_indices, device)
    params_by_operator, audits = {}, {}
    for operator in OPERATORS:
        params, audit = adaptation_params(
            model, support, operator, ridge_lambda=ridge_lambda)
        params_by_operator[operator] = params
        for index, item in zip(support_indices, audit["support_residuals"]):
            item["query_hash"] = store.rows[index]["query_hash"]
        audits[operator] = audit
        if state_digest(model) != before:
            raise ValueError("Adaptation mutated the shared checkpoint")

    log_pred = {operator: [] for operator in OPERATORS}
    true_values = []
    query_hash = []
    query_id = []
    for start in range(0, len(query_indices), batch_size):
        indices = query_indices[start:start + batch_size]
        batch = store.batch(indices, device)
        expected_ids = [store.rows[index]["id"] for index in indices]
        if list(batch["ids"]) != expected_ids:
            raise ValueError("Query identity order changed")
        query_hash.extend(store.rows[index]["query_hash"] for index in indices)
        query_id.extend(expected_ids)
        true_values.extend(
            batch["true"].detach().cpu().numpy().astype(np.float64).tolist()
        )
        with torch.no_grad():
            for operator in OPERATORS:
                values = forward(
                    model, params_by_operator[operator], batch
                ).detach().cpu().numpy().astype(np.float64)
                if not np.isfinite(values).all():
                    raise FloatingPointError(
                        f"Nonfinite raw prediction for {operator}")
                log_pred[operator].append(values)
    if state_digest(model) != before:
        raise ValueError("Evaluation mutated the shared checkpoint")
    true = np.asarray(true_values, dtype=np.float64)
    tables = table_counts[np.asarray(query_indices, dtype=np.int64)]
    identity = {
        "query_hash": np.asarray(query_hash, dtype=str),
        "query_id": np.asarray(query_id, dtype=str),
        "true": true,
        "n_tables": tables.astype(np.int16),
    }
    predictions = {
        operator: np.concatenate(values)
        for operator, values in log_pred.items()
    }
    metrics = {
        operator: metrics_for_identity(predictions[operator], identity)
        for operator in OPERATORS
    }
    for operator in OPERATORS:
        for bucket in ("four_plus", "low4"):
            if metrics[operator][bucket]["count"] == 0:
                raise ValueError(
                    f"Required {bucket} bucket is empty for {operator}")
    return {
        "database": database,
        "run": run,
        "identity": identity,
        "operators": {
            operator: {
                "log_pred": predictions[operator],
                "metrics": metrics[operator],
                "support_audit": audits[operator],
            }
            for operator in OPERATORS
        },
        "state_before": before,
        "state_after": state_digest(model),
    }


def write_pair_archive(path, result):
    import numpy as np

    identity = result["identity"]
    values = {
        "query_hash": np.asarray(identity["query_hash"], dtype="S64"),
        "query_id": np.asarray(identity["query_id"], dtype="S128"),
        "true": np.asarray(identity["true"], dtype=np.float64),
        "n_tables": np.asarray(identity["n_tables"], dtype=np.int16),
        "four_plus": np.asarray(identity["n_tables"] >= 4, dtype=np.bool_),
        "low4": np.asarray(
            (identity["true"] < 100) & (identity["n_tables"] >= 4),
            dtype=np.bool_,
        ),
    }
    for operator in OPERATORS:
        values[f"{operator}_log_pred"] = np.asarray(
            result["operators"][operator]["log_pred"], dtype=np.float64)
        values[f"{operator}_q"] = q_errors(
            result["operators"][operator]["log_pred"], identity["true"])[0]
    for pairing in PAIRINGS:
        left_q, left_log_q = q_errors(
            result["operators"][pairing["left"]]["log_pred"],
            identity["true"],
        )
        right_q, right_log_q = q_errors(
            result["operators"][pairing["right"]]["log_pred"],
            identity["true"],
        )
        values[f"{pairing['name']}_delta_log_q"] = left_log_q - right_log_q
        values[f"{pairing['name']}_q_ratio"] = left_q / right_q
    np.savez_compressed(path, **values)


def _metric_fields():
    return {
        f"{bucket}_{metric}": None
        for bucket in ("all", "four_plus", "low4")
        for metric in ("q50", "q95")
    }


def summarize_cells(cells):
    expected = {
        (operator, database, run)
        for operator in OPERATORS
        for database in DATABASES
        for run in RUNS
    }
    indexed = {}
    for cell in cells:
        key = (cell["operator"], cell["database"], cell["run"])
        if key not in expected or key in indexed:
            raise ValueError("Unexpected or duplicate screen cell")
        for bucket in ("all", "four_plus", "low4"):
            metric = cell["metrics"][bucket]
            if (
                metric["count"] < 1
                or metric["q50"] is None
                or metric["q95"] is None
                or not math.isfinite(float(metric["q50"]))
                or not math.isfinite(float(metric["q95"]))
                or float(metric["q50"]) < 1
                or float(metric["q95"]) < 1
            ):
                raise ValueError("Screen metrics must be finite Q-errors >= 1")
        indexed[key] = cell
    if set(indexed) != expected:
        raise ValueError("Screen requires exactly 60 cells")
    for database in DATABASES:
        for run in RUNS:
            reference = indexed[(OPERATORS[0], database, run)]
            for operator in OPERATORS[1:]:
                cell = indexed[(operator, database, run)]
                for field in ("query_digest", "support_digest"):
                    if cell[field] != reference[field]:
                        raise ValueError(f"Screen pairing mismatch: {field}")

    def mean_metric(records, bucket, metric):
        values = [float(row["metrics"][bucket][metric]) for row in records]
        return math.fsum(values) / len(values)

    summaries = {}
    field_names = tuple(_metric_fields())
    for operator in OPERATORS:
        per_database = {}
        for database in DATABASES:
            records = [
                indexed[(operator, database, run)] for run in RUNS
            ]
            per_database[database] = {
                field: mean_metric(
                    records,
                    field.rsplit("_", 1)[0],
                    field.rsplit("_", 1)[1],
                )
                for field in field_names
            }
        macro = {
            field: math.fsum(
                per_database[database][field] for database in DATABASES
            ) / len(DATABASES)
            for field in field_names
        }
        summaries[operator] = {
            "per_database": per_database,
            "macro": macro,
            "cell_count": len(RUNS) * len(DATABASES),
        }
    return {
        "schema": 1,
        "experiment": EXPERIMENT,
        "k": K,
        "metric": (
            "five-run arithmetic mean within each database, then equal "
            "database macro; standard unique-query Q50/Q95"
        ),
        "operators": summaries,
        "selection": "not_performed_by_runner",
    }


def _bucket_mask(identity, bucket):
    import numpy as np

    true = np.asarray(identity["true"], dtype=np.float64)
    tables = np.asarray(identity["n_tables"], dtype=np.int16)
    if bucket == "all":
        return np.ones(len(true), dtype=bool)
    if bucket == "four_plus":
        return tables >= 4
    if bucket == "low4":
        return (true < 100) & (tables >= 4)
    raise ValueError(f"Unknown bucket: {bucket}")


def _pair_stats(left_log_pred, right_log_pred, identity, bucket):
    import numpy as np

    left_q, left_log_q = q_errors(left_log_pred, identity["true"])
    right_q, right_log_q = q_errors(right_log_pred, identity["true"])
    mask = _bucket_mask(identity, bucket)
    count = int(mask.sum())
    if count == 0:
        return {"query_count": 0, "status": "empty"}
    left_q, right_q = left_q[mask], right_q[mask]
    left_log_q, right_log_q = left_log_q[mask], right_log_q[mask]
    delta_log_q = left_log_q - right_log_q
    q95_ratio = float(
        np.percentile(left_q, 95, method="linear")
        / np.percentile(right_q, 95, method="linear")
    )
    q50_ratio = float(
        np.percentile(left_q, 50, method="linear")
        / np.percentile(right_q, 50, method="linear")
    )
    if not math.isfinite(q95_ratio) or not math.isfinite(q50_ratio):
        raise FloatingPointError("Nonfinite paired Q-ratio")
    return {
        "query_count": count,
        "status": "below_min_n" if count < 30 else "ok",
        "q50_ratio": q50_ratio,
        "q95_ratio": q95_ratio,
        "mean_delta_log_q": float(np.mean(delta_log_q)),
        "median_delta_log_q": float(np.median(delta_log_q)),
        "mean_delta_q": float(np.mean(left_q - right_q)),
        "improved_count": int((left_q < right_q).sum()),
        "worsened_count": int((left_q > right_q).sum()),
        "tied_count": int((left_q == right_q).sum()),
        "improved_fraction": float(np.mean(left_q < right_q)),
        "lower_q_error_is_better": True,
    }


def summarize_pairings(entries):
    import numpy as np

    required_entry = {"database", "run", "identity", "log_pred"}
    runs = []
    for entry in entries:
        if set(entry) != required_entry:
            raise ValueError("Unexpected paired-run fields")
        database = entry["database"]
        run = entry["run"]
        if database not in DATABASES or run not in RUNS:
            raise ValueError("Target or unknown paired database/run")
        identity = entry["identity"]
        expected = {"query_hash", "query_id", "true", "n_tables"}
        if set(identity) != expected:
            raise ValueError("Unexpected paired identity fields")
        lengths = {len(np.asarray(value)) for value in identity.values()}
        if len(lengths) != 1 or not lengths or next(iter(lengths)) < 1:
            raise ValueError("Paired identity lengths differ")
        if set(entry["log_pred"]) != set(OPERATORS):
            raise ValueError("Paired run must contain all operators")
        if any(
            len(np.asarray(entry["log_pred"][operator])) != next(iter(lengths))
            for operator in OPERATORS
        ):
            raise ValueError("Paired prediction lengths differ")
        pair_runs = {}
        for pairing in PAIRINGS:
            pair_runs[pairing["name"]] = {
                bucket: _pair_stats(
                    entry["log_pred"][pairing["left"]],
                    entry["log_pred"][pairing["right"]],
                    identity,
                    bucket,
                )
                for bucket in ("all", "four_plus", "low4")
            }
        runs.append({
            "database": database,
            "run": run,
            "query_digest": digest([
                (str(query_hash), str(query_id))
                for query_hash, query_id in zip(
                    np.asarray(identity["query_hash"]).tolist(),
                    np.asarray(identity["query_id"]).tolist(),
                )
            ]),
            "pairs": pair_runs,
        })
    if len(runs) != len(DATABASES) * len(RUNS):
        raise ValueError("Paired screen requires 15 database/runs")
    if len({(item["database"], item["run"]) for item in runs}) != len(runs):
        raise ValueError("Duplicate paired database/run")
    macro = {}
    for pairing in PAIRINGS:
        macro[pairing["name"]] = {}
        for bucket in ("all", "four_plus", "low4"):
            values = [
                item["pairs"][pairing["name"]][bucket]
                for item in runs
            ]
            if any(item["query_count"] == 0 for item in values):
                raise ValueError(f"Empty paired bucket: {bucket}")
            fields = (
                "q50_ratio",
                "q95_ratio",
                "mean_delta_log_q",
                "mean_delta_q",
                "improved_fraction",
            )
            macro[pairing["name"]][bucket] = {
                f"arithmetic_mean_{field}": (
                    math.fsum(float(item[field]) for item in values) / len(values)
                    if values else None
                )
                for field in fields
            }
    return {
        "schema": 1,
        "experiment": EXPERIMENT,
        "contrasts": [dict(item) for item in CONTRASTS],
        "runs": runs,
        "macro": macro,
        "note": (
            "Q95 is not additively decomposable; per-run Q-ratios and paired "
            "delta-log values are reported without pooling queries across runs."
        ),
    }


def _source_binding(store):
    path = store.path
    return {
        "manifest_content_hash": store.manifest.get("content_hash"),
        "file_sha256": {name: file_hash(path / name) for name in CACHE_FILES},
    }


def validate_source_cache_path(cache_root, database):
    if database not in DATABASES:
        raise ValueError("Only source validation databases are permitted")
    root = checked_path(cache_root)
    pretrain = root / "pretrain"
    database_dir = pretrain / database
    paths = (pretrain, database_dir, *(database_dir / name for name in CACHE_FILES))
    for path in paths:
        if path.is_symlink():
            raise ValueError(f"Source cache symlink forbidden: {path}")
        if not path.exists():
            raise ValueError(f"Missing source cache input: {path}")
        if not path.resolve().is_relative_to(root):
            raise ValueError(f"Source cache path escapes its root: {path}")
    return database_dir


def validate_source_metadata(store, database):
    import numpy as np

    if (
        database not in DATABASES
        or store.path.name != database
        or store.path.parent.name != "pretrain"
    ):
        raise ValueError("Only source validation pretrain stores are permitted")
    offsets = store.offsets
    if (
        not len(store)
        or offsets.dtype.kind not in "iu"
        or (offsets < 0).any()
        or (offsets[0] != 0).any()
        or (offsets[1:] < offsets[:-1]).any()
    ):
        raise ValueError("Invalid source offsets")
    tables = np.diff(offsets[:, 2]).astype(np.int16)
    if (tables < 1).any():
        raise ValueError("Invalid source table counts")
    first = {}
    for index, row in enumerate(store.rows):
        if (
            not isinstance(row.get("query_hash"), str)
            or not row["query_hash"]
            or row.get("id") != f"{database}/pretrain/{index + 1}"
        ):
            raise ValueError("Invalid source identity")
        first.setdefault(row["query_hash"], index)
    unique = store.unique_indices()
    if len(unique) != store.manifest.get("unique_queries"):
        raise ValueError("Manifest unique-query count changed")
    if len(unique) != UNIQUE[database]:
        raise ValueError("Frozen three-database denominator changed")
    return unique, tables


def validate_source_labels(store, database):
    import numpy as np

    cards, offsets = store.cards, store.offsets
    if not np.isfinite(cards).all() or (cards < 0).any():
        raise ValueError("Invalid source cards")
    first = {}
    for index, row in enumerate(store.rows):
        query_hash = row["query_hash"]
        if query_hash in first:
            other = first[query_hash]
            if not np.array_equal(cards[index], cards[other]):
                raise ValueError("Conflicting duplicate source labels")
            for group, values in enumerate(store.groups):
                left = values[offsets[index, group]:offsets[index + 1, group]]
                right = values[offsets[other, group]:offsets[other + 1, group]]
                if not np.array_equal(left, right):
                    raise ValueError("Conflicting duplicate source features")
        else:
            first[query_hash] = index


def _environment(device):
    import numpy as np
    import torch

    if device not in ("cpu", "cuda", "cuda:0"):
        raise ValueError("Only cpu or one visible cuda:0 is permitted")
    if device != "cpu":
        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        if not visible or "," in visible or not (
            visible.isdigit() or visible.startswith("GPU-")
        ):
            raise ValueError("Explicit single CUDA_VISIBLE_DEVICES is required")
        if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
            raise ValueError("CUBLAS_WORKSPACE_CONFIG must be :4096:8")
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise ValueError("Explicit CUDA selection is unavailable")
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[name] = "1"
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    return {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": device,
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "threads": 1,
        "deterministic_algorithms": True,
        "tf32": False,
        "training": False,
    }


def _load_fixed_checkpoint(path, expected_sha256, device):
    from spatial_meta_ce.checkpoints import load_checkpoint

    path = checked_path(path)
    if file_hash(path) != expected_sha256:
        raise ValueError("Checkpoint SHA256 changed")
    model, state = load_checkpoint(path, device)
    if model.config != MODEL_CONFIG:
        raise ValueError("Checkpoint model configuration changed")
    if state.get("next_step") != 4000:
        raise ValueError("Gate 1 requires the frozen 4000-step checkpoint")
    metadata = state.get("metadata", {})
    if metadata.get("seed") != 42:
        raise ValueError("Checkpoint seed changed")
    if metadata.get("adapted_database"):
        raise ValueError("Adapted checkpoint cannot be a meta-initialization")
    for database in (*metadata.get("train_databases", ()),
                     *metadata.get("validation_databases", ())):
        if database in TARGETS:
            raise ValueError("Checkpoint exposes a target database")
    if metadata.get("validation_databases") != list(DATABASES):
        raise ValueError("Checkpoint source validation split changed")
    return model, state


def _cell_record(result, database, run, operator, plan):
    metrics = result["operators"][operator]["metrics"]
    support_audit = result["operators"][operator]["support_audit"]
    return {
        "operator": operator,
        "database": database,
        "run": run,
        "query_digest": plan["query_digest"],
        "support_digest": plan["support_digests"][str(K)],
        "query_count": metrics["all"]["count"],
        "metrics": metrics,
        "support_audit": support_audit,
        "state_before": result["state_before"],
        "state_after": result["state_after"],
    }


def _write_metrics_csv(path, cells):
    rows = []
    for cell in cells:
        metrics = cell["metrics"]
        audit = cell["support_audit"]
        rows.append({
            "operator": cell["operator"],
            "database": cell["database"],
            "run": cell["run"],
            "query_count": cell["query_count"],
            "all_q50": metrics["all"]["q50"],
            "all_q95": metrics["all"]["q95"],
            "four_plus_count": metrics["four_plus"]["count"],
            "four_plus_q50": metrics["four_plus"]["q50"],
            "four_plus_q95": metrics["four_plus"]["q95"],
            "low4_count": metrics["low4"]["count"],
            "low4_q50": metrics["low4"]["q50"],
            "low4_q95": metrics["low4"]["q95"],
            "support_mse_before": audit["support_mse_before"],
            "support_mse_after": audit["support_mse_after"],
            "support_smooth_l1_before": audit["support_smooth_l1_before"],
            "support_smooth_l1_after": audit["support_smooth_l1_after"],
            "update_norm": audit["update_norm"],
            "condition_number": audit["condition_number"],
            "solver_condition_number": audit.get("solver_condition_number"),
        })
    with Path(path).open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def execute(
    *,
    checkpoint,
    checkpoint_sha256,
    plan_path,
    plan_sha256,
    cache_root,
    lambda_registry_path,
    output,
    device="cpu",
    root=ROOT,
):
    """Run the read-only screen. This function never calls training code."""
    from spatial_meta_ce.data import FeatureStore

    output = safe_output(output, root=root)
    checkpoint = checked_path(checkpoint)
    plan_path = checked_path(plan_path)
    cache_root = checked_path(cache_root)
    lambda_registry_path = checked_path(lambda_registry_path)
    registry, registry_sha256 = load_lambda_registry(lambda_registry_path)
    ridge_lambda = registry["selected_lambda"]
    if file_hash(checkpoint) != checkpoint_sha256:
        raise ValueError("Checkpoint SHA256 mismatch before execution")
    if file_hash(plan_path) != plan_sha256:
        raise ValueError("Plan SHA256 mismatch before execution")
    plan = read_json(plan_path)
    if not isinstance(plan, dict) or set(plan) != set(DATABASES):
        raise ValueError("Plan must contain exactly the three source databases")
    for database in DATABASES:
        if not isinstance(plan[database], list) or len(plan[database]) != len(RUNS):
            raise ValueError("Plan must contain five runs per source database")
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    status = {
        "schema": 1,
        "experiment": EXPERIMENT,
        "status": "running",
        "k": K,
        "operators": list(OPERATORS),
        "lambda_registry_sha256": registry_sha256,
        "selected_lambda": ridge_lambda,
        "training": False,
        "target_read": False,
    }
    write_json(output / "status.json", status)
    try:
        write_json(output / "environment.json", _environment(device))
        inputs = {
            "schema": 1,
            "checkpoint": {
                "path": str(checkpoint),
                "sha256": checkpoint_sha256,
            },
            "plan": {
                "path": str(plan_path),
                "sha256": plan_sha256,
            },
            "lambda_registry": {
                "path": str(lambda_registry_path),
                "sha256": registry_sha256,
                "value": registry,
            },
            "cache_root": str(cache_root),
            "source_before": {},
        }
        source_before = {}
        cells = []
        pair_entries = []
        for database in DATABASES:
            store_path = validate_source_cache_path(cache_root, database)
            store = FeatureStore(store_path)
            unique_indices, table_counts = validate_source_metadata(store, database)
            source_before[database] = _source_binding(store)
            inputs["source_before"][database] = source_before[database]
            write_json(output / "inputs.json", inputs)
            for run in RUNS:
                certified = certify_plan(
                    plan[database][run],
                    store.rows,
                    database,
                    run,
                    expected_unique=len(unique_indices),
                )
                model, _ = _load_fixed_checkpoint(
                    checkpoint, checkpoint_sha256, device)
                result = evaluate_database_run(
                    model,
                    store,
                    database,
                    run,
                    support_indices=certified["support"],
                    query_indices=certified["query"],
                    table_counts=table_counts,
                    ridge_lambda=ridge_lambda,
                    device=device,
                )
                pair_path = output / f"pair-{database}-run{run}.npz"
                write_pair_archive(pair_path, result)
                pair_entries.append({
                    "database": database,
                    "run": run,
                    "identity": result["identity"],
                    "log_pred": {
                        operator: result["operators"][operator]["log_pred"]
                        for operator in OPERATORS
                    },
                })
                for operator in OPERATORS:
                    cell = _cell_record(
                        result, database, run, operator, plan[database][run])
                    cell["prediction_file"] = pair_path.name
                    cell["checkpoint_state_digest"] = result["state_before"]
                    cells.append(cell)
                write_json(
                    output / "cells.json",
                    {
                        "schema": 1,
                        "experiment": EXPERIMENT,
                        "cells": cells,
                    },
                )
                status.update({
                    "status": "running",
                    "active_database": database,
                    "active_run": run,
                    "completed_cells": len(cells),
                    "elapsed_seconds": time.monotonic() - started,
                })
                write_json(output / "status.json", status)
                del model, result
            validate_source_labels(store, database)
        cells_by_key = {
            (cell["operator"], cell["database"], cell["run"]): cell
            for cell in cells
        }
        ordered_cells = [
            cells_by_key[(operator, database, run)]
            for operator in OPERATORS
            for database in DATABASES
            for run in RUNS
        ]
        macro = summarize_cells(ordered_cells)
        paired = summarize_pairings(pair_entries)
        _write_metrics_csv(output / "metrics.csv", ordered_cells)
        write_json(output / "macro.json", macro)
        write_json(output / "paired_summary.json", paired)
        write_json(
            output / "screen_plan.json",
            {
                "schema": 1,
                "experiment": EXPERIMENT,
                "k": K,
                "support_max": SUPPORT_MAX,
                "runs": list(RUNS),
                "databases": list(DATABASES),
                "plan_sha256": plan_sha256,
                "runs_binding": {
                    database: [
                        {
                            "database": run_plan["database"],
                            "run": run_plan["run"],
                            "support": run_plan["supports"][str(K)],
                            "query_count": len(run_plan["query"]),
                            "query_digest": run_plan["query_digest"],
                            "support_digest": run_plan["support_digests"][str(K)],
                        }
                        for run_plan in plan[database]
                    ]
                    for database in DATABASES
                },
            },
        )
        source_after = {}
        for database in DATABASES:
            store_path = validate_source_cache_path(cache_root, database)
            store = FeatureStore(store_path)
            validate_source_metadata(store, database)
            validate_source_labels(store, database)
            source_after[database] = _source_binding(store)
        if source_before != source_after:
            raise ValueError("Source cache hashes changed during the screen")
        if file_hash(checkpoint) != checkpoint_sha256:
            raise ValueError("Checkpoint changed during the screen")
        if file_hash(plan_path) != plan_sha256:
            raise ValueError("Plan changed during the screen")
        if file_hash(lambda_registry_path) != registry_sha256:
            raise ValueError("Lambda registry changed during the screen")
        inputs["source_after"] = source_after
        write_json(output / "inputs.json", inputs)
        status.update({
            "status": "completed",
            "completed_cells": len(ordered_cells),
            "elapsed_seconds": time.monotonic() - started,
            "target_read": False,
            "training": False,
            "gate": "screen_completed_not_selected",
        })
        write_json(output / "status.json", status)
        return {
            "status": "completed",
            "cells": len(ordered_cells),
            "macro": macro,
            "paired": paired,
            "target_read": False,
            "training": False,
        }
    except BaseException as error:
        status.update({
            "status": "failed",
            "error_type": type(error).__name__,
            "error": str(error),
            "elapsed_seconds": time.monotonic() - started,
            "target_read": False,
            "training": False,
        })
        write_json(output / "status.json", status)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--checkpoint-sha256")
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--plan-sha256")
    parser.add_argument("--cache-root", type=Path)
    parser.add_argument("--lambda-registry", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    if not args.execute:
        print(json.dumps({
            "purpose": "dry-run-not-execution",
            "experiment": EXPERIMENT,
            "operators": list(OPERATORS),
            "k": K,
            "support_max": SUPPORT_MAX,
            "runs": list(RUNS),
            "databases": list(DATABASES),
            "lambda_registry_required": True,
            "target_read": False,
            "training": False,
            "output": str(args.out),
        }, indent=2))
        return
    required = {
        "--checkpoint": args.checkpoint,
        "--checkpoint-sha256": args.checkpoint_sha256,
        "--plan": args.plan,
        "--plan-sha256": args.plan_sha256,
        "--cache-root": args.cache_root,
        "--lambda-registry": args.lambda_registry,
    }
    missing = [name for name, value in required.items() if value is None]
    if missing:
        parser.error("execute requires: " + ", ".join(missing))
    result = execute(
        checkpoint=args.checkpoint,
        checkpoint_sha256=args.checkpoint_sha256,
        plan_path=args.plan,
        plan_sha256=args.plan_sha256,
        cache_root=args.cache_root,
        lambda_registry_path=args.lambda_registry,
        output=args.out,
        device=args.device,
    )
    print(json.dumps({
        "status": result["status"],
        "cells": result["cells"],
        "target_read": result["target_read"],
        "training": result["training"],
    }, indent=2))


if __name__ == "__main__":
    main()
