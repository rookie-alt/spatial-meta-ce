"""Source-only full-population evaluation for doc 112.

The evaluator deliberately does not reuse ``evaluate.py``: that path samples
support from ``finetune`` stores.  Here support comes only from the sealed
source plan and Q_d is the plan's complete remaining query population.
"""

from __future__ import annotations

import csv
import json
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch

from .checkpoints import load_checkpoint
from .data import StorePool
from .manifests import file_hash
from .meta import adapt, forward
from .plans import (
    EVAL_DATABASES,
    TARGET_DATABASES,
    load_source_eval_plan,
    verify_source_eval_plan_against_stores,
)
from .price_protocol import A_ARMS, validate_a_checkpoint, validate_a_plan
from .metrics import q_values, summarize
from .train import FIXED_MODEL, FIXED_SELECTION, FIXED_STEPS

ARMS = {
    "A_erm": ("erm", "post"),
    "B_fomaml": ("fomaml", "post"),
    "C_joint": ("fomaml", "joint"),
}
KS = (0, 5, 20)
PREDICTION_COLUMNS = (
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


def _is_sha256(value):
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def validate_eval_checkpoint(state, arm, seed):
    if arm in A_ARMS:
        return validate_a_checkpoint(state, arm, seed)
    if arm not in ARMS:
        raise ValueError(f"Unknown arm {arm!r}")
    method, objective = ARMS[arm]
    metadata = state.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("Checkpoint has no metadata")
    if state.get("model_config") != FIXED_MODEL:
        raise ValueError("Checkpoint model configuration is not the fixed 112 model")
    if state.get("next_step") != FIXED_STEPS:
        raise ValueError("Checkpoint is not the 4000-update last checkpoint")
    if metadata.get("fixed_budget") is not True:
        raise ValueError("Checkpoint was not trained in fixed-budget mode")
    if metadata.get("selection") != FIXED_SELECTION:
        raise ValueError("Checkpoint selection is not last.pt-only")
    if metadata.get("method") != method:
        raise ValueError(f"Checkpoint method does not match arm {arm}")
    if metadata.get("meta_objective") != objective:
        raise ValueError(f"Checkpoint objective does not match arm {arm}")
    if metadata.get("seed") != seed:
        raise ValueError("Checkpoint seed does not match the requested seed")
    if metadata.get("purpose") != "research":
        raise ValueError("Checkpoint purpose is not research")
    if metadata.get("fixed_steps") != FIXED_STEPS:
        raise ValueError("Checkpoint fixed_steps does not match the 4000-update budget")
    if objective == "joint" and metadata.get("joint_pre_weight") != 0.5:
        raise ValueError("Joint checkpoint does not record the 0.5 pre weight")
    for field in (
        "validation_databases",
        "selection_databases",
        "validation_monitor_reads",
    ):
        if metadata.get(field) != []:
            raise ValueError(f"Checkpoint {field} must be empty")
    if not _is_sha256(metadata.get("training_plan_sha256")):
        raise ValueError("Checkpoint has no sealed training-plan hash")
    train_databases = metadata.get("train_databases")
    if not isinstance(train_databases, list) or not train_databases:
        raise ValueError("Checkpoint has no train database list")
    if set(train_databases) & TARGET_DATABASES:
        raise ValueError("Checkpoint training used a target database")
    if set(train_databases) & set(EVAL_DATABASES):
        raise ValueError("Checkpoint training used a source evaluation database")
    return metadata


def _predict(model, params, store, indices, device, batch_size):
    indices = np.asarray(indices, dtype=np.int64)
    if not len(indices):
        raise ValueError("Cannot predict an empty query population")
    model.eval()
    chunks = []
    with torch.no_grad():
        for start in range(0, len(indices), batch_size):
            chunk = indices[start : start + batch_size]
            prediction = forward(model, params, store.batch(chunk, device))
            values = prediction.detach().cpu().numpy().astype(np.float64).reshape(-1)
            if not np.isfinite(values).all():
                raise FloatingPointError("Non-finite source evaluation prediction")
            chunks.append(values)
    result = np.concatenate(chunks)
    if len(result) != len(indices):
        raise ValueError("Prediction length does not match the query population")
    return result


def _write_rows(writer, arm, seed, database, store, indices, k, run, predictions):
    for index, prediction in zip(indices, predictions):
        row = store.rows[int(index)]
        writer.writerow(
            [
                arm,
                seed,
                database,
                k,
                run,
                row["id"],
                row["query_hash"],
                format(float(store.cards[int(index), 0]), ".17g"),
                format(float(prediction), ".17g"),
            ]
        )


def _unit_diagnostics(store, indices, prediction, k0_prediction):
    truth = np.asarray(store.cards[np.asarray(indices), 0], dtype=np.float64)
    metrics = summarize(prediction, truth)
    log_error = q_values(prediction, truth)[2]
    base_error = q_values(k0_prediction, truth)[2]
    counts = np.diff(store.offsets, axis=0)[np.asarray(indices), 2]
    four_plus = counts >= 4
    return {
        "metrics": metrics,
        "four_plus": summarize(prediction[four_plus], truth[four_plus]) if four_plus.any() else None,
        "worsened_fraction_vs_same_model_k0": float(np.mean(log_error > base_error)),
        "negative_transfer_delta_log_q95": metrics["log_q95"] - summarize(k0_prediction, truth)["log_q95"],
    }


def evaluate_source(
    checkpoint,
    arm,
    seed,
    plan_path,
    data_config,
    output_dir,
    device="cpu",
    batch_size=128,
):
    if arm not in (*ARMS, *A_ARMS):
        raise ValueError(f"Unknown arm {arm!r}")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    model, state = load_checkpoint(checkpoint, device)
    metadata = validate_eval_checkpoint(state, arm, seed)
    plan = load_source_eval_plan(plan_path, data_config)
    if arm in A_ARMS:
        validate_a_plan(plan, data_config)
        from .price_capacity import build_capacity, scan_source_capacity, UnsupportedCapacity

        capacity = build_capacity(data_config)
        if hasattr(model, "capacity") and model.capacity != capacity:
            raise ValueError("Source evaluation capacity/cache binding changed")
        coverage = scan_source_capacity(data_config, capacity)
        if coverage["unsupported"]:
            raise UnsupportedCapacity(coverage["unsupported"])
    if tuple(plan["databases"]) != EVAL_DATABASES:
        raise ValueError(f"Source plan must cover exactly {EVAL_DATABASES}")
    pool = StorePool(data_config["cache_root"], max_open=len(EVAL_DATABASES))
    stores = {db: pool.get("pretrain", db) for db in EVAL_DATABASES}
    verify_source_eval_plan_against_stores(plan, stores)
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError("Source evaluation requires a new empty output directory")
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "predictions.csv"
    units = []
    with csv_path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(PREDICTION_COLUMNS)
        for db in EVAL_DATABASES:
            store = stores[db]
            entry = plan["databases"][db]
            query = np.asarray(entry["query"], dtype=np.int64)
            base = OrderedDict(model.named_parameters())
            started = time.perf_counter()
            predictions = _predict(model, base, store, query, device, batch_size)
            predict_seconds = time.perf_counter() - started
            k0_predictions = predictions.copy()
            _write_rows(writer, arm, seed, db, store, query, 0, 0, predictions)
            units.append(
                {
                    "database": db,
                    "k": 0,
                    "run": 0,
                    "queries": len(query),
                    "adapt_seconds": 0.0,
                    "predict_seconds": predict_seconds,
                    "unit_wall_seconds": time.perf_counter() - started,
                    **({"arm": arm, "seed": seed, "source_eval_plan_sha256": plan["plan_sha256"],
                        **_unit_diagnostics(store, query, predictions, k0_predictions)} if arm in A_ARMS else {}),
                }
            )
            for k in (5, 20):
                for run, block in enumerate(entry["support_blocks"], start=1):
                    started = time.perf_counter()
                    support = np.asarray(block["support"][:k], dtype=np.int64)
                    adapt_started = time.perf_counter()
                    params = adapt(
                        model,
                        store.batch(support, device),
                        metadata["training_signature"]["inner_lr"],
                        metadata["training_signature"]["inner_steps"],
                        metadata["training_signature"]["adapt_scope"],
                        metadata["training_signature"]["loss"],
                        training=False,
                    )
                    adapt_seconds = time.perf_counter() - adapt_started
                    predict_started = time.perf_counter()
                    predictions = _predict(
                        model, params, store, query, device, batch_size
                    )
                    predict_seconds = time.perf_counter() - predict_started
                    _write_rows(
                        writer, arm, seed, db, store, query, k, run, predictions
                    )
                    units.append(
                        {
                            "database": db,
                            "k": k,
                            "run": run,
                            "queries": len(query),
                            "adapt_seconds": adapt_seconds,
                            "predict_seconds": predict_seconds,
                            "unit_wall_seconds": time.perf_counter() - started,
                            **({
                                "arm": arm, "seed": seed, "source_eval_plan_sha256": plan["plan_sha256"],
                                "support_ids": block["support_ids"][:k],
                                "support_hashes": block["support_hashes"][:k],
                                **_unit_diagnostics(store, query, predictions, k0_predictions),
                            } if arm in A_ARMS else {}),
                        }
                    )
    protocol = {
        "schema": 1,
        "kind": "price_source_eval_122_a" if arm in A_ARMS else "fomaml_joint_source_eval_protocol",
        "arm": arm,
        "seed": seed,
        "checkpoint": str(Path(checkpoint).resolve()),
        "checkpoint_sha256": file_hash(checkpoint),
        "source_eval_plan": str(Path(plan_path).resolve()),
        "source_eval_plan_sha256": plan["plan_sha256"],
        "source_eval_plan_file_sha256": file_hash(plan_path),
        "batch_size": batch_size,
        "device": device,
        "units": units,
        "predictions_sha256": file_hash(csv_path),
        "target_databases_read": [],
        "completed": True,
    }
    (output_dir / "eval_protocol.json").write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )
    return protocol
