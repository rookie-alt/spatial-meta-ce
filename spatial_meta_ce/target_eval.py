"""D-stage zero-shot (K=0) evaluation on the four held-out test databases.

Only reachable through a dedicated, signed evaluation authorization: the
existing A-stage evaluator keeps refusing 122 candidates, and this module adds
its own guards instead of relaxing them. Results never feed back into any
training, selection or calibration step.
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
from .official_split import OFFICIAL_TRAIN_DATABASES, TEST_DATABASES
from .plans import _canonical_unique_indices
from .source_eval import PREDICTION_COLUMNS, _predict, _unit_diagnostics, _write_rows

CANDIDATE_MODEL_TYPES = (
    "grouped_set_residual",
    "sum_anchor_only_residual",
    "price_lite_residual",
    "price_lite_pg_residual",
    "price_tiny_residual",
)


def pg_log_predictions(cards, indices) -> np.ndarray:
    """The PG estimate already stored in each cache row, in log space."""
    estimates = np.asarray(cards[np.asarray(indices, dtype=np.int64), 1], dtype=np.float64)
    if (estimates < 0).any():
        raise ValueError("PG estimates must be non-negative")
    return np.log(np.maximum(estimates, 1.0))


def _price_maxima(data_config: dict, pool) -> tuple[int, ...]:
    """PRICE padding maxima over the test databases' available splits."""
    maxima = np.ones(4, dtype=np.int64)
    seen = 0
    for database in TEST_DATABASES:
        for split in ("test", "finetune"):
            manifest = Path(data_config["cache_root"]) / split / database / "manifest.json"
            if not manifest.is_file():
                continue
            maxima = np.maximum(
                maxima, pool.get(split, database).manifest["inspection"]["max_counts"]
            )
            seen += 1
    if not seen:
        raise ValueError("No test-database cache manifest was found for PRICE padding")
    return tuple(int(value) for value in maxima)


def evaluate_target_baseline(
    name: str,
    data_config: dict,
    output_dir,
    *,
    price_checkpoint=None,
    device: str = "cuda",
    batch_size: int = 128,
) -> dict:
    """Zero-shot PG or PRICE baseline on the same four test populations."""
    if name not in ("pg", "price"):
        raise ValueError("Unknown target baseline")
    if name == "price" and not price_checkpoint:
        raise ValueError("PRICE baseline requires its reference weights")

    out = Path(output_dir)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("Target baseline requires a new empty output directory")
    out.mkdir(parents=True, exist_ok=True)
    pool = StorePool(data_config["cache_root"], max_open=len(TEST_DATABASES))
    stores = {db: pool.get("test", db) for db in TEST_DATABASES}
    adapter = None
    if name == "price":
        from .baselines import PriceAdapter, price_log_predictions

        adapter = PriceAdapter(price_checkpoint, _price_maxima(data_config, pool), device)

    csv_path = out / "predictions.csv"
    units = []
    with csv_path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(PREDICTION_COLUMNS)
        for database in TEST_DATABASES:
            store = stores[database]
            query = np.asarray(_canonical_unique_indices(store), dtype=np.int64)
            started = time.perf_counter()
            if name == "pg":
                predictions = pg_log_predictions(store.cards, query)
            else:
                chunks = []
                for offset in range(0, len(query), batch_size):
                    batch = store.batch(query[offset : offset + batch_size], device)
                    with torch.no_grad():
                        chunks.append(adapter(batch).detach().cpu().numpy())
                predictions = price_log_predictions(np.concatenate(chunks))
            predict_seconds = time.perf_counter() - started
            _write_rows(writer, name.upper(), 0, database, store, query, 0, 0, predictions)
            units.append(
                {
                    "database": database,
                    "k": 0,
                    "run": 0,
                    "queries": int(len(query)),
                    "predict_seconds": predict_seconds,
                    **_unit_diagnostics(store, query, predictions, predictions),
                }
            )
    protocol = {
        "schema": 1,
        "kind": "target_zero_shot_baseline",
        "baseline": name,
        "databases": list(TEST_DATABASES),
        "units": units,
        "predictions_sha256": file_hash(csv_path),
        "price_checkpoint_sha256": file_hash(price_checkpoint) if price_checkpoint else None,
        "target_databases_read": list(TEST_DATABASES),
        "selection_use": "none",
    }
    protocol["macro_log_q95"] = float(
        np.mean([unit["metrics"]["log_q95"] for unit in units])
    )
    (out / "eval_protocol.json").write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )
    return protocol


def evaluate_target_zero_shot(
    checkpoint,
    arm: str,
    seed: int,
    data_config: dict,
    output_dir,
    *,
    device: str = "cuda",
    batch_size: int = 128,
) -> dict:
    model, state = load_checkpoint(checkpoint, device)
    metadata = state["metadata"]
    if state.get("model_type") not in CANDIDATE_MODEL_TYPES:
        raise ValueError("D-stage evaluation requires a 122 candidate checkpoint")
    if metadata.get("adapted_database"):
        raise ValueError("An adapted checkpoint is not an independent zero-shot model")
    exposed = set(metadata.get("train_databases") or []) | set(
        metadata.get("validation_databases") or []
    )
    if exposed & set(TEST_DATABASES):
        raise ValueError("Checkpoint was trained on a test database")
    if tuple(metadata.get("train_databases") or []) != OFFICIAL_TRAIN_DATABASES:
        raise ValueError("D-stage evaluation requires the official 26-database training set")

    out = Path(output_dir)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("Target evaluation requires a new empty output directory")
    out.mkdir(parents=True, exist_ok=True)
    pool = StorePool(data_config["cache_root"], max_open=len(TEST_DATABASES))
    stores = {db: pool.get("test", db) for db in TEST_DATABASES}

    csv_path = out / "predictions.csv"
    units = []
    with csv_path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(PREDICTION_COLUMNS)
        for database in TEST_DATABASES:
            store = stores[database]
            query = np.asarray(_canonical_unique_indices(store), dtype=np.int64)
            base = OrderedDict(model.named_parameters())
            started = time.perf_counter()
            predictions = _predict(model, base, store, query, device, batch_size)
            predict_seconds = time.perf_counter() - started
            _write_rows(writer, arm, seed, database, store, query, 0, 0, predictions)
            units.append(
                {
                    "database": database,
                    "k": 0,
                    "run": 0,
                    "queries": int(len(query)),
                    "predict_seconds": predict_seconds,
                    **_unit_diagnostics(store, query, predictions, predictions),
                }
            )
    protocol = {
        "schema": 1,
        "kind": "target_zero_shot_d_stage",
        "arm": arm,
        "seed": seed,
        "databases": list(TEST_DATABASES),
        "units": units,
        "predictions_sha256": file_hash(csv_path),
        "checkpoint_sha256": file_hash(checkpoint),
        "training_plan_sha256": metadata.get("training_plan_sha256"),
        "training_databases": list(metadata["train_databases"]),
        "target_databases_read": list(TEST_DATABASES),
        "selection_use": "none",
    }
    protocol["macro_log_q95"] = float(
        np.mean([unit["metrics"]["log_q95"] for unit in units])
    )
    (out / "eval_protocol.json").write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )
    return protocol