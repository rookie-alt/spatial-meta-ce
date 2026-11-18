#!/usr/bin/env python3
"""Measure how source-side accuracy moves with the number of training steps.

Read-only with respect to training: this script never updates weights and never
opens a target database.  It evaluates one or more checkpoints from the *same*
training run on the source validation databases, using the sealed support
blocks and query population of ``spatial_meta_ce.plans.build_source_eval_plan``.

All checkpoints therefore see identical support queries and identical query
populations, so differences between rungs are paired and cannot come from
resampling.  Target databases are rejected before any cache is opened.
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from spatial_meta_ce.checkpoints import load_checkpoint
from spatial_meta_ce.cli import data_settings, read_json
from spatial_meta_ce.data import StorePool
from spatial_meta_ce.meta import adapt, forward
from spatial_meta_ce.metrics import q_values
from spatial_meta_ce.plans import (
    EVAL_DATABASES,
    TARGET_DATABASES,
    build_source_eval_plan,
    verify_source_eval_plan_against_stores,
)
from spatial_meta_ce.train import seed_all

FOUR_PLUS_MIN_TABLES = 4


def assert_source_databases(databases):
    for db in databases:
        if db in TARGET_DATABASES:
            raise ValueError(f"Refusing held-out target database {db!r}")
        if not db or Path(db).name != db:
            raise ValueError(f"Invalid database name {db!r}")


def table_counts(store, indices):
    offsets = np.asarray(store.offsets)
    indices = np.asarray(indices, dtype=np.int64)
    return offsets[indices + 1, 2] - offsets[indices, 2]


def summarize_population(predictions, truths, n_tables):
    """Unique-query Q50/Q95 plus the >=4-table bucket, pooled over queries."""
    q, _, _ = q_values(predictions, truths)
    n_tables = np.asarray(n_tables)
    if q.shape != n_tables.shape:
        raise ValueError("Table counts do not align with predictions")
    result = {
        "n": int(len(q)),
        "q50": float(np.percentile(q, 50)),
        "q95": float(np.percentile(q, 95)),
        "four_plus_n": int((n_tables >= FOUR_PLUS_MIN_TABLES).sum()),
        "four_plus_q95": None,
    }
    if result["four_plus_n"]:
        selected = q[n_tables >= FOUR_PLUS_MIN_TABLES]
        result["four_plus_q95"] = float(np.percentile(selected, 95))
    return result


def _predict(model, params, store, indices, device, batch_size):
    indices = np.asarray(indices, dtype=np.int64)
    chunks = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(indices), batch_size):
            batch = store.batch(indices[start : start + batch_size], device)
            chunk = forward(model, params, batch)
            values = chunk.detach().cpu().numpy().astype(np.float64).reshape(-1)
            if not np.isfinite(values).all():
                raise FloatingPointError("Non-finite source prediction")
            chunks.append(values)
    predictions = np.concatenate(chunks)
    if len(predictions) != len(indices):
        raise ValueError("Prediction length does not match the query population")
    return predictions


def evaluate_checkpoint(
    checkpoint,
    plan,
    stores,
    device,
    batch_size,
    ks,
    blocks,
    adaptation,
):
    adaptive = [k for k in ks if k > 0]
    if adaptive and adaptation["inner_steps"] == 0:
        # adapt(steps=0) returns the base parameters unchanged, so "adapted"
        # numbers would silently equal the zero-shot numbers.
        raise ValueError(
            "Few-shot rungs require inner_steps > 0; "
            f"{adaptive} would be identical to K=0"
        )
    model, state = load_checkpoint(checkpoint, device)
    metadata = state["metadata"]
    exposed = set(metadata["train_databases"]) | set(metadata["validation_databases"])
    if exposed & TARGET_DATABASES:
        raise ValueError("Checkpoint was trained or selected on a target database")
    if metadata.get("adapted_database"):
        raise ValueError("Adapted checkpoints are not independent initializations")
    if metadata.get("purpose") != "research":
        raise ValueError("Only research checkpoints may enter the learning curve")
    if list(metadata["validation_databases"]) != list(EVAL_DATABASES):
        raise ValueError("Checkpoint validation split differs from the eval plan")
    base = dict(model.named_parameters())
    rows = []
    for db in EVAL_DATABASES:
        store = stores[db]
        entry = plan["databases"][db]
        query = np.asarray(entry["query"], dtype=np.int64)
        truths = np.asarray(store.cards[query, 0], dtype=np.float64)
        tables = table_counts(store, query)
        for k in ks:
            if k < 0:
                raise ValueError("Support size must be non-negative")
            if k == 0:
                predictions = _predict(model, base, store, query, device, batch_size)
                rows.append(
                    {
                        "database": db,
                        "k": 0,
                        "run": 0,
                        **summarize_population(predictions, truths, tables),
                    }
                )
                continue
            for run, block in enumerate(entry["support_blocks"][:blocks], start=1):
                if k > len(block["support"]):
                    raise ValueError(
                        f"K={k} exceeds the sealed support block size "
                        f"{len(block['support'])}; the plan cannot supply it"
                    )
                support = np.asarray(block["support"][:k], dtype=np.int64)
                params = adapt(
                    model,
                    store.batch(support, device),
                    adaptation["inner_lr"],
                    adaptation["inner_steps"],
                    adaptation["adapt_scope"],
                    adaptation["loss"],
                    training=False,
                )
                predictions = _predict(model, params, store, query, device, batch_size)
                rows.append(
                    {
                        "database": db,
                        "k": k,
                        "run": run,
                        **summarize_population(predictions, truths, tables),
                    }
                )
    return rows, int(state["next_step"])


def macro(rows, key):
    values = [
        row[key]
        for row in rows
        if row["database"] in EVAL_DATABASES and row[key] is not None
    ]
    return float(np.mean(values)) if values else None


def geometric_macro_ratio(short_database, long_database, key):
    """P0's aggregation: per-database ratios, geometric mean over databases.

    P0 (doc 10 §7) froze this rule so that a large database cannot dominate the
    decision.  Averaging Q95 before taking the ratio would weight chembl's
    absolute scale far above legalacts and telstra.

    Databases whose bucket is undefined (empty >=4-table bucket) are skipped,
    matching ``arch_select.macro_four_plus``: an absent bucket is "not
    reported", not a zero.
    """
    ratios = []
    for db in EVAL_DATABASES:
        short = short_database.get(db, {}).get(key)
        long = long_database.get(db, {}).get(key)
        if short is None or long is None:
            continue
        if short <= 0:
            raise ValueError(f"Non-positive {key} for {db}: {short}")
        ratios.append(long / short)
    if not ratios:
        return None, {}
    return float(np.exp(np.mean(np.log(ratios)))), {
        db: ratio for db, ratio in zip(EVAL_DATABASES, ratios)
    }


def _bind_checkpoint_to_plan(state, plan):
    """Refuse to score a checkpoint that was trained on a different cache."""
    hashes = state["metadata"].get("manifest_hashes") or {}
    for db in EVAL_DATABASES:
        expected = plan["databases"][db]["cache_hash"]
        if hashes.get(db) != expected:
            raise ValueError(
                f"Checkpoint {db} cache hash {hashes.get(db)!r} does not match "
                f"the evaluation plan {expected!r}"
            )


def _assert_same_run(states):
    """All rungs must come from one training run, or the curve is meaningless."""
    reference = states[0][1]
    keys = ("model_config", "metadata")
    for label, state in states[1:]:
        for key in keys:
            if state[key] != reference[key]:
                raise ValueError(
                    f"Checkpoint at {label} has a different {key}; "
                    "all rungs must share one training run"
                )


def main(argv=None):
    parser = argparse.ArgumentParser(description="Source-side learning curve")
    parser.add_argument("--config", default="configs/data.json")
    parser.add_argument(
        "--checkpoints", nargs="+", required=True, help="step=path/to/step-N.pt"
    )
    parser.add_argument("--ks", nargs="+", type=int, default=[0])
    parser.add_argument("--blocks", type=int, default=1)
    parser.add_argument(
        "--inner-steps",
        type=int,
        default=5,
        help="few-shot adaptation steps (frozen evaluation protocol default: 5)",
    )
    parser.add_argument("--inner-lr", type=float, default=0.001)
    parser.add_argument("--adapt-scope", choices=["all", "head", "last"], default="all")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", required=True, help="output JSON path")
    args = parser.parse_args(argv)
    if args.blocks < 1 or args.blocks > 5:
        raise ValueError("blocks must be within 1..5")
    if any(k < 0 for k in args.ks):
        raise ValueError("Support sizes must be non-negative")
    if args.inner_steps < 0 or args.inner_lr <= 0:
        raise ValueError("Adaptation budget must be non-negative with positive lr")
    if len(set(args.ks)) != len(args.ks):
        raise ValueError("Duplicate support sizes in --ks")

    torch.set_num_threads(1)
    data_config = data_settings(read_json(args.config), args.config)
    assert_source_databases(data_config["splits"]["validation"])
    if data_config["splits"]["validation"] != list(EVAL_DATABASES):
        raise ValueError(f"Source validation must be exactly {EVAL_DATABASES}")
    seed_all(2026)
    plan = build_source_eval_plan(data_config)
    pool = StorePool(data_config["cache_root"], max_open=len(EVAL_DATABASES))
    stores = {db: pool.get("pretrain", db) for db in EVAL_DATABASES}
    verify_source_eval_plan_against_stores(plan, stores)

    results = []
    csv_rows = []
    loaded = []
    for spec in args.checkpoints:
        if "=" not in spec:
            raise ValueError(f"Checkpoint must be step=path, got {spec!r}")
        label, path = spec.split("=", 1)
        step = int(label)
        started = time.perf_counter()
        _, state = load_checkpoint(path, "cpu")
        _bind_checkpoint_to_plan(state, plan)
        loaded.append((step, state))
        adaptation = {
            "inner_lr": args.inner_lr,
            "inner_steps": args.inner_steps,
            "adapt_scope": args.adapt_scope,
            "loss": state["metadata"]["training_signature"]["loss"],
        }
        rows, checkpoint_step = evaluate_checkpoint(
            path,
            plan,
            stores,
            args.device,
            args.batch_size,
            args.ks,
            args.blocks,
            adaptation,
        )
        if checkpoint_step != step:
            raise ValueError(
                f"Checkpoint {path} advanced to step {checkpoint_step}, not {step}"
            )
        for row in rows:
            csv_rows.append({"step": step, **row})
        for k in args.ks:
            subset = [row for row in rows if row["k"] == k]
            results.append(
                {
                    "step": step,
                    "k": k,
                    "per_database": {
                        row["database"]: {
                            key: row[key]
                            for key in (
                                "n",
                                "q50",
                                "q95",
                                "four_plus_n",
                                "four_plus_q95",
                            )
                        }
                        for row in subset
                    },
                    "macro_q95": macro(subset, "q95"),
                    "macro_four_plus_q95": macro(subset, "four_plus_q95"),
                }
            )
        print(
            json.dumps(
                {
                    "step": step,
                    "seconds": round(time.perf_counter() - started, 1),
                    "macro_q95": results[-len(args.ks)]["macro_q95"],
                }
            ),
            flush=True,
        )

    if len({step for step, _ in loaded}) != len(loaded):
        raise ValueError("Duplicate step labels in --checkpoints")
    ordered = sorted(loaded, key=lambda item: item[0])
    _assert_same_run(ordered)
    steps = [step for step, _ in ordered]
    ratios = {}
    for key in ("q95", "four_plus_q95"):
        for k in args.ks:
            for previous, current in zip(steps, steps[1:]):
                short_rows = [
                    r for r in results if r["step"] == previous and r["k"] == k
                ]
                long_rows = [r for r in results if r["step"] == current and r["k"] == k]
                try:
                    ratio, per_database = geometric_macro_ratio(
                        short_rows[0]["per_database"] if short_rows else {},
                        long_rows[0]["per_database"] if long_rows else {},
                        key,
                    )
                except (ValueError, IndexError):
                    continue
                if ratio is None:
                    continue
                ratios[f"{key}_k{k}_{previous}_to_{current}"] = {
                    "ratio": ratio,
                    "per_database": per_database,
                }

    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    csv_path = output.parent / "learning_curve.csv"
    if csv_path.exists():
        raise FileExistsError(f"Refusing to overwrite {csv_path}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "step",
                "database",
                "k",
                "run",
                "n",
                "q50",
                "q95",
                "four_plus_n",
                "four_plus_q95",
            ],
        )
        writer.writeheader()
        writer.writerows(csv_rows)
    payload = {
        "schema": 1,
        "kind": "source_learning_curve",
        "config": args.config,
        "source_eval_plan_sha256": plan["plan_sha256"],
        "support_seed": plan["support_seed"],
        "blocks": args.blocks,
        "ks": args.ks,
        "adaptation": {
            "inner_steps": args.inner_steps,
            "inner_lr": args.inner_lr,
            "adapt_scope": args.adapt_scope,
            "source": "frozen evaluation protocol (configs/evaluation.json)",
        },
        "consecutive_ratios": ratios,
        "batch_size": args.batch_size,
        "device": args.device,
        "results": results,
        "target_databases_read": [],
    }
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )
    print(json.dumps({"written": str(output)}, ensure_ascii=False))
    return payload


if __name__ == "__main__":
    main()
