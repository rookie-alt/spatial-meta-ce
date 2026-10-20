"""Export per-query source-side predictions for the P0-3 audit (read-only).

Why this exists instead of calling ``evaluate()``: ``evaluate`` iterates
``data_config['splits']['test']`` and ``validate_splits`` forbids any overlap
between validation and test. The P0-3 audit needs K=0 and K>0 predictions for
the *source validation* databases, so pointing ``test`` at them is rejected by
design, and pointing it at the real targets would read held-out data.

This script therefore bypasses the split contract and iterates an explicitly
named, source-only database list. It refuses target databases, never trains,
and writes one CSV for the audit.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from spatial_meta_ce.checkpoints import load_checkpoint
from spatial_meta_ce.cli import read_json
from spatial_meta_ce.data import StorePool
from spatial_meta_ce.meta import adapt, forward
from spatial_meta_ce.train import seed_all

TARGETS = frozenset(("imdb", "stats", "genome", "ergastf1"))
SOURCE_VALIDATION = ("chembl", "legalacts", "telstra")


def assert_source(databases) -> None:
    for db in databases:
        if db in TARGETS:
            raise ValueError(f"Refusing held-out target database {db!r}")
        if not db or Path(db).name != db:
            raise ValueError(f"Invalid database name {db!r}")


def _predict(model, params, store, indices, device, batch_size):
    predictions = []
    for start in range(0, len(indices), batch_size):
        batch = store.batch(indices[start : start + batch_size], device)
        with torch.no_grad():
            chunk = model(batch) if params is None else forward(model, params, batch)
        predictions.extend(chunk.cpu().tolist())
    return predictions


def export(config, checkpoint, output, device, databases, batch_size):
    assert_source(databases)
    seed_all(config.get("support_seed", 2026))
    pool = StorePool(config["cache_root"])
    model, state = load_checkpoint(checkpoint, device)
    model.eval()
    metadata = state["metadata"]
    exposed = set(metadata["train_databases"]) | set(metadata["validation_databases"])
    if exposed & TARGETS:
        raise ValueError("Checkpoint was trained on a held-out target")
    if set(databases) - exposed:
        raise ValueError("Requested database was not a source database")
    if state["model_config"] != config["model"]:
        raise ValueError("Evaluation model config differs from checkpoint")

    output.parent.mkdir(parents=True, exist_ok=True)
    rows = 0
    with output.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow([
            "database", "run", "k", "query_id", "query_hash",
            "true_cardinality", "log_prediction",
        ])
        for db in databases:
            store = pool.get("pretrain", db)
            indices = store.unique_indices()
            pool_indices = store.unique_indices()
            for run in range(config["support_runs"]):
                # Source-side audit only: support and query come from the same
                # source database, so no target-leakage concern applies here
                # (unlike evaluate(), which must exclude the test queries).
                rng = np.random.default_rng(
                    np.random.SeedSequence([config["support_seed"], run])
                )
                support_indices = rng.choice(
                    pool_indices, size=max(config["shots"]), replace=False
                )
                for k in config["shots"]:
                    support = store.batch(support_indices[:k], device) if k else None
                    params = (
                        adapt(
                            model, support, config["inner_lr"], config["inner_steps"],
                            config["adapt_scope"], config["loss"], training=False,
                        )
                        if k
                        else None
                    )
                    for index, pred in zip(
                        indices, _predict(model, params, store, indices, device, batch_size)
                    ):
                        row = store.rows[int(index)]
                        writer.writerow([
                            db, run, k, row["id"], row["query_hash"],
                            float(store.cards[int(index), 0]), float(pred),
                        ])
                        rows += 1
    return {
        "output": str(output),
        "rows": rows,
        "databases": list(databases),
        "target_databases_read": [],
        "checkpoint": str(checkpoint),
        "model_config": state["model_config"],
        "shots": list(config["shots"]),
        "support_runs": config["support_runs"],
    }


def main(argv):
    parser = argparse.ArgumentParser(description="Export source-side predictions (read-only)")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--databases", nargs="+", default=list(SOURCE_VALIDATION))
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args(argv)
    torch.set_num_threads(1)
    config = read_json(args.config)
    # Read the cache root from the untouched data config. data_settings() runs
    # validate_splits(), which rejects any test/validation overlap, and this
    # export deliberately scopes itself to source databases instead.
    raw = read_json(config["data_config"])
    cache_root = Path(raw["cache_root"])
    if not cache_root.is_absolute():
        cache_root = Path(__file__).resolve().parents[1] / cache_root
    config = {**config, "cache_root": str(cache_root.resolve())}
    report = export(
        config, args.checkpoint, Path(args.output), args.device,
        args.databases, args.batch_size,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main(sys.argv[1:])
