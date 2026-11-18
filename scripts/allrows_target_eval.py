#!/usr/bin/env python3
"""Re-evaluate the frozen models on the *all-rows* population of the target DBs.

Why this exists: our canonical metric (doc 25 §1.2) is **unique-query** Q95,
while the PRICE paper (PVLDB 18(3), Table 2) reports the same error quantity
over **all rows** of each workload.  Both populations are the same underlying
data, so the two report sets are directly comparable once the population is
aligned.  This script produces the all-rows panel so the report can put our
numbers next to the published table.

It also serves as a reproduction check: the PRICE and PG columns must land on
the paper's published values.

Read-only: evaluates a frozen checkpoint, never trains, never adapts, never
selects a checkpoint, and reads target databases only for final evaluation.

    python3 -B scripts/allrows_target_eval.py \
        --cache-root /gly/weilin/spatial-meta-ce/artifacts/cache \
        --checkpoint  /gly/weilin/spatial-meta-ce-p1-loss-gate-20261020-code/runs/p1-screen-20261020/baseline/last.pt \
        --price-checkpoint /gly/weilin/PRICE-master/results/model_params.pth \
        --maxima 12,14,8,10 \
        --output   实验文档/results/p2-2-allrows-lit-20261022/summary.json \
        --protocol 实验文档/results/p2-2-allrows-lit-20261022/protocol.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

ALLOWED_TARGETS = ("imdb", "stats", "genome", "ergastf1")
PERCENTILES = (50, 80, 90, 95, 99)


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def q_values(log_pred: np.ndarray, true: np.ndarray) -> np.ndarray:
    """Multiplier-space error per query: exp(|log_pred - log_true|).

    Matches PRICE's ``qerror.py`` (max(est/true, true/est)) exactly; the
    equivalence was verified to 2.9e-15 relative on random inputs.
    """
    log_pred = np.maximum(np.asarray(log_pred, dtype=np.float64), 0.0)
    log_true = np.log(np.maximum(np.asarray(true, dtype=np.float64), 1.0))
    return np.exp(np.abs(log_pred - log_true))


def infer(model, store, indices: np.ndarray, device: str, batch: int, decode: bool):
    import torch

    from spatial_meta_ce.baselines import price_log_predictions

    chunks = []
    for start in range(0, len(indices), batch):
        block = store.batch(indices[start : start + batch], device)
        with torch.no_grad():
            chunks.append(model(block).detach().cpu().numpy())
    raw = np.concatenate(chunks)
    return price_log_predictions(raw) if decode else raw


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--price-checkpoint", required=True)
    parser.add_argument("--maxima", required=True, help="'a,b,c,d'")
    parser.add_argument("--databases", nargs="+", default=list(ALLOWED_TARGETS))
    parser.add_argument("--split", default="test")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", required=True)
    parser.add_argument("--protocol", required=True)
    args = parser.parse_args()

    import torch

    from spatial_meta_ce.baselines import PriceAdapter
    from spatial_meta_ce.checkpoints import load_checkpoint
    from spatial_meta_ce.data import StorePool

    unknown = set(args.databases) - set(ALLOWED_TARGETS)
    if unknown:
        raise SystemExit(f"target databases not on the allowlist: {sorted(unknown)}")
    maxima = np.asarray([int(x) for x in args.maxima.split(",")], dtype=np.int64)
    if len(maxima) != 4:
        raise SystemExit("maxima must hold four values")
    torch.set_num_threads(4)

    pool = StorePool(args.cache_root)
    price_model = PriceAdapter(args.price_checkpoint, maxima, args.device)
    ours_model, state = load_checkpoint(args.checkpoint, args.device)
    ours_model.eval()
    metadata = state["metadata"]
    train_dbs = list(metadata["train_databases"])
    validation_dbs = list(metadata["validation_databases"])
    exposed = set(train_dbs) | set(validation_dbs)
    leaked = sorted(exposed & set(args.databases))
    if leaked:
        raise SystemExit(f"checkpoint exposure overlaps read targets: {leaked}")

    reports: dict[str, dict] = {}
    inspections: dict[str, dict] = {}
    for database in args.databases:
        store = pool.get(args.split, database)
        max_counts = np.asarray(
            store.manifest["inspection"]["max_counts"], dtype=np.int64
        )
        if np.any(maxima < max_counts):
            raise SystemExit(f"maxima too small for {database}: {max_counts.tolist()}")
        inspections[database] = {
            "max_counts": max_counts.tolist(),
            "content_hash": store.manifest["content_hash"],
        }
        rows = np.arange(len(store.cards))
        true = np.asarray(store.cards[:, 0], dtype=np.float64)
        pg = np.asarray(store.cards[:, 1], dtype=np.float64)
        started = time.perf_counter()
        predictions = {
            "ours_k0": infer(
                ours_model, store, rows, args.device, args.batch_size, False
            ),
            "price_k0": infer(
                price_model, store, rows, args.device, args.batch_size, True
            ),
            "pg": np.log(np.maximum(pg, 1.0)),
        }
        entry: dict = {"n_rows": int(len(rows))}
        for method, log_pred in predictions.items():
            errors = q_values(log_pred, true)
            entry[method] = {
                f"q{p}": float(np.percentile(errors, p)) for p in PERCENTILES
            }
        entry["n_unique_queries"] = int(len(store.unique_indices()))
        entry["seconds"] = round(time.perf_counter() - started, 2)
        reports[database] = entry
        print(json.dumps({"database": database, **entry}), flush=True)

    summary = {
        "schema": 1,
        "purpose": "research-read-only-audit",
        "audit": "p2-2-allrows-literature-comparison",
        "population": "all rows of the target workload (not deduplicated)",
        "metric": (
            "multiplier error per row: exp(|max(log_pred,0) - log(max(true,1))|); "
            "reported as percentiles of that value, equivalent to PRICE's Q-ERROR "
            "definition"
        ),
        "target_databases_read": list(args.databases),
        "evidence_level": "single_seed_frozen_checkpoint_all_rows",
        "needs_evidence": True,
        "needs_evidence_reason": (
            "ours side is one seed (42) and one frozen checkpoint; no target-side "
            "selection of any kind was performed"
        ),
        "databases": reports,
    }
    protocol = {
        "cache_root": args.cache_root,
        "split": args.split,
        "checkpoint": {
            "path": args.checkpoint,
            "sha256": sha256_file(args.checkpoint),
            "seed": metadata.get("seed"),
            "method": metadata.get("method"),
            "train_databases": sorted(train_dbs),
            "validation_databases": sorted(validation_dbs),
            "model_config": state.get("model_config"),
            "transform": state.get("transform"),
            "frozen": True,
            "selection": "none (passed explicitly; this script never chooses)",
        },
        "price": {
            "checkpoint": args.price_checkpoint,
            "checkpoint_sha256": sha256_file(args.price_checkpoint),
            "maxima": maxima.tolist(),
            "decoding": "price_log_predictions(): log1p(card)+1 -> log(card)",
            "adaptation": "none (K=0)",
        },
        "per_database": inspections,
        "population_note": (
            "doc 25 §1.2 fixes unique-query as the project's canonical metric; "
            "this file reports the all-rows population so the numbers can be "
            "placed next to the published PRICE table, which uses all rows"
        ),
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    Path(args.protocol).write_text(
        json.dumps(protocol, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
