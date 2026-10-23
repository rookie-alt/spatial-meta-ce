#!/usr/bin/env python3
"""Probe the frozen representation for source-database identity.

Why this exists: doc 103 showed our Q95 gap to PRICE is a *ranking-consistency*
gap that tracks training exposure exactly; doc 96 showed the direction signal is
learnable within a database but does not transfer across databases.  Both point
at the same hypothesis: the representation encodes "which database am I" as a
shortcut.  This script measures that directly.

Three statistics, each answering a different question:

``k_way``
    Train a linear classifier to name the source database from the pooled
    feature vector that enters the output MLP (``_head_input``)
    and test on held-out *queries of the same databases* (5-fold).  High accuracy
    (vs ``1/k`` chance) means database identity is linearly readable — the
    shortcut exists.

``negative_control``
    Within one database, split its queries in half at random and try to predict
    which half a row came from.  There is no real signal, so this must land near
    0.5.  Materially above 0.5 would mean the features leak row order or another
    artefact, and would invalidate the ``k_way`` number.

``lodo_centroid_ratio``
    Leave one database out entirely: how far is its centroid from the nearest
    *trained* database centroid, in units of the typical within-database spread?
    A ratio near 1 means the unseen database sits inside the training cloud
    (no database-specific island); a large ratio means each database occupies
    its own region, which is what a shortcut representation looks like.

Read-only: loads a frozen checkpoint, never trains, never adapts, never selects
a checkpoint, and never reads a target database.

    python3 -B scripts/db_identity_probe.py \
        --cache-root /gly/weilin/spatial-meta-ce/artifacts/cache \
        --checkpoint /gly/weilin/.../runs/p1-screen-20261020/baseline/last.pt \
        --databases accidents airline baseball ... \
        --max-per-db 2000 \
        --output 实验文档/results/p2-2-db-identity-probe-20261023/summary.json \
        --protocol 实验文档/results/p2-2-db-identity-probe-20261023/protocol.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

# never a legal probe input: the four target databases are evaluation-only
FORBIDDEN = ("imdb", "stats", "genome", "ergastf1")


def _pooled_features(model, batch):
    """Reproduce ``model.forward``'s pooled input vector, without the output head.

    This rebuilds exactly the tensor that ``model._head_input`` produces (the
    same pooling, counts, pg and cartesian terms), so the probe works on both
    current and older checkpoints.
    """
    import torch

    masks = batch["masks"]
    tokens = [
        encoder(x.masked_fill(~mask.unsqueeze(-1), 0))
        for encoder, x, mask in zip(model.encoders, batch["groups"], masks)
    ]
    if model.arch == "sum_map":
        tokens = model._map_tokens(tokens, masks, batch["mapping"])
    elif model.arch in ("attn", "attn_pair"):
        tokens = model._mix(tokens, masks)
    pooled = []
    for t, mask in zip(tokens, masks):
        valid = mask.unsqueeze(-1)
        total = (t * valid).sum(1)
        mean = total / valid.sum(1).clamp_min(1)
        largest = t.masked_fill(~valid, -torch.inf).amax(1)
        largest = torch.where(
            mask.any(1, keepdim=True), largest, torch.zeros_like(largest)
        )
        pooled.extend(
            [mean, largest] if model.arch == "set" else [mean, largest, total]
        )
    pg = batch["pg_log"] if model.use_pg else torch.zeros_like(batch["pg_log"])
    log_cartesian = (batch["groups"][2][:, :, 0] * masks[2]).sum(1)
    return torch.cat(
        [
            *pooled,
            torch.log1p(batch["counts"]),
            pg[:, None],
            log_cartesian[:, None],
        ],
        dim=1,
    )


def load_features(model, store, indices, device, batch):
    """Return (n, d) pooled features for the given row indices."""
    import torch

    chunks = []
    for start in range(0, len(indices), batch):
        block = store.batch(indices[start : start + batch], device)
        with torch.no_grad():
            # _pooled_features mirrors _head_input, the pooling path present on
            # every model version.  Do not use head_features: it appends a
            # Linear+ReLU, which would put the probe on a different layer than
            # the checkpoints that predate it.
            chunks.append(_pooled_features(model, block).detach().cpu().numpy())
    return np.concatenate(chunks, axis=0)


def standardise(train_x: np.ndarray, test_x: np.ndarray):
    """Fit the scaler on train only."""
    mean = train_x.mean(axis=0, keepdims=True)
    scale = train_x.std(axis=0, keepdims=True)
    scale[scale < 1e-8] = 1.0
    return (train_x - mean) / scale, (test_x - mean) / scale


def logistic_fit(x, y, n_classes, epochs=400, lr=0.5, l2=1e-3):
    """Multinomial logistic regression by full-batch gradient descent (numpy).

    Numpy only: the project venv guarantees torch/numpy/sqlglot, not sklearn.
    """
    n, d = x.shape
    xb = np.concatenate([x, np.ones((n, 1))], axis=1)
    w = np.zeros((d + 1, n_classes))
    onehot = np.zeros((n, n_classes))
    onehot[np.arange(n), y] = 1.0
    for _ in range(epochs):
        z = xb @ w
        z -= z.max(axis=1, keepdims=True)
        p = np.exp(z)
        p /= p.sum(axis=1, keepdims=True)
        grad = xb.T @ (p - onehot) / n
        grad[:-1] += l2 * w[:-1]
        w -= lr * grad
    return w


def logistic_predict(x, w):
    xb = np.concatenate([x, np.ones((x.shape[0], 1))], axis=1)
    return np.argmax(xb @ w, axis=1)


def k_way(features, labels, n_classes, folds, rng):
    """5-fold accuracy of naming the database from its features."""
    order = rng.permutation(len(labels))
    chunks = np.array_split(order, folds)
    correct = total = 0
    for i, test_idx in enumerate(chunks):
        train_idx = np.concatenate([c for j, c in enumerate(chunks) if j != i])
        xtr, xte = standardise(features[train_idx], features[test_idx])
        w = logistic_fit(xtr, labels[train_idx], n_classes)
        correct += int((logistic_predict(xte, w) == labels[test_idx]).sum())
        total += len(test_idx)
    return correct / total


def negative_control(features, labels, n_classes, folds, rng):
    """Permutation control: shuffle the database labels, rerun the same probe.

    With the label-to-row link destroyed, accuracy must fall to ~1/k.  If it
    stays high, the k-way result reflects something other than database
    identity (leaked row order, a preprocessing artefact, or over-fitting of
    the probe); if it drops, the k-way number is trustworthy.
    """
    shuffled = labels.copy()
    rng.shuffle(shuffled)
    return k_way(features, shuffled, n_classes, folds, rng)


def lodo_centroid_ratio(features, labels):
    """Distance of each held-out centroid to the nearest trained one, / spread."""
    classes = np.unique(labels)
    global_scale = float(np.median(features.std(axis=0)))
    rows = []
    for held in classes:
        test = labels == held
        train = ~test
        xtr, xte = standardise(features[train], features[test])
        train_centroids = np.stack(
            [xtr[labels[train] == c].mean(axis=0) for c in classes if c != held]
        )
        held_centroid = xte.mean(axis=0)
        dist = np.linalg.norm(train_centroids - held_centroid, axis=1)
        # typical within-database spread in the same standardised units
        spread = float(
            np.median(
                [
                    np.linalg.norm(
                        xtr[labels[train] == c] - xtr[labels[train] == c].mean(0),
                        axis=1,
                    ).mean()
                    for c in classes
                    if c != held
                ]
            )
        )
        rows.append(
            {
                "held_out_label": int(held),
                "nearest_trained_centroid_distance": round(float(dist.min()), 4),
                "mean_within_database_spread": round(spread, 4),
                "ratio": round(float(dist.min()) / spread, 3) if spread > 0 else None,
            }
        )
    return rows, round(global_scale, 4)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--databases", nargs="+", required=True)
    parser.add_argument("--split", default="pretrain")
    parser.add_argument("--max-per-db", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", required=True)
    parser.add_argument("--protocol", required=True)
    args = parser.parse_args()

    import torch

    from spatial_meta_ce.checkpoints import load_checkpoint
    from spatial_meta_ce.data import StorePool

    leaked = sorted(set(args.databases) & set(FORBIDDEN))
    if leaked:
        raise SystemExit(f"refusing to probe target databases: {leaked}")
    if args.split != "pretrain":
        raise SystemExit("the probe only reads the pretrain split")
    if len(args.databases) < 3:
        raise SystemExit("need at least three databases for a meaningful probe")

    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    pool = StorePool(args.cache_root)
    model, state = load_checkpoint(args.checkpoint, args.device)
    model.eval()
    metadata = state["metadata"]
    exposed = set(metadata["train_databases"]) | set(metadata["validation_databases"])
    outside = sorted(set(args.databases) - exposed)
    if outside:
        raise SystemExit(f"not an exposed source database: {outside}")

    started = time.perf_counter()
    features, labels = [], []
    for label, database in enumerate(sorted(args.databases)):
        store = pool.get(args.split, database)
        rows = np.asarray(store.unique_indices(), dtype=np.int64)
        if len(rows) > args.max_per_db:
            rows = rng.choice(rows, size=args.max_per_db, replace=False)
        x = load_features(model, store, rows, args.device, args.batch_size)
        features.append(x)
        labels.append(np.full(len(x), label, dtype=np.int64))
        print(json.dumps({"database": database, "rows": int(len(x))}), flush=True)

    x_all = np.asarray(np.concatenate(features, axis=0), dtype=np.float64)
    y_all = np.concatenate(labels, axis=0)
    n_classes = len(args.databases)

    chance = 1.0 / n_classes
    acc = k_way(x_all, y_all, n_classes, args.folds, rng)
    control = negative_control(x_all, y_all, n_classes, args.folds, rng)
    centroid_rows, global_scale = lodo_centroid_ratio(x_all, y_all)
    names = sorted(args.databases)
    for row in centroid_rows:
        row["held_out"] = names[row["held_out_label"]]
        del row["held_out_label"]

    summary = {
        "schema": 1,
        "purpose": "research-read-only-audit",
        "audit": "p2-2-db-identity-probe",
        "hypothesis": (
            "the frozen representation encodes source-database identity; if so, "
            "database identity is linearly readable from the pooled features, and each "
            "database occupies its own region of the feature space"
        ),
        "population": "unique queries of each probed source database, capped per database",
        "probe_inputs": sorted(args.databases),
        "max_per_db": args.max_per_db,
        "n_databases": n_classes,
        "n_rows": int(len(y_all)),
        "feature_dim": int(x_all.shape[1]),
        "feature_scale_median_std": global_scale,
        "k_way": {
            "folds": args.folds,
            "accuracy": round(acc, 4),
            "chance": round(chance, 4),
            "excess_over_chance": round(acc - chance, 4),
            "note": (
                "queries of the same database appear in both train and test folds; "
                "this measures readability of database identity, not generalisation"
            ),
        },
        "negative_control": {
            "accuracy": round(control, 4),
            "chance": round(chance, 4),
            "note": (
                "the same probe with the database labels permuted; a value near "
                "1/k confirms the k-way result reflects database identity rather "
                "than an artefact of the probe or of row order"
            ),
        },
        "lodo_centroid_ratio": centroid_rows,
        "evidence_level": "single_seed_frozen_checkpoint_probe",
        "needs_evidence": True,
        "needs_evidence_reason": (
            "one seed (42), one frozen checkpoint, linear probe only; "
            "no target database was read"
        ),
        "target_databases_read": [],
    }
    protocol = {
        "cache_root": args.cache_root,
        "split": args.split,
        "checkpoint": {
            "path": args.checkpoint,
            "seed": metadata.get("seed"),
            "method": metadata.get("method"),
            "train_databases": sorted(metadata["train_databases"]),
            "validation_databases": sorted(metadata["validation_databases"]),
            "model_config": state.get("model_config"),
            "transform": state.get("transform"),
            "frozen": True,
            "selection": "none (passed explicitly; this script never chooses)",
        },
        "probe": {
            "feature": "SetEstimator._head_input (pooled vector entering the output MLP)",
            "classifier": "multinomial logistic regression, numpy, full batch",
            "epochs": 400,
            "learning_rate": 0.5,
            "l2": 1e-3,
            "standardisation": "fit on the training fold only",
            "k_way_protocol": f"{args.folds}-fold over queries, folds are random",
            "negative_control": "database labels permuted, same k-way probe; expect 1/k",
            "lodo_centroid_ratio": (
                "standardise on the training databases, then compare the held-out "
                "centroid distance to the nearest trained centroid against the mean "
                "within-database spread"
            ),
        },
        "seconds": round(time.perf_counter() - started, 1),
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    Path(args.protocol).write_text(
        json.dumps(protocol, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    print(json.dumps({"k_way": acc, "control": control}, ensure_ascii=False))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
