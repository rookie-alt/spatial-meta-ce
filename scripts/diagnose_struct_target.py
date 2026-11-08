"""Read-only diagnosis of the structural anchor on the four target caches.

Why this exists: the K=0 population result (doc 120) says the struct-anchored
model is worse than the PG-anchored baseline on three of four targets.  Two
explanations are possible -- the sidecar is wrong, or the anchor is genuinely
weak on these databases -- and they call for opposite responses.  This script
separates them:

* error of the bare anchor (``struct_log``) by table count: a 1-table query's
  structural estimate is just the filtered row count, so the anchor must be
  near-exact at n=1; large error there would mean the sidecar/queries path is
  broken, not that the estimator is weak;
* the residual the trained head adds (``prediction - struct_log``), per seed,
  to show whether the head fixes the anchor or merely shifts it.

Nothing here trains, adapts or selects.  It reads target caches, so every
database must be named explicitly on ``--databases``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from spatial_meta_ce.checkpoints import load_checkpoint  # noqa: E402
from spatial_meta_ce.data import StorePool  # noqa: E402

TARGET_DATABASES = ("imdb", "stats", "genome", "ergastf1")


def q95(values: np.ndarray) -> float:
    return float(np.percentile(np.abs(values), 95))


def assert_struct_checkpoint(state, databases) -> None:
    """The residual is only meaningful for a struct-anchored, PG-free checkpoint.

    ``residual`` is computed as ``prediction - batch["struct_log"]``; with any
    other anchor that difference mixes the head with an anchor swap.  The
    exposure check mirrors ``scripts/target_population_eval.py``.
    """
    model_config = state["model_config"]
    if (
        model_config.get("anchor") != "struct"
        or model_config.get("use_pg") is not False
    ):
        raise ValueError(
            f"Checkpoint is not struct-anchored without PG: {model_config!r}"
        )
    metadata = state["metadata"]
    if metadata.get("adapted_database"):
        raise ValueError(
            "Checkpoint was adapted on a database and is not an independent "
            f"initialisation: {metadata['adapted_database']!r}"
        )
    exposed = set(metadata["train_databases"]) | set(metadata["validation_databases"])
    overlap = exposed & set(databases)
    if overlap:
        raise ValueError(
            f"Checkpoint exposure overlaps read targets: {sorted(overlap)}"
        )


def diagnose_database(store, checkpoint_paths: dict) -> dict:
    indices = store.unique_indices()
    true = np.asarray(store.cards[indices, 0], dtype=np.float64)
    batch = store.batch(indices, "cpu")
    anchor = batch["struct_log"].detach().cpu().numpy().astype(np.float64)
    log_true = np.log(np.maximum(true, 1.0))
    n_tables = np.asarray(store.offsets[indices + 1, 2], dtype=np.int64) - np.asarray(
        store.offsets[indices, 2], dtype=np.int64
    )

    by_tables = {}
    for count in sorted(set(n_tables.tolist())):
        mask = n_tables == count
        error = np.abs(anchor[mask] - log_true[mask])
        by_tables[int(count)] = {
            "count": int(mask.sum()),
            "median_error": float(np.median(error)),
            "q95_error": float(np.percentile(error, 95)),
            "max_error": float(error.max()),
        }

    seeds = {}
    for seed, path in sorted(checkpoint_paths.items()):
        model, state = load_checkpoint(path, "cpu")
        assert_struct_checkpoint(state, TARGET_DATABASES)
        model.eval()
        with torch.no_grad():
            prediction = model(batch).detach().cpu().numpy().astype(np.float64)
        residual = prediction - anchor
        seeds[int(seed)] = {
            "q95_error": q95(prediction - log_true),
            "residual_mean": float(residual.mean()),
            "residual_median": float(np.median(residual)),
            "residual_p95": float(np.percentile(residual, 95)),
            "residual_max": float(residual.max()),
        }

    return {
        "queries": int(len(indices)),
        "anchor_q95_error": q95(anchor - log_true),
        "anchor_median_error": float(np.median(np.abs(anchor - log_true))),
        "anchor_by_table_count": by_tables,
        "seeds": seeds,
        "sidecar": sidecar_identity(store),
    }


def sidecar_identity(store) -> dict:
    """Which sidecar produced these numbers; a rebuilt one changes every value."""
    manifest = json.loads((store.path / "struct_manifest.json").read_text())
    return {
        "struct_manifest": manifest,
        "struct_log_sha256": hashlib.sha256(
            (store.path / "struct_log.npy").read_bytes()
        ).hexdigest(),
        "cache_content_hash": store.manifest.get("content_hash"),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--databases", nargs="+", required=True)
    parser.add_argument(
        "--checkpoint",
        action="append",
        required=True,
        metavar="SEED=PATH",
        help="trained checkpoint as SEED=PATH; repeatable",
    )
    args = parser.parse_args(argv)

    checkpoints = {}
    for item in args.checkpoint:
        seed, _, path = item.partition("=")
        if not seed.isdigit() or not path:
            parser.error(f"--checkpoint wants SEED=PATH, got {item!r}")
        checkpoints[int(seed)] = path

    for database in args.databases:
        if Path(database).name != database or database in (".", ".."):
            parser.error(f"Invalid database name {database!r}")

    torch.set_num_threads(4)
    pool = StorePool(str(args.cache_root))
    for database in args.databases:
        store = pool.get(args.split, database)
        report = diagnose_database(store, checkpoints)
        print(
            f"== {database} queries={report['queries']} "
            f"anchor q95={report['anchor_q95_error']:.3f} "
            f"median={report['anchor_median_error']:.3f}"
        )
        sidecar = report["sidecar"]
        print(
            f"   sidecar bins={sidecar['struct_manifest'].get('bins')} "
            f"rows={sidecar['struct_manifest'].get('rows')} "
            f"cache={str(sidecar['cache_content_hash'])[:12]} "
            f"struct_log_sha256={sidecar['struct_log_sha256'][:12]}"
        )
        for count, row in report["anchor_by_table_count"].items():
            print(
                f"   n_tables={count:2d} count={row['count']:5d} "
                f"median={row['median_error']:6.3f} q95={row['q95_error']:6.3f} "
                f"max={row['max_error']:6.3f}"
            )
        for seed, row in report["seeds"].items():
            print(
                f"   seed{seed}: q95={row['q95_error']:6.3f} "
                f"residual mean={row['residual_mean']:+.3f} "
                f"median={row['residual_median']:+.3f} "
                f"p95={row['residual_p95']:+.3f} max={row['residual_max']:+.3f}"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
