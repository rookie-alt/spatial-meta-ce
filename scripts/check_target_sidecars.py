"""Read-only preflight: a cache's struct sidecar must load through FeatureStore.

Why this exists: the sidecar is bound to the cache by content hash and loaded
lazily, so a missing or stale ``struct_log.npy`` only surfaces when a training
or evaluation batch is first built.  This script walks the same path
(``FeatureStore`` -> ``batch``) and fails fast instead.

Reading a target database's cache is an act of record: pass ``--databases``
explicitly and keep the stdout next to the evidence it supports.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from spatial_meta_ce.data import FeatureStore  # noqa: E402

TARGET_DATABASES = ("imdb", "stats", "genome", "ergastf1")


def check(cache_root: Path, split: str, database: str) -> tuple[dict, bool]:
    root = cache_root / split / database
    store = FeatureStore(root)
    sidecar = store.struct_log
    if sidecar is None:
        print(f"{database}: MISSING sidecar")
        return {}, False
    probe = np.arange(min(2, len(store.rows)), dtype=np.int64)
    batch = store.batch(probe, "cpu")
    finite = bool(np.isfinite(sidecar).all())
    ok = sidecar.shape[0] == len(store.rows) and "struct_log" in batch and finite
    manifest = json.loads((root / "struct_manifest.json").read_text())
    record = {
        "database": database,
        "rows": int(sidecar.shape[0]),
        "finite": finite,
        "batch_has_struct_log": "struct_log" in batch,
        "target_database": bool(manifest.get("target_database")),
        "bins": int(manifest.get("bins", 0)),
        "failure_count": int(manifest.get("failure_count", -1)),
        "cache_content_hash": manifest.get("cache_content_hash"),
        "anchor_median_log": float(np.median(sidecar)),
    }
    print(f"{database}: rows={record['rows']} ok={ok}")
    return record, ok


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--databases", nargs="+", default=list(TARGET_DATABASES))
    args = parser.parse_args(argv)
    for database in args.databases:
        if Path(database).name != database or database in (".", ".."):
            parser.error(f"Invalid database name {database!r}")

    records, failed = [], False
    for database in args.databases:
        record, ok = check(args.cache_root, args.split, database)
        failed = failed or not ok
        if record:
            records.append(record)
    print(json.dumps(records, indent=2))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
