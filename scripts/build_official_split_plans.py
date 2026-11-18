#!/usr/bin/env python3
"""Freeze the official PRICE split inputs: 26 training databases, 4 test ones.

Writes new JSON files only. It never trains, evaluates, reads a test database
or signs an authorization; those remain separate, separately approved steps.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from spatial_meta_ce.manifests import file_hash  # noqa: E402
from spatial_meta_ce.official_split import (  # noqa: E402
    OFFICIAL_TRAIN_DATABASES,
    TEST_DATABASES,
    build_official_capacity,
    official_data_config,
    validate_official_capacity,
)
from spatial_meta_ce.plans import build_training_plan  # noqa: E402

SEEDS = (42, 43, 44)


def write_new(path: Path, payload: dict) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(path)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-config", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--meta-batch", type=int, default=4)
    parser.add_argument("--accumulation", type=int, default=1)
    parser.add_argument("--query-size", type=int, default=64)
    parser.add_argument("--support-sizes", default="5,10,20,50,100")
    args = parser.parse_args()

    source = json.loads(Path(args.data_config).read_text())
    config = official_data_config(source)
    capacity = validate_official_capacity(build_official_capacity(config))
    support_sizes = [int(value) for value in args.support_sizes.split(",")]

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    write_new(out / "data_config_official_26.json", config)
    write_new(out / "capacity_official_26.json", capacity)

    manifest = {
        "schema": 1,
        "status": "unsigned",
        "kind": "official_split_26_train_4_test",
        "train_databases": list(OFFICIAL_TRAIN_DATABASES),
        "test_databases": list(TEST_DATABASES),
        "capacity_sha256": capacity["capacity_sha256"],
        "max_counts": capacity["max_counts"],
        "source_data_config": str(Path(args.data_config).resolve()),
        "source_data_config_sha256": file_hash(args.data_config),
        "plans": {},
    }
    for seed in SEEDS:
        plan = build_training_plan(
            config,
            seed=seed,
            steps=args.steps,
            meta_batch=args.meta_batch,
            accumulation=args.accumulation,
            support_sizes=support_sizes,
            query_size=args.query_size,
            allow_empty_validation=True,
        )
        path = out / f"training_plan_seed{seed}.json"
        write_new(path, plan)
        manifest["plans"][str(seed)] = {
            "path": str(path),
            "plan_sha256": plan["plan_sha256"],
            "file_sha256": file_hash(path),
            "episodes": len(plan["episodes"]),
        }
    write_new(out / "official_split_manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())