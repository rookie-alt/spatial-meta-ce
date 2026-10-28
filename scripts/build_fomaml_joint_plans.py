#!/usr/bin/env python3
"""Build the sealed source-only plans for doc 112.

This script reads the source caches, writes deterministic training and source
evaluation plans, and stops.  It never trains, evaluates a checkpoint, or
opens a target database.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from spatial_meta_ce.cli import data_settings, read_json  # noqa: E402
from spatial_meta_ce.manifests import file_hash  # noqa: E402
from spatial_meta_ce.plans import (  # noqa: E402
    EVAL_DATABASES,
    EXPECTED_UNIQUE,
    TARGET_DATABASES,
    build_source_eval_plan,
    build_training_plan,
)
from spatial_meta_ce.train import (  # noqa: E402
    FIXED_ACCUMULATION,
    FIXED_META_BATCH,
    FIXED_QUERY_SIZE,
    FIXED_STEPS,
    FIXED_SUPPORT_SIZES,
    FIXED_SEEDS,
)


def _write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def build_plans(
    data_config,
    seeds,
    out_dir,
    config_path=None,
    steps=FIXED_STEPS,
    expected_unique=EXPECTED_UNIQUE,
):
    seeds = tuple(int(seed) for seed in seeds)
    if not seeds:
        raise ValueError("At least one seed is required")
    if any(seed not in FIXED_SEEDS for seed in seeds):
        raise ValueError(f"Seeds must be a subset of {FIXED_SEEDS}")
    splits = data_config["splits"]
    if tuple(splits["validation"]) != EVAL_DATABASES:
        raise ValueError(f"Validation split must be exactly {EVAL_DATABASES}")
    seen = set(splits["train"]) | set(splits["validation"])
    leaked = seen & TARGET_DATABASES
    if leaked:
        raise ValueError(f"Target databases must not enter the plan: {sorted(leaked)}")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    training = {}
    for seed in seeds:
        plan = build_training_plan(
            data_config,
            seed=seed,
            steps=steps,
            meta_batch=FIXED_META_BATCH,
            accumulation=FIXED_ACCUMULATION,
            support_sizes=list(FIXED_SUPPORT_SIZES),
            query_size=FIXED_QUERY_SIZE,
        )
        path = out_dir / f"training_plan_seed{seed}.json"
        _write_json(path, plan)
        training[str(seed)] = {
            "path": str(path),
            "file_sha256": file_hash(path),
            "plan_sha256": plan["plan_sha256"],
            "episodes": len(plan["episodes"]),
        }
    eval_plan = build_source_eval_plan(data_config, expected_unique=expected_unique)
    eval_path = out_dir / "source_eval_plan.json"
    _write_json(eval_path, eval_plan)
    binding = {
        "schema": 1,
        "kind": "fomaml_joint_source_plan_binding",
        "seeds": list(seeds),
        "steps": steps,
        "meta_batch": FIXED_META_BATCH,
        "accumulation": FIXED_ACCUMULATION,
        "query_size": FIXED_QUERY_SIZE,
        "support_sizes": list(FIXED_SUPPORT_SIZES),
        "expected_unique": dict(EXPECTED_UNIQUE),
        "training_plans": training,
        "source_eval_plan": {
            "path": str(eval_path),
            "file_sha256": file_hash(eval_path),
            "plan_sha256": eval_plan["plan_sha256"],
        },
        "config_sha256": file_hash(config_path) if config_path else None,
        "target_databases_read": [],
    }
    binding_path = out_dir / "plan_binding.json"
    _write_json(binding_path, binding)
    return binding


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-config", default="configs/data.json")
    parser.add_argument("--out", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(FIXED_SEEDS))
    parser.add_argument("--config", help="training config to bind by file hash")
    args = parser.parse_args()
    config = read_json(args.config) if args.config else None
    data_config_path = (
        config.get("data_config", "configs/data.json") if config else args.data_config
    )
    data_config = data_settings(read_json(data_config_path), data_config_path)
    binding = build_plans(
        data_config,
        args.seeds,
        args.out,
        config_path=args.config,
    )
    print(json.dumps(binding, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
