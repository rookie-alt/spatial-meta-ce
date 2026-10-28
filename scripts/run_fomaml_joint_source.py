#!/usr/bin/env python3
"""Run the nine fixed-budget 112 training cells. Dry-run by default."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from spatial_meta_ce.checkpoints import load_checkpoint  # noqa: E402
from spatial_meta_ce.cli import data_settings, read_json  # noqa: E402
from spatial_meta_ce.manifests import file_hash  # noqa: E402
from spatial_meta_ce.plans import TARGET_DATABASES  # noqa: E402
from spatial_meta_ce.source_eval import (  # noqa: E402
    ARMS,
    validate_eval_checkpoint,
)
from spatial_meta_ce.train import FIXED_MODEL, FIXED_SEEDS, train  # noqa: E402


def build_runs(base_config, plans_dir, out_dir, seeds, arms):
    if base_config.get("fixed_budget") is not True:
        raise ValueError("Base config must set fixed_budget=true")
    if base_config.get("model") != FIXED_MODEL:
        raise ValueError("Base config does not use the frozen 112 model")
    seeds = tuple(int(seed) for seed in seeds)
    arms = tuple(arms)
    if any(seed not in FIXED_SEEDS for seed in seeds):
        raise ValueError(f"Seeds must be a subset of {FIXED_SEEDS}")
    if any(arm not in ARMS for arm in arms):
        raise ValueError(f"Arms must be a subset of {tuple(ARMS)}")
    plans_dir = Path(plans_dir)
    out_dir = Path(out_dir)
    runs = []
    for arm in arms:
        method, objective = ARMS[arm]
        for seed in seeds:
            config = {
                **base_config,
                "method": method,
                "meta_objective": objective,
                "seed": seed,
                "training_plan": str(
                    (plans_dir / f"training_plan_seed{seed}.json").resolve()
                ),
            }
            runs.append(
                {
                    "arm": arm,
                    "seed": seed,
                    "config": config,
                    "output": str(out_dir / arm / f"seed{seed}"),
                }
            )
    return runs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/fomaml_joint_sum_4k.json")
    parser.add_argument("--plans", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(FIXED_SEEDS))
    parser.add_argument("--arms", nargs="+", default=list(ARMS))
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    base = read_json(args.config)
    data_config = data_settings(read_json(base["data_config"]), base["data_config"])
    splits = data_config["splits"]
    seen = set(splits["train"]) | set(splits["validation"])
    leaked = seen & TARGET_DATABASES
    if leaked:
        raise SystemExit(f"Target databases present in data split: {sorted(leaked)}")
    runs = build_runs(base, args.plans, args.out, args.seeds, args.arms)
    specification = {
        "config": args.config,
        "plans": args.plans,
        "runs": [
            {"arm": run["arm"], "seed": run["seed"], "output": run["output"]}
            for run in runs
        ],
        "device": args.device,
        "target_databases_read": [],
    }
    print(json.dumps(specification, ensure_ascii=False, indent=2))
    if not args.execute:
        print("Dry run only. Add --execute after a separate execution authorization.")
        return
    import torch

    torch.set_num_threads(args.threads)
    output = Path(args.out)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Training requires a new empty output directory")
    protocols = []
    for run in runs:
        plan_path = Path(run["config"]["training_plan"])
        if not plan_path.is_file():
            raise FileNotFoundError(f"Missing sealed plan: {plan_path}")
        result = train(run["config"], data_config, run["output"], args.device)
        checkpoint = Path(run["output"]) / "last.pt"
        if not checkpoint.is_file():
            raise FileNotFoundError("Fixed-budget training did not write last.pt")
        if (Path(run["output"]) / "best.pt").exists():
            raise ValueError("Fixed-budget training wrote a forbidden best.pt")
        _, state = load_checkpoint(checkpoint, "cpu")
        validate_eval_checkpoint(state, run["arm"], run["seed"])
        audit_path = Path(run["output"]) / "gradient_audit.json"
        protocol = {
            "schema": 1,
            "kind": "fomaml_joint_source_training_protocol",
            "arm": run["arm"],
            "seed": run["seed"],
            "config": run["config"],
            "checkpoint": str(checkpoint.resolve()),
            "checkpoint_sha256": file_hash(checkpoint),
            "training_plan_sha256": state["metadata"]["training_plan_sha256"],
            "gradient_audit_sha256": file_hash(audit_path),
            "counts": result["counts"],
            "total_seconds": result["total_seconds"],
            "purpose": "research",
        }
        (Path(run["output"]) / "run_protocol.json").write_text(
            json.dumps(protocol, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        )
        protocols.append(protocol)
        print(json.dumps(protocol, ensure_ascii=False))
    (output / "runs.json").write_text(
        json.dumps(protocols, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )


if __name__ == "__main__":
    main()
