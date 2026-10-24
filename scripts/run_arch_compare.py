#!/usr/bin/env python3
"""Train the four architectures over six seeds, source validation only.

Why a new runner: ``scripts/run_experiments.py`` hardcodes the config names
(``meta.json`` / ``erm.json``) and the seed list ``[42, 43, 44]``, so it cannot
express an architecture matrix.

Two deliberate departures from the 2026-09-27 selection:

* **``last.pt``, not ``best.pt``.** The old runs let each arm early-stop on
  ``validation_seed=1729`` and then reported that arm on ``plan_seed=2718`` --
  two different episode plans, so every arm had been tuned against a slice of
  the same databases it was reported on. Fixed 4000 steps with ``last.pt``
  removes that selection entirely.
* **Fixed step count.** ``patience`` is raised to the full budget so no arm
  stops early; every arm spends the same number of updates.

Target databases (imdb/stats/genome/ergastf1) are never opened: the training
configs only name source splits, and ``arch_select`` re-checks the checkpoint
metadata before evaluating.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

ARCHS = ("set", "sum", "attn", "attn_pair")
SEEDS = (42, 43, 44, 45, 46, 47)
STEPS = 4000
TARGETS = frozenset(("imdb", "stats", "genome", "ergastf1"))
# The `set` arm keeps the historical two-key config shape so old checkpoints
# stay loadable (see spatial_meta_ce/model.py SetEstimator.config).
EXPECTED_MODEL = {
    "set": {"width": 64, "use_pg": True},
    "sum": {"width": 64, "use_pg": True, "arch": "sum", "layers": 2, "heads": 4},
    "attn": {"width": 64, "use_pg": True, "arch": "attn", "layers": 2, "heads": 4},
    "attn_pair": {
        "width": 64,
        "use_pg": True,
        "arch": "attn_pair",
        "layers": 2,
        "heads": 4,
    },
}
CONFIG_NAME = {
    "set": "meta_4k.json",
    "sum": "meta_sum_4k.json",
    "attn": "meta_attn_4k.json",
    "attn_pair": "meta_attn_pair_4k.json",
}


def arch_config(arch: str, seed: int) -> dict:
    """The 4K config for one architecture, with the seed applied.

    Seeds 43/44 were historically produced exactly this way (the configs pin
    ``seed: 42`` and the runner overrides it), so seeds 45-47 follow the same
    path rather than a separate one.
    """
    if arch not in ARCHS:
        raise ValueError(f"Unknown architecture {arch!r}")
    from spatial_meta_ce.cli import read_json

    config = read_json(ROOT / "configs" / CONFIG_NAME[arch])
    if config.get("model") != EXPECTED_MODEL[arch]:
        raise ValueError(f"Frozen model config changed for arch {arch}")
    if config.get("steps") != STEPS:
        raise ValueError(f"Frozen step budget changed for arch {arch}")
    return {**config, "seed": seed, "steps": STEPS, "patience": STEPS}


def matrix(output_root: Path) -> list[dict]:
    """Seed-major so one seed's four arms train back to back."""
    return [
        {
            "arch": arch,
            "seed": seed,
            "output": str(output_root / f"{arch}-seed{seed}"),
            "checkpoint": str(output_root / f"{arch}-seed{seed}" / "last.pt"),
        }
        for seed in SEEDS
        for arch in ARCHS
    ]


def verify(records: list[dict], expected: int) -> dict:
    """Every cell present, source-only, fixed budget, and ``last.pt``."""
    seen = set()
    for record in records:
        key = (record["arch"], record["seed"])
        if key in seen:
            raise ValueError(f"Duplicate cell {key}")
        seen.add(key)
        exposed = set(record["train_databases"]) | set(record["validation_databases"])
        if exposed & TARGETS:
            raise ValueError(f"Cell {key} trained on a target database")
        if record["optimizer_steps"] != STEPS:
            raise ValueError(f"Cell {key} did not run the frozen budget")
        if record.get("checkpoint_kind") != "last":
            raise ValueError(f"Cell {key} did not use last.pt")
    if len(records) != expected:
        raise ValueError(f"Expected {expected} cells, got {len(records)}")
    return {
        "cells": len(records),
        "seeds": sorted({r["seed"] for r in records}),
        "archs": sorted({r["arch"] for r in records}),
        "steps": STEPS,
        "checkpoint_kind": "last",
    }


def run_one(spec: dict, data: dict, device: str) -> dict:
    from spatial_meta_ce.checkpoints import load_checkpoint
    from spatial_meta_ce.train import train

    output = Path(spec["output"])
    train(arch_config(spec["arch"], spec["seed"]), data, output, device=device)
    model, state = load_checkpoint(output / "last.pt", "cpu")
    metadata = state["metadata"]
    return {
        "arch": spec["arch"],
        "seed": spec["seed"],
        "checkpoint": str(output / "last.pt"),
        "checkpoint_kind": "last",
        "optimizer_steps": state["next_step"],
        "train_databases": sorted(metadata["train_databases"]),
        "validation_databases": sorted(metadata["validation_databases"]),
        "parameters": sum(p.numel() for p in model.parameters()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="actually train; without it the workload is printed",
    )
    parser.add_argument(
        "--reuse",
        nargs="*",
        default=[],
        help="cells like sum-42 whose last.pt already exists",
    )
    args = parser.parse_args()

    plan = matrix(args.output)
    print(
        json.dumps(
            {
                "cells": len(plan),
                "steps": STEPS,
                "checkpoint_kind": "last",
                "device": args.device,
            },
            indent=2,
        )
    )
    if not args.execute:
        print("Dry run only. Add --execute AFTER reserving a GPU budget.")
        return 0

    from spatial_meta_ce.cli import data_settings, read_json, safe_output

    data_config = data_settings(
        read_json(ROOT / "configs/data.json"), ROOT / "configs/data.json"
    )
    safe_output(args.output.resolve(), data_config)
    args.output.mkdir(parents=True, exist_ok=False)

    reusable = set(args.reuse)
    records = []
    for spec in plan:
        cell = f"{spec['arch']}-seed{spec['seed']}"
        if cell in reusable:
            print(f"reusing {cell}", flush=True)
            continue
        records.append(run_one(spec, data_config, args.device))
    trained = len(plan) - len(reusable)
    report = {
        "runs": records,
        "reused": sorted(reusable),
        "verification": verify(records, trained) if records else None,
    }
    (args.output / "runs.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(f"Matrix outputs: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
