#!/usr/bin/env python3
"""Frozen P1 mapping-ablation protocol: 3 arms x 3 seeds, source-only.

Arms differ only in the mapping input. `sum` is the un-mapped baseline; `true`
and `shuffled` share the same sum_map architecture and parameter count, so their
difference isolates the join/filter association rather than capacity. Target
databases are never opened.

Dry run (default) prints the matrix and the pairing invariants. `--execute`
trains the nine runs sequentially and stops at the first violated invariant.
Endpoint analysis lives in scripts/mapping_ablation_endpoint.py.
"""

import argparse
import datetime
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from scripts.mapping_smoke import available_gpu, state_hash
from spatial_meta_ce.checkpoints import load_checkpoint
from spatial_meta_ce.cli import data_settings, read_json
from spatial_meta_ce.manifests import file_hash
from spatial_meta_ce.model import SetEstimator
from spatial_meta_ce.train import seed_all, train

ROOT = Path(__file__).resolve().parents[1]

ARMS = ("sum", "true", "shuffled")
SEEDS = (42, 43, 44)
TARGETS = frozenset(("imdb", "stats", "genome", "ergastf1"))
EXPECTED_ARCH = {"sum": "sum", "true": "sum_map", "shuffled": "sum_map"}
EXPECTED_MODE = {"sum": None, "true": "mapping", "shuffled": "shuffled_mapping"}
SOURCE26 = "/gly/weilin/spatial-meta-ce/runs/mapping-sidecars-source26-20261004"
EXPECTED_MODEL = {"width": 64, "use_pg": True, "layers": 2, "heads": 4}
FROZEN = {
    "method": "erm",
    "steps": 4000,
    "meta_batch": 4,
    "accumulation": 1,
    "outer_lr": 0.0003,
    "inner_lr": 0.001,
    "inner_steps": 5,
    "adapt_scope": "all",
    "support_sizes": [5, 10, 20, 50, 100],
    "query_size": 64,
    "loss": "smooth_l1",
    "validate_every": 100,
    "validation_runs": 5,
    "validation_seed": 1729,
    "patience": 41,
    "min_delta": 0.0001,
    "data_config": "configs/data.json",
}


def arm_config_path(arm: str) -> Path:
    if arm not in ARMS:
        raise ValueError(f"Unknown ablation arm: {arm}")
    return ROOT / "configs" / f"p1_map_{arm}.json"


def run_matrix(output_root: str) -> list:
    """Seed-major order so each seed's three arms train consecutively."""
    root = Path(output_root)
    return [
        {
            "arm": arm,
            "seed": seed,
            "config": str(arm_config_path(arm)),
            "output": str(root / f"{arm}-seed{seed}"),
        }
        for seed in SEEDS
        for arm in ARMS
    ]


def load_run_config(arm: str, seed: int) -> dict:
    config = read_json(arm_config_path(arm))
    expected_model = {**EXPECTED_MODEL, "arch": EXPECTED_ARCH[arm]}
    expected_mapping = (
        None
        if EXPECTED_MODE[arm] is None
        else {"root": SOURCE26, "mode": EXPECTED_MODE[arm]}
    )
    if config.get("model") != expected_model:
        raise ValueError(f"Frozen model config changed for arm {arm}")
    if config.get("mapping") != expected_mapping:
        raise ValueError(f"Frozen mapping source or mode changed for arm {arm}")
    changed = {k: (config.get(k), v) for k, v in FROZEN.items() if config.get(k) != v}
    if changed:
        raise ValueError(f"Frozen training config changed: {sorted(changed)}")
    return {**config, "seed": seed}


def initial_state(config: dict) -> dict:
    seed_all(config["seed"])
    model = SetEstimator(**config["model"])
    return {
        "initialization_hash": state_hash(model),
        "parameters": sum(p.numel() for p in model.parameters()),
    }


SHARED_KEYS = (
    "sample_plan_digest",
    "sample_plan_episodes",
    "counts",
    "validation_plan_hash",
    "optimizer_steps",
    "manifest_hashes",
)


def verify_seed_group(group: dict, seed: int) -> dict:
    """One seed's three arms must be perfectly paired; returns the capacity note."""
    if set(group) != set(ARMS):
        raise ValueError(f"Seed {seed} is missing an arm")
    for key in SHARED_KEYS:
        if len({json.dumps(group[a][key], sort_keys=True) for a in ARMS}) != 1:
            raise ValueError(f"Seed {seed} arms disagree on {key}")
    if group["true"]["initialization_hash"] != group["shuffled"]["initialization_hash"]:
        raise ValueError(f"Seed {seed} mapping arms do not share initialization")
    if group["true"]["parameters"] != group["shuffled"]["parameters"]:
        raise ValueError(f"Seed {seed} mapping arms differ in capacity")
    left = {
        db: (v["manifest_hash"], v["jsonl_hash"])
        for db, v in group["true"]["mapping_provenance"].items()
    }
    right = {
        db: (v["manifest_hash"], v["jsonl_hash"])
        for db, v in group["shuffled"]["mapping_provenance"].items()
    }
    if left != right or not left:
        raise ValueError(f"Seed {seed} mapping arms read different sidecars")
    for record in group.values():
        if (record["train_databases"] | record["validation_databases"]) & TARGETS:
            raise ValueError(f"Seed {seed} run trained on a target database")
    return {"sum": group["sum"]["parameters"], "sum_map": group["true"]["parameters"]}


def verify_pairing(records: list) -> dict:
    """Every arm of a seed must share the same sample stream and validation plan."""
    if {r["seed"] for r in records} != set(SEEDS):
        raise ValueError("Ablation must cover exactly the frozen seeds")
    if len(records) != len(ARMS) * len(SEEDS):
        raise ValueError("Ablation must contain exactly nine runs")
    by_seed = defaultdict(dict)
    for record in records:
        by_seed[record["seed"]][record["arm"]] = record
    capacity = {
        seed: verify_seed_group(group, seed) for seed, group in sorted(by_seed.items())
    }
    return {
        "seeds": sorted(by_seed),
        "arms": list(ARMS),
        "shared_digest": by_seed[SEEDS[0]]["sum"]["sample_plan_digest"],
        "parameters": capacity,
    }


def run_one(spec: dict, data: dict, device: str, gpu_uuid: str) -> dict:
    config = load_run_config(spec["arm"], spec["seed"])
    initial = initial_state(config)
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    started = time.perf_counter()
    output = Path(spec["output"])
    summary = train(config, data, output, device=device)
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    _, state = load_checkpoint(output / "last.pt", device)
    extra = state["extra"]
    if extra["counts"]["optimizer_steps"] != config["steps"]:
        raise ValueError("Early stopping truncated the frozen budget")
    return {
        **{k: spec[k] for k in ("arm", "seed", "config", "output")},
        "config_sha256": file_hash(spec["config"]),
        **initial,
        "optimizer_steps": extra["counts"]["optimizer_steps"],
        "sample_plan_digest": extra["sample_plan_digest"],
        "sample_plan_episodes": extra["sample_plan_episodes"],
        "counts": extra["counts"],
        "validation_plan_hash": file_hash(output / "validation_manifest.json"),
        "mapping_provenance": state["metadata"].get("mapping_provenance", {}),
        "manifest_hashes": state["metadata"]["manifest_hashes"],
        "train_databases": set(state["metadata"]["train_databases"]),
        "validation_databases": set(state["metadata"]["validation_databases"]),
        "checkpoint_sha256": file_hash(output / "last.pt"),
        "device": device,
        "gpu_uuid": gpu_uuid,
        "peak_memory_bytes": torch.cuda.max_memory_allocated()
        if device.startswith("cuda")
        else 0,
        "seconds": elapsed,
        "summary": summary,
    }


def serialise(record: dict) -> dict:
    return {k: (sorted(v) if isinstance(v, set) else v) for k, v in record.items()}


def configure_determinism() -> None:
    # CuBLAS needs this before first use or deterministic mode aborts the run.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def run_all(output: Path, data: dict, device: str, gpu_index: str | None) -> dict:
    # Check occupancy once: after run one starts, our own process owns the GPU.
    gpu_uuid = available_gpu(gpu_index) if gpu_index is not None else "cpu"
    records = []
    for spec in run_matrix(str(output)):
        record = run_one(spec, data, device, gpu_uuid)
        records.append(record)
        if len(records) % len(ARMS) == 0:
            verify_seed_group(
                {r["arm"]: r for r in records[-len(ARMS) :]}, record["seed"]
            )
        print(
            json.dumps(
                {
                    "arm": record["arm"],
                    "seed": record["seed"],
                    "optimizer_steps": record["optimizer_steps"],
                    "seconds": round(record["seconds"], 1),
                    "peak_memory_bytes": record["peak_memory_bytes"],
                }
            ),
            flush=True,
        )
    verdict = verify_pairing(records)
    report = {
        "purpose": "p1 mapping ablation; source-only; not target-tuned",
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "device": device,
        "seeds": list(SEEDS),
        "arms": list(ARMS),
        "pairing": verdict,
        "runs": [serialise(r) for r in records],
    }
    (output / "ablation_runs.json").write_text(
        json.dumps(report, indent=2, allow_nan=False)
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Frozen P1 mapping-ablation training matrix"
    )
    parser.add_argument(
        "--output", required=True, help="new run directory for the nine trainings"
    )
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--gpu-index", help="physical GPU index; must equal CUDA_VISIBLE_DEVICES"
    )
    args = parser.parse_args()
    if not args.execute:
        for arm in ARMS:
            load_run_config(arm, SEEDS[0])
        print(
            json.dumps(
                {
                    "runs": run_matrix(args.output),
                    "steps_per_run": 4000,
                    "invariants": [
                        "sample_plan_digest",
                        "counts",
                        "validation_plan_hash",
                        "optimizer_steps",
                        "manifest_hashes",
                        "sum_map initialization and capacity",
                    ],
                    "endpoint": "4+ table unique-query zero-shot Q95 on source validation",
                },
                indent=2,
            )
        )
        print("Dry run only. Add --execute AFTER reserving a GPU.")
        return
    if not args.device.startswith("cuda"):
        raise ValueError("Formal runs require --device cuda")
    if (
        args.gpu_index is None
        or os.environ.get("CUDA_VISIBLE_DEVICES") != args.gpu_index
    ):
        raise ValueError("Set CUDA_VISIBLE_DEVICES to exactly --gpu-index")
    config_path = str(arm_config_path("sum"))
    data = data_settings(read_json(config_path), config_path)
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError("Use a new run directory")
    output.mkdir(parents=True)
    configure_determinism()
    run_all(output, data, args.device, args.gpu_index)
    print("Ablation outputs:", output)


if __name__ == "__main__":
    main()
