#!/usr/bin/env python3
"""Official 26-train/4-test runner: six supervised training runs, no evaluation.

The parent re-verifies every frozen input before any model import or output
directory, supervises one worker, and seals the six run records. It never
reads a test database and never signs its own authorization.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PROTOCOL = "122-official-26"
PURPOSE = "official-split-training"
ALLOWED_ARMS = ("A1", "A1-PG")
ALLOWED_SEEDS = (42, 43, 44)
MIN_STEPS = 1000


def file_sha256(path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_bound(path, expected: str) -> dict:
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError(f"Bound input changed: {path}")
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def run_key(arm: str, seed: int) -> str:
    return f"{arm}-seed{seed}"


def derive_config(
    template: dict, spec: dict, capacity: dict, arm: str, seed: int, steps: int
) -> dict:
    """Carry the frozen A1 hyperparameters onto the official split."""
    from spatial_meta_ce.price_protocol import exploratory_model_config

    if template.get("fixed_stage") != "122-A1" or template.get("seed") != seed:
        raise ValueError(f"Official split template is not frozen A1 seed {seed}")
    config = dict(template)
    config.pop("source_preflight", None)
    config.pop("source_preflight_sha256", None)
    config["seed"] = seed
    config["steps"] = steps
    if steps > 4000:
        config["preregistered_step_budget"] = steps
        config["preregistration"] = spec.get("preregistration", "")
    config["fixed_budget"] = True
    config["fixed_stage"] = f"122-{arm}"
    config["data_config"] = spec["bindings"]["data_config"]["path"]
    config["data_config_sha256"] = spec["bindings"]["data_config"]["sha256"]
    config["training_plan"] = spec["bindings"]["training_plans"][str(seed)]["path"]
    config["model"] = (
        {**template["model"], "capacity": capacity}
        if arm == "A1"
        else exploratory_model_config(capacity)
    )
    return config


def verify_spec(spec: dict) -> dict:
    from scripts.run_price_meta_a import validate_budget

    if (
        spec.get("protocol") != PROTOCOL
        or spec.get("purpose") != PURPOSE
        or not spec.get("arms")
        or not set(spec.get("arms")) <= set(ALLOWED_ARMS)
        or len(set(spec["arms"])) != len(spec["arms"])
        or not spec.get("seeds")
        or not set(spec.get("seeds")) <= set(ALLOWED_SEEDS)
        or len(set(spec["seeds"])) != len(spec["seeds"])
        or type(spec.get("steps")) is not int
        or spec.get("steps") < MIN_STEPS
        or (spec.get("steps") > 4000
            and not str(spec.get("preregistration", "")).strip())
        or spec.get("target_databases_read") != []
    ):
        raise ValueError("Official split specification is not the frozen plan")
    budget = validate_budget(spec.get("budget"))
    bindings = spec["bindings"]
    if Path(bindings["code_dir"]).resolve() != ROOT:
        raise ValueError("Official split runner must execute from the bound code tree")
    for name, expected in bindings["code_sha256"].items():
        if file_sha256(ROOT / name) != expected:
            raise ValueError(f"Official split code differs from the freeze: {name}")
    data = read_bound(bindings["data_config"]["path"], bindings["data_config"]["sha256"])
    capacity = read_bound(bindings["capacity"]["path"], bindings["capacity"]["sha256"])
    from spatial_meta_ce.official_split import validate_official_capacity

    validate_official_capacity(capacity)
    if capacity["capacity_sha256"] != bindings["capacity"]["capacity_sha256"]:
        raise ValueError("Official capacity digest differs from the specification")
    read_bound(bindings["split_manifest"]["path"], bindings["split_manifest"]["sha256"])
    plans = {}
    from spatial_meta_ce.plans import validate_training_plan

    for seed in spec["seeds"]:
        record = bindings["training_plans"][str(seed)]
        plan = read_bound(record["path"], record["sha256"])
        validate_training_plan(plan, data, allow_empty_validation=True)
        if plan["plan_sha256"] != record["plan_sha256"] or plan["seed"] != seed:
            raise ValueError(f"Official training plan binding differs: seed {seed}")
        if plan["steps"] != spec["steps"]:
            raise ValueError(f"Official training plan step count differs: seed {seed}")
        plans[str(seed)] = plan
    configs = {}
    for seed in spec["seeds"]:
        record = bindings["templates"][str(seed)]
        template = read_bound(record["path"], record["sha256"])
        for arm in spec["arms"]:
            configs[run_key(arm, seed)] = derive_config(
                template, spec, capacity, arm, seed, spec["steps"]
            )
    return {"budget": budget, "data": data, "capacity": capacity, "plans": plans, "configs": configs}


def validate_authorization(authorization: dict, spec_sha256: str, budget: dict, output: Path) -> dict:
    from scripts.run_price_meta_a import canonical_digest

    if (
        authorization.get("protocol") != PROTOCOL
        or authorization.get("purpose") != PURPOSE
        or authorization.get("authorized") is not True
        or authorization.get("formal_training_authorized") is not True
        or authorization.get("target_databases_read") != []
    ):
        raise ValueError("Official split authorization is incomplete; still unsigned?")
    if (
        authorization.get("spec_sha256") != spec_sha256
        or authorization.get("budget_sha256") != canonical_digest(budget)
        or authorization.get("output_dir") != str(output)
    ):
        raise ValueError("Official split authorization does not bind this run")
    granted_by = authorization.get("granted_by")
    granted_at = authorization.get("granted_at")
    if not isinstance(granted_by, str) or not granted_by.strip():
        raise ValueError("Official split authorization lacks an approving identity")
    if not isinstance(granted_at, str):
        raise ValueError("Official split authorization lacks a dated approval")
    issued = datetime.fromisoformat(granted_at)
    if issued.tzinfo is None or issued.utcoffset() is None:
        raise ValueError("Official approval timestamp requires a timezone")
    return authorization


def worker_main(args) -> int:
    from scripts.run_price_meta_a import _write_phase, write_json

    if os.environ.get("META65_SUPERVISED_MODE") != "formal":
        raise ValueError("Official split worker requires the formal supervisor")
    if os.environ.get("META65_SUPERVISOR_PID") != str(os.getppid()):
        raise ValueError("Official split worker requires its own supervisor parent")
    spec = read_bound(args.spec, args.spec_sha256)
    resolved = verify_spec(spec)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    _write_phase(output, "preparation")

    import torch
    from spatial_meta_ce.train import train
    from scripts.run_price_meta_a import CudaMemoryTelemetry

    torch.set_num_threads(int(spec.get("threads", 1)))
    torch.set_num_interop_threads(1)
    if spec["device"] == "cuda":
        CudaMemoryTelemetry(output, torch).start()

    records = []
    for arm in spec["arms"]:
        for seed in spec["seeds"]:
            key = run_key(arm, seed)
            run_dir = output / "runs" / key
            result = train(resolved["configs"][key], resolved["data"], run_dir, spec["device"])
            checkpoint = run_dir / "last.pt"
            if not checkpoint.is_file() or (run_dir / "best.pt").exists():
                raise ValueError(f"Official run wrote an invalid checkpoint set: {key}")
            counts = result.get("counts", {})
            if counts.get("optimizer_steps") != spec["steps"]:
                raise ValueError(
                    f"Official run did not complete {spec['steps']} updates: {key}"
                )
            records.append(
                {
                    "key": key,
                    "arm": arm,
                    "seed": seed,
                    "checkpoint": str(checkpoint),
                    "checkpoint_sha256": file_sha256(checkpoint),
                    "training_plan_sha256": resolved["plans"][str(seed)]["plan_sha256"],
                    "optimizer_steps": counts.get("optimizer_steps"),
                    "parameters": result.get("parameters"),
                    "wall_seconds": result.get("total_seconds"),
                }
            )
    write_json(output / "training_records.json", {"runs": records})
    _write_phase(output, "certification")
    write_json(
        output / "status.json",
        {
            "status": "worker_complete",
            "mode": "formal",
            "cells": len(records),
            "target_databases_read": [],
        },
    )
    return 0


def parent_main(args) -> int:
    from scripts.meta65_joint_supervisor import sample_process, supervise
    from scripts.run_price_meta_a import (
        BudgetTracker,
        UnitDeadline,
        check_output_path,
        parent_sample_fn,
        supervisor_budget,
        write_json,
    )

    spec_path = Path(args.spec)
    spec_sha256 = file_sha256(spec_path)
    if spec_sha256 != args.spec_sha256:
        raise ValueError("Official split specification bytes changed")
    spec = json.loads(spec_path.read_text())
    resolved = verify_spec(spec)
    output = Path(args.output).absolute()
    authorization = read_bound(args.authorization, args.authorization_sha256)
    validate_authorization(authorization, spec_sha256, resolved["budget"], output)
    check_output_path(output, resolved["data"]["data_root"])
    if not args.execute:
        print(
            json.dumps(
                {"runs": [run_key(a, s) for a in spec["arms"] for s in spec["seeds"]]},
                indent=2,
            )
        )
        print("Dry run only. Add --execute after the authorization is signed.")
        return 0

    if spec["device"] == "cuda":
        if not spec.get("cuda_device"):
            raise ValueError("Official split CUDA execution needs an explicit device index")
        os.environ["CUDA_VISIBLE_DEVICES"] = str(spec["cuda_device"])
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    output.mkdir(parents=True)
    worker_output = output / "worker"
    command = [
        sys.executable,
        "-I",
        "-B",
        str(Path(__file__).resolve()),
        "worker",
        "--spec",
        str(spec_path.resolve()),
        "--spec-sha256",
        spec_sha256,
        "--authorization",
        str(Path(args.authorization).resolve()),
        "--authorization-sha256",
        args.authorization_sha256,
        "--output",
        str(worker_output),
    ]
    tracker = BudgetTracker(resolved["budget"], requires_gpu_usage=spec["device"] == "cuda")
    monitor = UnitDeadline(
        worker_output,
        resolved["budget"]["run_deadline_seconds"],
        resolved["budget"]["eval_deadline_seconds"],
    )
    sample = parent_sample_fn(
        monitor,
        tracker,
        lambda target: sample_process(target, mode="formal", output=worker_output),
    )
    record = supervise(
        command,
        worker_output,
        mode="formal",
        budget=supervisor_budget(resolved["budget"]),
        sample_fn=sample,
        root=str(ROOT),
        grace=10,
        formal_cells=len(spec["arms"]) * len(spec["seeds"]),
    )
    if record["status"] != "completed":
        raise RuntimeError(f"Official split runs failed: {record.get('failure')}")
    tracker.finalize()
    write_json(
        output / "completion.json",
        {
            "status": "official_split_completed",
            "runs": len(spec["arms"]) * len(spec["seeds"]),
            "spec_sha256": spec_sha256,
            "authorization_sha256": args.authorization_sha256,
            "supervisor": record,
            "budget_usage": tracker.state(),
            "target_databases_read": [],
        },
    )
    print(json.dumps({"status": "official_split_completed"}, indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("parent", "worker"))
    parser.add_argument("--spec", required=True)
    parser.add_argument("--spec-sha256", required=True)
    parser.add_argument("--authorization", required=True)
    parser.add_argument("--authorization-sha256", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    return parent_main(args) if args.mode == "parent" else worker_main(args)


if __name__ == "__main__":
    raise SystemExit(main())