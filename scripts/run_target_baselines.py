#!/usr/bin/env python3
"""D-stage zero-shot baselines: PG and PRICE on the four test databases.

Runs only under a signed evaluation authorization and writes one frozen
protocol per baseline; results never feed back into any selection.
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
PURPOSE = "target-zero-shot-baselines"
BASELINES = ("pg", "price")


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


def verify_spec(spec: dict) -> dict:
    from scripts.run_price_meta_a import validate_budget
    from spatial_meta_ce.official_split import TEST_DATABASES

    if (
        spec.get("protocol") != PROTOCOL
        or spec.get("purpose") != PURPOSE
        or spec.get("shots") != [0]
        or spec.get("baselines") != list(BASELINES)
        or spec.get("databases") != list(TEST_DATABASES)
        or spec.get("batch_size") != 128
    ):
        raise ValueError("Target baseline specification is not the frozen D plan")
    budget = validate_budget(spec.get("budget"))
    bindings = spec["bindings"]
    if Path(bindings["code_dir"]).resolve() != ROOT:
        raise ValueError("Target baseline must execute from the bound code tree")
    for name, expected in bindings["code_sha256"].items():
        if file_sha256(ROOT / name) != expected:
            raise ValueError(f"Target baseline code differs from the freeze: {name}")
    data = read_bound(bindings["data_config"]["path"], bindings["data_config"]["sha256"])
    price_path = Path(bindings["price_checkpoint"]["path"])
    if not price_path.is_file() or price_path.is_symlink():
        raise ValueError("PRICE reference weights are missing")
    if file_sha256(price_path) != bindings["price_checkpoint"]["sha256"]:
        raise ValueError("PRICE reference weights changed since the freeze")
    return {"budget": budget, "data": data, "price": str(price_path)}


def validate_authorization(authorization: dict, spec_sha256: str, budget: dict, output: Path) -> dict:
    from scripts.run_price_meta_a import canonical_digest

    databases = authorization.get("target_databases_read")
    if (
        authorization.get("protocol") != PROTOCOL
        or authorization.get("purpose") != PURPOSE
        or authorization.get("authorized") is not True
        or authorization.get("evaluation_authorized") is not True
        or not isinstance(databases, list)
        or not databases
    ):
        raise ValueError("Target baseline authorization is incomplete or unsigned")
    if (
        authorization.get("spec_sha256") != spec_sha256
        or authorization.get("budget_sha256") != canonical_digest(budget)
        or authorization.get("output_dir") != str(output)
    ):
        raise ValueError("Target baseline authorization does not bind this run")
    granted_by = authorization.get("granted_by")
    granted_at = authorization.get("granted_at")
    if not isinstance(granted_by, str) or not granted_by.strip():
        raise ValueError("Target baseline authorization lacks an approving identity")
    if not isinstance(granted_at, str):
        raise ValueError("Target baseline authorization lacks a dated approval")
    issued = datetime.fromisoformat(granted_at)
    if issued.tzinfo is None or issued.utcoffset() is None:
        raise ValueError("Target baseline approval timestamp requires a timezone")
    return authorization


def worker_main(args) -> int:
    from scripts.run_price_meta_a import _write_phase, write_json
    from spatial_meta_ce.target_eval import evaluate_target_baseline

    if os.environ.get("META65_SUPERVISED_MODE") != "formal":
        raise ValueError("Target baseline worker requires the formal supervisor")
    if os.environ.get("META65_SUPERVISOR_PID") != str(os.getppid()):
        raise ValueError("Target baseline worker requires its own supervisor parent")
    spec = read_bound(args.spec, args.spec_sha256)
    resolved = verify_spec(spec)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    _write_phase(output, "preparation")

    import torch
    from scripts.run_price_meta_a import CudaMemoryTelemetry

    if spec["device"] == "cuda":
        CudaMemoryTelemetry(output, torch).start()

    protocols = {}
    for name in BASELINES:
        protocols[name] = evaluate_target_baseline(
            name,
            resolved["data"],
            output / "baseline" / name,
            price_checkpoint=resolved["price"] if name == "price" else None,
            device=spec["device"],
            batch_size=spec["batch_size"],
        )
    _write_phase(output, "certification")
    summary = {
        "schema": 1,
        "kind": "target_zero_shot_baseline",
        "shots": [0],
        "databases": spec["databases"],
        "baselines": {
            name: {
                "macro_log_q95": protocol["macro_log_q95"],
                "per_database": {
                    unit["database"]: unit["metrics"] for unit in protocol["units"]
                },
                "predictions_sha256": protocol["predictions_sha256"],
            }
            for name, protocol in protocols.items()
        },
        "selection_use": "none",
        "target_databases_read": spec["databases"],
    }
    write_json(output / "summary.json", summary)
    write_json(
        output / "status.json",
        {
            "status": "worker_complete",
            "mode": "formal",
            "cells": len(protocols),
            "target_databases_read": spec["databases"],
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
        raise ValueError("Target baseline specification bytes changed")
    spec = json.loads(spec_path.read_text())
    resolved = verify_spec(spec)
    output = Path(args.output).absolute()
    authorization = read_bound(args.authorization, args.authorization_sha256)
    validate_authorization(authorization, spec_sha256, resolved["budget"], output)
    check_output_path(output, resolved["data"]["data_root"])
    if not args.execute:
        print(json.dumps({"baselines": list(BASELINES)}, indent=2))
        print("Dry run only. Add --execute after the baseline authorization is signed.")
        return 0

    if spec["device"] == "cuda":
        if not spec.get("cuda_device"):
            raise ValueError("Target baseline CUDA execution needs an explicit device index")
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
        formal_cells=len(BASELINES),
    )
    if record["status"] != "completed":
        raise RuntimeError(f"Target baseline failed: {record.get('failure')}")
    tracker.finalize()
    write_json(
        output / "completion.json",
        {
            "status": "target_baselines_completed",
            "baselines": list(BASELINES),
            "spec_sha256": spec_sha256,
            "authorization_sha256": args.authorization_sha256,
            "supervisor": record,
            "budget_usage": tracker.state(),
            "target_databases_read": spec["databases"],
            "selection_use": "none",
        },
    )
    print(json.dumps({"status": "target_baselines_completed"}, indent=2))
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