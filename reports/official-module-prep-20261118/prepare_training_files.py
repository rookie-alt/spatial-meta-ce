"""Prepare unsigned 15-run files after input certification, without training."""

import copy
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

BASE = Path("/gly/weilin/module26-prep-20261118")
CODE = BASE / "code-training"
PLANS = Path("/gly/weilin/official-split-20261117/plans-24k")
PREP = BASE / "training-ready-v2"
sys.path.insert(0, str(CODE))


def main():
    os.environ.update(
        CUDA_VISIBLE_DEVICES="",
        PYTHONDONTWRITEBYTECODE="1",
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
    )
    audit = BASE / "spec-preparation-v2"
    if "--worker" not in sys.argv:
        from scripts.meta65_joint_supervisor import supervise

        os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
        result = supervise(
            [sys.executable, "-I", "-B", str(Path(__file__).resolve()), "--worker"],
            audit,
            mode="cpu_preflight",
            root=CODE,
            preflight_micro_cells=1,
            budget={
                "total_seconds": 1800,
                "phase_seconds": {
                    "preparation": 1700,
                    "evaluation": 1,
                    "certification": 99,
                },
                "rss_bytes": 16 * 1024**3,
                "vram_bytes": 0,
                "output_bytes": 16 * 1024**2,
                "poll_seconds": 2,
            },
        )
        print(json.dumps(result, indent=2))
        if result["status"] != "preflight_completed":
            raise SystemExit(1)
        return
    import resource

    resource.setrlimit(resource.RLIMIT_AS, (24 * 1024**3, 24 * 1024**3))
    audit.mkdir()
    from scripts import run_official_module_experiment as runner

    if PREP.exists():
        raise FileExistsError(PREP)
    report = BASE / "inputs-parallel-v3/official_module_inputs.json"
    if json.loads(report.read_text())["state"] != "completed":
        raise ValueError("Complete real input certification is required")
    budget = json.loads((BASE / "evidence/training-budget-draft.json").read_text())
    started = time.monotonic()
    master = runner.prepare_training(
        data_config=PLANS / "data_config_official_26.json",
        capacity=PLANS / "capacity_official_26.json",
        split_manifest=PLANS / "official_split_manifest.json",
        training_plans={
            str(s): PLANS / f"training_plan_seed{s}.json" for s in runner.SEEDS
        },
        mapping_report=report,
        output_dir=BASE / "outputs/matrix-preparation-only",
        budget=budget,
        cuda_device=1,
    )
    PREP.mkdir()
    records = []
    for arm in runner.ARMS:
        for seed in runner.SEEDS:
            key = runner.run_key(arm, seed)
            header = runner.spec_header(
                "train",
                BASE / "outputs" / key,
                budget,
                cuda_device=1,
                execution={"mode": "single_run", "arm": arm, "seed": seed},
            )
            spec = {
                **header,
                "bindings": copy.deepcopy(master["bindings"]),
                "config_sha256": copy.deepcopy(master["config_sha256"]),
            }
            runner.validate_header(spec)
            if spec["bindings"] != master["bindings"]:
                raise AssertionError("Common verified input bindings changed")
            folder = PREP / key
            folder.mkdir()
            spec_path = folder / "training_spec.json"
            auth_path = folder / "authorization_template.json"
            runner.write_json(spec_path, spec)
            binding = runner.file_binding(spec_path)
            runner.write_json(
                auth_path, runner.authorization_template(spec, binding["sha256"])
            )
            auth_binding = runner.file_binding(auth_path)
            args = SimpleNamespace(
                spec=str(spec_path),
                spec_sha256=binding["sha256"],
                authorization=str(auth_path),
                authorization_sha256=auth_binding["sha256"],
                execute=True,
            )
            try:
                runner.parent_main(args)
            except ValueError as exc:
                if "unsigned" not in str(exc):
                    raise
                rejected = str(exc)
            else:
                raise AssertionError("Unsigned authorization must reject execution")
            if Path(spec["output_dir"]).exists():
                raise AssertionError("Rejected execution created output")
            records.append(
                {
                    "key": key,
                    "spec": binding,
                    "authorization_template": auth_binding,
                    "output_dir": spec["output_dir"],
                    "unsigned_execute_rejected": rejected,
                }
            )
    first = records[0]
    args = SimpleNamespace(
        spec=first["spec"]["path"],
        spec_sha256=first["spec"]["sha256"],
        authorization=first["authorization_template"]["path"],
        authorization_sha256=first["authorization_template"]["sha256"],
        execute=False,
    )
    runner.parent_main(args)
    result = {
        "prepared": True,
        "training_run": False,
        "model_predictions_run": False,
        "authorized": False,
        "target_databases_read_for_training": [],
        "runs": records,
        "common_bindings": master["bindings"],
        "common_config_sha256": master["config_sha256"],
        "budget_per_run": budget,
        "gpu_allocated_seconds_total_upper_bound": 15 * budget["gpu_allocated_seconds"],
        "seconds": time.monotonic() - started,
        "validation": "Full common matrix inputs validated; each single-run header checked; first spec full dry-run; all 15 unsigned execution attempts rejected before output creation. Actual run revalidates its spec.",
    }
    runner.write_json(PREP / "readiness.json", result)
    runner.write_json(audit / "phase.json", {"phase": "certification"})
    runner.write_json(
        audit / "status.json",
        {
            "status": "worker_complete",
            "mode": "cpu_preflight",
            "cells": 0,
            "micro_cells": 1,
            "training_run": False,
            "model_predictions_run": False,
        },
    )
    print(json.dumps({"prepared": True, "runs": len(records), "authorized": False}))


if __name__ == "__main__":
    main()
