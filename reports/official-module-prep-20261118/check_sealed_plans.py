"""CPU-only check of the three real sealed plans, without model construction."""

import json
import os
import sys
import time
from pathlib import Path

BASE = Path("/gly/weilin/module26-prep-20261118")
CODE = BASE / "code-ready"
PLANS = Path("/gly/weilin/official-split-20261117/plans-24k")
OUTPUT = BASE / "plan-validation"
sys.path.insert(0, str(CODE))


def main():
    os.environ.update(
        CUDA_VISIBLE_DEVICES="",
        PYTHONDONTWRITEBYTECODE="1",
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
    )
    if "--worker" not in sys.argv:
        from scripts.meta65_joint_supervisor import supervise

        os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
        budget = {
            "total_seconds": 1800,
            "phase_seconds": {
                "preparation": 1700,
                "evaluation": 1,
                "certification": 99,
            },
            "rss_bytes": 16 * 1024**3,
            "vram_bytes": 0,
            "output_bytes": 1024**2,
            "poll_seconds": 2,
        }
        result = supervise(
            [sys.executable, "-I", "-B", str(Path(__file__).resolve()), "--worker"],
            OUTPUT,
            mode="cpu_preflight",
            budget=budget,
            root=CODE,
            preflight_micro_cells=1,
        )
        print(json.dumps(result, indent=2))
        if result["status"] != "preflight_completed":
            raise SystemExit(1)
        return
    import resource
    from scripts import run_official_module_experiment as runner
    from spatial_meta_ce.module_experiment import canonical_digest
    from spatial_meta_ce.official_split import build_official_capacity
    from spatial_meta_ce.plans import _store_data_sha256

    resource.setrlimit(resource.RLIMIT_AS, (24 * 1024**3, 24 * 1024**3))
    OUTPUT.mkdir()
    code_before = runner.compute_code_sha256(CODE)
    data_path = PLANS / "data_config_official_26.json"
    capacity_path = PLANS / "capacity_official_26.json"
    data = json.loads(data_path.read_text())
    capacity = json.loads(capacity_path.read_text())
    if canonical_digest(build_official_capacity(data)) != canonical_digest(capacity):
        raise ValueError("Current 26 training cache capacity changed")
    stores = runner._cache_stores(data["cache_root"], "pretrain")
    data_hashes = {db: _store_data_sha256(store) for db, store in stores.items()}
    results = {}
    for seed in runner.SEEDS:
        started = time.monotonic()
        binding = runner.file_binding(
            PLANS / f"training_plan_seed{seed}.json", plan=True
        )
        summary = runner._verify_training_plan(
            binding, seed, capacity, stores, data_hashes
        )
        results[str(seed)] = {
            "binding": binding,
            "summary": summary,
            "seconds": time.monotonic() - started,
        }
    after = runner.compute_code_sha256(CODE)
    if code_before != after:
        raise ValueError("Code changed during sealed plan validation")
    (OUTPUT / "result.json").write_text(
        json.dumps(
            {
                "passed": True,
                "model_run": False,
                "target_data_read": [],
                "data_config": runner.file_binding(data_path),
                "capacity": runner.file_binding(capacity_path),
                "plans": results,
                "code_sha256_before": code_before,
                "code_sha256_after": after,
            },
            indent=2,
        )
        + "\n"
    )
    (OUTPUT / "phase.json").write_text(json.dumps({"phase": "certification"}))
    (OUTPUT / "status.json").write_text(
        json.dumps(
            {
                "status": "worker_complete",
                "mode": "cpu_preflight",
                "cells": 0,
                "micro_cells": 1,
            }
        )
    )


if __name__ == "__main__":
    main()
