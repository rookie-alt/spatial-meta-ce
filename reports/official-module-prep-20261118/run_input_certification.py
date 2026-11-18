"""Explicit bounded CPU-only official input certification, with no model run."""

import json
import os
import sys
from pathlib import Path

BASE = Path("/gly/weilin/module26-prep-20261118")
CODE = BASE / "code-ready"
OUTPUT = BASE / "inputs-parallel-v3"
sys.path.insert(0, str(CODE))


def main():
    os.environ.update(
        CUDA_VISIBLE_DEVICES="",
        PYTHONDONTWRITEBYTECODE="1",
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
    )
    if sys.platform != "linux":
        raise ValueError("Input certification requires server Linux")
    if "--worker" in sys.argv:
        import resource
        from scripts.prepare_official_module_inputs_parallel import (
            prepare_inputs_parallel,
        )

        resource.setrlimit(resource.RLIMIT_AS, (32 * 1024**3, 32 * 1024**3))
        result = prepare_inputs_parallel(
            "/gly/weilin/official-split-20261117/plans-24k/data_config_official_26.json",
            OUTPUT,
            "/gly/weilin/spatial-meta-ce/runs/mapping-sidecars-source26-20261004",
            True,
            BASE / "evidence/historical_mapping_builder.py.txt",
            workers=4,
        )
        (OUTPUT / "phase.json").write_text(json.dumps({"phase": "certification"}))
        (OUTPUT / "status.json").write_text(
            json.dumps(
                {
                    "status": "worker_complete",
                    "mode": "cpu_preflight",
                    "cells": 0,
                    "micro_cells": 1,
                    "model_predictions_run": False,
                    "training_run": False,
                }
            )
        )
        print(json.dumps({"state": result["state"], "databases": 30}))
        return
    from scripts.meta65_joint_supervisor import supervise

    available = sorted(os.sched_getaffinity(0))
    os.sched_setaffinity(0, available[:4])
    budget = {
        "total_seconds": 5400,
        "phase_seconds": {"preparation": 5100, "evaluation": 1, "certification": 299},
        "rss_bytes": 16 * 1024**3,
        "vram_bytes": 0,
        "output_bytes": 3 * 1024**3,
        "poll_seconds": 5,
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


if __name__ == "__main__":
    main()
