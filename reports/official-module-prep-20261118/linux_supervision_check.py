"""Bounded CPU-only Linux checks for the module resource sampler; no model."""

import hashlib
import json
import os
import sys
import time
from pathlib import Path

from scripts import meta65_joint_supervisor as supervisor
from scripts.run_official_module_experiment import module_resource_sample


def main():
    if sys.platform != "linux":
        raise ValueError("This evidence check requires Linux")
    base = Path(sys.argv[1]).resolve()
    if base.exists():
        raise FileExistsError(base)
    base.mkdir()
    os.environ.update(
        CUDA_VISIBLE_DEVICES="",
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        PYTHONDONTWRITEBYTECODE="1",
    )
    # Synthetic counters; actual worker and /proc RSS. No torch/CUDA or cache.
    worker = """
import json,sys,time
from pathlib import Path
output=Path(sys.argv[1]); output.mkdir()
(output/'phase.json').write_text(json.dumps({'phase':'preparation'}))
if sys.argv[2]=='valid':
    for _ in range(12):
        value={'allocated_bytes':20,'reserved_bytes':30,
               'max_allocated_bytes':40,'max_reserved_bytes':50,
               'interval_seconds':0.2}
        temporary=output/'memory.tmp'; temporary.write_text(json.dumps(value))
        temporary.replace(output/'memory.json'); time.sleep(0.2)
else:
    time.sleep(3)
(output/'phase.json').write_text(json.dumps({'phase':'certification'}))
(output/'status.json').write_text(json.dumps({'status':'worker_complete',
    'mode':'cpu_preflight','cells':0,'micro_cells':1}))
"""
    budget = {
        "total_seconds": 15,
        "phase_seconds": {
            "preparation": 10,
            "evaluation": 1,
            "certification": 4,
        },
        "rss_bytes": 256 * 1024**2,
        "vram_bytes": 1024,
        "output_bytes": 1024**2,
        "poll_seconds": 0.1,
    }
    original = supervisor._gpu_bytes
    queries = []

    def forbidden_query(*args):
        queries.append(args)
        raise AssertionError("Global nvidia-smi query must not run")

    supervisor._gpu_bytes = forbidden_query
    records = {}
    try:
        for scenario in ("valid", "missing"):
            output = base / scenario
            started = time.monotonic()
            record = supervisor.supervise(
                [sys.executable, "-I", "-B", "-c", worker, str(output), scenario],
                output,
                mode="cpu_preflight",
                budget=budget,
                grace=1,
                preflight_micro_cells=1,
                sample_fn=lambda child: module_resource_sample(
                    child,
                    output,
                    "cuda",
                    telemetry_required=(output / "memory.json").exists()
                    or time.monotonic() - started > 1.5,
                ),
            )
            records[scenario] = record
    finally:
        supervisor._gpu_bytes = original
    if records["valid"]["status"] != "preflight_completed":
        raise AssertionError(records["valid"])
    if records["valid"]["peak_vram_bytes"] != 50:
        raise AssertionError("Synthetic counters were not measured")
    if records["missing"]["status"] != "failed" or not records["missing"]["failure"]:
        raise AssertionError(records["missing"])
    if queries:
        raise AssertionError("Unexpected global GPU query")
    value = {
        "kind": "linux-real-process-synthetic-cuda-telemetry",
        "passed": True,
        "model_run": False,
        "cuda_used": False,
        "global_gpu_queries": len(queries),
        "records": records,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    (base / "result.json").write_text(json.dumps(value, indent=2) + "\n")
    print(json.dumps({"passed": True, "output": str(base)}))


if __name__ == "__main__":
    main()
