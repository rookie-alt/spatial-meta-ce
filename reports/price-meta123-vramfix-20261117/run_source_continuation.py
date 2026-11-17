"""One bounded source-only continuation; no retries or work on import."""

from __future__ import annotations

import datetime
import json
import os
import sys
import traceback
from pathlib import Path

ROOT = Path("/gly/weilin/price-meta-123-vramfix-reselect-20261117")
BINDING_SHA = "5d81667b8ad958c92f15fab4b28dbd5c421e7fe8fe02796846298f200d8934f3"
GPU_INDEX = 2
GPU_UUID = "GPU-cf1ca2b2-761c-66cb-7b36-221d730920c7"
ACTIONS = (
    "train:E:42",
    "train:M:42",
    "train:E:43",
    "train:M:43",
    "train:E:44",
    "train:M:44",
    "seal-training",
    "calibrate-e",
    "report",
)


def main():
    sys.path.insert(0, str(ROOT / "code"))
    from spatial_meta_ce.price_meta123_runtime import (
        _calibration,
        _completed,
        _cost_gate,
        file_sha,
        load_binding,
        read_json,
        run_stage,
        validate_authorization,
        write_json,
    )

    binding_path = ROOT / "inputs/binding.json"
    binding = load_binding(binding_path)
    if binding["binding_sha256"] != BINDING_SHA or _calibration(binding) != 1e-6:
        raise ValueError("Continuation binding or frozen alpha differs")
    _cost_gate(binding, "train:E:42")
    script_sha = file_sha(Path(__file__))
    authorizations = {}
    for action in ACTIONS:
        path = ROOT / "inputs" / f"authorization-{action.replace(':', '-')}.json"
        auth = read_json(path)
        validate_authorization(auth, binding, action)
        if (
            auth.get("actions") != [action]
            or auth.get("attempts") != 1
            or auth.get("continuation_sha256") != script_sha
        ):
            raise ValueError("Continuation authorization changed")
        if action != "seal-training" and (
            auth.get("cuda_device") != GPU_INDEX or auth.get("cuda_uuid") != GPU_UUID
        ):
            raise ValueError("Continuation GPU differs")
        authorizations[action] = (path, file_sha(path))
        output = ROOT / "outputs" / action.replace(":", "-")
        if (
            output.exists()
            or output.with_name(output.name + ".supervisor.json").exists()
        ):
            raise FileExistsError("Continuation never resumes or overwrites a stage")

    state = {
        "schema": 1,
        "status": "running",
        "binding_sha256": BINDING_SHA,
        "continuation_sha256": script_sha,
        "pid": os.getpid(),
        "started_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "actions": list(ACTIONS),
        "authorization_sha256": {a: pair[1] for a, pair in authorizations.items()},
        "completed": [],
        "current_action": None,
        "target_databases_read": [],
    }
    write_json(ROOT / "evidence/continuation-started.json", state)
    try:
        for action in ACTIONS:
            state["current_action"] = action
            write_json(ROOT / "evidence/continuation-status.json", state, replace=True)
            path, expected_sha = authorizations[action]
            if file_sha(path) != expected_sha or file_sha(Path(__file__)) != script_sha:
                raise ValueError("Continuation execution inputs changed")
            device = "cpu" if action == "seal-training" else "cuda"
            record = run_stage(binding_path, path, action, device)
            if record["status"] != "completed":
                raise RuntimeError(f"Stage {action} stopped: {record.get('failure')}")
            output, completion = _completed(binding, action.replace(":", "-"), action)
            if completion.get("target_databases_read") != []:
                raise ValueError("Stage source-only declaration differs")
            state["completed"].append(
                {
                    "action": action,
                    "completion_sha256": file_sha(output / "completion.json"),
                    "elapsed_seconds": record["elapsed_seconds"],
                }
            )
            write_json(ROOT / "evidence/continuation-status.json", state, replace=True)
        state.update(status="completed", current_action=None)
        write_json(ROOT / "evidence/continuation-completed.json", state)
        write_json(ROOT / "evidence/continuation-status.json", state, replace=True)
        print(json.dumps(state), flush=True)
        return 0
    except BaseException as exc:
        state.update(
            status="failed",
            error={"type": type(exc).__name__, "message": str(exc)},
            traceback=traceback.format_exc(),
        )
        write_json(ROOT / "evidence/continuation-failure.json", state)
        write_json(ROOT / "evidence/continuation-status.json", state, replace=True)
        print(json.dumps(state), flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
