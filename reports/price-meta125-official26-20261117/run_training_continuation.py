"""Fixed official26 training sequence, with no retries or target evaluation."""

from __future__ import annotations

import datetime
import json
import os
import sys
import traceback
from pathlib import Path

ROOT = Path("/gly/weilin/price-meta-125-official26-20261117")
BINDING_SHA = "220400d2d1a8797d3124e084caba4f1d7e1baac5ea953274884899c3206f0e17"
ACTIONS = (
    "train:E:42",
    "train:M:42",
    "train:E:43",
    "train:M:43",
    "train:E:44",
    "train:M:44",
    "seal-training",
)


def main():
    sys.path.insert(0, str(ROOT / "code"))
    from spatial_meta_ce.price_meta125_runtime import (
        ALPHA,
        completed,
        cost_gate,
        file_sha,
        load_binding,
        read_json,
        run_stage,
        validate_authorization,
        write_json,
    )

    binding_path = ROOT / "inputs/binding.json"
    binding = load_binding(binding_path)
    if binding["binding_sha256"] != BINDING_SHA or binding["alpha"] != ALPHA:
        raise ValueError("Frozen continuation binding differs")
    cost_gate(binding)
    script_sha = file_sha(Path(__file__))
    authorizations = {}
    for action in ACTIONS:
        path = ROOT / "inputs" / f"authorization-{action.replace(':', '-')}.json"
        value = read_json(path)
        validate_authorization(value, binding, action)
        if value.get("continuation_sha256") != script_sha or value.get("attempts") != 1:
            raise ValueError("Continuation authorization differs")
        authorizations[action] = (path, file_sha(path))
        output = ROOT / "outputs" / action.replace(":", "-")
        if (
            output.exists()
            or output.with_name(output.name + ".supervisor.json").exists()
        ):
            raise FileExistsError("Continuation never resumes or overwrites a stage")
    state = {
        "schema": 1,
        "protocol": 125,
        "status": "running",
        "pid": os.getpid(),
        "binding_sha256": BINDING_SHA,
        "continuation_sha256": script_sha,
        "started_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "actions": list(ACTIONS),
        "completed": [],
        "current_action": None,
        "authorization_sha256": {a: pair[1] for a, pair in authorizations.items()},
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
            record = run_stage(
                binding_path,
                path,
                action,
                "cpu" if action == "seal-training" else "cuda",
            )
            if record["status"] != "completed":
                raise RuntimeError(f"Stage {action} stopped: {record.get('failure')}")
            output, value = completed(binding, action.replace(":", "-"))
            if value.get("target_databases_read") != []:
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
