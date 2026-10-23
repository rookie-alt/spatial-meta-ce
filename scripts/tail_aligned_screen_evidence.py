"""Protocol, decision and supervision helpers for the P1 tail-aligned screen.

The success gate is frozen in doc 93 §0: a candidate must improve the source
4+ table Q95 macro by at least 10%, and no per-database bucket may regress
beyond 1.10x the baseline. Target databases never enter this module.

Nothing here executes training on import; `supervise` is the only function
that spawns a process, and it is used by the runner's parent mode.
"""

import datetime
import json
import math
import os
import signal
import subprocess
import sys
from pathlib import Path

VALIDATION = ("chembl", "legalacts", "telstra")
TARGETS = frozenset(("imdb", "stats", "genome", "ergastf1"))
BUCKETS = ("all_tables", "four_plus_tables")
STOP_GRACE = 10


def write_json(path: Path, value: dict) -> None:
    """Atomic write; NaN/Infinity are refused because they cannot round-trip."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def _finite_q95(value, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{label} Q95 must be numeric")
    if not math.isfinite(value) or value < 1:
        raise ValueError(f"{label} Q95 must be finite and at least one")
    return float(value)


def validate_endpoint(endpoint: dict, label: str) -> None:
    if set(endpoint.get("per_database", {})) != set(VALIDATION):
        raise ValueError(
            f"{label} endpoint must cover the three source validation databases"
        )
    _finite_q95(endpoint.get("macro_four_plus_q95"), f"{label} macro")
    for db in VALIDATION:
        for bucket in BUCKETS:
            _finite_q95(
                endpoint["per_database"][db][bucket]["q95"], f"{label} {db}/{bucket}"
            )


def choose_candidates(
    records: dict,
    baseline: str,
    macro_improvement: float = 0.10,
    per_database: float = 1.10,
) -> dict:
    """Apply the frozen doc-93 gate to a {arm: endpoint} mapping."""
    if not isinstance(records, dict) or baseline not in records:
        raise ValueError("Records must be a mapping containing the baseline arm")
    if len(records) < 2:
        raise ValueError("Screening requires at least one non-baseline arm")
    for name, endpoint in records.items():
        validate_endpoint(endpoint, name)
    base = records[baseline]
    base_macro = base["macro_four_plus_q95"]
    candidates = []
    improvement = {}
    for name, endpoint in records.items():
        if name == baseline:
            continue
        improvement[name] = 1 - endpoint["macro_four_plus_q95"] / base_macro
        if endpoint["macro_four_plus_q95"] > (1 - macro_improvement) * base_macro:
            continue
        if all(
            endpoint["per_database"][db][bucket]["q95"]
            <= per_database * base["per_database"][db][bucket]["q95"]
            for db in VALIDATION
            for bucket in BUCKETS
        ):
            candidates.append(name)
    candidates.sort(key=lambda name: (records[name]["macro_four_plus_q95"], name))
    return {
        "decision": "promising_single_seed" if candidates else "no_candidate",
        "candidates": candidates,
        "baseline": baseline,
        "thresholds": {
            "macro_improvement": macro_improvement,
            "per_database": per_database,
        },
        "primary_improvement": {
            name: improvement[name] for name in records if name != baseline
        },
    }


def status(path: Path, state: str, **extra) -> None:
    write_json(
        path / "status.json",
        {
            "status": state,
            "updated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            **extra,
        },
    )


def _log(message: str) -> None:
    print(message, flush=True)


def _terminate(process: subprocess.Popen, grace: int) -> None:
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(process.pid, sig)
        except ProcessLookupError:
            return
        try:
            process.wait(timeout=grace)
            return
        except subprocess.TimeoutExpired:
            continue


def supervise(
    command: list, output: Path, timeout: int = 43200, grace: int = STOP_GRACE
) -> None:
    """Run one arm as a child process with a wall-clock budget.

    A hang is killed and recorded as `failed`; a nonzero child exit is an
    error. Both leave `status.json` readable for a later audit.
    """
    output.mkdir(parents=True, exist_ok=True)
    status(output, "running", command=command, timeout_seconds=timeout)
    process = subprocess.Popen(command, start_new_session=True)
    try:
        code = process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        _terminate(process, grace)
        message = f"arm timed out: exceeded the {timeout}s wall-clock budget"
        status(output, "failed", error=message, command=command, exit_code=None)
        raise RuntimeError(message) from None
    except BaseException:
        _terminate(process, grace)
        raise
    if code != 0:
        status(
            output,
            "failed",
            error=f"child exited with code {code}",
            command=command,
            exit_code=code,
        )
        raise RuntimeError(f"child exited with code {code}")
    status(output, "completed", command=command, exit_code=0)
    _log(f"arm completed: {output}")


def child_command(
    script: Path, arm: str, config: Path, data_config: Path, output: Path, device: str
) -> list:
    return [
        sys.executable,
        str(script),
        "--arm",
        arm,
        "--config",
        str(config),
        "--data-config",
        str(data_config),
        "--output",
        str(output),
        "--device",
        device,
        "--child",
    ]
