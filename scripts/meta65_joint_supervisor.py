#!/usr/bin/env python3
"""Independent standard-library supervisor for the Meta65 joint runner.

This module deliberately has no dependency on the project runner.  It owns one
new session/process group, samples only that group and its descendants, and
writes evidence beside the worker output rather than inside it.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import resource
import signal
import subprocess
import sys
import time
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

GiB = 1024**3
MiB = 1024**2
PHASES = ("preparation", "evaluation", "certification")
MODES = ("formal", "cpu_preflight")

_DEFAULTS = {
    "formal": {
        "total_seconds": 4800.0,
        "phase_seconds": {"preparation": 300.0, "evaluation": 3600.0, "certification": 900.0},
        "rss_bytes": 4 * GiB,
        "vram_bytes": 8 * GiB,
        "output_bytes": 1 * GiB,
        "poll_seconds": 1.0,
    },
    "cpu_preflight": {
        "total_seconds": 900.0,
        "phase_seconds": {"preparation": 300.0, "evaluation": 120.0, "certification": 300.0},
        "rss_bytes": 4 * GiB,
        "vram_bytes": 0,
        "output_bytes": 1 * GiB,
        "poll_seconds": 1.0,
    },
}


class UnsafeOutputError(ValueError):
    """Raised when a worker output path contains a symlink."""


def artifact_paths(output: str | os.PathLike[str]) -> tuple[Path, Path]:
    """Return the two sidecar paths without creating the worker output."""
    output = Path(output)
    return (
        output.with_name(output.name + ".supervisor.json"),
        output.with_name(output.name + ".supervisor.log"),
    )


def _number(value: Any, *, name: str, positive: bool = True) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number")  # noqa: TRY004
    value = float(value)
    if not math.isfinite(value) or (positive and value <= 0) or (not positive and value < 0):
        raise ValueError(f"{name} must be finite and {'positive' if positive else 'non-negative'}")
    return value


def limits_for(mode: str, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return frozen limits, allowing only smaller values for test helpers."""
    if mode not in MODES:
        raise ValueError(f"unsupported mode: {mode}")
    if overrides is None:
        overrides = {}
    if not isinstance(overrides, dict):
        raise ValueError("limits must be a mapping")  # noqa: TRY004
    unknown = set(overrides) - set(_DEFAULTS[mode])
    if unknown:
        raise ValueError(f"unknown limit(s): {sorted(unknown)}")

    result = {
        key: (value.copy() if isinstance(value, dict) else value)
        for key, value in _DEFAULTS[mode].items()
    }
    for key, value in overrides.items():
        if key == "phase_seconds":
            if not isinstance(value, dict) or set(value) - set(PHASES):
                raise ValueError("phase_seconds must contain known phases only")
            for phase, phase_limit in value.items():
                limit = _number(phase_limit, name=f"phase_seconds.{phase}")
                if limit > result[key][phase]:
                    raise ValueError("test limits cannot exceed the frozen budget")
                result[key][phase] = limit
        else:
            positive = key not in {"rss_bytes", "vram_bytes", "output_bytes"}
            limit = _number(value, name=key, positive=positive)
            if limit > result[key]:
                raise ValueError("test limits cannot exceed the frozen budget")
            result[key] = limit
    return result


def _json_path(path: Path) -> dict[str, Any] | None:
    try:
        if path.is_symlink() or not path.is_file():
            return None
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _reject_symlinks(path: Path) -> None:
    if path.is_symlink():
        raise UnsafeOutputError(f"symlink is not allowed: {path}")
    if not path.exists():
        return
    if path.is_dir():
        for child in path.iterdir():
            _reject_symlinks(child)


def output_bytes(output: str | os.PathLike[str], log_path: str | os.PathLike[str]) -> int:
    """Count worker files and the bypass log, rejecting symlink traversal."""
    output = Path(output)
    log_path = Path(log_path)
    _reject_symlinks(output)
    if log_path.is_symlink():
        raise UnsafeOutputError(f"symlink is not allowed: {log_path}")
    total = 0
    if output.exists():
        for path in output.rglob("*"):
            if path.is_symlink():
                raise UnsafeOutputError(f"symlink is not allowed: {path}")
            if path.is_file():
                total += path.stat().st_size
    if log_path.exists():
        total += log_path.stat().st_size
    return total


def _linux_rows(proc_root: Path = Path("/proc")) -> dict[int, tuple[int, int, int, int, str]]:
    """Read Linux process metadata, tolerating PID disappearance races."""
    rows: dict[int, tuple[int, int, int, int, str]] = {}
    page_size = os.sysconf("SC_PAGE_SIZE")
    try:
        entries = list(proc_root.iterdir())
    except OSError:
        return rows
    for entry in entries:
        if not entry.name.isdigit():
            continue
        try:
            text = (entry / "stat").read_text(encoding="ascii")
            closing = text.rfind(")")
            if closing < 0:
                continue
            fields = text[closing + 2 :].split()
            if len(fields) < 22:
                continue
            pid = int(entry.name)
            state = fields[0]
            ppid, pgrp, sid = (int(fields[index]) for index in (1, 2, 3))
            rss = int(fields[21]) * page_size
            rows[pid] = (ppid, pgrp, sid, rss, state)
        except (OSError, UnicodeError, ValueError, IndexError):
            continue
    return rows


def _ps_rows() -> dict[int, tuple[int, int, int, int, str]]:
    try:
        result = subprocess.run(
            ["ps", "-axo", "pid=,ppid=,pgid=,rss=,state="],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.SubprocessError):
        return {}
    rows: dict[int, tuple[int, int, int, int, str]] = {}
    for line in result.stdout.splitlines():
        fields = line.split()
        if len(fields) < 5:
            continue
        try:
            pid, ppid, pgrp, rss_kib = (int(item) for item in fields[:4])
        except ValueError:
            continue
        rows[pid] = (ppid, pgrp, 0, rss_kib * 1024, fields[4])
    return rows


def _process_rows() -> dict[int, tuple[int, int, int, int, str]]:
    return _linux_rows() if sys.platform.startswith("linux") else _ps_rows()


def _owned_pids(rows: dict[int, tuple[int, int, int, int, str]], pid: int) -> set[int]:
    if pid not in rows:
        return set()
    group = rows[pid][1]
    owned = {candidate for candidate, row in rows.items() if row[1] == group}
    owned.add(pid)
    changed = True
    while changed:
        changed = False
        for candidate, row in rows.items():
            if candidate not in owned and row[0] in owned:
                owned.add(candidate)
                changed = True
    return owned


def _gpu_bytes(pids: Iterable[int]) -> int:
    owned = {int(pid) for pid in pids}
    if not owned:
        return 0
    query = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,used_memory",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    total = 0
    for line in query.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) < 2:
            continue
        try:
            pid, mib = int(fields[0]), int(fields[1])
        except ValueError:
            continue
        if pid in owned:
            total += mib * MiB
    return total


def _worker_reported_vram(output: Path) -> int:
    status = _json_path(output / "status.json") or {}
    values = [status.get("allocated_vram_bytes", 0), status.get("reserved_vram_bytes", 0)]
    memory = status.get("memory")
    if isinstance(memory, dict):
        values.extend((memory.get("allocated_bytes", 0), memory.get("reserved_bytes", 0)))
    valid = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
            raise ValueError("worker VRAM report must be non-negative bytes")
        valid.append(int(value))
    return max(valid, default=0)


def sample_process(child: Any, *, mode: str, output: Path | None = None) -> dict[str, int]:
    """Sample only the child group/tree; CPU preflight never calls nvidia-smi."""
    rows = _process_rows()
    pids = _owned_pids(rows, int(child.pid))
    rss = sum(rows[pid][3] for pid in pids if rows[pid][4] != "Z")
    if rss == 0:
        try:
            self_rss = int(resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss)
            rss = self_rss * (1024 if sys.platform.startswith("linux") else 1)
        except (AttributeError, ValueError):
            pass
    vram = 0
    if mode == "formal":
        vram = _gpu_bytes(pids)
        if output is not None:
            vram = max(vram, _worker_reported_vram(output))
    return {"rss_bytes": int(rss), "vram_bytes": int(vram)}


def process_is_live(pid: int) -> bool:
    try:
        rows = _process_rows()
        row = rows.get(int(pid))
        if row is not None:
            return row[4] != "Z"
        os.kill(int(pid), 0)
    except (OSError, ValueError):
        return False
    return True


def _write_json(path: Path, value: dict[str, Any], *, replace: bool = True) -> None:
    if path.is_symlink():
        raise UnsafeOutputError(f"symlink is not allowed: {path}")
    if not replace and path.exists():
        raise FileExistsError(path)
    temporary = path.with_name(path.name + ".tmp")
    if temporary.is_symlink():
        raise UnsafeOutputError(f"symlink is not allowed: {temporary}")
    if temporary.exists():
        if not replace:
            raise FileExistsError(temporary)
        temporary.unlink()
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, sort_keys=True, indent=2)
        handle.write("\n")
    os.replace(temporary, path)


def _read_phase(output: Path, current: str) -> str:
    path = output / "phase.json"
    if not path.exists():
        return current
    value = _json_path(path)
    if value is None or value.get("phase") not in PHASES:
        raise ValueError("invalid phase")
    phase = value["phase"]
    if PHASES.index(phase) < PHASES.index(current):
        raise ValueError("phase regression")
    return phase


def _status_failed(output: Path, record: dict[str, Any]) -> None:
    if output.is_symlink():
        return
    if not output.exists():
        # The worker may fail before creating output.  Never fabricate a
        # completed marker or create the directory, but do keep the failure in
        # the sidecar record so the caller can distinguish this case.
        return
    if not output.is_dir():
        return
    status_path = output / "status.json"
    if status_path.is_symlink():
        return
    temporary = status_path.with_suffix(status_path.suffix + ".supervisor.tmp")
    if temporary.is_symlink() or status_path.parent.is_symlink():
        return
    status = _json_path(status_path) or {}
    status.update(status="failed", supervisor_record=record.get("supervisor_json"))
    _write_json(status_path, status)


def _kill_group(child: subprocess.Popen[Any], grace: float, signals: list[int]) -> None:
    def send(sig: signal.Signals) -> None:
        try:
            os.killpg(child.pid, sig)
            signals.append(int(sig))
        except ProcessLookupError:
            pass

    send(signal.SIGTERM)
    deadline = time.monotonic() + grace
    while child.poll() is None and time.monotonic() < deadline:
        time.sleep(min(0.02, max(0.0, deadline - time.monotonic())))
    if child.poll() is None:
        send(signal.SIGKILL)
    child.wait()


def _validate_command(command: Iterable[str]) -> list[str]:
    if isinstance(command, (str, bytes)):
        raise ValueError("command must be argv, not a shell string")  # noqa: TRY004
    argv = list(command)
    if not argv or any(not isinstance(arg, str) or not arg for arg in argv):
        raise ValueError("command must be a non-empty argv")
    return argv


def _sample_value(value: dict[str, Any]) -> tuple[int, int]:
    if not isinstance(value, dict):
        raise ValueError("sample_fn must return a mapping")  # noqa: TRY004
    rss = value.get("rss_bytes")
    vram = value.get("vram_bytes")
    if any(isinstance(item, bool) or not isinstance(item, int) or item < 0 for item in (rss, vram)):
        raise ValueError("sample values must be non-negative integers")
    return rss, vram


def supervise(
    command: Iterable[str],
    output: str | os.PathLike[str],
    *,
    mode: str = "cpu_preflight",
    limits: dict[str, Any] | None = None,
    sample_fn: Callable[[subprocess.Popen[Any]], dict[str, int]] | None = None,
    grace: float = 10,
    root: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    """Run and supervise one worker, returning and persisting a record dict."""
    argv = _validate_command(command)
    if mode not in MODES:
        raise ValueError(f"unsupported mode: {mode}")
    budget = limits_for(mode, limits)
    grace = _number(grace, name="grace")
    output = Path(output)
    root_path = Path(root) if root is not None else None
    record_path, log_path = artifact_paths(output)
    for path in (output, record_path, log_path):
        if path.is_symlink():
            raise UnsafeOutputError(f"symlink is not allowed: {path}")
        if path.exists():
            raise FileExistsError(f"supervisor path already exists: {path}")
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.parent.is_symlink():
        raise UnsafeOutputError(f"symlink is not allowed: {output.parent}")

    expected_cells = 135 if mode == "formal" else 0
    started = time.monotonic()
    record: dict[str, Any] = {
        "schema": 1,
        "status": "running",
        "mode": mode,
        "command": argv,
        "output": str(output),
        "pid": None,
        "pgid": None,
        "returncode": None,
        "elapsed": 0.0,
        "peak_rss_bytes": 0,
        "peak_vram_bytes": 0,
        "outputbytes": 0,
        "phase": "preparation",
        "signals": [],
        "received_signals": [],
        "failure": None,
        "supervisor_json": str(record_path),
    }
    child: subprocess.Popen[Any] | None = None
    old_handlers: dict[int, Any] = {}
    pending_signal: int | None = None

    def receive(signum: int, _frame: Any) -> None:
        nonlocal pending_signal
        pending_signal = signum
        record["received_signals"].append(int(signum))

    def persist() -> None:
        _write_json(record_path, record)

    try:
        with log_path.open("x", encoding="utf-8") as log:
            persist()
            for sig in (signal.SIGTERM, signal.SIGINT):
                old_handlers[sig] = signal.signal(sig, receive)
            env = os.environ.copy()
            env.update(META65_SUPERVISED_MODE=mode, META65_SUPERVISOR_PID=str(os.getpid()))
            if mode == "cpu_preflight":
                env.update(CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
                           OPENBLAS_NUM_THREADS="1", VECLIB_MAXIMUM_THREADS="1")
            child = subprocess.Popen(
                argv,
                cwd=str(root_path) if root_path is not None else None,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            record.update(pid=child.pid, pgid=child.pid)
            persist()
            current_phase = "preparation"
            phase_started = started
            total_deadline = started + budget["total_seconds"]
            while True:
                now = time.monotonic()
                if pending_signal is not None:
                    record["failure"] = "parent_signal"
                    _kill_group(child, grace, record["signals"])
                    break

                try:
                    current_phase = _read_phase(output, current_phase)
                except (OSError, ValueError, UnsafeOutputError) as exc:
                    record["failure"] = "invalid_phase" if isinstance(exc, ValueError) else "unsafe_output"
                    _kill_group(child, grace, record["signals"])
                    break
                if current_phase != record["phase"]:
                    record["phase"] = current_phase
                    phase_started = now

                try:
                    if sample_fn is None:
                        sample = sample_process(child, mode=mode, output=output)
                    else:
                        sample = sample_fn(child)
                    rss, vram = _sample_value(sample)
                    record["peak_rss_bytes"] = max(record["peak_rss_bytes"], rss)
                    record["peak_vram_bytes"] = max(record["peak_vram_bytes"], vram)
                    record["outputbytes"] = output_bytes(output, log_path)
                    _reject_symlinks(output)
                except UnsafeOutputError:
                    record["failure"] = "unsafe_output"
                    _kill_group(child, grace, record["signals"])
                    break
                except Exception:  # noqa: BLE001 - sampler boundary must fail closed
                    if child.poll() is None:
                        record["failure"] = "sample_error"
                        _kill_group(child, grace, record["signals"])
                        break
                    child.wait()
                    # Re-read the final phase before deciding whether a late
                    # exit crossed the external deadline.
                    try:
                        current_phase = _read_phase(output, current_phase)
                        record["phase"] = current_phase
                    except (OSError, ValueError, UnsafeOutputError):
                        pass
                    final_now = time.monotonic()
                    if final_now > total_deadline:
                        record["failure"] = "total_timeout"
                    elif child.returncode not in (0, None):
                        record["failure"] = "sample_error"
                    break

                child_code = child.poll()
                if now > total_deadline:
                    record["failure"] = "total_timeout"
                    _kill_group(child, grace, record["signals"])
                    break
                if now - phase_started > budget["phase_seconds"][current_phase]:
                    record["failure"] = "phase_timeout"
                    _kill_group(child, grace, record["signals"])
                    break
                if record["outputbytes"] > budget["output_bytes"]:
                    record["failure"] = "output_limit"
                    _kill_group(child, grace, record["signals"])
                    break
                if record["peak_rss_bytes"] > budget["rss_bytes"]:
                    record["failure"] = "rss_limit"
                    _kill_group(child, grace, record["signals"])
                    break
                if record["peak_vram_bytes"] > budget["vram_bytes"]:
                    record["failure"] = "vram_limit"
                    _kill_group(child, grace, record["signals"])
                    break
                if child_code is not None:
                    child.wait()
                    break
                time.sleep(budget["poll_seconds"])

            if child.poll() is None:
                child.wait()
            record["returncode"] = child.returncode
            record["elapsed"] = time.monotonic() - started
            try:
                record["phase"] = _read_phase(output, record["phase"])
            except (OSError, ValueError, UnsafeOutputError):
                pass
            try:
                record["outputbytes"] = output_bytes(output, log_path)
            except UnsafeOutputError:
                record["failure"] = record["failure"] or "unsafe_output"

            if record["failure"] is None and record["returncode"] == 0:
                try:
                    phase = _read_phase(output, record["phase"])
                    status = _json_path(output / "status.json")
                    valid = (
                        phase == "certification"
                        and status is not None
                        and status.get("status") == "worker_complete"
                        and status.get("mode") == mode
                        and status.get("cells") == expected_cells and not isinstance(status.get("cells"), bool)
                        and (mode != "cpu_preflight" or status.get("micro_cells") == 9 and not isinstance(status.get("micro_cells"), bool))
                    )
                except (OSError, ValueError, UnsafeOutputError):
                    valid = False
                if valid:
                    record["status"] = "completed" if mode == "formal" else "preflight_completed"
                    final_status = _json_path(output / "status.json") or {}
                    final_status["status"] = "completed" if mode == "formal" else "preflight_completed"
                    _write_json(output / "status.json", final_status)
                    try:
                        record["outputbytes"] = output_bytes(output, log_path)
                    except UnsafeOutputError:
                        record["failure"] = "unsafe_output"
                else:
                    record["failure"] = "invalid_worker_status"
            elif record["failure"] is None:
                record["failure"] = "child_exit"
            if record["failure"] is not None:
                record["status"] = "failed"
                if output.is_dir() and not (output / "status.json").is_symlink():
                    _write_json(output / "status.json", {
                        "status": "failed", "mode": mode, "failure": record["failure"],
                        "supervisor_record": record.get("supervisor_json"), "formal_result": False,
                    })
                _status_failed(output, record)
            persist()
    except BaseException as exc:
        if child is not None and child.poll() is None:
            record["failure"] = record["failure"] or "supervisor_error"
            _kill_group(child, grace, record["signals"])
            record["returncode"] = child.returncode
        if isinstance(exc, KeyboardInterrupt):
            record["failure"] = "parent_signal"
            record["status"] = "failed"
        elif record["failure"] is None:
            record["failure"] = "supervisor_error"
            record["status"] = "failed"
        if record["failure"] is not None:
            record["status"] = "failed"
            try:
                _status_failed(output, record)
            except (OSError, ValueError, UnsafeOutputError):
                pass
        record["elapsed"] = time.monotonic() - started
        if log_path.exists() and not log_path.is_symlink():
            try:
                record["outputbytes"] = output_bytes(output, log_path)
            except (OSError, UnsafeOutputError):
                pass
        try:
            persist()
        except (OSError, UnsafeOutputError):
            pass
        if isinstance(exc, (FileExistsError, UnsafeOutputError, ValueError)) and child is None:
            raise
    finally:
        for sig, handler in old_handlers.items():
            signal.signal(sig, handler)
    return record


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--mode", choices=MODES, default="formal")
    parser.add_argument("--root", default=None)
    parser.add_argument("--grace", type=float, default=10.0)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    command = args.command
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        parser.error("a command is required after --")
    record = supervise(command, args.output, mode=args.mode, grace=args.grace, root=args.root)
    return 0 if record["status"] in {"completed", "preflight_completed"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
