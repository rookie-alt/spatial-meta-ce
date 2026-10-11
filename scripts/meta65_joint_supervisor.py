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
MODES = ("formal", "cpu_preflight", "gpu_preflight")

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
    "gpu_preflight": {
        "total_seconds": 900.0,
        "phase_seconds": {"preparation": 300.0, "evaluation": 300.0, "certification": 300.0},
        "rss_bytes": 4 * GiB,
        "vram_bytes": 8 * GiB,
        "output_bytes": 1 * GiB,
        "poll_seconds": 1.0,
    },
}


class UnsafeOutputError(ValueError):
    """Raised when a worker output path contains a symlink."""


class GPUQueryError(RuntimeError):
    """Raised when GPU usage cannot be measured safely."""


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
    """Count worker files and supervisor sidecars, rejecting symlink traversal."""
    output = Path(output)
    log_path = Path(log_path)
    record_path, _ = artifact_paths(output)
    _reject_symlinks(output)
    total = 0
    if output.exists():
        for path in output.rglob("*"):
            if path.is_symlink():
                raise UnsafeOutputError(f"symlink is not allowed: {path}")
            if path.is_file():
                total += path.stat().st_size
    for path in (log_path, record_path):
        if path.is_symlink():
            raise UnsafeOutputError(f"symlink is not allowed: {path}")
        if path.exists():
            total += path.stat().st_size
    return total


def _linux_rows(proc_root: Path = Path("/proc")) -> dict[int, tuple[int, int, int, int, str]]:
    """Read Linux process metadata, tolerating PID disappearance races."""
    rows: dict[int, tuple[int, int, int, int, str]] = {}
    page_size = os.sysconf("SC_PAGE_SIZE")
    try:
        entries = list(proc_root.iterdir())
    except OSError as exc:
        raise GPUQueryError("Linux process table cannot be enumerated") from exc
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
    except (OSError, subprocess.SubprocessError) as exc:
        raise GPUQueryError("process table cannot be enumerated") from exc
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


def _peek_exit(child: subprocess.Popen[Any]) -> int | None:
    """Observe a Linux leader without reaping its PID until group cleanup."""
    known = getattr(child, "returncode", None)
    if known is not None:
        return known
    if sys.platform.startswith("linux"):
        try:
            info = os.waitid(os.P_PID, child.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT)
        except ChildProcessError:
            return child.poll()  # A test callback may already have waited.
        if info is None:
            return None
        return int(info.si_status) if info.si_code == os.CLD_EXITED else -int(info.si_status)
    return child.poll()


def _owned_pids(rows: dict[int, tuple[int, int, int, int, str]], pid: int) -> set[int]:
    leader = rows.get(pid)
    if leader is not None and (leader[1] != pid or leader[2] not in (0, pid)):
        raise GPUQueryError("worker no longer belongs to its new process group")
    same_group = {candidate for candidate, row in rows.items() if row[1] == pid}
    if any(rows[candidate][2] not in (0, pid) for candidate in same_group):
        raise GPUQueryError("observed an unrelated process group with the worker PGID")
    owned = same_group
    changed = True
    while changed:
        changed = False
        for candidate, row in rows.items():
            if candidate not in owned and row[0] in owned:
                owned.add(candidate)
                changed = True
    return owned


def _live_owned_descendants(pgid: int, leader_pid: int) -> list[int]:
    rows = _process_rows()
    group = {pid: row for pid, row in rows.items() if row[1] == pgid}
    if any(row[2] not in (0, pgid) for row in group.values()):
        raise GPUQueryError("cannot certify owned process group identity")
    return [pid for pid, row in group.items() if pid != leader_pid and row[4] != "Z"]


def _gpu_bytes(pids: Iterable[int]) -> int:
    owned = {int(pid) for pid in pids}
    if not owned:
        return 0
    try:
        query = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,used_memory",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise GPUQueryError("nvidia-smi compute-apps query failed") from exc
    total = 0
    for line in query.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        try:
            if len(fields) != 2:
                raise ValueError("invalid compute-apps row")
            pid, mib = int(fields[0]), int(fields[1])
            if pid <= 0 or mib < 0:
                raise ValueError("invalid compute-apps values")
        except ValueError as exc:
            raise GPUQueryError("invalid nvidia-smi compute-apps output") from exc
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
    for _ in range(2):
        if pids or _peek_exit(child) is not None:
            break
        time.sleep(0.01)
        rows = _process_rows()
        pids = _owned_pids(rows, int(child.pid))
    if not pids and _peek_exit(child) is None:
        raise GPUQueryError("live worker process tree cannot be observed")
    rss = sum(rows[pid][3] for pid in pids if rows[pid][4] != "Z")
    if rss == 0:
        try:
            self_rss = int(resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss)
            rss = self_rss * (1024 if sys.platform.startswith("linux") else 1)
        except (AttributeError, ValueError):
            pass
    vram = 0
    if mode in ("formal", "gpu_preflight"):
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


def _kill_group(
    child: subprocess.Popen[Any], grace: float, signals: list[int], *, owned_pgid: int,
) -> None:
    if owned_pgid != child.pid or owned_pgid <= 0 or owned_pgid == os.getpgrp():
        raise ValueError("refusing to signal an unowned process group")

    def send(sig: signal.Signals) -> None:
        # After poll() reaps the leader, PGID reuse would make a blind killpg
        # unsafe.  A surviving group must still belong to the original SID.
        if _peek_exit(child) is not None:
            if sys.platform.startswith("linux") and child.returncode is not None:
                raise GPUQueryError("refusing to signal after the leader PID was reaped")
            try:
                rows = _process_rows()
            except GPUQueryError:
                # Linux's unreaped WNOWAIT leader still reserves this PGID;
                # fail closed for the sampler while cleaning only that group.
                if not sys.platform.startswith("linux") or child.returncode is not None:
                    raise
            else:
                remaining = [row for row in rows.values() if row[1] == owned_pgid and row[4] != "Z"]
                if not remaining:
                    return
                if any(row[2] not in (0, owned_pgid) for row in remaining):
                    raise GPUQueryError("refusing to signal a reused process group")
        try:
            os.killpg(owned_pgid, sig)
            signals.append(int(sig))
        except ProcessLookupError:
            pass

    send(signal.SIGTERM)
    deadline = time.monotonic() + grace
    while time.monotonic() < deadline:
        try:
            os.killpg(owned_pgid, 0)
        except ProcessLookupError:
            break
        time.sleep(min(0.02, max(0.0, deadline - time.monotonic())))
    else:
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

    def persist_measured() -> int:
        # The record contains its own byte count; settle the digit count after
        # every write so the reported value includes the final evidence file.
        while True:
            persist()
            measured = output_bytes(output, log_path)
            if measured == record["outputbytes"]:
                return measured
            record["outputbytes"] = measured

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
                if pending_signal is not None:
                    record["failure"] = "parent_signal"
                    _kill_group(child, grace, record["signals"], owned_pgid=record["pgid"])
                    break

                try:
                    current_phase = _read_phase(output, current_phase)
                except (OSError, ValueError, UnsafeOutputError) as exc:
                    record["failure"] = "invalid_phase" if isinstance(exc, ValueError) else "unsafe_output"
                    _kill_group(child, grace, record["signals"], owned_pgid=record["pgid"])
                    break
                if current_phase != record["phase"]:
                    transition_now = time.monotonic()
                    if transition_now - phase_started > budget["phase_seconds"][record["phase"]]:
                        record["failure"] = "phase_timeout"
                        _kill_group(child, grace, record["signals"], owned_pgid=record["pgid"])
                        break
                    record["phase"] = current_phase
                    phase_started = transition_now

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
                    _kill_group(child, grace, record["signals"], owned_pgid=record["pgid"])
                    break
                except GPUQueryError:
                    record["failure"] = "sample_error"
                    _kill_group(child, grace, record["signals"], owned_pgid=record["pgid"])
                    break
                except Exception:  # noqa: BLE001 - sampler boundary must fail closed
                    if _peek_exit(child) is None:
                        record["failure"] = "sample_error"
                        _kill_group(child, grace, record["signals"], owned_pgid=record["pgid"])
                        break
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
                    elif _peek_exit(child) != 0:
                        record["failure"] = "sample_error"
                    break

                child_code = _peek_exit(child)
                observed_now = time.monotonic()
                if observed_now > total_deadline:
                    record["failure"] = "total_timeout"
                    _kill_group(child, grace, record["signals"], owned_pgid=record["pgid"])
                    break
                if observed_now - phase_started > budget["phase_seconds"][current_phase]:
                    record["failure"] = "phase_timeout"
                    _kill_group(child, grace, record["signals"], owned_pgid=record["pgid"])
                    break
                if record["outputbytes"] > budget["output_bytes"]:
                    record["failure"] = "output_limit"
                    _kill_group(child, grace, record["signals"], owned_pgid=record["pgid"])
                    break
                if record["peak_rss_bytes"] > budget["rss_bytes"]:
                    record["failure"] = "rss_limit"
                    _kill_group(child, grace, record["signals"], owned_pgid=record["pgid"])
                    break
                if record["peak_vram_bytes"] > budget["vram_bytes"]:
                    record["failure"] = "vram_limit"
                    _kill_group(child, grace, record["signals"], owned_pgid=record["pgid"])
                    break
                if child_code is not None:
                    break
                time.sleep(budget["poll_seconds"])

            if _peek_exit(child) is None:
                child.wait()
            record["returncode"] = _peek_exit(child)
            record["elapsed"] = time.monotonic() - started
            try:
                final_phase = _read_phase(output, record["phase"])
                if final_phase != record["phase"]:
                    transition_now = time.monotonic()
                    if (record["failure"] is None
                            and transition_now - phase_started > budget["phase_seconds"][record["phase"]]):
                        record["failure"] = "phase_timeout"
                    phase_started = transition_now
                record["phase"] = final_phase
            except (OSError, ValueError, UnsafeOutputError):
                pass
            try:
                record["outputbytes"] = output_bytes(output, log_path)
            except UnsafeOutputError:
                record["failure"] = record["failure"] or "unsafe_output"

            if record["failure"] is None:
                if time.monotonic() > total_deadline:
                    record["failure"] = "total_timeout"
                elif time.monotonic() - phase_started > budget["phase_seconds"][record["phase"]]:
                    record["failure"] = "phase_timeout"
            # Check owned descendants on *all* exit paths while Linux's
            # unreaped leader still reserves its process-group identity.
            try:
                descendants = _live_owned_descendants(record["pgid"], child.pid)
                if descendants:
                    if record["failure"] is None:
                        record["failure"] = ("orphan_descendant" if record["returncode"] == 0
                                             else "child_exit")
                    _kill_group(child, grace, record["signals"], owned_pgid=record["pgid"])
            except GPUQueryError:
                record["failure"] = record["failure"] or "sample_error"
                if child.returncode is None and sys.platform.startswith("linux"):
                    _kill_group(child, grace, record["signals"], owned_pgid=record["pgid"])

            # Only now, after group identity and descendants have been checked,
            # release the leader PID so it cannot be reused during cleanup.
            if child.returncode is None:
                child.wait()

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
                        and (mode == "formal" or status.get("micro_cells") == 9 and not isinstance(status.get("micro_cells"), bool))
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
            final_size = persist_measured()
            if record["status"] in {"completed", "preflight_completed"}:
                if final_size > budget["output_bytes"]:
                    record["failure"] = "output_limit"
                elif time.monotonic() > total_deadline:
                    record["failure"] = "total_timeout"
                elif time.monotonic() - phase_started > budget["phase_seconds"][record["phase"]]:
                    record["failure"] = "phase_timeout"
            if record["failure"] is not None and record["status"] in {"completed", "preflight_completed"}:
                record["status"] = "failed"
                _write_json(output / "status.json", {
                    "status": "failed", "mode": mode, "failure": record["failure"],
                    "supervisor_record": record["supervisor_json"], "formal_result": False,
                })
                record["elapsed"] = time.monotonic() - started
                persist_measured()
    except BaseException as exc:
        if child is not None:
            record["failure"] = record["failure"] or "supervisor_error"
            try:
                if (_peek_exit(child) is None
                        or _live_owned_descendants(record["pgid"], child.pid)):
                    _kill_group(child, grace, record["signals"], owned_pgid=record["pgid"])
            except GPUQueryError:
                if child.returncode is None and sys.platform.startswith("linux"):
                    _kill_group(child, grace, record["signals"], owned_pgid=record["pgid"])
                else:
                    record["cleanup_unverified"] = True
            if child.returncode is None:
                child.wait()
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
