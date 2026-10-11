#!/usr/bin/env python3
"""Fail-closed supervision for the independent 60-cell Meta65 scope runner.

Workers write status.json, verification.json (including engineering_verified: true), and
an unpublished decision.provisional.json. Only this parent publishes decisions.
Budget overrides are a synthetic-test API, never a formal CLI option.
"""
from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

if __package__ in (None, ""):  # Direct invocation uses the same module identity.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import meta65_joint_supervisor as low

artifact_paths = low.artifact_paths
output_bytes = low.output_bytes
_write_json = low._write_json
UnsafeOutputError = low.UnsafeOutputError
GPUQueryError = low.GPUQueryError
MODES = low.MODES
CLEANUP_SECONDS = 10.0
Row = tuple[int, int, int, int, str]
Table = dict[int, tuple[Row, str]]


def limits_for(mode: str, override: dict[str, Any] | None = None) -> dict[str, float]:
    if mode not in MODES:
        raise ValueError(f"unsupported mode: {mode}")
    frozen = {"total_seconds": 14400.0 if mode == "formal" else 900.0, "rss_bytes": float(4 * low.GiB),
              "vram_bytes": float(8 * low.GiB), "output_bytes": float(low.GiB),
              "poll_seconds": 1.0}
    if override is not None and not isinstance(override, dict):
        raise ValueError("budget_override must be a mapping")  # noqa: TRY004
    values = override or {}
    if set(values) - set(frozen):
        raise ValueError("unknown budget key")
    checked = {key: low._number(value, name=key, positive=key.endswith("seconds"))
               for key, value in values.items()}
    if any(value > frozen[key] for key, value in checked.items()):
        raise ValueError("test overrides cannot expand frozen budgets")
    return {**frozen, **checked}


def _table() -> Table:
    """Include birth identity, not just reusable PID/PGID numbers."""
    if sys.platform.startswith("linux"):
        rows = low._linux_rows()
        result: Table = {}
        for pid, row in rows.items():
            try:
                text = Path(f"/proc/{pid}/stat").read_text(encoding="ascii")
                fields = text[text.rfind(")") + 2:].split()
                # Re-read ownership and RSS with the same birth identity.
                current: Row = (int(fields[1]), int(fields[2]), int(fields[3]),
                                int(fields[21]) * os.sysconf("SC_PAGE_SIZE"), fields[0])
                result = {**result, pid: (current, fields[19])}
            except (FileNotFoundError, ProcessLookupError):
                continue
            except (OSError, ValueError, IndexError) as exc:
                raise GPUQueryError("process identity unavailable") from exc
        return result
    try:
        query = subprocess.run(["ps", "-axo", "pid=,ppid=,pgid=,rss=,state=,lstart="],
                               check=True, capture_output=True, text=True, timeout=1)
        result: Table = {}
        for line in query.stdout.splitlines():
            fields = line.split(None, 5)
            if len(fields) != 6:
                raise ValueError("incomplete process identity")
            pid, ppid, pgid, rss = (int(value) for value in fields[:4])
            result = {**result, pid: ((ppid, pgid, 0, rss * 1024, fields[4]), fields[5])}
        return result
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        raise GPUQueryError("process table unavailable") from exc


def process_is_live(pid: int) -> bool:
    entry = _table().get(pid)
    return entry is not None and not entry[0][4].startswith("Z")


def _owned(table: Table, child: subprocess.Popen[Any], known: dict[int, str]) -> dict[int, str]:
    leader = table.get(child.pid)
    if leader is not None:
        if leader[0][1] != child.pid or leader[0][2] not in (0, child.pid):
            raise GPUQueryError("worker changed its owned PGID/SID")
        if child.pid in known and leader[1] != known[child.pid]:
            raise GPUQueryError("leader PID identity changed")
    owned = {pid: birth for pid, birth in known.items()
             if pid in table and table[pid][1] == birth}
    # A reaped leader no longer reserves its PGID: do not acquire new group
    # members just because the numerical PGID matches an old one.
    group = low._owned_pids({pid: entry[0] for pid, entry in table.items()}, child.pid) if leader else set()
    owned = {**owned, **{pid: table[pid][1] for pid in group}}
    while True:
        descendants = {pid: birth for pid, (row, birth) in table.items() if row[0] in owned}
        expanded = {**owned, **descendants}
        if expanded == owned:
            return owned
        owned = expanded


def _gpu_bytes(pids: set[int], uuid: str) -> int:
    try:
        query = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,gpu_uuid,used_memory",
             "--format=csv,noheader,nounits"],
            check=True, capture_output=True, text=True, timeout=1,
        )
        total = 0
        for line in query.stdout.splitlines():
            fields = [value.strip() for value in line.split(",")]
            if len(fields) != 3:
                raise ValueError("malformed GPU sample")
            pid = int(fields[0])
            if pid not in pids:
                continue  # Other users' allocations are not this worker's usage.
            memory = int(fields[2])
            if pid <= 0 or memory < 0 or fields[1] != uuid:
                raise ValueError("owned GPU allocation outside binding")
            total += memory * low.MiB
        return total
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        raise GPUQueryError("owned GPU usage unavailable") from exc


def _cleanup(child: subprocess.Popen[Any], known: dict[int, str]) -> tuple[bool, list[int]]:
    """Bounded TERM/KILL; never signal a PID whose birth identity has changed."""
    deadline = time.monotonic() + CLEANUP_SECONDS
    sent: list[int] = []
    killed = False
    while time.monotonic() < deadline:
        try:
            table = _table()
            known = _owned(table, child, known)
            live = {pid: birth for pid, birth in known.items()
                    if pid in table and not table[pid][0][4].startswith("Z")}
            if not live:
                child.wait(timeout=max(0.001, deadline - time.monotonic()))
                return True, sent
            sig = signal.SIGKILL if killed else signal.SIGTERM
            # Verify identity again immediately before each signal; no blind killpg.
            for pid, birth in live.items():
                current = _table().get(pid)
                if current is not None and current[1] == birth:
                    try:
                        os.kill(pid, sig)
                    except ProcessLookupError:
                        continue
                    sent = [*sent, int(sig)]
            killed = True
            time.sleep(min(0.1, max(0, deadline - time.monotonic())))
        except (OSError, GPUQueryError, subprocess.TimeoutExpired):
            # Popen owns the unreaped direct child even if the process table
            # fails. Kill/reap that child, but never certify unknown descendants.
            if child.returncode is None:
                try:
                    child.kill()
                    sent = [*sent, int(signal.SIGKILL)]
                    child.wait(timeout=max(0.001, deadline - time.monotonic()))
                except (OSError, subprocess.TimeoutExpired):
                    return False, sent
            return False, sent
    return False, sent


def _completion(output: Path, mode: str) -> str | None:
    cells = 60 if mode == "formal" else 12
    status = low._json_path(output / "status.json") or {}
    if (status.get("status") != "worker_complete" or status.get("mode") != mode
            or type(status.get("cells")) is not int or status["cells"] != cells):
        return "invalid_worker_status"
    verification = low._json_path(output / "verification.json") or {}
    if (verification.get("status") != ("formal_verified" if mode == "formal" else "preflight_verified")
            or verification.get("mode") != mode
            or type(verification.get("cells_verified")) is not int
            or verification["cells_verified"] != cells or verification.get("engineering_verified") is not True
            or verification.get("formal_result") is not (mode == "formal")):
        return "invalid_verification"
    if (output / "decision.json").exists():
        return "premature_decision"
    if mode == "formal":
        decision = low._json_path(output / "decision.provisional.json") or {}
        summary = verification.get("summary")
        if (decision.get("status") not in {"promising_single_seed_K20", "no_candidate"}
                or not isinstance(summary, dict) or summary.get("status") != decision["status"]
                or summary.get("selected_arm") != decision.get("selected_arm")):
            return "invalid_decision"
    return None


def _revoke(output: Path, record: dict[str, Any]) -> None:
    # A malicious unrelated worker symlink must not stop revocation of the
    # canonical decision name. Rename links themselves; never follow them.
    if any(path.is_symlink() for path in (output, *output.parents)):
        raise UnsafeOutputError("cannot revoke through symlinked output")
    if not output.is_dir():
        return
    decision = output / "decision.json"
    if decision.exists():
        invalid = output / "decision.invalid.json"
        index = 0
        while invalid.exists():
            index += 1
            invalid = output / f"decision.invalid.{index}.json"
        decision.rename(invalid)
    _write_json(output / "status.json", {"status": "failed", "mode": record["mode"],
                "failure": record["failure"], "formal_result": False,
                "supervisor_record": record["supervisor_json"]})


def _persist(record: dict[str, Any], output: Path, log: Path) -> dict[str, Any]:
    for _ in range(5):  # Settle sidecar self-size without an unbounded loop.
        _write_json(Path(record["supervisor_json"]), record)
        measured = output_bytes(output, log)
        if measured == record["outputbytes"]:
            return record
        record = {**record, "outputbytes": measured}
    raise RuntimeError("sidecar byte count did not settle")


def _budget_failure(record: dict[str, Any], budget: dict[str, float], started: float) -> str | None:
    if time.monotonic() - started > budget["total_seconds"]:
        return "total_timeout"
    for field, limit, reason in (("peak_rss_bytes", "rss_bytes", "rss_limit"),
                                 ("peak_vram_bytes", "vram_bytes", "vram_limit"),
                                 ("outputbytes", "output_bytes", "output_limit")):
        if record[field] > budget[limit]:
            return reason
    return None


def supervise(command: list[str], output: Path, *, mode: str = "formal",
              gpu_uuid: str | None = None,
              budget_override: dict[str, Any] | None = None) -> dict[str, Any]:
    argv = low._validate_command(command)
    budget = limits_for(mode, budget_override)
    test_only = budget_override is not None
    if gpu_uuid is not None and (not gpu_uuid.startswith("GPU-") or len(gpu_uuid) != 40
                                 or any(c not in "0123456789abcdefABCDEF-" for c in gpu_uuid[4:])):
        raise ValueError("gpu_uuid must be a full NVIDIA GPU UUID")
    if mode != "cpu_preflight" and gpu_uuid is None and not test_only:
        raise ValueError("GPU mode requires a frozen gpu_uuid")
    if mode == "cpu_preflight" and gpu_uuid is not None:
        raise ValueError("CPU preflight cannot bind a GPU")
    output = Path(os.path.abspath(output))
    record_path, log_path = artifact_paths(output)
    for path in (*output.parents, output, record_path, log_path):
        if path.is_symlink():
            raise UnsafeOutputError(f"symlink is not allowed: {path}")
    if any(path.exists() for path in (output, record_path, log_path)):
        raise FileExistsError("supervisor output must be fresh")
    output.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    record: dict[str, Any] = {
        "schema": 1, "status": "running", "mode": mode, "test_only": test_only,
        "formal_result": False, "command": argv, "output": str(output),
        "gpu_uuid": gpu_uuid, "budget": budget, "pid": None, "pgid": None,
        "returncode": None, "elapsed": 0.0, "peak_rss_bytes": 0, "peak_vram_bytes": 0,
        "outputbytes": 0, "signals": [], "received_signals": [], "failure": None,
        "cleanup_verified": False, "supervisor_json": str(record_path),
    }
    pending: int | None = None
    received: tuple[int, ...] = ()
    old_handlers: dict[int, Any] = {}
    child: subprocess.Popen[Any] | None = None
    known: dict[int, str] = {}

    def receive(signum: int, _frame: Any) -> None:
        nonlocal pending, received
        pending = signum
        received = (*received, signum)

    try:
        for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
            old_handlers = {**old_handlers, sig: signal.signal(sig, receive)}
        with log_path.open("x", encoding="utf-8") as log:
            record = _persist(record, output, log_path)
            env = {**os.environ, "META65_SCOPE_SUPERVISED_MODE": mode,
                   "META65_SCOPE_SUPERVISOR_PID": str(os.getpid()),
                   "CUDA_VISIBLE_DEVICES": gpu_uuid or "", "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
                   "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
                   "VECLIB_MAXIMUM_THREADS": "1"}
            child = subprocess.Popen(argv, env=env, stdout=log, stderr=subprocess.STDOUT,
                                     start_new_session=True)
            record = {**record, "pid": child.pid, "pgid": child.pid}
            while True:
                if pending is not None:
                    record = {**record, "failure": "parent_signal"}
                    break
                try:
                    table = _table()
                    known = _owned(table, child, known)
                    code = low._peek_exit(child)
                    if not known and code is None:
                        raise GPUQueryError("live worker cannot be observed")
                    live = {pid for pid in known if pid in table and not table[pid][0][4].startswith("Z")}
                    rss = sum(table[pid][0][3] for pid in live)
                    vram = _gpu_bytes(live, gpu_uuid) if gpu_uuid is not None else 0
                    record = {**record, "peak_rss_bytes": max(record["peak_rss_bytes"], rss),
                              "peak_vram_bytes": max(record["peak_vram_bytes"], vram),
                              "outputbytes": output_bytes(output, log_path)}
                except UnsafeOutputError:
                    record = {**record, "failure": "unsafe_output"}
                    break
                except (OSError, ValueError, GPUQueryError):
                    record = {**record, "failure": "sample_error"}
                    break
                failure = _budget_failure(record, budget, started)
                if failure is not None:
                    record = {**record, "failure": failure}
                    break
                if code is not None:
                    record = {**record, "returncode": code}
                    if code != 0:
                        record = {**record, "failure": "child_exit"}
                    elif live - {child.pid}:
                        record = {**record, "failure": "orphan_descendant"}
                    break
                time.sleep(min(budget["poll_seconds"], max(0, budget["total_seconds"] -
                                                          (time.monotonic() - started))))
            clean, sent = _cleanup(child, known)
            record = {**record, "cleanup_verified": clean, "signals": sent,
                      "returncode": child.returncode,
                      "failure": record["failure"] or (None if clean else "cleanup_unverified")}
            if record["failure"] is None:
                record = {**record, "failure": _completion(output, mode)}
            if record["failure"] is None:
                record = {**record, "status": "completed" if mode == "formal" else "preflight_completed",
                          "formal_result": mode == "formal" and not test_only}
                _write_json(output / "status.json", {**(low._json_path(output / "status.json") or {}),
                            "status": record["status"], "formal_result": record["formal_result"],
                            "test_only": test_only, "supervisor_record": str(record_path)})
                record = _persist(record, output, log_path)
                record = {**record, "failure": "parent_signal" if pending is not None else
                          _budget_failure(record, budget, started)}
                if record["failure"] is None and mode == "formal":
                    (output / "decision.provisional.json").rename(output / "decision.json")
    except BaseException as exc:  # No exception may leave a published candidate behind.
        record = {**record, "failure": "parent_signal" if isinstance(exc, KeyboardInterrupt)
                  else record["failure"] or "supervisor_error"}
        if child is not None and not record["cleanup_verified"]:
            clean, sent = _cleanup(child, known)
            record = {**record, "cleanup_verified": clean, "signals": [*record["signals"], *sent],
                      "returncode": child.returncode}
    try:
        record = {**record, "elapsed": time.monotonic() - started,
                  "received_signals": list(received)}
        record = _persist(record, output, log_path)
        record = {**record, "failure": record["failure"] or
                  ("parent_signal" if pending is not None else _budget_failure(record, budget, started))}
        if record["failure"] is not None:
            record = {**record, "status": "failed", "formal_result": False}
            _revoke(output, record)
        record = _persist(record, output, log_path)
        # The final record itself is evidence and consumes the same budget.
        # Once failed, failure-evidence writes cannot restore success.
        if record["failure"] is None:
            failure = "parent_signal" if pending is not None else _budget_failure(record, budget, started)
            if failure is not None:
                record = {**record, "failure": failure, "status": "failed", "formal_result": False}
                _revoke(output, record)
                record = _persist(record, output, log_path)
    except (OSError, ValueError, RuntimeError) as exc:
        record = {**record, "status": "failed", "formal_result": False,
                  "failure": record["failure"] or "supervisor_error", "persistence_error": str(exc)}
        try:
            _revoke(output, record)
            record = _persist(record, output, log_path)
        except (OSError, ValueError, RuntimeError) as failure:
            record = {**record, "revocation_error": str(failure)}
    finally:
        for old_sig, handler in old_handlers.items():
            signal.signal(old_sig, handler)
    return record


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=MODES, default="formal")
    parser.add_argument("--gpu-uuid")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("worker command required after --")
    record = supervise(command, args.output, mode=args.mode, gpu_uuid=args.gpu_uuid)
    return 0 if record["status"] in {"completed", "preflight_completed"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
