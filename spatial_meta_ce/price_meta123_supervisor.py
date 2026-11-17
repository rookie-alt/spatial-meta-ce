"""Independent standard-library supervisor for the PRICE meta-learning runs.

The supervisor owns exactly one worker process group.  It starts the worker in
a new session, samples only that group and its verifiable descendants, and
records wall clock, resident memory, device memory and output volume.  Any
budget breach, monitoring error or parent signal terminates the owned group
and leaves evidence beside the worker output.

The module deliberately depends on the standard library only.  Actual
supervision requires Linux; every other platform is rejected instead of being
approximated.
"""

from __future__ import annotations

import json
import math
import os
import signal
import subprocess
import sys
import threading
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SUPERVISED_ENV = "PRICE_META123_SUPERVISED"
SUPERVISOR_PID_ENV = "PRICE_META123_SUPERVISOR_PID"

COMPLETION_PROTOCOL = 123
REQUIRED_COMPLETION_FIELDS = ("status", "protocol", "binding_sha256")

_STALE_ENV_PREFIXES = ("META65_", "PRICE_META123_")
_ALLOWED_ENV_KEYS = frozenset(
    {
        "CUDA_VISIBLE_DEVICES",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "PYTHONDONTWRITEBYTECODE",
    }
)
_BUDGET_FIELDS = (
    "max_wall_seconds",
    "max_rss_bytes",
    "max_vram_bytes",
    "max_output_bytes",
    "poll_seconds",
)
_MIB = 1024**2
_KILL_GRACE_SECONDS = 5.0
_LEADER_WAIT_SECONDS = 5.0
_IDLE_VRAM_BYTES = 16 * _MIB
_IDLE_UTILIZATION_PERCENT = 0


class SupervisorError(RuntimeError):
    """Base class for supervisor failures."""


class BudgetError(SupervisorError, ValueError):
    """Raised when the caller budget does not match the frozen schema."""


class UnsafeOutputError(SupervisorError, ValueError):
    """Raised when an output path would escape the supervised workspace."""


class MonitoringError(SupervisorError):
    """Raised when resource monitoring cannot be performed reliably."""


class PidReuseError(MonitoringError):
    """Raised when a sampled pid now refers to a different process."""


class UnsupportedPlatformError(SupervisorError):
    """Raised when supervision is requested on a platform without /proc."""


class GpuNotIdleError(SupervisorError):
    """Raised when the authorized GPU is busy before the worker is started."""


@dataclass(frozen=True)
class _ProcRow:
    """Minimal /proc view of one process."""

    ppid: int
    pgid: int
    sid: int
    rss_bytes: int
    state: str
    start_time: str = ""


def _is_linux() -> bool:
    return sys.platform.startswith("linux")


def _validate_budget(budget: Any) -> dict[str, float]:
    """Validate the frozen budget schema; every field is required."""
    if not isinstance(budget, dict):
        raise BudgetError("budget must be a mapping")
    missing = sorted(set(_BUDGET_FIELDS) - set(budget))
    unknown = sorted(set(budget) - set(_BUDGET_FIELDS))
    if missing or unknown:
        raise BudgetError(
            f"budget schema mismatch (missing={missing}, unknown={unknown})"
        )
    limits: dict[str, float] = {}
    for field in _BUDGET_FIELDS:
        value = budget[field]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise BudgetError(f"{field} must be a number")
        number = float(value)
        if not math.isfinite(number):
            raise BudgetError(f"{field} must be finite")
        if field == "max_vram_bytes":
            if number < 0:
                raise BudgetError(
                    "max_vram_bytes must be non-negative (0 means CPU only)"
                )
        elif number <= 0:
            raise BudgetError(f"{field} must be positive")
        limits[field] = number
    return limits


def _validate_command(command: Any) -> list[str]:
    """Accept an argv list only; the worker never goes through a shell."""
    if isinstance(command, (str, bytes, bytearray)):
        raise ValueError("command must be an argv list, not a shell string")
    if not isinstance(command, (list, tuple)):
        raise ValueError("command must be a list of arguments")
    argv = list(command)
    if not argv:
        raise ValueError("command must not be empty")
    for argument in argv:
        if not isinstance(argument, str):
            raise ValueError("command arguments must be strings")
        if not argument:
            raise ValueError("command arguments must not be empty")
        if "\0" in argument:
            raise ValueError("command arguments must not contain NUL")
    return argv


def _sidecar_paths(output: Path) -> tuple[Path, Path]:
    """Return the supervisor record and log paths beside the worker output."""
    return (
        output.with_name(output.name + ".supervisor.json"),
        output.with_name(output.name + ".supervisor.log"),
    )


def _validate_root(root: Path) -> Path:
    """Require an existing absolute directory that can host a supervised run."""
    if not root.is_absolute():
        raise ValueError("cwd must be an absolute path")
    if not root.exists():
        raise ValueError(f"cwd does not exist: {root}")
    if not root.is_dir():
        raise ValueError(f"cwd is not a directory: {root}")
    return root


def _validate_output(output: Path, *, boundary: Path | None = None) -> Path:
    """Require a fresh output directory that stays inside the supervised root.

    Only the chain between ``output`` and ``boundary`` is checked for symlinks;
    system prefixes above the boundary are outside the supervised workspace.
    """
    if not output.is_absolute():
        raise ValueError("output must be an absolute path")
    if any(part == ".." for part in output.parts):
        raise UnsafeOutputError(f"output must not contain '..': {output}")
    if output.is_symlink():
        raise UnsafeOutputError(f"symlink is not allowed: {output}")
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    if output.parent.is_symlink():
        raise UnsafeOutputError(f"symlink is not allowed: {output.parent}")
    if boundary is None:
        return output
    if not boundary.is_absolute():
        raise ValueError("boundary must be an absolute path")
    try:
        output.relative_to(boundary)
    except ValueError as exc:
        raise UnsafeOutputError(
            f"output escapes the supervised root: {output}"
        ) from exc
    current = output.parent
    while True:
        if current.is_symlink():
            raise UnsafeOutputError(f"symlink is not allowed: {current}")
        if current == boundary:
            break
        if current == current.parent:
            raise UnsafeOutputError(f"output escapes the supervised root: {output}")
        current = current.parent
    resolved_boundary = boundary.resolve()
    resolved_output = Path(os.path.normpath(str(output))).resolve()
    if (
        resolved_output != resolved_boundary
        and resolved_boundary not in resolved_output.parents
    ):
        raise UnsafeOutputError(f"output escapes the supervised root: {output}")
    return output


def _build_child_env(env: dict[str, str] | None) -> dict[str, str]:
    """Inherit the environment, drop stale supervision fields, inject context."""
    if env is not None:
        if not isinstance(env, dict):
            raise ValueError("env must be a mapping")
        for key, value in env.items():
            if not isinstance(key, str) or not isinstance(value, str):
                raise ValueError("env keys and values must be strings")
            if key not in _ALLOWED_ENV_KEYS:
                raise ValueError(f"env key is not allowed: {key}")
    child_env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(_STALE_ENV_PREFIXES)
    }
    if env:
        child_env.update(env)
    child_env[SUPERVISED_ENV] = "1"
    child_env[SUPERVISOR_PID_ENV] = str(os.getpid())
    return child_env


def _read_proc_rows(proc_root: Path = Path("/proc")) -> dict[int, _ProcRow]:
    """Read Linux process metadata, tolerating PID disappearance races."""
    page_size = os.sysconf("SC_PAGE_SIZE")
    try:
        entries = list(proc_root.iterdir())
    except OSError as exc:
        raise MonitoringError(f"process table cannot be enumerated: {exc}") from exc
    rows: dict[int, _ProcRow] = {}
    for entry in entries:
        if not entry.name.isdigit():
            continue
        try:
            text = (entry / "stat").read_text(encoding="ascii")
        except (OSError, UnicodeError):
            continue
        closing = text.rfind(")")
        if closing < 0:
            continue
        fields = text[closing + 1 :].split()
        if len(fields) < 22:
            continue
        try:
            state = fields[0]
            ppid = int(fields[1])
            pgid = int(fields[2])
            sid = int(fields[3])
            rss_bytes = int(fields[21]) * page_size
            start_time = fields[19]
        except (ValueError, IndexError):
            continue
        rows[int(entry.name)] = _ProcRow(ppid, pgid, sid, rss_bytes, state, start_time)
    return rows


def _pid_identity_matches(pid: int, start_time: str, identity: dict[int, str]) -> bool:
    """Report whether a pid still refers to the process that was sampled."""
    if not start_time:
        return False
    return identity.get(int(pid)) == start_time


def _owned_pids(
    rows: dict[int, _ProcRow],
    pid: int,
    identity: dict[int, str] | None = None,
) -> set[int]:
    """Return the worker group plus descendants, including ones that left it.

    A descendant that called ``setsid`` leaves the process group but stays a
    child of the worker, so it is still owned and must be cleaned up.  When an
    identity map is given, a pid whose start time changed is a recycled pid
    and is ignored instead of being trusted or signalled.  Descendants that
    were sampled earlier keep being counted even after their parent exited
    and they were reparented.
    """
    leader = rows.get(pid)
    if leader is not None and (leader.pgid != pid or leader.sid not in (0, pid)):
        raise MonitoringError("worker no longer belongs to its own process group")
    if identity is not None and pid in rows and pid in identity:
        if not _pid_identity_matches(pid, rows[pid].start_time, identity):
            raise PidReuseError("worker pid was reused by another process")
    same_group = {candidate for candidate, row in rows.items() if row.pgid == pid}
    if any(rows[candidate].sid not in (0, pid) for candidate in same_group):
        raise MonitoringError("an unrelated process shares the worker process group")
    owned = set(same_group)
    changed = True
    while changed:
        changed = False
        for candidate, row in rows.items():
            if candidate in owned:
                continue
            if identity is not None and row.start_time and candidate in identity:
                if not _pid_identity_matches(candidate, row.start_time, identity):
                    continue
            if row.ppid in owned:
                owned.add(candidate)
                changed = True
    if identity:
        for candidate, start_time in identity.items():
            if candidate in owned:
                continue
            row = rows.get(candidate)
            if row is None or row.state == "Z":
                continue
            if not _pid_identity_matches(candidate, row.start_time, identity):
                continue
            owned.add(candidate)
    return owned


def _query_vram_bytes(pids: set[int]) -> int:
    """Sum device memory of the owned processes; failures are not zero."""
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
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise MonitoringError(f"nvidia-smi compute-apps query failed: {exc}") from exc
    total = 0
    for line in query.stdout.splitlines():
        fields = [item.strip() for item in line.split(",")]
        if len(fields) != 2:
            raise MonitoringError(f"invalid nvidia-smi compute-apps row: {line!r}")
        try:
            pid, used_mib = int(fields[0]), int(fields[1])
        except ValueError as exc:
            raise MonitoringError(
                f"invalid nvidia-smi compute-apps values: {line!r}"
            ) from exc
        if pid <= 0 or used_mib < 0:
            raise MonitoringError(f"invalid nvidia-smi compute-apps values: {line!r}")
        if pid in owned:
            total += used_mib * _MIB
    return total


def _run_nvidia_smi(args: list[str], *, timeout: int = 10) -> str:
    """Run one nvidia-smi query and return its standard output."""
    try:
        query = subprocess.run(
            ["nvidia-smi", *args],
            check=True,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise MonitoringError(f"nvidia-smi query failed: {exc}") from exc
    return query.stdout


def _gpu_index_uuid_map() -> dict[int, str]:
    """Map every physical GPU index to its full UUID."""
    output = _run_nvidia_smi(
        ["--query-gpu=index,uuid", "--format=csv,noheader,nounits"]
    )
    mapping: dict[int, str] = {}
    for line in output.splitlines():
        fields = [item.strip() for item in line.split(",")]
        if len(fields) != 2:
            raise MonitoringError(f"invalid nvidia-smi index row: {line!r}")
        try:
            index = int(fields[0])
        except ValueError as exc:
            raise MonitoringError(f"invalid nvidia-smi index values: {line!r}") from exc
        if index < 0 or not fields[1]:
            raise MonitoringError(f"invalid nvidia-smi index values: {line!r}")
        mapping[index] = fields[1]
    if not mapping:
        raise MonitoringError("nvidia-smi reported no GPU index")
    return mapping


def _device_state(gpu_uuid: str) -> tuple[int, int]:
    """Read the whole-card memory use and utilization of one authorized GPU.

    The device total is a conservative upper bound: it also covers contexts,
    caches and any unrelated workload that lands on the same card.  A query
    that fails, returns no row or returns an unusable value is an error, never
    a zero measurement.
    """
    output = _run_nvidia_smi(
        [
            "-i",
            gpu_uuid,
            "--query-gpu=uuid,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ]
    )
    rows = [line for line in output.splitlines() if line.strip()]
    if len(rows) != 1:
        raise MonitoringError(
            f"nvidia-smi device query for {gpu_uuid} did not return one row"
        )
    fields = [item.strip() for item in rows[0].split(",")]
    if len(fields) != 3:
        raise MonitoringError(f"invalid nvidia-smi device row: {rows[0]!r}")
    if fields[0] != gpu_uuid:
        raise MonitoringError(
            f"nvidia-smi reported {fields[0]!r} instead of {gpu_uuid!r}"
        )
    try:
        used_mib = int(fields[1])
    except ValueError as exc:
        raise MonitoringError(
            f"invalid nvidia-smi device memory value: {rows[0]!r}"
        ) from exc
    if used_mib < 0:
        raise MonitoringError(f"invalid nvidia-smi device memory value: {rows[0]!r}")
    try:
        utilization = int(fields[2])
    except ValueError as exc:
        raise MonitoringError(
            f"invalid nvidia-smi utilization value: {rows[0]!r}"
        ) from exc
    if utilization < 0:
        raise MonitoringError(f"invalid nvidia-smi utilization value: {rows[0]!r}")
    return used_mib * _MIB, utilization


def _device_used_bytes(gpu_uuid: str) -> int:
    """Read the whole-card memory use of one authorized GPU."""
    used_bytes, _ = _device_state(gpu_uuid)
    return used_bytes


def _device_compute_pids(gpu_uuid: str) -> list[int]:
    """List the compute-app pids that NVML reports for one authorized GPU.

    The rows are filtered on the reported ``gpu_uuid`` instead of trusting the
    ``-i`` selector, because the compute-apps query does not honour that
    selector on every driver, and a busy neighbour card must never be read as
    a busy authorized card.
    """
    output = _run_nvidia_smi(
        [
            "--query-compute-apps=pid,gpu_uuid",
            "--format=csv,noheader,nounits",
        ]
    )
    pids = []
    for line in output.splitlines():
        fields = [item.strip() for item in line.split(",")]
        if len(fields) != 2:
            if not line.strip():
                continue
            raise MonitoringError(f"invalid nvidia-smi compute-app row: {line!r}")
        if fields[1] != gpu_uuid:
            continue
        try:
            pid = int(fields[0])
        except ValueError as exc:
            raise MonitoringError(
                f"invalid nvidia-smi compute-app pid: {line!r}"
            ) from exc
        if pid <= 0:
            raise MonitoringError(f"invalid nvidia-smi compute-app pid: {line!r}")
        pids.append(pid)
    return pids


def _authorize_gpu(gpu_uuid: str | None, env: dict[str, str] | None) -> tuple[int, str]:
    """Bind one authorized GPU UUID to the physical index the worker selects.

    The caller may name the card by physical index or by its full UUID; both
    must resolve to the authorized UUID.  The worker is then pinned to that
    UUID, so the card it sees is the card the supervisor accounts even when
    the CUDA enumeration order differs from the physical order.
    """
    if not isinstance(gpu_uuid, str) or not gpu_uuid.strip():
        raise ValueError("A GPU budget requires an explicit gpu_uuid")
    selected = (env or {}).get("CUDA_VISIBLE_DEVICES")
    if selected is None or not selected.strip():
        raise ValueError(
            "A GPU budget requires CUDA_VISIBLE_DEVICES to name one physical index"
        )
    mapping = _gpu_index_uuid_map()
    selected = selected.strip()
    if selected.isdigit():
        index = int(selected)
        if index not in mapping:
            raise ValueError(f"CUDA_VISIBLE_DEVICES index {index} does not exist")
        if mapping[index] != gpu_uuid:
            raise ValueError(
                f"authorized gpu_uuid {gpu_uuid!r} is not physical index {index} "
                f"({mapping[index]!r})"
            )
        return index, gpu_uuid
    if selected != gpu_uuid:
        raise ValueError(
            f"CUDA_VISIBLE_DEVICES {selected!r} does not name the authorized "
            f"gpu_uuid {gpu_uuid!r}"
        )
    matches = [index for index, value in mapping.items() if value == gpu_uuid]
    if not matches:
        raise ValueError(f"authorized gpu_uuid {gpu_uuid!r} is not a visible GPU")
    if len(matches) > 1:
        raise ValueError(
            f"authorized gpu_uuid {gpu_uuid!r} matches several physical indices"
        )
    index = matches[0]
    if index not in mapping:
        raise ValueError(f"CUDA_VISIBLE_DEVICES index {index} does not exist")
    if mapping[index] != gpu_uuid:
        raise ValueError(
            f"authorized gpu_uuid {gpu_uuid!r} is not physical index {index} "
            f"({mapping[index]!r})"
        )
    return index, gpu_uuid


def _assert_gpu_idle(gpu_uuid: str) -> int:
    """Require an idle, unused card before the worker is started."""
    used_bytes, utilization = _device_state(gpu_uuid)
    compute_pids = _device_compute_pids(gpu_uuid)
    if compute_pids:
        raise GpuNotIdleError(
            f"authorized GPU {gpu_uuid} runs compute apps {compute_pids}"
        )
    if used_bytes > _IDLE_VRAM_BYTES:
        raise GpuNotIdleError(
            f"authorized GPU {gpu_uuid} holds {used_bytes} bytes before the run"
        )
    if utilization > _IDLE_UTILIZATION_PERCENT:
        raise GpuNotIdleError(
            f"authorized GPU {gpu_uuid} is {utilization}% busy before the run"
        )
    return used_bytes


def _safe_stat(path: Path) -> os.stat_result:
    """Stat one path without following symlinks."""
    try:
        return path.stat(follow_symlinks=False)
    except TypeError:  # Python < 3.10 has no follow_symlinks argument.
        return os.lstat(path)


def _measure_output_bytes(output: Path, log_path: Path, record_path: Path) -> int:
    """Count worker files, the log and the record; only vanished files are free."""
    total = 0
    try:
        if output.exists():
            for path in sorted(output.rglob("*")):
                if path.is_symlink():
                    raise UnsafeOutputError(f"symlink is not allowed: {path}")
                if not path.is_file():
                    continue
                try:
                    total += _safe_stat(path).st_size
                except FileNotFoundError:
                    continue
                except OSError as exc:
                    raise MonitoringError(
                        f"output file cannot be measured: {path}: {exc}"
                    ) from exc
    except UnsafeOutputError:
        raise
    except OSError as exc:
        raise MonitoringError(f"output tree cannot be walked: {output}: {exc}") from exc
    for path in (log_path, record_path):
        if path.is_symlink():
            raise UnsafeOutputError(f"symlink is not allowed: {path}")
        if not path.exists():
            continue
        try:
            total += _safe_stat(path).st_size
        except FileNotFoundError:
            continue
        except OSError as exc:
            raise MonitoringError(f"sidecar cannot be measured: {path}: {exc}") from exc
    return total


def _peek_exit(child: subprocess.Popen[Any]) -> int | None:
    """Observe the leader exit status without releasing its PID when possible."""
    known = getattr(child, "returncode", None)
    if known is not None:
        return known
    if not hasattr(os, "waitid"):
        return child.poll()
    try:
        info = os.waitid(os.P_PID, child.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT)
    except ChildProcessError:
        return child.poll()
    except OSError as exc:
        raise MonitoringError(f"worker exit status cannot be observed: {exc}") from exc
    if info is None:
        return None
    if info.si_code == os.CLD_EXITED:
        return int(info.si_status)
    if info.si_code in (os.CLD_KILLED, os.CLD_DUMPED):
        return -int(info.si_status)
    return None


def _sample_usage(
    child: subprocess.Popen[Any],
    *,
    max_vram_bytes: float,
    identity: dict[int, str] | None = None,
    gpu_uuid: str | None = None,
) -> dict[str, int]:
    """Sample the owned process group; device memory is only queried when budgeted."""
    deadline = time.monotonic() + _LEADER_WAIT_SECONDS
    while True:
        rows = _read_proc_rows()
        if child.pid not in rows:
            if _peek_exit(child) is not None:
                # The worker finished before the first sample; there is nothing
                # left to measure and the exit status decides the outcome.
                return {"rss_bytes": 0, "vram_bytes": 0}
            if time.monotonic() > deadline:
                raise MonitoringError("worker process tree cannot be observed")
            time.sleep(0.02)
            continue
        try:
            owned = _owned_pids(rows, child.pid, identity=identity)
        except PidReuseError:
            raise
        except MonitoringError:
            if time.monotonic() > deadline:
                raise
            time.sleep(0.02)
            continue
        break
    if identity is not None:
        for pid in owned:
            row = rows[pid]
            if row.start_time:
                identity[pid] = row.start_time
    rss = sum(rows[pid].rss_bytes for pid in owned if rows[pid].state != "Z")
    vram = 0
    if max_vram_bytes > 0:
        if gpu_uuid is None:
            raise MonitoringError("a GPU budget requires an authorized gpu_uuid")
        # The whole-card total is used on purpose: the container's process
        # table cannot see the NVML pids, so a pid intersection would report
        # zero for a card that is really in use.
        vram = _device_used_bytes(gpu_uuid)
    return {"rss_bytes": int(rss), "vram_bytes": int(vram)}


def _sample_usage_if_running(
    child: subprocess.Popen[Any],
    *,
    max_vram_bytes: float,
    identity: dict[int, str] | None = None,
    gpu_uuid: str | None = None,
) -> dict[str, int] | None:
    """Sample the worker group, or report that the leader already exited."""
    rows = _read_proc_rows()
    if child.pid not in rows and _peek_exit(child) is not None:
        return None
    return _sample_usage(
        child,
        max_vram_bytes=max_vram_bytes,
        identity=identity,
        gpu_uuid=gpu_uuid,
    )


def _group_alive(pgid: int) -> bool:
    """Report whether the owned group still holds a live process."""
    try:
        rows = _read_proc_rows()
    except MonitoringError:
        rows = None
    if rows is not None:
        return any(row.pgid == pgid and row.state != "Z" for row in rows.values())
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _group_members(pgid: int) -> list[int]:
    """Return live members of the owned group, ignoring zombies."""
    rows = _read_proc_rows()
    return sorted(
        pid for pid, row in rows.items() if row.pgid == pgid and row.state != "Z"
    )


def _escaped_members(identity: dict[int, str], pgid: int) -> list[int]:
    """Return sampled descendants that left the worker process group.

    A descendant that called ``setsid`` is no longer reachable through the
    process group, but it was observed as a child of the worker.  It is
    reported here so the cleanup can terminate it explicitly.
    """
    rows = _read_proc_rows()
    escaped = []
    for pid, start_time in identity.items():
        row = rows.get(pid)
        if row is None or row.state == "Z":
            continue
        if row.pgid == pgid:
            continue
        if not _pid_identity_matches(pid, row.start_time, identity):
            continue
        escaped.append(pid)
    return sorted(escaped)


def _escaped_still_alive(
    identity: dict[int, str], pgid: int, candidates: list[int]
) -> list[int]:
    """Re-read the process table and keep the candidates that still match."""
    rows = _read_proc_rows()
    alive = []
    for pid in candidates:
        row = rows.get(pid)
        if row is None or row.state == "Z":
            continue
        if row.pgid == pgid:
            continue
        if not _pid_identity_matches(pid, row.start_time, identity):
            continue
        alive.append(pid)
    return sorted(alive)


def _terminate_escaped(
    identity: dict[int, str] | None, pgid: int | None, cleanup: dict[str, Any]
) -> list[int]:
    """Terminate sampled descendants that left the worker process group.

    The process table is re-read before every signal and only pids whose
    start time still matches the sampled identity are signalled, so a recycled
    pid is never hit.  Escaped descendants that survive are returned so the
    residual report stays truthful.
    """
    if not identity or pgid is None:
        return []
    try:
        escaped = _escaped_members(identity, pgid)
    except MonitoringError as exc:
        cleanup["error"] = (
            cleanup["error"] or f"escaped descendants cannot be verified: {exc}"
        )
        cleanup["verified"] = False
        return []
    for pid in escaped:
        try:
            current = _escaped_still_alive(identity, pgid, [pid])
        except MonitoringError as exc:
            cleanup["error"] = (
                cleanup["error"] or f"escaped descendants cannot be verified: {exc}"
            )
            cleanup["verified"] = False
            continue
        if not current:
            continue
        try:
            os.kill(pid, signal.SIGTERM)
            cleanup["signals"].append(f"SIGTERM(escaped:{pid})")
        except ProcessLookupError:
            cleanup["signals"].append(f"SIGTERM(escaped:{pid}:missing)")
        except PermissionError as exc:
            cleanup["error"] = f"cannot signal escaped descendant {pid}: {exc}"
    deadline = time.monotonic() + 1.0
    while time.monotonic() < deadline and escaped:
        try:
            remaining = _escaped_still_alive(identity, pgid, escaped)
        except MonitoringError as exc:
            cleanup["error"] = (
                cleanup["error"] or f"escaped descendants cannot be verified: {exc}"
            )
            cleanup["verified"] = False
            break
        if not remaining:
            break
        time.sleep(0.05)
    else:
        try:
            remaining = _escaped_still_alive(identity, pgid, escaped)
        except MonitoringError as exc:
            cleanup["error"] = (
                cleanup["error"] or f"escaped descendants cannot be verified: {exc}"
            )
            cleanup["verified"] = False
            remaining = []
        for pid in remaining:
            try:
                current = _escaped_still_alive(identity, pgid, [pid])
            except MonitoringError as exc:
                cleanup["error"] = (
                    cleanup["error"] or f"escaped descendants cannot be verified: {exc}"
                )
                cleanup["verified"] = False
                continue
            if not current:
                continue
            try:
                os.kill(pid, signal.SIGKILL)
                cleanup["signals"].append(f"SIGKILL(escaped:{pid})")
            except ProcessLookupError:
                cleanup["signals"].append(f"SIGKILL(escaped:{pid}:missing)")
            except PermissionError as exc:
                cleanup["error"] = f"cannot signal escaped descendant {pid}: {exc}"
    try:
        return _escaped_still_alive(identity, pgid, escaped)
    except MonitoringError:
        return escaped


def _group_identity_is_certified(pgid: int) -> bool:
    """Certify that every member of the group still belongs to that session."""
    rows = _read_proc_rows()
    members = [row for row in rows.values() if row.pgid == pgid]
    if not members:
        return False
    return all(row.sid in (0, pgid) for row in members)


def _terminate_owned_group(
    child: subprocess.Popen[Any],
    pgid: int | None,
    grace: float,
    cleanup: dict[str, Any],
    identity: dict[int, str] | None = None,
) -> None:
    """Terminate the owned group with TERM, a grace window and then KILL."""
    cleanup["attempted"] = True
    if pgid is None or pgid <= 0 or pgid == os.getpgrp():
        cleanup["error"] = "refusing to signal an unowned process group"
        return
    certified = False
    try:
        certified = _group_identity_is_certified(pgid)
    except MonitoringError as exc:
        cleanup["error"] = f"process group identity cannot be verified: {exc}"
    if identity:
        try:
            rows = _read_proc_rows()
        except MonitoringError as exc:
            cleanup["error"] = (
                cleanup["error"] or f"worker identity cannot be verified: {exc}"
            )
        else:
            leader_row = rows.get(child.pid)
            if leader_row is not None and not _pid_identity_matches(
                child.pid, leader_row.start_time, identity
            ):
                cleanup["error"] = (
                    cleanup["error"]
                    or "refusing to signal a worker pid that was reused"
                )
                cleanup["verified"] = False
                return
    try:
        if certified:
            os.killpg(pgid, signal.SIGTERM)
            cleanup["signals"].append("SIGTERM")
        else:
            os.kill(child.pid, signal.SIGTERM)
            cleanup["signals"].append("SIGTERM(leader-only)")
    except ProcessLookupError:
        cleanup["signals"].append("SIGTERM(missing)")
    except PermissionError as exc:
        cleanup["error"] = f"cannot signal the worker: {exc}"
        return
    cleanup["escaped_pids"] = _terminate_escaped(identity, pgid, cleanup)
    deadline = time.monotonic() + grace
    while time.monotonic() < deadline:
        if not _group_alive(pgid):
            break
        time.sleep(0.05)
    else:
        try:
            if certified:
                os.killpg(pgid, signal.SIGKILL)
                cleanup["signals"].append("SIGKILL")
            else:
                os.kill(child.pid, signal.SIGKILL)
                cleanup["signals"].append("SIGKILL(leader-only)")
        except ProcessLookupError:
            cleanup["signals"].append("SIGKILL(missing)")
        except PermissionError as exc:
            cleanup["error"] = f"cannot signal the worker: {exc}"
    cleanup["escaped_pids"] = _terminate_escaped(identity, pgid, cleanup)
    # The leader PID is released only after the group identity was certified,
    # so the process-group ID cannot be recycled between check and signal.
    try:
        child.wait(timeout=max(1.0, grace))
    except subprocess.TimeoutExpired:
        cleanup["error"] = cleanup["error"] or "worker leader did not exit"
    try:
        residual = _group_members(pgid)
        residual = sorted(set(residual) | set(cleanup.get("escaped_pids", [])))
    except MonitoringError as exc:
        cleanup["error"] = (
            cleanup["error"] or f"residual processes cannot be verified: {exc}"
        )
        cleanup["verified"] = False
    else:
        cleanup["residual_pids"] = residual
        if cleanup.get("error") is None:
            cleanup["verified"] = not residual


def _read_completion(
    output: Path, completion_protocol: int = COMPLETION_PROTOCOL
) -> tuple[dict[str, Any] | None, str, str]:
    """Read the worker completion marker; missing or partial markers fail."""
    path = output / "completion.json"
    if path.is_symlink():
        return None, "unsafe_completion", f"symlink is not allowed: {path}"
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None, "missing_completion", f"worker did not write {path}"
    except OSError as exc:
        return None, "unreadable_completion", f"{path} cannot be read: {exc}"
    try:
        value = json.loads(text)
    except (UnicodeError, json.JSONDecodeError) as exc:
        return None, "invalid_completion", f"{path} is not valid JSON: {exc}"
    if not isinstance(value, dict):
        return None, "invalid_completion", f"{path} must contain a JSON object"
    missing = [field for field in REQUIRED_COMPLETION_FIELDS if field not in value]
    if missing:
        return None, "incomplete_completion", f"completion marker misses {missing}"
    if value.get("status") != "completed":
        return (
            None,
            "incomplete_completion",
            f"completion status is {value.get('status')!r}",
        )
    if value.get("protocol") != completion_protocol:
        return (
            None,
            "incomplete_completion",
            f"completion protocol is {value.get('protocol')!r}",
        )
    binding_sha = value.get("binding_sha256")
    if (
        not isinstance(binding_sha, str)
        or not binding_sha
        or len(binding_sha) != 64
        or any(character not in "0123456789abcdef" for character in binding_sha.lower())
    ):
        return (
            None,
            "incomplete_completion",
            "completion binding_sha256 must be a 64-hex digest",
        )
    return value, "", ""


def _write_record(path: Path, record: dict[str, Any]) -> None:
    """Persist the supervisor record atomically."""
    temporary = path.with_name(path.name + ".tmp")
    if temporary.is_symlink():
        raise UnsafeOutputError(f"symlink is not allowed: {temporary}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(record, handle, sort_keys=True, indent=2, default=str)
        handle.write("\n")
    os.replace(temporary, path)


def _set_failure(
    record: dict[str, Any],
    stage: str,
    failure_type: str,
    message: str,
    tb: str | None = None,
) -> None:
    """Keep the first failure only; later events must not overwrite the cause."""
    if record["failure"] is None:
        record["failure"] = {
            "stage": stage,
            "type": failure_type,
            "message": message,
            "traceback": tb,
        }


def _first_limit_breach(
    usage: dict[str, int], output_bytes: int, limits: dict[str, float]
) -> tuple[str, str, str] | None:
    """Return the first breached limit as (stage, type, message)."""
    if usage["rss_bytes"] > limits["max_rss_bytes"]:
        return (
            "limits",
            "rss_limit",
            f"resident memory {usage['rss_bytes']} exceeds "
            f"{int(limits['max_rss_bytes'])} bytes",
        )
    if usage["vram_bytes"] > limits["max_vram_bytes"]:
        return (
            "limits",
            "vram_limit",
            f"device memory {usage['vram_bytes']} exceeds "
            f"{int(limits['max_vram_bytes'])} bytes",
        )
    if output_bytes > limits["max_output_bytes"]:
        return (
            "limits",
            "output_limit",
            f"output volume {output_bytes} exceeds "
            f"{int(limits['max_output_bytes'])} bytes",
        )
    return None


def supervise(
    command: list[str],
    output: Path,
    *,
    cwd: Path,
    budget: dict[str, Any],
    env: dict[str, str] | None = None,
    root: Path | None = None,
    gpu_uuid: str | None = None,
    completion_protocol: int = COMPLETION_PROTOCOL,
) -> dict[str, Any]:
    """Run one worker under a hard resource budget and return the evidence record.

    The worker owns its output directory and must write ``completion.json``
    with ``{"status": "completed"}``.  A run counts as completed only when the
    worker exits with code zero, leaves no owned descendant behind and reports
    that completion marker.

    A GPU budget requires an explicit ``gpu_uuid`` that is proven to be the
    physical index the worker selects.  Device memory is then accounted as the
    whole-card total of that GPU, because the container's process table cannot
    be intersected with the NVML pids reliably.
    """
    if not _is_linux():
        raise UnsupportedPlatformError("resource supervision requires Linux /proc")
    if type(completion_protocol) is not int or completion_protocol not in (123, 125):
        raise ValueError("Unsupported worker completion protocol")
    limits = _validate_budget(budget)
    argv = _validate_command(command)
    child_env = _build_child_env(env)
    work_dir = _validate_root(Path(cwd))
    if root is None:
        # The worker output usually sits beside the code root (outputs/ under
        # the experiment root); the parent of the code root is that boundary.
        boundary = work_dir.parent
    else:
        boundary = _validate_root(Path(root))
    output = _validate_output(Path(output), boundary=boundary)
    record_path, log_path = _sidecar_paths(output)
    for path in (record_path, log_path):
        if path.is_symlink():
            raise UnsafeOutputError(f"symlink is not allowed: {path}")
        if path.exists():
            raise FileExistsError(f"supervisor path already exists: {path}")
    output.parent.mkdir(parents=True, exist_ok=True)

    started = time.monotonic()
    deadline = started + limits["max_wall_seconds"]
    authorized_uuid: str | None = None
    gpu_index: int | None = None
    vram_baseline_bytes = 0
    preflight_failure: tuple[str, str, str] | None = None
    if limits["max_vram_bytes"] > 0:
        try:
            gpu_index, authorized_uuid = _authorize_gpu(gpu_uuid, env)
        except MonitoringError as exc:
            # A missing or unusable index map is a monitoring failure; it must
            # stop the run and leave evidence instead of guessing a card.
            preflight_failure = ("preflight", "monitoring_error", str(exc))
        if authorized_uuid is not None:
            # Pin the worker to the authorized card by UUID: the CUDA
            # enumeration order need not match the physical order, so the
            # worker and the supervisor must name the device the same way.
            child_env["CUDA_VISIBLE_DEVICES"] = authorized_uuid
    record: dict[str, Any] = {
        "schema": 1,
        "status": "running",
        "command": argv,
        "cwd": str(work_dir),
        "output": str(output),
        "budget": limits,
        "supervisor_pid": os.getpid(),
        "pid": None,
        "pgid": None,
        "returncode": None,
        "elapsed_seconds": 0.0,
        "peak_rss_bytes": 0,
        "peak_vram_bytes": 0,
        "output_bytes": 0,
        "samples": 0,
        "vram_accounting": "device_total" if authorized_uuid else "none",
        "gpu_uuid": authorized_uuid,
        "gpu_index": gpu_index,
        "vram_baseline_bytes": 0,
        "failure": None,
        "completion": None,
        "cleanup": {
            "attempted": False,
            "signals": [],
            "residual_pids": [],
            "verified": False,
            "error": None,
        },
        "supervisor_json": str(record_path),
        "supervisor_log": str(log_path),
    }
    pending_signal: int | None = None
    old_handlers: dict[int, Any] = {}
    child: subprocess.Popen[Any] | None = None
    identity: dict[int, str] = {}

    def _receive(signum: int, _frame: Any) -> None:
        nonlocal pending_signal
        pending_signal = signum

    def _fail(
        stage: str, failure_type: str, message: str, tb: str | None = None
    ) -> None:
        _set_failure(record, stage, failure_type, message, tb)
        _terminate_owned_group(
            child,
            record["pgid"],
            _KILL_GRACE_SECONDS,
            record["cleanup"],
            identity,
        )

    try:
        with log_path.open("x", encoding="utf-8") as log:
            if preflight_failure is not None:
                _set_failure(
                    record,
                    preflight_failure[0],
                    preflight_failure[1],
                    preflight_failure[2],
                )
                record["status"] = "failed"
                record["elapsed_seconds"] = time.monotonic() - started
                _write_record(record_path, record)
                return record
            if authorized_uuid is not None:
                try:
                    vram_baseline_bytes = _assert_gpu_idle(authorized_uuid)
                except GpuNotIdleError as exc:
                    _set_failure(
                        record,
                        "preflight",
                        "gpu_not_idle",
                        str(exc),
                        traceback.format_exc(),
                    )
                    record["status"] = "failed"
                    record["elapsed_seconds"] = time.monotonic() - started
                    _write_record(record_path, record)
                    return record
                except MonitoringError as exc:
                    _set_failure(
                        record,
                        "preflight",
                        "monitoring_error",
                        str(exc),
                        traceback.format_exc(),
                    )
                    record["status"] = "failed"
                    record["elapsed_seconds"] = time.monotonic() - started
                    _write_record(record_path, record)
                    return record
                record["vram_baseline_bytes"] = vram_baseline_bytes
                _write_record(record_path, record)
            try:
                child = subprocess.Popen(
                    argv,
                    cwd=str(work_dir),
                    env=child_env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
            except OSError as exc:
                _set_failure(
                    record,
                    "start",
                    "worker_start",
                    f"{type(exc).__name__}: {exc}",
                    traceback.format_exc(),
                )
                child = None
            if child is None:
                record["status"] = "failed"
                record["elapsed_seconds"] = time.monotonic() - started
                _write_record(record_path, record)
                return record
            record["pid"] = child.pid
            record["pgid"] = child.pid
            if threading.current_thread() is threading.main_thread():
                for sig in (signal.SIGTERM, signal.SIGINT):
                    old_handlers[sig] = signal.signal(sig, _receive)
            _write_record(record_path, record)
            while True:
                if pending_signal is not None:
                    _fail(
                        "signal",
                        "parent_signal",
                        f"supervisor received signal {pending_signal}",
                    )
                    break
                if time.monotonic() > deadline:
                    _fail(
                        "limits",
                        "wall_limit",
                        f"wall clock exceeded {limits['max_wall_seconds']} seconds",
                    )
                    break
                try:
                    usage = _sample_usage_if_running(
                        child,
                        max_vram_bytes=limits["max_vram_bytes"],
                        identity=identity,
                        gpu_uuid=authorized_uuid,
                    )
                    if usage is None:
                        # The leader exited between two samples; the exit status
                        # and the completion marker decide the outcome.
                        record["returncode"] = _peek_exit(child)
                        break
                except Exception as exc:  # noqa: BLE001 - sampling must fail closed
                    _fail(
                        "sampling",
                        "monitoring_error",
                        f"{type(exc).__name__}: {exc}",
                        traceback.format_exc(),
                    )
                    break
                record["samples"] += 1
                record["peak_rss_bytes"] = max(
                    record["peak_rss_bytes"], usage["rss_bytes"]
                )
                record["peak_vram_bytes"] = max(
                    record["peak_vram_bytes"], usage["vram_bytes"]
                )
                breach = _first_limit_breach(usage, record["output_bytes"], limits)
                if breach is not None:
                    _fail(*breach)
                    break
                try:
                    record["output_bytes"] = _measure_output_bytes(
                        output, log_path, record_path
                    )
                except UnsafeOutputError as exc:
                    _fail(
                        "output_scan", "unsafe_output", str(exc), traceback.format_exc()
                    )
                    break
                except Exception as exc:  # noqa: BLE001 - output scan must fail closed
                    _fail(
                        "output_scan",
                        "monitoring_error",
                        f"{type(exc).__name__}: {exc}",
                        traceback.format_exc(),
                    )
                    break
                breach = _first_limit_breach(usage, record["output_bytes"], limits)
                if breach is not None:
                    _fail(*breach)
                    break
                try:
                    code = _peek_exit(child)
                except MonitoringError as exc:
                    _fail(
                        "worker", "monitoring_error", str(exc), traceback.format_exc()
                    )
                    break
                if code is not None:
                    record["returncode"] = code
                    break
                time.sleep(limits["poll_seconds"])

            if record["failure"] is None:
                try:
                    residual = [
                        pid
                        for pid in _group_members(record["pgid"])
                        if pid != child.pid
                    ]
                except MonitoringError as exc:
                    _fail(
                        "cleanup",
                        "monitoring_error",
                        f"owned descendants cannot be verified: {exc}",
                        traceback.format_exc(),
                    )
                    residual = []
                if record["failure"] is None and residual:
                    _fail(
                        "cleanup",
                        "residual_descendant",
                        f"owned descendants survived the worker: {residual}",
                    )
            if record["failure"] is None and record["returncode"] not in (0, None):
                _fail(
                    "worker",
                    "worker_exit",
                    f"worker exited with code {record['returncode']}",
                )
            if record["failure"] is None:
                completion, failure_type, message = _read_completion(
                    output, completion_protocol
                )
                if completion is None:
                    _fail("completion", failure_type, message)
                else:
                    record["completion"] = completion
            if record["failure"] is None and record["returncode"] is None:
                _fail(
                    "worker", "worker_exit", "worker exit status could not be observed"
                )
            if record["returncode"] is None:
                record["returncode"] = child.returncode
            # The leader is reaped only after the owned group was certified, so
            # the process-group ID cannot be recycled during the checks above.
            if child.returncode is None:
                try:
                    child.wait(timeout=_LEADER_WAIT_SECONDS)
                except subprocess.TimeoutExpired:
                    if record["failure"] is None:
                        _fail("worker", "worker_exit", "worker leader did not exit")
    finally:
        for sig, handler in old_handlers.items():
            signal.signal(sig, handler)

    record["elapsed_seconds"] = time.monotonic() - started
    try:
        record["output_bytes"] = _measure_output_bytes(output, log_path, record_path)
    except (MonitoringError, UnsafeOutputError) as exc:
        if record["failure"] is None:
            _fail(
                "output_scan",
                "monitoring_error",
                f"{type(exc).__name__}: {exc}",
                traceback.format_exc(),
            )
    if (
        record["failure"] is None
        and record["output_bytes"] > limits["max_output_bytes"]
    ):
        _fail(
            "limits",
            "output_limit",
            f"output volume {record['output_bytes']} exceeds "
            f"{int(limits['max_output_bytes'])} bytes",
        )
    record["status"] = "failed" if record["failure"] is not None else "completed"
    for _ in range(3):
        _write_record(record_path, record)
        try:
            measured = _measure_output_bytes(output, log_path, record_path)
        except (MonitoringError, UnsafeOutputError):
            break
        if measured == record["output_bytes"]:
            break
        record["output_bytes"] = measured
    _write_record(record_path, record)
    return record
