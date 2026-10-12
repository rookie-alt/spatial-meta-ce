"""Diagnostic RSS-trajectory probe for the Meta65 scope R training recipe.

Diagnostic only: not a formal run, no approval, no certified outputs, and no
frozen file is modified. One child trains the frozen R recipe for a bounded
step count while this parent samples process-group RSS every poll and reads the
child's monitor lines without blocking, then enforces hard time/RSS/output caps
with process-group cleanup.

Two instrument defects in the first (2026-10-12) probe are repaired here and
covered by tests/test_meta65_scope_memory_probe.py:

* the step parser matched a token starting with ``"step":`` while the monitor
  emits ``{"step": 100, ...``, so ``final_step_seen`` was always 0;
* the sample loop read one monitor line before every sample, so a sample was
  taken per 100-step validation line instead of once per poll interval.

Samples are JSON-serialised with ``allow_nan=False`` and each sample keeps the
child's last reported step, so the trajectory stays relatable to progress.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import select
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

DEFAULT_ROOT = Path("/gly/weilin/spatial-meta-ce-meta65-scope-20261011-code")
HARD_SECONDS = 3600.0
RSS_KILL_BYTES = 16 * 1024**3
OUTPUT_KILL_BYTES = 2 * 1024**3
POLL_SECONDS = 1.0
DEFAULT_STEPS = 4000

STEP_PATTERN = re.compile(r'"step"\s*:\s*(\d+)')


def log(message: str) -> None:
    print(json.dumps({"probe": message}, ensure_ascii=False), flush=True)


def parse_step(line: str, previous: int) -> int:
    """Return the largest step number reported on one monitor line.

    The monitor emits ``{"step": 100, ...}``, so the value is found anywhere in
    the line rather than only at the start of one whitespace-separated token.
    """
    found = [int(value) for value in STEP_PATTERN.findall(line)]
    return max([previous, *found])


def train_worker(root: Path, out: Path, steps: int) -> None:
    import torch  # noqa: F401  (device runtime must initialise inside the child)

    from scripts import meta65_scope_contract as contract
    from scripts import run_meta65_screen as screen
    from spatial_meta_ce.train import train

    config = contract.arm_config(
        contract.load_config(root / "configs/meta65_scope.json"), "R_head3", "formal"
    )
    config = {**config, "steps": steps}  # Probe cap; every other field is the R recipe.
    data = screen.validate_data(
        contract.read_json(root / "configs/meta65_server_data.json"), root
    )
    out.mkdir(parents=True, exist_ok=False)
    result = train(config, data, out, device="cuda")
    log(
        json.dumps(
            {"done": True, "last_step": result["last_step"], "counts": result["counts"]}
        )
    )


def _proc_fields(pid: int) -> tuple[str, int, int] | None:
    """Return (state, pgid, rss_pages) for one pid, or None if unreadable."""
    try:
        text = Path(f"/proc/{pid}/stat").read_text(encoding="ascii")
        fields = text[text.rfind(")") + 2 :].split()
        return fields[0], int(fields[2]), int(fields[21])
    except (OSError, ValueError, IndexError):
        return None


def group_rss(pgid: int) -> int:
    """Sum RSS over live members of one process group."""
    total = 0
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        fields = _proc_fields(int(entry.name))
        if fields is None:
            continue
        state, pgrp, rss_pages = fields
        if pgrp == pgid and state != "Z":
            total += rss_pages * 4096
    return total


def group_alive(pgid: int) -> list[int]:
    alive = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        fields = _proc_fields(int(entry.name))
        if fields is None:
            continue
        state, pgrp, _ = fields
        if pgrp == pgid and state != "Z":
            alive.append(int(entry.name))
    return alive


def output_bytes(path: Path, log_path: Path) -> int:
    size = log_path.stat().st_size if log_path.is_file() else 0
    if path.is_dir():
        size += sum(p.stat().st_size for p in path.rglob("*") if p.is_file())
    return size


def cleanup(child: subprocess.Popen[Any]) -> tuple[bool, list[int]]:
    sent: list[int] = []
    for sig in (signal.SIGTERM, signal.SIGKILL):
        if child.poll() is not None and not group_alive(child.pid):
            break
        try:
            os.killpg(child.pid, sig)
            sent.append(int(sig))
        except ProcessLookupError:
            break
        deadline = time.monotonic() + (10.0 if sig == signal.SIGTERM else 5.0)
        while group_alive(child.pid) and time.monotonic() < deadline:
            time.sleep(0.2)
    return not group_alive(child.pid) and child.poll() is not None, sent


def drain(stream: Any, log_stream: Any, state: dict[str, Any]) -> int:
    """Consume whatever the child has written without blocking.

    Complete lines are copied to the log and mined for a step number; a partial
    trailing line is buffered so a step split across two reads is neither lost
    nor double counted.
    """
    chunk = b""
    while True:
        ready, _, _ = select.select([stream], [], [], 0)
        if not ready:
            break
        piece = os.read(stream.fileno(), 65536)
        if not piece:
            break
        chunk += piece
    if not chunk and not state.get("pending"):
        return state["step"]
    buffered = state.get("pending", "") + chunk.decode("utf-8", errors="replace")
    *lines, tail = buffered.split("\n")
    for line in lines:
        if line:
            log_stream.write(line + "\n")
            state["step"] = parse_step(line, state["step"])
    log_stream.flush()
    state["pending"] = tail
    return state["step"]


def supervise(out: Path, steps: int) -> dict:
    log_path = out.with_name(out.name + ".probe.log")
    record_path = out.with_name(out.name + ".probe.json")
    child = subprocess.Popen(
        [sys.executable, "-B", __file__, "--train", "--out", str(out), "--steps", str(steps)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
    )
    started = time.monotonic()
    samples: list[dict] = []
    state: dict[str, Any] = {"step": 0, "pending": ""}
    failure = None
    with log_path.open("w", encoding="utf-8") as stream:
        while True:
            state["step"] = drain(child.stdout, stream, state)
            elapsed = time.monotonic() - started
            rss = group_rss(child.pid)
            samples.append({"t": round(elapsed, 2), "rss_bytes": rss, "step": state["step"]})
            size = output_bytes(out, log_path)
            if elapsed > HARD_SECONDS:
                failure = "probe_timeout"
            elif rss > RSS_KILL_BYTES:
                failure = "probe_rss_kill"
            elif size > OUTPUT_KILL_BYTES:
                failure = "probe_output_kill"
            if failure is not None:
                break
            if child.poll() is not None and not group_alive(child.pid):
                break
            time.sleep(POLL_SECONDS)
        # Drain to completion: a finished child may still have buffered monitor
        # lines (including its final "done" record) that no sample has consumed.
        for _ in range(100):  # Bounded: a dead child stops producing output.
            state["step"] = drain(child.stdout, stream, state)
            if not state.get("pending"):
                break
            if child.poll() is None:
                time.sleep(0.05)
                continue
            stream.write(state["pending"] + "\n")
            state["step"] = parse_step(state["pending"], state["step"])
            state["pending"] = ""
            stream.flush()
            break
    clean, sent = cleanup(child)
    if child.stdout is not None:
        child.stdout.close()
    record = {
        "schema": 1,
        "probe": "meta65-scope-r-memory",
        "steps_requested": steps,
        "device": "cuda",
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "returncode": child.returncode,
        "failure": failure,
        "elapsed": time.monotonic() - started,
        "samples": samples,
        "cleanup_verified": clean,
        "signals": sent,
        "final_step_seen": state["step"],
        "output_bytes": output_bytes(out, log_path),
    }
    record_path.write_text(
        json.dumps(record, ensure_ascii=False, allow_nan=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return record


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--train", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.steps <= 0:
        parser.error("steps must be positive")
    if args.train:
        train_worker(args.root, args.out, args.steps)
        return 0
    record = supervise(args.out, args.steps)
    log(
        json.dumps(
            {
                "returncode": record["returncode"],
                "failure": record["failure"],
                "elapsed": record["elapsed"],
                "final_step_seen": record["final_step_seen"],
                "peak_rss_bytes": max((s["rss_bytes"] for s in record["samples"]), default=0),
                "samples": len(record["samples"]),
                "cleanup_verified": record["cleanup_verified"],
            }
        )
    )
    return 0 if record["failure"] is None and record["returncode"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
