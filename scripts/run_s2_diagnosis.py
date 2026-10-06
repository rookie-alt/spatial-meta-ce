import argparse
import csv
import ctypes
import datetime
import json
import os
import resource
import secrets
import signal
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_s1_diagnosis import (
    MEMORY_BYTES,
    SUPERVISOR_MEMORY_BYTES,
    WORKER_MEMORY_BYTES,
    THREAD_ENV,
    TIMEOUT,
    stop_child,
    interrupted,
)

ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = (
    "protocol.json",
    "summary.json",
    "pool.csv",
    "exposure.csv",
    "residuals.csv",
    "comparison.csv",
    "verification.json",
)
CODE_FILES = (
    "scripts/run_s2_diagnosis.py",
    "scripts/s2_source_replay.py",
    "scripts/s2_exposure_stats.py",
    "scripts/s2_verify.py",
    "scripts/run_s1_diagnosis.py",
)


def safe(root: Path, path: Path) -> Path:
    if any(p.is_symlink() for p in (path.absolute(), *path.absolute().parents)):
        raise ValueError("Symlink paths are forbidden")
    if ".." in path.parts or not path.resolve().is_relative_to(root.resolve()):
        raise ValueError("S2 path escapes root")
    return path


def file_hash(path: Path) -> str:
    import hashlib

    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def write_json(path: Path, value: dict) -> None:
    text = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(text)


def update_status(output: Path, state: str, **values: object) -> None:
    temp = output / "status.tmp"
    temp.write_text(
        json.dumps(
            {
                "status": state,
                "updated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                **values,
            },
            indent=2,
            allow_nan=False,
        ),
        encoding="utf-8",
    )
    temp.replace(output / "status.json")


def csv_output(output: Path, name: str, rows: list) -> None:
    columns = sorted({key for row in rows for key in row})
    with (output / name).open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def validate_output(root: Path, output: Path) -> Path:
    safe(root / "runs", output)
    resolved = output.resolve()
    if resolved == (root / "runs").resolve() or resolved.is_relative_to(
        root / "runs/precision-screen-20261006"
    ):
        raise ValueError("Output overlaps frozen inputs or runs root")
    if output.exists():
        raise FileExistsError("S2 requires a new run directory")
    return resolved


def analyze(root: Path, output: Path) -> dict:
    import numpy as np
    from scripts import s2_source_replay as core, s2_exposure_stats as stats, s2_verify

    started = time.monotonic()
    if np.__version__ != "1.26.1":
        raise ValueError("Formal S2 requires recorded NumPy 1.26.1")
    contract = core.load_contract(root)
    before = core.snapshot(root, contract)
    code_before = {name: file_hash(safe(root, root / name)) for name in CODE_FILES}
    protocol = {
        "schema": 1,
        "mode": "source_only_metadata_diagnosis",
        "plan_sha256": core.FROZEN["实验文档/16-S2源训练曝光与残差监督分布诊断方案.md"],
        "train": contract["train"],
        "validation": contract["validation"],
        "config": contract["config"],
        "input_sha256": before,
        "code_sha256": code_before,
        "timeout_seconds": TIMEOUT,
        "address_space_bytes": MEMORY_BYTES,
        "supervisor_address_space_bytes": SUPERVISOR_MEMORY_BYTES,
        "worker_address_space_bytes": WORKER_MEMORY_BYTES,
        "prefix_steps": [1000, 2000, 4000],
        "numpy": "1.26.1",
        "training": False,
        "inference": False,
        "target_data_read": False,
        "limitations": "One seed; posthoc S1 groups; no causal intervention, true gradients, checkpoint RNG certification, or independently counted SQL labels.",
    }
    write_json(output / "protocol.json", protocol)
    caches = {
        db: core.read_cache(root, db, contract["sources"][db])
        for db in contract["train"] + contract["validation"]
    }
    update_status(output, "running", stage="S2-0 replay certification")
    replay = core.replay(caches, contract["train"], contract["config"])
    core.certify(replay)
    update_status(
        output, "running", stage="S2-1/S2-2 fixed tables", replay_verified=True
    )
    tables = stats.build_tables(
        caches, replay, contract["train"], contract["validation"]
    )
    update_status(
        output, "running", stage="independent reconstruction", replay_verified=True
    )
    verification = s2_verify.verify(
        caches,
        replay,
        tables,
        contract["train"],
        contract["validation"],
        contract["config"],
    )
    if (
        core.snapshot(root, contract) != before
        or {name: file_hash(root / name) for name in CODE_FILES} != code_before
    ):
        raise ValueError("Inputs or execution code changed during S2")
    verification = {
        **verification,
        "source_rehashed_after": True,
        "code_rehashed_after": True,
        "replay_verified": True,
    }
    write_json(output / "verification.json", verification)
    for name in ("pool", "exposure", "residuals", "comparison"):
        csv_output(output, name + ".csv", tables[name])
    summary = {
        "schema": 1,
        "purpose": "S2 public source-only exposure and residual supervision diagnosis; not model selection or causal verification",
        "replay_verified": True,
        "causal_explanation_verified": False,
        "training": False,
        "inference": False,
        "adaptation": False,
        "target_data_read": False,
        "database_count": False,
        "input_sha256": before,
        "code_sha256": code_before,
        "sample_plan_digest": replay["digest"],
        "prefixes": {
            str(step): {
                "digest": v["digest"],
                "episodes": v["episodes"],
                "presentations": v["presentations"],
                "support_role_presentations": sum(
                    int(d["support"].sum()) for d in v["databases"].values()
                ),
                "query_role_presentations": sum(
                    int(d["query"].sum()) for d in v["databases"].values()
                ),
                "nominal_mass": sum(
                    float(d["mass"].sum()) for d in v["databases"].values()
                ),
            }
            for step, v in replay["prefixes"].items()
        },
        "source_pool": {
            db: {
                "raw_rows": d["raw_rows"],
                "unique_queries": len(d["true"]),
                "duplicates": d["raw_rows"] - len(d["true"]),
            }
            for db, d in caches.items()
        },
        "distribution_summary": tables["distribution_summary"],
        "definitions": {
            "nominal_mass": "1/[4*(K+64)] per presentation, not actual gradient contribution",
            "support_counter": "Historical ERM support_presentations=0; support-role rows are supervised in merged query counter",
            "r_ref": "FP64 log(max(true,1))-log(max(pg,1))",
            "r_fp32": "Each FP64 log separately cast FP32 then promoted and subtracted",
            "pg_direction": "FP64 r_ref positive/negative/zero only",
            "sparse": "<200 pool or sampled distinct unique queries: counts/mass only",
            "weighted_quantiles": "inverse weighted empirical CDF; unweighted: NumPy linear",
            "directory_trust": "Input and output directories must not be concurrently modified; hashes do not defend against same-privilege malicious replacement",
        },
        "runtime": {
            "python": sys.version,
            "numpy": np.__version__,
            "seconds": time.monotonic() - started,
            "peak_rss_kib_linux": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "threads": {k: os.environ.get(k) for k in THREAD_ENV},
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "memory_budget_bytes": MEMORY_BYTES,
            "timeout_seconds": TIMEOUT,
        },
        "completed_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    write_json(output / "summary.json", summary)
    write_json(
        output / "output-manifest.json",
        {name: file_hash(safe(output, output / name)) for name in OUTPUTS},
    )
    return summary


def verify_outputs(output: Path) -> dict:
    manifest = json.loads(safe(output, output / "output-manifest.json").read_text())
    if set(manifest) != set(OUTPUTS):
        raise ValueError("Incomplete S2 output manifest")
    for name, sha in manifest.items():
        if file_hash(safe(output, output / name)) != sha:
            raise ValueError("S2 output failed supervisor byte verification")
    return manifest


def supervise(command: list, output: Path, timeout: float = TIMEOUT) -> None:
    started = time.monotonic()
    token = secrets.token_hex(16)
    write_json(output / "supervision.json", {"pid": os.getpid(), "token": token})
    env = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": "",
        "S2_PARENT": str(os.getpid()),
        "S2_TOKEN": token,
        **{k: "1" for k in THREAD_ENV},
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    previous = signal.signal(signal.SIGTERM, interrupted)
    child = None
    try:
        if timeout <= 0:
            raise TimeoutError("S2 budget exhausted before worker launch")
        child = subprocess.Popen(command, start_new_session=True, env=env)
        code = child.wait(timeout=max(0.001, timeout - (time.monotonic() - started)))
        current = json.loads((output / "status.json").read_text())
        if code != 0 or current["status"] != "outputs_ready":
            raise RuntimeError(
                f"S2 worker did not complete, exit={code}, status={current['status']}"
            )
        hashes = verify_outputs(output)
        if time.monotonic() - started >= timeout:
            raise TimeoutError("S2 budget exhausted during output verification")
        update_status(
            output,
            "completed",
            output_sha256=hashes,
            replay_verified=True,
            causal_explanation_verified=False,
            training=False,
            inference=False,
            target_data_read=False,
            seconds=time.monotonic() - started,
            supervisor_peak_rss_kib_linux=resource.getrusage(
                resource.RUSAGE_SELF
            ).ru_maxrss,
        )
    except BaseException as error:
        signal.setitimer(signal.ITIMER_REAL, 0)
        if child is not None:
            stop_child(child)
        current = json.loads((output / "status.json").read_text())
        state = (
            "blocked_replay_unverified"
            if current["status"] == "blocked_replay_unverified"
            else "failed"
        )
        update_status(
            output,
            state,
            error=f"{type(error).__name__}: {error}",
            seconds=time.monotonic() - started,
        )
        if isinstance(error, (subprocess.TimeoutExpired, KeyboardInterrupt)):
            raise TimeoutError("S2 interrupted or exceeded 30-minute bound") from error
        raise
    finally:
        signal.signal(signal.SIGTERM, previous)


def worker(root: Path, output: Path) -> None:
    if (
        not sys.platform.startswith("linux")
        or os.environ.get("CUDA_VISIBLE_DEVICES") != ""
    ):
        raise RuntimeError("Formal S2 requires Linux CPU-only execution")
    if any(os.environ.get(k) != "1" for k in THREAD_ENV):
        raise ValueError("S2 requires single-thread numeric libraries")
    safe(root / "runs", output)
    if {p.name for p in output.iterdir()} != {"status.json", "supervision.json"}:
        raise ValueError("Worker requires fresh supervised output")
    parent = int(os.environ.get("S2_PARENT", "0"))
    if parent != os.getppid() or json.loads(
        (output / "supervision.json").read_text()
    ) != {"pid": parent, "token": os.environ.get("S2_TOKEN")}:
        raise ValueError("Worker ownership mismatch")
    if (
        ctypes.CDLL(None, use_errno=True).prctl(1, signal.SIGTERM) != 0
        or os.getppid() != parent
    ):
        raise RuntimeError("Could not bind worker to supervisor")
    resource.setrlimit(resource.RLIMIT_AS, (WORKER_MEMORY_BYTES, WORKER_MEMORY_BYTES))
    try:
        from scripts.s2_source_replay import ReplayMismatch

        update_status(output, "running", stage="input verification")
        analyze(root, output)
        update_status(output, "outputs_ready", replay_verified=True)
    except Exception as error:
        state = (
            "blocked_replay_unverified"
            if isinstance(error, ReplayMismatch)
            else "failed"
        )
        update_status(output, state, error=f"{type(error).__name__}: {error}")
        raise


def run_main(deadline: float) -> None:
    if sys.platform.startswith("linux"):
        limit = (
            WORKER_MEMORY_BYTES if "--worker" in sys.argv else SUPERVISOR_MEMORY_BYTES
        )
        resource.setrlimit(resource.RLIMIT_AS, (limit, MEMORY_BYTES))
    parser = argparse.ArgumentParser(
        description="CPU-only S2 frozen source metadata replay and diagnosis"
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        worker(ROOT, Path(args.output))
        return
    output = validate_output(ROOT, Path(args.output))
    if not args.execute:
        print(
            json.dumps(
                {
                    "purpose": "S2 dry-run, no inputs opened or analysis executed",
                    "output": str(output),
                    "timeout_seconds": TIMEOUT,
                    "address_space_bytes": MEMORY_BYTES,
                }
            )
        )
        return
    if not sys.platform.startswith("linux"):
        raise RuntimeError("Formal S2 requires the server Linux runtime")
    output.mkdir(parents=True)
    update_status(output, "planned")
    supervise(
        [
            sys.executable,
            "-B",
            str(Path(__file__).resolve()),
            "--execute",
            "--worker",
            "--output",
            str(output),
        ],
        output,
        timeout=deadline - time.monotonic(),
    )


def deadline_expired(_signal: int, _frame: object) -> None:
    raise TimeoutError("S2 absolute 30-minute deadline reached")


def main() -> None:
    deadline = time.monotonic() + TIMEOUT
    parent = (
        sys.platform.startswith("linux")
        and "--worker" not in sys.argv
        and "--execute" in sys.argv
    )
    previous = signal.signal(signal.SIGALRM, deadline_expired) if parent else None
    if parent:
        signal.setitimer(signal.ITIMER_REAL, TIMEOUT)
    try:
        run_main(deadline)
    finally:
        if parent:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, previous)


if __name__ == "__main__":
    main()
