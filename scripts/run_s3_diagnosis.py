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
from contextlib import ExitStack
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_s2_diagnosis import (
    MEMORY_BYTES,
    SUPERVISOR_MEMORY_BYTES,
    WORKER_MEMORY_BYTES,
    THREAD_ENV,
    TIMEOUT,
    safe,
    file_hash,
    write_json,
    update_status,
    stop_child,
    interrupted,
)

ROOT = Path(__file__).resolve().parents[1]
PLAN = "实验文档/17-S3条件残差监督分布诊断方案.md"
PLAN_SHA = "c0e8ac08dbc2ec854fed4c55d416f00ff9d6f84629e8d5b0285139e058b361d1"
S2 = "实验文档/results/s2-source-exposure-20261006"
ADDITIONAL = {
    PLAN: PLAN_SHA,
    S2
    + "/summary.json": "cce4e01da46f7d25893f9250cfd8267fb269366589953c9ef5fc14742f57f48e",
    S2
    + "/status.json": "e1fb35dc79a35b5fd59e5d54ca106a2082d31e9420a21974bc64d611b62cd8cb",
    S2
    + "/verification.json": "6fb51bd97947adfe39d1b3080d2fd656ed31f9fb306956c65b8a41ff56bab525",
    S2
    + "/protocol.json": "dbbf3eac6a9db3c7b92a04738e27332184d3a878202ffb09927d5357b34cbcc2",
}
OUTPUTS = (
    "protocol.json",
    "summary.json",
    "conditions.csv",
    "residuals.csv",
    "alignment.csv",
    "verification.json",
)
CODE_FILES = (
    "scripts/run_s3_diagnosis.py",
    "scripts/s3_conditional_analysis.py",
    "scripts/s3_verify.py",
    "scripts/s2_source_replay.py",
    "scripts/s2_exposure_stats.py",
    "scripts/run_s2_diagnosis.py",
    "scripts/run_s1_diagnosis.py",
)


def load_contract(root: Path) -> dict:
    from scripts import s2_source_replay as core

    contract = core.load_contract(root)
    for name, sha in ADDITIONAL.items():
        if file_hash(safe(root, root / name)) != sha:
            raise ValueError(f"Frozen S3/S2 evidence changed: {name}")
    status = json.loads((root / S2 / "status.json").read_text())
    verification = json.loads((root / S2 / "verification.json").read_text())
    if (
        status["status"] != "completed"
        or not status["replay_verified"]
        or not verification["source_rehashed_after"]
    ):
        raise ValueError("S3 requires certified S2 evidence")
    return {**contract, "input_sha256": {**contract["input_sha256"], **ADDITIONAL}}


def read_metadata(root: Path, db: str, frozen: dict) -> dict:
    import numpy as np
    from scripts import s2_source_replay as core

    data = core.read_cache(root, db, frozen)
    directory = root / "artifacts/cache/pretrain" / db
    offsets = np.load(safe(root, directory / "offsets.npy"), allow_pickle=False)
    if (offsets > np.iinfo(np.int64).max).any():
        raise ValueError("Offsets exceed supported integer range")
    counts = np.diff(offsets.astype(np.int64), axis=0)
    rows = json.loads(safe(root, directory / "manifest.json").read_text())["rows"]
    _, first, inverse = np.unique(
        [r["query_hash"] for r in rows], return_index=True, return_inverse=True
    )
    if not np.array_equal(counts, counts[first[inverse]]):
        raise ValueError("Duplicate identity has conflicting token counts")
    selected = counts[data["indices"]]
    if not np.array_equal(selected[:, 2], data["n_tables"]):
        raise ValueError("Table token counts disagree")
    return {**data, "token_counts": selected}


def check_denominators(root: Path, caches: dict) -> None:
    import numpy as np
    from scripts.s2_exposure_stats import CARDINALITY, TABLES

    old = json.loads((root / S2 / "summary.json").read_text())
    if set(caches) != set(old["source_pool"]):
        raise ValueError("S2 source denominator changed")
    for db, data in caches.items():
        previous = old["source_pool"][db]
        if (
            len(data["true"]) != previous["unique_queries"]
            or data["raw_rows"] != previous["raw_rows"]
        ):
            raise ValueError(f"S2 pool denominator changed: {db}")
    s1 = json.loads(
        (
            root / "实验文档/results/s1-source-diagnosis-20261006-verified/summary.json"
        ).read_text()
    )
    for db, report in s1["per_database"].items():
        data = caches[db]
        true, tables = data["true"], data["n_tables"]
        card = np.where(
            true == 0,
            0,
            np.searchsorted([10, 100, 1000, 10000, 100000], true, side="right") + 1,
        )
        table = np.clip(tables - 3, 0, 3).astype(int)
        masks = {
            **{f"all:true={label}": card == i for i, label in enumerate(CARDINALITY)},
            **{
                f"all:tables={'6plus' if label == '6+' else label}": table == i
                for i, label in enumerate(TABLES)
            },
        }
        masks = {
            **masks,
            **{
                f"all:true={c}|tables={'6plus' if t == '6+' else t}": (card == i)
                & (table == j)
                for i, c in enumerate(CARDINALITY)
                for j, t in enumerate(TABLES)
            },
        }
        for name, mask in masks.items():
            if int(mask.sum()) != report["strata"][name]["n"]:
                raise ValueError(f"S1 stratum denominator changed: {db}/{name}")


def validate_output(root: Path, output: Path) -> Path:
    safe(root / "runs", output)
    resolved = output.resolve()
    forbidden = (
        "precision-screen-20261006",
        "s1-source-diagnosis-20261006",
        "s1-source-diagnosis-20261006-verified",
        "s2-source-exposure-20261006",
    )
    if resolved == (root / "runs").resolve() or any(
        resolved.is_relative_to(root / "runs" / n) for n in forbidden
    ):
        raise ValueError("Output overlaps old inputs or runs root")
    if output.exists():
        raise FileExistsError("S3 requires a fresh output directory")
    return resolved


def write_chunk(writers: dict, streams: dict, chunk: dict) -> None:
    for name in ("conditions", "residuals", "alignment"):
        rows = chunk[name]
        if not rows:
            continue
        fields = sorted(rows[0])
        if name not in writers:
            writer = csv.DictWriter(
                streams[name], fieldnames=fields, lineterminator="\n"
            )
            writer.writeheader()
            writers[name] = writer
        if set(fields) != set(writers[name].fieldnames):
            raise ValueError("S3 CSV schema changed across strata")
        writers[name].writerows(rows)


def analyze(root: Path, output: Path) -> dict:
    import numpy as np
    from scripts import s2_source_replay as core, s3_conditional_analysis as stats
    from scripts.s3_verify import Verification

    started = time.monotonic()
    if np.__version__ != "1.26.1":
        raise ValueError("Formal S3 requires NumPy1.26.1")
    contract = load_contract(root)
    before = core.snapshot(root, contract)
    code_before = {name: file_hash(safe(root, root / name)) for name in CODE_FILES}
    write_json(
        output / "protocol.json",
        {
            "schema": 1,
            "mode": "source_metadata_conditional_diagnosis",
            "plan_sha256": PLAN_SHA,
            "input_sha256": before,
            "code_sha256": code_before,
            "train": contract["train"],
            "validation": contract["validation"],
            "condition": "FP64 PG bucket plus four exact token counts",
            "minimum_unique": 200,
            "training": False,
            "inference": False,
            "target_data_read": False,
            "timeout_seconds": TIMEOUT,
            "memory_limit_bytes": MEMORY_BYTES,
            "supervisor_address_limit_bytes": SUPERVISOR_MEMORY_BYTES,
            "worker_address_limit_bytes": WORKER_MEMORY_BYTES,
            "directory_trust": "User-owned frozen directories must not be concurrently modified; hashes do not defend against malicious same-privilege substitution.",
        },
    )
    caches = {
        db: read_metadata(root, db, contract["sources"][db])
        for db in contract["train"] + contract["validation"]
    }
    check_denominators(root, caches)
    update_status(output, "running", stage="sampling certification")
    replay = core.replay(caches, contract["train"], contract["config"])
    core.certify(replay)
    verifier = Verification(caches, replay, contract["train"], contract["validation"])
    alignment = []
    with ExitStack() as stack:
        streams = {
            name: stack.enter_context(
                (output / (name + ".csv")).open("x", encoding="utf-8", newline="")
            )
            for name in ("conditions", "residuals", "alignment")
        }
        writers = {}
        for chunk in stats.iter_analysis(
            caches, replay, contract["train"], contract["validation"]
        ):
            verifier.check(chunk)
            write_chunk(writers, streams, chunk)
            alignment.extend(chunk["alignment"])
            update_status(
                output,
                "running",
                stage=f"{chunk['axis']}:{chunk['bucket']}",
                replay_verified=True,
            )
    verification = verifier.finish()
    if set(writers) != {"conditions", "residuals", "alignment"}:
        raise ValueError("Incomplete S3 CSV outputs")
    after = core.snapshot(root, contract)
    if (
        after != before
        or {name: file_hash(safe(root, root / name)) for name in CODE_FILES}
        != code_before
    ):
        raise ValueError("S3 sources or execution code changed")
    write_json(
        output / "verification.json",
        {
            **verification,
            "source_rehashed_after": True,
            "code_rehashed_after": True,
            "s2_pool_denominators_reproduced": True,
            "s1_strata_denominators_reproduced": True,
            "sampling_digest": replay["digest"],
            "sampling_certification": "Exact full-sequence digest and historical counters; no checkpoint RNG read",
        },
    )
    final = replay["prefixes"][4000]
    summary = {
        "schema": 1,
        "purpose": "S3 recorded source-only conditional residual diagnosis, not causality or model selection",
        "input_sha256": before,
        "code_sha256": code_before,
        "replay_verified": True,
        "causal_explanation_verified": False,
        "training": False,
        "inference": False,
        "adaptation": False,
        "target_data_read": False,
        "database_count": False,
        "sample_plan_digest": replay["digest"],
        "episodes": final["episodes"],
        "presentations": final["presentations"],
        "nominal_mass": sum(
            float(v["mass"].sum()) for v in final["databases"].values()
        ),
        "source_pool": {
            db: {"raw_rows": d["raw_rows"], "unique_queries": len(d["true"])}
            for db, d in caches.items()
        },
        "row_counts": verification.get("rows"),
        "alignment": alignment,
        "definitions": {
            "r_ref": "FP64 log(max(true,1))-log(max(pg,1))",
            "r_fp32": "Separate FP64 logs cast FP32 then promoted before subtraction",
            "label_strata": "Posthoc true-cardinality groups are not deployable input conditions",
            "support": "Unique and sampled-unique >=200 per condition; absent cells and sparse denominators retained",
            "nominal_mass": "Not actual gradient contribution",
            "conditions": "PG bucket and counts only, not complete features or SQL equivalence",
            "standardization": "Validation condition mass, independent of residual values; no learned calibration",
        },
        "completed_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "runtime": {
            "python": sys.version,
            "numpy": np.__version__,
            "seconds": time.monotonic() - started,
            "peak_rss_kib_linux": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "threads": {n: os.environ.get(n) for n in THREAD_ENV},
            "timeout_seconds": TIMEOUT,
            "address_limit_bytes": MEMORY_BYTES,
        },
    }
    write_json(output / "summary.json", summary)
    write_json(
        output / "output-manifest.json",
        {n: file_hash(safe(output, output / n)) for n in OUTPUTS},
    )
    return summary


def verify_outputs(output: Path) -> dict:
    manifest = json.loads(safe(output, output / "output-manifest.json").read_text())
    if set(manifest) != set(OUTPUTS):
        raise ValueError("Incomplete S3 output manifest")
    for name, sha in manifest.items():
        if file_hash(safe(output, output / name)) != sha:
            raise ValueError("S3 output failed supervisor verification")
    return manifest


def supervise(command: list, output: Path, timeout: float = TIMEOUT) -> None:
    started = time.monotonic()
    token = secrets.token_hex(16)
    write_json(output / "supervision.json", {"pid": os.getpid(), "token": token})
    env = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": "",
        "S3_PARENT": str(os.getpid()),
        "S3_TOKEN": token,
        **{k: "1" for k in THREAD_ENV},
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    previous = signal.signal(signal.SIGTERM, interrupted)
    child = None
    try:
        if timeout <= 0:
            raise TimeoutError("S3 budget exhausted before worker launch")
        child = subprocess.Popen(command, start_new_session=True, env=env)
        code = child.wait(timeout=max(0.001, timeout - (time.monotonic() - started)))
        current = json.loads((output / "status.json").read_text())
        if code != 0 or current["status"] != "outputs_ready":
            raise RuntimeError(
                f"S3 worker did not complete: exit={code}, status={current['status']}"
            )
        hashes = verify_outputs(output)
        if time.monotonic() - started >= timeout:
            raise TimeoutError("S3 budget exhausted during output verification")
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
        cleanup_error = None
        if child is not None:
            try:
                stop_child(child)
            except BaseException as cleanup:
                cleanup_error = f"{type(cleanup).__name__}: {cleanup}"
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
            cleanup_error=cleanup_error,
            seconds=time.monotonic() - started,
        )
        if isinstance(error, (subprocess.TimeoutExpired, KeyboardInterrupt)):
            raise TimeoutError("S3 interrupted or exceeded 30-minute bound") from error
        raise
    finally:
        signal.signal(signal.SIGTERM, previous)


def worker(root: Path, output: Path) -> None:
    if (
        not sys.platform.startswith("linux")
        or os.environ.get("CUDA_VISIBLE_DEVICES") != ""
    ):
        raise RuntimeError("Formal S3 requires Linux CPU-only execution")
    if any(os.environ.get(k) != "1" for k in THREAD_ENV):
        raise ValueError("S3 requires single-thread numeric libraries")
    safe(root / "runs", output)
    if {p.name for p in output.iterdir()} != {"status.json", "supervision.json"}:
        raise ValueError("Worker requires fresh supervised output")
    parent = int(os.environ.get("S3_PARENT", "0"))
    owner = json.loads(safe(output, output / "supervision.json").read_text())
    if parent != os.getppid() or owner != {
        "pid": parent,
        "token": os.environ.get("S3_TOKEN"),
    }:
        raise ValueError("Worker ownership mismatch")
    if (
        ctypes.CDLL(None, use_errno=True).prctl(1, signal.SIGTERM) != 0
        or os.getppid() != parent
    ):
        raise RuntimeError("Could not bind worker to supervisor")
    resource.setrlimit(resource.RLIMIT_AS, (WORKER_MEMORY_BYTES, WORKER_MEMORY_BYTES))
    from scripts.s2_source_replay import ReplayMismatch

    try:
        update_status(output, "running", stage="input verification")
        analyze(root, output)
        update_status(output, "outputs_ready", replay_verified=True)
    except Exception as error:
        update_status(
            output,
            "blocked_replay_unverified"
            if isinstance(error, ReplayMismatch)
            else "failed",
            error=f"{type(error).__name__}: {error}",
        )
        raise


def run_main(deadline: float) -> None:
    if sys.platform.startswith("linux"):
        limit = (
            WORKER_MEMORY_BYTES if "--worker" in sys.argv else SUPERVISOR_MEMORY_BYTES
        )
        resource.setrlimit(resource.RLIMIT_AS, (limit, MEMORY_BYTES))
    parser = argparse.ArgumentParser(
        description="CPU-only S3 conditional metadata diagnosis"
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
                    "purpose": "S3 dry-run, no inputs opened or analysis executed",
                    "output": str(output),
                    "timeout_seconds": TIMEOUT,
                    "address_space_bytes": MEMORY_BYTES,
                }
            )
        )
        return
    if not sys.platform.startswith("linux"):
        raise RuntimeError("Formal S3 requires server Linux runtime")
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
        deadline - time.monotonic(),
    )


def deadline_expired(_signal: int, _frame: object) -> None:
    raise TimeoutError("S3 absolute 30-minute deadline reached")


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
