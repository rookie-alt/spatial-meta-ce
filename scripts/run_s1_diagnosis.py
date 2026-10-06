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

import hashlib

ROOT = Path(__file__).resolve().parents[1]
FORMAL_NAME = "precision-screen-20261006"
ARMS = ("A_w64_smooth", "B_w64_mse", "C_w128_smooth", "D_w128_mse")
DATABASES = ("chembl", "legalacts", "telstra")
EXPECTED = {
    "summary.json": "3fd40103ba5ce19c93c3fa7861a5012c902a4237da5ba7502b5add7b04eae78a",
    "status.json": "eb535f744e7006fdbf9c6c031314bbda2936d1a12b6d4d801d791431822fcc99",
    "preflight.json": "3b037edd777a7e39259033b020ffff8015cbb4cd6bd169ab8453b16bc5d8bfe8",
}
COVERAGE = {
    "chembl": (50000, 33947),
    "legalacts": (49504, 9178),
    "telstra": (50000, 20311),
}
TIMEOUT = 1800
MEMORY_BYTES = 4 * 1024**3
SUPERVISOR_MEMORY_BYTES = 256 * 1024**2
WORKER_MEMORY_BYTES = MEMORY_BYTES - SUPERVISOR_MEMORY_BYTES
THREAD_ENV = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def safe(root: Path, path: Path) -> Path:
    if ".." in path.parts or not path.resolve().is_relative_to(root.resolve()):
        raise ValueError("S1 input path escaped its frozen root")
    if any(p.is_symlink() for p in (path.absolute(), *path.absolute().parents)):
        raise ValueError("S1 refuses symlink paths")
    return path


def file_hash(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def validate_output(path: Path, root: Path) -> Path:
    safe(root / "runs", path)
    resolved, formal = path.resolve(), (root / "runs" / FORMAL_NAME).resolve()
    if (
        resolved == (root / "runs").resolve()
        or resolved.is_relative_to(formal)
        or formal.is_relative_to(resolved)
    ):
        raise ValueError("S1 output overlaps input or runs root")
    if path.exists():
        raise FileExistsError("Use a fresh S1 output directory")
    return resolved


def write_json(path: Path, value: dict) -> None:
    with path.open("x") as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + "\n")


def update_status(output: Path, state: str, **values) -> None:
    temporary = output / "status.tmp"
    temporary.write_text(
        json.dumps(
            {
                "status": state,
                "updated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                **values,
            },
            indent=2,
        )
    )
    temporary.replace(output / "status.json")


def load_contract(root: Path) -> dict:
    directory = root / "实验文档/results" / FORMAL_NAME
    records = {}
    for name, sha in EXPECTED.items():
        path = safe(root, directory / name)
        if file_hash(path) != sha:
            raise ValueError(f"Frozen formal evidence changed: {name}")
        records[name[:-5]] = json.loads(path.read_text())
    summary, status = records["summary"], records["status"]
    if (
        status["status"] != "completed"
        or status["smoke"]
        or summary["decision"] != "no_candidate"
    ):
        raise ValueError("S1 requires the completed frozen source screen")
    runs = summary["runs"]
    if len(runs) != 4 or [r["arm"] for r in runs] != list(ARMS):
        raise ValueError("Formal matrix changed")
    for r in runs:
        if (
            r["optimizer_steps"] != 4000
            or r["config"]["seed"] != 42
            or r["config"]["method"] != "erm"
        ):
            raise ValueError("Formal training identity changed")
        for db in DATABASES:
            if (
                r["manifest_hashes"][db]
                != records["preflight"]["sources"][db]["manifest_hash"]
            ):
                raise ValueError("Prediction and frozen source hashes disagree")
    return {**records, "runs": {r["arm"]: r for r in runs}}


def snapshot(root: Path, contract: dict) -> dict:
    hashes = {
        f"实验文档/results/{FORMAL_NAME}/{name}": sha for name, sha in EXPECTED.items()
    }
    for arm, r in contract["runs"].items():
        relative = f"runs/{FORMAL_NAME}/{arm}/endpoint_rows.jsonl"
        path = safe(root, root / relative)
        if file_hash(path) != r["endpoint_rows_sha256"]:
            raise ValueError(f"Frozen predictions changed: {arm}")
        hashes[relative] = r["endpoint_rows_sha256"]
    for db in DATABASES:
        frozen = contract["preflight"]["sources"][db]
        for name in ("manifest.json", "cards.npy", "offsets.npy"):
            sha = frozen["file_sha256"][name]
            relative = f"artifacts/cache/pretrain/{db}/{name}"
            path = safe(root, root / relative)
            if file_hash(path) != sha:
                raise ValueError(f"Frozen source cache changed: {db}/{name}")
            hashes[relative] = sha
    return hashes


def csv_output(output: Path, name: str, rows: list) -> None:
    columns = sorted({key for row in rows for key in row})
    with (output / name).open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def analyze(root: Path, output: Path) -> dict:
    import numpy as np

    from scripts import s1_source_analysis as core

    started = time.monotonic()
    contract = load_contract(root)
    before = snapshot(root, contract)
    caches = {
        db: core.read_cache(
            root / "artifacts/cache/pretrain" / db, contract["preflight"]["sources"][db]
        )
        for db in DATABASES
    }
    for db, data in caches.items():
        n, primary = COVERAGE[db]
        if len(data["true"]) != n or int((data["n_tables"] >= 4).sum()) != primary:
            raise ValueError("Frozen unique-query coverage changed")
    predictions = {
        a: core.read_predictions(
            root / "runs" / FORMAL_NAME / a / "endpoint_rows.jsonl",
            caches,
            contract["runs"][a]["endpoint_rows_sha256"],
        )
        for a in ARMS
    }
    per_database = {}
    for db, cache in caches.items():
        data = {**cache, "predictions": {a: predictions[a][db] for a in ARMS}}
        identity = digest(
            sorted(
                (db, h, float(t), int(n))
                for h, t, n in zip(data["query_hash"], data["true"], data["n_tables"])
            )
        )
        report = core.analyze_database(data)
        for arm in ARMS:
            recorded = contract["runs"][arm]["endpoint"]["per_database"][db]
            for bucket in ("all_tables", "four_plus_tables"):
                computed = report["global_metrics"][bucket][arm]
                if computed["n"] != recorded[bucket]["queries"]:
                    raise ValueError("Frozen endpoint denominator changed")
                for metric in ("q50", "q95"):
                    if not np.isclose(
                        computed[metric],
                        recorded[bucket][metric],
                        rtol=1e-12,
                        atol=1e-12,
                    ):
                        raise ValueError("Frozen endpoint failed reproduction")
        per_database[db] = {"identity_digest": identity, **report}
        update_status(
            output,
            "running",
            stage=f"{db}: analyzed",
            databases_completed=list(per_database),
        )
    if snapshot(root, contract) != before:
        raise ValueError("Inputs changed during S1")
    if load_contract(root)["summary"] != contract["summary"]:
        raise ValueError("Frozen evidence changed during S1")
    report = {
        "purpose": "S1 source-only posthoc recorded-prediction diagnosis; not selection or causality",
        "status": "analysis_complete_requires_supervisor_verification",
        "training": False,
        "inference": False,
        "adaptation": False,
        "target_data_read": False,
        "database_count": False,
        "completed_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "input_sha256": before,
        "per_database": per_database,
        "definitions": {
            "primary": "unchanged three-source equal arithmetic macro 4+ unique Q95",
            "official": "abs(max(log_pred,0)-log(max(true,1)))",
            "raw": "log_pred-log(max(true,1)); not lower-bounded prediction",
            "paired_delta": "candidate minus reference absolute official log error; below zero is improvement",
            "tail": "each per-db/per-bucket Q95 threshold, all ties included; A fixed tail is shared by B/C/D",
            "sparse": "n<200 counts only; cell Q99 only if n>=1000",
            "pg": "reference log in FP64; actual model PG input FP32 cast then promoted to FP64",
            "directory_trust": "User-owned frozen input/output directories must not be modified concurrently; hash checks detect ordinary changes, not malicious same-privilege directory substitution",
            "limitations": "one training seed, posthoc group analysis, no confidence interval or causal attribution; no model/threshold selection",
        },
        "checks": {
            "source_databases": list(DATABASES),
            "unique_queries": 149504,
            "four_plus_queries": 63436,
            "paired": True,
            "source_rehashed_after": True,
            "formal_q50_q95_reproduced": True,
        },
        "runtime": {
            "python": sys.version,
            "numpy": np.__version__,
            "seconds": time.monotonic() - started,
            "peak_rss_kib_linux": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "memory_limit_bytes": MEMORY_BYTES,
            "worker_address_limit_bytes": WORKER_MEMORY_BYTES,
            "supervisor_address_limit_bytes": SUPERVISOR_MEMORY_BYTES,
            "timeout_seconds": TIMEOUT,
            "threads": {k: os.environ.get(k) for k in THREAD_ENV},
        },
        "code_sha256": {
            p.name: file_hash(p)
            for p in (Path(__file__), root / "scripts/s1_source_analysis.py")
        },
    }
    report["macro_four_plus_q95"] = {
        a: float(
            np.mean(
                [
                    r["global_metrics"]["four_plus_tables"][a]["q95"]
                    for r in per_database.values()
                ]
            )
        )
        for a in ARMS
    }
    for a in ARMS:
        if not np.isclose(
            report["macro_four_plus_q95"][a],
            contract["runs"][a]["endpoint"]["macro_four_plus_q95"],
            rtol=1e-12,
        ):
            raise ValueError("Frozen macro changed")
    global_rows, strata_rows, tail_rows = [], [], []
    for db, r in per_database.items():
        for bucket, arms in r["global_metrics"].items():
            global_rows.extend(
                {"database": db, "bucket": bucket, "arm": a, **v}
                for a, v in arms.items()
            )
        for cell, info in r["strata"].items():
            if info["status"] != "descriptive":
                strata_rows.append({"database": db, "cell": cell, **info})
            else:
                strata_rows.extend(
                    {
                        "database": db,
                        "cell": cell,
                        "arm": a,
                        "status": info["status"],
                        "fraction_of_database": info["fraction_of_database"],
                        **v,
                    }
                    for a, v in info["arms"].items()
                )
        for bucket, pairs in r["tail_migration"].items():
            tail_rows.extend(
                {
                    "database": db,
                    "bucket": bucket,
                    "comparison": key,
                    **{
                        k: value
                        for k, value in stats.items()
                        if not isinstance(value, dict)
                    },
                }
                for key, stats in pairs.items()
            )
    write_json(output / "summary.json", report)
    csv_output(output, "global.csv", global_rows)
    csv_output(output, "strata.csv", strata_rows)
    csv_output(output, "tails.csv", tail_rows)
    write_json(
        output / "output-manifest.json",
        {
            p.name: file_hash(p)
            for p in output.iterdir()
            if p.name in ("summary.json", "global.csv", "strata.csv", "tails.csv")
        },
    )
    return report


def stop_child(child) -> None:
    deadline = time.monotonic() + 10
    try:
        os.killpg(child.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        child.wait(timeout=min(9, max(0.001, deadline - time.monotonic())))
    except subprocess.TimeoutExpired:
        try:
            os.killpg(child.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        child.wait(timeout=max(0.001, deadline - time.monotonic()))
    try:
        os.killpg(child.pid, 0)
    except ProcessLookupError:
        return
    os.killpg(child.pid, signal.SIGKILL)


def interrupted(_signal, _frame) -> None:
    raise KeyboardInterrupt("S1 supervisor interrupted")


def supervise(command: list, output: Path, timeout: int = TIMEOUT) -> None:
    started = time.monotonic()
    token = secrets.token_hex(16)
    write_json(output / "supervision.json", {"pid": os.getpid(), "token": token})
    env = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": "",
        "S1_PARENT": str(os.getpid()),
        "S1_TOKEN": token,
        **{name: "1" for name in THREAD_ENV},
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    previous = signal.signal(signal.SIGTERM, interrupted)
    child = None
    try:
        if timeout <= 0:
            raise TimeoutError("S1 budget exhausted before worker launch")
        child = subprocess.Popen(command, start_new_session=True, env=env)
        code = child.wait(timeout=max(0.001, timeout - (time.monotonic() - started)))
        current = json.loads((output / "status.json").read_text())
        if code != 0 or current["status"] != "outputs_ready":
            raise RuntimeError(f"S1 worker did not complete, exit={code}")
        manifest = json.loads(safe(output, output / "output-manifest.json").read_text())
        if set(manifest) != {"summary.json", "global.csv", "strata.csv", "tails.csv"}:
            raise ValueError("Incomplete S1 output manifest")
        for name, sha in manifest.items():
            if file_hash(safe(output, output / name)) != sha:
                raise ValueError("S1 output failed supervisor verification")
        if time.monotonic() - started >= timeout:
            raise TimeoutError("S1 budget exhausted during output verification")
        update_status(
            output,
            "completed",
            training=False,
            inference=False,
            target_data_read=False,
            seconds=time.monotonic() - started,
        )
    except BaseException as error:
        signal.setitimer(signal.ITIMER_REAL, 0)
        if child is not None:
            stop_child(child)
        update_status(
            output,
            "failed",
            error=f"{type(error).__name__}: {error}",
            seconds=time.monotonic() - started,
        )
        if isinstance(error, (subprocess.TimeoutExpired, KeyboardInterrupt)):
            raise TimeoutError("S1 interrupted or exceeded 30-minute bound") from error
        raise
    finally:
        signal.signal(signal.SIGTERM, previous)


def worker(root: Path, output: Path) -> None:
    if (
        not sys.platform.startswith("linux")
        or os.environ.get("CUDA_VISIBLE_DEVICES") != ""
    ):
        raise RuntimeError("Formal S1 requires Linux CPU-only execution")
    if any(os.environ.get(k) != "1" for k in THREAD_ENV):
        raise ValueError("S1 requires single-thread numeric libraries")
    safe(root / "runs", output)
    if output.resolve() == (root / "runs").resolve() or output.resolve().is_relative_to(
        root / "runs" / FORMAL_NAME
    ):
        raise ValueError("Worker output overlaps inputs")
    if {p.name for p in output.iterdir()} != {"status.json", "supervision.json"}:
        raise ValueError("Worker requires a fresh supervised directory")
    owner = json.loads(safe(output, output / "supervision.json").read_text())
    parent = int(os.environ.get("S1_PARENT", "0"))
    if parent != os.getppid() or owner != {
        "pid": parent,
        "token": os.environ.get("S1_TOKEN"),
    }:
        raise ValueError("Worker ownership mismatch")
    if (
        ctypes.CDLL(None, use_errno=True).prctl(1, signal.SIGTERM) != 0
        or os.getppid() != parent
    ):
        raise RuntimeError("Could not bind worker to its original supervisor")
    resource.setrlimit(resource.RLIMIT_AS, (WORKER_MEMORY_BYTES, WORKER_MEMORY_BYTES))
    try:
        update_status(output, "running", stage="input verification")
        analyze(root, output)
        update_status(
            output,
            "outputs_ready",
            training=False,
            inference=False,
            target_data_read=False,
        )
    except Exception as error:
        update_status(output, "failed", error=f"{type(error).__name__}: {error}")
        raise


def run_main(deadline: float) -> None:
    if sys.platform.startswith("linux"):
        limit = (
            WORKER_MEMORY_BYTES if "--worker" in sys.argv else SUPERVISOR_MEMORY_BYTES
        )
        resource.setrlimit(resource.RLIMIT_AS, (limit, MEMORY_BYTES))
    parser = argparse.ArgumentParser(
        description="CPU-only S1 recorded source prediction diagnosis"
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        worker(ROOT, Path(args.output))
        return
    load_contract(ROOT)
    if time.monotonic() >= deadline:
        raise TimeoutError("S1 preflight exhausted the total budget")
    output = validate_output(Path(args.output), ROOT)
    if not args.execute:
        print(
            json.dumps(
                {
                    "purpose": "S1 dry-run, no analysis executed",
                    "output": str(output),
                    "databases": DATABASES,
                    "timeout_seconds": TIMEOUT,
                    "memory_bytes": MEMORY_BYTES,
                }
            )
        )
        return
    if not sys.platform.startswith("linux"):
        raise RuntimeError("Formal S1 requires the server Linux runtime")
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


def deadline_expired(_signal, _frame) -> None:
    raise TimeoutError("S1 absolute 30-minute deadline reached")


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
