import argparse
import json
import os
import platform
import secrets
import signal
import subprocess
import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from scripts import run_s4_anchor as anchor
from scripts import s6_tail_evidence as verifier
from scripts.s6_loss_verify import verify_loss_evidence
from scripts.precision_screen_evidence import (
    COVERAGE,
    SOURCE_BINDING,
    checked_path,
    historical_sources,
    source_evidence,
    write_json,
)
from scripts.run_mapping_ablation import configure_determinism
from spatial_meta_ce.cli import data_settings, read_json
from spatial_meta_ce.data import StorePool
from spatial_meta_ce.manifests import file_hash

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/s6_tail.json"
PROTOCOL = "实验文档/20-S6平均与上尾混合损失单因素方案.md"
PROTOCOL_HASH = "9420a8e98664eb037ea95fa27d1e198241c99e624eeeb854253bd4d387b6c96e"
BASELINE = "实验文档/results/s5-complex-weighting-20261007/summary.json"
BASELINE_HASH = "f944da91da44faf1216508f3b224c3c31d340615910ef975024bc55beeb09106"
ARMS = ("A_mean", "B_mean_tail10")
TIMEOUT = 21600
FROZEN = dict(anchor.FROZEN)
MODEL = {**anchor.MODEL, "pg_output_scale": 1.0}
SPECS = {
    ARMS[0]: {"tail_risk_mode": "mean"},
    ARMS[1]: {"tail_risk_mode": "mean_tail10"},
}


def load_plan(config: Path | None = None) -> dict:
    expected = {
        "schema": 1,
        "protocol_sha256": PROTOCOL_HASH,
        "data_config": "configs/data.json",
        "source_binding_sha256": SOURCE_BINDING,
        "timeout_seconds": TIMEOUT,
        "training": FROZEN,
        "model": MODEL,
        "arms": SPECS,
    }
    path = CONFIG if config is None else config
    if read_json(checked_path(path.parent, path)) != expected:
        raise ValueError("Frozen S6 matrix changed")
    for name, sha in {
        PROTOCOL: PROTOCOL_HASH,
        "configs/data.json": anchor.precision.DATA_HASH,
    }.items():
        if file_hash(checked_path(ROOT, ROOT / name)) != sha:
            raise ValueError("Frozen S6 protocol or split changed")
    return {
        **expected,
        "arms": {
            a: {
                **FROZEN,
                **spec,
                "model": dict(MODEL),
                "data_config": "configs/data.json",
                "purpose": "research",
            }
            for a, spec in SPECS.items()
        },
        "splits": read_json(ROOT / "configs/data.json")["splits"],
    }


def validate_data(data: dict, plan: dict) -> None:
    if data["splits"] != plan["splits"] or Path(data["data_root"]).resolve() != Path(
        "/gly/weilin/PRICE-master"
    ):
        raise ValueError("Frozen source split or PRICE root changed")
    checked_path(Path(data["data_root"]), Path(data["data_root"]))
    if Path(data["cache_root"]).resolve() != (ROOT / "artifacts/cache").resolve():
        raise ValueError("Frozen cache root changed")
    checked_path(ROOT, Path(data["cache_root"]))


def frozen_inputs() -> dict:
    bindings = {**anchor.BINDINGS, BASELINE: BASELINE_HASH, PROTOCOL: PROTOCOL_HASH}
    for name, sha in bindings.items():
        if file_hash(checked_path(ROOT, ROOT / name)) != sha:
            raise ValueError(f"Frozen evidence changed: {name}")
    return bindings


def code_hashes() -> dict:
    return {
        str(p.relative_to(ROOT)): file_hash(checked_path(ROOT, p))
        for folder in ("scripts", "spatial_meta_ce", "vendor")
        for p in sorted((ROOT / folder).rglob("*.py"))
        if "__pycache__" not in p.parts
    }


def preflight(data: dict, index: str) -> tuple:
    import sqlglot

    uuid = anchor.gpu_gate("cuda", index, initial=True)
    runtime = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "sqlglot": sqlglot.__version__,
    }
    if runtime != anchor.DEPS:
        raise ValueError("Frozen dependency environment changed")
    inputs = frozen_inputs()
    databases = data["splits"]["train"] + data["splits"]["validation"]
    historical = historical_sources(ROOT, databases)
    sources = source_evidence(data, historical)
    old = read_json(ROOT / (anchor.FORMAL + "preflight.json"))
    if sources != old["sources"] or len(sources) != 26:
        raise ValueError("Frozen 182-file source evidence changed")
    return (
        {
            **runtime,
            "python": sys.version,
            "gpu_uuid": uuid,
            "gpu_index": index,
            "process_id": os.getpid(),
            "nvml_process_id": anchor.own_nvml_pid(),
            "protocol_sha256": PROTOCOL_HASH,
            "config_sha256": file_hash(CONFIG),
            "source_binding": SOURCE_BINDING,
            "code_sha256": code_hashes(),
            "input_sha256": inputs,
            "sources": sources,
            "smoke": False,
        },
        historical,
    )


def loss_stores(data: dict) -> dict:
    pool = StorePool(data["cache_root"], max_open=23)
    stores = {}
    for db in data["splits"]["train"]:
        store = pool.get("pretrain", db)
        stores[db] = {
            "query_hashes": [r["query_hash"] for r in store.rows],
            "unique": store.unique_indices(),
            "tables": np.diff(store.offsets[:, 2]),
            "labels": np.log(np.maximum(store.cards[:, 0], 1))
            .astype(np.float32)
            .astype(np.float64),
        }
    return stores


def status(output: Path, state: str, arms: dict, **extra) -> None:
    anchor.status(output, state, arms, **extra)


def execute_arms(
    configs: dict,
    data: dict,
    output: Path,
    evidence: dict,
    historical: dict,
    index: str,
    smoke: bool = False,
) -> dict:
    states = dict.fromkeys(ARMS, "not_started")
    records, checks = [], []
    output.mkdir(exist_ok=True)
    stores = None
    for arm in ARMS:
        states = {**states, arm: "running"}
        status(output, "running", states)
        baseline_check = False
        try:
            anchor.gpu_gate("cuda", index, expected_uuid=evidence["gpu_uuid"])
            if source_evidence(data, historical) != evidence["sources"]:
                raise ValueError("Source inputs changed before arm")
            record = anchor.run_one(
                arm,
                configs[arm],
                data,
                output / arm,
                "cuda",
                None if smoke else COVERAGE,
                smoke,
            )
            if stores is None:
                stores = loss_stores(data)
            loss_check = anchor.require_passed(
                verify_loss_evidence(output / arm, configs[arm], stores)
            )
            if source_evidence(data, historical) != evidence["sources"]:
                raise ValueError("Source inputs changed during arm")
            record = {
                **record,
                "gpu_uuid": evidence["gpu_uuid"],
                "loss_verification": loss_check,
            }
            write_json(output / arm / "loss-verification.json", loss_check)
            write_json(output / arm / "result.json", record)
            checks.append(
                anchor.require_passed(
                    verifier.verify_run(record, evidence, formal=not smoke)
                )
            )
            records.append(record)
            write_json(output / "runs.json", {"runs": records})
            if not smoke and arm == ARMS[0]:
                baseline_check = True
                checks.append(
                    anchor.require_passed(
                        verifier.verify_baseline(record, read_json(ROOT / BASELINE))
                    )
                )
                baseline_check = False
            states = {**states, arm: "verified"}
            status(output, "running", states)
        except BaseException as error:
            status(
                output,
                "reproduction_blocked" if baseline_check else "failed",
                {**states, arm: "failed"},
                error=f"{type(error).__name__}: {error}",
            )
            raise
    checks.append(
        anchor.require_passed(verifier.verify_pair(records, formal=not smoke))
    )
    report = {
        "purpose": "gpu-smoke-not-research"
        if smoke
        else "research-source-only-single-seed",
        "runs": records,
        **({} if smoke else verifier.choose(records)),
    }
    write_json(output / "summary.json", report)
    write_json(output / "verification.json", {"passed": True, "checks": checks})
    return report


def verify_record_files(output: Path) -> None:
    fields = {
        "checkpoint_sha256": "last.pt",
        "endpoint_rows_sha256": "endpoint_rows.jsonl",
        "initial_endpoint_rows_sha256": "initial_endpoint_rows.jsonl",
        "validation_plan_hash": "validation_manifest.json",
    }
    try:
        for prefix in ("", "smoke/"):
            records = read_json(checked_path(output, output / f"{prefix}runs.json"))[
                "runs"
            ]
            if [r["arm"] for r in records] != list(ARMS):
                raise ValueError("Incomplete S6 verified arm matrix")
            for record in records:
                directory = output / prefix / record["arm"]
                result = read_json(checked_path(output, directory / "result.json"))
                for field, name in fields.items():
                    actual = file_hash(checked_path(output, directory / name))
                    if actual != record[field] or actual != result[field]:
                        raise ValueError("Verified record file hash changed")
                for field, name in (
                    ("audit_sha256", "loss-audit.jsonl"),
                    ("snapshots_sha256", "loss-snapshots.jsonl"),
                ):
                    actual = file_hash(checked_path(output, directory / name))
                    if (
                        actual != record["loss_verification"][field]
                        or actual != result["loss_verification"][field]
                    ):
                        raise ValueError("Verified loss evidence file hash changed")
    except (KeyError, TypeError) as error:
        raise ValueError("Incomplete verified record hashes") from error


def verify_content(output: Path) -> None:
    plan = load_plan()
    if read_json(output / "protocol.json") != plan:
        raise ValueError("Supervisor protocol mismatch")
    evidence = read_json(output / "preflight.json")
    if (
        evidence["input_sha256"] != frozen_inputs()
        or evidence["code_sha256"] != code_hashes()
    ):
        raise ValueError("Supervisor input or code mismatch")
    data = data_settings(plan["arms"][ARMS[0]], CONFIG)
    validate_data(data, plan)
    historical = historical_sources(
        ROOT, data["splits"]["train"] + data["splits"]["validation"]
    )
    if source_evidence(data, historical) != evidence["sources"]:
        raise ValueError("Supervisor source evidence mismatch")
    stores = loss_stores(data)
    for prefix, formal in (("smoke/", False), ("", True)):
        folder = output / prefix
        records = read_json(folder / "runs.json")["runs"]
        summary = read_json(folder / "summary.json")
        if summary["runs"] != records:
            raise ValueError("Supervisor summary/record mismatch")
        for record in records:
            directory = folder / record["arm"]
            if read_json(directory / "result.json") != record:
                raise ValueError("Supervisor result/record mismatch")
            checked_loss = verify_loss_evidence(directory, record["config"], stores)
            if checked_loss != record["loss_verification"] or checked_loss != read_json(
                directory / "loss-verification.json"
            ):
                raise ValueError("Supervisor loss content verification mismatch")
            for initial in (False, True):
                name = "initial_endpoint" if initial else "endpoint"
                rows = {db: [] for db in anchor.VALIDATION}
                with checked_path(
                    output, directory / f"{name}_rows.jsonl"
                ).open() as stream:
                    for line in stream:
                        row = json.loads(line)
                        rows[row["database"]].append(row)
                anchor.require_passed(
                    verifier.verify_rows(rows, 1.0, record[name], formal=formal)
                )
            _, checkpoint = anchor.load_checkpoint(directory / "last.pt", "cpu")
            signature = {
                k: v
                for k, v in record["config"].items()
                if k not in ("steps", "data_config")
            }
            if (
                checkpoint.get("next_step") != record["config"]["steps"]
                or checkpoint["metadata"]["training_signature"] != signature
                or checkpoint["extra"]["gradient_stats"]
                != record["summary"]["gradient_stats"]
                or checkpoint["extra"]["counts"] != record["counts"]
                or checkpoint["extra"]["loss_evidence"]
                != record["summary"]["loss_evidence"]
            ):
                raise ValueError("Supervisor checkpoint content mismatch")
            anchor.require_passed(verifier.verify_run(record, evidence, formal))
        anchor.require_passed(verifier.verify_pair(records, formal))
        if formal:
            anchor.require_passed(
                verifier.verify_baseline(records[0], read_json(ROOT / BASELINE))
            )
            decision = verifier.choose(records)
            if any(summary.get(key) != value for key, value in decision.items()):
                raise ValueError("Supervisor candidate decision mismatch")


def verify_outputs(output: Path) -> dict:
    hashes = read_json(checked_path(output, output / "output-manifest.json"))
    names = {
        str(p.relative_to(output))
        for p in output.rglob("*")
        if p.is_file() and p.name not in ("status.json", "output-manifest.json")
    }
    required = {
        "protocol.json",
        "preflight.json",
        "summary.json",
        "runs.json",
        "verification.json",
        "smoke/summary.json",
        "smoke/verification.json",
    }
    required |= {
        f"{prefix}{arm}/{name}"
        for prefix in ("", "smoke/")
        for arm in ARMS
        for name in (
            "last.pt",
            "result.json",
            "endpoint_rows.jsonl",
            "initial_endpoint_rows.jsonl",
            "validation_manifest.json",
            "loss-verification.json",
            "loss-audit.jsonl",
            "loss-snapshots.jsonl",
        )
    }
    if set(hashes) != names or not required <= names:
        raise ValueError("Incomplete S6 output manifest")
    for name, sha in hashes.items():
        if file_hash(checked_path(output, output / name)) != sha:
            raise ValueError("Supervisor output hash mismatch")
    anchor.require_passed(read_json(output / "verification.json"))
    verify_record_files(output)
    verify_content(output)
    return hashes


def deadline_expired(_signal: int, _frame: object) -> None:
    raise TimeoutError("S6 absolute six-hour deadline reached")


def worker(output: Path, index: str) -> None:
    owner = read_json(checked_path(ROOT / "runs", output / "supervision.json"))
    parent = os.getppid()
    if owner != {"pid": parent, "token": os.environ.get("S6_TOKEN")} or os.environ.get(
        "S6_PARENT"
    ) != str(parent):
        raise ValueError("Worker requires its owning S6 supervisor")
    if {p.name for p in output.iterdir()} != {
        "status.json",
        "supervision.json",
    } or read_json(output / "status.json")["status"] != "planned":
        raise ValueError("Worker requires fresh supervised output")
    anchor.precision.bind_parent_lifetime(parent)
    previous = signal.signal(signal.SIGALRM, deadline_expired)
    remaining = float(os.environ["S6_DEADLINE"]) - time.monotonic()
    if remaining <= 0:
        raise TimeoutError("S6 deadline exhausted")
    signal.setitimer(signal.ITIMER_REAL, remaining)
    try:
        plan = load_plan()
        data = data_settings(plan["arms"][ARMS[0]], CONFIG)
        validate_data(data, plan)
        evidence, historical = preflight(data, index)
        write_json(output / "preflight.json", evidence)
        write_json(output / "protocol.json", plan)
        configure_determinism()
        execute_arms(
            anchor.smoke_configs(plan["arms"]),
            data,
            output / "smoke",
            evidence,
            historical,
            index,
            True,
        )
        execute_arms(plan["arms"], data, output, evidence, historical, index)
        if (
            source_evidence(data, historical) != evidence["sources"]
            or code_hashes() != evidence["code_sha256"]
            or frozen_inputs() != evidence["input_sha256"]
        ):
            raise ValueError("Sources or code changed after execution")
        anchor.output_manifest(output)
        status(output, "outputs_ready", dict.fromkeys(ARMS, "verified"))
    except BaseException as error:
        current = read_json(output / "status.json")
        status(
            output,
            "reproduction_blocked"
            if current.get("status") == "reproduction_blocked"
            else "failed",
            current.get("arms", dict.fromkeys(ARMS, "not_started")),
            error=f"{type(error).__name__}: {error}",
        )
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def supervise(command: list, output: Path, deadline: float) -> None:
    token = secrets.token_hex(16)
    write_json(output / "supervision.json", {"pid": os.getpid(), "token": token})
    child = None
    previous = signal.signal(signal.SIGTERM, anchor.precision.interrupted)
    try:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("S6 budget exhausted before worker")
        child = subprocess.Popen(
            command,
            start_new_session=True,
            env={
                **os.environ,
                "S6_PARENT": str(os.getpid()),
                "S6_TOKEN": token,
                "S6_DEADLINE": str(deadline),
            },
        )
        code = child.wait(timeout=remaining)
        current = read_json(output / "status.json")
        if code != 0 or current["status"] != "outputs_ready":
            raise RuntimeError(f"S6 worker did not complete: {code}")
        hashes = verify_outputs(output)
        if time.monotonic() >= deadline:
            raise TimeoutError("S6 deadline exhausted during output verification")
        status(
            output, "completed", dict.fromkeys(ARMS, "completed"), output_sha256=hashes
        )
    except BaseException as error:
        signal.setitimer(signal.ITIMER_REAL, 0)
        cleanup_error = None
        if child is not None:
            try:
                anchor.stop_child(child)
            except BaseException as cleanup:
                cleanup_error = f"{type(cleanup).__name__}: {cleanup}"
        current = read_json(output / "status.json")
        states = {
            a: "partial" if v == "running" else v
            for a, v in current.get("arms", dict.fromkeys(ARMS, "not_started")).items()
        }
        state = (
            "failed"
            if cleanup_error
            else "reproduction_blocked"
            if current["status"] == "reproduction_blocked"
            else "partial"
            if isinstance(
                error, (TimeoutError, subprocess.TimeoutExpired, KeyboardInterrupt)
            )
            else "failed"
        )
        status(
            output,
            state,
            states,
            error=f"{type(error).__name__}: {error}",
            worker_error=current.get("worker_error", current.get("error")),
            cleanup_error=cleanup_error,
        )
        raise
    finally:
        signal.signal(signal.SIGTERM, previous)


def run_main(deadline: float) -> None:
    parser = argparse.ArgumentParser(
        description="Frozen source-only S6 mean/upper-tail loss contrast"
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--gpu-index")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        worker(Path(args.output), args.gpu_index)
        return
    plan = load_plan()
    data = data_settings(plan["arms"][ARMS[0]], CONFIG)
    validate_data(data, plan)
    output = anchor.precision.validate_output(Path(args.output), data, ROOT)
    if not args.execute:
        print(
            json.dumps(
                {"purpose": "dry-run-not-execution", "output": str(output), **plan},
                indent=2,
            )
        )
        return
    if not sys.platform.startswith("linux"):
        raise RuntimeError("Formal S6 requires Linux CUDA execution")
    output.mkdir(parents=True)
    status(output, "planned", dict.fromkeys(ARMS, "not_started"))
    try:
        anchor.gpu_gate(args.device, args.gpu_index, initial=True)
    except anchor.GPUBusy as error:
        status(
            output,
            "deferred_gpu_busy",
            dict.fromkeys(ARMS, "not_started"),
            error=str(error),
        )
        return
    except BaseException as error:
        status(
            output,
            "failed",
            dict.fromkeys(ARMS, "not_started"),
            error=f"{type(error).__name__}: {error}",
        )
        raise
    supervise(
        [
            sys.executable,
            "-B",
            str(Path(__file__).resolve()),
            "--execute",
            "--worker",
            "--output",
            str(output),
            "--gpu-index",
            args.gpu_index,
        ],
        output,
        deadline,
    )


def main() -> None:
    deadline = time.monotonic() + TIMEOUT
    parent = "--execute" in sys.argv and "--worker" not in sys.argv
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
