import argparse
import json
import os
import platform
import re
import secrets
import signal
import subprocess
import sys
import time
from pathlib import Path

# Dependency helpers are unchanged: this runner has its own S4 gates.
sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from scripts import run_precision_screen as precision
from scripts.precision_screen_evidence import (
    COVERAGE,
    SOURCE_BINDING,
    VALIDATION as VALIDATION,
    checked_path,
    endpoint,
    historical_sources,
    source_evidence,
    write_json,
)
from scripts.mapping_smoke import state_hash
from scripts.run_mapping_ablation import configure_determinism
from spatial_meta_ce.checkpoints import load_checkpoint
from spatial_meta_ce.cli import data_settings, read_json
from spatial_meta_ce.data import StorePool
from spatial_meta_ce.manifests import digest, file_hash
from spatial_meta_ce.model import SetEstimator
from spatial_meta_ce.train import seed_all, train

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/s4_anchor.json"
PROTOCOL = "实验文档/18-S4PG输出锚定单因素训练对照方案.md"
PROTOCOL_HASH = "d63c4f503f55eb87b59896e54f0379768e8829c8d48f2bad0a3f6224b53e0be3"
FORMAL = "实验文档/results/precision-screen-20261006/"
BINDINGS = {
    "configs/data.json": precision.DATA_HASH,
    FORMAL
    + "summary.json": "3fd40103ba5ce19c93c3fa7861a5012c902a4237da5ba7502b5add7b04eae78a",
    FORMAL
    + "preflight.json": "3b037edd777a7e39259033b020ffff8015cbb4cd6bd169ab8453b16bc5d8bfe8",
    "实验文档/results/s2-source-exposure-20261006/summary.json": "cce4e01da46f7d25893f9250cfd8267fb269366589953c9ef5fc14742f57f48e",
    "实验文档/results/s2-source-exposure-20261006/status.json": "e1fb35dc79a35b5fd59e5d54ca106a2082d31e9420a21974bc64d611b62cd8cb",
    "实验文档/results/s2-source-exposure-20261006/verification.json": "6fb51bd97947adfe39d1b3080d2fd656ed31f9fb306956c65b8a41ff56bab525",
    "实验文档/results/s2-source-exposure-20261006/protocol.json": "dbbf3eac6a9db3c7b92a04738e27332184d3a878202ffb09927d5357b34cbcc2",
}
ARMS = ("A_anchor1", "B_anchor0")
TIMEOUT = 21600
STOP_GRACE = 10
FROZEN = {**precision.FROZEN, "loss": "smooth_l1"}
MODEL = {**precision.MODEL, "width": 64}
SPECS = {ARMS[0]: {"pg_output_scale": 1.0}, ARMS[1]: {"pg_output_scale": 0.0}}
DEPS = {
    "python": "3.11.4",
    "torch": "2.1.0+cu121",
    "numpy": "1.26.1",
    "sqlglot": "18.17.0",
}
INITIAL_HASH = "41e594b03ef19c56c3f236237da850929f2ec5c919702b44402dd9b2864d284a"


class GPUBusy(RuntimeError):
    pass


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
    config_path = CONFIG if config is None else config
    if read_json(checked_path(config_path.parent, config_path)) != expected:
        raise ValueError("Frozen S4 matrix changed")
    for name, sha in {
        PROTOCOL: PROTOCOL_HASH,
        "configs/data.json": precision.DATA_HASH,
    }.items():
        if file_hash(checked_path(ROOT, ROOT / name)) != sha:
            raise ValueError("Frozen S4 protocol or source split changed")
    return {
        **expected,
        "arms": {
            a: {
                **FROZEN,
                "support_sizes": list(FROZEN["support_sizes"]),
                "model": {**MODEL, **spec},
                "data_config": "configs/data.json",
                "purpose": "research",
            }
            for a, spec in SPECS.items()
        },
        "splits": read_json(checked_path(ROOT, ROOT / "configs/data.json"))["splits"],
    }


def canonical_model(config: dict) -> dict:
    return {**config, "pg_output_scale": config.get("pg_output_scale", 1.0)}


def require_passed(value: dict) -> dict:
    if not isinstance(value, dict) or value.get("passed") is not True:
        raise ValueError("Independent verification did not pass")
    return value


def validate_data(data: dict, plan: dict) -> None:
    if data["splits"] != plan["splits"] or Path(data["data_root"]).resolve() != Path(
        "/gly/weilin/PRICE-master"
    ):
        raise ValueError("Frozen source split or PRICE root changed")
    checked_path(Path(data["data_root"]), Path(data["data_root"]))
    checked_path(ROOT, ROOT / "artifacts/cache")
    if Path(data["cache_root"]).resolve() != (ROOT / "artifacts/cache").resolve():
        raise ValueError("Frozen cache root changed")
    checked_path(ROOT, Path(data["cache_root"]))


def validate_output(path: Path, data: dict) -> Path:
    return precision.validate_output(path, data, ROOT)


def own_nvml_pid() -> int:
    try:
        fields = {
            line.partition(":")[0]: line.partition(":")[2].split()
            for line in Path("/proc/self/status").read_text().splitlines()
        }
        if fields.get("Pid") != [str(os.getpid())]:
            raise ValueError("Process metadata does not identify this process")
        header = Path("/proc/self/sched").read_text().splitlines()[0]
        match = re.search(r" \((\d+), #threads: \d+\)$", header)
        if match is None or int(match[1]) < 1:
            raise ValueError("Unknown kernel scheduler PID format")
        pid = int(match[1])
        if "NSpid" in fields:
            chain = [int(value) for value in fields["NSpid"]]
            if (
                not chain
                or chain[-1] != os.getpid()
                or any(value < 1 for value in chain)
            ):
                raise ValueError("Invalid PID namespace chain")
            if len(chain) > 1 and chain[0] != pid:
                raise ValueError("Kernel PID sources disagree")
        return pid
    except (OSError, ValueError, IndexError) as error:
        raise RuntimeError("Cannot authenticate own NVML process PID") from error


def gpu_gate(
    device: str,
    index: str,
    expected_uuid: str | None = None,
    initial: bool = False,
) -> str:
    if (
        device != "cuda"
        or not isinstance(index, str)
        or not index.isdecimal()
        or os.environ.get("CUDA_VISIBLE_DEVICES") != index
    ):
        raise ValueError("Formal S4 needs one explicit physical CUDA index")
    rows = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    apps = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    selected = [
        r.split(",") for r in rows.splitlines() if r.split(",")[0].strip() == index
    ]
    if len(selected) != 1 or len(selected[0]) != 4:
        raise ValueError("Selected GPU does not exist or identity is ambiguous")
    _, uuid, memory, utilization = [v.strip() for v in selected[0]]
    if expected_uuid is not None and uuid != expected_uuid:
        raise ValueError("GPU UUID changed")
    allowed = set() if initial else {own_nvml_pid()}
    for row in apps.splitlines():
        gpu, pid = [p.strip() for p in row.split(",")]
        if gpu == uuid and (initial or int(pid) not in allowed):
            raise GPUBusy("Selected GPU has external processes; no retry")
    if initial and (int(memory) > 64 or int(utilization) != 0):
        raise GPUBusy("Selected GPU is occupied; no retry")
    return uuid


def frozen_inputs() -> dict:
    for name, sha in BINDINGS.items():
        if file_hash(checked_path(ROOT, ROOT / name)) != sha:
            raise ValueError(f"Frozen historical evidence changed: {name}")
    return dict(BINDINGS)


def code_hashes() -> dict:
    return {
        str(p.relative_to(ROOT)): file_hash(checked_path(ROOT, p))
        for folder in ("scripts", "spatial_meta_ce", "vendor")
        for p in sorted((ROOT / folder).rglob("*.py"))
        if "__pycache__" not in p.parts
    }


def preflight(data: dict, index: str) -> tuple:
    import sqlglot

    uuid = gpu_gate("cuda", index, initial=True)  # before any CUDA initialization
    runtime = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "sqlglot": sqlglot.__version__,
    }
    if runtime != DEPS:
        raise ValueError("Frozen Python/torch/NumPy/sqlglot environment changed")
    inputs = frozen_inputs()
    databases = data["splits"]["train"] + data["splits"]["validation"]
    historical = historical_sources(ROOT, databases)
    sources = source_evidence(data, historical)
    old = read_json(checked_path(ROOT, ROOT / (FORMAL + "preflight.json")))
    if sources != old["sources"] or len(sources) != 26:
        raise ValueError("Full 182-file formal source evidence changed")
    return (
        {
            **runtime,
            "python": sys.version,
            "gpu_uuid": uuid,
            "gpu_index": index,
            "process_id": os.getpid(),
            "nvml_process_id": own_nvml_pid(),
            "protocol_sha256": file_hash(ROOT / PROTOCOL),
            "config_sha256": file_hash(CONFIG),
            "source_binding": SOURCE_BINDING,
            "code_sha256": code_hashes(),
            "input_sha256": inputs,
            "sources": sources,
            "smoke": False,
        },
        historical,
    )


def enrich_rows(rows_by_db: dict, pool, scale: float) -> dict:
    result = {}
    for db, rows in rows_by_db.items():
        store = pool.get("pretrain", db)
        indices = {
            store.rows[int(i)]["query_hash"]: int(i) for i in store.unique_indices()
        }
        enriched = []
        for row in rows:
            i = indices[row["query_hash"]]
            reference = float(np.log(max(float(store.cards[i, 1]), 1.0)))
            pg = float(np.float32(reference))
            anchor = scale * pg
            head = row["log_pred"] - anchor
            enriched.append(
                {
                    **row,
                    "pg_input_log": pg,
                    "pg_reference_log": reference,
                    "anchor": anchor,
                    "head_output": head,
                    "total_correction_vs_pg": head + (scale - 1) * pg,
                    "token_counts": [
                        int(v) for v in store.offsets[i + 1] - store.offsets[i]
                    ],
                }
            )
        result[db] = enriched
    return result


def pg_identity(rows: dict) -> str:
    return digest(
        [
            (
                db,
                r["query_hash"],
                r["true"],
                r["n_tables"],
                r["pg_input_log"],
                r["pg_reference_log"],
            )
            for db, group in rows.items()
            for r in group
        ]
    )


def save_rows(path: Path, rows: dict) -> str:
    with path.open("x", encoding="utf-8") as stream:
        for db, group in rows.items():
            for row in group:
                stream.write(
                    json.dumps({"database": db, **row}, allow_nan=False) + "\n"
                )
    return file_hash(path)


def run_one(
    arm: str,
    config: dict,
    data: dict,
    output: Path,
    device: str,
    expected_coverage: dict | None = COVERAGE,
    smoke: bool = False,
) -> dict:
    started = time.monotonic()
    seed_all(config["seed"])
    initial = SetEstimator(**config["model"]).to(device)
    initial_hash = state_hash(initial)
    parameters = sum(p.numel() for p in initial.parameters())
    if not smoke and (initial_hash != INITIAL_HASH or parameters != 132545):
        raise ValueError("Historical initial tensors or capacity changed")
    pool = StorePool(data["cache_root"], max_open=26)
    initial_endpoint, rows = endpoint(initial, pool, device, expected_coverage, smoke)
    initial_rows = enrich_rows(rows, pool, config["model"]["pg_output_scale"])
    if any(
        r["log_pred"] != config["model"]["pg_output_scale"] * r["pg_input_log"]
        for group in initial_rows.values()
        for r in group
    ):
        raise ValueError("Zero-head initial function changed")
    del initial
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    summary = train(config, data, output, device=device)
    model, state = load_checkpoint(checked_path(output, output / "last.pt"), device)
    extra = state["extra"]
    if (
        extra["counts"]["optimizer_steps"] != config["steps"]
        or state["next_step"] != config["steps"]
    ):
        raise ValueError("Frozen budget incomplete")
    signature = {k: v for k, v in config.items() if k not in ("steps", "data_config")}
    if (
        canonical_model(state["model_config"]) != canonical_model(config["model"])
        or state["metadata"]["training_signature"] != signature
    ):
        raise ValueError("Checkpoint does not match arm configuration")
    episodes = config["steps"] * config["meta_batch"]
    if (
        extra["sample_plan_episodes"] != episodes
        or extra["counts"]["backward_calls"] != episodes
        or extra["counts"]["support_presentations"] != 0
    ):
        raise ValueError("ERM episode or backward budget changed")
    pool = precision.checkpoint_pool(state, data)
    checks = precision.checkpoint_smoke(
        model, state, output, pool, device, initial_hash
    )
    final, rows = endpoint(model, pool, device, expected_coverage, smoke)
    rows = enrich_rows(rows, pool, config["model"]["pg_output_scale"])
    from scripts import s4_anchor_evidence as verifier

    row_checks = require_passed(
        verifier.verify_rows(
            rows, config["model"]["pg_output_scale"], final, formal=not smoke
        )
    )
    initial_checks = require_passed(
        verifier.verify_rows(
            initial_rows,
            config["model"]["pg_output_scale"],
            initial_endpoint,
            formal=not smoke,
        )
    )
    final = {
        **final,
        "pg_identity_digest": row_checks["endpoint"]["pg_identity_digest"],
    }
    initial_endpoint = {
        **initial_endpoint,
        "pg_identity_digest": initial_checks["endpoint"]["pg_identity_digest"],
    }
    record = {
        "arm": arm,
        "config": config,
        "output": str(output),
        "initialization_hash": initial_hash,
        "parameters": parameters,
        "optimizer_steps": extra["counts"]["optimizer_steps"],
        "counts": extra["counts"],
        "sample_plan_digest": extra["sample_plan_digest"],
        "sample_plan_episodes": extra["sample_plan_episodes"],
        "validation_plan_hash": file_hash(
            checked_path(output, output / "validation_manifest.json")
        ),
        "manifest_hashes": state["metadata"]["manifest_hashes"],
        "checkpoint_sha256": file_hash(output / "last.pt"),
        "checkpoint_checks": checks,
        "endpoint_rows_sha256": save_rows(output / "endpoint_rows.jsonl", rows),
        "initial_endpoint_rows_sha256": save_rows(
            output / "initial_endpoint_rows.jsonl", initial_rows
        ),
        "initial_endpoint": initial_endpoint,
        "endpoint": final,
        "pg_identity_digest": pg_identity(rows),
        "initial_pg_identity_digest": pg_identity(initial_rows),
        "rows_verification": row_checks,
        "initial_rows_verification": initial_checks,
        "strata": verifier.analyze_rows(rows, config["model"]["pg_output_scale"]),
        "summary": summary,
        "seconds": time.monotonic() - started,
        "peak_memory_bytes": torch.cuda.max_memory_allocated()
        if device == "cuda"
        else 0,
    }
    if (
        record["pg_identity_digest"] != record["initial_pg_identity_digest"]
        or final["identity_digest"] != initial_endpoint["identity_digest"]
    ):
        raise ValueError("Initial/final endpoint identity mismatch")
    write_json(output / "result.json", record)
    return record


def status(output: Path, state: str, arms: dict, **extra) -> None:
    precision.status(output, state, arms, **extra)


def execute_arms(
    configs: dict,
    data: dict,
    output: Path,
    evidence: dict,
    historical: dict,
    index: str,
    smoke: bool = False,
) -> dict:
    from scripts import s4_anchor_evidence as verifier

    states = dict.fromkeys(ARMS, "not_started")
    records, checks = [], []
    output.mkdir(exist_ok=True)
    for arm in ARMS:
        states = {**states, arm: "running"}
        status(output, "running", states)
        try:
            gpu_gate("cuda", index, expected_uuid=evidence["gpu_uuid"])
            if source_evidence(data, historical) != evidence["sources"]:
                raise ValueError("Source inputs changed before arm")
            record = run_one(
                arm,
                configs[arm],
                data,
                output / arm,
                "cuda",
                None if smoke else COVERAGE,
                smoke,
            )
            if source_evidence(data, historical) != evidence["sources"]:
                raise ValueError("Source inputs changed during arm")
            record = {**record, "gpu_uuid": evidence["gpu_uuid"]}
            records.append(record)
            write_json(output / "runs.json", {"runs": records})
            checks.append(
                require_passed(verifier.verify_run(record, evidence, formal=not smoke))
            )
            if not smoke and arm == ARMS[0]:
                checks.append(
                    require_passed(
                        verifier.verify_baseline(
                            record, read_json(ROOT / (FORMAL + "summary.json"))
                        )
                    )
                )
            states = {**states, arm: "verified"}
            status(output, "running", states)
        except BaseException as error:
            blocked = not smoke and arm == ARMS[0] and records and len(checks) == 1
            status(
                output,
                "reproduction_blocked" if blocked else "failed",
                {**states, arm: "failed"},
                error=f"{type(error).__name__}: {error}",
            )
            raise
    checks.append(require_passed(verifier.verify_pair(records, formal=not smoke)))
    decision = {} if smoke else verifier.choose(records)
    report = {
        "purpose": "gpu-smoke-not-research"
        if smoke
        else "research-source-only-single-seed",
        "runs": records,
        **decision,
    }
    write_json(output / "summary.json", report)
    write_json(output / "verification.json", {"passed": True, "checks": checks})
    return report


def smoke_configs(configs: dict) -> dict:
    return {
        a: {
            **c,
            "steps": 2,
            "support_sizes": [5],
            "query_size": 8,
            "validate_every": 1,
            "validation_runs": 1,
            "purpose": "gpu-smoke-not-research",
        }
        for a, c in configs.items()
    }


def output_manifest(output: Path) -> dict:
    hashes = {
        str(p.relative_to(output)): file_hash(checked_path(output, p))
        for p in sorted(output.rglob("*"))
        if p.is_file() and p.name not in ("status.json", "output-manifest.json")
    }
    write_json(output / "output-manifest.json", hashes)
    return hashes


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
                raise ValueError("Incomplete verified record matrix")
            for record in records:
                directory = output / prefix / record["arm"]
                result = read_json(checked_path(output, directory / "result.json"))
                for field, name in fields.items():
                    actual = file_hash(checked_path(output, directory / name))
                    if actual != record[field] or actual != result[field]:
                        raise ValueError(
                            f"Verified record hash mismatch: {prefix}{record['arm']}/{name}"
                        )
    except (KeyError, TypeError) as error:
        raise ValueError("Incomplete verified record hashes") from error


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
    for arm in ARMS:
        required |= {
            f"{prefix}{arm}/{name}"
            for prefix in ("", "smoke/")
            for name in (
                "last.pt",
                "result.json",
                "endpoint_rows.jsonl",
                "initial_endpoint_rows.jsonl",
                "validation_manifest.json",
            )
        }
    if set(hashes) != names or not required <= names:
        raise ValueError("Incomplete output hash manifest")
    for name, sha in hashes.items():
        if file_hash(checked_path(output, output / name)) != sha:
            raise ValueError("Supervisor output hash mismatch")
    if read_json(output / "verification.json").get("passed") is not True:
        raise ValueError("Independent output verification failed")
    verify_record_files(output)
    return hashes


def worker(output: Path, index: str) -> None:
    owner = read_json(checked_path(ROOT / "runs", output / "supervision.json"))
    parent = os.getppid()
    if owner != {"pid": parent, "token": os.environ.get("S4_TOKEN")} or os.environ.get(
        "S4_PARENT"
    ) != str(parent):
        raise ValueError("Worker requires its owning supervisor")
    if {p.name for p in output.iterdir()} != {
        "status.json",
        "supervision.json",
    } or read_json(output / "status.json")["status"] != "planned":
        raise ValueError("Worker requires fresh supervised output")
    precision.bind_parent_lifetime(parent)
    previous = signal.signal(signal.SIGALRM, deadline_expired)
    remaining = float(os.environ["S4_DEADLINE"]) - time.monotonic()
    if remaining <= 0:
        raise TimeoutError("S4 deadline exhausted")
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
            smoke_configs(plan["arms"]),
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
        output_manifest(output)
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


def deadline_expired(_signal: int, _frame: object) -> None:
    raise TimeoutError("S4 absolute six-hour deadline reached")


def stop_child(child) -> None:
    deadline = time.monotonic() + STOP_GRACE
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(child.pid, sig)
        except ProcessLookupError:
            pass
        try:
            child.wait(
                timeout=max(
                    0.001,
                    (deadline - time.monotonic()) / (2 if sig == signal.SIGTERM else 1),
                )
            )
            break
        except subprocess.TimeoutExpired:
            if sig == signal.SIGKILL:
                raise RuntimeError("Own process group cleanup exceeded ten seconds")
    try:
        os.killpg(child.pid, signal.SIGKILL)  # descendants may survive leader exit
    except ProcessLookupError:
        pass


def supervise(command: list, output: Path, deadline: float) -> None:
    token = secrets.token_hex(16)
    write_json(output / "supervision.json", {"pid": os.getpid(), "token": token})
    child = None
    previous = signal.signal(signal.SIGTERM, precision.interrupted)
    try:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("S4 budget exhausted before worker")
        child = subprocess.Popen(
            command,
            start_new_session=True,
            env={
                **os.environ,
                "S4_PARENT": str(os.getpid()),
                "S4_TOKEN": token,
                "S4_DEADLINE": str(deadline),
            },
        )
        code = child.wait(timeout=remaining)
        current = read_json(output / "status.json")
        if code != 0 or current["status"] != "outputs_ready":
            raise RuntimeError(f"S4 worker did not complete: {code}")
        hashes = verify_outputs(output)
        if time.monotonic() >= deadline:
            raise TimeoutError("S4 deadline exhausted during output verification")
        status(
            output, "completed", dict.fromkeys(ARMS, "completed"), output_sha256=hashes
        )
    except BaseException as error:
        signal.setitimer(signal.ITIMER_REAL, 0)
        cleanup_error = None
        if child is not None:
            try:
                stop_child(child)
            except BaseException as cleanup:
                cleanup_error = f"{type(cleanup).__name__}: {cleanup}"
        current = read_json(output / "status.json")
        states = {
            a: "partial" if v == "running" else v
            for a, v in current.get("arms", dict.fromkeys(ARMS, "not_started")).items()
        }
        status(
            output,
            "failed"
            if cleanup_error is not None
            else "reproduction_blocked"
            if current["status"] == "reproduction_blocked"
            else "partial"
            if isinstance(
                error, (TimeoutError, subprocess.TimeoutExpired, KeyboardInterrupt)
            )
            else "failed",
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
        description="Frozen source-only PG output anchor contrast"
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
    output = validate_output(Path(args.output), data)
    if not args.execute:
        print(
            json.dumps(
                {"purpose": "dry-run-not-execution", "output": str(output), **plan},
                indent=2,
            )
        )
        return
    if not sys.platform.startswith("linux"):
        raise RuntimeError("Formal S4 requires Linux CUDA execution")
    output.mkdir(parents=True)
    status(output, "planned", dict.fromkeys(ARMS, "not_started"))
    try:
        gpu_gate(args.device, args.gpu_index, initial=True)
    except GPUBusy as error:
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
