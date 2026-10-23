import argparse
import datetime
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from scripts.arch_select import checkpoint_pool
from scripts.mapping_smoke import available_gpu, state_hash
from scripts.precision_screen_evidence import (
    COVERAGE,
    SOURCE_BINDING,
    VALIDATION,
    checked_path,
    endpoint,
    finite_q95,
    historical_sources,
    source_evidence,
    write_json,
)
from scripts.run_mapping_ablation import configure_determinism, initial_state
from spatial_meta_ce.checkpoints import load_checkpoint
from spatial_meta_ce.cli import data_settings, read_json
from spatial_meta_ce.manifests import file_hash
from spatial_meta_ce.train import train

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/precision_screen.json"
PROTOCOL = ROOT / "实验文档/15-离线精度提升首轮实验计划.md"
PROTOCOL_HASH = "ccc5d39f4f8acc87e8daeee2cd65c55f05730e42802c6321ca51bfd598a87152"
DATA_HASH = "a9057d65c7cb220bd21e7cd67dacbc15c0c20dcc87f61d9b3e6470c0ea094d90"
ARMS = ("A_w64_smooth", "B_w64_mse", "C_w128_smooth", "D_w128_mse")
TIMEOUT = 43200
STOP_GRACE = 10
FROZEN = {
    "method": "erm",
    "seed": 42,
    "steps": 4000,
    "meta_batch": 4,
    "accumulation": 1,
    "query_size": 64,
    "support_sizes": [5, 10, 20, 50, 100],
    "outer_lr": 0.0003,
    "inner_lr": 0.001,
    "inner_steps": 0,
    "adapt_scope": "all",
    "validate_every": 100,
    "validation_runs": 5,
    "validation_seed": 1729,
    "patience": 41,
    "min_delta": 0.0001,
}
MODEL = {"arch": "sum", "use_pg": True, "layers": 2, "heads": 4}
SPECS = {
    a: {"width": 64 if i < 2 else 128, "loss": "smooth_l1" if i % 2 == 0 else "mse"}
    for i, a in enumerate(ARMS)
}


def load_plan() -> dict:
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
    if read_json(CONFIG) != expected or file_hash(PROTOCOL) != PROTOCOL_HASH:
        raise ValueError("Frozen precision protocol or matrix changed")
    if file_hash(ROOT / "configs/data.json") != DATA_HASH:
        raise ValueError("Frozen source split or data settings changed")
    arms = {
        a: {
            **FROZEN,
            "support_sizes": list(FROZEN["support_sizes"]),
            "model": {**MODEL, "width": s["width"]},
            "loss": s["loss"],
            "data_config": expected["data_config"],
            "purpose": "research",
        }
        for a, s in SPECS.items()
    }
    return {
        **expected,
        "arms": arms,
        "splits": read_json(ROOT / "configs/data.json")["splits"],
    }


def validate_data(data: dict, plan: dict) -> None:
    if data["splits"] != plan["splits"]:
        raise ValueError("Exact frozen 23/3/4 split required")
    if Path(data["data_root"]).resolve() != Path("/gly/weilin/PRICE-master"):
        raise ValueError("Frozen PRICE source root changed")
    checked_path(ROOT, ROOT / "artifacts/cache")
    if Path(data["cache_root"]).resolve() != (ROOT / "artifacts/cache").resolve():
        raise ValueError("Frozen source cache root changed")
    for db in data["splits"]["train"] + data["splits"]["validation"]:
        checked_path(
            Path(data["cache_root"]), Path(data["cache_root"]) / "pretrain" / db
        )


def validate_output(path: Path, data: dict, root: Path | None = None) -> Path:
    root = ROOT if root is None else root
    absolute = path.absolute()
    if any(p.is_symlink() for p in (absolute, *absolute.parents)):
        raise ValueError("Output cannot traverse symlinks")
    resolved = path.resolve()
    runs = (root / "runs").resolve()
    if resolved == runs or not resolved.is_relative_to(runs):
        raise ValueError("Output must be a new subdirectory of project runs")
    for field in ("data_root", "cache_root"):
        source = Path(data[field]).resolve()
        if resolved.is_relative_to(source) or source.is_relative_to(resolved):
            raise ValueError("Output overlaps read-only inputs")
    if path.exists():
        raise FileExistsError("Output already exists")
    return resolved


def formal_gpu(device: str, index: str | None) -> str:
    if (
        device != "cuda"
        or index is None
        or os.environ.get("CUDA_VISIBLE_DEVICES") != index
    ):
        raise ValueError(
            "Use cuda with exactly the physical GPU index in CUDA_VISIBLE_DEVICES"
        )
    return available_gpu(index)


def checkpoint_smoke(
    model, state: dict, output: Path, pool, device: str, initial_hash: str
) -> dict:
    if state_hash(model) == initial_hash:
        raise ValueError("Optimizer did not change weights")
    if not all(torch.isfinite(t).all() for t in model.state_dict().values()):
        raise FloatingPointError("Nonfinite checkpoint weights")
    store = pool.get("pretrain", "legalacts")
    indices = store.unique_indices()[:8]
    reloaded, _ = load_checkpoint(output / "last.pt", device)
    cpu, _ = load_checkpoint(output / "last.pt", "cpu")
    with torch.no_grad():
        actual = model.eval()(store.batch(indices, device)).cpu()
        repeated = reloaded.eval()(store.batch(indices, device)).cpu()
        reference = cpu.eval()(store.batch(indices, "cpu"))
    torch.testing.assert_close(actual, repeated, rtol=0, atol=0)
    torch.testing.assert_close(actual, reference, rtol=1e-4, atol=1e-4)
    return {
        "roundtrip_exact": True,
        "optimizer_changed_weights": True,
        "cpu_device_max_abs_difference": float((actual - reference).abs().max()),
        "checkpoint_steps": state["extra"]["counts"]["optimizer_steps"],
    }


def run_one(
    arm: str,
    config: dict,
    data: dict,
    output: Path,
    device: str,
    expected_coverage: dict | None = COVERAGE,
    smoke: bool = False,
) -> dict:
    initial = initial_state(config)
    started = time.monotonic()
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    summary = train(config, data, output, device=device)
    model, state = load_checkpoint(output / "last.pt", device)
    extra = state["extra"]
    if (
        extra["counts"]["optimizer_steps"] != config["steps"]
        or state["next_step"] != config["steps"]
    ):
        raise ValueError("Frozen budget incomplete")
    if state["model_config"] != config["model"] or state["metadata"][
        "training_signature"
    ] != {k: v for k, v in config.items() if k not in ("steps", "data_config")}:
        raise ValueError("Checkpoint does not match arm configuration")
    episodes = config["steps"] * config["meta_batch"]
    if (
        extra["sample_plan_episodes"] != episodes
        or extra["counts"]["backward_calls"] != episodes
        or extra["counts"]["support_presentations"] != 0
    ):
        raise ValueError("ERM episode or backward budget changed")
    pool = checkpoint_pool(state, data)
    checks = checkpoint_smoke(
        model, state, output, pool, device, initial["initialization_hash"]
    )
    result, rows_by_db = endpoint(model, pool, device, expected_coverage, smoke)
    with (output / "endpoint_rows.jsonl").open("x") as stream:
        for db, rows in rows_by_db.items():
            for row in rows:
                stream.write(
                    json.dumps({"database": db, **row}, allow_nan=False) + "\n"
                )
    record = {
        "arm": arm,
        "config": config,
        "output": str(output),
        **initial,
        "optimizer_steps": extra["counts"]["optimizer_steps"],
        "counts": extra["counts"],
        "sample_plan_digest": extra["sample_plan_digest"],
        "sample_plan_episodes": extra["sample_plan_episodes"],
        "validation_plan_hash": file_hash(output / "validation_manifest.json"),
        "manifest_hashes": state["metadata"]["manifest_hashes"],
        "checkpoint_sha256": file_hash(output / "last.pt"),
        "endpoint_rows_sha256": file_hash(output / "endpoint_rows.jsonl"),
        "checkpoint_checks": checks,
        "endpoint": result,
        "summary": summary,
        "seconds": time.monotonic() - started,
        "peak_memory_bytes": torch.cuda.max_memory_allocated()
        if device == "cuda"
        else 0,
    }
    write_json(output / "result.json", record)
    return record


def ordered_records(records: list) -> dict:
    if len(records) != len(ARMS) or {r["arm"] for r in records} != set(ARMS):
        raise ValueError("Exactly the four frozen arms required")
    return {r["arm"]: r for r in records}


def verify_partial(records: list, steps: int) -> None:
    if not records or len(records) > len(ARMS):
        raise ValueError("Invalid partial matrix")
    if [r["arm"] for r in records] != list(ARMS[: len(records)]):
        raise ValueError("Arm order or identity changed")
    for key in (
        "counts",
        "sample_plan_digest",
        "sample_plan_episodes",
        "validation_plan_hash",
        "manifest_hashes",
    ):
        if len({json.dumps(r[key], sort_keys=True) for r in records}) != 1:
            raise ValueError(f"Four-arm pairing mismatch: {key}")
    if len({r["endpoint"]["identity_digest"] for r in records}) != 1:
        raise ValueError("Endpoint query identities, truth or table counts differ")
    by_arm = {r["arm"]: r for r in records}
    for record in records:
        if (
            record["optimizer_steps"] != steps
            or record["counts"]["optimizer_steps"] != steps
        ):
            raise ValueError("Optimizer budget incomplete")
    for left, right in ((ARMS[0], ARMS[1]), (ARMS[2], ARMS[3])):
        if right not in by_arm:
            continue
        for key in ("initialization_hash", "parameters"):
            if by_arm[left][key] != by_arm[right][key]:
                raise ValueError("Same-width arm initializations or capacities differ")


def verify_group(records: list, steps: int = 4000) -> None:
    ordered_records(records)
    verify_partial(records, steps)


def choose_candidates(records: list) -> dict:
    by_arm = ordered_records(records)
    for r in records:
        e = r["endpoint"]
        finite_q95(e["macro_four_plus_q95"])
        if set(e["per_database"]) != set(VALIDATION):
            raise ValueError("Missing source endpoint")
        for db in VALIDATION:
            for bucket in ("all_tables", "four_plus_tables"):
                finite_q95(e["per_database"][db][bucket]["q95"])
    baseline = by_arm[ARMS[0]]["endpoint"]
    candidates = [
        a
        for a in ARMS[1:]
        if by_arm[a]["endpoint"]["macro_four_plus_q95"]
        <= 0.9 * baseline["macro_four_plus_q95"]
        and all(
            by_arm[a]["endpoint"]["per_database"][db][b]["q95"]
            <= 1.1 * baseline["per_database"][db][b]["q95"]
            for db in VALIDATION
            for b in ("all_tables", "four_plus_tables")
        )
    ]
    from functools import cmp_to_key

    def compare(a: str, b: str) -> int:
        delta = (
            by_arm[a]["endpoint"]["macro_four_plus_q95"]
            - by_arm[b]["endpoint"]["macro_four_plus_q95"]
        )
        if abs(delta) > 1e-9:
            return -1 if delta < 0 else 1
        return -1 if (by_arm[a]["parameters"], a) < (by_arm[b]["parameters"], b) else 1

    candidates.sort(key=cmp_to_key(compare))
    return {
        "decision": "promising_single_seed" if candidates else "no_candidate",
        "candidates": candidates,
        "baseline": ARMS[0],
        "primary_improvement": {
            a: 1
            - by_arm[a]["endpoint"]["macro_four_plus_q95"]
            / baseline["macro_four_plus_q95"]
            for a in ARMS
        },
    }


def status(output: Path, state: str, arms: dict, **extra) -> None:
    write_json(
        output / "status.json",
        {
            "status": state,
            "arms": arms,
            "updated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            **extra,
        },
    )


def execute_arms(
    configs: dict,
    data: dict,
    output: Path,
    device: str,
    expected_coverage: dict | None = COVERAGE,
    smoke: bool = False,
    historical: dict | None = None,
    sources: dict | None = None,
) -> dict:
    states = dict.fromkeys(ARMS, "planned")
    records = []
    for arm in ARMS:
        states = {**states, arm: "running"}
        status(output, "running", states)
        try:
            if historical is not None and source_evidence(data, historical) != sources:
                raise ValueError("Source inputs changed before arm")
            record = run_one(
                arm,
                configs[arm],
                data,
                output / arm,
                device,
                expected_coverage=expected_coverage,
                smoke=smoke,
            )
            if historical is not None and source_evidence(data, historical) != sources:
                raise ValueError("Source inputs changed during arm")
            records.append(record)
            write_json(output / "runs.json", {"runs": records})
            verify_partial(records, configs[ARMS[0]]["steps"])
            states = {**states, arm: "completed"}
            write_json(output / "runs.json", {"runs": records})
            status(output, "running", states)
        except Exception as error:
            status(
                output,
                "failed",
                {**states, arm: "failed"},
                error=f"{type(error).__name__}: {error}",
            )
            raise
    verify_group(records, steps=configs[ARMS[0]]["steps"])
    report = {
        "purpose": "gpu-smoke-not-research"
        if smoke
        else "research-source-only-single-seed",
        "runs": records,
        **choose_candidates(records),
    }
    write_json(output / "summary.json", report)
    return report


def terminate_child(child) -> None:
    try:
        os.killpg(child.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        child.wait(timeout=STOP_GRACE)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(child.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        child.wait(timeout=STOP_GRACE)
    try:
        os.killpg(child.pid, 0)
    except ProcessLookupError:
        return
    os.killpg(child.pid, signal.SIGKILL)


def interrupted(_signum, _frame) -> None:
    raise KeyboardInterrupt("Supervisor received SIGTERM")


def supervise(command: list, output: Path, timeout: int = TIMEOUT) -> None:
    import secrets

    started = time.monotonic()
    token = secrets.token_hex(16)
    write_json(output / "supervision.json", {"parent_pid": os.getpid(), "token": token})
    child = subprocess.Popen(
        command,
        start_new_session=True,
        env={
            **os.environ,
            "PRECISION_SUPERVISOR_PID": str(os.getpid()),
            "PRECISION_SUPERVISOR_TOKEN": token,
        },
    )
    previous = signal.signal(signal.SIGTERM, interrupted)
    try:
        code = child.wait(timeout=max(0.001, timeout - (time.monotonic() - started)))
    except BaseException as error:
        try:
            terminate_child(child)
        finally:
            existing = (
                read_json(output / "status.json")
                if (output / "status.json").exists()
                else {}
            )
            states = {
                a: ("partial" if s == "running" else s)
                for a, s in existing.get("arms", dict.fromkeys(ARMS, "planned")).items()
            }
            status(
                output,
                "partial",
                states,
                error=type(error).__name__,
                seconds=time.monotonic() - started,
            )
        if isinstance(error, (subprocess.TimeoutExpired, KeyboardInterrupt)):
            raise TimeoutError(
                "Precision task exceeded its budget or was interrupted"
            ) from error
        raise
    finally:
        signal.signal(signal.SIGTERM, previous)
    existing = (
        read_json(output / "status.json") if (output / "status.json").exists() else {}
    )
    if code != 0 or existing.get("status") != "completed":
        status(
            output,
            "failed",
            existing.get("arms", dict.fromkeys(ARMS, "planned")),
            error=existing.get(
                "error", f"Worker exit {code} without verified completion"
            ),
            seconds=time.monotonic() - started,
        )
        raise RuntimeError(f"Precision worker did not complete: {code}")
    status(
        output,
        "completed",
        existing["arms"],
        **{
            k: v
            for k, v in existing.items()
            if k not in ("status", "arms", "updated_at", "seconds")
        },
        seconds=time.monotonic() - started,
    )


def validate_worker_output(output: Path) -> None:
    checked_path(ROOT / "runs", output)
    if output.resolve() == (ROOT / "runs").resolve():
        raise ValueError("Worker cannot write to runs root")
    if not output.is_dir() or {p.name for p in output.iterdir()} != {
        "status.json",
        "supervision.json",
    }:
        raise ValueError("Worker output is not a fresh supervised run")
    owner = read_json(checked_path(output, output / "supervision.json"))
    if owner["parent_pid"] != os.getppid() or owner["token"] != os.environ.get(
        "PRECISION_SUPERVISOR_TOKEN"
    ):
        raise ValueError("Worker directory does not belong to its supervisor")
    initial = read_json(checked_path(output, output / "status.json"))
    if initial["status"] != "planned" or initial["arms"] != dict.fromkeys(
        ARMS, "planned"
    ):
        raise ValueError("Worker output has already been used")


def bind_parent_lifetime(parent: int) -> None:
    if not sys.platform.startswith("linux"):
        raise RuntimeError("Formal worker requires Linux parent-death protection")
    import ctypes

    if ctypes.CDLL(None, use_errno=True).prctl(1, signal.SIGTERM) != 0:
        raise RuntimeError("Could not bind worker lifetime to supervisor")
    if os.getppid() != parent:
        raise RuntimeError("Supervisor exited during worker startup")


def worker(output: Path, index: str, smoke: bool) -> None:
    if os.environ.get("PRECISION_SUPERVISOR_PID") != str(os.getppid()):
        raise RuntimeError("Worker requires its timeout supervisor")
    validate_worker_output(output)
    bind_parent_lifetime(int(os.environ["PRECISION_SUPERVISOR_PID"]))
    plan = load_plan()
    data = data_settings(plan["arms"][ARMS[0]], CONFIG)
    validate_data(data, plan)
    gpu_uuid = formal_gpu("cuda", index)
    configure_determinism()
    historical = historical_sources(
        ROOT, data["splits"]["train"] + data["splits"]["validation"]
    )
    evidence = source_evidence(data, historical)
    import numpy
    import sqlglot

    runtime = {
        "python": sys.version,
        "torch": torch.__version__,
        "numpy": numpy.__version__,
        "sqlglot": sqlglot.__version__,
        "gpu_uuid": gpu_uuid,
        "gpu_index": index,
        "protocol_sha256": file_hash(PROTOCOL),
        "config_sha256": file_hash(CONFIG),
        "source_binding": SOURCE_BINDING,
        "code_sha256": {
            str(p.relative_to(ROOT)): file_hash(p)
            for folder in ("scripts", "spatial_meta_ce", "vendor")
            for p in sorted((ROOT / folder).rglob("*.py"))
            if "__pycache__" not in p.parts
        },
    }
    write_json(
        output / "preflight.json", {**runtime, "sources": evidence, "smoke": smoke}
    )
    configs = plan["arms"]
    if smoke:
        configs = {
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
    execute_arms(
        configs,
        data,
        output,
        "cuda",
        expected_coverage=None if smoke else COVERAGE,
        smoke=smoke,
        historical=historical,
        sources=evidence,
    )
    if source_evidence(data, historical) != evidence:
        raise ValueError("Source inputs changed during execution")
    status(
        output,
        "completed",
        dict.fromkeys(ARMS, "completed"),
        gpu_uuid=gpu_uuid,
        smoke=smoke,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Frozen source-only precision four-arm screen"
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--gpu-index")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    plan = load_plan()
    data = data_settings(plan["arms"][ARMS[0]], CONFIG)
    validate_data(data, plan)
    if args.worker:
        worker(Path(args.output), args.gpu_index, args.smoke)
        return
    output = validate_output(Path(args.output), data)
    if not args.execute:
        print(
            json.dumps(
                {"purpose": "dry-run-not-execution", "output": str(output), **plan},
                indent=2,
            )
        )
        return
    formal_gpu(args.device, args.gpu_index)
    output.mkdir(parents=True)
    status(output, "planned", dict.fromkeys(ARMS, "planned"))
    command = [
        sys.executable,
        "-B",
        str(Path(__file__).resolve()),
        "--execute",
        "--worker",
        "--output",
        str(output),
        "--gpu-index",
        args.gpu_index,
    ]
    if args.smoke:
        command.append("--smoke")
    supervise(command, output, timeout=TIMEOUT)


if __name__ == "__main__":
    main()
