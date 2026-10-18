"""Supervised single-arm 21/3/2 source-holdout baseline runner."""
from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import build_meta65_holdout_plan as plan_builder
from scripts import meta65_holdout_contract as contract


def environment(mode: str) -> dict:
    import torch

    if mode not in ("cpu_preflight", "gpu_preflight", "formal"):
        raise ValueError("Training mode required")
    if mode == "cpu_preflight" and os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("CPU preflight requires explicitly hidden CUDA")
    if mode != "cpu_preflight" and (not os.environ.get("CUDA_VISIBLE_DEVICES")
                                     or not torch.cuda.is_available()
                                     or torch.cuda.device_count() != 1):
        raise ValueError("GPU modes require one explicitly selected CUDA device")
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise ValueError("CUBLAS_WORKSPACE_CONFIG must be :4096:8")
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        if os.environ.get(name) != "1":
            raise ValueError("Single-thread BLAS environment required")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    return {
        "schema": 1,
        "mode": mode,
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "threads": 1,
        "deterministic_algorithms": True,
        "tf32": False,
        "training": True,
        "target_read": False,
    }


def manifest_hashes(data: dict) -> dict[str, str]:
    cache_root = Path(data["cache_root"]) / "pretrain"
    hashes = {}
    for database in contract.ALL_SOURCES:
        manifest = contract.read_json(cache_root / database / "manifest.json")
        value = manifest.get("content_hash")
        if not isinstance(value, str) or len(value) != 64:
            raise ValueError(f"Invalid manifest content hash: {database}")
        hashes[database] = value
    return hashes


def load_plan(plan_path: Path, data: dict, config: dict) -> dict:
    from spatial_meta_ce.data import FeatureStore

    path = contract.checked_path(plan_path)
    plan = contract.read_json(path)
    cache_root = Path(data["cache_root"]) / "pretrain"
    identities = {}
    for database in contract.EVALUATION:
        store = FeatureStore(cache_root / database)
        identities[database] = plan_builder.validate_store(
            store, database, config["expected"]["unique"][database],
            config["expected"]["four_plus"][database],
        )
    contract.certify_plan(plan, identities)
    return plan


def train_holdout(config: dict, data: dict, signature: str, output: Path,
                  mode: str, deadline) -> tuple[object, dict]:
    from spatial_meta_ce.checkpoints import load_checkpoint
    from spatial_meta_ce.train import train

    training_config = contract.arm_config(config, mode, signature)
    arm_output = output / contract.ARM
    started = time.monotonic()
    deadline()
    summary = train(training_config, contract.training_data(data), arm_output,
                    device="cpu" if mode == "cpu_preflight" else "cuda")
    deadline()
    checkpoint_model, checkpoint = load_checkpoint(arm_output / "last.pt",
                                                   "cpu" if mode == "cpu_preflight" else "cuda")
    metadata = checkpoint["metadata"]
    counts = checkpoint["extra"]["counts"]
    steps = config["training"]["steps"] if mode == "formal" else config["preflight"]["training_steps"]
    if (checkpoint["next_step"] != steps or summary.get("last_step") != steps
            or counts.get("optimizer_steps") != steps
            or counts.get("query_presentations") != steps * config["training"]["meta_batch"] * config["training"]["query_size"]
            or metadata.get("train_databases") != list(contract.TRAIN)
            or metadata.get("validation_databases") != list(contract.VALIDATION)
            or metadata.get("training_signature", {}).get("data_signature") != signature):
        raise ValueError("Holdout training metadata/count chain changed")
    if any(not np.isfinite(value.detach().cpu().numpy()).all()
           for value in checkpoint_model.state_dict().values()):
        raise FloatingPointError("Nonfinite holdout model weights")
    record = {
        "schema": 1,
        "arm": contract.ARM,
        "mode": mode,
        "data_signature": signature,
        "metadata": metadata,
        "counts": counts,
        "checkpoint_sha256": contract.sha256(arm_output / "last.pt"),
        "checkpoint": "last.pt",
        "last_step": checkpoint["next_step"],
        "training_seconds": time.monotonic() - started,
        "summary": summary,
    }
    contract.write_json(arm_output / "metadata.json", record)
    return checkpoint_model, record


def evaluate_cells(model: object, data: dict, plan: dict, config: dict,
                   output: Path, mode: str, deadline) -> list[dict]:
    import torch

    from scripts import meta65_joint_runner as joint
    from spatial_meta_ce.data import FeatureStore
    from spatial_meta_ce.meta import forward

    formal = mode == "formal"
    cells: list[dict] = []
    prediction_dir = output / "predictions"
    prediction_dir.mkdir()
    state = None
    for database in contract.EVALUATION:
        store = FeatureStore(Path(data["cache_root"]) / "pretrain" / database)
        plan_builder.validate_store(
            store, database, config["expected"]["unique"][database],
            config["expected"]["four_plus"][database],
        )
        identity_all = joint.store_identity(store, database)
        for run in (contract.RUNS if formal else (0,)):
            deadline()
            frozen = plan[database][run]
            query = list(frozen["query"] if formal else frozen["query"][:32])
            support = list(frozen["supports"]["20"])
            identity = {key: value[query] for key, value in identity_all.items()}
            for operator in contract.OPERATORS:
                deadline()
                before = joint.state_digest(model)
                batch = None if operator == "none" else joint.certified_batch(
                    store, support, "cpu" if mode == "cpu_preflight" else "cuda"
                )
                params, operator_record = joint.operator_params(model, batch, operator)
                after = joint.state_digest(model)
                if before != after:
                    raise ValueError("Evaluation mutated shared checkpoint")
                if state is None:
                    state = before
                if before != state:
                    raise ValueError("Initialization changed across holdout cells")
                predictions = []
                for start in range(0, len(query), config["evaluation"]["batch_size"]):
                    deadline()
                    chunk = query[start:start + config["evaluation"]["batch_size"]]
                    batch = joint.certified_batch(store, chunk, "cpu" if mode == "cpu_preflight" else "cuda")
                    with torch.no_grad():
                        values = forward(model, params, batch).detach().cpu().numpy().astype(np.float64)
                    if not np.isfinite(values).all():
                        raise FloatingPointError("Nonfinite holdout prediction")
                    predictions.append(values)
                pred = np.concatenate(predictions)
                true = identity["true"]
                tables = identity["n_tables"]
                path = prediction_dir / f"{contract.ARM}-{operator}-{database}-run{run}.npz"
                with path.open("wb") as stream:
                    np.savez(stream,
                             log_pred=pred,
                             true=np.asarray(true, dtype=np.float64),
                             n_tables=np.asarray(tables, dtype=np.int16),
                             query_hash=np.asarray(identity["query_hash"]).astype("S64"),
                             query_id=np.asarray(identity["query_id"]).astype("S128"))
                cell = {
                    "schema": 1,
                    "mode": mode,
                    "arm": contract.ARM,
                    "operator": operator,
                    "database": database,
                    "run": run,
                    "k20": 20,
                    "query_count": len(query),
                    "query_digest": frozen["query_digest"],
                    "support_digest": frozen["support_digests"]["20"],
                    "prediction_file": str(path.relative_to(output)),
                    "prediction_sha256": contract.sha256(path),
                    "state_before": before,
                    "state_after": after,
                    "metrics": contract.metric_buckets(pred, true, tables),
                    **operator_record,
                }
                cells.append(cell)
                contract.write_json(
                    output / ("cells.json" if formal else "microcells.json"),
                    {"schema": 1, "experiment": contract.EXPERIMENT,
                     "mode": mode, "cells": cells},
                )
    return cells


def _status(output: Path, value: dict) -> None:
    contract.write_json(output / "status.json", value)


def _phase(output: Path, name: str) -> None:
    contract.write_json(output / "phase.json", {"phase": name})


def require_supervisor(mode: str) -> None:
    if (os.environ.get("META65_SUPERVISED_MODE") != mode
            or os.environ.get("META65_SUPERVISOR_PID") != str(os.getppid())):
        raise ValueError("Holdout worker requires its owning external supervisor")


def _formal_worker_gate(approval: Path | None, output: Path, binding: Path | None,
                        config_path: Path, data_config_path: Path,
                        plan_path: Path) -> None:
    if approval is None or binding is None:
        raise ValueError("Formal holdout execution requires approval and launch binding")
    from scripts import meta65_holdout_launch as launch
    uuid = os.environ.get("CUDA_VISIBLE_DEVICES")
    launch.verify_worker_approval(approval, output, binding, config_path,
                                  root=ROOT, gpu_uuid=uuid, parent_pid=os.getppid(),
                                  data_config_path=data_config_path, plan_path=plan_path)


def worker(output: Path, mode: str, config_path: Path, data_config_path: Path,
           plan_path: Path, *, approval: Path | None = None,
           launch_binding: Path | None = None) -> dict:
    require_supervisor(mode)
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                 "VECLIB_MAXIMUM_THREADS"):
        os.environ.setdefault(name, "1")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    config = contract.load_config(config_path)
    data = contract.validate_data(contract.read_json(data_config_path))
    output = contract.safe_output(output, [data["cache_root"], data["data_root"], plan_path])
    if mode == "formal":
        _formal_worker_gate(approval, output, launch_binding, config_path,
                            data_config_path, plan_path)
    runtime = environment(mode)
    output.mkdir(parents=True, exist_ok=False)
    _status(output, {"schema": 1, "experiment": contract.EXPERIMENT,
                     "status": "running", "mode": mode, "cells": 0,
                     "micro_cells": 0, "training": True, "target_read": False,
                     "selection_performed": False, "formal_result": mode == "formal"})
    _phase(output, "preparation")
    started = time.monotonic()
    deadline_seconds = config["limits"]["total_seconds"] if mode == "formal" else config["preflight"]["total_seconds"]

    def deadline() -> None:
        if time.monotonic() - started > deadline_seconds:
            raise TimeoutError("Holdout worker deadline exceeded")

    try:
        signature = contract.dataset_signature(manifest_hashes(data), data)
        plan = load_plan(plan_path, data, config)
        binding = contract.input_binding(ROOT, config_path, data_config_path, plan_path)
        contract.write_json(output / "environment.json", runtime)
        contract.write_json(output / "input-binding.json", {
            "schema": 1, "mode": mode, "plan_file": str(contract.checked_path(plan_path)),
            "plan_sha256": contract.sha256(plan_path), "input_sha256": binding,
            "config_sha256": contract.sha256(config_path),
            "data_config_sha256": contract.sha256(data_config_path),
        })
        contract.write_json(output / "plan_summary.json", contract.plan_summary(plan))
        _phase(output, "evaluation")
        model, _ = train_holdout(config, data, signature, output, mode, deadline)
        cells = evaluate_cells(model, data, plan, config, output, mode, deadline)
        expected = 50 if mode == "formal" else 10
        if len(cells) != expected:
            raise ValueError("Holdout cell count changed")
        for path in (output / "predictions").glob("*.npz"):
            if contract.checked_path(path).is_symlink():
                raise ValueError("Symlink prediction forbidden")
        contract.verify_bound_inputs(binding)
        _phase(output, "certification")
        if mode == "formal":
            _status(output, {"schema": 1, "experiment": contract.EXPERIMENT,
                             "status": "worker_complete", "mode": mode, "cells": 50,
                             "micro_cells": 0, "training": True, "target_read": False,
                             "selection_performed": False, "formal_result": True})
            from scripts.meta65_holdout_report import build_report
            report = build_report(output)
            contract.write_json(output / "report.json", report)
        else:
            _status(output, {"schema": 1, "experiment": contract.EXPERIMENT,
                             "status": "worker_complete", "mode": mode, "cells": 0,
                             "micro_cells": 10, "training": True, "target_read": False,
                             "selection_performed": False, "formal_result": False})
        verification = schedule_verification(output, mode)
        contract.write_json(output / "verification.json", verification)
        contract.write_json(output / "source_after.json", binding)
        return verification
    except BaseException as error:
        _status(output, {"schema": 1, "experiment": contract.EXPERIMENT,
                         "status": "failed", "mode": mode,
                         "failure": f"{type(error).__name__}: {error}",
                         "formal_result": False})
        raise


def schedule_verification(output: Path, mode: str) -> dict:
    from scripts import meta65_holdout_verify as verifier
    return verifier.verify(output, formal=mode == "formal")


def verify_gpu_idle(uuid: str) -> None:
    if not uuid:
        raise ValueError("GPU UUID required")
    query = subprocess.run(
        ["nvidia-smi", "-i", uuid, "--query-gpu=uuid,memory.used,utilization.gpu",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True, timeout=10,
    )
    fields = [item.strip() for item in query.stdout.splitlines()[0].split(",")]
    if len(fields) != 3 or fields[0] != uuid or int(fields[1]) > 100 or int(fields[2]) != 0:
        raise ValueError("Selected GPU is not idle")
    apps = subprocess.run(
        ["nvidia-smi", "-i", uuid, "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True, timeout=10,
    )
    if apps.stdout.strip():
        raise ValueError("Selected GPU already has compute processes")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/meta65_source_holdout.json")
    parser.add_argument("--data-config", type=Path,
                        default=ROOT / "configs/meta65_source_holdout_data.json")
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--mode", choices=("dry_run", *contract.MODES[1:]), default="dry_run")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--approval", type=Path)
    parser.add_argument("--launch-binding", type=Path)
    args = parser.parse_args(argv)
    if args.mode == "dry_run":
        print(json.dumps({"status": "prepared_not_executed", "matrix": contract.matrix("formal"),
                          "formal_result": False}, indent=2))
        return 0
    if args.out is None or args.plan is None:
        parser.error("--out and --plan are required for execution")
    if not args.execute:
        print(json.dumps({"status": "prepared_not_executed", "mode": args.mode,
                          "matrix": contract.matrix("formal" if args.mode == "formal" else "cpu_preflight"),
                          "formal_result": False}, indent=2))
        return 0
    config_path = contract.checked_path(args.config)
    data_config_path = contract.checked_path(args.data_config)
    plan_path = contract.checked_path(args.plan)
    contract.load_config(config_path)
    if args.worker:
        worker(contract.checked_path(args.out), args.mode, config_path, data_config_path, plan_path,
               approval=args.approval, launch_binding=args.launch_binding)
        return 0
    data = contract.validate_data(contract.read_json(data_config_path))
    output = contract.safe_output(args.out, [data["cache_root"], data["data_root"], plan_path])
    if args.mode == "formal":
        if args.approval is None or args.launch_binding is None:
            raise ValueError("Formal execution requires approval and launch binding")
        from scripts import meta65_holdout_launch as launch
        uuid = os.environ.get("CUDA_VISIBLE_DEVICES")
        launch.authorize_formal(args.approval, output, args.launch_binding, config_path,
                                root=ROOT, gpu_uuid=uuid)
    elif args.mode == "gpu_preflight":
        verify_gpu_idle(os.environ.get("CUDA_VISIBLE_DEVICES", ""))
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    from scripts.meta65_joint_supervisor import supervise
    command = [sys.executable, "-B", str(Path(__file__).resolve()), "--worker", "--execute",
               "--config", str(config_path), "--data-config", str(data_config_path),
               "--plan", str(plan_path), "--out", str(output), "--mode", args.mode]
    if (args.approval is None) != (args.launch_binding is None):
        parser.error("--approval and --launch-binding must be supplied together")
    if args.approval is not None:
        command += ["--approval", str(args.approval), "--launch-binding", str(args.launch_binding)]
    if args.mode == "formal":
        supervisor_budget = {**contract.FROZEN["limits"]}
    else:
        supervisor_budget = {
            **contract.FROZEN["limits"],
            "total_seconds": contract.FROZEN["preflight"]["total_seconds"],
            "phase_seconds": dict(contract.FROZEN["preflight"]["phase_seconds"]),
            "poll_seconds": contract.FROZEN["preflight"]["poll_seconds"],
        }
        if args.mode == "cpu_preflight":
            supervisor_budget["vram_bytes"] = 0
    record = supervise(command, output, mode=args.mode, root=ROOT,
                       budget=supervisor_budget,
                       formal_cells=50, preflight_micro_cells=10)
    print(json.dumps(record, indent=2))
    if record["status"] not in ("completed", "preflight_completed"):
        raise SystemExit(1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
