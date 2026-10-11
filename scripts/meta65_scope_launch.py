"""Read-only launch certification and explicit, one-shot Meta65 scope authorization.

No API here creates an approval. An operator supplies an exact approval separately;
only the supervising parent consumes it, before starting its owned worker.
"""
from __future__ import annotations

import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Any

from scripts import meta65_scope_contract as contract
from scripts import meta65_scope_verify as scope_verify
from scripts.meta65_joint_runner import verify_gpu_idle
from scripts.meta65_joint_supervisor import artifact_paths

EXPERIMENT = "meta65-training-scope-20261011"
PERMISSION = "one_shot_meta65_scope_formal"
MODES = ("cpu_preflight", "gpu_preflight")
SHA = re.compile(r"[0-9a-f]{64}\Z")
GPU = re.compile(r"GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}\Z")
Json = dict[str, Any]
FROZEN: Json = contract.FROZEN


def _path(value: Path | str, *, absolute: bool = True) -> Path:
    path = Path(value)
    if ".." in path.parts or (absolute and not path.is_absolute()):
        raise ValueError("Absolute path without '..' required")
    return contract.checked_path(path)


def _inside(root: Path, relative: str) -> Path:
    if not isinstance(relative, str) or not relative:
        raise ValueError("Nonempty relative path required")
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("Bound code/config/protocol must be relative inside root")
    result = _path(root / path)
    if not result.is_relative_to(root) or not result.is_file():
        raise ValueError("Bound file missing or outside isolated root")
    return result


def _json(path: Path) -> Json:
    if not path.is_file():
        raise ValueError(f"Evidence file missing: {path}")
    value = contract.read_json(path)
    if not isinstance(value, dict):
        raise ValueError("JSON object required")
    return value


def _sha(path: Path, expected: Any) -> None:
    if not isinstance(expected, str) or SHA.fullmatch(expected) is None:
        raise ValueError("Invalid SHA256")
    if not path.is_file() or contract.sha256(path) != expected:
        raise ValueError(f"SHA256 mismatch: {path}")


def _exact(actual: Any, expected: Any) -> bool:
    # Python equality alone accepts True as 1, including in nested budgets.
    return contract.digest(actual) == contract.digest(expected)


def _uuid(value: Any) -> str:
    if not isinstance(value, str) or GPU.fullmatch(value) is None:
        raise ValueError("Full NVIDIA GPU UUID required")
    return value


def _input_map(value: Any) -> dict[str, str]:
    if not isinstance(value, dict) or not value:
        raise ValueError("Nonempty complete input_sha256 required")
    for name, expected in value.items():
        if not isinstance(name, str):
            raise ValueError("Input path must be a string")
        path = _path(name)
        if str(path) != name or scope_verify.TARGET_PATTERN.search(name):
            raise ValueError("Noncanonical or target input path forbidden")
        _sha(path, expected)
    return value


def _inventory(output: Path, supervisor: Path, raw: Any) -> None:
    if not output.is_dir() or not isinstance(raw, dict) or not raw:
        raise ValueError("Preflight raw output/manifest missing")
    entries = list(output.rglob("*"))
    if any(path.is_symlink() or (not path.is_file() and not path.is_dir()) for path in entries):
        raise ValueError("Raw output contains symlink or special file")
    expected_sidecar, log = artifact_paths(output)
    if supervisor != expected_sidecar:
        raise ValueError("Preflight supervisor path not bound to raw output")
    files = [*(p for p in entries if p.is_file()), supervisor, log]
    names = {str(path) for path in files}
    required = {"status.json", "verification.json", "input-binding.json", "microcells.json",
                "environment.json", "training-records.json", "reproduction.json"}
    if set(raw) != names or not required <= {str(p.relative_to(output)) for p in entries if p.is_file()}:
        raise ValueError("Preflight raw manifest must cover every artifact and sidecar exactly")
    for name, expected in raw.items():
        path = _path(name)
        if str(path) != name:
            raise ValueError("Noncanonical raw manifest path")
        _sha(path, expected)


def _runtime_versions() -> Json:
    import numpy as np
    import torch

    return {"python": sys.version, "numpy": np.__version__, "torch": torch.__version__,
            "cuda": torch.version.cuda}


def _environment(output: Path, mode: str, uuid: str) -> None:
    env = _json(output / "environment.json")
    expected = {**_runtime_versions(), "mode": mode, "threads": 1,
                "precision": "FP32 model/FP64 true", "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
                "CUDA_VISIBLE_DEVICES": "" if mode == MODES[0] else uuid,
                "deterministic_algorithms": True, "tf32": False}
    if any(not _exact(env.get(key), value) for key, value in expected.items()):
        raise ValueError("Preflight environment differs from frozen/current runtime")


def preflight_budget(mode: str) -> Json:
    """Frozen supervisor budget for one preflight: mode wall clock, formal resource caps."""
    if mode not in MODES:
        raise ValueError("Unknown preflight mode")
    return {**dict(contract.FROZEN["limits"]),
            "total_seconds": contract.FROZEN[mode]["total_seconds"], "poll_seconds": 1.0}


def _supervisor(output: Path, sidecar: Path, mode: str, uuid: str) -> None:
    record = _json(sidecar)
    expected: Json = {"schema": 1, "status": "preflight_completed", "mode": mode,
                "test_only": False, "formal_result": False, "returncode": 0, "failure": None,
                "cleanup_verified": True, "output": str(output), "supervisor_json": str(sidecar),
                "gpu_uuid": None if mode == MODES[0] else uuid,
                "budget": preflight_budget(mode)}
    # Supervisor serializes numeric budgets as floats; reject bool, allow equal finite numbers.
    for key, value in expected.items():
        if key != "budget" and not _exact(record.get(key), value):
            raise ValueError("Preflight supervisor incomplete or unsuccessful")
    budget = record.get("budget")
    if not isinstance(budget, dict) or set(budget) != set(expected["budget"]):
        raise ValueError("Preflight supervisor budget changed")
    for key, limit in expected["budget"].items():
        _number(budget.get(key), lower=limit, upper=limit)
    _number(record.get("elapsed"), lower=0, upper=contract.FROZEN[mode]["total_seconds"], positive=True)
    for field, limit in (("peak_rss_bytes", "rss_bytes"), ("peak_vram_bytes", "vram_bytes"),
                         ("outputbytes", "output_bytes")):
        _number(record.get(field), lower=0, upper=contract.FROZEN["limits"][limit])
    if mode == MODES[0] and record["peak_vram_bytes"] != 0:
        raise ValueError("CPU preflight allocated GPU memory")


def _number(value: Any, *, lower: float, upper: float, positive: bool = False) -> None:
    if (type(value) not in (int, float) or not math.isfinite(value)
            or not lower <= value <= upper or (positive and value <= 0)):
        raise ValueError("Invalid resource evidence or budget exceeded")


def _training(output: Path) -> None:
    payload = _json(output / "training-records.json")
    records = payload.get("records")
    if (not _exact(payload.get("schema"), 1) or not isinstance(records, list) or len(records) != 2
            or any(not isinstance(record, dict) for record in records)
            or {record.get("arm") for record in records} != set(contract.ARMS)):
        raise ValueError("Exactly two preflight training records required")
    for record in records:
        if (not isinstance(record.get("config"), dict) or not _exact(record["config"].get("steps"), 2)
                or not _exact(record.get("last_step"), 2)
                or not _exact(record.get("optimizer_steps"), 2)):
            raise ValueError("Preflight training must execute two steps per arm")
        checkpoint_name = record.get("checkpoint")
        if not isinstance(checkpoint_name, str) or not checkpoint_name:
            raise ValueError("Preflight checkpoint missing")
        path = Path(checkpoint_name)
        checkpoint = _path(path if path.is_absolute() else output / record["arm"] / path)
        if not checkpoint.is_relative_to(output) or checkpoint.name != "last.pt":
            raise ValueError("Checkpoint must be raw last.pt inside preflight output")
        _sha(checkpoint, record.get("checkpoint_sha256"))


def _verification(value: Json, mode: str) -> None:
    expected = {"status": "preflight_verified", "mode": mode, "formal_result": False,
                "engineering_verified": True, "reproduction_verified": False,
                "cells_verified": 12, "all_metrics_verified": True, "summary": None}
    if any(not _exact(value.get(key), item) for key, item in expected.items()):
        raise ValueError("Independent 12-cell preflight certification required")


def _preflight(entry: Any, mode: str, binding: Json) -> None:
    if not isinstance(entry, dict) or set(entry) != {"output", "supervisor_file", "raw_sha256"}:
        raise ValueError("Complete preflight raw binding required")
    if not isinstance(entry["output"], str) or not isinstance(entry["supervisor_file"], str):
        raise ValueError("Preflight paths must be absolute strings")
    output, supervisor = _path(entry["output"]), _path(entry["supervisor_file"])
    if str(output) != entry["output"] or str(supervisor) != entry["supervisor_file"]:
        raise ValueError("Canonical absolute raw paths required")
    _inventory(output, supervisor, entry["raw_sha256"])
    status = _json(output / "status.json")
    expected = {"status": "preflight_completed", "mode": mode, "cells": 12, "formal_result": False,
                "test_only": False}
    if any(not _exact(status.get(key), value) for key, value in expected.items()):
        raise ValueError("Preflight worker incomplete or not successful")
    inputs = _json(output / "input-binding.json")
    if (not _exact(inputs.get("schema"), 1) or inputs.get("mode") != mode
            or inputs.get("code_sha256") != binding["code_sha256"]
            or inputs.get("input_sha256") != binding["input_sha256"]):
        raise ValueError("Preflight code/input SHA binding differs from launch")
    cells = _json(output / "microcells.json")
    if (not _exact(cells.get("schema"), 1) or cells.get("mode") != mode
            or not isinstance(cells.get("cells"), list) or len(cells["cells"]) != 12):
        raise ValueError("Complete 12-cell preflight matrix required")
    _environment(output, mode, binding["gpu_uuid"])
    _supervisor(output, supervisor, mode, binding["gpu_uuid"])
    _training(output)
    recorded = _json(output / "verification.json")
    _verification(recorded, mode)
    certified = scope_verify.verify(output, formal=False)
    _verification(certified, mode)
    if any(not _exact(recorded.get(key), value) for key, value in certified.items()):
        raise ValueError("Recorded certification differs from independent raw recheck")
    _inventory(output, supervisor, entry["raw_sha256"])


def verify_launch_binding(binding_path: Path, *, root: Path, config_path: Path) -> Json:
    """Certify frozen launch evidence, never permission; missing raw data fails closed."""
    root, binding_path, config_path = _path(root), _path(binding_path), _path(config_path)
    if not root.is_dir() or not binding_path.is_relative_to(root) or not config_path.is_relative_to(root):
        raise ValueError("Launch binding/config must reside inside isolated root")
    original_sha = contract.sha256(binding_path)
    binding = _json(binding_path)
    flags = {"schema": 1, "experiment": EXPERIMENT, "launch_ready": True,
             "execution_authorized": False, "formal_executed": False,
             "resource_budget": contract.FROZEN["limits"]}
    if any(not _exact(binding.get(key), value) for key, value in flags.items()):
        raise ValueError("Launch status/schema/experiment/resource budget changed")
    _uuid(binding.get("gpu_uuid"))
    code = binding.get("code_sha256")
    if not isinstance(code, dict) or set(code) != set(contract.CODE_FILES):
        raise ValueError("Launch code_sha256 must match exact runtime closure")
    for relative, expected in code.items():
        _sha(_inside(root, relative), expected)
    protocol = _inside(root, contract.PROTOCOL)
    config_path = _inside(root, str(config_path.relative_to(root)))
    data_path = _inside(root, "configs/meta65_server_data.json")
    _sha(config_path, binding.get("config_sha256"))
    _sha(protocol, binding.get("protocol_sha256"))
    _sha(protocol, contract.PROTOCOL_SHA)
    _sha(data_path, binding.get("data_config_sha256"))
    contract.load_config(config_path, root)
    inputs = _input_map(binding.get("input_sha256"))
    complete = contract.bind_inputs(root=root, config_path=config_path)
    if not complete or inputs != complete:
        raise ValueError("Launch input SHA coverage differs from complete contract binding")
    preflights = binding.get("preflights")
    if not isinstance(preflights, dict) or set(preflights) != set(MODES):
        raise ValueError("Both CPU and GPU preflights required")
    outputs = [_path(preflights[mode]["output"]) for mode in MODES if isinstance(preflights[mode], dict)
               and isinstance(preflights[mode].get("output"), str)]
    if len(outputs) != 2 or outputs[0].is_relative_to(outputs[1]) or outputs[1].is_relative_to(outputs[0]):
        raise ValueError("CPU and GPU preflight outputs must be distinct/nonoverlapping")
    for mode in MODES:
        _preflight(preflights[mode], mode, binding)
    _input_map(inputs)
    _sha(binding_path, original_sha)
    return binding


def _receipt_path(approval_path: Path) -> Path:
    return _path(approval_path.with_name(approval_path.name + ".consumed.json"))


def _output(output: Path, binding: Json, root: Path, *, worker: bool) -> Path:
    output = _path(output)
    data = _json(_inside(root, "configs/meta65_server_data.json"))
    protected = [contract.FROZEN_ROOT, contract.RETAINED, Path("/gly/weilin/PRICE-master"),
                 *(root / name for name in ("scripts", "spatial_meta_ce", "configs", "实验文档", "tests", "vendor")),
                 *(Path(name) for name in binding["input_sha256"]),
                 *(Path(entry["output"]) for entry in binding["preflights"].values())]
    for key in ("cache_root", "data_root"):
        name = data.get(key)
        if not isinstance(name, str) or not name:
            raise ValueError("Data source/cache root missing")
        protected = [*protected, _path(name)]
    if (output == root or "PRICE-master" in output.parts
            or scope_verify.TARGET_PATTERN.search(str(output))
            or any(output.is_relative_to(_path(path)) or _path(path).is_relative_to(output)
                   for path in protected)):
        raise ValueError("Output overlaps read-only source/cache/PRICE/retained/preflight inputs")
    if output.exists() and (not worker or not output.is_dir()):
        raise ValueError("Formal output must be a fresh path (worker directory only exception)")
    return output


def _approval(approval_path: Path, output: Path, binding_path: Path, config_path: Path,
              binding: Json, gpu_uuid: str, *, worker: bool) -> Json:
    approval_path, binding_path, config_path = _path(approval_path), _path(binding_path), _path(config_path)
    if approval_path.parent != output.parent or approval_path == output or not approval_path.is_file():
        raise ValueError("Existing approval must be beside the unique formal output")
    receipt_path = _receipt_path(approval_path)
    if receipt_path.exists() and not worker:
        raise ValueError("Formal approval already consumed or used")
    if _uuid(gpu_uuid) != binding["gpu_uuid"]:
        raise ValueError("Approved UUID differs from GPU preflight binding")
    expected = {"schema": 1, "permission": PERMISSION,
                "binding_sha256": contract.sha256(binding_path), "code_sha256": binding["code_sha256"],
                "config_sha256": contract.sha256(config_path), "protocol_sha256": binding["protocol_sha256"],
                "gpu_uuid": gpu_uuid, "output": str(output), "resource_budget": binding["resource_budget"]}
    approval = _json(approval_path)
    if not _exact(approval, expected):
        raise ValueError("Approval must exactly bind scope launch, SHA, output, UUID and budget")
    return approval


def authorize_formal(approval_path: Path, output: Path, binding_path: Path, config_path: Path,
                     *, root: Path, gpu_uuid: str) -> Json:
    """Parent rechecks everything and atomically consumes an operator's explicit approval."""
    root, approval_path = _path(root), _path(approval_path)
    binding = verify_launch_binding(binding_path, root=root, config_path=config_path)
    output = _output(output, binding, root, worker=False)
    approval = _approval(approval_path, output, binding_path, config_path, binding, gpu_uuid, worker=False)
    approved_sha = contract.sha256(approval_path)
    verify_gpu_idle(gpu_uuid)
    # Idle queries may take seconds. Recheck against any intervening artifact/input change.
    if verify_launch_binding(binding_path, root=root, config_path=config_path) != binding:
        raise ValueError("Launch binding changed before consumption")
    _output(output, binding, root, worker=False)
    if _approval(approval_path, output, binding_path, config_path, binding, gpu_uuid, worker=False) != approval:
        raise ValueError("Approval changed before consumption")
    _sha(approval_path, approved_sha)
    receipt = {"schema": 1, "status": "consumed", "approval_sha256": approved_sha,
               "binding_sha256": approval["binding_sha256"], "output": str(output),
               "gpu_uuid": gpu_uuid, "supervisor_pid": os.getpid()}
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(_receipt_path(approval_path), flags, 0o600)
    except FileExistsError as exc:
        raise ValueError("Formal approval already consumed or used") from exc
    # Any write failure leaves a poison receipt: never silently retry permission.
    with os.fdopen(fd, "w", encoding="utf-8") as stream:
        json.dump(receipt, stream, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    return receipt


def verify_worker_approval(approval_path: Path, output: Path, binding_path: Path, config_path: Path,
                           *, root: Path, gpu_uuid: str, parent_pid: int) -> Json:
    """Read-only worker verification; cannot create, consume, or renew approval."""
    if (type(parent_pid) is not int or parent_pid <= 0 or parent_pid != os.getppid()
            or os.environ.get("META65_SCOPE_SUPERVISOR_PID") != str(parent_pid)
            or os.environ.get("META65_SCOPE_SUPERVISED_MODE") != "formal"
            or os.environ.get("CUDA_VISIBLE_DEVICES") != gpu_uuid):
        raise ValueError("Worker approval requires owning actual supervisor PID and environment")
    root, approval_path = _path(root), _path(approval_path)
    binding = verify_launch_binding(binding_path, root=root, config_path=config_path)
    output = _output(output, binding, root, worker=True)
    approval = _approval(approval_path, output, binding_path, config_path, binding, gpu_uuid, worker=True)
    expected = {"schema": 1, "status": "consumed", "approval_sha256": contract.sha256(approval_path),
                "binding_sha256": approval["binding_sha256"], "output": str(output),
                "gpu_uuid": gpu_uuid, "supervisor_pid": parent_pid}
    receipt = _json(_receipt_path(approval_path))
    if not _exact(receipt, expected):
        raise ValueError("Consumed receipt differs or belongs to another supervisor")
    return receipt
