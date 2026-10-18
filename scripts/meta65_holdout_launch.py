"""Launch evidence and one-shot approval for the Meta65 source-holdout run."""
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = "meta65-source-holdout-20261018"
PERMISSION = "one_shot_meta65_source_holdout_formal"
SHA = re.compile(r"[0-9a-f]{64}\Z")
GPU = re.compile(r"GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}\Z")


def checked_path(path: Path | str) -> Path:
    path = Path(path).absolute()
    if any(item.is_symlink() for item in (path, *path.parents)):
        raise ValueError(f"Symlink forbidden: {path}")
    return path.resolve()


def sha256(path: Path | str) -> str:
    value = hashlib.sha256()
    with checked_path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def read_json(path: Path | str) -> dict:
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result

    value = json.loads(
        checked_path(path).read_text(encoding="utf-8"),
        object_pairs_hook=pairs,
    )
    if not isinstance(value, dict):
        raise TypeError("JSON root must be an object")
    return value


def write_json(path: Path | str, value: dict) -> None:
    path = checked_path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    if temporary.exists():
        raise FileExistsError(f"Temporary path exists: {temporary}")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False,
                                    allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _sha(value: object, label: str) -> str:
    if not isinstance(value, str) or SHA.fullmatch(value) is None:
        raise ValueError(f"Invalid {label} SHA")
    return value


def _uuid(value: object) -> str:
    if not isinstance(value, str) or GPU.fullmatch(value) is None:
        raise ValueError("Invalid GPU UUID")
    return value


def _inventory(output: Path, sidecar: Path, manifest: object) -> None:
    if not isinstance(manifest, dict):
        raise TypeError("Raw inventory missing")
    expected = set(manifest)
    actual = {
        str(path.relative_to(output)): path
        for path in output.rglob("*") if path.is_file()
    }
    actual[str(sidecar.name)] = sidecar
    log = sidecar.with_name(sidecar.name.removesuffix(".supervisor.json") + ".supervisor.log")
    actual[log.name] = log
    if set(actual) != expected:
        raise ValueError("Raw output inventory changed")
    for name, path in actual.items():
        if sha256(path) != _sha(manifest[name], f"raw {name}"):
            raise ValueError(f"Raw output SHA changed: {name}")


def _input_map(value: object) -> dict[str, str]:
    if not isinstance(value, dict) or not value:
        raise ValueError("Input SHA map missing")
    for name, expected in value.items():
        if not isinstance(name, str) or not Path(name).is_absolute():
            raise ValueError("Input binding paths must be absolute")
        _sha(expected, "input")
    return value


def _verify_preflight(entry: object, mode: str, root: Path) -> None:
    if not isinstance(entry, dict) or set(entry) != {"output", "supervisor_file", "raw_sha256"}:
        raise ValueError("Preflight entry schema changed")
    output = checked_path(entry["output"])
    sidecar = checked_path(entry["supervisor_file"])
    if not output.is_relative_to(root) or not sidecar.is_relative_to(root):
        raise ValueError("Preflight evidence must reside in isolated root")
    _inventory(output, sidecar, entry["raw_sha256"])
    status = read_json(output / "status.json")
    verification = read_json(output / "verification.json")
    supervisor = read_json(sidecar)
    if (status.get("status") != "preflight_completed" or status.get("mode") != mode
            or status.get("cells") != 0 or status.get("micro_cells") != 10
            or status.get("formal_result") is not False):
        raise ValueError("Preflight status incomplete")
    if (verification.get("status") != "preflight_verified"
            or verification.get("mode") != mode
            or verification.get("cells_verified") != 10
            or verification.get("all_metrics_verified") is not True
            or verification.get("formal_result") is not False):
        raise ValueError("Preflight independent verification incomplete")
    if (supervisor.get("status") != "preflight_completed" or supervisor.get("mode") != mode
            or supervisor.get("returncode") != 0 or supervisor.get("failure") is not None):
        raise ValueError("Preflight supervisor did not complete cleanly")
    from scripts import meta65_holdout_verify as verifier
    verifier.verify(output, formal=False)


def verify_launch_binding(binding_path: Path | str, *, root: Path, config_path: Path) -> dict:
    root = checked_path(root)
    binding_path = checked_path(binding_path)
    config_path = checked_path(config_path)
    if not binding_path.is_relative_to(root) or not config_path.is_relative_to(root):
        raise ValueError("Binding/config must reside in isolated root")
    binding = read_json(binding_path)
    expected_flags = {
        "schema": 1, "experiment": EXPERIMENT, "launch_ready": True,
        "execution_authorized": False, "formal_executed": False,
    }
    for key, value in expected_flags.items():
        if binding.get(key) != value:
            raise ValueError(f"Launch binding flag changed: {key}")
    data_path = checked_path(root / "configs/meta65_source_holdout_data.json")
    plan_path = checked_path(binding["plan_file"])
    if not plan_path.is_relative_to(root):
        raise ValueError("Plan must reside in isolated root")
    from scripts import meta65_holdout_contract as contract
    contract.load_config(config_path, root)
    if sha256(root / contract.PROTOCOL) != contract.PROTOCOL_SHA:
        raise ValueError("Protocol SHA changed")
    if sha256(config_path) != binding.get("config_sha256"):
        raise ValueError("Config SHA changed")
    if sha256(data_path) != binding.get("data_config_sha256"):
        raise ValueError("Data config SHA changed")
    if sha256(plan_path) != binding.get("plan_sha256"):
        raise ValueError("Plan SHA changed")
    code = binding.get("code_sha256")
    if not isinstance(code, dict) or set(code) != set(contract.CODE_FILES):
        raise ValueError("Code closure changed")
    for name, expected in code.items():
        if sha256(root / name) != expected:
            raise ValueError(f"Code SHA changed: {name}")
    complete = contract.input_binding(root, config_path, data_path, plan_path)
    if binding.get("input_sha256") != complete:
        raise ValueError("Input SHA closure changed")
    if binding.get("resource_budget") != contract.FROZEN["limits"]:
        raise ValueError("Resource budget changed")
    _uuid(binding.get("gpu_uuid"))
    preflights = binding.get("preflights")
    if not isinstance(preflights, dict) or set(preflights) != {"cpu_preflight", "gpu_preflight"}:
        raise ValueError("Both preflights required")
    for mode in ("cpu_preflight", "gpu_preflight"):
        _verify_preflight(preflights[mode], mode, root)
    return binding


def verify_gpu_idle(uuid: str) -> None:
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


def _receipt_path(approval: Path) -> Path:
    return approval.with_name(approval.name + ".consumed.json")


def _expected_approval(output: Path, binding_path: Path, config_path: Path,
                       binding: dict, gpu_uuid: str) -> dict:
    return {
        "schema": 1,
        "permission": PERMISSION,
        "binding_sha256": sha256(binding_path),
        "code_sha256": binding["code_sha256"],
        "config_sha256": sha256(config_path),
        "protocol_sha256": binding["protocol_sha256"],
        "plan_sha256": binding["plan_sha256"],
        "gpu_uuid": gpu_uuid,
        "output": str(output),
        "resource_budget": binding["resource_budget"],
    }


def _approval_payload(approval: Path, output: Path, binding_path: Path,
                      config_path: Path, binding: dict, gpu_uuid: str) -> dict:
    approval = checked_path(approval)
    if approval.parent != output.parent or not approval.is_file():
        raise ValueError("Approval must be beside the unique output")
    receipt = _receipt_path(approval)
    if receipt.exists() or receipt.is_symlink():
        raise ValueError("Approval already consumed")
    expected = _expected_approval(output, binding_path, config_path, binding, gpu_uuid)
    if read_json(approval) != expected:
        raise ValueError("Approval does not exactly bind launch/output/UUID/budget")
    return expected


def authorize_formal(approval: Path, output: Path, binding_path: Path, config_path: Path,
                     *, root: Path, gpu_uuid: str) -> dict:
    root = checked_path(root)
    output = checked_path(output)
    binding_path = checked_path(binding_path)
    config_path = checked_path(config_path)
    binding = verify_launch_binding(binding_path, root=root, config_path=config_path)
    if gpu_uuid != _uuid(binding["gpu_uuid"]):
        raise ValueError("Approval UUID differs from GPU preflight")
    verify_gpu_idle(gpu_uuid)
    expected = _approval_payload(checked_path(approval), output, binding_path, config_path, binding, gpu_uuid)
    approval_path = checked_path(approval)
    receipt = _receipt_path(approval_path)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(receipt, flags, 0o600)
    except FileExistsError as error:
        raise ValueError("Approval already consumed") from error
    payload = {"schema": 1, "status": "consumed", "approval_sha256": sha256(approval_path),
               **{key: expected[key] for key in ("binding_sha256", "output", "gpu_uuid")},
               "supervisor_pid": os.getpid()}
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        json.dump(payload, stream, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    return payload


def verify_worker_approval(approval: Path, output: Path, binding_path: Path, config_path: Path,
                           *, root: Path, gpu_uuid: str | None, parent_pid: int) -> dict:
    if (not isinstance(parent_pid, int) or parent_pid <= 0
            or parent_pid != os.getppid()
            or os.environ.get("META65_SUPERVISOR_PID") != str(parent_pid)
            or os.environ.get("META65_SUPERVISED_MODE") != "formal"
            or os.environ.get("CUDA_VISIBLE_DEVICES") != gpu_uuid):
        raise ValueError("Worker approval requires owning supervisor")
    binding = verify_launch_binding(binding_path, root=root, config_path=config_path)
    bound_uuid = _uuid(binding["gpu_uuid"])
    if gpu_uuid != bound_uuid:
        raise ValueError("Worker UUID differs from launch binding")
    approval_path = checked_path(approval)
    expected = _expected_approval(checked_path(output), checked_path(binding_path),
                                  checked_path(config_path), binding, bound_uuid)
    receipt = read_json(_receipt_path(approval_path))
    required = {"schema": 1, "status": "consumed", "approval_sha256": sha256(approval_path),
                "binding_sha256": expected["binding_sha256"], "output": expected["output"],
                "gpu_uuid": expected["gpu_uuid"], "supervisor_pid": parent_pid}
    if receipt != required:
        raise ValueError("Consumed approval receipt differs")
    return receipt
