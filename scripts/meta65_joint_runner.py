#!/usr/bin/env python3
"""Fixed-checkpoint evaluation only. Formal execution needs a separate approval."""
import argparse
import hashlib
import json
import os
import platform
import resource
import subprocess
import sys
import time
from collections import OrderedDict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DATABASES = ("chembl", "legalacts", "telstra")
CHECKPOINTS = {"A": "A_all5_head3", "B": "B_head3_head3", "C": "C_last3_last3"}
OPERATORS = ("none", "head3", "last3")
COUNTS = {"chembl": 49900, "legalacts": 49404, "telstra": 49900}
CHECKPOINT_SHA = {
    "A": "ae0c386d35ac368964b4160d634ca063d7801c65e5296ac679c459dd01951b12",
    "B": "a2d777a4dcaad6d23c3f1212929d062768f5157a3fc37bce9909903a22f18db9",
    "C": "58f2f4db6f54cd4578fd6127d5a4aa350dbdc2268a36a057b56f7161cb6097a4",
}
RETAINED = Path("/gly/weilin/spatial-meta-ce-meta65-20261009-frozen/runs/meta65-source-screen-20261009")
FROZEN_ROOT = RETAINED.parents[1]
CACHE = Path("/gly/weilin/spatial-meta-ce/artifacts/cache/pretrain")
PLAN_SHA = "ebeb178de15b134722bd9fbc24ff4b4752f795a311580c1a17cb23c550bfd0f0"
PROTOCOL = "实验文档/56-Meta65同初始化适应算子最小受控协议.md"
LAUNCH_BINDING = "reports/meta65-cublas-preflight-20261011/launch-binding.json"
PREFLIGHT_EVIDENCE_DIR = "reports/meta65-cublas-preflight-20261011/cpu-preflight"
GPU_PREFLIGHT_EVIDENCE_DIR = "reports/meta65-cublas-preflight-20261011/gpu-preflight"
GPU_PREFLIGHT_RUNS = Path("/gly/weilin/spatial-meta-ce-meta65-cublas-20261011-runs")
CUDA_WORKSPACE_CONFIG = ":4096:8"
PROTOCOL_SHA = "cc87ef93451e2664aba4aa51257a1cdba47c6638c004cb8e4b943773171ca4d2"
FROZEN = {
    "schema": 1, "experiment": "meta65-joint-control-20261010", "k": 20,
    "databases": list(DATABASES), "checkpoints": list(CHECKPOINTS), "operators": list(OPERATORS),
    "runs": list(range(5)), "batch_size": 512, "inner_lr": 0.001, "inner_steps": 3,
    "loss": "smooth_l1", "cuda_workspace_config": CUDA_WORKSPACE_CONFIG,
    "model": {"arch": "sum", "use_pg": True, "width": 64, "layers": 2, "heads": 4},
    "protocol_file": PROTOCOL, "protocol_sha256": PROTOCOL_SHA,
    "limits": {"total_seconds": 4800, "preparation_seconds": 300,
               "evaluation_seconds": 3600, "certification_seconds": 900,
               "rss_bytes": 4 * 1024**3, "vram_bytes": 8 * 1024**3,
               "output_bytes": 1024**3},
    "cpu_preflight": {"micro_queries": 32, "total_seconds": 900,
                      "preparation_seconds": 300, "evaluation_seconds": 120,
                      "certification_seconds": 300},
    "gpu_preflight": {"micro_queries": 32, "total_seconds": 900,
                      "preparation_seconds": 300, "evaluation_seconds": 300,
                      "certification_seconds": 300},
}
CODE_FILES = (
    "scripts/meta65_joint_runner.py", "scripts/meta65_joint_supervisor.py", "scripts/meta65_joint_verify.py",
    "spatial_meta_ce/data.py", "spatial_meta_ce/manifests.py", "spatial_meta_ce/mappings.py",
    "spatial_meta_ce/model.py", "spatial_meta_ce/meta.py", "spatial_meta_ce/checkpoints.py",
    "tests/test_meta65_joint_runner.py", "tests/test_meta65_joint_supervisor.py",
    "tests/test_meta65_joint_verify.py",
)
PROVENANCE = (
    "实验文档/results/meta65-source-screen-20261009/source-before.json",
    "实验文档/results/meta65-telstra-diagnosis-20261010/telstra_diagnosis.json",
    "实验文档/results/meta65-endpoint-completion-20261009/inputs.json",
)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def read_json(path):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result

    def constant(value):
        raise ValueError(f"Nonfinite JSON: {value}")

    return json.loads(Path(path).read_text(), object_pairs_hook=pairs, parse_constant=constant)


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    if temporary.exists():
        raise ValueError("Unexpected temporary output")
    temporary.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + "\n")
    temporary.replace(path)


def checked_path(path):
    path = Path(path).absolute()
    # `/var` is a system alias for `/private/var` on macOS; rejecting every
    # symlink in the parent chain would reject otherwise safe temp paths.
    if any(item.is_symlink() and item not in (Path("/var"), Path("/tmp"))
           for item in (path, *path.parents)):
        raise ValueError(f"Symlink forbidden: {path}")
    return path.resolve()


def safe_output(path, inputs):
    path = checked_path(path)
    if path.exists() or any(path.is_relative_to(checked_path(item))
                            or checked_path(item).is_relative_to(path) for item in inputs):
        raise ValueError("Output exists or overlaps frozen inputs")
    if path.is_relative_to(checked_path(FROZEN_ROOT)):
        raise ValueError("frozen root is read-only")
    if "PRICE-master" in path.parts or path.is_relative_to(ROOT / "spatial_meta_ce"):
        raise ValueError("Read-only/source output path")
    return path


def verify_launch_binding(binding_path, *, root=ROOT, config_path=None):
    """Recheck the committed launch evidence before *each* formal invocation.

    The file records a launch configuration, not permission to execute; that
    permission still requires a separate direct user confirmation and receipt.
    """
    root = checked_path(root)
    binding_path = checked_path(binding_path)
    if not binding_path.is_relative_to(root):
        raise ValueError("Launch binding must reside in the isolated code root")

    def bound_file(relative):
        relative_path = Path(relative)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError("Launch binding path escapes isolated code root")
        path = checked_path(root / relative_path)
        if not path.is_relative_to(root):
            raise ValueError("Launch binding path escapes isolated code root")
        return path

    binding = read_json(binding_path)
    if (binding.get("launch_ready") is not True
            or binding.get("execution_authorized") is not False
            or binding.get("gpu_formal_executed") is not False
            or binding.get("resource_budget") != FROZEN["limits"]):
        raise ValueError("Launch binding status or resource budget changed")
    code = binding.get("code_sha256")
    if not isinstance(code, dict) or set(code) != set(CODE_FILES):
        raise ValueError("Launch binding code SHA set changed")
    for relative, expected in code.items():
        if sha256(bound_file(relative)) != expected:
            raise ValueError(f"Launch binding code SHA changed: {relative}")
    sources = binding.get("input_sha256")
    config_path = checked_path(config_path or root / "configs/meta65_joint_control.json")
    if not config_path.is_relative_to(root):
        raise ValueError("Launch binding config must reside in isolated code root")
    config_rel = str(config_path.relative_to(root))
    if (not isinstance(sources, dict)
            or not {config_rel, PROTOCOL, *PROVENANCE} <= set(sources)
            or sources[PROTOCOL] != PROTOCOL_SHA):
        raise ValueError("Launch binding config/protocol SHA missing")
    for relative, expected in sources.items():
        if sha256(bound_file(relative)) != expected:
            raise ValueError(f"Launch binding input SHA changed: {relative}")
    evidence = binding.get("cpu_preflight_evidence_sha256")
    evidence_paths = {str(Path(PREFLIGHT_EVIDENCE_DIR) / filename) for filename in (
        "status.json", "verification.json", "input-binding.json", "microcells.json",
        "environment.json", "supervisor.json")}
    if not isinstance(evidence, dict) or set(evidence) != evidence_paths:
        raise ValueError("Launch binding lacks the fixed CPU preflight evidence set")
    for relative, expected in evidence.items():
        if sha256(bound_file(relative)) != expected:
            raise ValueError(f"Launch binding preflight SHA changed: {relative}")
    evidence_root = root / PREFLIGHT_EVIDENCE_DIR
    status = read_json(evidence_root / "status.json")
    verification = read_json(evidence_root / "verification.json")
    preflight_binding = read_json(evidence_root / "input-binding.json")
    microcells = read_json(evidence_root / "microcells.json")
    environment = read_json(evidence_root / "environment.json")
    supervisor = read_json(evidence_root / "supervisor.json")
    if (status.get("status") != "preflight_completed" or status.get("mode") != "cpu_preflight"
            or status.get("cells") != 0 or status.get("micro_cells") != 9
            or status.get("formal_result") is not False
            or verification.get("status") != "preflight_verified"
            or verification.get("mode") != "cpu_preflight"
            or verification.get("cells_verified") != 9
            or verification.get("all_metrics_verified") is not True
            or verification.get("formal_result") is not False
            or verification.get("summary") is not None
            or microcells.get("mode") != "cpu_preflight"
            or len(microcells.get("cells", [])) != 9
            or len(microcells.get("none_checks", [])) != 9
            or environment.get("mode") != "cpu_preflight"
            or environment.get("CUDA_VISIBLE_DEVICES") != ""
            or environment.get("threads") != 1
            or supervisor.get("status") != "preflight_completed"
            or supervisor.get("mode") != "cpu_preflight"
            or supervisor.get("returncode") != 0
            or supervisor.get("failure") is not None
            or supervisor.get("phase") != "certification"
            or supervisor.get("peak_vram_bytes") != 0
            or not 0 <= supervisor.get("peak_rss_bytes", -1) <= FROZEN["limits"]["rss_bytes"]
            or not 0 <= supervisor.get("outputbytes", -1) <= FROZEN["limits"]["output_bytes"]
            or not 0 < supervisor.get("elapsed", 0) <= FROZEN["cpu_preflight"]["total_seconds"]):
        raise ValueError("CPU preflight success/status/resource evidence is not certified")
    if (preflight_binding.get("mode") != "cpu_preflight"
            or preflight_binding.get("code_sha256") != code):
        raise ValueError("CPU preflight code SHA is not bound to this launch")
    preflight_inputs = preflight_binding.get("input_sha256")
    if not isinstance(preflight_inputs, dict):
        raise TypeError("CPU preflight input SHA evidence is missing")
    for relative, expected in {**code, config_rel: sources[config_rel], PROTOCOL: sources[PROTOCOL]}.items():
        matches = [value for name, value in preflight_inputs.items()
                   if name.endswith("/" + relative)]
        if matches != [expected]:
            raise ValueError(f"CPU preflight input SHA differs from launch binding: {relative}")
    _verify_gpu_preflight_evidence(binding, root, code, sources, config_rel)
    return binding


def _verify_gpu_preflight_evidence(binding, root, code, sources, config_rel):
    uuid = binding.get("gpu_preflight_uuid")
    evidence = binding.get("gpu_preflight_evidence_sha256")
    names = ("status.json", "verification.json", "input-binding.json", "microcells.json",
             "environment.json", "supervisor.json")
    paths = {str(Path(GPU_PREFLIGHT_EVIDENCE_DIR) / name) for name in names}
    if not isinstance(uuid, str) or not uuid.startswith("GPU-") or not isinstance(evidence, dict) or set(evidence) != paths:
        raise ValueError("GPU preflight identity or evidence set missing")
    for relative, expected in evidence.items():
        path = checked_path(root / relative)
        if not path.is_relative_to(root) or sha256(path) != expected:
            raise ValueError(f"GPU preflight evidence SHA changed: {relative}")
    folder = root / GPU_PREFLIGHT_EVIDENCE_DIR
    status, verified, inputs, micro, env, supervisor = [read_json(folder / name) for name in names]
    if (status.get("status") != "preflight_completed" or status.get("mode") != "gpu_preflight"
            or status.get("cells") != 0 or status.get("micro_cells") != 9
            or status.get("formal_result") is not False
            or verified.get("status") != "preflight_verified" or verified.get("mode") != "gpu_preflight"
            or verified.get("cells_verified") != 9 or verified.get("all_metrics_verified") is not True
            or verified.get("formal_result") is not False or verified.get("summary") is not None
            or micro.get("mode") != "gpu_preflight" or len(micro.get("cells", [])) != 9
            or len(micro.get("none_checks", [])) != 9
            or env.get("mode") != "gpu_preflight" or env.get("CUDA_VISIBLE_DEVICES") != uuid
            or env.get("CUBLAS_WORKSPACE_CONFIG") != CUDA_WORKSPACE_CONFIG or env.get("threads") != 1
            or env.get("deterministic_algorithms") is not True or env.get("tf32") is not False
            or supervisor.get("status") != "preflight_completed" or supervisor.get("mode") != "gpu_preflight"
            or supervisor.get("returncode") != 0 or supervisor.get("failure") is not None
            or supervisor.get("phase") != "certification"
            or not 0 <= supervisor.get("peak_rss_bytes", -1) <= FROZEN["limits"]["rss_bytes"]
            or not 0 <= supervisor.get("peak_vram_bytes", -1) <= FROZEN["limits"]["vram_bytes"]
            or not 0 <= supervisor.get("outputbytes", -1) <= FROZEN["limits"]["output_bytes"]
            or not 0 < supervisor.get("elapsed", 0) <= FROZEN["gpu_preflight"]["total_seconds"]):
        raise ValueError("GPU preflight status, environment or resource evidence invalid")
    if inputs.get("mode") != "gpu_preflight" or inputs.get("code_sha256") != code:
        raise ValueError("GPU preflight code SHA binding invalid")
    recorded = inputs.get("input_sha256")
    if not isinstance(recorded, dict):
        raise ValueError("GPU preflight input SHA binding missing")
    required = {**code, **{name: sources[name] for name in (config_rel, PROTOCOL, *PROVENANCE)}}
    for relative, expected in required.items():
        if [value for name, value in recorded.items() if name.endswith("/" + relative)] != [expected]:
            raise ValueError(f"GPU preflight input SHA differs: {relative}")


def verify_gpu_preflight_output(binding, *, allowed_root=GPU_PREFLIGHT_RUNS,
                                archive_root=ROOT / GPU_PREFLIGHT_EVIDENCE_DIR):
    """Recompute the bound GPU microcheck from its retained raw outputs."""
    name = binding.get("gpu_preflight_output")
    manifest = binding.get("gpu_preflight_artifact_sha256")
    if not isinstance(name, str) or not isinstance(manifest, dict):
        raise ValueError("GPU preflight raw output binding missing")
    output = checked_path(name)
    base = checked_path(allowed_root)
    if not output.is_dir() or output == base or not output.is_relative_to(base):
        raise ValueError("GPU preflight raw output path is invalid")
    _verify_gpu_output_inventory(output, manifest)
    _verify_gpu_output_evidence(output, binding, archive_root)
    from scripts.meta65_joint_verify import verify
    if verify(output) != read_json(output / "verification.json"):
        raise ValueError("GPU preflight raw metrics differ from archived verification")


def _verify_gpu_output_inventory(output, manifest):
    entries = list(output.rglob("*"))
    if any(path.is_symlink() for path in entries):
        raise ValueError("GPU preflight raw output contains symlinks")
    actual = {str(path.relative_to(output)) for path in entries if path.is_file()}
    metadata = {"status.json", "verification.json", "environment.json", "input-binding.json",
                "microcells.json", "phase.json"}
    identities = {name for name in actual if name.startswith("identities/") and name.endswith(".npz")}
    predictions = {name for name in actual if name.startswith("predictions/") and name.endswith(".npz")}
    if (set(manifest) != actual or not metadata <= actual
            or len(identities) != 15 or len(predictions) != 9
            or actual != metadata | identities | predictions):
        raise ValueError("GPU preflight raw output inventory changed")
    for relative, expected in manifest.items():
        path = checked_path(output / relative)
        if not path.is_relative_to(output) or sha256(path) != expected:
            raise ValueError(f"GPU preflight raw output SHA changed: {relative}")


def _verify_gpu_output_evidence(output, binding, archive_root):
    archived = checked_path(archive_root)
    evidence = binding.get("gpu_preflight_evidence_sha256", {})
    for name in ("status.json", "verification.json", "input-binding.json",
                 "microcells.json", "environment.json"):
        expected = evidence.get(str(Path(GPU_PREFLIGHT_EVIDENCE_DIR) / name))
        if not expected or sha256(archived / name) != expected or sha256(output / name) != expected:
            raise ValueError(f"GPU preflight raw and archived evidence differ: {name}")
    sidecar = output.with_name(output.name + ".supervisor.json")
    expected_sidecar = evidence.get(str(Path(GPU_PREFLIGHT_EVIDENCE_DIR) / "supervisor.json"))
    if (sidecar.is_symlink() or not expected_sidecar
            or sha256(archived / "supervisor.json") != expected_sidecar
            or sha256(sidecar) != expected_sidecar):
        raise ValueError("GPU preflight supervisor sidecar changed")
    status, environment_record, supervisor = [read_json(path) for path in
        (output / "status.json", output / "environment.json", sidecar)]
    import numpy as np
    import torch
    if (environment_record.get("python") != sys.version
            or environment_record.get("numpy") != np.__version__
            or environment_record.get("torch") != torch.__version__
            or environment_record.get("cuda") != torch.version.cuda):
        raise ValueError("GPU preflight runtime versions differ from formal environment")
    if (status.get("status") != "preflight_completed" or status.get("mode") != "gpu_preflight"
            or status.get("formal_result") is not False or status.get("cells") != 0
            or status.get("micro_cells") != 9
            or environment_record.get("CUDA_VISIBLE_DEVICES") != binding.get("gpu_preflight_uuid")
            or environment_record.get("CUBLAS_WORKSPACE_CONFIG") != CUDA_WORKSPACE_CONFIG
            or supervisor.get("status") != "preflight_completed" or supervisor.get("returncode") != 0
            or supervisor.get("failure") is not None or supervisor.get("output") != str(output)):
        raise ValueError("GPU preflight raw status or environment invalid")


def _formal_receipt_path(approval_path):
    approval_path = checked_path(approval_path)
    return approval_path.with_name(approval_path.name + ".consumed.json")


def validate_formal_approval(approval_path, output, launch_binding_path, config_path,
                             code_sha256, *, gpu_uuid, allow_consumed=False):
    """Match an operator-created receipt to the frozen launch and unique output."""
    approval_path = checked_path(approval_path)
    output = checked_path(output)
    if approval_path.parent != output.parent or not approval_path.is_file():
        raise ValueError("Formal approval must be an existing file beside the isolated output")
    receipt = _formal_receipt_path(approval_path)
    if receipt.is_symlink() or (receipt.exists() and not allow_consumed):
        raise ValueError("Formal approval already consumed or used")
    try:
        binding_sha = sha256(checked_path(launch_binding_path))
    except OSError as exc:
        raise ValueError("Launch binding SHA cannot be checked") from exc
    approval = read_json(approval_path)
    expected = {
        "schema": 2,
        "permission": "one_shot_formal_gpu_matrix",
        "launch_binding_sha256": binding_sha,
        "config_sha256": sha256(checked_path(config_path)),
        "code_sha256": code_sha256,
        "protocol_sha256": PROTOCOL_SHA,
        "output": str(output),
        "gpu_uuid": gpu_uuid,
    }
    if (not gpu_uuid or not gpu_uuid.startswith("GPU-")
            or set(approval) != set(expected) or approval != expected):
        raise ValueError("Formal approval does not bind this launch, SHA or GPU UUID")
    return approval


def claim_formal_approval(approval_path, output, approval):
    """Atomically consume approval *before* supervising; never auto-retry it."""
    approval_path = checked_path(approval_path)
    output = checked_path(output)
    receipt = _formal_receipt_path(approval_path)
    if receipt.is_symlink() or receipt.exists():
        raise ValueError("Formal approval already consumed or used")
    if read_json(approval_path) != approval or approval_path.parent != output.parent:
        raise ValueError("Formal approval changed before consumption")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(receipt, flags, 0o600)
    except FileExistsError as exc:
        raise ValueError("Formal approval already consumed or used") from exc
    with os.fdopen(fd, "w", encoding="utf-8") as stream:
        json.dump({"schema": 1, "status": "consumed", "approval_sha256": sha256(approval_path),
                   "output": str(output), "supervisor_pid": os.getpid()}, stream,
                  sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    return receipt


def verify_formal_claim(approval_path, output, *, parent_pid):
    receipt = read_json(_formal_receipt_path(approval_path))
    if (receipt.get("schema") != 1 or receipt.get("status") != "consumed"
            or receipt.get("approval_sha256") != sha256(checked_path(approval_path))
            or receipt.get("output") != str(checked_path(output))
            or receipt.get("supervisor_pid") != parent_pid):
        raise ValueError("Formal approval consumption receipt changed or used by another supervisor")
    return receipt


def verify_gpu_idle(gpu_uuid):
    """Check the specifically approved device immediately before claiming it."""
    if not isinstance(gpu_uuid, str) or not gpu_uuid.startswith("GPU-"):
        raise ValueError("Explicit GPU UUID required for formal execution")
    details = subprocess.run(
        ["nvidia-smi", "-i", gpu_uuid,
         "--query-gpu=uuid,memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
        check=True, capture_output=True, text=True, timeout=5)
    lines = details.stdout.strip().splitlines()
    if len(lines) != 1:
        raise ValueError("Selected GPU idle state could not be verified")
    fields = [field.strip() for field in lines[0].split(",")]
    try:
        uuid, memory_mib, utilization = fields
        memory_mib, utilization = int(memory_mib), int(utilization)
    except (ValueError, TypeError) as exc:
        raise ValueError("Selected GPU idle state is malformed") from exc
    if uuid != gpu_uuid or not 0 <= memory_mib <= 100 or utilization != 0:
        raise ValueError("Selected GPU is not idle (<=100 MiB and utilization=0 required)")
    apps = subprocess.run(
        ["nvidia-smi", "-i", gpu_uuid, "--query-compute-apps=pid",
         "--format=csv,noheader,nounits"],
        check=True, capture_output=True, text=True, timeout=5)
    if apps.stdout.strip():
        raise ValueError("Selected GPU is not idle: compute process present")
    return {"uuid": uuid, "memory_used_mib": memory_mib, "utilization_percent": utilization}


def validate_config(value):
    if digest(value) != digest(FROZEN):
        raise ValueError("Frozen configuration changed")
    return value


def source_path(database):
    if database not in DATABASES:
        raise ValueError("Only source validation databases permitted")
    return CACHE / database


def matrix():
    return [(checkpoint, operator, db, run) for checkpoint in CHECKPOINTS
            for operator in OPERATORS for db in DATABASES for run in range(5)]


def recheck(binding):
    for path, expected in binding.items():
        if sha256(checked_path(path)) != expected:
            raise ValueError(f"Input SHA changed: {path}")


def encode_identity(data):
    import numpy as np

    if set(data) != {"query_hash", "query_id", "true", "n_tables"}:
        raise ValueError("Unexpected identity fields")
    true, tables = np.asarray(data["true"]), np.asarray(data["n_tables"])
    n = len(true)
    if (not n or true.dtype != np.float64 or true.ndim != 1
            or any(np.asarray(value).shape != (n,) for value in data.values())
            or not np.isfinite(true).all() or (true < 0).any()
            or not np.isfinite(tables).all() or (tables < 1).any() or (tables > 32767).any()
            or not np.array_equal(tables, tables.astype(np.int16))):
        raise ValueError("Invalid FP64 truth or table counts")
    result = {"true": true, "n_tables": tables.astype(np.int16)}
    for key, width in (("query_hash", 64), ("query_id", 128)):
        values = [str(value).encode("utf-8") for value in data[key]]
        if any(not value or len(value) > width or b"\0" in value for value in values) or len(set(values)) != n:
            raise ValueError("Duplicate, empty or overlong identity")
        result[key] = np.asarray(values, dtype=f"S{width}")
    return result


def certify_plan(plan, data, database, run, expected_n=None):
    import numpy as np

    if database not in DATABASES or plan["database"] != database or plan["run"] != run:
        raise ValueError("Wrong plan database/run")
    n = len(data["true"])
    support, query = plan["support_max"], plan["query"]
    if (len(support) != 100 or len(set(support)) != 100 or len(set(query)) != len(query)
            or any(type(i) is not int or not 0 <= i < n for i in (*support, *query))
            or set(support) & set(query) or len(query) != (expected_n if expected_n is not None else COUNTS[database])):
        raise ValueError("Plan changed support exclusion or denominator")
    hashes = np.asarray(data["query_hash"]).astype(str)
    first = {}
    for index, value in enumerate(hashes):
        first.setdefault(value, index)
    indices = list(first.values())
    if query != [i for i in indices if i not in set(support)] or not set(support) <= set(indices):
        raise ValueError("Plan differs from complete first-unique query population")
    if set(plan["supports"]) != {"5", "10", "20", "50", "100"}:
        raise ValueError("Full nested plan required")
    for k in (5, 10, 20, 50, 100):
        if (plan["supports"][str(k)] != support[:k]
                or plan["support_digests"][str(k)] != digest([(i, hashes[i]) for i in support[:k]])):
            raise ValueError("Support prefix/digest changed")
    if plan["query_digest"] != digest([(i, hashes[i]) for i in query]):
        raise ValueError("Query digest changed")
    return {"query": query, "support": support[:20]}


def cell_metrics(log_pred, identity):
    import numpy as np

    true, tables = identity["true"], identity["n_tables"]
    log_pred = np.asarray(log_pred, dtype=np.float64)
    if log_pred.shape != true.shape or not np.isfinite(log_pred).all():
        raise ValueError("Invalid raw predictions")
    with np.errstate(over="raise", invalid="raise"):
        q = np.exp(np.abs(np.maximum(log_pred, 0) - np.log(np.maximum(true, 1))))

    def row(mask):
        n = int(mask.sum())
        return {"count": n, "status": "empty" if not n else "below_min_n" if n < 30 else "ok",
                **{f"q{p}": float(np.percentile(q[mask], p, method="linear")) if n else None
                   for p in (50, 95)}}

    card = np.searchsorted([100, 1000, 10000, 100000, 1000000], true, side="right")
    table = np.minimum(tables, 5) - 1
    labels = ("<100", "100-1e3", "1e3-1e4", "1e4-1e5", "1e5-1e6", ">=1e6")
    joint = {f"{c}|{t}": row((card == ci) & (table == ti)) for ci, c in enumerate(labels)
             for ti, t in enumerate(("1", "2", "3", "4", "5+"))}
    return {"all": row(np.ones(len(true), dtype=bool)), "4+": row(tables >= 4), "joint": joint}


def mean_metric(rows, bucket, metric):
    values = [row[bucket][metric] for row in rows]
    return sum(values) / len(values) if values and all(v is not None for v in values) else None


def verify_anchor_identity(identity, anchor):
    import numpy as np

    if set(anchor) != {"log_pred", "true", "n_tables", "query_hash", "query_id"}:
        raise ValueError("Unexpected anchor fields")
    order = np.argsort(np.asarray(anchor["query_hash"]).astype(str))
    aligned = {key: np.asarray(value)[order] for key, value in anchor.items()}
    for key in ("query_hash", "query_id", "true", "n_tables"):
        actual, expected = identity[key], aligned[key]
        if key in ("query_hash", "query_id"):
            actual, expected = np.asarray(actual).astype(str), np.asarray(expected).astype(str)
        if not np.array_equal(actual, expected) or key == "true" and expected.dtype != np.float64:
            raise ValueError(f"Regression anchor {key} differs")


def verify_anchor(log_pred, identity, anchor, positions=None):
    import numpy as np

    if set(anchor) != {"log_pred", "true", "n_tables", "query_hash", "query_id"}:
        raise ValueError("Unexpected anchor fields")
    order = np.argsort(np.asarray(anchor["query_hash"]).astype(str))
    aligned = {key: np.asarray(value)[order] for key, value in anchor.items()}
    if positions is not None:
        aligned = {key: value[positions] for key, value in aligned.items()}
    for key in ("query_hash", "query_id", "true", "n_tables"):
        actual, expected = identity[key], aligned[key]
        if key in ("query_hash", "query_id"):
            actual, expected = np.asarray(actual).astype(str), np.asarray(expected).astype(str)
        if not np.array_equal(actual, expected) or key == "true" and expected.dtype != np.float64:
            raise ValueError(f"Regression anchor {key} differs")
    if len(set(np.asarray(anchor["query_hash"]).astype(str))) != len(anchor["true"]):
        raise ValueError("Duplicate anchor queries")
    if not np.allclose(log_pred, aligned["log_pred"], rtol=1e-5, atol=1e-6):
        raise ValueError("Regression raw log_pred tolerance failed")
    expected = cell_metrics(aligned["log_pred"], identity)["all"]["q95"]
    actual = cell_metrics(log_pred, identity)["all"]["q95"]
    if not np.isclose(actual, expected, rtol=1e-4, atol=0):
        raise ValueError("Regression standard Q95 tolerance failed")


def state_digest(model):
    value = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        value.update(name.encode())
        value.update(str(tensor.dtype).encode())
        value.update(str(tuple(tensor.shape)).encode())
        value.update(tensor.detach().cpu().numpy().tobytes())
    return value.hexdigest()


def check_cell_peaks(record):
    """Catch a transient memory peak that the one-second sampler might miss."""
    for field, budget in (("peak_rss_bytes", FROZEN["limits"]["rss_bytes"]),
                          ("peak_vram_bytes", FROZEN["limits"]["vram_bytes"])):
        value = record.get(field)
        if type(value) is not int or value < 0:
            raise ValueError(f"Invalid {field} high-water mark")
        if value > budget:
            raise ValueError(f"{field} exceeded frozen limit")
    return record


def operator_params(model, support, operator):
    import torch

    from spatial_meta_ce.meta import _adaptation_names, adapt
    if operator not in OPERATORS:
        raise ValueError("Unknown operator")
    before = state_digest(model)
    original = OrderedDict(model.named_parameters())
    selected = [] if operator == "none" else _adaptation_names(model, original, operator[:-1])
    params = original if operator == "none" else adapt(
        model, support, lr=0.001, steps=3, scope=operator[:-1], loss="smooth_l1", training=False)
    changed = [name for name, tensor in original.items() if not torch.equal(tensor, params[name])]
    if state_digest(model) != before or not set(changed) <= set(selected):
        raise ValueError("Shared initialization or frozen parameters contaminated")
    record = {"selected_parameter_names": selected,
              "selected_parameter_count": sum(original[name].numel() for name in selected),
              "changed_parameter_names": changed,
              "changed_parameter_count": sum(original[name].numel() for name in changed),
              "steps": 0 if operator == "none" else 3}
    expected = {"none": 0, "head3": 107521, "last3": 65}[operator]
    if record["selected_parameter_count"] != expected:
        raise ValueError("Unexpected adaptation parameter capacity")
    return params, record


def require_cuda_workspace(mode):
    if mode not in ("formal", "gpu_preflight"):
        return
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != CUDA_WORKSPACE_CONFIG:
        raise ValueError(f"CUBLAS_WORKSPACE_CONFIG must be {CUDA_WORKSPACE_CONFIG} before CUDA use")
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        if os.environ.get(name) != "1":
            raise ValueError(f"{name} must be 1 before GPU execution")


def environment(mode):
    require_cuda_workspace(mode)
    import numpy as np
    import torch

    if mode == "cpu_preflight" and os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("CPU preflight requires explicitly hidden CUDA")
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        if os.environ.get(key) != "1":
            raise ValueError("Explicit single-thread environment required")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if mode in ("formal", "gpu_preflight") and (not os.environ.get("CUDA_VISIBLE_DEVICES")
                                                 or not torch.cuda.is_available() or torch.cuda.device_count() != 1):
        raise ValueError("GPU execution requires one explicitly selected CUDA GPU")
    return {"python": sys.version, "executable": sys.executable, "platform": platform.platform(),
            "numpy": np.__version__, "torch": torch.__version__, "cuda": torch.version.cuda,
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "threads": 1, "precision": "FP32 model/FP64 true", "deterministic_algorithms": True,
            "tf32": False, "mode": mode}


def input_binding(root=ROOT):
    old_source, diagnosis, retained = [read_json(root / p) for p in PROVENANCE]
    binding = {str(root / p): sha256(root / p) for p in (*CODE_FILES, *PROVENANCE, PROTOCOL)}
    if binding[str(root / PROTOCOL)] != PROTOCOL_SHA:
        raise ValueError("Frozen protocol changed")
    for checkpoint, arm in CHECKPOINTS.items():
        path = checked_path(RETAINED / arm / "last.pt")
        if path.stat().st_size != 2574664 or sha256(path) != CHECKPOINT_SHA[checkpoint]:
            raise ValueError("Checkpoint SHA/size changed")
        binding[str(path)] = CHECKPOINT_SHA[checkpoint]
    plan_path = checked_path(RETAINED / "endpoint_plan.json")
    if sha256(plan_path) != PLAN_SHA or retained["endpoint_plan_sha256"] != PLAN_SHA:
        raise ValueError("Plan SHA changed")
    binding[str(plan_path)] = PLAN_SHA
    for db in DATABASES:
        for name, expected in old_source[db]["file_sha256"].items():
            path = checked_path(source_path(db) / name)
            if name not in ("manifest.json", "cards.npy", "offsets.npy", *(f"group{i}.npy" for i in range(4))):
                raise ValueError("Unexpected source cache file")
            binding[str(path)] = expected
    anchors = {}
    for name, expected in diagnosis["input_sha256"].items():
        path = Path(name)
        if path.suffix == ".npz" and "-k20-run" in path.name:
            db, shot, run = path.stem.split("-")
            checkpoint = next((c for c, arm in CHECKPOINTS.items() if path.parent.parent.name == arm), None)
            if db not in DATABASES or checkpoint is None or shot != "k20":
                raise ValueError("Unexpected regression anchor")
            key = (checkpoint, db, int(run[3:]))
            if key in anchors:
                raise ValueError("Duplicate anchor")
            anchors[key] = checked_path(path)
            binding[str(path)] = expected
    if set(anchors) != {(c, db, r) for c in CHECKPOINTS for db in DATABASES for r in range(5)}:
        raise ValueError("Incomplete 45-cell regression anchors")
    recheck(binding)
    return binding, read_json(plan_path), anchors


def store_identity(store, database):
    import numpy as np

    if store.cards.dtype != np.float64 or not np.isfinite(store.cards).all() or (store.cards < 0).any():
        raise ValueError("Source cards must be finite FP64")
    tables = np.diff(store.offsets[:, 2])
    if (tables < 1).any() or store.offsets.dtype.kind not in "iu" or (np.diff(store.offsets, axis=0) < 0).any():
        raise ValueError("Invalid offsets/table counts")
    first = {}
    for index, row in enumerate(store.rows):
        if row["id"] != f"{database}/pretrain/{index+1}" or not isinstance(row["query_hash"], str):
            raise ValueError("Source manifest identity format changed")
        h = row["query_hash"]
        if h in first:
            other = first[h]
            if not np.array_equal(store.cards[index], store.cards[other]):
                raise ValueError("Conflicting duplicate source label")
            for g, values in enumerate(store.groups):
                left = values[store.offsets[index, g]:store.offsets[index+1, g]]
                right = values[store.offsets[other, g]:store.offsets[other+1, g]]
                if not np.array_equal(left, right):
                    raise ValueError("Conflicting duplicate source features")
        else:
            first[h] = index
    for group in store.groups:
        if not np.isfinite(group).all():
            raise ValueError("Nonfinite source features")
    return {"query_hash": np.array([r["query_hash"] for r in store.rows]),
            "query_id": np.array([r["id"] for r in store.rows]),
            "true": np.asarray(store.cards[:, 0]), "n_tables": tables}


def certified_batch(store, indices, device):
    import numpy as np
    import torch

    batch = store.batch(indices, device)
    expected = np.asarray(store.cards[indices, 0])
    if batch["true"].dtype != torch.float64 or not np.array_equal(batch["true"].cpu().numpy(), expected):
        raise ValueError("FP64 cards/sample/batch chain changed")
    return batch


def evaluate_cell(checkpoint_path, store, query_indices, support_indices, operator, device):
    import numpy as np
    import torch

    from spatial_meta_ce.checkpoints import load_checkpoint
    from spatial_meta_ce.meta import forward
    checkpoint = next((key for key, arm in CHECKPOINTS.items() if checkpoint_path.parent.name == arm), None)
    if checkpoint is None or sha256(checkpoint_path) != CHECKPOINT_SHA[checkpoint]:
        raise ValueError("Checkpoint changed before trusted deserialization")
    model, state = load_checkpoint(checkpoint_path, device)
    if state["next_step"] != 4000 or state["metadata"]["seed"] != 42 or model.config != FROZEN["model"]:
        raise ValueError("Checkpoint model/seed/budget changed")
    model.eval()
    before = state_digest(model)
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    support = None if operator == "none" else certified_batch(store, support_indices, device)
    params, parameter_record = operator_params(model, support, operator)
    if device == "cuda":
        torch.cuda.synchronize()
    adaptation_seconds = time.monotonic() - started
    predictions, labels = [], []
    started = time.monotonic()
    with torch.no_grad():
        for start in range(0, len(query_indices), 512):
            batch = certified_batch(store, query_indices[start:start+512], device)
            values = forward(model, params, batch).detach().cpu().numpy().astype(np.float64)
            if not np.isfinite(values).all():
                raise ValueError("Nonfinite raw predictions")
            predictions.append(values)
            labels.append(batch["true"].cpu().numpy())
    if device == "cuda":
        torch.cuda.synchronize()
    after = state_digest(model)
    if after != before:
        raise ValueError("Cell mutated shared initialization")
    record = {**parameter_record, "state_before": before, "state_after": after,
              "adaptation_seconds": adaptation_seconds,
              "prediction_seconds": time.monotonic() - started,
              "peak_vram_bytes": int(max(torch.cuda.max_memory_allocated(), torch.cuda.max_memory_reserved()))
              if device == "cuda" else 0,
              "peak_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
              * (1 if sys.platform == "darwin" else 1024),
              "rss_measurement": "process lifetime high-water mark; watchdog also samples owned process tree"}
    return np.concatenate(predictions), np.concatenate(labels), record


def worker(output, mode, config_path, root=ROOT):
    require_cuda_workspace(mode)
    import numpy as np

    from spatial_meta_ce.data import FeatureStore
    validate_config(read_json(config_path))
    out = safe_output(output, [RETAINED, CACHE, "/gly/weilin/PRICE-master"])
    out.mkdir(parents=True)
    write_json(out / "status.json", {"status": "running", "mode": mode, "cells": 0})
    write_json(out / "phase.json", {"phase": "preparation"})
    environment_record = environment(mode)
    write_json(out / "environment.json", environment_record)
    binding, plan, anchors = input_binding(root)
    binding[str(checked_path(config_path))] = sha256(config_path)
    if set(plan) != set(DATABASES) or any(len(plan[db]) != 5 for db in DATABASES):
        raise ValueError("Full source plan is missing")
    stores, identities, certified = {}, {}, {}
    (out / "identities").mkdir()
    (out / "predictions").mkdir()
    for db in DATABASES:
        store = FeatureStore(source_path(db))
        stores[db] = store
        raw = store_identity(store, db)
        for run in range(5):
            selected = certify_plan(plan[db][run], raw, db, run)
            certified[(db, run)] = selected
            indices = np.asarray(selected["query"], dtype=np.int64)
            order = np.argsort(raw["query_hash"][indices])
            indices = indices[order]
            selected["query_sorted"] = indices.tolist()
            identity = encode_identity({key: value[indices] for key, value in raw.items()})
            identity_path = out / "identities" / f"{db}-run{run}.npz"
            np.savez(identity_path, **identity)
            identities[(db, run)] = (identity, identity_path)
            for checkpoint in CHECKPOINTS:
                with np.load(anchors[(checkpoint, db, run)], allow_pickle=False) as archive:
                    anchor = {key: archive[key] for key in archive.files}
                # Preparation certifies identity/truth/table/plan binding only.
                # Raw log_pred tolerances are checked on the exact regression
                # cells during evaluation, not over the full query population.
                verify_anchor_identity(identity, anchor)
    estimated = sum(p.stat().st_size for p in (out / "identities").glob("*.npz")) + 6714180 * 8
    if estimated > 0.30 * 1024**3:
        raise ValueError("Frozen shared identity schema exceeds capacity budget")
    write_json(out / "input-binding.json", {"schema": 1, "mode": mode, "input_sha256": binding,
                                           "code_sha256": {p: binding[str(root / p)] for p in CODE_FILES},
                                           "estimated_formal_output_bytes": estimated})
    write_json(out / "phase.json", {"phase": "evaluation"})
    cells = []
    none_checks = []
    selected_matrix = matrix() if mode == "formal" else [
        (c, "last3" if c == "C" else "head3", db, 0) for c in CHECKPOINTS for db in DATABASES]
    device = "cpu" if mode == "cpu_preflight" else "cuda"
    for checkpoint, operator, db, run in selected_matrix:
        identity, identity_path = identities[(db, run)]
        query = certified[(db, run)]["query_sorted"]
        positions = None if mode == "formal" else list(range(32))
        query = query if positions is None else query[:32]
        selected_identity = identity if positions is None else {key: value[positions] for key, value in identity.items()}
        checkpoint_path = RETAINED / CHECKPOINTS[checkpoint] / "last.pt"
        if mode != "formal":
            none_pred, none_true, none_record = evaluate_cell(
                checkpoint_path, stores[db], query,
                certified[(db, run)]["support"], "none", device)
            check_cell_peaks(none_record)
            if none_record["selected_parameter_count"] or none_record["steps"]:
                raise ValueError("None microcheck called adaptation")
            if none_record["state_before"] != none_record["state_after"]:
                raise ValueError("None microcheck mutated initialization")
            none_checks.append({"checkpoint": checkpoint, "database": db, "run": run,
                                "operator": "none", "query_positions": positions,
                                "count": len(none_pred), "state_before": none_record["state_before"],
                                "state_after": none_record["state_after"],
                                "selected_parameter_count": none_record["selected_parameter_count"],
                                "steps": none_record["steps"]})
            if not np.array_equal(none_true, selected_identity["true"]):
                raise ValueError("None microcheck FP64 truth differs")
        pred, true, record = evaluate_cell(checkpoint_path, stores[db], query,
                                           certified[(db, run)]["support"], operator, device)
        check_cell_peaks(record)
        if not np.array_equal(true, selected_identity["true"]):
            raise ValueError("Output FP64 truth changed")
        diagonal = operator == ("last3" if checkpoint == "C" else "head3")
        if diagonal:
            with np.load(anchors[(checkpoint, db, run)], allow_pickle=False) as archive:
                verify_anchor(pred, selected_identity, {key: archive[key] for key in archive.files}, positions)
        path = out / "predictions" / f"{checkpoint}-{operator}-{db}-run{run}.npz"
        np.savez(path, log_pred=pred)
        cells.append({**record, "checkpoint": checkpoint, "operator": operator, "database": db,
                      "run": run, "k": 20, "k20": 20, "count": len(pred), "regression_anchor_checked": diagonal,
                      "identity_file": str(identity_path.relative_to(out)), "identity_sha256": sha256(identity_path),
                      "prediction_file": str(path.relative_to(out)), "prediction_sha256": sha256(path),
                      "metrics": cell_metrics(pred, selected_identity),
                      **({"query_positions": positions} if positions is not None else {})})
        write_json(out / ("cells.json" if mode == "formal" else "microcells.json"),
                   {"schema": 1, "mode": mode, "cells": cells,
                    **({"none_checks": none_checks} if mode != "formal" else {})})
    write_json(out / "phase.json", {"phase": "certification"})
    from scripts.meta65_joint_verify import verify
    verified = verify(out)
    recheck(binding)
    write_json(out / "verification.json", verified)
    write_json(out / "status.json", {"status": "worker_complete", "mode": mode,
                                     "cells": len(cells) if mode == "formal" else 0,
                                     "micro_cells": len(cells) if mode != "formal" else 0,
                                     "formal_result": mode == "formal"})
    return verified


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/meta65_joint_control.json")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--mode", choices=("cpu_preflight", "gpu_preflight", "formal"), default="cpu_preflight")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--approval", type=Path, help="Separate user-approved formal launch contract")
    args = parser.parse_args(argv)
    config_path = checked_path(args.config)
    validate_config(read_json(config_path))
    if args.mode == "formal" and args.approval is None:
        raise ValueError("Formal GPU matrix is not authorized by implementation/preflight")
    require_cuda_workspace(args.mode)
    output = checked_path(args.out)
    binding_path = ROOT / LAUNCH_BINDING
    approval = None
    gpu_uuid = os.environ.get("CUDA_VISIBLE_DEVICES") if args.mode != "cpu_preflight" else None
    if args.mode != "formal" and args.approval is not None:
        raise ValueError("Preflight cannot use a formal approval")
    if args.mode == "formal":
        launch = verify_launch_binding(binding_path, root=ROOT, config_path=config_path)
        if launch["gpu_preflight_uuid"] != gpu_uuid:
            raise ValueError("Formal GPU UUID differs from certified GPU preflight")
        if args.execute and not args.worker:
            verify_gpu_preflight_output(launch)
        approval = validate_formal_approval(
            args.approval, output, binding_path, config_path,
            {p: sha256(ROOT / p) for p in CODE_FILES},
            gpu_uuid=gpu_uuid, allow_consumed=args.worker)
    if not args.execute:
        print(json.dumps({"status": "prepared_not_executed", "mode": args.mode,
                          "cells": 135 if args.mode == "formal" else 0,
                          "microcells": 9 if args.mode != "formal" else 0,
                          "formal_result": False}, indent=2))
        return
    safe_output(output, [RETAINED, CACHE, "/gly/weilin/PRICE-master"])
    if args.worker:
        if (os.environ.get("META65_SUPERVISED_MODE") != args.mode
                or os.environ.get("META65_SUPERVISOR_PID") != str(os.getppid())):
            raise ValueError("Worker requires its owning external supervisor")
        if args.mode == "formal":
            verify_formal_claim(args.approval, output, parent_pid=os.getppid())
        try:
            worker(output, args.mode, config_path)
        except BaseException as exc:
            if output.exists():
                write_json(output / "status.json", {"status": "failed", "mode": args.mode,
                                                    "failure": f"{type(exc).__name__}: {exc}",
                                                    "formal_result": False})
            raise
    else:
        from scripts.meta65_joint_supervisor import supervise
        if args.mode in ("formal", "gpu_preflight"):
            verify_gpu_idle(gpu_uuid)
        if args.mode == "formal":
            claim_formal_approval(args.approval, output, approval)
        command = [sys.executable, "-B", str(Path(__file__).resolve()), "--worker", "--execute",
                   "--config", str(config_path), "--out", str(output), "--mode", args.mode]
        if args.approval:
            command += ["--approval", str(checked_path(args.approval))]
        record = supervise(command, output, mode=args.mode, root=ROOT)
        print(json.dumps(record, indent=2))
        if record["status"] not in ("completed", "preflight_completed"):
            raise SystemExit(1)


if __name__ == "__main__":
    main()
