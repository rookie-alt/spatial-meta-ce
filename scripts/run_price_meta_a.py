#!/usr/bin/env python3
"""Doc 122 A-stage formal matrix runner: 12 training runs and 396 source units.

The parent process validates the frozen inputs, the explicit execution
authorization and the complete budget before any model library is imported or
any output directory is created.  It then supervises one worker that trains
all twelve arms/seeds and evaluates the sealed source population, and finally
recomputes every source unit independently before sealing the A verdict.

This runner never authorizes itself: the authorization record is a separate
artifact whose bytes are bound to the specification and to the budget digest.
The check is fail-closed protection against a mistyped run, not a tamper-proof
authorization service.
"""

from __future__ import annotations

import argparse
import atexit
import csv
import hashlib
import json
import os
import sys
import threading
import time
from array import array
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PROTOCOL = 122
PURPOSE = "122-a-formal-training"
ARMS = ("A0-ref", "A0", "A1", "A2")
EXPLORATORY_ARM = "A1-PG"
SEEDS = (42, 43, 44)
EVAL_DATABASES = ("chembl", "legalacts", "telstra")
KS = (0, 5, 20)
SUPPORT_RUNS = 5
UNITS_PER_RUN = len(EVAL_DATABASES) * (1 + 2 * SUPPORT_RUNS)
TOTAL_RUNS = len(ARMS) * len(SEEDS)
TOTAL_CELLS = TOTAL_RUNS * UNITS_PER_RUN
PHASES = ("preparation", "evaluation", "certification")
BUDGET_FIELDS = (
    "total_seconds",
    "phase_seconds",
    "rss_bytes",
    "vram_bytes",
    "output_bytes",
    "poll_seconds",
    "cpu_core_seconds",
    "gpu_allocated_seconds",
    "run_deadline_seconds",
    "eval_deadline_seconds",
)
SUPERVISOR_BUDGET_FIELDS = (
    "total_seconds",
    "phase_seconds",
    "rss_bytes",
    "vram_bytes",
    "output_bytes",
    "poll_seconds",
)
REQUIRED_CODE = (
    "scripts/run_price_meta_a.py",
    "scripts/verify_p0_certificate.py",
    "scripts/meta65_joint_supervisor.py",
    "spatial_meta_ce/__init__.py",
    "spatial_meta_ce/train.py",
    "spatial_meta_ce/source_eval.py",
    "spatial_meta_ce/price_protocol.py",
    "spatial_meta_ce/price_capacity.py",
    "spatial_meta_ce/plans.py",
    "spatial_meta_ce/metrics.py",
    "spatial_meta_ce/model.py",
    "spatial_meta_ce/price_model.py",
    "spatial_meta_ce/data.py",
    "spatial_meta_ce/checkpoints.py",
    "spatial_meta_ce/meta.py",
    "spatial_meta_ce/manifests.py",
    "spatial_meta_ce/mappings.py",
    "spatial_meta_ce/tail_risk.py",
    "vendor/price_baseline/module.py",
)


def canonical_digest(value) -> str:
    raw = json.dumps(value, sort_keys=True, ensure_ascii=False).encode()
    return hashlib.sha256(raw).hexdigest()


def file_sha256(path) -> str:
    hasher = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def read_bound_json(path, expected_sha256: str) -> dict:
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError(f"Input file SHA256 mismatch: {path}")
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def write_json(path: Path, payload: dict, *, replace: bool = True) -> None:
    path = Path(path)
    if path.is_symlink():
        raise ValueError(f"Refusing to write through a symlink: {path}")
    if not replace and path.exists():
        raise FileExistsError(path)
    temporary = path.with_name(path.name + ".tmp")
    if temporary.exists():
        if not replace:
            raise FileExistsError(temporary)
        temporary.unlink()
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    os.replace(temporary, path)


def _number(value, name: str, *, positive: bool = True) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number")
    value = float(value)
    if value != value or value in (float("inf"), float("-inf")):
        raise ValueError(f"{name} must be finite")
    if positive and value <= 0:
        raise ValueError(f"{name} must be positive")
    if not positive and value < 0:
        raise ValueError(f"{name} must be non-negative")
    return value


def validate_budget(budget) -> dict:
    if not isinstance(budget, dict) or set(budget) != set(BUDGET_FIELDS):
        raise ValueError("122 A budget must contain the complete frozen schema")
    phase = budget["phase_seconds"]
    if not isinstance(phase, dict) or set(phase) != set(PHASES):
        raise ValueError("122 A budget phase_seconds must cover every phase")
    normalized = {
        "total_seconds": _number(budget["total_seconds"], "budget.total_seconds"),
        "phase_seconds": {
            name: _number(phase[name], f"budget.phase_seconds.{name}")
            for name in PHASES
        },
        "rss_bytes": int(
            _number(budget["rss_bytes"], "budget.rss_bytes", positive=False)
        ),
        "vram_bytes": int(
            _number(budget["vram_bytes"], "budget.vram_bytes", positive=False)
        ),
        "output_bytes": int(
            _number(budget["output_bytes"], "budget.output_bytes", positive=False)
        ),
        "poll_seconds": _number(budget["poll_seconds"], "budget.poll_seconds"),
        "cpu_core_seconds": int(
            _number(
                budget["cpu_core_seconds"], "budget.cpu_core_seconds", positive=False
            )
        ),
        "gpu_allocated_seconds": int(
            _number(
                budget["gpu_allocated_seconds"],
                "budget.gpu_allocated_seconds",
                positive=False,
            )
        ),
        "run_deadline_seconds": _number(
            budget["run_deadline_seconds"], "budget.run_deadline_seconds"
        ),
        "eval_deadline_seconds": _number(
            budget["eval_deadline_seconds"], "budget.eval_deadline_seconds"
        ),
    }
    if sum(normalized["phase_seconds"].values()) > normalized["total_seconds"]:
        raise ValueError("122 A budget phase limits exceed the parent wall limit")
    return normalized


def validate_exploratory_budget(budget: dict) -> dict:
    """Reject the old training-only envelope before authorizing 33 source units."""
    checked = validate_budget(budget)
    if (
        checked["total_seconds"] < 5700
        or checked["phase_seconds"]["preparation"] < 1800
        or checked["phase_seconds"]["evaluation"] < 1200
        or checked["phase_seconds"]["certification"] < 299
        or checked["run_deadline_seconds"] < 1800
        or checked["eval_deadline_seconds"] < 1200
        or checked["output_bytes"] < 1024**3
        or checked["cpu_core_seconds"] < 4500
        or checked["gpu_allocated_seconds"] < 3500
    ):
        raise ValueError("A1-PG requires a separately approved training and evaluation budget")
    return checked


def supervisor_budget(budget: dict) -> dict:
    return {key: budget[key] for key in SUPERVISOR_BUDGET_FIELDS}


def validate_authorization(
    authorization, spec_sha256: str, budget: dict, output: Path
) -> dict:
    if not isinstance(authorization, dict):
        raise ValueError("122 A authorization must be a JSON object")
    if (
        authorization.get("protocol") != PROTOCOL
        or authorization.get("purpose") != PURPOSE
        or authorization.get("authorized") is not True
        or authorization.get("formal_training_authorized") is not True
        or authorization.get("target_databases_read") != []
    ):
        raise ValueError("122 A execution authorization is incomplete")
    if authorization.get("spec_sha256") != spec_sha256:
        raise ValueError("122 A authorization does not bind this specification")
    if authorization.get("budget_sha256") != canonical_digest(budget):
        raise ValueError("122 A authorization does not bind this budget")
    granted_by = authorization.get("granted_by")
    if not isinstance(granted_by, str) or not granted_by.strip():
        raise ValueError("122 A authorization lacks an approving identity")
    granted_at = authorization.get("granted_at")
    if not isinstance(granted_at, str):
        raise ValueError("122 A authorization lacks a dated approval")
    try:
        timestamp = datetime.fromisoformat(granted_at)
    except ValueError as exc:
        raise ValueError("122 A approval timestamp must be ISO 8601") from exc
    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        raise ValueError("122 A approval timestamp requires a timezone")
    bound_output = authorization.get("output_dir")
    requested_output = Path(output)
    if (
        not isinstance(bound_output, str)
        or not Path(bound_output).is_absolute()
        or ".." in Path(bound_output).parts
        or ".." in requested_output.parts
        or Path(bound_output) != requested_output.absolute()
        or any(
            parent.is_symlink()
            for parent in (requested_output, *requested_output.parents)
        )
    ):
        raise ValueError("122 A authorization does not bind this output directory")
    return authorization


def compute_code_sha256(code) -> dict:
    """Bind the modules the A matrix depends on, not the whole deployment tree."""
    code = Path(code)
    if not code.is_dir() or code.is_symlink():
        raise ValueError("122 A requires a regular code tree")
    hashes = {}
    for name in REQUIRED_CODE:
        path = code / name
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"122 A code tree lacks a regular module: {name}")
        hashes[name] = file_sha256(path)
    return hashes


def verify_code_tree(code, hashes) -> None:
    if compute_code_sha256(code) != hashes:
        raise ValueError("122 A code tree differs from the frozen specification")


def check_output_path(path, forbidden_root) -> Path:
    raw = Path(path).absolute()
    if ".." in raw.parts:
        raise ValueError("122 A output path cannot contain ..")
    if raw.exists():
        raise FileExistsError(f"122 A output already exists: {raw}")
    if any(parent.is_symlink() for parent in (raw, *raw.parents)):
        raise ValueError("122 A output path traverses a symlink")
    forbidden = Path(forbidden_root).absolute()
    if raw == forbidden or forbidden in raw.parents:
        raise ValueError("122 A output must stay outside the original PRICE tree")
    return raw


def _plan_digest(plan: dict) -> str:
    return canonical_digest({k: v for k, v in plan.items() if k != "plan_sha256"})


def execution_keys(spec: dict) -> list[str]:
    keys = [f"{arm}-seed{seed}" for arm in ARMS for seed in SEEDS]
    mode = spec.get("execution_mode", "matrix")
    if mode == "matrix" and "run_key" not in spec and "exploratory_arm" not in spec:
        return keys
    if mode == "single_run" and spec.get("run_key") in keys and not spec.get("exploratory_arm"):
        return [spec["run_key"]]
    if (
        mode == "single_run"
        and spec.get("exploratory_arm") == EXPLORATORY_ARM
        and spec.get("run_key") in (f"{EXPLORATORY_ARM}-seed{s}" for s in SEEDS)
    ):
        return [spec["run_key"]]
    raise ValueError("122 A execution scope must be a frozen arm/seed or isolated A1-PG")


def validate_exploratory_spec(spec: dict) -> None:
    """Require one PG-input arm/seed without changing the frozen A matrix."""
    if (
        spec.get("execution_mode") != "single_run"
        or spec.get("exploratory_arm") != EXPLORATORY_ARM
        or spec.get("run_key") not in (f"{EXPLORATORY_ARM}-seed{s}" for s in SEEDS)
        or spec.get("arms") != list(ARMS)
        or spec.get("seeds") != list(SEEDS)
        or spec.get("total_runs") != TOTAL_RUNS
        or spec.get("total_cells") != TOTAL_CELLS
        or not isinstance(spec.get("exploratory_config_dir"), str)
        or not Path(spec["exploratory_config_dir"]).is_absolute()
        or ".." in Path(spec["exploratory_config_dir"]).parts
        or (
            "rebound" in spec
            and Path(spec["exploratory_config_dir"]).resolve()
            == Path(spec["rebound"]).resolve()
        )
    ):
        raise ValueError("A1-PG requires one labeled run and the frozen A matrix")


def exploratory_config(name: str, original: dict) -> dict:
    """Derive the selected PG-input arm from its sealed A1 twin."""
    from spatial_meta_ce.price_protocol import exploratory_model_config

    key = name.removesuffix(".json")
    arm, seed_text = key.rsplit("-seed", 1)
    if (
        arm != EXPLORATORY_ARM
        or int(seed_text) not in SEEDS
        or original.get("fixed_stage") != "122-A1"
        or original.get("seed") != int(seed_text)
    ):
        raise ValueError("A1-PG must reuse the matching frozen A1 seed")
    return {
        **original,
        "fixed_stage": "122-A1-PG",
        "model": exploratory_model_config(original["model"]["capacity"]),
    }


def prepare_exploratory_configs(rebound: Path, output: Path) -> None:
    """Create a separate PG arm from the byte-verified A1 configs."""
    rebound, output = Path(rebound), Path(output)
    if output.exists() or output.is_symlink() or rebound.resolve() == output.resolve():
        raise FileExistsError("PG-input config directory must be new and separate")
    manifest = read_bound_json(
        rebound / "binding_manifest.json", file_sha256(rebound / "binding_manifest.json")
    )
    if manifest.get("p0_gate_passed") is not False or manifest.get("formal_training_authorized") is not False:
        raise ValueError("Inherited 122 binding manifest must remain closed")
    configs = {}
    for seed in SEEDS:
        source = f"A1-seed{seed}.json"
        original = read_bound_json(rebound / source, manifest["files"][source]["sha256"])
        configs[f"{EXPLORATORY_ARM}-seed{seed}.json"] = exploratory_config(
            f"{EXPLORATORY_ARM}-seed{seed}.json", original
        )
    output.mkdir(parents=True, exist_ok=False)
    for name, config in configs.items():
        write_json(output / name, config, replace=False)


def prepare_exploratory_certificate(
    source: Path, code: Path, config_dir: Path, output: Path
) -> dict:
    """Build an unsigned certificate from independently bound inputs."""
    from scripts.verify_p0_certificate import verify

    source, code, config_dir, output = map(Path, (source, code, config_dir, output))
    if verify(source):
        raise ValueError("Inherited P0 certificate does not verify")
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    old = read_bound_json(source, file_sha256(source))
    digest_path = output.with_name("pg-code-digests.json")
    if digest_path.exists() or digest_path.is_symlink():
        raise FileExistsError(digest_path)
    hashes = compute_code_sha256(code)
    write_json(digest_path, hashes, replace=False)
    frozen = {
        "path": str(digest_path.resolve()),
        "sha256": file_sha256(digest_path),
        "manifest_digest": canonical_digest(hashes),
        "code_files": [
            {"name": name, "path": str(code / name), "sha256": digest}
            for name, digest in hashes.items()
        ],
    }
    configs = [
        {
            "name": f"{EXPLORATORY_ARM}-seed{seed}.json",
            "path": str(config_dir / f"{EXPLORATORY_ARM}-seed{seed}.json"),
            "sha256": file_sha256(config_dir / f"{EXPLORATORY_ARM}-seed{seed}.json"),
        }
        for seed in SEEDS
    ]
    record = {
        **old,
        "status": "unsigned",
        "issued_at": None,
        "issued_by": None,
        "approval_basis": None,
        "note": "A1-PG exploration: code/config binding draft; independent P0 review required before signing.",
        "truth": {
            "p0_gate_passed": None,
            "formal_training_authorized": False,
            "research_training_started": False,
            "protocol_122_executed": False,
        },
        "bindings": {
            **old["bindings"],
            "frozen_code": frozen,
            "a_training_configs": [*old["bindings"]["a_training_configs"], *configs],
        },
    }
    write_json(output, record, replace=False)
    return record


def verify_single_protocol(training: list, key: str) -> dict:
    if len(training) != 1 or training[0].get("key") != key:
        raise ValueError("122 single-run protocol has an unexpected run")
    entry = training[0]
    arm, seed_text = key.rsplit("-seed", 1)
    if (
        entry.get("arm") != arm
        or entry.get("seed") != int(seed_text)
        or entry.get("optimizer_steps") != 4000
        or file_sha256(entry["checkpoint"]) != entry.get("checkpoint_sha256")
    ):
        raise ValueError("122 single-run checkpoint/count binding differs")
    return entry


def read_p0_certificate(record: dict) -> dict:
    from scripts.verify_p0_certificate import verify

    if not isinstance(record, dict) or not {"path", "sha256"} <= record.keys():
        raise ValueError("122 A requires a hash-bound P0 certificate")
    certificate = read_bound_json(record["path"], record["sha256"])
    if certificate.get("schema") != 2:
        raise ValueError("122 A execution requires a schema-2 P0-only certificate")
    differences = verify(Path(record["path"]))
    if differences:
        raise ValueError("122 A P0 certificate rejected: " + "; ".join(differences))
    return certificate


def _require_binding(entry: dict, path, sha256: str) -> None:
    if (
        not isinstance(entry, dict)
        or not isinstance(entry.get("path"), str)
        or Path(entry["path"]).resolve() != Path(path).resolve()
        or entry.get("sha256") != sha256
    ):
        raise ValueError(f"122 A P0 certificate binding differs: {path}")


def require_exploratory_certificate(spec: dict, resolved: dict) -> dict:
    """Do not reuse a formal four-arm P0 certificate for this new code/config."""
    validate_exploratory_spec(spec)
    certificate = read_p0_certificate(spec.get("p0_certificate"))
    if certificate.get("purpose") != "p0-certificate":
        raise ValueError("A1-PG requires an independently reviewed P0 certificate")
    bindings = certificate.get("bindings", {})
    entries = bindings.get("a_training_configs", [])
    name = spec["run_key"] + ".json"
    matches = [entry for entry in entries if entry.get("name") == name]
    if len(matches) != 1 or len(entries) != len(ARMS) * len(SEEDS) + len(SEEDS):
        raise ValueError("A1-PG certificate must bind all three exploratory configs")
    directory = Path(spec["exploratory_config_dir"])
    for seed in SEEDS:
        bound_name = f"{EXPLORATORY_ARM}-seed{seed}.json"
        bound = next((entry for entry in entries if entry.get("name") == bound_name), None)
        path = directory / bound_name
        _require_binding(bound, path, file_sha256(path))
    return certificate


def validate_certificate_bindings(
    certificate: dict, spec: dict, resolved: dict
) -> None:
    bindings = certificate["bindings"]
    rebound = Path(spec["rebound"])
    manifest = resolved["manifest"]
    for field, name in (
        ("source_preflight", "preflight.json"),
        ("data_config_122", "data_config_122.json"),
    ):
        _require_binding(
            bindings[field], rebound / name, manifest["files"][name]["sha256"]
        )
    _require_binding(
        bindings["source_eval_plan"],
        spec["source_eval_plan"]["path"],
        spec["source_eval_plan"]["file_sha256"],
    )
    expected_groups = {
        "a_training_configs": {
            f"{key}.json": (
                rebound / f"{key}.json",
                manifest["files"][f"{key}.json"]["sha256"],
            )
            for key in resolved["configs"]
            if not key.startswith(f"{EXPLORATORY_ARM}-")
        },
        "training_plans": {
            f"seed{seed}": (record["path"], record["file_sha256"])
            for seed, record in resolved["training_plans"].items()
        },
    }
    preflight = resolved["preflight"]
    capacity = preflight["capacity"]
    source_records = preflight["coverage"]["databases"]
    cache_root = Path(resolved["data"]["cache_root"]) / "pretrain"
    expected_groups["cache_manifests"] = {
        database: (
            cache_root / database / "manifest.json",
            capacity["manifest_sha256"].get(database)
            if database in capacity["databases"]
            else source_records[database]["manifest_sha256"],
        )
        for database in (*capacity["databases"], *EVAL_DATABASES)
    }
    if spec.get("exploratory_arm") == EXPLORATORY_ARM:
        directory = Path(spec["exploratory_config_dir"])
        if not directory.is_absolute() or directory.is_symlink():
            raise ValueError("A1-PG requires a regular absolute config directory")
        expected_groups["a_training_configs"].update(
            {
                f"{EXPLORATORY_ARM}-seed{seed}.json": (
                    directory / f"{EXPLORATORY_ARM}-seed{seed}.json",
                    file_sha256(directory / f"{EXPLORATORY_ARM}-seed{seed}.json"),
                )
                for seed in SEEDS
            }
        )
    for field, expected in expected_groups.items():
        entries = bindings[field]
        indexed = {entry.get("name"): entry for entry in entries}
        if len(entries) != len(indexed) or set(indexed) != set(expected):
            raise ValueError(f"122 A P0 certificate lacks the complete {field}")
        for name, (path, sha256) in expected.items():
            _require_binding(indexed[name], path, sha256)
    frozen = bindings["frozen_code"]
    code_entries = frozen.get("code_files", [])
    code_map = {entry.get("name"): entry for entry in code_entries}
    if len(code_entries) != len(code_map) or set(code_map) != set(spec["code_sha256"]):
        raise ValueError("122 A P0 certificate does not bind the complete code tree")
    for name, sha256 in spec["code_sha256"].items():
        _require_binding(code_map[name], Path(spec["code"]) / name, sha256)


def verify_spec(spec: dict) -> dict:
    """Re-verify every frozen input recorded in the specification."""
    selected_keys = execution_keys(spec)
    if spec.get("exploratory_arm") == EXPLORATORY_ARM:
        validate_exploratory_spec(spec)
    elif "exploratory_arm" in spec:
        raise ValueError("Unknown exploratory arm")
    if (
        spec.get("protocol") != PROTOCOL
        or spec.get("purpose") != PURPOSE
        or spec.get("arms") != list(ARMS)
        or spec.get("seeds") != list(SEEDS)
        or spec.get("total_runs") != TOTAL_RUNS
        or spec.get("total_cells") != TOTAL_CELLS
        or spec.get("device") not in ("cpu", "cuda")
        or spec.get("target_databases_read", []) != []
    ):
        raise ValueError("122 A specification is not the frozen A matrix")
    budget = (
        validate_exploratory_budget(spec.get("budget"))
        if spec.get("exploratory_arm") == EXPLORATORY_ARM
        else validate_budget(spec.get("budget"))
    )
    if Path(spec["code"]).resolve() != ROOT:
        raise ValueError("122 A runner must execute from the frozen code tree")
    verify_code_tree(spec["code"], spec["code_sha256"])

    rebound = Path(spec["rebound"])
    manifest = read_bound_json(
        rebound / "binding_manifest.json", spec["binding_manifest_sha256"]
    )
    if (
        manifest.get("p0_gate_passed") is not False
        or manifest.get("formal_training_authorized") is not False
    ):
        raise ValueError("122 A must preserve the closed historical gates")
    configs = {}
    for arm in ARMS:
        for seed in SEEDS:
            name = f"{arm}-seed{seed}.json"
            record = manifest["files"].get(name)
            if not isinstance(record, dict) or "sha256" not in record:
                raise ValueError(f"122 A binding manifest lacks {name}")
            config = read_bound_json(rebound / name, record["sha256"])
            expected_stage = f"122-{arm}"
            if (
                config.get("fixed_stage") != expected_stage
                or config.get("seed") != seed
                or config.get("method") != "erm"
                or config.get("meta_objective") != "post"
                or config.get("fixed_budget") is not True
                or config.get("steps") != 4000
                or config.get("purpose") != "research"
            ):
                raise ValueError(
                    f"122 A configuration differs from the frozen arm: {name}"
                )
            configs[name[: -len(".json")]] = config
    data = read_bound_json(
        rebound / "data_config_122.json",
        manifest["files"]["data_config_122.json"]["sha256"],
    )
    preflight = read_bound_json(
        rebound / "preflight.json", manifest["files"]["preflight.json"]["sha256"]
    )
    if (
        preflight.get("p0_gate_passed") is not False
        or preflight.get("formal_training_authorized") is not False
        or preflight.get("target_databases_read") != []
        or preflight.get("status") != "data_preflight_passed"
    ):
        raise ValueError("122 A inherited preflight is not the frozen source preflight")
    from spatial_meta_ce.price_protocol import _validate_data_config_binding

    for config in configs.values():
        if (
            Path(config.get("source_preflight", "")).resolve()
            != (rebound / "preflight.json").resolve()
            or config.get("source_preflight_sha256")
            != manifest["files"]["preflight.json"]["sha256"]
        ):
            raise ValueError(
                "122 A training configuration has an unbound source preflight"
            )
        _validate_data_config_binding(config, preflight, data)
    if spec.get("exploratory_arm") == EXPLORATORY_ARM:
        from spatial_meta_ce.train import _validate_fixed_budget

        for seed in SEEDS:
            name = f"{EXPLORATORY_ARM}-seed{seed}.json"
            path = Path(spec["exploratory_config_dir"]) / name
            if path.is_symlink() or not path.is_file():
                raise ValueError(f"Missing PG-input binding: {name}")
            config = read_bound_json(path, file_sha256(path))
            original = configs[f"A1-seed{seed}"]
            if config != exploratory_config(name, original):
                raise ValueError(f"PG-input config differs from paired A1: {name}")
            _validate_fixed_budget(config)
            _validate_data_config_binding(config, preflight, data)
            configs[name.removesuffix(".json")] = config
    plan_record = spec["source_eval_plan"]
    plan = read_bound_json(plan_record["path"], plan_record["file_sha256"])
    if plan.get("plan_sha256") != _plan_digest(plan):
        raise ValueError("122 A source evaluation plan binding changed")
    if plan.get("plan_sha256") != preflight["source_eval_plan"]["bound_plan_sha256"]:
        raise ValueError("122 A source evaluation plan differs from the preflight")
    if (
        plan.get("protocol") != PROTOCOL
        or plan.get("analysis_role") != "population_primary"
        or tuple(plan.get("databases", {})) != EVAL_DATABASES
    ):
        raise ValueError("122 A source evaluation plan is not the sealed population")
    for entry in plan["databases"].values():
        if entry.get("excluded_unique") != 100:
            raise ValueError(
                "122 A source evaluation plan lacks the 100-query exclusion"
            )
    training_plans = {}
    for seed in SEEDS:
        record = preflight["training_plans"][str(seed)]
        path = Path(record["path"])
        actual = file_sha256(path)
        if actual != record["file_sha256"]:
            raise ValueError("122 A sealed training plan changed since the preflight")
        training_plans[str(seed)] = {
            "path": str(path),
            "file_sha256": actual,
            "plan_sha256": record["plan_sha256"],
        }
    resolved = {
        "budget": budget,
        "execution_keys": selected_keys,
        "manifest": manifest,
        "configs": configs,
        "exploratory_arm": spec.get("exploratory_arm"),
        "data": data,
        "preflight": preflight,
        "plan": plan,
        "training_plans": training_plans,
    }
    certificate = read_p0_certificate(spec.get("p0_certificate"))
    validate_certificate_bindings(certificate, spec, resolved)
    if spec.get("exploratory_arm") == EXPLORATORY_ARM:
        require_exploratory_certificate(spec, resolved)
    return resolved


def verify_execution_evidence(preflight: dict, evidence: dict, config=None) -> None:
    if (
        not isinstance(evidence, dict)
        or not {"spec", "authorization", "output_dir"} <= evidence.keys()
    ):
        raise ValueError("122 training requires bound execution evidence")
    records = {}
    for name in ("spec", "authorization"):
        record = evidence[name]
        if not isinstance(record, dict) or not {"path", "sha256"} <= record.keys():
            raise ValueError("122 training requires bound execution evidence")
        records[name] = read_bound_json(record["path"], record["sha256"])
    spec = records["spec"]
    if spec.get("protocol") != PROTOCOL or spec.get("purpose") != PURPOSE:
        raise ValueError("122 execution evidence has an invalid specification")
    resolved = verify_spec(spec)
    check_output = Path(evidence["output_dir"]).absolute()
    forbidden = Path(resolved["data"]["data_root"]).resolve()
    if (
        check_output.resolve() == forbidden
        or forbidden in check_output.resolve().parents
    ):
        raise ValueError(
            "122 execution output must stay outside the original PRICE tree"
        )
    certificate = read_p0_certificate(spec.get("p0_certificate"))
    source = certificate["bindings"]["source_preflight"]
    original = read_bound_json(source["path"], source["sha256"])
    if (
        original != preflight
        or resolved["preflight"] != preflight
        or preflight.get("p0_gate_passed") is not False
        or preflight.get("formal_training_authorized") is not False
    ):
        raise ValueError(
            "122 execution evidence does not bind the closed source preflight"
        )
    validate_authorization(
        records["authorization"],
        evidence["spec"]["sha256"],
        validate_budget(spec.get("budget")),
        Path(evidence["output_dir"]),
    )
    if config is not None:
        rebound = (
            Path(spec["exploratory_config_dir"])
            if config.get("fixed_stage") == "122-A1-PG"
            else Path(spec["rebound"])
        )
        name = f"{config.get('fixed_stage', '').removeprefix('122-')}-seed{config.get('seed')}.json"
        if name.removesuffix(".json") not in execution_keys(spec):
            raise ValueError(
                "122 runtime configuration is outside the authorized scope"
            )
        entries = certificate["bindings"]["a_training_configs"]
        matches = [entry for entry in entries if entry.get("name") == name]
        if len(matches) != 1:
            raise ValueError("122 execution evidence lacks this training configuration")
        entry = matches[0]
        _require_binding(entry, rebound / name, file_sha256(rebound / name))
        frozen = read_bound_json(entry["path"], entry["sha256"])
        expected = {
            **frozen,
            "source_preflight": config["source_preflight"],
            "source_preflight_sha256": config["source_preflight_sha256"],
        }
        if config != expected:
            raise ValueError(
                "122 runtime configuration differs from its P0 certificate"
            )
        verify_code_tree(spec["code"], spec["code_sha256"])


def validate_execution_snapshot(snapshot: dict, config: dict) -> None:
    evidence = snapshot.get("execution_evidence")
    if not isinstance(evidence, dict):
        raise ValueError("122 training requires bound execution evidence")
    spec = read_bound_json(evidence["spec"]["path"], evidence["spec"]["sha256"])
    certificate = read_p0_certificate(spec.get("p0_certificate"))
    source = certificate["bindings"]["source_preflight"]
    preflight = read_bound_json(source["path"], source["sha256"])
    verify_execution_evidence(preflight, evidence, config)
    if snapshot != build_execution_snapshot(preflight, evidence):
        raise ValueError("122 execution snapshot differs from its certified evidence")


def build_execution_snapshot(preflight: dict, evidence: dict) -> dict:
    """Validate independent certification and approval, preserving frozen evidence."""
    verify_execution_evidence(preflight, evidence)
    spec = read_bound_json(evidence["spec"]["path"], evidence["spec"]["sha256"])
    if spec.get("exploratory_arm") == EXPLORATORY_ARM:
        snapshot = {**preflight, "exploratory_pg_head_input_authorized": True}
    else:
        snapshot = preflight
    return {
        **snapshot,
        "purpose": PURPOSE,
        "p0_gate_passed": True,
        "formal_training_authorized": True,
        "research_training_started": False,
        "protocol_122_executed": False,
        "execution_evidence": evidence,
        "inherited_preflight": {
            "p0_gate_passed": preflight.get("p0_gate_passed"),
            "formal_training_authorized": preflight.get("formal_training_authorized"),
            "status": preflight.get("status"),
        },
    }


def _write_phase(output: Path, phase: str) -> None:
    if phase not in PHASES:
        raise ValueError("Unknown 122 A phase")
    write_json(Path(output) / "phase.json", {"phase": phase})


PROGRESS_FILE = "unit_progress.json"
UNIT_KINDS = ("run", "eval", "eval_unit")
MEMORY_FILE = "memory.json"


def cuda_memory_bytes(torch_module) -> dict:
    """Return this process's CUDA counters; independent of PID namespaces."""
    cuda = torch_module.cuda
    if not cuda.is_available():
        return {
            "allocated_bytes": 0,
            "reserved_bytes": 0,
            "max_allocated_bytes": 0,
            "max_reserved_bytes": 0,
        }
    return {
        "allocated_bytes": int(cuda.memory_allocated()),
        "reserved_bytes": int(cuda.memory_reserved()),
        "max_allocated_bytes": int(cuda.max_memory_allocated()),
        "max_reserved_bytes": int(cuda.max_memory_reserved()),
    }


class CudaMemoryTelemetry:
    """Publish worker CUDA peaks so the supervisor bills this process only.

    ``nvidia-smi`` reports host PIDs that may not exist in the worker's PID
    namespace, so a driver-side match silently reads zero.  The worker writes
    its own counters instead; a query failure is published as an error so the
    supervisor fails closed rather than treating it as zero usage.
    """

    def __init__(self, output: Path, torch_module, interval: float = 1.0):
        self.path = Path(output) / MEMORY_FILE
        self.torch_module = torch_module
        self.interval = interval
        self._stop = threading.Event()
        self._thread = None

    def sample(self) -> dict:
        try:
            value = cuda_memory_bytes(self.torch_module)
        except Exception as exc:  # noqa: BLE001 - telemetry failure must reach the supervisor
            value = {"error": f"{type(exc).__name__}: {exc}"}
        value["interval_seconds"] = self.interval
        try:
            write_json(self.path, value)
        except OSError as exc:
            # Keep sampling so a later write can succeed; if none does, the
            # supervisor's staleness check stops the run instead of trusting it.
            print(
                json.dumps({"telemetry_write_error": f"{type(exc).__name__}: {exc}"}),
                flush=True,
            )
        return value

    def _loop(self) -> None:
        while not self._stop.wait(self.interval):
            self.sample()

    def start(self) -> None:
        self.sample()
        atexit.register(self.close)
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=5)
        self.sample()


def gpu_accounting_note(device: str) -> str:
    """State that GPU usage is measured inside the worker, not from PIDs."""
    if device == "cpu":
        return "CPU mode hides CUDA: gpu_allocated_seconds is 0 by construction."
    return (
        "CUDA mode: peak and interval VRAM come from the worker's own "
        "torch.cuda counters (memory.json), which are independent of PID "
        "namespaces; GPU seconds accrue over monotonic sample intervals with "
        "nonzero worker VRAM."
    )


class UnitDeadline:
    """Parent-side online hard stop for per-run and per-evaluation-unit deadlines.

    The worker records unit start/done markers in an atomic JSON file.  The
    parent times each unit from the monotonic instant it first observes that
    unit, so no cross-process clock comparison is needed.  A malformed marker
    file fails closed; an overrun raises inside the supervisor's sample_fn,
    which already kills the owned process group.
    """

    def __init__(self, output: Path, run_deadline: float, eval_deadline: float):
        self.path = Path(output) / PROGRESS_FILE
        self.deadlines = {
            "run": run_deadline,
            "eval": eval_deadline,
            "eval_unit": eval_deadline,
        }
        self._active: dict[tuple[str, str], float] = {}

    def write_unit(self, kind: str, key: str, state: str) -> None:
        if kind not in UNIT_KINDS or state not in ("start", "done"):
            raise ValueError("122 A invalid unit progress marker")
        if self.path.is_symlink():
            raise ValueError("122 A unit progress file must not be a symlink")
        units = []
        if self.path.is_file():
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            if not isinstance(raw, dict) or not isinstance(raw.get("units"), list):
                raise ValueError("122 A unit progress file is malformed")
            units = raw["units"]
        units.append({"kind": kind, "key": key, "state": state})
        write_json(self.path, {"units": units})

    def check(self) -> None:
        if self.path.is_symlink():
            raise ValueError("122 A unit progress file must not be a symlink")
        if not self.path.is_file():
            return
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError("122 A unit progress file is unreadable") from exc
        if not isinstance(raw, dict) or not isinstance(raw.get("units"), list):
            raise ValueError("122 A unit progress file is malformed")
        now = time.monotonic()
        for marker in raw["units"]:
            if (
                not isinstance(marker, dict)
                or marker.get("kind") not in UNIT_KINDS
                or not isinstance(marker.get("key"), str)
                or marker.get("state") not in ("start", "done")
            ):
                raise ValueError("122 A unit progress marker is malformed")
            kind, key, state = marker["kind"], marker["key"], marker["state"]
            active_key = (kind, key)
            if state == "done":
                self._active.pop(active_key, None)
            elif active_key not in self._active:
                self._active[active_key] = now
        for (kind, key), started in self._active.items():
            if now - started > self.deadlines[kind]:
                label = "evaluation unit" if kind != "run" else "run"
                raise TimeoutError(f"122 A {label} deadline exceeded: {key}")


def parent_sample_fn(monitor: UnitDeadline, tracker, sampler):
    """Check unit deadlines inside the supervisor's fail-closed sample boundary."""

    def sample(child) -> dict:
        monitor.check()
        return tracker.sample(child, sampler)

    return sample


class BudgetTracker:
    """Account for the whole tree (parent + live worker + reaped children).

    CPU core-seconds are user+system seconds across the supervisor itself, the
    live worker process tree and already-reaped children. ``gpu_allocated_seconds``
    is wall time integrated over monotonic sampling intervals where the worker
    reported nonzero CUDA memory (not GiB·s); the GiB·s quantity is kept
    separately as ``gpu_allocated_gib_seconds``. Sampling failures must
    propagate to the supervisor's fail-closed process-group stop and poison
    finalization. A CUDA run that never observes nonzero worker VRAM cannot
    finalize: a zero would otherwise pass as a measurement.
    """

    def __init__(self, budget: dict, *, requires_gpu_usage: bool = False) -> None:
        self.budget = budget
        self.requires_gpu_usage = requires_gpu_usage
        self.cpu_seconds = 0.0
        self.gpu_seconds = 0.0
        self.gpu_gib_seconds = 0.0
        self.samples = 0
        self.sampling_failed = None
        self._baseline_reaped_cpu = self._child_cpu_seconds()
        self._baseline_self_cpu = self._self_cpu_seconds()
        self._previous_time = time.monotonic()

    @staticmethod
    def _child_cpu_seconds() -> float:
        import resource

        usage = resource.getrusage(resource.RUSAGE_CHILDREN)
        return float(usage.ru_utime + usage.ru_stime)

    @staticmethod
    def _self_cpu_seconds() -> float:
        import resource

        usage = resource.getrusage(resource.RUSAGE_SELF)
        return float(usage.ru_utime + usage.ru_stime)

    @staticmethod
    def _live_cpu_seconds(child) -> float:
        from scripts.meta65_joint_supervisor import _owned_pids, _process_rows

        rows = _process_rows()
        pids = _owned_pids(rows, int(child.pid))
        if not pids:
            raise ValueError("122 A live worker process tree cannot be sampled")
        ticks = os.sysconf("SC_CLK_TCK")
        if ticks <= 0:
            raise ValueError("122 A CPU clock ticks cannot be sampled")
        cpu_ticks = 0
        for pid in pids:
            try:
                raw = (Path("/proc") / str(pid) / "stat").read_text(encoding="ascii")
                closing = raw.rfind(")")
                fields = raw[closing + 2 :].split()
                if closing < 0 or len(fields) < 20:
                    raise ValueError("invalid process stat")
                if (int(fields[2]), int(fields[3])) != rows[pid][1:3]:
                    raise ValueError("process group changed during CPU sample")
                cpu_ticks += sum(int(fields[index]) for index in (11, 12, 13, 14))
            except (OSError, UnicodeError, ValueError, IndexError) as exc:
                raise ValueError(f"122 A CPU sample failed for pid {pid}") from exc
        return cpu_ticks / ticks

    def _check_cpu(self) -> None:
        if self.cpu_seconds > self.budget["cpu_core_seconds"]:
            raise ValueError("122 A cumulative CPU core-second budget exceeded")

    def sample(self, child, sampler) -> dict:
        try:
            value = sampler(child)
            live_cpu = self._live_cpu_seconds(child)
            reaped_cpu = self._child_cpu_seconds() - self._baseline_reaped_cpu
            self_cpu = self._self_cpu_seconds() - self._baseline_self_cpu
            self.cpu_seconds = max(self.cpu_seconds, self_cpu + reaped_cpu + live_cpu)
            now = time.monotonic()
            elapsed = now - self._previous_time
            if elapsed < 0:
                raise ValueError("122 A monotonic sampling clock moved backwards")
            self._previous_time = now
            gib = float(value["vram_bytes"]) / (1024**3)
            self.gpu_gib_seconds += gib * elapsed
            if gib > 0:
                self.gpu_seconds += elapsed
            self.samples += 1
        except Exception as exc:
            self.sampling_failed = f"{type(exc).__name__}: {exc}"
            raise
        self._check_cpu()
        if self.gpu_seconds > self.budget["gpu_allocated_seconds"]:
            raise ValueError("122 A cumulative GPU allocated-second budget exceeded")
        return value

    def finalize(self) -> None:
        if self.sampling_failed is not None:
            raise ValueError(
                f"122 A sampling failed, refusing to finalize: {self.sampling_failed}"
            )
        if self.requires_gpu_usage and self.gpu_seconds <= 0:
            raise ValueError(
                "122 A CUDA run observed no nonzero worker VRAM; "
                "refusing to certify a zero GPU measurement"
            )
        self.cpu_seconds = max(
            self.cpu_seconds,
            self._self_cpu_seconds()
            - self._baseline_self_cpu
            + self._child_cpu_seconds()
            - self._baseline_reaped_cpu,
        )
        self._check_cpu()

    def state(self) -> dict:
        return {
            "cpu_core_seconds": round(self.cpu_seconds, 6),
            "gpu_allocated_seconds": round(self.gpu_seconds, 6),
            "gpu_allocated_gib_seconds": round(self.gpu_gib_seconds, 6),
            "sampling_failed": self.sampling_failed,
            "samples": self.samples,
            "cpu_core_seconds_budget": self.budget["cpu_core_seconds"],
            "gpu_allocated_seconds_budget": self.budget["gpu_allocated_seconds"],
        }


def train_all(
    resolved: dict,
    output: Path,
    device: str,
    snapshot_path: Path,
    deadline: float,
    progress=None,
) -> list:
    from spatial_meta_ce.train import train

    def report(kind: str, key: str, state: str) -> None:
        if progress is not None:
            progress(kind, key, state)

    protocols = []
    data_config = resolved["data"]
    for arm in (*ARMS, *((EXPLORATORY_ARM,) if resolved.get("exploratory_arm") else ())):
        for seed in SEEDS:
            key = f"{arm}-seed{seed}"
            if key not in resolved.get("execution_keys", execution_keys({})):
                continue
            config = dict(resolved["configs"][key])
            config["source_preflight"] = str(snapshot_path)
            config["source_preflight_sha256"] = file_sha256(snapshot_path)
            run_dir = Path(output) / "runs" / key
            report("run", key, "start")
            started = time.monotonic()
            result = train(config, data_config, run_dir, device)
            elapsed = time.monotonic() - started
            report("run", key, "done")
            if elapsed > deadline:
                raise TimeoutError(f"122 A training run exceeded its deadline: {key}")
            checkpoint = run_dir / "last.pt"
            if not checkpoint.is_file():
                raise FileNotFoundError(f"122 A training did not write last.pt: {key}")
            if (run_dir / "best.pt").exists():
                raise ValueError(f"122 A training wrote a forbidden best.pt: {key}")
            counts = result.get("counts", {})
            if counts.get("optimizer_steps") != 4000:
                raise ValueError(f"122 A training did not complete 4000 updates: {key}")
            protocols.append(
                {
                    "key": key,
                    "arm": arm,
                    "seed": seed,
                    "checkpoint": str(checkpoint),
                    "checkpoint_sha256": file_sha256(checkpoint),
                    "training_plan_sha256": resolved["training_plans"][str(seed)][
                        "plan_sha256"
                    ],
                    "optimizer_steps": counts.get("optimizer_steps"),
                    "backward_calls": counts.get("backward_calls"),
                    "query_presentations": counts.get("query_presentations"),
                    "total_seconds": result.get("total_seconds"),
                    "wall_seconds": elapsed,
                    "parameters": result.get("parameters"),
                }
            )
    return protocols


def evaluate_all(
    spec: dict,
    resolved: dict,
    output: Path,
    device: str,
    training: list,
    deadline: float,
    progress=None,
) -> list:
    from spatial_meta_ce.source_eval import evaluate_source

    def report(kind: str, key: str, state: str) -> None:
        if progress is not None:
            progress(kind, key, state)

    def make_on_unit(run_key: str):
        return lambda state, unit_key: report(
            "eval_unit", f"{run_key}/{unit_key}", state
        )

    protocols = []
    plan_path = spec["source_eval_plan"]["path"]
    for entry in training:
        eval_dir = Path(output) / "eval" / entry["key"]
        report("eval", entry["key"], "start")
        started = time.monotonic()
        protocol = evaluate_source(
            entry["checkpoint"],
            entry["arm"],
            entry["seed"],
            plan_path,
            resolved["data"],
            eval_dir,
            device=device,
            batch_size=int(spec.get("batch_size", 128)),
            on_unit=make_on_unit(entry["key"]),
        )
        elapsed = time.monotonic() - started
        report("eval", entry["key"], "done")
        if elapsed > deadline:
            raise TimeoutError(
                f"122 A source evaluation exceeded its deadline: {entry['key']}"
            )
        protocols.append(
            {
                "key": entry["key"],
                "arm": entry["arm"],
                "seed": entry["seed"],
                "output": str(eval_dir),
                "protocol_sha256": file_sha256(eval_dir / "eval_protocol.json"),
                "predictions_sha256": protocol["predictions_sha256"],
                "units": len(protocol["units"]),
                "wall_seconds": elapsed,
            }
        )
    return protocols


def recompute_units(csv_path) -> dict:
    """Independent log-Q95 recomputation from the prediction CSV."""
    import numpy as np

    groups: dict = {}
    with Path(csv_path).open("r", newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        required = {
            "arm",
            "seed",
            "database",
            "k",
            "run",
            "query_id",
            "query_hash",
            "true_cardinality",
            "log_prediction",
        }
        if not required <= set(reader.fieldnames or ()):
            raise ValueError("122 A predictions CSV lacks the frozen columns")
        for row in reader:
            key = (row["database"], int(row["k"]), int(row["run"]))
            entry = groups.setdefault(key, (array("d"), array("d"), []))
            entry[0].append(float(row["log_prediction"]))
            entry[1].append(float(row["true_cardinality"]))
            entry[2].append((row["query_id"], row["query_hash"]))
    result = {}
    for key, (predictions, truths, identities) in groups.items():
        prediction = np.asarray(predictions, dtype=np.float64)
        truth = np.asarray(truths, dtype=np.float64)
        if not np.isfinite(prediction).all() or not np.isfinite(truth).all():
            raise ValueError("122 A predictions contain non-finite values")
        log_q = np.abs(np.maximum(prediction, 0.0) - np.log(np.maximum(truth, 1.0)))
        result[key] = {
            "log_q95": float(np.percentile(log_q, 95)),
            "n": int(len(truth)),
            "identities": identities,
        }
    return result


def verify_exploratory_evaluation(protocol, csv_path, plan, seed: int) -> dict:
    """Independently check every PG-input source unit against the sealed plan."""
    import numpy as np

    if (
        protocol.get("kind") != "price_source_eval_122_pg_input_exploratory"
        or protocol.get("arm") != EXPLORATORY_ARM
        or protocol.get("seed") != seed
        or protocol.get("completed") is not True
        or protocol.get("target_databases_read") != []
        or protocol.get("source_eval_plan_sha256") != plan.get("plan_sha256")
        or file_sha256(csv_path) != protocol.get("predictions_sha256")
    ):
        raise ValueError("A1-PG source evaluation or prediction binding differs")
    units = protocol.get("units", [])
    with Path(csv_path).open("r", newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if row["arm"] != EXPLORATORY_ARM or row["seed"] != str(seed):
                raise ValueError("A1-PG CSV arm/seed identity differs")
    recomputed = recompute_units(csv_path)
    expected = {
        (db, 0, 0) for db in EVAL_DATABASES
    } | {
        (db, k, run) for db in EVAL_DATABASES for k in (5, 20) for run in range(1, 6)
    }
    keyed = {(unit["database"], unit["k"], unit["run"]): unit for unit in units}
    if len(units) != UNITS_PER_RUN or set(keyed) != expected or set(recomputed) != expected:
        raise ValueError("A1-PG must contain the complete 33 source units")
    for (db, k, run), unit in keyed.items():
        row = recomputed[db, k, run]
        entry = plan["databases"][db]
        identities = list(zip(entry["query_ids"], entry["query_hashes"]))
        if (
            row["n"] != len(entry["query"])
            or row["identities"] != identities
            or not np.isclose(unit["metrics"]["log_q95"], row["log_q95"], rtol=1e-9, atol=1e-9)
        ):
            raise ValueError("A1-PG source query population or metric differs")
        if k and unit.get("support_ids") != entry["support_blocks"][run - 1]["support_ids"][:k]:
            raise ValueError("A1-PG source support identity differs")
    return {"units": UNITS_PER_RUN, "independent_recompute": True}


def expected_unit_keys() -> set:
    keys = set()
    for arm in ARMS:
        for seed in SEEDS:
            for database in EVAL_DATABASES:
                keys.add((arm, seed, database, 0, 0))
                for k in (5, 20):
                    for run in range(1, SUPPORT_RUNS + 1):
                        keys.add((arm, seed, database, k, run))
    return keys


def verify_matrix(worker_dir, plan: dict) -> dict:
    """Rebuild the 396 units from the sealed evaluations and recheck them."""
    import numpy as np

    units = []
    seen = set()
    for arm in ARMS:
        for seed in SEEDS:
            key = f"{arm}-seed{seed}"
            eval_dir = Path(worker_dir) / "eval" / key
            protocol_path = eval_dir / "eval_protocol.json"
            if not protocol_path.is_file():
                raise FileNotFoundError(f"122 A source evaluation is missing: {key}")
            protocol = json.loads(protocol_path.read_text())
            if (
                protocol.get("kind") != "price_source_eval_122_a"
                or protocol.get("arm") != arm
                or protocol.get("seed") != seed
                or protocol.get("completed") is not True
                or protocol.get("target_databases_read") != []
            ):
                raise ValueError(f"122 A source evaluation protocol is invalid: {key}")
            recomputed = recompute_units(eval_dir / "predictions.csv")
            if len(protocol["units"]) != UNITS_PER_RUN:
                raise ValueError(f"122 A source evaluation unit count differs: {key}")
            for unit in protocol["units"]:
                unit_key = (arm, seed, unit["database"], unit["k"], unit["run"])
                if unit_key in seen:
                    raise ValueError(f"122 A duplicate source unit: {unit_key}")
                seen.add(unit_key)
                recomputed_row = recomputed.get(
                    (unit["database"], unit["k"], unit["run"])
                )
                if recomputed_row is None:
                    raise ValueError(
                        f"122 A prediction CSV lacks a protocol unit: {unit_key}"
                    )
                expected = plan["databases"][unit["database"]]
                if recomputed_row["n"] != len(expected["query"]):
                    raise ValueError(f"122 A unit denominator differs: {unit_key}")
                reported = float(unit["metrics"]["log_q95"])
                if not np.isclose(
                    reported, recomputed_row["log_q95"], rtol=1e-9, atol=1e-9
                ):
                    raise ValueError(
                        f"122 A reported and recomputed log-Q95 differ: {unit_key}"
                    )
                if unit["k"] == 0:
                    expected_identities = list(
                        zip(expected["query_ids"], expected["query_hashes"])
                    )
                    if recomputed_row["identities"] != expected_identities:
                        raise ValueError(
                            f"122 A K0 query identity binding differs: {unit_key}"
                        )
                else:
                    block = expected["support_blocks"][unit["run"] - 1]
                    limit = unit["k"]
                    if unit.get("support_ids") != list(block["support_ids"][:limit]):
                        raise ValueError(
                            f"122 A support identity binding differs: {unit_key}"
                        )
                units.append(
                    {
                        "arm": arm,
                        "seed": seed,
                        "database": unit["database"],
                        "k": unit["k"],
                        "run": unit["run"],
                        "queries": recomputed_row["n"],
                        "metrics": {"log_q95": recomputed_row["log_q95"]},
                        "source_eval_plan_sha256": protocol["source_eval_plan_sha256"],
                    }
                )
    if len(units) != TOTAL_CELLS or len(seen) != TOTAL_CELLS:
        raise ValueError(f"122 A matrix is incomplete: {len(units)} units")
    if seen != expected_unit_keys():
        raise ValueError("122 A matrix does not cover the frozen 396 units")
    return {"units": units, "independent_recompute": True}


def worker_main(args) -> int:
    sys.dont_write_bytecode = True
    if os.environ.get("META65_SUPERVISED_MODE") != "formal":
        raise ValueError("122 A worker requires the formal supervisor")
    if os.environ.get("META65_SUPERVISOR_PID") != str(os.getppid()):
        raise ValueError("122 A worker requires its own supervisor parent")
    if sys.platform != "linux":
        raise ValueError("122 A execution is server Linux only")
    started = time.monotonic()
    spec_raw = Path(args.spec).read_bytes()
    if hashlib.sha256(spec_raw).hexdigest() != args.spec_sha256:
        raise ValueError("122 A specification bytes changed")
    spec = json.loads(spec_raw)
    resolved = verify_spec(spec)
    budget = resolved["budget"]
    authorization = read_bound_json(args.authorization, args.authorization_sha256)
    validate_authorization(
        authorization, args.spec_sha256, budget, Path(args.output).parent
    )

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "input_spec.json", spec)
    snapshot_dir = output / "execution"
    snapshot_dir.mkdir()
    snapshot_path = snapshot_dir / "preflight.json"
    write_json(
        snapshot_path,
        build_execution_snapshot(
            resolved["preflight"],
            {
                "spec": {
                    "path": str(Path(args.spec).resolve()),
                    "sha256": args.spec_sha256,
                },
                "authorization": {
                    "path": str(Path(args.authorization).resolve()),
                    "sha256": args.authorization_sha256,
                },
                "output_dir": str(output.parent),
            },
        ),
    )

    device = spec["device"]
    import torch

    torch.set_num_threads(int(spec.get("threads", 1)))
    torch.set_num_interop_threads(1)
    if device == "cpu" and torch.cuda.is_available():
        raise ValueError("122 A CPU execution must hide the GPU")

    monitor = UnitDeadline(
        output,
        budget["run_deadline_seconds"],
        budget["eval_deadline_seconds"],
    )
    progress = monitor.write_unit
    if device == "cuda":
        CudaMemoryTelemetry(output, torch).start()
    _write_phase(output, "preparation")
    training = train_all(
        resolved,
        output,
        device,
        snapshot_path,
        budget["run_deadline_seconds"],
        progress=progress,
    )
    write_json(output / "training_protocols.json", {"runs": training})
    if spec.get("execution_mode") == "single_run":
        _write_phase(
            output, "evaluation" if spec.get("exploratory_arm") == EXPLORATORY_ARM else "certification"
        )
        entry = verify_single_protocol(training, spec["run_key"])
        from spatial_meta_ce.checkpoints import load_checkpoint
        from spatial_meta_ce.price_protocol import (
            validate_a_checkpoint, validate_exploratory_checkpoint,
        )

        model, state = load_checkpoint(entry["checkpoint"])
        if spec.get("exploratory_arm") == EXPLORATORY_ARM:
            validate_exploratory_checkpoint(state, entry["seed"])
        else:
            validate_a_checkpoint(state, entry["arm"], entry["seed"])
        del model, state
        if spec.get("exploratory_arm") == EXPLORATORY_ARM:
            evaluations = evaluate_all(
                spec, resolved, output, device, training, budget["eval_deadline_seconds"],
                progress=progress,
            )
            if len(evaluations) != 1 or evaluations[0]["units"] != UNITS_PER_RUN:
                raise ValueError("A1-PG requires 33 supervised source units")
            eval_dir = output / "eval" / entry["key"]
            protocol = json.loads((eval_dir / "eval_protocol.json").read_text())
            verified = verify_exploratory_evaluation(
                protocol, eval_dir / "predictions.csv", resolved["plan"], entry["seed"]
            )
            write_json(
                output / "exploratory_source_seal.json",
                {
                    "spec_sha256": args.spec_sha256,
                    "authorization_sha256": args.authorization_sha256,
                    "p0_certificate_sha256": spec["p0_certificate"]["sha256"],
                    "checkpoint_sha256": entry["checkpoint_sha256"],
                    "source_eval_plan_sha256": resolved["plan"]["plan_sha256"],
                    "eval_protocol_sha256": file_sha256(eval_dir / "eval_protocol.json"),
                    "predictions_sha256": protocol["predictions_sha256"],
                    **verified, "target_databases_read": [],
                },
                replace=False,
            )
            _write_phase(output, "certification")
        write_json(
            output / "single_run_seal.json",
            {
                "protocol": PROTOCOL,
                "execution_mode": "single_run",
                "run": entry,
                "spec_sha256": args.spec_sha256,
                "authorization_sha256": args.authorization_sha256,
                "source_evaluation_started": spec.get("exploratory_arm") == EXPLORATORY_ARM,
                "source_evaluation_completed": spec.get("exploratory_arm") == EXPLORATORY_ARM,
                "a_verdict": None,
                "target_databases_read": [],
            },
        )
        write_json(
            output / "status.json",
            {
                "status": "worker_complete",
                "mode": "formal",
                "cells": 1 + (UNITS_PER_RUN if spec.get("exploratory_arm") == EXPLORATORY_ARM else 0),
                "runs": 1,
                "execution_mode": "single_run",
                "research_training_started": True,
                "protocol_122_executed": False,
                "wall_seconds": time.monotonic() - started,
                "target_databases_read": [],
            },
        )
        return 0
    _write_phase(output, "evaluation")
    evaluations = evaluate_all(
        spec,
        resolved,
        output,
        device,
        training,
        budget["eval_deadline_seconds"],
        progress=progress,
    )
    write_json(output / "evaluation_protocols.json", {"runs": evaluations})
    _write_phase(output, "certification")
    matrix = verify_matrix(output, resolved["plan"])
    write_json(
        output / "units.json",
        {"schema": 1, "protocol": PROTOCOL, "units": matrix["units"]},
    )
    write_json(
        output / "status.json",
        {
            "status": "worker_complete",
            "mode": "formal",
            "cells": TOTAL_CELLS,
            "runs": TOTAL_RUNS,
            "device": device,
            "wall_seconds": time.monotonic() - started,
            "formal_result": True,
            "target_databases_read": [],
        },
    )
    print(json.dumps({"status": "worker_complete", "cells": TOTAL_CELLS}), flush=True)
    return 0


def parent_main(args) -> int:
    spec_path = Path(args.spec)
    spec_raw = spec_path.read_bytes()
    spec_sha256 = hashlib.sha256(spec_raw).hexdigest()
    if spec_sha256 != args.spec_sha256:
        raise ValueError("122 A specification bytes changed")
    spec = json.loads(spec_raw)
    budget = validate_budget(spec.get("budget"))
    output = Path(args.output).absolute()
    authorization = read_bound_json(args.authorization, args.authorization_sha256)
    validate_authorization(authorization, spec_sha256, budget, output)
    resolved = verify_spec(spec)
    output = check_output_path(output, resolved["data"]["data_root"])

    plan = {
        "protocol": PROTOCOL,
        "purpose": PURPOSE,
        "spec_sha256": spec_sha256,
        "authorization_sha256": args.authorization_sha256,
        "runs": TOTAL_RUNS,
        "cells": TOTAL_CELLS,
        "device": spec["device"],
        "budget": budget,
        "gpu_allocated_seconds_accounting": gpu_accounting_note(spec["device"]),
        "target_databases_read": [],
    }
    single_run = spec.get("execution_mode") == "single_run"
    if single_run:
        plan.update(
            runs=1, cells=UNITS_PER_RUN if spec.get("exploratory_arm") == EXPLORATORY_ARM else 0, execution_mode="single_run", run_key=spec["run_key"]
        )
    if not args.execute:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        print("Dry run only. Add --execute after the authorization record is complete.")
        return 0

    from scripts.meta65_joint_supervisor import sample_process, supervise

    if spec["device"] == "cuda":
        if not spec.get("cuda_device"):
            raise ValueError("122 A CUDA execution requires an explicit device index")
        os.environ["CUDA_VISIBLE_DEVICES"] = str(spec["cuda_device"])
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    threads = int(spec.get("threads", 1))
    os.environ["OMP_NUM_THREADS"] = str(threads)
    os.environ["MKL_NUM_THREADS"] = str(threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(threads)
    output.mkdir(parents=True)
    write_json(output / "execution_plan.json", plan)
    worker_output = output / "worker"
    command = [
        sys.executable,
        "-I",
        "-B",
        str(Path(spec["code"]) / "scripts" / "run_price_meta_a.py"),
        "worker",
        "--spec",
        str(spec_path.resolve()),
        "--spec-sha256",
        spec_sha256,
        "--authorization",
        str(Path(args.authorization).resolve()),
        "--authorization-sha256",
        args.authorization_sha256,
        "--output",
        str(worker_output),
    ]
    tracker = BudgetTracker(budget, requires_gpu_usage=spec["device"] == "cuda")
    monitor = UnitDeadline(
        worker_output,
        budget["run_deadline_seconds"],
        budget["eval_deadline_seconds"],
    )
    sample = parent_sample_fn(
        monitor,
        tracker,
        lambda target: sample_process(target, mode="formal", output=worker_output),
    )

    try:
        record = supervise(
            command,
            worker_output,
            mode="formal",
            budget=supervisor_budget(budget),
            sample_fn=sample,
            root=spec["code"],
            grace=10,
            formal_cells=(1 + UNITS_PER_RUN if spec.get("exploratory_arm") == EXPLORATORY_ARM else 1) if single_run else TOTAL_CELLS,
        )
    except Exception as exc:
        write_json(
            output / "failure.json",
            {
                "status": "failed",
                "failure": f"{type(exc).__name__}: {exc}",
                "supervisor": None,
                "budget_usage": tracker.state(),
                "budget_usage_complete": False,
                "formal_result": False,
            },
            replace=False,
        )
        raise
    budget_state = tracker.state()
    try:
        tracker.finalize()
        budget_state = tracker.state()
        if record.get("status") != "completed":
            raise RuntimeError(f"122 A matrix failed: {record.get('failure')}")

        if single_run:
            seal = json.loads((worker_output / "single_run_seal.json").read_text())
            if (
                seal.get("spec_sha256") != spec_sha256
                or seal.get("authorization_sha256") != args.authorization_sha256
            ):
                raise ValueError("122 single-run seal authorization binding differs")
            verify_single_protocol([seal["run"]], spec["run_key"])
            if spec.get("exploratory_arm") == EXPLORATORY_ARM:
                source = json.loads(
                    (worker_output / "exploratory_source_seal.json").read_text()
                )
                if (
                    source.get("units") != UNITS_PER_RUN
                    or source.get("independent_recompute") is not True
                    or source.get("spec_sha256") != spec_sha256
                    or source.get("authorization_sha256") != args.authorization_sha256
                    or source.get("checkpoint_sha256") != seal["run"]["checkpoint_sha256"]
                    or source.get("target_databases_read") != []
                ):
                    raise ValueError("A1-PG supervised source seal is incomplete")
            write_json(
                output / "completion.json",
                {
                    "status": "single_run_completed",
                    "runs": 1,
                    "evaluation_cells": UNITS_PER_RUN if spec.get("exploratory_arm") == EXPLORATORY_ARM else 0,
                    "run_key": spec["run_key"],
                    "supervisor": record,
                    "budget_usage": budget_state,
                    "gpu_accounting": gpu_accounting_note(spec["device"]),
                    "a_verdict": None,
                    "target_databases_read": [],
                },
            )
            print(
                json.dumps(
                    {"status": "single_run_completed", "run_key": spec["run_key"]}
                ),
                flush=True,
            )
            return 0

        status = json.loads((worker_output / "status.json").read_text())
        if (
            status.get("cells") != TOTAL_CELLS
            or status.get("mode") != "formal"
            or status.get("target_databases_read") != []
        ):
            raise ValueError("122 A worker status does not certify the full matrix")
        matrix = verify_matrix(worker_output, resolved["plan"])
        from spatial_meta_ce.price_protocol import summarize_a

        verdict = summarize_a(matrix["units"])
        training = json.loads((worker_output / "training_protocols.json").read_text())[
            "runs"
        ]
        evaluations = json.loads(
            (worker_output / "evaluation_protocols.json").read_text()
        )["runs"]
        seal = {
            "schema": 1,
            "protocol": PROTOCOL,
            "purpose": PURPOSE,
            "spec_sha256": spec_sha256,
            "authorization_sha256": args.authorization_sha256,
            "budget": budget,
            "device": spec["device"],
            "code_sha256": spec["code_sha256"],
            "source_eval_plan": spec["source_eval_plan"],
            "training_plans": resolved["training_plans"],
            "training": training,
            "evaluations": evaluations,
            "supervisor": record,
            "budget_usage": budget_state,
            "gpu_accounting": gpu_accounting_note(spec["device"]),
            "units": len(matrix["units"]),
            "independent_recompute": True,
            "target_databases_read": [],
        }
        write_json(output / "a_matrix.json", {"units": matrix["units"]})
        write_json(output / "a_summary.json", verdict)
        write_json(output / "a_seal.json", seal)
        write_json(
            output / "completion.json",
            {
                "status": "a_matrix_completed",
                "cells": TOTAL_CELLS,
                "runs": TOTAL_RUNS,
                "a_status": verdict["status"],
                "target_databases_read": [],
            },
        )
    except Exception as exc:
        write_json(
            output / "failure.json",
            {
                "status": "failed",
                "failure": f"{type(exc).__name__}: {exc}",
                "supervisor": record,
                "budget_usage": tracker.state(),
                "formal_result": False,
            },
            replace=False,
        )
        raise
    print(
        json.dumps({"status": "a_matrix_completed", "cells": TOTAL_CELLS}), flush=True
    )
    return 0


def plan_main(args) -> int:
    """Freeze the specification and emit an unauthorized authorization template."""
    code = Path(args.code).resolve()
    rebound = Path(args.rebound).resolve()
    manifest = read_bound_json(
        rebound / "binding_manifest.json",
        file_sha256(rebound / "binding_manifest.json"),
    )
    preflight = read_bound_json(
        rebound / "preflight.json", manifest["files"]["preflight.json"]["sha256"]
    )
    source_plan_path = Path(args.source_eval_plan).resolve()
    source_plan = read_bound_json(source_plan_path, file_sha256(source_plan_path))
    if (
        source_plan.get("plan_sha256")
        != preflight["source_eval_plan"]["bound_plan_sha256"]
    ):
        raise ValueError(
            "122 A source evaluation plan differs from the frozen preflight"
        )
    budget = (
        validate_exploratory_budget(json.loads(Path(args.budget).read_text()))
        if getattr(args, "exploratory_arm", False)
        else validate_budget(json.loads(Path(args.budget).read_text()))
    )
    certificate_path = Path(args.p0_certificate).resolve()
    certificate_record = {
        "path": str(certificate_path),
        "sha256": file_sha256(certificate_path),
    }
    certificate = read_bound_json(certificate_path, certificate_record["sha256"])
    from scripts.verify_p0_certificate import verify_format, verify_truth

    if certificate.get("schema") != 2 or verify_format(certificate_path):
        raise ValueError(
            "122 A planning requires a well-formed schema-2 P0 certificate"
        )
    differences = verify_truth(certificate_path)
    if differences:
        raise ValueError(
            "122 A planning requires a signed P0 certificate: " + "; ".join(differences)
        )
    data = read_bound_json(
        rebound / "data_config_122.json",
        manifest["files"]["data_config_122.json"]["sha256"],
    )
    output_dir = check_output_path(args.output_dir, data["data_root"])
    spec = {
        "schema": 1,
        "protocol": PROTOCOL,
        "purpose": PURPOSE,
        "arms": list(ARMS),
        "seeds": list(SEEDS),
        "total_runs": TOTAL_RUNS,
        "total_cells": TOTAL_CELLS,
        "device": args.device,
        "cuda_device": args.cuda_device,
        "threads": args.threads,
        "batch_size": args.batch_size,
        "code": str(code),
        "code_sha256": compute_code_sha256(code),
        "rebound": str(rebound),
        "binding_manifest_sha256": file_sha256(rebound / "binding_manifest.json"),
        "source_eval_plan": {
            "path": str(source_plan_path),
            "file_sha256": file_sha256(source_plan_path),
            "plan_sha256": source_plan["plan_sha256"],
        },
        "p0_certificate": certificate_record,
        "budget": budget,
        "target_databases_read": [],
    }
    if getattr(args, "run_key", None):
        spec.update(execution_mode="single_run", run_key=args.run_key)
    if getattr(args, "exploratory_arm", False):
        if not args.exploratory_config_dir:
            raise ValueError("A1-PG requires --exploratory-config-dir")
        spec["exploratory_arm"] = EXPLORATORY_ARM
        spec["exploratory_config_dir"] = str(Path(args.exploratory_config_dir).absolute())
        validate_exploratory_spec(spec)
        require_exploratory_certificate(spec, {})
    execution_keys(spec)
    spec_path = Path(args.out_spec)
    if spec_path.exists():
        raise FileExistsError(f"122 A specification already exists: {spec_path}")
    write_json(spec_path, spec, replace=False)
    spec_sha256 = file_sha256(spec_path)
    authorization = {
        "schema": 1,
        "protocol": PROTOCOL,
        "purpose": PURPOSE,
        "authorized": False,
        "formal_training_authorized": False,
        "spec_sha256": spec_sha256,
        "budget_sha256": canonical_digest(budget),
        "output_dir": str(output_dir),
        "granted_by": None,
        "granted_at": None,
        "target_databases_read": [],
    }
    authorization_path = Path(args.out_authorization)
    if authorization_path.exists():
        raise FileExistsError(
            f"122 A authorization already exists: {authorization_path}"
        )
    write_json(authorization_path, authorization, replace=False)
    print(
        json.dumps(
            {
                "spec": str(spec_path),
                "spec_sha256": spec_sha256,
                "authorization_template": str(authorization_path),
                "authorization_sha256": file_sha256(authorization_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    plan = subparsers.add_parser("plan", help="freeze the specification")
    plan.add_argument("--code", required=True)
    plan.add_argument("--rebound", required=True)
    plan.add_argument("--source-eval-plan", required=True)
    plan.add_argument("--budget", required=True)
    plan.add_argument("--p0-certificate", required=True)
    plan.add_argument("--out-spec", required=True)
    plan.add_argument("--out-authorization", required=True)
    plan.add_argument("--output-dir", required=True)
    plan.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    plan.add_argument("--cuda-device", default=None)
    plan.add_argument("--threads", type=int, default=1)
    plan.add_argument("--batch-size", type=int, default=128)
    plan.add_argument(
        "--run-key", choices=[f"{arm}-seed{seed}" for arm in (*ARMS, EXPLORATORY_ARM) for seed in SEEDS]
    )
    plan.add_argument("--exploratory-arm", action="store_true")
    plan.add_argument("--exploratory-config-dir")

    pg_configs = subparsers.add_parser("prepare-pg", help="prepare isolated PG configurations")
    pg_configs.add_argument("--rebound", required=True)
    pg_configs.add_argument("--out", required=True)

    pg_certificate = subparsers.add_parser("prepare-pg-certificate", help="create unsigned PG P0 certificate")
    pg_certificate.add_argument("--source", required=True)
    pg_certificate.add_argument("--code", required=True)
    pg_certificate.add_argument("--config-dir", required=True)
    pg_certificate.add_argument("--out", required=True)

    for name in ("run", "worker"):
        command = subparsers.add_parser(name)
        command.add_argument("--spec", required=True)
        command.add_argument("--spec-sha256", required=True)
        command.add_argument("--authorization", required=True)
        command.add_argument("--authorization-sha256", required=True)
        command.add_argument("--output", required=True)
        command.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.command == "prepare-pg":
        prepare_exploratory_configs(Path(args.rebound), Path(args.out))
        return 0
    if args.command == "prepare-pg-certificate":
        prepare_exploratory_certificate(
            Path(args.source), Path(args.code), Path(args.config_dir), Path(args.out)
        )
        return 0
    if args.command == "plan":
        return plan_main(args)
    if args.command == "worker":
        return worker_main(args)
    return parent_main(args)


if __name__ == "__main__":
    raise SystemExit(main())
