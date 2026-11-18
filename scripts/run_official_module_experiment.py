#!/usr/bin/env python3
"""Doc-132 official26/4 module preparation and separately authorized execution.

prepare / verify / evaluate-ready never train or predict. run is a dry run
unless --execute is present, and worker requires its own formal supervisor.
Neither command signs an authorization. Training reads pretrain26 only;
evaluation freezes all fifteen last.pt files before reading test4 only.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if __name__ == "__main__":
    # train() imports the named runner for its evidence validator. Reuse this
    # module and its content-bound plan summaries instead of parsing twice.
    sys.modules["scripts.run_official_module_experiment"] = sys.modules[__name__]

from spatial_meta_ce.module_experiment import (  # noqa: E402
    ARMS,
    PROTOCOL,
    SEEDS,
    STAGE,
    STEPS,
    TEST_DATABASES,
    TRAIN_DATABASES,
    FIXED_SELECTION,
    binding_shape,
    canonical_digest,
    exact_equal,
    is_sha256,
    mapping_mode,
    regular_path,
    run_key,
    training_config,
    validate_checkpoint,
    validate_config_shape,
    validate_splits,
)

PURPOSES = {
    "train": "official-module-training",
    "evaluate": "official-module-k0-evaluation",
}
REQUIRED_CODE = (
    "scripts/run_official_module_experiment.py",
    "scripts/run_price_meta_a.py",
    "scripts/meta65_joint_supervisor.py",
    "scripts/verify_p0_certificate.py",
    "scripts/prepare_official_module_inputs.py",
    "scripts/prepare_official_module_inputs_parallel.py",
    "scripts/build_official_split_plans.py",
    "scripts/audit_input_mapping.py",
    "scripts/build_mapping_sidecar.py",
    "spatial_meta_ce/__init__.py",
    "spatial_meta_ce/module_experiment.py",
    "spatial_meta_ce/official_mappings.py",
    "spatial_meta_ce/official_split.py",
    "spatial_meta_ce/train.py",
    "spatial_meta_ce/model.py",
    "spatial_meta_ce/price_model.py",
    "spatial_meta_ce/price_relations.py",
    "spatial_meta_ce/checkpoints.py",
    "spatial_meta_ce/price_protocol.py",
    "spatial_meta_ce/price_capacity.py",
    "spatial_meta_ce/source_eval.py",
    "spatial_meta_ce/data.py",
    "spatial_meta_ce/meta.py",
    "spatial_meta_ce/plans.py",
    "spatial_meta_ce/metrics.py",
    "spatial_meta_ce/manifests.py",
    "spatial_meta_ce/mappings.py",
    "spatial_meta_ce/tail_risk.py",
    "spatial_meta_ce/features.py",
    "spatial_meta_ce/cli.py",
    "spatial_meta_ce/evaluate.py",
    "spatial_meta_ce/baselines.py",
    "vendor/price_baseline/module.py",
    "vendor/price_baseline/encoder.py",
    "vendor/price_features/features_tool.py",
    "reports/official-module-prep-20261118/input-producer-official-mappings.py.txt",
    "reports/official-module-prep-20261118/input-producer-parallel-v3-mappings.py.txt",
)
PREDICTION_COLUMNS = (
    "arm",
    "seed",
    "database",
    "population",
    "k",
    "run",
    "query_id",
    "query_hash",
    "true_cardinality",
    "log_prediction",
)
POPULATIONS = ("unique-query", "all-rows")
_PLAN_METADATA = {}
_VERIFIED_PLANS = {}


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Repeated JSON object key: {key}")
        result[key] = value
    return result


class _PlanJSONReader:
    """Bounded JSON decoding of one top-level value or episode at a time."""

    def __init__(self, stream):
        self.stream, self.buffer, self.pos, self.eof = stream, "", 0, False
        self.decoder = json.JSONDecoder(object_pairs_hook=_unique_object)

    def fill(self):
        self.buffer = self.buffer[self.pos :]
        self.pos = 0
        chunk = self.stream.read(65536)
        self.buffer += chunk
        self.eof = not chunk
        if len(self.buffer) > 4 * 1024 * 1024:
            raise ValueError(
                "Training plan contains an oversized individual JSON value"
            )

    def peek(self):
        while True:
            while self.pos < len(self.buffer) and self.buffer[self.pos].isspace():
                self.pos += 1
            if self.pos < len(self.buffer):
                return self.buffer[self.pos]
            if self.eof:
                return ""
            self.fill()

    def take(self, token):
        if self.peek() != token:
            raise ValueError(f"Invalid training plan JSON; expected {token!r}")
        self.pos += 1

    def value(self):
        self.peek()
        while True:
            try:
                value, end = self.decoder.raw_decode(self.buffer, self.pos)
                # A number can end at a chunk boundary; read its delimiter first.
                if end == len(self.buffer) and not self.eof:
                    self.fill()
                    continue
                self.pos = end
                return value
            except json.JSONDecodeError as exc:
                if self.eof:
                    raise ValueError("Invalid/truncated training plan JSON") from exc
                self.fill()


def _plan_items(path):
    """Yield headers and individual episodes without retaining a full plan."""
    with regular_path(path).open(encoding="utf-8") as stream:
        reader = _PlanJSONReader(stream)
        reader.take("{")
        seen = set()
        while reader.peek() != "}":
            key = reader.value()
            if (
                not isinstance(key, str)
                or key in seen
                or key in ("episode", "episode_count")
            ):
                raise ValueError("Training plan has repeated or invalid top-level keys")
            seen.add(key)
            reader.take(":")
            if key == "episodes":
                reader.take("[")
                count = 0
                while reader.peek() != "]":
                    yield "episode", reader.value()
                    count += 1
                    if reader.peek() == ",":
                        reader.take(",")
                        if reader.peek() == "]":
                            raise ValueError(
                                "Training plan has a trailing episode comma"
                            )
                    elif reader.peek() != "]":
                        raise ValueError("Training plan episodes are not a JSON array")
                reader.take("]")
                yield "episode_count", count
            else:
                yield key, reader.value()
            if reader.peek() == ",":
                reader.take(",")
                if reader.peek() == "}":
                    raise ValueError("Training plan has a trailing object comma")
            elif reader.peek() != "}":
                raise ValueError("Training plan has an invalid top-level separator")
        reader.take("}")
        if reader.peek():
            raise ValueError("Trailing content after training plan JSON")


def _plan_metadata(path, *, expected_sha256=None):
    actual = file_sha256(path)
    if expected_sha256 is not None and actual != expected_sha256:
        raise ValueError("Training plan file SHA256 changed")
    cache_key = (str(path), actual)
    if cache_key not in _PLAN_METADATA:
        metadata = {key: value for key, value in _plan_items(path) if key != "episode"}
        expected_fields = {
            "schema",
            "kind",
            "seed",
            "steps",
            "meta_batch",
            "accumulation",
            "support_sizes",
            "query_size",
            "databases",
            "data_hashes",
            "plan_sha256",
            "episode_count",
        }
        if set(metadata) != expected_fields:
            raise ValueError("Training plan has missing or extra fields")
        _PLAN_METADATA[cache_key] = metadata
    # Every cache lookup follows a complete file SHA256 check, never stat/mtime.
    return dict(_PLAN_METADATA[cache_key])


def _verify_training_plan(record, seed, capacity, stores, data_hashes):
    metadata = _plan_metadata(record["path"], expected_sha256=record["sha256"])
    expected = {
        "schema": 1,
        "kind": "fomaml_joint_source_training_plan",
        "seed": seed,
        "steps": STEPS,
        "meta_batch": 4,
        "accumulation": 1,
        "support_sizes": [5, 10, 20, 50, 100],
        "query_size": 64,
        "databases": capacity["cache_hashes"],
        "data_hashes": data_hashes,
        "plan_sha256": record["plan_sha256"],
        "episode_count": STEPS * 4,
    }
    if not exact_equal(metadata, expected):
        raise ValueError(
            "Sealed training plan recipe/cache/label/feature hashes differ"
        )
    cache_key = (record["sha256"], canonical_digest(expected))
    if cache_key in _VERIFIED_PLANS:
        return dict(_VERIFIED_PLANS[cache_key])
    fields = sorted(set(metadata) - {"plan_sha256", "episode_count"} | {"episodes"})
    digest = hashlib.sha256()
    digest.update(b"{")
    for position, field in enumerate(fields):
        if position:
            digest.update(b", ")
        digest.update(json.dumps(field).encode() + b": ")
        if field != "episodes":
            digest.update(
                json.dumps(
                    metadata[field], sort_keys=True, ensure_ascii=False, allow_nan=False
                ).encode()
            )
            continue
        digest.update(b"[")
        count = 0
        for key, episode in _plan_items(record["path"]):
            if key != "episode":
                continue
            if not isinstance(episode, dict) or set(episode) != {
                "db",
                "k",
                "support",
                "query",
                "support_hashes",
                "query_hashes",
            }:
                raise ValueError("Invalid sealed training episode schema")
            db, k = episode["db"], episode["k"]
            if (
                db not in TRAIN_DATABASES
                or type(k) is not int
                or k not in metadata["support_sizes"]
            ):
                raise ValueError(
                    "Training episode uses a non-source database or invalid support size"
                )
            store = stores[db]
            for name, size in (("support", k), ("query", 64)):
                indices, hashes = episode[name], episode[name + "_hashes"]
                if (
                    not isinstance(indices, list)
                    or len(indices) != size
                    or not isinstance(hashes, list)
                    or len(hashes) != size
                    or any(
                        type(i) is not int or i < 0 or i >= len(store.rows)
                        for i in indices
                    )
                    or any(not is_sha256(h) for h in hashes)
                    or len(set(indices)) != size
                    or len(set(hashes)) != size
                    or hashes != [store.rows[i]["query_hash"] for i in indices]
                ):
                    raise ValueError(
                        "Training episode ID/hash/size differs from the actual source cache"
                    )
            if set(episode["support"]) & set(episode["query"]) or set(
                episode["support_hashes"]
            ) & set(episode["query_hashes"]):
                raise ValueError("Training episode support/query identities overlap")
            if count:
                digest.update(b", ")
            digest.update(
                json.dumps(
                    episode, sort_keys=True, ensure_ascii=False, allow_nan=False
                ).encode()
            )
            count += 1
        if count != STEPS * 4:
            raise ValueError(
                "Training episode count differs from the fixed update budget"
            )
        digest.update(b"]")
    digest.update(b"}")
    if (
        digest.hexdigest() != record["plan_sha256"]
        or file_sha256(record["path"]) != record["sha256"]
    ):
        raise ValueError("Training plan canonical digest or file bytes changed")
    _VERIFIED_PLANS[cache_key] = metadata
    return dict(metadata)


def file_sha256(path) -> str:
    hasher = hashlib.sha256()
    with regular_path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def file_binding(path, *, plan=False) -> dict:
    path = regular_path(path)
    result = {"path": str(path), "sha256": file_sha256(path)}
    if plan:
        result["plan_sha256"] = _plan_metadata(path)["plan_sha256"]
    return result


def read_bound(record: dict) -> dict:
    binding_shape(record, plan=isinstance(record, dict) and "plan_sha256" in record)
    raw = regular_path(record["path"]).read_bytes()
    if hashlib.sha256(raw).hexdigest() != record["sha256"]:
        raise ValueError(f"Bound input SHA256 changed: {record['path']}")
    value = json.loads(raw, object_pairs_hook=_unique_object)
    if not isinstance(value, dict):
        raise ValueError("Bound JSON input must be an object")
    return value


def write_json(path, value) -> None:
    from scripts.run_price_meta_a import write_json as atomic_write

    regular_path(path, exists=False)
    atomic_write(Path(path), value, replace=False)


def preparation_path(value, data, report) -> Path:
    """Reject writes overlapping any read-only data/cache/mapping input root."""
    path = regular_path(value, exists=False)
    roots = [
        data.get("data_root"),
        data.get("cache_root"),
        report.get("mapping_root"),
        report.get("source_mapping_root"),
    ]
    for value in roots:
        if value is None:
            continue
        root = regular_path(value, exists=False)
        if path.is_relative_to(root) or root.is_relative_to(path):
            raise ValueError("Preparation/output path overlaps a read-only input root")
    return path


def module_resource_sample(
    child, output: Path, device: str, *, telemetry_required=True
) -> dict:
    """Measure owned RSS and worker CUDA counters without a global driver query."""
    from scripts.meta65_joint_supervisor import TELEMETRY_STALE_SECONDS, sample_process

    sample = sample_process(child, mode="cpu_preflight", output=output)
    if device == "cpu":
        return sample
    if device != "cuda":
        raise ValueError("Unknown module sampling device")
    memory_path = output / "memory.json"
    if not telemetry_required and not memory_path.exists():
        # Worker verification precedes model import and telemetry. Runtime unit
        # markers/ready evidence below make the counters mandatory thereafter.
        return sample
    if not memory_path.is_file():
        raise ValueError("Worker CUDA telemetry is missing")
    path = regular_path(memory_path)
    age = time.time() - path.stat().st_mtime
    if age < -1 or age > TELEMETRY_STALE_SECONDS:
        raise ValueError("Worker CUDA telemetry is stale or has a future timestamp")
    values = json.loads(path.read_text())
    fields = (
        "allocated_bytes",
        "reserved_bytes",
        "max_allocated_bytes",
        "max_reserved_bytes",
    )
    if (
        not isinstance(values, dict)
        or "error" in values
        or any(type(values.get(name)) is not int or values[name] < 0 for name in fields)
        or type(values.get("interval_seconds")) not in (int, float)
        or not 0 < values["interval_seconds"] <= TELEMETRY_STALE_SECONDS
        or not math.isfinite(values["interval_seconds"])
        or values["reserved_bytes"] < values["allocated_bytes"]
        or values["max_allocated_bytes"] < values["allocated_bytes"]
        or values["max_reserved_bytes"]
        < max(values["reserved_bytes"], values["max_allocated_bytes"])
    ):
        raise ValueError("Worker CUDA telemetry has invalid counters or an error")
    return {**sample, "vram_bytes": max(values[name] for name in fields)}


def compute_code_sha256(code) -> dict:
    code = regular_path(code, exists=False)
    if not code.is_dir():
        raise ValueError("Module experiment requires a regular code directory")
    result = {}
    for name in REQUIRED_CODE:
        if Path(name).is_absolute() or ".." in Path(name).parts:
            raise ValueError("Invalid code closure path")
        path = regular_path(code / name)
        if not path.resolve().is_relative_to(code.resolve()):
            raise ValueError("Code closure path escapes code directory")
        result[name] = file_sha256(path)
    return result


def verify_code_tree(code, hashes, *, expected_root=None) -> None:
    expected = ROOT if expected_root is None else Path(expected_root)
    code = regular_path(code, exists=False)
    if code != expected or code.resolve() != expected.resolve():
        raise ValueError("Module code_dir must be the executing ROOT")
    if not isinstance(hashes, dict) or set(hashes) != set(REQUIRED_CODE):
        raise ValueError("Module code closure has missing or extra files")
    if not all(is_sha256(value) for value in hashes.values()):
        raise ValueError("Module code closure contains invalid SHA256")
    if compute_code_sha256(code) != hashes:
        raise ValueError("Module code closure differs from its frozen SHA256")


def checked_budget(value) -> dict:
    from scripts.run_price_meta_a import validate_budget

    result = validate_budget(value)
    for field in (
        "rss_bytes",
        "vram_bytes",
        "output_bytes",
        "cpu_core_seconds",
        "gpu_allocated_seconds",
    ):
        if type(value[field]) is not int or value[field] <= 0:
            raise ValueError(
                f"Module budget {field} must be an explicit positive integer"
            )
    return result


def execution_keys(spec) -> list[str]:
    if not exact_equal(spec.get("arms"), list(ARMS)) or not exact_equal(
        spec.get("seeds"), list(SEEDS)
    ):
        raise ValueError(
            "Module matrix must predeclare all five arms and three integer seeds"
        )
    execution = spec.get("execution")
    if exact_equal(execution, {"mode": "matrix"}):
        return [run_key(arm, seed) for arm in ARMS for seed in SEEDS]
    if (
        isinstance(execution, dict)
        and set(execution) == {"mode", "arm", "seed"}
        and execution["mode"] == "single_run"
    ):
        return [run_key(execution["arm"], execution["seed"])]
    raise ValueError(
        "Execution must explicitly select the matrix or one declared arm/seed"
    )


def spec_header(
    action,
    output_dir,
    budget,
    *,
    device="cuda",
    cuda_device=None,
    threads=1,
    batch_size=128,
    execution=None,
):
    if action not in PURPOSES:
        raise ValueError("Unknown module experiment action")
    output = regular_path(output_dir, exists=False)
    budget = checked_budget(budget)
    spec = {
        "schema": 1,
        "protocol": PROTOCOL,
        "purpose": PURPOSES[action],
        "action": action,
        "status": "unsigned",
        "stage": STAGE,
        "arms": list(ARMS),
        "seeds": list(SEEDS),
        "steps": STEPS,
        "selection": FIXED_SELECTION,
        "k": 0,
        "selection_use": "none",
        "execution": {"mode": "matrix"} if execution is None else execution,
        "train_databases": list(TRAIN_DATABASES),
        "validation_databases": [],
        "test_databases": list(TEST_DATABASES),
        "target_databases_read": [] if action == "train" else list(TEST_DATABASES),
        "source_databases_read": list(TRAIN_DATABASES) if action == "train" else [],
        "output_dir": str(output),
        "budget": budget,
        "budget_sha256": canonical_digest(budget),
        "device": device,
        "cuda_device": cuda_device,
        "threads": threads,
        "batch_size": batch_size,
        "bindings": {},
        "config_sha256": {},
    }
    kind = "runs" if action == "train" else "eval"
    spec["outputs"] = {
        key: str(output / "worker" / kind / key) for key in execution_keys(spec)
    }
    return spec


def validate_header(spec) -> dict:
    if not isinstance(spec, dict) or spec.get("action") not in PURPOSES:
        raise ValueError("Not an official module experiment specification")
    action = spec["action"]
    expected = spec_header(
        action,
        spec.get("output_dir"),
        spec.get("budget"),
        device=spec.get("device"),
        cuda_device=spec.get("cuda_device"),
        threads=spec.get("threads"),
        batch_size=spec.get("batch_size"),
        execution=spec.get("execution"),
    )
    for field in expected.keys() - {"bindings", "config_sha256"}:
        if not exact_equal(spec.get(field), expected[field]):
            raise ValueError(f"Official module specification {field} differs")
    if set(spec) != set(expected):
        raise ValueError("Official module specification has missing or extra fields")
    if spec["device"] not in ("cuda", "cpu"):
        raise ValueError("Module device must be cuda or cpu")
    if spec["device"] == "cuda":
        if type(spec["cuda_device"]) is not int or spec["cuda_device"] < 0:
            raise ValueError(
                "Module CUDA execution needs an explicit nonnegative device index"
            )
    elif spec["cuda_device"] is not None:
        raise ValueError("CPU specification cannot declare a CUDA index")
    for field in ("threads", "batch_size"):
        if type(spec[field]) is not int or spec[field] < 1:
            raise ValueError(f"Module {field} must be a positive integer")
    return expected["budget"]


def authorization_template(spec, spec_sha256) -> dict:
    return {
        "schema": 1,
        "protocol": PROTOCOL,
        "purpose": PURPOSES[spec["action"]],
        "action": spec["action"],
        "authorized": False,
        "formal_training_authorized": False,
        "formal_evaluation_authorized": False,
        "spec_sha256": spec_sha256,
        "budget_sha256": spec["budget_sha256"],
        "output_dir": spec["output_dir"],
        "execution_keys": execution_keys(spec),
        "target_databases_read": spec["target_databases_read"],
        "selection_use": "none",
        "granted_by": None,
        "granted_at": None,
    }


def validate_authorization(value, spec, spec_sha256) -> dict:
    # Validate only the lightweight header before any cache/model/output access.
    validate_header(spec)
    if not is_sha256(spec_sha256):
        raise ValueError("Invalid signed spec SHA256")
    expected = authorization_template(spec, spec_sha256)
    expected.update(
        authorized=True,
        formal_training_authorized=spec["action"] == "train",
        formal_evaluation_authorized=spec["action"] == "evaluate",
    )
    if not isinstance(value, dict) or set(value) != set(expected):
        raise ValueError("Module authorization is incomplete or still unsigned")
    for field in expected.keys() - {"granted_by", "granted_at"}:
        if not exact_equal(value[field], expected[field]):
            raise ValueError(
                f"Module authorization {field} differs or is still unsigned"
            )
    if not isinstance(value["granted_by"], str) or not value["granted_by"].strip():
        raise ValueError("Module authorization requires an approving identity")
    try:
        timestamp = datetime.fromisoformat(value["granted_at"])
    except (ValueError, TypeError) as exc:
        raise ValueError("Module authorization needs an ISO8601 approval time") from exc
    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        raise ValueError("Module authorization approval time requires a timezone")
    return value


def _cache_stores(cache_root, split):
    from spatial_meta_ce.data import StorePool

    cache = regular_path(cache_root, exists=False)
    databases = TRAIN_DATABASES if split == "pretrain" else TEST_DATABASES
    for db in databases:
        for name in (
            "manifest.json",
            "cards.npy",
            "offsets.npy",
            "group0.npy",
            "group1.npy",
            "group2.npy",
            "group3.npy",
        ):
            regular_path(cache / split / db / name)
    pool = StorePool(cache, max_open=len(databases))
    return {db: pool.get(split, db) for db in databases}


def _mapping_report(record, data, *, split):
    report = read_bound(record)
    if (
        type(report.get("schema")) is not int
        or report["schema"] != 1
        or report.get("state") != "completed"
        or report.get("scope") != "official_26_4"
        or report.get("cache_root") != data["cache_root"]
        or report.get("data_config_hash") != canonical_digest(data)
        or report.get("resolved_data_config_hash") != canonical_digest(data)
        or not exact_equal(report.get("splits"), data["splits"])
        or report.get("failed_databases") != []
        or type(report.get("unsupported_count")) is not int
        or report["unsupported_count"] != 0
        or report.get("unsupported_count_complete") is not True
        or report.get("model_predictions_run") is not False
        or report.get("training_run") is not False
        or type(report.get("include_test")) is not bool
    ):
        raise ValueError("Module mapping input report scope/cache root differs")
    from spatial_meta_ce.official_mappings import verify_input_report

    root = regular_path(report.get("mapping_root"), exists=False)
    if Path(record["path"]) != root / "official_module_inputs.json":
        raise ValueError("Module mapping report is not at its certified root")
    verify_input_report(report, splits=(split,))
    return report


def _check_input_overlap(output, data, report):
    preparation_path(output, data, report)


def _training_bindings(spec, *, verify_stores):
    from spatial_meta_ce.official_split import (
        build_official_capacity,
        validate_official_capacity,
    )
    from spatial_meta_ce.plans import _store_data_sha256
    from spatial_meta_ce.price_capacity import validate_capacity

    bindings = spec["bindings"]
    fields = {
        "code_dir",
        "code_sha256",
        "data_config",
        "capacity",
        "split_manifest",
        "training_plans",
        "mapping_report",
    }
    if not isinstance(bindings, dict) or set(bindings) not in (
        fields,
        fields | {"templates"},
    ):
        raise ValueError("Module training bindings have missing or extra fields")
    verify_code_tree(bindings["code_dir"], bindings["code_sha256"])
    data = read_bound(bindings["data_config"])
    validate_splits(data)
    regular_path(data.get("cache_root"), exists=False)
    regular_path(data.get("data_root"), exists=False)
    capacity = read_bound(bindings["capacity"])
    validate_official_capacity(capacity)
    validate_capacity(capacity)
    if verify_stores and not exact_equal(capacity, build_official_capacity(data)):
        raise ValueError("Module capacity differs from the actual 26 training caches")
    manifest = read_bound(bindings["split_manifest"])
    if (
        manifest.get("kind") != "official_split_26_train_4_test"
        or not exact_equal(manifest.get("train_databases"), list(TRAIN_DATABASES))
        or not exact_equal(manifest.get("test_databases"), list(TEST_DATABASES))
        or manifest.get("capacity_sha256") != capacity["capacity_sha256"]
        or not exact_equal(manifest.get("max_counts"), capacity["max_counts"])
    ):
        raise ValueError("Module official split manifest differs")
    if not isinstance(bindings["training_plans"], dict) or set(
        bindings["training_plans"]
    ) != {str(s) for s in SEEDS}:
        raise ValueError("Module requires all three sealed official training plans")
    stores = _cache_stores(data["cache_root"], "pretrain") if verify_stores else None
    data_hashes = (
        {db: _store_data_sha256(store) for db, store in stores.items()}
        if verify_stores
        else None
    )
    if (
        verify_stores
        and {db: store.manifest["content_hash"] for db, store in stores.items()}
        != capacity["cache_hashes"]
    ):
        raise ValueError("Module current training cache manifests differ from capacity")
    selected_seeds = (
        {spec["execution"]["seed"]}
        if spec["execution"]["mode"] == "single_run"
        else set(SEEDS)
    )
    plans = {}
    for seed in SEEDS:
        record = bindings["training_plans"][str(seed)]
        binding_shape(record, plan=True)
        entry = manifest.get("plans", {}).get(str(seed), {})
        if (
            entry.get("path") != record["path"]
            or entry.get("file_sha256") != record["sha256"]
            or entry.get("plan_sha256") != record["plan_sha256"]
            or type(entry.get("episodes")) is not int
            or entry["episodes"] != STEPS * 4
        ):
            raise ValueError("Module split manifest training-plan binding differs")
        if verify_stores and seed in selected_seeds:
            plans[str(seed)] = _verify_training_plan(
                record, seed, capacity, stores, data_hashes
            )
        elif verify_stores and file_sha256(record["path"]) != record["sha256"]:
            raise ValueError("Unselected declared training plan file SHA256 changed")
    if verify_stores:
        report = _mapping_report(bindings["mapping_report"], data, split="pretrain")
    else:
        report = read_bound(bindings["mapping_report"])
        if report.get("scope") != "official_26_4":
            raise ValueError("Module training mapping scope differs")
    configs = {}
    for arm in ARMS:
        for seed in SEEDS:
            key = run_key(arm, seed)
            configs[key] = training_config(
                arm,
                seed,
                capacity,
                data_config_binding=bindings["data_config"],
                training_plan_binding=bindings["training_plans"][str(seed)],
                mapping_root=report["mapping_root"] if mapping_mode(arm) else None,
            )
    if "templates" in bindings:
        if set(bindings["templates"]) != {str(s) for s in SEEDS}:
            raise ValueError("Module templates must cover the three seeds exactly")
        for seed in SEEDS:
            if not exact_equal(
                read_bound(bindings["templates"][str(seed)]),
                configs[run_key(ARMS[0], seed)],
            ):
                raise ValueError(
                    "Module template is not its canonical baseline configuration"
                )
    expected_hashes = {key: canonical_digest(value) for key, value in configs.items()}
    if spec["config_sha256"] != expected_hashes:
        raise ValueError("Module canonical config SHA256 bindings differ")
    return {
        "data": data,
        "capacity": capacity,
        "plans": plans,
        "mapping_report": report,
        "configs": {key: configs[key] for key in execution_keys(spec)},
        "all_configs": configs,
    }


def verify_spec(spec) -> dict:
    budget = validate_header(spec)
    if spec["action"] == "train":
        resolved = _training_bindings(spec, verify_stores=True)
    else:
        resolved = _evaluation_bindings(spec)
    _check_input_overlap(
        spec["output_dir"], resolved["data"], resolved["mapping_report"]
    )
    return {**resolved, "budget": budget, "execution_keys": execution_keys(spec)}


def prepare_training(
    *,
    data_config,
    capacity,
    split_manifest,
    training_plans,
    mapping_report,
    output_dir,
    budget,
    cuda_device,
    device="cuda",
    threads=1,
    execution=None,
    templates=None,
):
    """Return an unsigned spec; no model is constructed and no file is written."""
    spec = spec_header(
        "train",
        output_dir,
        budget,
        device=device,
        cuda_device=cuda_device,
        threads=threads,
        execution=execution,
    )
    bindings = {
        "code_dir": str(ROOT),
        "code_sha256": compute_code_sha256(ROOT),
        "data_config": file_binding(data_config),
        "capacity": file_binding(capacity),
        "split_manifest": file_binding(split_manifest),
        "mapping_report": file_binding(mapping_report),
        "training_plans": {
            str(seed): file_binding(training_plans[str(seed)], plan=True)
            for seed in SEEDS
        },
    }
    if templates is not None:
        bindings["templates"] = {
            str(seed): file_binding(templates[str(seed)]) for seed in SEEDS
        }
    spec["bindings"] = bindings
    report = read_bound(bindings["mapping_report"])
    preparation_path(output_dir, read_bound(bindings["data_config"]), report)
    cap = read_bound(bindings["capacity"])
    spec["config_sha256"] = {
        run_key(arm, seed): canonical_digest(
            training_config(
                arm,
                seed,
                cap,
                data_config_binding=bindings["data_config"],
                training_plan_binding=bindings["training_plans"][str(seed)],
                mapping_root=report["mapping_root"] if mapping_mode(arm) else None,
            )
        )
        for arm in ARMS
        for seed in SEEDS
    }
    verify_spec(spec)
    return spec


def _population_entry(store):
    from spatial_meta_ce.plans import _canonical_unique_indices, _store_data_sha256

    indices = _canonical_unique_indices(store).tolist()
    all_indices = list(range(len(store.rows)))
    if not all_indices:
        raise ValueError("Official test population cannot be empty")
    return {
        "cache_hash": store.manifest["content_hash"],
        "data_sha256": _store_data_sha256(store),
        "populations": {
            name: {
                "indices": selected,
                "query_ids": [store.rows[i]["id"] for i in selected],
                "query_hashes": [store.rows[i]["query_hash"] for i in selected],
                "true_cardinality": [float(store.cards[i, 0]) for i in selected],
            }
            for name, selected in zip(POPULATIONS, (indices, all_indices))
        },
    }


def build_evaluation_plan(data):
    validate_splits(data)
    stores = _cache_stores(data["cache_root"], "test")
    payload = {
        "schema": 1,
        "protocol": PROTOCOL,
        "kind": "official-module-k0-population",
        "k": 0,
        "selection_use": "none",
        "databases": {db: _population_entry(stores[db]) for db in TEST_DATABASES},
    }
    return {**payload, "plan_sha256": canonical_digest(payload)}


def _validate_evaluation_plan(plan, data):
    payload = {key: value for key, value in plan.items() if key != "plan_sha256"}
    if (
        plan.get("plan_sha256") != canonical_digest(payload)
        or plan.get("schema") != 1
        or plan.get("protocol") != PROTOCOL
        or plan.get("kind") != "official-module-k0-population"
        or type(plan.get("k")) is not int
        or plan["k"] != 0
        or plan.get("selection_use") != "none"
        or set(plan.get("databases", {})) != set(TEST_DATABASES)
    ):
        raise ValueError("Official module evaluation plan identity differs")
    # Reads test4 only. No training cache, target capacity, support or adaptation.
    actual = build_evaluation_plan(data)
    if not exact_equal(plan, actual) or canonical_digest(plan) != canonical_digest(
        actual
    ):
        raise ValueError(
            "Official module evaluation population/label/feature binding differs"
        )
    return plan


def _frozen_training_records(records_bindings):
    """Verify all checkpoint files before a test store is opened; no model load."""
    if not isinstance(records_bindings, list) or not records_bindings:
        raise ValueError("Evaluation requires the frozen complete training records")
    records = {}
    training = {}
    shared_capacity = None
    shared_data = None
    completions = {}
    for binding in records_bindings:
        document = read_bound(binding)
        if (
            document.get("protocol") != PROTOCOL
            or document.get("completed") is not True
        ):
            raise ValueError(
                "Evaluation requires completed official module training records"
            )
        for entry in document.get("runs", []):
            arm, seed = entry.get("arm"), entry.get("seed")
            key = run_key(arm, seed)
            if key in records or entry.get("key") != key:
                raise ValueError(
                    "Evaluation training records contain repeated/mismatched runs"
                )
            evidence = entry.get("module_execution")
            if not isinstance(evidence, dict) or set(evidence) != {
                "spec",
                "authorization",
                "output_dir",
                "run_key",
            }:
                raise ValueError(
                    "Evaluation training record lacks its execution authorization"
                )
            spec_binding = evidence["spec"]
            spec = read_bound(spec_binding)
            if spec.get("action") != "train":
                raise ValueError(
                    "Evaluation training record uses a non-training specification"
                )
            validate_authorization(
                read_bound(evidence["authorization"]), spec, spec_binding["sha256"]
            )
            completion_path = str(Path(spec["output_dir"]) / "completion.json")
            if completion_path not in completions:
                completion_binding = file_binding(completion_path)
                completion = read_bound(completion_binding)
                usage = completion.get("budget_usage", {})
                if (
                    completion.get("protocol") != PROTOCOL
                    or completion.get("action") != "train"
                    or completion.get("status") != "completed"
                    or completion.get("spec_sha256") != spec_binding["sha256"]
                    or completion.get("authorization_sha256")
                    != evidence["authorization"]["sha256"]
                    or completion.get("supervisor", {}).get("status") != "completed"
                    or completion.get("target_databases_read") != []
                    or usage.get("sampling_failed") is not None
                    or type(usage.get("samples")) is not int
                    or usage["samples"] < 1
                    or not isinstance(usage.get("cpu_core_seconds"), (int, float))
                    or not math.isfinite(usage["cpu_core_seconds"])
                    or not 0
                    <= usage["cpu_core_seconds"]
                    <= spec["budget"]["cpu_core_seconds"]
                    or not isinstance(usage.get("gpu_allocated_seconds"), (int, float))
                    or not math.isfinite(usage["gpu_allocated_seconds"])
                    or not 0
                    <= usage["gpu_allocated_seconds"]
                    <= spec["budget"]["gpu_allocated_seconds"]
                    or (
                        spec["device"] == "cuda" and usage["gpu_allocated_seconds"] <= 0
                    )
                ):
                    raise ValueError(
                        "Evaluation requires the parent-supervised budget-certified training completion"
                    )
                completions[completion_path] = completion_binding
            if (
                evidence["output_dir"] != spec["output_dir"]
                or evidence["run_key"] != key
                or key not in execution_keys(spec)
            ):
                raise ValueError("Evaluation training record differs from signed scope")
            cache_key = spec_binding["sha256"]
            if cache_key not in training:
                validate_header(spec)
                training[cache_key] = _training_bindings(spec, verify_stores=False)
            resolved = training[cache_key]
            config = resolved["configs"][key]
            checkpoint = regular_path(entry.get("checkpoint"))
            if (
                checkpoint.name != "last.pt"
                or checkpoint != Path(spec["outputs"][key]) / "last.pt"
            ):
                raise ValueError("Evaluation only accepts the signed last.pt path")
            if file_sha256(checkpoint) != entry.get("checkpoint_sha256"):
                raise ValueError("Evaluation checkpoint bytes changed")
            if (
                type(entry.get("optimizer_steps")) is not int
                or entry["optimizer_steps"] != STEPS
                or entry.get("training_plan_sha256") != config["training_plan_sha256"]
                or type(entry.get("sample_plan_episodes")) is not int
                or entry["sample_plan_episodes"] != STEPS * 4
                or entry.get("sample_plan_digest") != config["training_plan_sha256"]
            ):
                raise ValueError("Evaluation training completion provenance differs")
            if shared_capacity is None:
                shared_capacity, shared_data = resolved["capacity"], resolved["data"]
            if not exact_equal(
                resolved["capacity"], shared_capacity
            ) or not exact_equal(resolved["data"], shared_data):
                raise ValueError("Evaluation runs do not share official capacity/data")
            records[key] = entry
    expected = {run_key(a, s) for a in ARMS for s in SEEDS}
    if set(records) != expected:
        raise ValueError(
            "Evaluation requires all fifteen frozen arm/seed checkpoints first"
        )
    configs = {}
    for key, entry in records.items():
        resolved = training[entry["module_execution"]["spec"]["sha256"]]
        configs[key] = resolved["configs"][key]
    for seed in SEEDS:
        if (
            len({configs[run_key(arm, seed)]["training_plan_sha256"] for arm in ARMS})
            != 1
        ):
            raise ValueError(
                "Evaluation arms did not consume the same per-seed training plan"
            )
    return {
        "records": records,
        "completion_bindings": completions,
        "all_configs": configs,
        "capacity": shared_capacity,
        "data": shared_data,
    }


def _evaluation_bindings(spec):
    bindings = spec["bindings"]
    required = {
        "code_dir",
        "code_sha256",
        "data_config",
        "training_records",
        "training_completions",
        "evaluation_plan",
        "mapping_report",
    }
    if not isinstance(bindings, dict) or set(bindings) != required:
        raise ValueError("Evaluation bindings have missing or extra fields")
    verify_code_tree(bindings["code_dir"], bindings["code_sha256"])
    frozen = _frozen_training_records(bindings["training_records"])
    if not exact_equal(bindings["training_completions"], frozen["completion_bindings"]):
        raise ValueError("Evaluation parent training completion hashes changed")
    data = read_bound(bindings["data_config"])
    validate_splits(data)
    if not exact_equal(data, frozen["data"]):
        raise ValueError("Evaluation data configuration differs from frozen training")
    expected_hashes = {
        key: canonical_digest(value) for key, value in frozen["all_configs"].items()
    }
    if spec["config_sha256"] != expected_hashes:
        raise ValueError("Evaluation training configuration SHA256 differs")
    plan = _validate_evaluation_plan(read_bound(bindings["evaluation_plan"]), data)
    report = _mapping_report(bindings["mapping_report"], data, split="test")
    return {**frozen, "evaluation_plan": plan, "mapping_report": report}


def prepare_evaluation(
    *,
    training_records,
    data_config,
    mapping_report,
    plan_path,
    output_dir,
    budget,
    cuda_device,
    device="cuda",
    threads=1,
    batch_size=128,
):
    """Freeze test4 populations only after all fifteen checkpoints are frozen."""
    record_bindings = [file_binding(path) for path in training_records]
    frozen = _frozen_training_records(record_bindings)
    data_binding = file_binding(data_config)
    data = read_bound(data_binding)
    if not exact_equal(data, frozen["data"]):
        raise ValueError("Evaluation data differs from training records")
    input_report = _mapping_report(file_binding(mapping_report), data, split="test")
    preparation_path(output_dir, data, input_report)
    path = preparation_path(plan_path, data, input_report)
    plan = build_evaluation_plan(data)
    if path.exists():
        if not exact_equal(json.loads(path.read_text()), plan):
            raise ValueError(
                "Existing evaluation plan differs from frozen test population"
            )
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        write_json(path, plan)
    spec = spec_header(
        "evaluate",
        output_dir,
        budget,
        device=device,
        cuda_device=cuda_device,
        threads=threads,
        batch_size=batch_size,
    )
    spec["bindings"] = {
        "code_dir": str(ROOT),
        "code_sha256": compute_code_sha256(ROOT),
        "data_config": data_binding,
        "training_records": record_bindings,
        "training_completions": frozen["completion_bindings"],
        "evaluation_plan": file_binding(path),
        "mapping_report": file_binding(mapping_report),
    }
    spec["config_sha256"] = {
        key: canonical_digest(value) for key, value in frozen["all_configs"].items()
    }
    verify_spec(spec)
    return spec


def recompute_evaluation(paths, *, plan=None):
    """Independent CSV statistics and exact ID/hash/truth population pairing.

    This intentionally does not call metrics.summarize or the prediction helper.
    Negative log predictions use the protocol's card>=1 Q-error definition;
    raw log predictions in CSV are never overwritten or clipped.
    """
    import numpy as np

    populations = {}
    units = []
    keyed = {}
    for key, path in paths.items():
        groups = {}
        arm, seed = None, None
        with regular_path(path).open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            if tuple(reader.fieldnames or ()) != PREDICTION_COLUMNS:
                raise ValueError("Evaluation CSV columns differ")
            for row in reader:
                arm, seed = row["arm"], int(row["seed"])
                if run_key(arm, seed) != key or row["k"] != "0" or row["run"] != "0":
                    raise ValueError("Evaluation CSV arm/seed/K0 identity differs")
                db, population = row["database"], row["population"]
                if db not in TEST_DATABASES or population not in POPULATIONS:
                    raise ValueError("Evaluation CSV has an unregistered population")
                truth, prediction = (
                    float(row["true_cardinality"]),
                    float(row["log_prediction"]),
                )
                if (
                    not math.isfinite(truth)
                    or truth < 0
                    or not math.isfinite(prediction)
                ):
                    raise ValueError(
                        "Evaluation CSV values must be finite; no failed row is filtered"
                    )
                if not row["query_id"] or not is_sha256(row["query_hash"]):
                    raise ValueError("Evaluation CSV query identity is invalid")
                groups.setdefault((db, population), []).append(
                    (row["query_id"], row["query_hash"], truth, prediction)
                )
        expected = {
            (db, population) for db in TEST_DATABASES for population in POPULATIONS
        }
        if set(groups) != expected:
            raise ValueError("Evaluation CSV is missing a complete test population")
        for db in TEST_DATABASES:
            all_rows = {(row[0], row[1]): row[2:] for row in groups[db, "all-rows"]}
            for row in groups[db, "unique-query"]:
                if all_rows.get((row[0], row[1])) != row[2:]:
                    raise ValueError(
                        "Evaluation unique-query and all-rows truth/prediction pairs differ"
                    )
        for (db, population), rows in groups.items():
            identities = [(r[0], r[1], r[2]) for r in rows]
            if len({r[0] for r in rows}) != len(rows):
                raise ValueError("Evaluation CSV repeats a query ID")
            if population == "unique-query" and len({r[1] for r in rows}) != len(rows):
                raise ValueError("Evaluation unique-query population repeats a hash")
            population_key = (db, population)
            if (
                population_key in populations
                and identities != populations[population_key]
            ):
                raise ValueError(
                    "Evaluation ID/hash/truth population differs across arms/seeds"
                )
            populations[population_key] = identities
            if plan is not None:
                entry = plan["databases"][db]["populations"][population]
                expected_rows = list(
                    zip(
                        entry["query_ids"],
                        entry["query_hashes"],
                        entry["true_cardinality"],
                    )
                )
                if identities != expected_rows:
                    raise ValueError(
                        "Evaluation ID/hash/truth population differs from the sealed plan"
                    )
            log_q = np.asarray(
                [abs(max(r[3], 0.0) - math.log(max(r[2], 1.0))) for r in rows]
            )
            with np.errstate(over="raise", invalid="raise"):
                q = np.exp(log_q)
            if not np.isfinite(q).all():
                raise ValueError(
                    "Evaluation Q-error overflow; raw rows retained without clipping"
                )
            metrics = {f"q{p}": float(np.percentile(q, p)) for p in (50, 90, 95, 99)}
            metrics.update(log_q95=float(np.percentile(log_q, 95)), n=len(rows))
            units.append(
                {
                    "arm": arm,
                    "seed": seed,
                    "database": db,
                    "population": population,
                    "k": 0,
                    "metrics": metrics,
                }
            )
            keyed[arm, seed, db, population] = metrics["log_q95"]
    macro = {}
    deltas = {}
    for arm, seed in sorted({(u["arm"], u["seed"]) for u in units}):
        macro.setdefault(arm, {})[str(seed)] = float(
            np.mean([keyed[arm, seed, db, "unique-query"] for db in TEST_DATABASES])
        )
    for arm, values in macro.items():
        if arm == ARMS[0]:
            continue
        deltas[arm] = {
            seed: value - macro[ARMS[0]][seed]
            for seed, value in values.items()
            if seed in macro.get(ARMS[0], {})
        }
    return {
        "protocol": PROTOCOL,
        "selection_use": "none",
        "k": 0,
        "units": units,
        "macro_log_q95": macro,
        "paired_seed_deltas": deltas,
        "per_database_paired_seed_deltas": {
            arm: {
                db: {
                    seed: keyed[arm, int(seed), db, "unique-query"]
                    - keyed[ARMS[0], int(seed), db, "unique-query"]
                    for seed in values
                }
                for db in TEST_DATABASES
            }
            for arm, values in deltas.items()
        },
        "independent_recompute": True,
    }


def _train_all(spec, resolved, spec_binding, authorization_binding, output, progress):
    import torch
    from spatial_meta_ce.train import train

    records = []
    for key, canonical in resolved["configs"].items():
        evidence = {
            "spec": spec_binding,
            "authorization": authorization_binding,
            "output_dir": spec["output_dir"],
            "run_key": key,
        }
        config = {**canonical, "module_execution": evidence}
        validate_config_shape(config)
        progress("run", key, "start")
        started = time.monotonic()
        result = train(config, resolved["data"], spec["outputs"][key], spec["device"])
        elapsed = time.monotonic() - started
        progress("run", key, "done")
        if elapsed > resolved["budget"]["run_deadline_seconds"]:
            raise TimeoutError(f"Official module run exceeded its deadline: {key}")
        checkpoint = Path(spec["outputs"][key]) / "last.pt"
        if not checkpoint.is_file() or (checkpoint.parent / "best.pt").exists():
            raise ValueError("Official module run must write last.pt only")
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        validate_checkpoint(
            state, config["module_arm"], config["seed"], expected_config=config
        )
        if result.get("counts", {}).get("optimizer_steps") != STEPS:
            raise ValueError("Official module run did not finish 24000 updates")
        records.append(
            {
                "key": key,
                "arm": config["module_arm"],
                "seed": config["seed"],
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": file_sha256(checkpoint),
                "training_plan_sha256": config["training_plan_sha256"],
                "optimizer_steps": STEPS,
                "sample_plan_digest": state["extra"]["sample_plan_digest"],
                "sample_plan_episodes": state["extra"]["sample_plan_episodes"],
                "parameters": result.get("parameters"),
                "wall_seconds": elapsed,
                "training_seconds": result.get("total_seconds"),
                "gpu_timing_scope": "Parent completion budget_usage is cumulative; per-run timing is measured wall time.",
                "module_execution": evidence,
                "memory": json.loads((output / "memory.json").read_text())
                if (output / "memory.json").is_file()
                else None,
            }
        )
    return {
        "protocol": PROTOCOL,
        "completed": True,
        "runs": records,
        "target_databases_read": [],
    }


def _evaluate_all(spec, resolved, output, progress):
    import numpy as np
    import torch
    from spatial_meta_ce.checkpoints import load_checkpoint
    from spatial_meta_ce.data import StorePool
    from spatial_meta_ce.metrics import summarize
    from spatial_meta_ce.official_mappings import OfficialStorePool
    from spatial_meta_ce.source_eval import _predict

    # Raw metadata for every checkpoint is checked before constructing the first
    # model. Pickles are trusted local outputs bound by their SHA256s.
    for key, entry in resolved["records"].items():
        state = torch.load(entry["checkpoint"], map_location="cpu", weights_only=False)
        validate_checkpoint(
            state,
            entry["arm"],
            entry["seed"],
            expected_config={
                **resolved["all_configs"][key],
                "module_execution": entry["module_execution"],
            },
        )
    records = []
    plan = resolved["evaluation_plan"]
    for key in execution_keys(spec):
        record = resolved["records"][key]
        arm, seed = record["arm"], record["seed"]
        progress("eval", key, "start")
        started = time.monotonic()
        model, state = load_checkpoint(record["checkpoint"], device=spec["device"])
        validate_checkpoint(
            state,
            arm,
            seed,
            expected_config={
                **resolved["all_configs"][key],
                "module_execution": record["module_execution"],
            },
        )
        mode = mapping_mode(arm)
        pool = (
            OfficialStorePool(
                resolved["data"]["cache_root"],
                resolved["mapping_report"]["mapping_root"],
                max_open=4,
                mapping_mode=mode,
            )
            if mode
            else StorePool(resolved["data"]["cache_root"], max_open=4)
        )
        eval_dir = Path(spec["outputs"][key])
        eval_dir.mkdir(parents=True, exist_ok=False)
        units = []
        with (eval_dir / "predictions.csv").open(
            "x", newline="", encoding="utf-8"
        ) as stream:
            writer = csv.writer(stream)
            writer.writerow(PREDICTION_COLUMNS)
            for db in TEST_DATABASES:
                progress("eval_unit", f"{key}/{db}", "start")
                store = pool.get("test", db)
                all_indices = plan["databases"][db]["populations"]["all-rows"][
                    "indices"
                ]
                prediction = _predict(
                    model,
                    dict(model.named_parameters()),
                    store,
                    all_indices,
                    spec["device"],
                    spec["batch_size"],
                )
                for population in POPULATIONS:
                    indices = plan["databases"][db]["populations"][population][
                        "indices"
                    ]
                    pred = prediction[np.asarray(indices, dtype=np.int64)]
                    truth = np.asarray(store.cards[indices, 0], dtype=np.float64)
                    for index, value, true in zip(indices, pred, truth):
                        row = store.rows[index]
                        writer.writerow(
                            [
                                arm,
                                seed,
                                db,
                                population,
                                0,
                                0,
                                row["id"],
                                row["query_hash"],
                                format(float(true), ".17g"),
                                format(float(value), ".17g"),
                            ]
                        )
                    stream.flush()
                    units.append(
                        {
                            "arm": arm,
                            "seed": seed,
                            "database": db,
                            "population": population,
                            "k": 0,
                            "metrics": summarize(pred, truth),
                        }
                    )
                progress("eval_unit", f"{key}/{db}", "done")
        progress("eval", key, "done")
        elapsed = time.monotonic() - started
        if elapsed > resolved["budget"]["eval_deadline_seconds"]:
            raise TimeoutError(f"Official module evaluation exceeded deadline: {key}")
        write_json(
            eval_dir / "evaluation.json",
            {
                "protocol": PROTOCOL,
                "arm": arm,
                "seed": seed,
                "completed": True,
                "k": 0,
                "selection_use": "none",
                "units": units,
                "checkpoint_sha256": record["checkpoint_sha256"],
                "population_plan_sha256": plan["plan_sha256"],
                "predictions_sha256": file_sha256(eval_dir / "predictions.csv"),
                "wall_seconds": elapsed,
            },
        )
        records.append({"key": key, "output": str(eval_dir)})
        del model
    return records


def _certify_evaluation(spec, resolved):
    paths = {
        key: Path(spec["outputs"][key]) / "predictions.csv"
        for key in execution_keys(spec)
    }
    summary = recompute_evaluation(paths, plan=resolved["evaluation_plan"])
    keyed = {
        (u["arm"], u["seed"], u["database"], u["population"]): u
        for u in summary["units"]
    }
    for key in execution_keys(spec):
        directory = Path(spec["outputs"][key])
        record = json.loads((directory / "evaluation.json").read_text())
        training = resolved["records"][key]
        if (
            record.get("protocol") != PROTOCOL
            or record.get("completed") is not True
            or record.get("arm") != training["arm"]
            or record.get("seed") != training["seed"]
            or record.get("selection_use") != "none"
            or record.get("k") != 0
            or record.get("predictions_sha256") != file_sha256(paths[key])
            or record.get("checkpoint_sha256") != training["checkpoint_sha256"]
            or record.get("population_plan_sha256")
            != resolved["evaluation_plan"]["plan_sha256"]
            or len(record.get("units", [])) != len(TEST_DATABASES) * len(POPULATIONS)
        ):
            raise ValueError("Evaluation output protocol differs from frozen inputs")
        seen = set()
        for unit in record["units"]:
            identity = (unit["arm"], unit["seed"], unit["database"], unit["population"])
            if identity in seen or identity not in keyed:
                raise ValueError("Evaluation output repeats or invents units")
            if (
                identity[:2] != (training["arm"], training["seed"])
                or type(unit.get("k")) is not int
                or unit["k"] != 0
            ):
                raise ValueError("Evaluation output unit belongs to another run or K")
            seen.add(identity)
            expected = keyed[identity]["metrics"]
            for field in ("n", "log_q95", "q50", "q90", "q95", "q99"):
                if not math.isclose(
                    unit["metrics"][field], expected[field], rel_tol=1e-9, abs_tol=1e-9
                ):
                    raise ValueError(
                        "Evaluation reported metric differs from independent CSV recomputation"
                    )
    return summary


def _read_execution(args, *, require_authorization):
    spec_binding = {"path": str(regular_path(args.spec)), "sha256": args.spec_sha256}
    spec = read_bound(spec_binding)
    validate_header(spec)
    authorization_binding = None
    if args.authorization is not None:
        authorization_binding = {
            "path": str(regular_path(args.authorization)),
            "sha256": args.authorization_sha256,
        }
    if require_authorization:
        if authorization_binding is None:
            raise ValueError("Execution needs separately signed authorization")
        validate_authorization(
            read_bound(authorization_binding), spec, args.spec_sha256
        )
    return spec, spec_binding, authorization_binding


def worker_main(args):
    sys.dont_write_bytecode = True
    if sys.platform != "linux":
        raise ValueError("Official module worker execution requires server Linux")
    if os.environ.get("META65_SUPERVISED_MODE") != "formal" or os.environ.get(
        "META65_SUPERVISOR_PID"
    ) != str(os.getppid()):
        raise ValueError(
            "Official module worker requires its own formal supervisor parent"
        )
    spec, spec_binding, authorization_binding = _read_execution(
        args, require_authorization=True
    )
    expected_visibility = str(spec["cuda_device"]) if spec["device"] == "cuda" else ""
    if os.environ.get("CUDA_VISIBLE_DEVICES") != expected_visibility:
        raise ValueError(
            "Worker CUDA visibility differs from its signed explicit device"
        )
    resolved = verify_spec(spec)
    output = Path(spec["output_dir"]) / "worker"
    output.mkdir(parents=True, exist_ok=False)
    from scripts.run_price_meta_a import CudaMemoryTelemetry, UnitDeadline, _write_phase
    import torch

    torch.set_num_threads(spec["threads"])
    torch.set_num_interop_threads(1)
    monitor = UnitDeadline(
        output,
        resolved["budget"]["run_deadline_seconds"],
        resolved["budget"]["eval_deadline_seconds"],
    )
    telemetry = CudaMemoryTelemetry(output, torch) if spec["device"] == "cuda" else None
    if telemetry:
        telemetry.start()
        write_json(
            output / "telemetry_ready.json", {"required": True, "device": "cuda"}
        )
    write_json(output / "code_before.json", compute_code_sha256(ROOT))
    _write_phase(output, "preparation")
    try:
        if spec["action"] == "train":
            result = _train_all(
                spec,
                resolved,
                spec_binding,
                authorization_binding,
                output,
                monitor.write_unit,
            )
            write_json(output / "training_records.json", result)
        else:
            _write_phase(output, "evaluation")
            _evaluate_all(spec, resolved, output, monitor.write_unit)
        _write_phase(output, "certification")
        verify_code_tree(ROOT, spec["bindings"]["code_sha256"])
        write_json(output / "code_after.json", compute_code_sha256(ROOT))
        write_json(
            output / "status.json",
            {
                "status": "worker_complete",
                "mode": "formal",
                "cells": len(execution_keys(spec)),
                "protocol": PROTOCOL,
            },
        )
    except Exception as exc:
        write_json(
            output / "failure.json",
            {"status": "failed", "error": f"{type(exc).__name__}: {exc}"},
        )
        raise
    finally:
        if telemetry:
            telemetry.close()
    return 0


def parent_main(args):
    from scripts.run_price_meta_a import (
        BudgetTracker,
        UnitDeadline,
        check_output_path,
        parent_sample_fn,
        supervisor_budget,
    )

    spec, spec_binding, authorization_binding = _read_execution(
        args, require_authorization=args.execute
    )
    resolved = verify_spec(spec)
    output = Path(spec["output_dir"])
    check_output_path(output, resolved["data"]["data_root"])
    preparation_path(output, resolved["data"], resolved["mapping_report"])
    if not args.execute:
        status = "not_provided"
        if authorization_binding is not None:
            try:
                validate_authorization(
                    read_bound(authorization_binding), spec, spec_binding["sha256"]
                )
                status = "signed_and_bound"
            except ValueError as exc:
                status = f"not_authorized: {exc}"
        print(
            json.dumps(
                {
                    "dry_run": True,
                    "executed": False,
                    "authorization": status,
                    "runs": execution_keys(spec),
                },
                indent=2,
            )
        )
        return 0
    if sys.platform != "linux":
        raise ValueError("Official module execution requires server Linux")
    os.environ["CUDA_VISIBLE_DEVICES"] = (
        str(spec["cuda_device"]) if spec["device"] == "cuda" else ""
    )
    from scripts.meta65_joint_supervisor import supervise

    execution_started = time.monotonic()
    output.mkdir(parents=True, exist_ok=False)
    worker_output = output / "worker"
    command = [
        sys.executable,
        "-I",
        "-B",
        str(Path(__file__).resolve()),
        "worker",
        "--spec",
        spec_binding["path"],
        "--spec-sha256",
        spec_binding["sha256"],
        "--authorization",
        authorization_binding["path"],
        "--authorization-sha256",
        authorization_binding["sha256"],
    ]
    tracker = BudgetTracker(
        resolved["budget"], requires_gpu_usage=spec["device"] == "cuda"
    )
    monitor = UnitDeadline(
        worker_output,
        resolved["budget"]["run_deadline_seconds"],
        resolved["budget"]["eval_deadline_seconds"],
    )
    measured_by_unit = {}
    previous_usage = [0.0, 0.0]
    sample_base = parent_sample_fn(
        monitor,
        tracker,
        lambda child: module_resource_sample(
            child,
            worker_output,
            spec["device"],
            telemetry_required=(worker_output / "telemetry_ready.json").exists()
            or (worker_output / "unit_progress.json").exists(),
        ),
    )

    def sample_units(child):
        value = sample_base(child)
        # UnitDeadline has already validated all observed markers. Attribute
        # sampled intervals to the currently observed outer run/eval unit;
        # sub-poll transitions are explicitly part of the sampling resolution.
        active = sorted(key for kind, key in monitor._active if kind in ("run", "eval"))
        key = active[0] if len(active) == 1 else "preparation_or_certification"
        usage = measured_by_unit.setdefault(
            key, {"cpu_core_seconds": 0.0, "gpu_allocated_seconds": 0.0}
        )
        for index, field in enumerate(("cpu_core_seconds", "gpu_allocated_seconds")):
            actual = tracker.cpu_seconds if index == 0 else tracker.gpu_seconds
            usage[field] += max(0.0, actual - previous_usage[index])
            previous_usage[index] = actual
        return value

    record = supervise(
        command,
        worker_output,
        mode="formal",
        budget=supervisor_budget(resolved["budget"]),
        sample_fn=sample_units,
        root=str(ROOT),
        grace=10,
        formal_cells=len(execution_keys(spec)),
    )
    if record["status"] != "completed":
        raise RuntimeError(f"Official module worker failed: {record.get('failure')}")
    tracker.finalize()
    verify_code_tree(ROOT, spec["bindings"]["code_sha256"])
    if spec["action"] == "evaluate":
        certified_at = time.monotonic()
        write_json(output / "summary.json", _certify_evaluation(spec, resolved))
        if (
            time.monotonic() - certified_at
            > resolved["budget"]["phase_seconds"]["certification"]
        ):
            raise TimeoutError(
                "Parent independent evaluation certification exceeded its budget"
            )
    tracker.finalize()
    if time.monotonic() - execution_started > resolved["budget"]["total_seconds"]:
        raise TimeoutError(
            "Parent total execution and certification exceeded its budget"
        )
    write_json(
        output / "completion.json",
        {
            "protocol": PROTOCOL,
            "action": spec["action"],
            "status": "completed",
            "spec_sha256": spec_binding["sha256"],
            "authorization_sha256": authorization_binding["sha256"],
            "supervisor": record,
            "budget_usage": tracker.state(),
            "sampled_usage_by_unit": measured_by_unit,
            "unit_accounting_resolution_seconds": resolved["budget"]["poll_seconds"],
            "unit_accounting_note": "Observed run/eval markers assign sampling intervals; boundary transitions within one poll can be attributed to the adjacent unit. Cumulative budget_usage is authoritative.",
            "target_databases_read": spec["target_databases_read"],
            "selection_use": "none",
        },
    )
    print(json.dumps({"status": "completed", "action": spec["action"]}))
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="mode", required=True)
    for mode in ("run", "worker", "verify"):
        command = commands.add_parser(mode)
        command.add_argument("--spec", required=True)
        command.add_argument("--spec-sha256", required=True)
        command.add_argument("--authorization")
        command.add_argument("--authorization-sha256")
        command.add_argument("--execute", action="store_true")
    for mode in ("prepare", "evaluate-ready"):
        command = commands.add_parser(mode)
        command.add_argument("--data-config", required=True)
        command.add_argument("--mapping-report", required=True)
        command.add_argument("--output-dir", required=True)
        command.add_argument(
            "--out", required=True, help="New preparation artifact directory"
        )
        command.add_argument(
            "--budget",
            required=True,
            help="Explicit resource budget JSON; never estimated here",
        )
        command.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
        command.add_argument("--cuda-device", type=int)
        command.add_argument("--threads", type=int, default=1)
        if mode == "prepare":
            command.add_argument("--capacity", required=True)
            command.add_argument("--split-manifest", required=True)
            command.add_argument(
                "--training-plan", action="append", required=True, metavar="SEED=PATH"
            )
            command.add_argument("--arm", choices=ARMS)
            command.add_argument("--seed", type=int, choices=SEEDS)
        else:
            command.add_argument("--training-records", action="append", required=True)
            command.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args(argv)
    if args.mode == "worker":
        return worker_main(args)
    if args.mode == "run":
        return parent_main(args)
    if args.mode == "verify":
        spec, binding, authorization = _read_execution(
            args, require_authorization=False
        )
        result = verify_spec(spec)
        status = "not_provided"
        if authorization is not None:
            try:
                validate_authorization(
                    read_bound(authorization), spec, binding["sha256"]
                )
                status = "signed_and_bound"
            except ValueError as exc:
                status = f"not_authorized: {exc}"
        print(
            json.dumps(
                {
                    "verified": True,
                    "executed": False,
                    "authorization": status,
                    "runs": result["execution_keys"],
                },
                indent=2,
            )
        )
        return 0
    common = {
        "data_config": args.data_config,
        "mapping_report": args.mapping_report,
        "output_dir": args.output_dir,
        "budget": json.loads(regular_path(args.budget).read_text()),
        "cuda_device": args.cuda_device,
        "device": args.device,
        "threads": args.threads,
    }
    out = regular_path(args.out, exists=False)
    if out.exists():
        raise FileExistsError(out)
    preparation_data = read_bound(file_binding(args.data_config))
    preparation_report = read_bound(file_binding(args.mapping_report))
    preparation_path(out, preparation_data, preparation_report)
    preparation_path(args.output_dir, preparation_data, preparation_report)
    if args.mode == "prepare":
        if (args.arm is None) != (args.seed is None):
            raise ValueError("A subset run requires both --arm and --seed")
        plans = {}
        for item in args.training_plan:
            seed, path = item.split("=", 1)
            if seed in plans:
                raise ValueError("Repeated training plan seed")
            plans[seed] = path
        if set(plans) != {str(seed) for seed in SEEDS}:
            raise ValueError("Preparation requires the three sealed seed plans")
        execution = (
            {"mode": "single_run", "arm": args.arm, "seed": args.seed}
            if args.arm
            else None
        )
        spec = prepare_training(
            **common,
            capacity=args.capacity,
            split_manifest=args.split_manifest,
            training_plans=plans,
            execution=execution,
        )
        out.mkdir(parents=True, exist_ok=False)
    else:
        # Only a preparation artifact is written. No model/inference is imported.
        out.mkdir(parents=True, exist_ok=False)
        spec = prepare_evaluation(
            **common,
            training_records=args.training_records,
            plan_path=str(out / "evaluation_plan.json"),
            batch_size=args.batch_size,
        )
    name = "training_spec.json" if args.mode == "prepare" else "evaluation_spec.json"
    write_json(out / name, spec)
    write_json(
        out / "authorization_template.json",
        authorization_template(spec, file_sha256(out / name)),
    )
    print(
        json.dumps(
            {"prepared": str(out / name), "authorized": False, "executed": False},
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
