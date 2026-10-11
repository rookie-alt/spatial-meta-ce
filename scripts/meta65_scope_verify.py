#!/usr/bin/env python3
"""Read-only consistency certification for static, trusted source evidence.

SHA bindings detect changed bytes; they do not authenticate a malicious producer.
The caller must freeze source/code bindings and protect the evidence tree from
concurrent replacement. This verifier never opens target-named paths.
"""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any

import numpy as np

from scripts.meta65_joint_verify import (
    _load_identity,
    _load_prediction,
    aggregate,
    joint_masks,
    read_json,
    sha256,
)

ARMS = ("R_head3", "T_all3")
DATABASES = ("chembl", "legalacts", "telstra")
COUNTS = {"chembl": 49900, "legalacts": 49404, "telstra": 49900}
TRAIN = (
    "accidents",
    "airline",
    "baseball",
    "basketball",
    "carcinogenesis",
    "ccs",
    "consumer",
    "credit",
    "employee",
    "financial",
    "fnhk",
    "grants",
    "hepatitis",
    "hockey",
    "movielens",
    "sakila",
    "sap",
    "seznam",
    "ssb",
    "talkingdata",
    "tournament",
    "tpc_h",
    "tubepricing",
)
HEAD_NAMES = frozenset(
    f"head.{layer}.{kind}" for layer in (0, 2, 4) for kind in ("weight", "bias")
)
INITIAL_HASH = "41e594b03ef19c56c3f236237da850929f2ec5c919702b44402dd9b2864d284a"
SAMPLE_HASH = "ad433d964fe562efacd19c8a03005bfe7df530d96bed0ca3535684854f6f4de8"
VALIDATION_HASH = "0a110551da2db90ff619d8461cd06a1a4b62eb82341f6c4219eb7f6436e3ddb2"
CONFIG = {
    "method": "fomaml",
    "seed": 42,
    "meta_batch": 4,
    "accumulation": 1,
    "query_size": 64,
    "support_sizes": [5, 10, 20, 50, 100],
    "outer_lr": 0.0003,
    "inner_lr": 0.001,
    "validate_every": 100,
    "validation_runs": 5,
    "validation_seed": 1729,
    "patience": 41,
    "min_delta": 0.0001,
    "model": {"arch": "sum", "use_pg": True, "layers": 2, "heads": 4, "width": 64},
    "loss": "smooth_l1",
    "inner_steps": 3,
}
TARGET_PATTERN = re.compile(
    r"(?:^|[/_.-])(imdb|stats|genome|ergastf1)(?:$|[/_.-])", re.I
)
SHA_PATTERN = re.compile(r"[0-9a-f]{64}\Z")


def _check_sha(value: Any) -> str:
    if not isinstance(value, str) or SHA_PATTERN.fullmatch(value) is None:
        raise ValueError("Invalid SHA256")
    return value


def _safe_path(root: Path, value: Any, *, external: bool = False) -> Path:
    if not isinstance(value, str) or not value or TARGET_PATTERN.search(value):
        raise ValueError("Invalid or target path forbidden")
    path = Path(value)
    if ".." in path.parts or (path.is_absolute() and not external):
        raise ValueError("Path must remain inside output")
    candidate = path if path.is_absolute() else root / path
    if any(part.is_symlink() for part in (candidate, *candidate.parents)):
        raise ValueError("Symlink path forbidden")
    resolved = candidate.resolve()
    if TARGET_PATTERN.search(str(resolved)):
        raise ValueError("Target path forbidden")
    if not external and not resolved.is_relative_to(root):
        raise ValueError("Path escapes output")
    if not resolved.is_file():
        raise ValueError(f"Evidence file missing: {value}")
    return resolved


def _json(root: Path, name: str) -> dict:
    payload = read_json(_safe_path(root, name))
    if not isinstance(payload, dict):
        raise ValueError("JSON object required")
    return payload


def _binding(root: Path) -> dict:
    binding = _json(root, "input-binding.json")
    if type(binding.get("schema")) is not int or binding["schema"] != 1:
        raise ValueError("Input binding schema changed")
    for field in ("input_sha256", "code_sha256"):
        values = binding.get(field)
        if not isinstance(values, dict) or not values:
            raise ValueError(f"Empty or invalid {field}")
        for path, digest in values.items():
            _check_sha(digest)
            if not isinstance(path, str) or TARGET_PATTERN.search(path):
                raise ValueError("Target input forbidden")
    for relative, expected in binding["input_sha256"].items():
        path = _safe_path(root, relative, external=True)
        if sha256(path) != expected:
            raise ValueError(f"Input SHA mismatch: {relative}")
    code_root = Path(__file__).resolve().parents[1]
    for relative, expected in binding["code_sha256"].items():
        path = _safe_path(code_root, relative)
        if sha256(path) != expected:
            raise ValueError(f"Code SHA mismatch: {relative}")
    return binding


def _bound_source(root: Path, value: Any, expected: Any, binding: dict) -> Path:
    path = _safe_path(root, value, external=True)
    _check_sha(expected)
    if binding["input_sha256"].get(value) != expected or sha256(path) != expected:
        raise ValueError(
            "Reproduction source missing from input binding or SHA mismatch"
        )
    return path


def _integer(value: Any, expected: int, label: str) -> None:
    if type(value) is not int or value != expected:
        raise ValueError(f"Invalid {label}: expected {expected}")


def _validate_training(record: dict, scope: str, *, formal: bool) -> None:
    config, metadata = record.get("config"), record.get("metadata")
    if not isinstance(config, dict) or not isinstance(metadata, dict):
        raise ValueError("Training configuration/metadata missing")
    expected = {**CONFIG, "adapt_scope": scope, "steps": 4000 if formal else 2}
    if any(
        config.get(k) != v or type(config.get(k)) is not type(v)
        for k, v in expected.items()
    ):
        raise ValueError("Frozen training configuration changed")
    allowed = set(expected) | {"purpose", "data_config"}
    if not set(config) <= allowed:
        raise ValueError("Unexpected training configuration fields")
    signature = {k: v for k, v in config.items() if k not in ("steps", "data_config")}
    if (
        metadata.get("training_signature") != signature
        or metadata.get("method") != "fomaml"
        or type(metadata.get("seed")) is not int
        or metadata["seed"] != 42
        or metadata.get("train_databases") != list(TRAIN)
        or metadata.get("validation_databases") != list(DATABASES)
    ):
        raise ValueError("Training provenance/signature changed")
    hashes = metadata.get("manifest_hashes")
    if not isinstance(hashes, dict) or set(hashes) != set(TRAIN) | set(DATABASES):
        raise ValueError("Source manifest set changed")
    for value in hashes.values():
        _check_sha(value)
    if record.get("initialization_digest") != INITIAL_HASH:
        raise ValueError("Frozen initialization changed")
    for field in ("sample_plan_digest", "validation_plan_digest", "checkpoint_sha256"):
        _check_sha(record.get(field))
    steps = 4000 if formal else 2
    for field, count in (
        ("parameters", 132545),
        ("last_step", steps),
        ("optimizer_steps", steps),
        ("sample_plan_episodes", steps * 4),
    ):
        _integer(record.get(field), count, field)
    support = record.get("support_sample_count")
    if type(support) is not int or support <= 0:
        raise ValueError("Invalid support sample count")
    counts = {
        "support_presentations": support * 3,
        "query_presentations": steps * 4 * 64,
        "backward_calls": steps * 4 * 4,
        "optimizer_steps": steps,
    }
    if not isinstance(record.get("counts"), dict) or set(record["counts"]) != set(
        counts
    ):
        raise ValueError("Training budget fields changed")
    for field, count in counts.items():
        _integer(record["counts"][field], count, field)
    if formal and (
        support != 597915
        or record["sample_plan_digest"] != SAMPLE_HASH
        or record["validation_plan_digest"] != VALIDATION_HASH
    ):
        raise ValueError("Frozen training sample/validation plan changed")
    if Path(str(record.get("checkpoint", ""))).name != "last.pt":
        raise ValueError("Training must select fixed last.pt, not best")


def _training_records(root: Path, old: dict, *, formal: bool) -> None:
    payload = _json(root, "training-records.json")
    records = payload.get("records")
    if payload.get("schema") != 1 or not isinstance(records, list) or len(records) != 2:
        raise ValueError("Exactly two training records required")
    if {r.get("arm") for r in records if isinstance(r, dict)} != set(ARMS):
        raise ValueError("Training arm matrix changed")
    for record in records:
        if (
            _check_sha(record.get("endpoint_plan_digest"))
            != "46d6bce098ab97cf3e442b98495f0321732319fdb26be54265af8ef1d7138b68"
        ):
            raise ValueError("Frozen endpoint plan digest changed")
        artifact = _safe_path(root, record.get("metadata_file"))
        if sha256(artifact) != _check_sha(record.get("metadata_sha256")):
            raise ValueError("Training metadata SHA mismatch")
        actual = read_json(artifact)
        fields = set(record) - {
            "metadata_file",
            "metadata_sha256",
            "endpoint_plan_digest",
        }
        if not isinstance(actual, dict) or any(
            actual.get(k) != record[k] for k in fields
        ):
            raise ValueError("Training record differs from authenticated metadata")
        _validate_training(
            record, "head" if record["arm"] == ARMS[0] else "all", formal=formal
        )
        if record["metadata"]["manifest_hashes"] != old["metadata"]["manifest_hashes"]:
            raise ValueError("Training source manifests differ from old B")
    paired = (
        "initialization_digest",
        "sample_plan_digest",
        "sample_plan_episodes",
        "support_sample_count",
        "validation_plan_digest",
        "parameters",
        "counts",
        "optimizer_steps",
        "last_step",
        "endpoint_plan_digest",
    )
    if any(records[0].get(k) != records[1].get(k) for k in paired):
        raise ValueError("Paired training stream/budget differs")
    configs = [
        {k: v for k, v in r["config"].items() if k != "adapt_scope"} for r in records
    ]
    if configs[0] != configs[1]:
        raise ValueError("Training changed more than inner scope")


def _recompute_metrics(prediction: np.ndarray, identity: dict) -> dict:
    true, tables = identity["true"], identity["n_tables"]
    log_q = np.abs(np.maximum(prediction, 0.0) - np.log(np.maximum(true, 1.0)))
    try:
        with np.errstate(over="raise", invalid="raise"):
            qerror = np.exp(log_q)
    except FloatingPointError as error:
        raise ValueError("Nonfinite Q-error") from error
    return {
        "all": aggregate(qerror.tolist()),
        "4+": aggregate(qerror[tables >= 4].tolist()),
        "joint": {
            name: aggregate(qerror[mask].tolist())
            for name, mask in joint_masks(true, tables).items()
        },
        "low4": aggregate(qerror[(true < 100.0) & (tables >= 4)].tolist()),
    }


def _assert_metrics(actual: Any, expected: dict) -> None:
    if not isinstance(actual, dict) or set(actual) != set(expected):
        raise ValueError("Metrics schema changed")
    for name, value in expected.items():
        given = actual[name]
        if isinstance(value, dict):
            _assert_metrics(given, value)
        elif isinstance(value, float):
            if (
                type(given) not in (int, float)
                or not math.isfinite(given)
                or not math.isclose(given, value, rel_tol=1e-12, abs_tol=0.0)
            ):
                raise ValueError(f"Metric mismatch: {name}")
        elif type(given) is not type(value) or given != value:
            raise ValueError(f"Metric mismatch: {name}")


def _identity(root: Path, cell: dict) -> dict:
    _safe_path(root, cell.get("identity_file"))
    identity = _load_identity(
        root, cell["identity_file"], _check_sha(cell.get("identity_sha256"))
    )
    if any(value.ndim != 1 for value in identity.values()):
        raise ValueError("Identity must be one-dimensional")
    hashes, ids = identity["query_hash"].astype(str), identity["query_id"].astype(str)
    if not np.array_equal(hashes, np.sort(hashes)):
        raise ValueError("Identity must be hash sorted")
    if any(not SHA_PATTERN.fullmatch(value) for value in hashes) or any(
        not value for value in ids
    ):
        raise ValueError("Invalid identity hash/id")
    if len(identity["true"]) != COUNTS[cell["database"]]:
        raise ValueError("Full identity denominator changed")
    return identity


def _cell_arrays(root: Path, cell: dict, *, formal: bool) -> tuple[dict, np.ndarray]:
    identity = _identity(root, cell)
    count = COUNTS[cell["database"]] if formal else 32
    _integer(cell.get("count"), count, "query count")
    if formal and "query_positions" in cell:
        raise ValueError("Formal cell must not subset identity")
    if not formal:
        positions = cell.get("query_positions")
        if positions != list(range(32)) or any(type(p) is not int for p in positions):
            raise ValueError("Preflight query positions changed")
        identity = {name: value[:32] for name, value in identity.items()}
    _safe_path(root, cell.get("prediction_file"))
    prediction = _load_prediction(
        root, cell["prediction_file"], _check_sha(cell.get("prediction_sha256")), count
    )
    return identity, prediction


def _cell_key(cell: Any) -> tuple:
    if not isinstance(cell, dict) or type(cell.get("run")) is not int:
        raise ValueError("Invalid cell/run")
    if cell.get("database") not in DATABASES:
        raise ValueError("Target or unknown database forbidden")
    if cell.get("arm") not in ARMS or cell.get("operator") not in ("none", "head3"):
        raise ValueError("Unknown arm/operator")
    if "k" not in cell and "k20" not in cell:
        raise ValueError("K20 missing")
    for field in ("k", "k20"):
        if field in cell:
            _integer(cell[field], 20, field)
    return cell["arm"], cell["operator"], cell["database"], cell["run"]


def _operator(cell: dict) -> None:
    head = cell["operator"] == "head3"
    _integer(
        cell.get("selected_parameter_count"),
        107521 if head else 0,
        "selected parameters",
    )
    _integer(cell.get("steps"), 3 if head else 0, "operator steps")
    selected, changed = (
        cell.get("selected_parameter_names"),
        cell.get("changed_parameter_names"),
    )
    for names in (selected, changed):
        if (
            not isinstance(names, list)
            or any(not isinstance(n, str) for n in names)
            or len(set(names)) != len(names)
        ):
            raise ValueError("Invalid parameter names")
    if not isinstance(selected, list) or not isinstance(changed, list):
        raise ValueError("Parameter name lists required")
    if (
        set(selected) != (HEAD_NAMES if head else set())
        or not set(changed) <= set(selected)
        or (head and not changed)
        or (not head and changed)
    ):
        raise ValueError("Deployment adaptation scope changed")
    if _check_sha(cell.get("state_before")) != _check_sha(cell.get("state_after")):
        raise ValueError("Shared initialization mutated")


def _verify_cells(root: Path, cells: list, *, formal: bool) -> list[dict]:
    expected = {
        (arm, operator, db, run)
        for arm in ARMS
        for operator in ("none", "head3")
        for db in DATABASES
        for run in range(5 if formal else 1)
    }
    keys = [_cell_key(cell) for cell in cells]
    if len(keys) != len(set(keys)) or set(keys) != expected:
        raise ValueError("Incomplete, duplicate or unexpected cell matrix")
    identities: dict[tuple[str, int], tuple[str, str]] = {}
    states: dict[str, str] = {}
    verified = []
    for cell in cells:
        _operator(cell)
        identity_key = cell["database"], cell["run"]
        reference = (cell["identity_file"], cell["identity_sha256"])
        if identities.get(identity_key, reference) != reference:
            raise ValueError("Cells do not share the same identity artifact")
        identities[identity_key] = reference
        if states.get(cell["arm"], cell["state_before"]) != cell["state_before"]:
            raise ValueError("Same-arm initialization differs across cells")
        states[cell["arm"]] = cell["state_before"]
        identity, prediction = _cell_arrays(root, cell, formal=formal)
        recomputed = _recompute_metrics(prediction, identity)
        _assert_metrics(cell.get("metrics"), recomputed)
        verified.append({**cell, "metrics": recomputed})
    return verified


def _old_metadata(root: Path, reproduction: dict, binding: dict) -> dict:
    path = _bound_source(
        root,
        reproduction.get("old_b_metadata_file"),
        reproduction.get("old_b_metadata_sha256"),
        binding,
    )
    old = read_json(path)
    if not isinstance(old, dict) or old.get("arm") != "B_head3_head3":
        raise ValueError("Old B training metadata missing")
    _validate_training(old, "head", formal=True)
    return old


def _anchor_arrays(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != {
            "query_hash",
            "query_id",
            "true",
            "n_tables",
            "log_pred",
        }:
            raise ValueError("Reproduction anchor fields changed")
        arrays = {name: archive[name] for name in archive.files}
    if (
        any(value.ndim != 1 for value in arrays.values())
        or len({len(v) for v in arrays.values()}) != 1
    ):
        raise ValueError("Reproduction anchor dimensions changed")
    if arrays["true"].dtype != np.float64 or arrays["log_pred"].dtype != np.float64:
        raise ValueError("Reproduction anchor FP64 changed")
    for name in ("true", "log_pred", "n_tables"):
        if not np.isfinite(arrays[name]).all():
            raise ValueError("Nonfinite reproduction anchor")
    if arrays["n_tables"].dtype.kind not in "iu" or (arrays["n_tables"] < 1).any():
        raise ValueError("Reproduction anchor tables changed")
    for name in ("query_hash", "query_id"):
        if arrays[name].dtype.kind not in "SU" or len(
            set(arrays[name].astype(str))
        ) != len(arrays[name]):
            raise ValueError("Duplicate or invalid reproduction anchor identity")
    order = np.argsort(arrays["query_hash"].astype(str))
    return {name: value[order] for name, value in arrays.items()}


def verify_reproduction(
    output: Path, cells: list[dict], *, formal: bool = True
) -> dict:
    """Certify R-head endpoints before T runs; never trust the manifest's status flag."""
    root = _root(output)
    binding = _binding(root)
    mode = binding.get("mode")
    if (formal and mode != "formal") or (
        not formal and mode not in ("cpu_preflight", "gpu_preflight")
    ):
        raise ValueError("Reproduction mode mismatch")
    reproduction = _json(root, "reproduction.json")
    _old_metadata(root, reproduction, binding)
    anchors = reproduction.get("anchors")
    if not isinstance(anchors, list):
        raise ValueError("Reproduction anchors missing")
    runs = range(5 if formal else 1)
    expected = {(db, run) for db in DATABASES for run in runs}
    by_key = {}
    for anchor in anchors:
        if (
            not isinstance(anchor, dict)
            or anchor.get("database") not in DATABASES
            or type(anchor.get("run")) is not int
        ):
            raise ValueError("Invalid reproduction anchor database/run")
        key = anchor["database"], anchor["run"]
        if key in by_key:
            raise ValueError("Duplicate reproduction anchor")
        by_key[key] = anchor
    if set(by_key) != expected:
        raise ValueError("Incomplete reproduction anchors")
    references = [cell for cell in cells if _cell_key(cell)[:2] == (ARMS[0], "head3")]
    if (
        len(references) != len(expected)
        or {(c["database"], c["run"]) for c in references} != expected
    ):
        raise ValueError("Incomplete R-head reproduction cells")
    for cell in references:
        _operator(cell)
        anchor = by_key[cell["database"], cell["run"]]
        path = _bound_source(root, anchor.get("path"), anchor.get("sha256"), binding)
        aligned = _anchor_arrays(path)
        if len(aligned["true"]) != COUNTS[cell["database"]]:
            raise ValueError("Reproduction anchor denominator changed")
        identity, prediction = _cell_arrays(root, cell, formal=formal)
        _assert_metrics(cell.get("metrics"), _recompute_metrics(prediction, identity))
        aligned = aligned if formal else {k: v[:32] for k, v in aligned.items()}
        for name, actual in identity.items():
            expected_array = aligned[name]
            if name in ("query_hash", "query_id"):
                actual, expected_array = actual.astype(str), expected_array.astype(str)
            if not np.array_equal(actual, expected_array):
                raise ValueError(f"Reproduction identity differs: {name}")
        if formal:
            if not np.allclose(prediction, aligned["log_pred"], rtol=1e-5, atol=1e-6):
                raise ValueError("Reproduction raw log_pred tolerance failed")
            actual = _recompute_metrics(prediction, identity)["all"]["q95"]
            expected_q95 = _recompute_metrics(aligned["log_pred"], identity)["all"][
                "q95"
            ]
            if not math.isclose(actual, expected_q95, rel_tol=1e-4, abs_tol=0.0):
                raise ValueError("Reproduction Q95 tolerance failed")
    _binding(root)
    return {
        **reproduction,
        "status": "verified" if formal else "not_applicable",
        "reproduction_verified": formal,
        "formal_result": formal,
        "anchors_verified": len(references),
    }


def _decision(cells: list[dict]) -> dict:
    """Compute all prespecified AND gates from independently recomputed per-run Q."""

    def mean(
        arm: str, operator: str, db: str, bucket: str, metric: str
    ) -> float | None:
        values = [
            c["metrics"][bucket][metric]
            for c in cells
            if (c["arm"], c["operator"], c["database"]) == (arm, operator, db)
        ]
        return (
            sum(values) / len(values)
            if len(values) == 5
            and all(type(v) in (int, float) and math.isfinite(v) for v in values)
            else None
        )

    def bounded(left: float | None, right: float | None) -> bool:
        return left is not None and right is not None and left <= 1.1 * right

    primary = {
        arm: [mean(arm, "head3", db, "4+", "q95") for db in DATABASES] for arm in ARMS
    }
    macros = {
        arm: sum(v for v in values if v is not None) / 3
        if all(v is not None for v in values)
        else None
        for arm, values in primary.items()
    }
    r, t = macros[ARMS[0]], macros[ARMS[1]]
    improvement = 1.0 - t / r if r is not None and t is not None and r > 0 else None
    gates = {"primary_10_percent": improvement is not None and improvement >= 0.1}
    for db in DATABASES:
        gates[f"{db}_all_four_q50_q95"] = all(
            bounded(
                mean(ARMS[1], "head3", db, bucket, metric),
                mean(ARMS[0], "head3", db, bucket, metric),
            )
            for bucket in ("all", "4+")
            for metric in ("q50", "q95")
        )
        gates[f"{db}_low4_reference"] = bounded(
            mean(ARMS[1], "head3", db, "low4", "q95"),
            mean(ARMS[0], "head3", db, "low4", "q95"),
        )
        gates[f"{db}_low4_no_negative_transfer"] = bounded(
            mean(ARMS[1], "head3", db, "low4", "q95"),
            mean(ARMS[1], "none", db, "low4", "q95"),
        )
    gates["low4_min_n"] = all(c["metrics"]["low4"]["count"] >= 30 for c in cells)
    decision = "promising_single_seed_K20" if all(gates.values()) else "no_candidate"
    return {
        "status": decision,
        "decision": decision,
        "primary_macro": macros,
        "primary_improvement": improvement,
        "gates": gates,
    }


def _root(output: Path) -> Path:
    path = Path(output).absolute()
    if any(part.is_symlink() for part in (path, *path.parents)) or not path.is_dir():
        raise ValueError("Output directory missing or symlinked")
    return path.resolve()


def verify(output: Path, *, formal: bool = True) -> dict:
    """Verify all raw source evidence, frozen training, R reproduction and decision."""
    root = _root(output)
    binding = _binding(root)
    mode = binding.get("mode")
    if (formal and mode != "formal") or (
        not formal and mode not in ("cpu_preflight", "gpu_preflight")
    ):
        raise ValueError("Verification mode mismatch")
    payload = _json(root, "cells.json" if formal else "microcells.json")
    if (
        type(payload.get("schema")) is not int
        or payload["schema"] != 1
        or payload.get("mode") != mode
        or not isinstance(payload.get("cells"), list)
    ):
        raise ValueError("Cell payload schema changed")
    cells = _verify_cells(root, payload["cells"], formal=formal)
    reproduction = _json(root, "reproduction.json")
    old = _old_metadata(root, reproduction, binding)
    _training_records(root, old, formal=formal)
    verified_reproduction = verify_reproduction(root, cells, formal=formal)
    summary = _decision(cells) if formal else None
    _binding(root)
    return {
        "status": "formal_verified" if formal else "preflight_verified",
        "mode": mode,
        "formal_result": formal,
        "engineering_verified": True,
        "reproduction_verified": verified_reproduction["reproduction_verified"],
        "cells_verified": len(cells),
        "all_metrics_verified": True,
        "summary": summary,
    }
