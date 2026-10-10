#!/usr/bin/env python3
"""Independent verifier for Meta65 joint-control outputs."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

DATABASES = ("chembl", "legalacts", "telstra")
CHECKPOINTS = ("A", "B", "C")
OPERATORS = ("none", "head3", "last3")
COUNTS = {"chembl": 49900, "legalacts": 49404, "telstra": 49900}
PARAM_COUNTS = {"none": 0, "head3": 107521, "last3": 65}
PARAM_STEPS = {"none": 0, "head3": 3, "last3": 3}
CARD_EDGES = (100.0, 1e3, 1e4, 1e5, 1e6)
CARD_LABELS = ("<100", "100-1e3", "1e3-1e4", "1e4-1e5", "1e5-1e6", ">=1e6")
TABLE_LABELS = ("1", "2", "3", "4", "5+")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result

    def constant(value):
        raise ValueError(f"Nonfinite JSON constant: {value}")

    return json.loads(Path(path).read_text(encoding="utf-8"),
                      object_pairs_hook=pairs, parse_constant=constant)


def _inside(root, path):
    path = Path(path)
    if path.is_absolute():
        raise ValueError("Reference must be relative")
    resolved = (root / path).resolve()
    if not resolved.is_relative_to(root.resolve()) or (root / path).is_symlink():
        raise ValueError("Reference escapes output")
    return resolved


def quantile(values, probability):
    values = sorted(float(value) for value in values)
    if not values:
        return None
    if probability <= 0:
        return values[0]
    if probability >= 1:
        return values[-1]
    # Frozen project semantics: NumPy linear interpolation at (n - 1) * p.
    position = probability * (len(values) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return values[lower]
    return values[lower] + (values[upper] - values[lower]) * (position - lower)


def aggregate(values):
    values = [float(value) for value in values]
    if not values:
        return {"count": 0, "q50": None, "q95": None, "status": "empty"}
    if any(not math.isfinite(value) or value < 1.0 for value in values):
        raise ValueError("Q-errors must be finite and >= 1")
    return {"count": len(values), "q50": quantile(values, 0.50),
            "q95": quantile(values, 0.95),
            "status": "ok" if len(values) >= 30 else "below_min_n"}


def joint_masks(true, tables):
    import numpy as np

    true = np.asarray(true, dtype=np.float64)
    tables = np.asarray(tables, dtype=np.int64)
    if true.shape != tables.shape or true.ndim != 1 or not np.isfinite(true).all() \
            or not np.isfinite(tables).all() or (true < 0).any() or (tables < 1).any():
        raise ValueError("Invalid true/table arrays")
    card_index = np.searchsorted(CARD_EDGES, true, side="right")
    table_index = np.minimum(tables, 5) - 1
    result = {}
    for ci, card in enumerate(CARD_LABELS):
        for ti, table in enumerate(TABLE_LABELS):
            result[f"{card}|{table}"] = (card_index == ci) & (table_index == ti)
    return result


def _load_identity(root, relative, expected_sha):
    import numpy as np

    path = _inside(root, relative)
    if not path.is_file() or sha256(path) != expected_sha:
        raise ValueError("Identity SHA mismatch")
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != {"query_hash", "query_id", "true", "n_tables"}:
            raise ValueError("Unexpected identity fields")
        query_hash = archive["query_hash"]
        query_id = archive["query_id"]
        true = archive["true"]
        n_tables = archive["n_tables"]
    if query_hash.dtype != np.dtype("S64") or query_id.dtype != np.dtype("S128") \
            or true.dtype != np.dtype("float64") or n_tables.dtype != np.dtype("int16"):
        raise ValueError("Identity dtypes changed")
    if not (len(query_hash) == len(query_id) == len(true) == len(n_tables)) or not len(true):
        raise ValueError("Identity lengths differ")
    if len(set(query_hash.astype(str))) != len(query_hash) or len(set(query_id.astype(str))) != len(query_id):
        raise ValueError("Duplicate identity")
    if not np.isfinite(true).all() or (true < 0).any() or not np.isfinite(n_tables).all() or (n_tables < 1).any():
        raise ValueError("Invalid identity values")
    return {"query_hash": query_hash, "query_id": query_id, "true": true, "n_tables": n_tables}


def _load_prediction(root, relative, expected_sha, count):
    import numpy as np

    path = _inside(root, relative)
    if not path.is_file() or sha256(path) != expected_sha:
        raise ValueError("Prediction SHA mismatch")
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != {"log_pred"}:
            raise ValueError("Unexpected prediction fields")
        values = archive["log_pred"]
    if values.dtype != np.dtype("float64") or values.ndim != 1 or len(values) != count \
            or not np.isfinite(values).all():
        raise ValueError("Invalid prediction vector")
    return values


def _recompute_metrics(prediction, identity):
    import numpy as np

    true = identity["true"]
    tables = identity["n_tables"]
    log_q = np.abs(np.maximum(prediction, 0.0) - np.log(np.maximum(true, 1.0)))
    with np.errstate(over="raise", invalid="raise"):
        qerror = np.exp(log_q)
    masks = joint_masks(true, tables)
    metrics = {
        "all": aggregate(qerror.tolist()),
        "4+": aggregate(qerror[tables >= 4].tolist()),
        "joint": {key: aggregate(qerror[mask].tolist()) for key, mask in masks.items()},
    }
    return metrics


def _assert_metrics_equal(actual, expected):
    if set(actual) != {"all", "4+", "joint"}:
        raise ValueError("Metrics schema changed")
    for section in ("all", "4+"):
        for key in ("count", "q50", "q95", "status"):
            if actual[section].get(key) != expected[section].get(key):
                if (key in ("q50", "q95") and actual[section].get(key) is not None
                        and expected[section].get(key) is not None
                        and math.isclose(actual[section][key], expected[section][key],
                                         rel_tol=1e-12, abs_tol=0.0)):
                    continue
                raise ValueError(f"Metric mismatch: {section}.{key}")
    if set(actual["joint"]) != set(expected["joint"]):
        raise ValueError("Joint metric keys changed")
    for key in expected["joint"]:
        for field in ("count", "q50", "q95", "status"):
            left = actual["joint"][key].get(field)
            right = expected["joint"][key].get(field)
            if left != right and not (isinstance(left, (int, float)) and isinstance(right, (int, float))
                                      and math.isclose(left, right, rel_tol=1e-12, abs_tol=0.0)):
                raise ValueError(f"Joint metric mismatch: {key}.{field}")


def _matrix_cells(mode):
    if mode == "formal":
        return {(checkpoint, operator, database, run)
                for checkpoint in CHECKPOINTS for operator in OPERATORS
                for database in DATABASES for run in range(5)}
    return {(checkpoint, "last3" if checkpoint == "C" else "head3", database, 0)
            for checkpoint in CHECKPOINTS for database in DATABASES}


def verify(output):
    root = Path(output).resolve()
    if not root.is_dir() or root.is_symlink():
        raise ValueError("Output directory missing or symlinked")
    binding = read_json(root / "input-binding.json")
    if set(binding) != {"schema", "mode", "input_sha256", "code_sha256", "estimated_formal_output_bytes"}:
        raise ValueError("Input binding schema changed")
    mode = binding["mode"]
    if mode not in ("formal", "cpu_preflight"):
        raise ValueError("Unknown mode")
    code = binding["code_sha256"]
    required_code = {
        "scripts/meta65_joint_runner.py", "scripts/meta65_joint_supervisor.py",
        "scripts/meta65_joint_verify.py", "spatial_meta_ce/data.py",
        "spatial_meta_ce/manifests.py", "spatial_meta_ce/mappings.py",
        "spatial_meta_ce/model.py", "spatial_meta_ce/meta.py",
        "spatial_meta_ce/checkpoints.py",
    }
    if not isinstance(code, dict) or not required_code <= set(code):
        raise ValueError("Incomplete code SHA binding")
    if not isinstance(binding["estimated_formal_output_bytes"], int) \
            or binding["estimated_formal_output_bytes"] < 0:
        raise ValueError("Invalid estimated output size")
    for relative, expected in binding["input_sha256"].items():
        path = Path(relative)
        if not path.is_file() or sha256(path) != expected:
            raise ValueError(f"Input SHA changed: {relative}")
    cells_name = "cells.json" if mode == "formal" else "microcells.json"
    payload = read_json(root / cells_name)
    if payload.get("schema") != 1 or payload.get("mode") != mode or not isinstance(payload.get("cells"), list):
        raise ValueError("Cell payload schema changed")
    cells = payload["cells"]
    expected_keys = _matrix_cells(mode)
    seen = set()
    identity_cache = {}
    shared_before = {}
    for cell in cells:
        key = (cell.get("checkpoint"), cell.get("operator"), cell.get("database"), cell.get("run"))
        if key not in expected_keys or key in seen:
            raise ValueError("Unexpected or duplicate cell")
        seen.add(key)
        checkpoint, operator, database, run = key
        if database not in DATABASES or cell.get("k") != 20 or cell.get("k20") != 20:
            raise ValueError("Cell database/K changed")
        if not isinstance(cell.get("count"), int):
            raise TypeError("Cell count missing")
        identity = identity_cache.get((database, run))
        if identity is None:
            identity = _load_identity(root, cell["identity_file"], cell["identity_sha256"])
            identity_cache[(database, run)] = identity
        else:
            if sha256(_inside(root, cell["identity_file"])) != cell["identity_sha256"]:
                raise ValueError("Identity SHA inconsistent across cells")
        if mode == "cpu_preflight":
            positions = cell.get("query_positions")
            if positions != list(range(32)) or cell["count"] != 32:
                raise ValueError("CPU micro query positions changed")
            identity_view = {name: value[positions] for name, value in identity.items()}
        else:
            if "query_positions" in cell:
                raise ValueError("Formal cell must use full identity")
            if cell["count"] != len(identity["true"]) or cell["count"] != COUNTS[database]:
                raise ValueError("Formal denominator changed")
            identity_view = identity
        prediction = _load_prediction(root, cell["prediction_file"], cell["prediction_sha256"], cell["count"])
        _assert_metrics_equal(cell["metrics"], _recompute_metrics(prediction, identity_view))
        if cell["state_before"] != cell["state_after"]:
            raise ValueError("Shared initialization mutated")
        prior = shared_before.setdefault((checkpoint, run), cell["state_before"])
        if prior != cell["state_before"]:
            raise ValueError("Checkpoint initialization differs across cells")
        expected_params = PARAM_COUNTS[operator]
        if cell["selected_parameter_count"] != expected_params or cell["steps"] != PARAM_STEPS[operator]:
            raise ValueError("Operator parameter contract changed")
        if operator == "none" and (cell["selected_parameter_names"] or cell["changed_parameter_names"]):
            raise ValueError("None modified parameters")
        if operator != "none" and not cell["changed_parameter_names"]:
            raise ValueError("Adaptation changed no parameter")
        if not set(cell["changed_parameter_names"]) <= set(cell["selected_parameter_names"]):
            raise ValueError("Changed parameters outside selected scope")
    if seen != expected_keys:
        raise ValueError("Incomplete matrix")
    result = {"status": "preflight_verified" if mode == "cpu_preflight" else "formal_verified",
              "mode": mode, "formal_result": mode == "formal", "cells_verified": len(cells),
              "all_metrics_verified": True, "summary": None}
    return result
