"""Independent NumPy-only verification of a Meta65 source-holdout output."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = "meta65-source-holdout-20261018"
ARM = "R_head3"
OPERATORS = ("none", "head3")
VALIDATION = ("chembl", "legalacts", "telstra")
HOLDOUT = ("hepatitis", "sakila")
DATABASES = (*VALIDATION, *HOLDOUT)
MODES = ("cpu_preflight", "gpu_preflight", "formal")
TARGETS = ("imdb", "stats", "genome", "ergastf1")
TRAIN = (
    "accidents", "airline", "baseball", "basketball", "carcinogenesis", "ccs",
    "consumer", "credit", "employee", "financial", "fnhk", "grants", "hockey",
    "movielens", "sap", "seznam", "ssb", "talkingdata", "tournament", "tpc_h",
    "tubepricing",
)
EXPECTED = {"cpu_preflight": 10, "gpu_preflight": 10, "formal": 50}
SHA = re.compile(r"[0-9a-f]{64}\Z")
RTOL = ATOL = 1e-12


def require(value: bool, message: str) -> None:
    if not value:
        raise ValueError(message)


def checked_path(path: Path | str) -> Path:
    path = Path(path).absolute()
    require(not any(item.is_symlink() and item not in (Path("/var"), Path("/tmp"))
                    for item in (path, *path.parents)),
            f"Symlink forbidden: {path}")
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
            require(key not in result, f"Duplicate JSON key: {key}")
            result[key] = value
        return result

    def constant(value):
        raise ValueError(f"Nonfinite JSON value: {value}")

    value = json.loads(
        checked_path(path).read_text(encoding="utf-8"),
        object_pairs_hook=pairs,
        parse_constant=constant,
    )
    require(isinstance(value, dict), "JSON root must be an object")
    return value


def metric(q: np.ndarray, min_count: int = 30) -> dict:
    if len(q) == 0:
        return {"count": 0, "status": "empty", "q50": None, "q95": None}
    return {
        "count": len(q),
        "status": "below_min_n" if len(q) < min_count else "ok",
        "q50": float(np.percentile(q, 50, method="linear")),
        "q95": float(np.percentile(q, 95, method="linear")),
    }


def buckets(log_pred: np.ndarray, true: np.ndarray, tables: np.ndarray,
            min_count: int = 30) -> dict:
    require(log_pred.shape == true.shape == tables.shape,
            "Prediction/identity shape mismatch")
    require(np.isfinite(log_pred).all() and np.isfinite(true).all(),
            "Nonfinite prediction or truth")
    require((true >= 0).all() and tables.dtype.kind in "iu" and (tables >= 1).all(),
            "Invalid truth/table counts")
    with np.errstate(over="raise", invalid="raise"):
        q = np.exp(np.abs(np.maximum(log_pred, 0) - np.log(np.maximum(true, 1))))
    masks = {
        "all": np.ones(len(true), dtype=bool),
        "4+": tables >= 4,
        "low4": (tables >= 4) & (true < 100),
    }
    return {name: metric(q[mask], min_count) for name, mask in masks.items()}


def compare_metric(actual: object, expected: dict, label: str) -> None:
    require(isinstance(actual, dict) and set(actual) == set(expected),
            f"{label}: metric schema changed")
    require(actual["count"] == expected["count"] and actual["status"] == expected["status"],
            f"{label}: count/status mismatch")
    for field in ("q50", "q95"):
        if expected[field] is None:
            require(actual[field] is None, f"{label}: {field} must be null")
            continue
        require(isinstance(actual[field], (int, float)) and not isinstance(actual[field], bool)
                and math.isclose(actual[field], expected[field], rel_tol=RTOL, abs_tol=ATOL),
                f"{label}: {field} mismatch")


def prediction_arrays(root: Path, cell: dict) -> dict[str, np.ndarray]:
    path = checked_path(root / cell["prediction_file"])
    require(path.is_relative_to(root), "Prediction escapes output root")
    require(sha256(path) == cell["prediction_sha256"], "Prediction SHA mismatch")
    try:
        with np.load(path, allow_pickle=False) as archive:
            require(len(archive.files) == len(set(archive.files)), "Duplicate NPZ field")
            values = {name: archive[name] for name in archive.files}
    except (OSError, ValueError, zipfile.BadZipFile, EOFError) as error:
        raise ValueError(f"Invalid prediction archive: {error}") from error
    require(set(values) == {"log_pred", "true", "n_tables", "query_hash", "query_id"},
            "Prediction field set changed")
    n = len(values["true"])
    require(n > 0 and all(value.ndim == 1 and len(value) == n for value in values.values()),
            "Prediction dimensions changed")
    for name in ("log_pred", "true"):
        require(values[name].dtype == np.float64 and np.isfinite(values[name]).all(),
                f"Prediction {name} must be finite FP64")
    require(values["n_tables"].dtype == np.int16 and (values["n_tables"] >= 1).all(),
            "Prediction table counts must be positive int16")
    for name in ("query_hash", "query_id"):
        require(values[name].dtype.kind in "SU" and all(values[name].astype(str)),
                f"Prediction {name} invalid")
        require(len(set(values[name].astype(str))) == n, f"Duplicate prediction {name}")
    return values


def verify_plan(plan: dict, mode: str) -> dict:
    require(set(plan) == set(DATABASES), "Plan database closure changed")
    expected_runs = 5
    for database in DATABASES:
        require(isinstance(plan[database], list) and len(plan[database]) == expected_runs,
                "Plan run count changed")
        for run, entry in enumerate(plan[database]):
            require(entry.get("database") == database and entry.get("run") == run,
                    "Plan identity mismatch")
            support = entry.get("support_max")
            query = entry.get("query")
            require(isinstance(support, list) and len(support) == 100
                    and len(set(support)) == 100, "Plan support must be 100 unique")
            require(isinstance(query, list) and query and len(set(query)) == len(query),
                    "Plan query must be nonempty and unique")
            require(not set(support) & set(query), "Plan support/query overlap")
            require(entry.get("supports") == {str(k): support[:k] for k in (5, 10, 20, 50, 100)},
                    "Plan nested supports changed")
            require(SHA.fullmatch(str(entry.get("query_digest", ""))) is not None,
                    "Plan query digest invalid")
            require(set(entry.get("support_digests", {})) == {"5", "10", "20", "50", "100"},
                    "Plan support digest fields changed")
    return plan


def verify_cells(output: Path | str, *, formal: bool) -> dict:
    root = checked_path(output)
    require(root.is_dir(), "Output directory missing")
    status = read_json(root / "status.json")
    mode = status.get("mode")
    require(mode in MODES and formal is (mode == "formal"), "Output mode mismatch")
    expected_cells = EXPECTED[mode]
    if formal:
        require(status.get("status") in ("worker_complete", "completed")
                and status.get("cells") == expected_cells,
                "Incomplete holdout worker matrix")
    else:
        require(status.get("status") in ("worker_complete", "preflight_completed")
                and status.get("cells") == 0
                and status.get("micro_cells") == expected_cells,
                "Incomplete holdout preflight matrix")
    require(status.get("training") is True, "Training flag missing")
    require(status.get("target_read") is False, "Target read flag must be false")
    require(status.get("selection_performed") is False, "Candidate selection is forbidden")
    environment = read_json(root / "environment.json")
    require(environment.get("mode") == mode and environment.get("training") is True,
            "Environment mode/training evidence changed")
    binding = read_json(root / "input-binding.json")
    require(binding.get("mode") == mode and isinstance(binding.get("input_sha256"), dict),
            "Input binding schema changed")
    for name, expected in binding["input_sha256"].items():
        require(SHA.fullmatch(str(expected)) is not None and sha256(name) == expected,
                f"Bound input changed: {name}")
    plan_path = checked_path(binding["plan_file"])
    require(sha256(plan_path) == binding["plan_sha256"], "Plan SHA mismatch")
    plan = verify_plan(read_json(plan_path), mode)
    cells_doc = read_json(root / ("cells.json" if formal else "microcells.json"))
    cells = cells_doc.get("cells")
    require(isinstance(cells, list) and len(cells) == expected_cells,
            "Cell document count changed")
    expected_keys = {
        (ARM, operator, database, run)
        for operator in OPERATORS for database in DATABASES
        for run in (range(5) if formal else range(1))
    }
    index = {}
    state = None
    for cell in cells:
        require(isinstance(cell, dict), "Cell must be object")
        key = (cell.get("arm"), cell.get("operator"), cell.get("database"), cell.get("run"))
        require(key in expected_keys and key not in index, "Unexpected or duplicate cell")
        values = prediction_arrays(root, cell)
        require(cell.get("k20") == 20 and cell.get("query_count") == len(values["true"]),
                "K20/query count changed")
        database, run = key[2], key[3]
        frozen = plan[database][run]
        require(cell.get("support_digest") == frozen["support_digests"]["20"],
                "Cell support digest differs from plan")
        require(cell.get("query_digest") == frozen["query_digest"],
                "Cell query digest differs from plan")
        expected_query = frozen["query"] if formal else frozen["query"][:32]
        require(len(expected_query) == len(values["true"]), "Cell query denominator changed")
        recomputed = buckets(values["log_pred"], values["true"], values["n_tables"])
        require(set(cell.get("metrics", {})) == {"all", "4+", "low4"},
                "Cell metric buckets changed")
        for name, expected in recomputed.items():
            compare_metric(cell["metrics"][name], expected, f"{key}:{name}")
        require(cell.get("state_before") == cell.get("state_after"),
                "Shared initialization was mutated")
        if state is None:
            state = cell.get("state_before")
        require(cell.get("state_before") == state, "Initialization differs across cells")
        if key[1] == "none":
            require(cell.get("steps") == 0 and cell.get("selected_parameter_count") == 0
                    and cell.get("changed_parameter_names") == [],
                    "none operator changed parameters")
        else:
            require(cell.get("steps") == 3 and cell.get("selected_parameter_count") == 107521,
                    "head3 operator budget changed")
            require(cell.get("changed_parameter_names"), "head3 did not change a parameter")
        index[key] = cell
    require(set(index) == expected_keys, "Incomplete cell matrix")
    metadata = read_json(root / ARM / "metadata.json")
    require(metadata.get("metadata", {}).get("train_databases") == list(TRAIN),
            "Training database list changed")
    require(metadata.get("metadata", {}).get("validation_databases") == list(VALIDATION),
            "Validation database list changed")
    require(isinstance(metadata.get("data_signature"), str)
            and SHA.fullmatch(metadata["data_signature"]) is not None,
            "Dataset signature missing")
    evidence_files = (
        root / "status.json", root / "environment.json", root / "input-binding.json",
        root / "plan_summary.json", root / "cells.json", root / "microcells.json",
        root / "report.json", root / "verification.json", root / ARM / "metadata.json",
    )
    for path in evidence_files:
        if path.is_file() and path.suffix in (".json", ".txt", ".csv"):
            text = checked_path(path).read_text(encoding="utf-8", errors="ignore")
            require(not re.search(rf"\b(?:{'|'.join(TARGETS)})\b", text, re.IGNORECASE),
                    f"Target database name found: {path}")
    return {
        "schema": 1,
        "experiment": EXPERIMENT,
        "status": "formal_verified" if formal else "preflight_verified",
        "mode": mode,
        "formal_result": formal,
        "cells_verified": expected_cells,
        "all_metrics_verified": True,
        "engineering_verified": True,
        "selection_performed": False,
        "target_read": False,
        "metric": "per-run standard unique-query Q50/Q95; groups never pooled",
    }


def verify(output: Path | str, *, formal: bool) -> dict:
    result = verify_cells(output, formal=formal)
    if formal:
        require(read_json(checked_path(output) / "report.json").get("status") == "baseline_verified",
                "Formal report status changed")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=MODES, required=True)
    args = parser.parse_args(argv)
    result = verify(args.output, formal=args.mode == "formal")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
