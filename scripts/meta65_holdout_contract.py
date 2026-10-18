"""Frozen contract for the Meta65 fixed-source-holdout baseline."""
from __future__ import annotations

import hashlib
import json
import math
import random
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = "meta65-source-holdout-20261018"
PROTOCOL = "实验文档/81-源库留出协议.md"
PROTOCOL_SHA = "da5a73d693ed886de25e5f1d904e6b5ff3eccc233c39e04f793514140852315b"
VALIDATION = ("chembl", "legalacts", "telstra")
HOLDOUT = ("hepatitis", "sakila")
EVALUATION = (*VALIDATION, *HOLDOUT)
TARGETS = ("imdb", "stats", "genome", "ergastf1")
TRAIN = (
    "accidents", "airline", "baseball", "basketball", "carcinogenesis", "ccs",
    "consumer", "credit", "employee", "financial", "fnhk", "grants", "hockey",
    "movielens", "sap", "seznam", "ssb", "talkingdata", "tournament", "tpc_h",
    "tubepricing",
)
ALL_SOURCES = (*TRAIN, *VALIDATION, *HOLDOUT)
ARM = "R_head3"
OPERATORS = ("none", "head3")
RUNS = tuple(range(5))
K = 20
SHOTS = (5, 10, 20, 50, 100)
MODES = ("dry_run", "cpu_preflight", "gpu_preflight", "formal")
CACHE_FILES = ("manifest.json", "cards.npy", "offsets.npy", *(f"group{i}.npy" for i in range(4)))
UNIQUE = {
    "chembl": 50000, "legalacts": 49504, "telstra": 50000,
    "hepatitis": 50000, "sakila": 50000,
}
FOUR_PLUS = {
    "chembl": 33947, "legalacts": 9178, "telstra": 20311,
    "hepatitis": 38538, "sakila": 35295,
}
QUERY_COUNTS = {name: count - 100 for name, count in UNIQUE.items()}

FROZEN = {
    "schema": 1,
    "experiment": EXPERIMENT,
    "protocol_file": PROTOCOL,
    "protocol_sha256": PROTOCOL_SHA,
    "training": {
        "method": "fomaml", "seed": 42, "steps": 4000, "meta_batch": 4,
        "accumulation": 1, "query_size": 64, "support_sizes": list(SHOTS),
        "outer_lr": 0.0003, "inner_lr": 0.001, "validate_every": 100,
        "validation_runs": 5, "validation_seed": 1729, "patience": 41,
        "min_delta": 0.0001,
    },
    "model": {"arch": "sum", "use_pg": True, "layers": 2, "heads": 4},
    "arm": {"name": ARM, "train_scope": "head", "train_steps": 3},
    "evaluation": {
        "support_seed": 2718, "support_runs": 5, "k": K, "batch_size": 512,
        "operators": list(OPERATORS),
    },
    "expected": {"unique": UNIQUE, "four_plus": FOUR_PLUS},
    "limits": {
        "total_seconds": 14400,
        "phase_seconds": {"preparation": 300, "evaluation": 13200, "certification": 900},
        "rss_bytes": 12 * 1024**3,
        "vram_bytes": 8 * 1024**3,
        "output_bytes": 1024**3,
        "poll_seconds": 1.0,
    },
    "preflight": {
        "training_steps": 2, "micro_queries": 32, "total_seconds": 900,
        "phase_seconds": {"preparation": 300, "evaluation": 300, "certification": 300},
        "poll_seconds": 1.0,
    },
    "report": {"min_bucket_count": 30, "selection_performed": False},
}

CODE_FILES = (
    "scripts/meta65_holdout_contract.py",
    "scripts/build_meta65_holdout_plan.py",
    "scripts/run_meta65_holdout.py",
    "scripts/meta65_holdout_verify.py",
    "scripts/meta65_holdout_report.py",
    "scripts/meta65_holdout_launch.py",
    "scripts/build_meta65_holdout_binding.py",
    "scripts/meta65_joint_runner.py",
    "scripts/meta65_joint_supervisor.py",
    "spatial_meta_ce/train.py",
    "spatial_meta_ce/meta.py",
    "spatial_meta_ce/data.py",
    "spatial_meta_ce/model.py",
    "spatial_meta_ce/checkpoints.py",
    "spatial_meta_ce/manifests.py",
    "spatial_meta_ce/metrics.py",
)


def digest(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":")
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def sha256(path: Path | str) -> str:
    result = hashlib.sha256()
    with checked_path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def checked_path(path: Path | str) -> Path:
    path = Path(path).absolute()
    if any(item.is_symlink() and item not in (Path("/var"), Path("/tmp"))
           for item in (path, *path.parents)):
        raise ValueError(f"Symlink forbidden: {path}")
    return path.resolve()


def read_json(path: Path | str) -> Any:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result

    def constant(value: str) -> None:
        raise ValueError(f"Nonfinite JSON value: {value}")

    return json.loads(
        checked_path(path).read_text(encoding="utf-8"),
        object_pairs_hook=pairs,
        parse_constant=constant,
    )


def write_json(path: Path | str, value: Any) -> None:
    path = checked_path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    if temporary.exists():
        raise FileExistsError(f"Temporary path exists: {temporary}")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def validate_config(config: Any) -> dict[str, Any]:
    if not isinstance(config, dict) or digest(config) != digest(FROZEN):
        raise ValueError("Frozen source-holdout config changed or unknown fields supplied")
    return config


def load_config(path: Path | str, root: Path = ROOT) -> dict[str, Any]:
    config = validate_config(read_json(path))
    protocol = checked_path(root / config["protocol_file"])
    if not protocol.is_relative_to(checked_path(root)) or sha256(protocol) != PROTOCOL_SHA:
        raise ValueError("Source-holdout protocol bytes changed")
    return config


def validate_data(data: Any, root: Path = ROOT) -> dict[str, Any]:
    expected = {
        "train": list(TRAIN),
        "validation": list(VALIDATION),
        "holdout": list(HOLDOUT),
        "test": list(TARGETS),
    }
    if not isinstance(data, dict) or set(data) != {"data_root", "cache_root", "splits"}:
        raise ValueError("Only source-holdout data/cache roots may change")
    if data["splits"] != expected:
        raise ValueError("Frozen 21/3/2 source split or target list changed")
    groups = list(expected.values())
    for index, group in enumerate(groups):
        if len(group) != len(set(group)):
            raise ValueError("Duplicate database in source split")
        for other in groups[index + 1:]:
            if set(group) & set(other):
                raise ValueError("Database leakage between source splits")
    result: dict[str, Any] = {"splits": expected}
    for key in ("data_root", "cache_root"):
        value = data[key]
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Invalid {key}")
        path = Path(value).expanduser()
        result[key] = str(checked_path(path if path.is_absolute() else checked_path(root) / path))
    if len(ALL_SOURCES) != 26 or len(set(ALL_SOURCES)) != 26:
        raise ValueError("Source closure changed")
    return result


def training_data(data: dict[str, Any]) -> dict[str, Any]:
    splits = data["splits"]
    return {
        "data_root": data["data_root"],
        "cache_root": data["cache_root"],
        "splits": {
            "train": list(splits["train"]),
            "validation": list(splits["validation"]),
            "test": list(splits["test"]),
        },
    }


def matrix(mode: str) -> list[tuple[str, str, str, int]]:
    if mode not in MODES:
        raise ValueError("Unknown source-holdout mode")
    runs = RUNS if mode == "formal" else (0,)
    return [(ARM, operator, database, run)
            for operator in OPERATORS for database in EVALUATION for run in runs]


def arm_config(config: dict[str, Any], mode: str, data_signature: str) -> dict[str, Any]:
    validate_config(config)
    if mode not in ("cpu_preflight", "gpu_preflight", "formal"):
        raise ValueError("Training mode required")
    steps = config["training"]["steps"] if mode == "formal" else config["preflight"]["training_steps"]
    return {
        **config["training"],
        "steps": steps,
        "model": {**config["model"], "width": 64},
        "loss": "smooth_l1",
        "adapt_scope": config["arm"]["train_scope"],
        "inner_steps": config["arm"]["train_steps"],
        "purpose": config["experiment"],
        "data_signature": data_signature,
    }


def safe_output(path: Path | str, protected: list[Path | str], root: Path = ROOT) -> Path:
    output = checked_path(path)
    blocked = [checked_path(ROOT / "scripts"), checked_path(ROOT / "configs"),
               checked_path(ROOT / "实验文档"), Path("/gly/weilin/PRICE-master"),
               *(checked_path(item) for item in protected)]
    if output.exists():
        raise FileExistsError("Output must be fresh")
    if any(output.is_relative_to(item) or item.is_relative_to(output) for item in blocked):
        raise ValueError("Output overlaps source/cache/config/docs/PRICE")
    if any(part in (*TARGETS, "PRICE-master", "finetune") for part in output.parts):
        raise ValueError("Target/PRICE output path forbidden")
    runs = checked_path(root) / "runs"
    if not output.is_relative_to(runs):
        raise ValueError("Raw outputs must stay under the isolated checkout runs/")
    return output


def dataset_signature(manifest_hashes: dict[str, str], data: dict[str, Any]) -> str:
    if set(manifest_hashes) != set(ALL_SOURCES):
        raise ValueError("Dataset signature requires all 26 source manifests")
    return digest({
        "schema": 1,
        "train": data["splits"]["train"],
        "validation": data["splits"]["validation"],
        "holdout": data["splits"]["holdout"],
        "manifest_hashes": manifest_hashes,
    })


def build_run_plan(database: str, indices: list[int], hashes: list[str],
                   *, seed: int, run: int) -> dict[str, Any]:
    if database not in EVALUATION or len(indices) != len(hashes) or not indices:
        raise ValueError("Invalid evaluation plan database")
    if len(set(indices)) != len(indices) or len(set(hashes)) != len(hashes):
        raise ValueError("Evaluation identities must be unique")
    if run not in RUNS:
        raise ValueError("Evaluation run outside 0..4")
    support = random.Random(f"{seed}:{database}:{run}").sample(indices, 100)
    excluded = set(support)
    query = [index for index in indices if index not in excluded]
    identity = dict(zip(indices, hashes, strict=True))
    return {
        "database": database,
        "run": run,
        "support_max": support,
        "supports": {str(k): support[:k] for k in SHOTS},
        "query": query,
        "query_digest": digest([(index, identity[index]) for index in query]),
        "support_digests": {
            str(k): digest([(index, identity[index]) for index in support[:k]])
            for k in SHOTS
        },
    }


def certify_run_plan(plan: Any, indices: list[int], hashes: list[str],
                     database: str, run: int) -> dict[str, Any]:
    expected = build_run_plan(
        database, indices, hashes, seed=FROZEN["evaluation"]["support_seed"], run=run
    )
    if plan != expected:
        raise ValueError("Frozen plan differs from deterministic regeneration")
    return {"query": expected["query"], "support": expected["support_max"][:K]}


def certify_plan(plan: Any, identities: dict[str, tuple[list[int], list[str]]]) -> dict[str, Any]:
    if not isinstance(plan, dict) or set(plan) != set(EVALUATION):
        raise ValueError("Plan database closure changed")
    result: dict[str, Any] = {}
    for database in EVALUATION:
        runs = plan[database]
        if not isinstance(runs, list) or len(runs) != len(RUNS):
            raise ValueError("Plan must contain five runs per evaluation database")
        indices, hashes = identities[database]
        result[database] = [
            certify_run_plan(run_plan, indices, hashes, database, run)
            for run, run_plan in enumerate(runs)
        ]
    return result


def plan_summary(plan: dict[str, Any]) -> dict[str, Any]:
    return {
        database: {
            "runs": len(plan[database]),
            "support_max": [len(run["support_max"]) for run in plan[database]],
            "query_counts": [len(run["query"]) for run in plan[database]],
            "query_digests": [run["query_digest"] for run in plan[database]],
        }
        for database in EVALUATION
    }


def metric(q: np.ndarray, min_count: int = 30) -> dict[str, Any]:
    values = np.asarray(q, dtype=np.float64)
    if len(values) == 0:
        return {"count": 0, "status": "empty", "q50": None, "q95": None}
    if not np.isfinite(values).all() or (values < 1).any():
        raise ValueError("Invalid Q-error values")
    return {
        "count": len(values),
        "status": "below_min_n" if len(values) < min_count else "ok",
        "q50": float(np.percentile(values, 50, method="linear")),
        "q95": float(np.percentile(values, 95, method="linear")),
    }


def metric_buckets(log_pred: np.ndarray, true: np.ndarray, tables: np.ndarray,
                   min_count: int = 30) -> dict[str, Any]:
    log_pred = np.asarray(log_pred, dtype=np.float64)
    true = np.asarray(true, dtype=np.float64)
    tables = np.asarray(tables)
    if (log_pred.shape != true.shape or log_pred.ndim != 1 or
            tables.shape != true.shape or not np.isfinite(log_pred).all() or
            not np.isfinite(true).all() or (true < 0).any() or
            tables.dtype.kind not in "iu" or (tables < 1).any()):
        raise ValueError("Invalid metric inputs")
    with np.errstate(over="raise", invalid="raise"):
        q = np.exp(np.abs(np.maximum(log_pred, 0) - np.log(np.maximum(true, 1))))
    masks = {
        "all": np.ones(len(true), dtype=bool),
        "4+": tables >= 4,
        "low4": (tables >= 4) & (true < 100),
    }
    return {bucket: metric(q[mask], min_count) for bucket, mask in masks.items()}


def input_binding(root: Path, config_path: Path, data_config_path: Path,
                  plan_path: Path) -> dict[str, str]:
    root = checked_path(root)
    files = [
        *(root / name for name in CODE_FILES),
        root / PROTOCOL,
        checked_path(config_path),
        checked_path(data_config_path),
        checked_path(plan_path),
    ]
    data = validate_data(read_json(data_config_path), root)
    cache_root = Path(data["cache_root"]) / "pretrain"
    for database in ALL_SOURCES:
        for filename in CACHE_FILES:
            files.append(cache_root / database / filename)
    binding = {str(path): sha256(path) for path in files}
    if binding[str(root / PROTOCOL)] != PROTOCOL_SHA:
        raise ValueError("Protocol SHA differs from frozen contract")
    return binding


def verify_bound_inputs(binding: dict[str, str]) -> None:
    if not isinstance(binding, dict) or not binding:
        raise ValueError("Empty input binding")
    for name, expected in binding.items():
        if not isinstance(name, str) or not isinstance(expected, str) or len(expected) != 64:
            raise ValueError("Malformed input binding")
        if sha256(name) != expected:
            raise ValueError(f"Bound input changed: {name}")


def finite_number(value: Any) -> bool:
    return (not isinstance(value, bool) and isinstance(value, (int, float))
            and math.isfinite(value))
