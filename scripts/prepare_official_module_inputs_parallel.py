#!/usr/bin/env python3
"""Prepare official mapping inputs with a bounded CPU process pool.

The serial predecessor remains the reference implementation. This driver
performs the same parent-side preflight, delegates each fixed database pair to
its original ``_database`` function in a spawned worker, and writes the root
certificate only in the parent process.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import sqlglot

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import prepare_official_module_inputs as serial  # noqa: E402
from scripts.audit_input_mapping import write_report  # noqa: E402
from spatial_meta_ce.manifests import file_hash  # noqa: E402
from spatial_meta_ce.official_mappings import (  # noqa: E402
    builder_compatibility,
    plain_path,
)
from spatial_meta_ce.official_split import TEST_DATABASES  # noqa: E402
from spatial_meta_ce.price_capacity import OFFICIAL_TRAIN_DATABASES  # noqa: E402

PARALLEL_SOURCE = "scripts/prepare_official_module_inputs_parallel.py"
PRODUCER_CODE_PATHS = (
    ROOT / "scripts/prepare_official_module_inputs.py",
    ROOT / "spatial_meta_ce/official_mappings.py",
    ROOT / "scripts/audit_input_mapping.py",
    ROOT / "scripts/build_mapping_sidecar.py",
    ROOT / "spatial_meta_ce/data.py",
    ROOT / "spatial_meta_ce/features.py",
    ROOT / "spatial_meta_ce/manifests.py",
    ROOT / "spatial_meta_ce/mappings.py",
    ROOT / "spatial_meta_ce/official_split.py",
    ROOT / "spatial_meta_ce/price_capacity.py",
    ROOT / "vendor/price_features/features_tool.py",
    ROOT / PARALLEL_SOURCE,
)


def _validate_workers(workers) -> int:
    if type(workers) is not int or not 1 <= workers <= 4:
        raise ValueError("workers must be an explicit integer from 1 through 4")
    return workers


def _failed_result(split: str, database: str, exc: Exception) -> dict:
    return {
        "database": database,
        "split": split,
        "state": "failed",
        "error": f"{type(exc).__name__}: {exc}",
        "unsupported_count": 0,
        "dedup_conflict_count": 0,
        "unsupported_count_complete": False,
    }


def _run_database(task: tuple) -> dict:
    (
        split,
        database,
        data,
        paths,
        output,
        legacy_path,
        compatibility,
    ) = task
    target_files = []
    try:
        result = serial._database(
            data,
            split,
            database,
            paths,
            output,
            legacy_path,
            target_files,
            compatibility,
        )
    except Exception as exc:  # noqa: BLE001 - preserve the serial full-population failure record.
        result = _failed_result(split, database, exc)
    return {
        "split": split,
        "database": database,
        "result": result,
        "target_files": target_files,
    }


def prepare_inputs_parallel(
    data_config,
    output,
    source_mapping_root=None,
    include_test=False,
    legacy_builder_source=None,
    workers=4,
):
    """Build the serial official mapping certificate with a bounded process pool.

    ``include_test`` enables structural preprocessing only. False never reads
    test inputs. A supplied source root uses legacy ``<root>/<database>``
    sidecars. The outer supervisor owns wall-clock limits and stopping.
    """
    workers = _validate_workers(workers)
    data, config_binding = serial._configuration(data_config)
    if type(include_test) is not bool:
        raise ValueError("include_test must be an explicit boolean")
    if sqlglot.__version__ != "18.17.0":
        raise ValueError("Official mapping preparation requires sqlglot==18.17.0")
    source_root = (
        plain_path(source_mapping_root) if source_mapping_root is not None else None
    )
    if legacy_builder_source is not None and source_root is None:
        raise ValueError(
            "Historical builder proof requires a legacy source mapping root"
        )
    compatibility = (
        builder_compatibility(legacy_builder_source)
        if legacy_builder_source is not None
        else None
    )
    output = serial._output_path(output, data, source_root)
    pairs = [("pretrain", database) for database in OFFICIAL_TRAIN_DATABASES]
    if include_test:
        pairs += [("test", database) for database in TEST_DATABASES]

    inventory = {
        (split, database): serial._input_paths(data, split, database)
        for split, database in pairs
    }
    legacy = {}
    if source_root is not None:
        for database in OFFICIAL_TRAIN_DATABASES:
            folder = plain_path(source_root / database, source_root)
            for name in ("manifest.json", "mapping.jsonl"):
                path = plain_path(folder / name, source_root)
                if not path.is_file():
                    raise FileNotFoundError(path)
            legacy[database] = folder

    code_hashes = {
        str(path.relative_to(ROOT)): file_hash(path) for path in PRODUCER_CODE_PATHS
    }
    executor_binding = {
        "kind": "cpu-process-pool",
        "workers": workers,
        "source_sha256": code_hashes[PARALLEL_SOURCE],
    }
    report = {
        "schema": 1,
        "scope": serial.OFFICIAL_MAPPING_SCOPE,
        "state": "running",
        **config_binding,
        "mapping_root": str(output),
        "cache_root": data["cache_root"],
        "data_root": data["data_root"],
        "splits": data["splits"],
        "include_test": include_test,
        "source_mapping_root": str(source_root) if source_root is not None else None,
        "databases": {"pretrain": {}, "test": {}},
        "failed_databases": [],
        "unsupported_count": 0,
        "dedup_conflict_count": 0,
        "unsupported_count_complete": True,
        "target_data_read": [database for split, database in pairs if split == "test"],
        "target_files_read": [],
        "model_predictions_run": False,
        "training_run": False,
        "code_sha256": code_hashes,
        "preparation_executor": executor_binding,
        "interpretation": serial.INTERPRETATION,
    }
    output.mkdir(parents=True, exist_ok=False)
    report_path = output / "official_module_inputs.json"
    write_report(report_path, report)

    tasks = [
        (
            split,
            database,
            data,
            inventory[(split, database)],
            output / split / database,
            legacy.get(database) if split == "pretrain" else None,
            compatibility if split == "pretrain" else None,
        )
        for split, database in pairs
    ]
    context = multiprocessing.get_context("spawn")
    executor = ProcessPoolExecutor(max_workers=workers, mp_context=context)
    task_results = [None] * len(tasks)
    try:
        futures = {
            executor.submit(_run_database, task): index
            for index, task in enumerate(tasks)
        }
        for future in as_completed(futures):
            index = futures[future]
            split, database = pairs[index]
            try:
                task_results[index] = future.result()
            except Exception as exc:  # noqa: BLE001 - retain pool failures in the full pair list.
                task_results[index] = {
                    "split": split,
                    "database": database,
                    "result": _failed_result(split, database, exc),
                    "target_files": [],
                }
    finally:
        executor.shutdown(wait=True, cancel_futures=False)

    for task_result in task_results:
        assert task_result is not None
        split = task_result["split"]
        database = task_result["database"]
        result = task_result["result"]
        report["databases"][split][database] = result
        if split == "test":
            report["target_files_read"].extend(task_result["target_files"])
        if result["state"] != "completed":
            report["failed_databases"].append(f"{split}/{database}")
        report["unsupported_count"] += result["unsupported_count"]
        report["dedup_conflict_count"] += result["dedup_conflict_count"]
        report["unsupported_count_complete"] &= result["unsupported_count_complete"]

    if any(
        file_hash(ROOT / name) != expected
        for name, expected in report["code_sha256"].items()
    ):
        report["failed_databases"].append("code_changed_during_preparation")
    report["target_files_read"] = list(dict.fromkeys(report["target_files_read"]))
    report["state"] = "failed" if report["failed_databases"] else "completed"
    write_report(report_path, report)
    if report["failed_databases"]:
        raise serial.InputPreparationError(report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", "--data-config", dest="config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--source-mapping-root")
    parser.add_argument("--legacy-builder-source")
    parser.add_argument(
        "--include-test",
        action="store_true",
        help="Explicit CPU-only structural preparation of all four test databases",
    )
    parser.add_argument(
        "--workers",
        type=int,
        choices=range(1, 5),
        default=4,
        help="Bounded CPU worker count (1-4); the outer supervisor owns time limits",
    )
    args = parser.parse_args()
    report = prepare_inputs_parallel(
        args.config,
        args.output,
        args.source_mapping_root,
        args.include_test,
        args.legacy_builder_source,
        args.workers,
    )
    print(
        json.dumps(
            {
                "state": report["state"],
                "mapping_root": report["mapping_root"],
                "target_data_read": report["target_data_read"],
                "model_predictions_run": False,
                "preparation_executor": report["preparation_executor"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
