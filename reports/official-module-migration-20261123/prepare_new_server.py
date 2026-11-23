"""Prepare fifteen unsigned single-run specs on a migrated server.

This helper performs preparation only. It certifies the migrated inputs
once through the frozen runner, derives one spec and one unsigned
authorization template per arm and seed, and writes a readiness record.
It never trains, never predicts, never signs and never executes.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

DEFAULT_BASE = Path("/gly/weilin/module26-prep-20261118")
DEFAULT_SPLIT = Path("/gly/weilin/official-split-20261117/plans-24k")
DEFAULT_BUDGET = DEFAULT_BASE / "evidence/training-budget-draft.json"
DEFAULT_MAPPING = DEFAULT_BASE / "inputs-parallel-v3/official_module_inputs.json"


def _load_runner(code_dir):
    """Import the frozen runner from the migrated code directory."""
    code_dir = Path(code_dir)
    if (
        not code_dir.is_absolute()
        or ".." in code_dir.parts
        or any(path.is_symlink() for path in (code_dir, *code_dir.parents))
    ):
        raise ValueError("Code directory must be an absolute path without symlinks")
    if not code_dir.is_dir():
        raise ValueError(f"Code directory does not exist: {code_dir}")
    resolved = str(code_dir.resolve())
    if resolved not in sys.path:
        sys.path.insert(0, resolved)
    from scripts import run_official_module_experiment as runner

    if runner.ROOT != code_dir:
        raise ValueError("Imported runner differs from the requested code directory")
    return runner


def _reject_overlap(path, roots, label):
    """Refuse a path that overlaps any read-only input root."""
    path = Path(path).absolute()
    for root in roots:
        root = Path(root).absolute()
        if path == root or path.is_relative_to(root) or root.is_relative_to(path):
            raise ValueError(f"{label} path overlaps a read-only input root: {path}")
    return path


def _refuse_existing(path, label):
    path = Path(path)
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"{label} already exists: {path}")
    return path


def prepare(
    *,
    out,
    output_root,
    code_dir,
    data_config=None,
    capacity=None,
    split_manifest=None,
    mapping_report=None,
    budget_path=None,
    cuda_device=1,
):
    """Certify the migrated inputs once and write fifteen unsigned specs."""
    runner = _load_runner(code_dir)
    data_config = Path(data_config or DEFAULT_SPLIT / "data_config_official_26.json")
    capacity = Path(capacity or DEFAULT_SPLIT / "capacity_official_26.json")
    split_manifest = Path(
        split_manifest or DEFAULT_SPLIT / "official_split_manifest.json"
    )
    mapping_report = Path(mapping_report or DEFAULT_MAPPING)
    budget_path = Path(budget_path or DEFAULT_BUDGET)

    data = runner.read_bound(runner.file_binding(data_config))
    report = runner.read_bound(runner.file_binding(mapping_report))
    read_only_roots = [
        value
        for value in (
            data.get("data_root"),
            data.get("cache_root"),
            report.get("mapping_root"),
            report.get("source_mapping_root"),
        )
        if value
    ]
    read_only_roots.extend(
        [str(data_config.parent), str(mapping_report.parent), str(capacity.parent)]
    )

    out = _refuse_existing(runner.regular_path(out, exists=False), "Preparation output")
    output_root = _refuse_existing(
        runner.regular_path(output_root, exists=False), "Run output root"
    )
    read_only_roots.append(str(Path(code_dir)))
    _reject_overlap(out, read_only_roots, "Preparation output")
    _reject_overlap(output_root, read_only_roots, "Run output root")
    if out.is_relative_to(output_root) or output_root.is_relative_to(out):
        raise ValueError("Preparation output and run output root must not overlap")

    expected_code = json.loads(
        (Path(__file__).resolve().parent / "required-code-sha256.json").read_text()
    )["code_sha256"]
    runner.verify_code_tree(Path(code_dir), expected_code)

    budget = json.loads(runner.regular_path(budget_path).read_text())
    runner.checked_budget(budget)
    manifest = runner.read_bound(runner.file_binding(split_manifest))
    plans = {}
    for seed in runner.SEEDS:
        entry = manifest.get("plans", {}).get(str(seed))
        if not isinstance(entry, dict) or not entry.get("path"):
            raise ValueError(f"Split manifest lacks the sealed plan for seed {seed}")
        plans[str(seed)] = entry["path"]
    started = time.monotonic()
    master = runner.prepare_training(
        data_config=data_config,
        capacity=capacity,
        split_manifest=split_manifest,
        training_plans=plans,
        mapping_report=mapping_report,
        output_dir=output_root / "matrix-preparation-only",
        budget=budget,
        cuda_device=cuda_device,
    )

    out.mkdir(parents=True)
    records = []
    for arm in runner.ARMS:
        for seed in runner.SEEDS:
            key = runner.run_key(arm, seed)
            header = runner.spec_header(
                "train",
                output_root / key,
                budget,
                cuda_device=cuda_device,
                execution={"mode": "single_run", "arm": arm, "seed": seed},
            )
            spec = {
                **header,
                "bindings": copy.deepcopy(master["bindings"]),
                "config_sha256": copy.deepcopy(master["config_sha256"]),
            }
            runner.validate_header(spec)
            if spec["bindings"] != master["bindings"]:
                raise AssertionError("Common verified input bindings changed")
            folder = out / key
            folder.mkdir()
            spec_path = folder / "training_spec.json"
            auth_path = folder / "authorization_template.json"
            runner.write_json(spec_path, spec)
            binding = runner.file_binding(spec_path)
            runner.write_json(
                auth_path, runner.authorization_template(spec, binding["sha256"])
            )
            auth_binding = runner.file_binding(auth_path)
            args = SimpleNamespace(
                spec=str(spec_path),
                spec_sha256=binding["sha256"],
                authorization=str(auth_path),
                authorization_sha256=auth_binding["sha256"],
                execute=True,
            )
            try:
                runner.parent_main(args)
            except ValueError as exc:
                if "unsigned" not in str(exc):
                    raise
                rejected = str(exc)
            else:
                raise AssertionError("Unsigned authorization must reject execution")
            if Path(spec["output_dir"]).exists():
                raise AssertionError("Rejected execution created output")
            records.append(
                {
                    "key": key,
                    "spec": binding,
                    "authorization_template": auth_binding,
                    "output_dir": spec["output_dir"],
                    "unsigned_execute_rejected": rejected,
                }
            )

    result = {
        "prepared": True,
        "training_run": False,
        "model_predictions_run": False,
        "authorized": False,
        "target_databases_read_for_training": [],
        "runs": records,
        "common_bindings": master["bindings"],
        "common_config_sha256": master["config_sha256"],
        "budget_per_run": budget,
        "code_dir": str(Path(code_dir).resolve()),
        "output_root": str(output_root),
        "cuda_device": cuda_device,
        "seconds": time.monotonic() - started,
        "validation": (
            "Migrated inputs certified once through the frozen runner; each "
            "single-run header checked against the common bindings; all 15 "
            "unsigned execution attempts rejected before output creation."
        ),
    }
    runner.write_json(out / "readiness.json", result)
    print(json.dumps({"prepared": True, "runs": len(records), "authorized": False}))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, help="New preparation directory")
    parser.add_argument("--output-root", required=True, help="New run output root")
    parser.add_argument("--code-dir", required=True, help="Migrated training code")
    parser.add_argument("--data-config")
    parser.add_argument("--capacity")
    parser.add_argument("--split-manifest")
    parser.add_argument("--mapping-report")
    parser.add_argument("--budget")
    parser.add_argument("--cuda-device", type=int, default=1)
    args = parser.parse_args(argv)
    prepare(
        out=args.out,
        output_root=args.output_root,
        code_dir=args.code_dir,
        data_config=args.data_config,
        capacity=args.capacity,
        split_manifest=args.split_manifest,
        mapping_report=args.mapping_report,
        budget_path=args.budget,
        cuda_device=args.cuda_device,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
