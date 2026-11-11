#!/usr/bin/env python3
"""122 P0/A preparation and bounded synthetic CPU smoke; never formal training."""

from __future__ import annotations

import argparse
import json
import os
import resource
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.meta65_joint_supervisor import supervise  # noqa: E402
from spatial_meta_ce.manifests import digest, file_hash  # noqa: E402

PURPOSE = "smoke-not-research"
MAX_UPDATES = 2


def write_json(path: Path, payload: dict) -> None:
    with path.open("x") as stream:
        stream.write(
            json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
        )


def prepare(data_config: dict, plans_dir: Path, output: Path) -> dict:
    """Refuse existing outputs and leave a failure record without hiding errors."""
    output.mkdir(parents=True, exist_ok=False)
    try:
        return _prepare(data_config, plans_dir, output)
    except (OSError, ValueError, KeyError, TypeError) as error:
        path = output / "preflight.json"
        record = {
            "schema": 1,
            "protocol": 122,
            "purpose": "implementation-preflight-not-research",
            "status": "failed",
            "error_type": type(error).__name__,
            "error": str(error),
            "p0_gate_passed": False,
            "formal_training_authorized": False,
            "target_databases_read": [],
        }
        if path.exists():
            path.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n")
        else:
            write_json(path, record)
        raise


def _prepare(data_config: dict, plans_dir: Path, output: Path) -> dict:
    from spatial_meta_ce.price_capacity import (
        build_capacity,
        scan_source_capacity,
        validate_source_splits,
    )
    from spatial_meta_ce.price_protocol import A_ARMS, A_SEEDS, a_model_config
    from spatial_meta_ce.plans import (
        load_training_plan,
        validate_plan_matches_config,
        verify_plan_against_stores,
        load_source_eval_plan,
        verify_source_eval_plan_against_stores,
        bind_source_eval_ids,
    )
    from spatial_meta_ce.data import StorePool
    from spatial_meta_ce.train import _validate_fixed_budget

    validate_source_splits(data_config)
    report = {
        "schema": 1,
        "protocol": 122,
        "purpose": "implementation-preflight-not-research",
        "status": "incomplete_missing_evidence",
        "p0_gate_passed": False,
        "formal_training_authorized": False,
        "target_databases_read": [],
        "missing": [],
    }
    try:
        capacity = build_capacity(data_config)
        coverage = scan_source_capacity(data_config, capacity)
    except FileNotFoundError as error:
        report = {**report, "missing": [str(error.filename)], "unsupported": None}
        write_json(output / "preflight.json", report)
        return report
    report = {**report, "capacity": capacity, "coverage": coverage}
    if coverage["unsupported"]:
        report = {**report, "status": "unsupported_stop_before_a"}
        write_json(output / "preflight.json", report)
        return report
    plan_records = {}
    base = json.loads((ROOT / "configs/fomaml_joint_sum_4k.json").read_text())
    pool = StorePool(
        data_config["cache_root"], max_open=len(data_config["splits"]["train"])
    )
    stores = {db: pool.get("pretrain", db) for db in data_config["splits"]["train"]}
    for seed in A_SEEDS:
        path = Path(plans_dir) / f"training_plan_seed{seed}.json"
        if not path.is_file():
            report = {**report, "missing": report["missing"] + [str(path)]}
            continue
        plan = load_training_plan(path, data_config)
        config = {**base, "seed": seed}
        validate_plan_matches_config(plan, config)
        verify_plan_against_stores(plan, stores)
        plan_records[str(seed)] = {
            "path": str(path.resolve()),
            "file_sha256": file_hash(path),
            "plan_sha256": plan["plan_sha256"],
        }
    source_plan_path = Path(plans_dir) / "source_eval_plan.json"
    if not source_plan_path.is_file():
        report = {**report, "missing": report["missing"] + [str(source_plan_path)]}
    if report["missing"]:
        write_json(output / "preflight.json", report)
        return report
    source_plan = load_source_eval_plan(source_plan_path, data_config)
    source_stores = {
        db: pool.get("pretrain", db) for db in data_config["splits"]["validation"]
    }
    verify_source_eval_plan_against_stores(source_plan, source_stores)
    bound_plan = bind_source_eval_ids(source_plan, source_stores)
    write_json(output / "source_eval_plan_122.json", bound_plan)
    report = {
        **report,
        "source_eval_plan": {
            "path": str(source_plan_path.resolve()),
            "original_file_sha256": file_hash(source_plan_path),
            "original_plan_sha256": source_plan["plan_sha256"],
            "bound_plan_sha256": bound_plan["plan_sha256"],
        },
        "status": "data_preflight_passed",
        "training_plans": plan_records,
        "requires_separate_p0_certification_and_formal_authorization": True,
    }
    write_json(output / "preflight.json", report)
    preflight_hash = file_hash(output / "preflight.json")
    for arm in A_ARMS:
        for seed in A_SEEDS:
            config = {
                **base,
                "model": a_model_config(arm, capacity),
                "method": "erm",
                "meta_objective": "post",
                "seed": seed,
                "fixed_stage": f"122-{arm}",
                "training_plan": plan_records[str(seed)]["path"],
                "source_preflight": str((output / "preflight.json").resolve()),
                "source_preflight_sha256": preflight_hash,
            }
            _validate_fixed_budget(config)
            write_json(output / f"{arm}-seed{seed}.json", config)
    return report


def _synthetic_capacity() -> dict:
    from spatial_meta_ce.data import TRANSFORM_VERSION
    from spatial_meta_ce.price_capacity import TRAIN_DATABASES, validate_capacity

    maxima = [4, 4, 4, 4]
    payload = {
        "schema": 1,
        "kind": "price_train_capacity_122",
        "split": "pretrain",
        "transform": TRANSFORM_VERSION,
        "databases": list(TRAIN_DATABASES),
        "max_counts": maxima,
        "database_max_counts": {db: maxima for db in TRAIN_DATABASES},
        "cache_hashes": {db: digest([PURPOSE, db]) for db in TRAIN_DATABASES},
        "manifest_sha256": {
            db: digest([PURPOSE, "manifest", db]) for db in TRAIN_DATABASES
        },
        "evidence_role": "synthetic_fixture_not_real_training_capacity",
    }
    return validate_capacity({**payload, "capacity_sha256": digest(payload)})


def synthetic_smoke(output: Path, updates: int = MAX_UPDATES) -> dict:
    if type(updates) is not int or not 1 <= updates <= MAX_UPDATES:
        raise ValueError("122 synthetic smoke permits at most 2 updates")
    if os.environ.get("META65_SUPERVISED_MODE") != "cpu_preflight":
        raise ValueError("Synthetic smoke must run through the CPU resource supervisor")
    import torch
    from spatial_meta_ce.data import batch_samples
    from spatial_meta_ce.model import create_model
    from spatial_meta_ce.meta import meta_loss, loss_value
    from spatial_meta_ce.checkpoints import save_checkpoint, load_checkpoint
    from spatial_meta_ce.price_protocol import A_ARMS, a_model_config

    torch.set_num_threads(1)
    torch.manual_seed(42)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "phase.json", {"phase": "preparation"})
    capacity = _synthetic_capacity()

    def sample(n: int, truth: float) -> dict:
        return {
            "groups": [
                torch.rand(n, 40),
                torch.rand(n, 40) * 10,
                torch.rand(n, 4),
                torch.rand(n, 43),
            ],
            "id": f"synthetic/{n}/{truth}",
            "hash": digest([n, truth]),
            "database": "synthetic",
            "true": truth,
            "pg": 2.0,
        }

    support = batch_samples([sample(2, 100), sample(3, 200)])
    query = batch_samples([sample(1, 300), sample(4, 400)])
    measurements = {}
    for arm in A_ARMS:
        model = create_model(a_model_config(arm, capacity)).eval()
        if not torch.equal(model(query), query["pg_log"]):
            raise AssertionError("Zero initialized candidate differs from PG")
        start = time.perf_counter()
        with torch.no_grad():
            for _ in range(10):
                model(query)
        measurements[arm] = {
            "parameters": sum(p.numel() for p in model.parameters()),
            "synthetic_batch2_forward_ms": (time.perf_counter() - start) * 100,
            "timing_not_comparable_to_research_workloads": True,
        }
    model = create_model({**a_model_config("A1", capacity), "lora": True})
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    started = time.perf_counter()
    losses = []
    for _ in range(updates):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        objective = meta_loss(model, support, query, 0.001, 5, scope="lora")
        objective.backward()
        if any(
            p.grad is not None and not torch.isfinite(p.grad).all()
            for p in model.parameters()
        ):
            raise FloatingPointError("Nonfinite smoke outer gradient")
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        losses.append(float(objective.detach()))
    seconds = time.perf_counter() - started
    model.eval()
    checkpoint = output / "synthetic.pt"
    save_checkpoint(
        checkpoint,
        model,
        optimizer,
        updates,
        None,
        0,
        {
            "purpose": PURPOSE,
            "manifest_hashes": capacity["cache_hashes"],
            "evidence_role": "synthetic_fixture",
        },
    )
    clone, _ = load_checkpoint(checkpoint)
    clone.eval()
    if not torch.equal(model(query), clone(query)):
        raise AssertionError("Smoke checkpoint roundtrip prediction mismatch")
    if not torch.isfinite(loss_value(clone(query), query["label"])):
        raise FloatingPointError("Nonfinite smoke checkpoint loss")
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (
        1024 if sys.platform.startswith("linux") else 1
    )
    report = {
        "purpose": PURPOSE,
        "protocol": 122,
        "updates": updates,
        "cpu_threads": 1,
        "device": "cpu",
        "synthetic_only": True,
        "target_databases_read": [],
        "losses": losses,
        "seconds": seconds,
        "peak_rss_bytes": rss,
        "parameters": measurements,
        "lora_inner_parameters": sum(
            p.numel() for n, p in model.named_parameters() if "lora_" in n
        ),
        "checkpoint_roundtrip_equal": True,
        "formal_training_started": False,
        "p0_real_data_gate_passed": False,
        "estimate_4000_synthetic_updates_seconds_not_research_budget": seconds
        / updates
        * 4000,
    }
    write_json(output / "smoke_protocol.json", report)
    (output / "phase.json").write_text(json.dumps({"phase": "certification"}))
    write_json(
        output / "status.json",
        {
            "status": "worker_complete",
            "mode": "cpu_preflight",
            "cells": 0,
            "micro_cells": 1,
        },
    )
    return report


def run_smoke(output: Path, updates: int = MAX_UPDATES) -> dict:
    if type(updates) is not int or not 1 <= updates <= MAX_UPDATES:
        raise ValueError("122 synthetic smoke permits at most 2 updates")
    command = [
        sys.executable,
        "-B",
        str(Path(__file__).resolve()),
        "smoke-worker",
        "--output",
        str(output.resolve()),
        "--updates",
        str(updates),
    ]
    return supervise(
        command,
        output,
        mode="cpu_preflight",
        root=ROOT,
        preflight_micro_cells=1,
        grace=1,
        budget={
            "total_seconds": 300,
            "phase_seconds": {"preparation": 290, "evaluation": 5, "certification": 5},
            "rss_bytes": 4 * 1024**3,
            "vram_bytes": 0,
            "output_bytes": 128 * 1024**2,
            "poll_seconds": 0.2,
        },
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "smoke", "smoke-worker"])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=ROOT / "configs/data.json")
    parser.add_argument("--plans", type=Path)
    parser.add_argument("--updates", type=int, default=MAX_UPDATES)
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(ROOT):
        raise ValueError("122 preparation/smoke outputs must stay inside this worktree")
    if args.command == "prepare":
        if args.plans is None:
            parser.error("prepare requires --plans (read-only sealed plans)")
        from spatial_meta_ce.cli import data_settings

        data = data_settings(json.loads(args.data.read_text()), args.data)
        report = prepare(data, args.plans, output)
        code = 0 if report["status"] == "data_preflight_passed" else 2
    elif args.command == "smoke":
        report = run_smoke(output, args.updates)
        code = 0 if report["status"] == "preflight_completed" else 1
    else:
        report = synthetic_smoke(output, args.updates)
        code = 0
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
