"""Frozen doc 122 A contracts, separate from historical 112 defaults."""

from __future__ import annotations

import copy

import numpy as np

from .plans import validate_source_eval_plan
from .price_capacity import validate_capacity, validate_source_splits

A_ARMS = ("A0-ref", "A0", "A1", "A2")
A_SEEDS = (42, 43, 44)
SUM_MODEL = {"width": 64, "use_pg": True, "arch": "sum", "layers": 2, "heads": 4}


def a_model_config(arm: str, capacity: dict | None = None) -> dict:
    if arm not in A_ARMS:
        raise ValueError("Unknown 122 A arm")
    if arm in ("A0-ref", "A0"):
        return {**SUM_MODEL, "arch": "sum" if arm == "A0-ref" else "sum_anchor_only"}
    validate_capacity(capacity)
    return {
        "arch": "price_lite" if arm == "A1" else "price_tiny",
        "d_model": 64 if arm == "A1" else 32,
        "layers": 2,
        "heads": 4,
        "dropout": 0.0,
        "capacity": copy.deepcopy(capacity),
        "lora": False,
        "lora_seed": 42,
    }


def fixed_model_scope(
    config: dict, default_model: dict, default_scope: str
) -> tuple[dict, str]:
    stage = config.get("fixed_stage")
    if stage is None:
        return default_model, default_scope
    if stage not in tuple(f"122-{arm}" for arm in A_ARMS):
        raise ValueError(
            "Unsupported fixed_stage; only 122 A infrastructure is implemented"
        )
    if config["method"] != "erm" or config["meta_objective"] != "post":
        raise ValueError(
            "122 A requires ERM with no training inner loop (post signature)"
        )
    arm = stage.removeprefix("122-")
    expected = a_model_config(arm, config["model"].get("capacity"))
    return expected, "all"


def validate_a_plan(plan: dict, data_config: dict) -> dict:
    validate_source_splits(data_config)
    if (
        plan.get("protocol") != 122
        or plan.get("analysis_role") != "population_primary"
        or "plan_seed" in plan
        or "shot_seed" in plan
    ):
        raise ValueError(
            "122 requires population_primary; historical plan_seed is diagnostic only"
        )
    validate_source_eval_plan(plan, data_config)
    for entry in plan["databases"].values():
        if (
            entry.get("excluded_unique") != 100
            or len(entry.get("query_ids", [])) != entry["query_count"]
        ):
            raise ValueError("122 population plan lacks IDs/exclusion denominator")
        for block in entry["support_blocks"]:
            if len(block.get("support_ids", [])) != 20:
                raise ValueError("122 support block lacks complete IDs")
    return plan


def validate_training_preflight(config: dict, data_config: dict) -> dict:
    """Training consumes source-only preflight evidence, never validation stores."""
    import json
    from pathlib import Path
    from .manifests import file_hash
    from .price_capacity import build_capacity

    path = config.get("source_preflight")
    if not path or file_hash(path) != config.get("source_preflight_sha256"):
        raise ValueError("122 training requires a hash-bound source_preflight")
    report = json.loads(Path(path).read_text())
    if (
        report.get("protocol") != 122
        or report.get("status") != "data_preflight_passed"
        or report.get("coverage", {}).get("full_population_supported") is not True
        or report.get("coverage", {}).get("unsupported") != []
        or report.get("target_databases_read") != []
    ):
        raise ValueError("122 source preflight did not pass capacity coverage")
    if report.get("p0_gate_passed") is not True:
        raise ValueError(
            "122 P0 gate is incomplete; data preflight alone cannot authorize A"
        )
    actual = build_capacity(data_config)
    if report.get("capacity") != actual:
        raise ValueError("122 training capacity/cache binding changed")
    if config["model"].get("capacity", actual) != actual:
        raise ValueError("122 model capacity differs from source preflight")
    record = report.get("training_plans", {}).get(str(config["seed"]), {})
    if record.get("file_sha256") != file_hash(config["training_plan"]):
        raise ValueError("122 sealed training plan changed since preflight")
    return report


def validate_a_checkpoint(state: dict, arm: str, seed: int) -> dict:
    from .train import FIXED_SELECTION, FIXED_STEPS, _validate_fixed_budget
    from .price_capacity import TRAIN_DATABASES

    if arm not in A_ARMS or seed not in A_SEEDS:
        raise ValueError("Unknown 122 A arm/seed")
    metadata = state.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("122 A checkpoint lacks metadata")
    signature = metadata.get("training_signature")
    required = {
        "method",
        "meta_batch",
        "accumulation",
        "query_size",
        "support_sizes",
        "training_plan",
        "loss",
        "inner_lr",
        "inner_steps",
        "adapt_scope",
        "outer_lr",
        "model",
        "seed",
        "fixed_stage",
    }
    if not isinstance(signature, dict) or not required <= signature.keys():
        raise ValueError("122 A checkpoint lacks a complete training_signature")
    if not isinstance(signature["model"], dict):
        raise ValueError("122 A checkpoint model config must be a dictionary")
    _validate_fixed_budget(
        {
            **signature,
            "steps": state.get("next_step"),
            "meta_objective": signature.get("meta_objective", "post"),
        }
    )
    expected = a_model_config(arm, state.get("model_config", {}).get("capacity"))
    if state.get("model_config") != expected or signature.get("model") != expected:
        raise ValueError("122 A checkpoint model configuration mismatch")
    if (
        metadata.get("fixed_stage") != f"122-{arm}"
        or signature.get("fixed_stage") != f"122-{arm}"
    ):
        raise ValueError("122 A checkpoint stage mismatch")
    if (
        state.get("next_step") != FIXED_STEPS
        or metadata.get("fixed_steps") != FIXED_STEPS
        or metadata.get("selection") != FIXED_SELECTION
        or metadata.get("fixed_budget") is not True
    ):
        raise ValueError("122 A requires the fixed 4000-update last.pt checkpoint")
    if (
        metadata.get("method") != "erm"
        or metadata.get("seed") != seed
        or signature.get("seed") != seed
        or metadata.get("purpose") != "research"
    ):
        raise ValueError("122 A checkpoint method/seed/purpose mismatch")
    if metadata.get("train_databases") != list(TRAIN_DATABASES):
        raise ValueError("122 A checkpoint does not bind the frozen training databases")
    for field in (
        "validation_databases",
        "selection_databases",
        "validation_monitor_reads",
    ):
        if metadata.get(field) != []:
            raise ValueError(f"122 A checkpoint {field} must be empty")
    plan_hash = metadata.get("training_plan_sha256")
    if (
        not isinstance(plan_hash, str)
        or len(plan_hash) != 64
        or any(c not in "0123456789abcdef" for c in plan_hash)
        or metadata.get("adapted_database")
    ):
        raise ValueError("122 A requires an independent sealed training initialization")
    return metadata


def summarize_a(units: list[dict]) -> dict:
    """A gate uses only K0; K5/20 must never select an architecture."""
    from .price_capacity import SOURCE_DATABASES

    keyed = {}
    plan_hashes = {row.get("source_eval_plan_sha256") for row in units}
    if len(plan_hashes) != 1 or not all(
        isinstance(h, str) and len(h) == 64 for h in plan_hashes
    ):
        raise ValueError("A summary requires one shared sealed population plan hash")
    for row in units:
        if row["k"] != 0:
            continue
        key = (row["arm"], row["seed"], row["database"])
        if key in keyed or row["run"] != 0:
            raise ValueError("A summary contains duplicate K0 units")
        value = row["metrics"]["log_q95"]
        if not np.isfinite(value):
            raise ValueError("Nonfinite A log-Q95")
        keyed[key] = float(value)
    expected = {(a, s, d) for a in A_ARMS for s in A_SEEDS for d in SOURCE_DATABASES}
    if set(keyed) != expected:
        raise ValueError(
            "A summary requires the complete 4-arm/3-seed/3-database K0 matrix"
        )
    comparisons = {}
    for arm, reference in (("A1", "A0"), ("A2", "A0"), ("A1", "A0-ref")):
        per_db = {
            d: [keyed[arm, s, d] - keyed[reference, s, d] for s in A_SEEDS]
            for d in SOURCE_DATABASES
        }
        deltas = np.asarray(
            [np.mean([v[i] for v in per_db.values()]) for i in range(3)]
        )
        mean, sd = float(deltas.mean()), float(deltas.std(ddof=1))
        protected = all(float(np.mean(v)) <= 0.1 for v in per_db.values())
        direction = (
            mean < 0 and abs(mean) >= sd and (sd != 0 or bool((deltas < 0).all()))
        )
        comparisons[f"{arm}-{reference}"] = {
            "mean_delta": mean,
            "sd_delta": sd,
            "per_seed_delta": deltas.tolist(),
            "per_database_delta": per_db,
            "database_guard_passed": protected,
            "passed": bool(direction and protected),
        }
    passed = all(comparisons[key]["passed"] for key in ("A1-A0", "A2-A0"))
    return {
        "status": "a_passed_source_screen" if passed else "a_no_candidate",
        "comparisons": comparisons,
        "source_development_screen_not_significance": True,
        "capacity_effect_not_identified": True,
    }
