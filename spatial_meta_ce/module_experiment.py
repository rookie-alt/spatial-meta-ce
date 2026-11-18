"""Independent doc-132 official26/4 contract; imports no models on load.

Shape validation permits unsigned preparation. Real training always requires
fresh hash-bound execution evidence, never a local allow-training flag.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import sys
from pathlib import Path

STAGE = "132-module26"
PROTOCOL = "official-module-26-4-v1"
ARMS = (
    "A1-PG",
    "A1-PG-GroupMean",
    "A1-PG-GroupAttention",
    "A1-PG-RelationTrue",
    "A1-PG-RelationShuffled",
)
SEEDS = (42, 43, 44)
STEPS = 24000
FIXED_SELECTION = "none (fixed 24000 updates, last.pt only)"
TRAIN_DATABASES = (
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
    "chembl",
    "legalacts",
    "telstra",
)
TEST_DATABASES = ("imdb", "stats", "genome", "ergastf1")
MODEL_TYPES = {
    "A1-PG": "price_lite_pg_residual",
    "A1-PG-GroupMean": "price_lite_pg_group_mean_residual",
    "A1-PG-GroupAttention": "price_lite_pg_group_attention_residual",
    "A1-PG-RelationTrue": "price_lite_pg_official_relation_residual",
    "A1-PG-RelationShuffled": "price_lite_pg_official_relation_residual",
}
RECIPE = {
    "purpose": "research",
    "fixed_budget": True,
    "fixed_stage": STAGE,
    "steps": STEPS,
    "method": "erm",
    "meta_objective": "post",
    "meta_batch": 4,
    "accumulation": 1,
    "support_sizes": [5, 10, 20, 50, 100],
    "query_size": 64,
    "inner_lr": 1e-3,
    "inner_steps": 5,
    "adapt_scope": "all",
    "outer_lr": 3e-4,
    "optimizer": "adam",
    "gradient_clip": 1.0,
    "loss": "smooth_l1",
    "validate_every": STEPS,
    "validation_runs": 1,
    "validation_seed": 2026,
    "patience": 1,
    "min_delta": 0.0,
}


def canonical_digest(value) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        ).encode()
    ).hexdigest()


def exact_equal(left, right) -> bool:
    """Reject bool/int, float/int and list/tuple aliases JSON hashes alone miss."""
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(
            exact_equal(left[k], right[k]) for k in left
        )
    if isinstance(left, (list, tuple)):
        return len(left) == len(right) and all(
            exact_equal(a, b) for a, b in zip(left, right)
        )
    return left == right


def is_sha256(value) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(c in "0123456789abcdef" for c in value)
    )


def regular_path(value, *, exists=True) -> Path:
    if not isinstance(value, (str, Path)):
        raise ValueError("A bound path must be an absolute path")
    path = Path(value)
    if not path.is_absolute() or ".." in path.parts:
        raise ValueError("A bound path must be absolute and cannot contain ..")
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError(f"Bound path traverses a symlink: {path}")
    if exists and not path.is_file():
        raise ValueError(f"Bound input is not a regular file: {path}")
    return path


def binding_shape(record, *, plan=False) -> dict:
    fields = {"path", "sha256"} | ({"plan_sha256"} if plan else set())
    if not isinstance(record, dict) or set(record) != fields:
        raise ValueError("Invalid file binding schema")
    regular_path(record["path"], exists=False)
    if any(not is_sha256(record[k]) for k in fields - {"path"}):
        raise ValueError("Invalid binding SHA256")
    return record


def validate_splits(data_config: dict) -> None:
    expected = {
        "train": list(TRAIN_DATABASES),
        "validation": [],
        "test": list(TEST_DATABASES),
    }
    if not isinstance(data_config, dict) or not exact_equal(
        data_config.get("splits"), expected
    ):
        raise ValueError(
            "Module experiment requires exact official26/4 lists and validation=[]"
        )


def mapping_mode(arm: str):
    if arm not in ARMS:
        raise ValueError("Unknown official module arm")
    return {
        "A1-PG-RelationTrue": "mapping",
        "A1-PG-RelationShuffled": "shuffled_mapping",
    }.get(arm)


def model_config(arm: str, capacity: dict) -> dict:
    from .official_split import validate_official_capacity
    from .price_capacity import validate_capacity
    from .price_protocol import exploratory_model_config

    mapping_mode(arm)
    validate_official_capacity(capacity)
    validate_capacity(capacity)
    result = exploratory_model_config(capacity)
    if arm in ARMS[1:3]:
        result["group_pooling"] = "mean" if arm == ARMS[1] else "attention"
    elif mapping_mode(arm):
        result.update(relation_attention=True, relation_scope="official_26_4")
    return result


def run_key(arm: str, seed: int) -> str:
    mapping_mode(arm)
    if type(seed) is not int or seed not in SEEDS:
        raise ValueError("Official module seed must be one of 42/43/44")
    return f"{arm}-seed{seed}"


def training_config(
    arm: str,
    seed: int,
    capacity: dict,
    *,
    data_config_binding: dict,
    training_plan_binding: dict,
    mapping_root=None,
    module_execution=None,
) -> dict:
    """Canonical config builder; the returned unsigned config cannot train."""
    run_key(arm, seed)
    binding_shape(data_config_binding)
    binding_shape(training_plan_binding, plan=True)
    result = {
        **copy.deepcopy(RECIPE),
        "module_arm": arm,
        "seed": seed,
        "model": model_config(arm, capacity),
        "data_config": data_config_binding["path"],
        "data_config_sha256": data_config_binding["sha256"],
        "training_plan": training_plan_binding["path"],
        "training_plan_file_sha256": training_plan_binding["sha256"],
        "training_plan_sha256": training_plan_binding["plan_sha256"],
    }
    if mapping_mode(arm):
        if mapping_root is None:
            raise ValueError("Relation arm requires a certified mapping root")
        regular_path(mapping_root, exists=False)
        result["mapping"] = {
            "root": str(mapping_root),
            "mode": mapping_mode(arm),
            "scope": "official_26_4",
            "split": "pretrain",
        }
    elif mapping_root is not None:
        raise ValueError("Only a relation arm accepts a mapping root")
    if module_execution is not None:
        result["module_execution"] = copy.deepcopy(module_execution)
    return result


def validate_config_shape(config: dict) -> dict:
    if not isinstance(config, dict):
        raise ValueError("Module training configuration must be an object")
    try:
        expected = training_config(
            config["module_arm"],
            config["seed"],
            config["model"]["capacity"],
            data_config_binding={
                "path": config["data_config"],
                "sha256": config["data_config_sha256"],
            },
            training_plan_binding={
                "path": config["training_plan"],
                "sha256": config["training_plan_file_sha256"],
                "plan_sha256": config["training_plan_sha256"],
            },
            mapping_root=config.get("mapping", {}).get("root"),
            module_execution=config.get("module_execution"),
        )
    except (KeyError, TypeError, AttributeError) as exc:
        raise ValueError("Module training configuration is incomplete") from exc
    if not exact_equal(config, expected) or canonical_digest(
        config
    ) != canonical_digest(expected):
        raise ValueError(
            "Module training configuration differs from its exact frozen recipe"
        )
    return config


def validate_training_config(
    config: dict, data_config: dict, resume=None, *, output_dir=None, device=None
) -> dict:
    """Call at train() entry, before a cache/model/output is opened or created."""
    if resume is not None:
        raise ValueError(
            "Official module training forbids resume and initialization checkpoints"
        )
    validate_config_shape(config)
    validate_splits(data_config)
    evidence = config.get("module_execution")
    if not isinstance(evidence, dict) or set(evidence) != {
        "spec",
        "authorization",
        "output_dir",
        "run_key",
    }:
        raise ValueError(
            "Official module training requires bound module_execution evidence"
        )
    from scripts.run_official_module_experiment import (
        read_bound,
        validate_authorization,
        verify_spec,
    )

    binding_shape(evidence["spec"])
    binding_shape(evidence["authorization"])
    spec = read_bound(evidence["spec"])
    if spec.get("action") != "train":
        raise ValueError("module_execution requires a training specification")
    authorization = read_bound(evidence["authorization"])
    validate_authorization(authorization, spec, evidence["spec"]["sha256"])
    validate_runtime(spec, device)
    if evidence["output_dir"] != spec["output_dir"]:
        raise ValueError("module_execution output differs from signed specification")
    resolved = verify_spec(spec)
    key = run_key(config["module_arm"], config["seed"])
    if evidence["run_key"] != key or key not in resolved["configs"]:
        raise ValueError("module_execution arm/seed is outside the signed scope")
    if (
        output_dir is None
        or str(regular_path(output_dir, exists=False)) != spec["outputs"][key]
    ):
        raise ValueError(
            "Runtime training output_dir differs from its signed run scope"
        )
    expected = {**resolved["configs"][key], "module_execution": evidence}
    if (
        not exact_equal(config, expected)
        or canonical_digest(config) != canonical_digest(expected)
        or not exact_equal(data_config, resolved["data"])
    ):
        raise ValueError("Runtime config/data differ from their signed bindings")
    return resolved


def validate_runtime(spec: dict, device) -> None:
    """Close the public train API over the signed device and owned supervisor."""
    if sys.platform != "linux":
        raise ValueError("Official module training requires server Linux")
    if not isinstance(device, str) or device != spec.get("device"):
        raise ValueError("Runtime training device differs from signed device")
    visibility = str(spec["cuda_device"]) if device == "cuda" else ""
    if os.environ.get("CUDA_VISIBLE_DEVICES") != visibility:
        raise ValueError("Runtime CUDA visibility differs from signed device")
    if os.environ.get("META65_SUPERVISED_MODE") != "formal" or os.environ.get(
        "META65_SUPERVISOR_PID"
    ) != str(os.getppid()):
        raise ValueError(
            "Official module training requires its own formal supervisor parent"
        )


def checkpoint_metadata(config: dict) -> dict:
    """Merge into train metadata after the normal cache/plan verification."""
    validate_config_shape(config)
    return {
        "protocol": PROTOCOL,
        "fixed_stage": STAGE,
        "module_arm": config["module_arm"],
        "method": "erm",
        "meta_objective": "post",
        "seed": config["seed"],
        "purpose": "research",
        "fixed_budget": True,
        "fixed_steps": STEPS,
        "selection": FIXED_SELECTION,
        "train_databases": list(TRAIN_DATABASES),
        "validation_databases": [],
        "selection_databases": [],
        "validation_monitor_reads": [],
        "target_databases_read": [],
        "adapted": False,
        "manifest_hashes": copy.deepcopy(config["model"]["capacity"]["cache_hashes"]),
        "training_plan": config["training_plan"],
        "training_plan_sha256": config["training_plan_sha256"],
        "training_signature": copy.deepcopy(config),
        "mapping_mode": mapping_mode(config["module_arm"]),
        "mapping_scope": "official_26_4"
        if mapping_mode(config["module_arm"])
        else None,
    }


def validate_checkpoint(
    state: dict, arm: str, seed: int, *, expected_config=None
) -> dict:
    """Validate raw trusted checkpoint metadata before constructing a model."""
    from .data import TRANSFORM_VERSION

    run_key(arm, seed)
    if not isinstance(state, dict) or state.get("transform") != TRANSFORM_VERSION:
        raise ValueError("Official module checkpoint preprocessing transform differs")
    metadata = state.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("Official module checkpoint lacks metadata")
    config = metadata.get("training_signature")
    validate_config_shape(config)
    expected_metadata = checkpoint_metadata(config)
    if config["module_arm"] != arm or config["seed"] != seed:
        raise ValueError("Official module checkpoint arm/seed differs")
    for field, expected in expected_metadata.items():
        if not exact_equal(metadata.get(field), expected):
            raise ValueError(f"Official module checkpoint {field} differs")
    model = model_config(arm, state.get("model_config", {}).get("capacity"))
    if (
        type(state.get("schema")) is not int
        or state["schema"] != 1
        or type(state.get("next_step")) is not int
        or state["next_step"] != STEPS
        or state.get("model_type") != MODEL_TYPES[arm]
        or not exact_equal(state.get("model_config"), model)
        or not exact_equal(model, config["model"])
        or state.get("model_config_sha256") != canonical_digest(model)
    ):
        raise ValueError(
            "Official module checkpoint model or 24000-update endpoint differs"
        )
    extra = state.get("extra", {})
    if (
        type(extra.get("counts", {}).get("optimizer_steps")) is not int
        or extra["counts"]["optimizer_steps"] != STEPS
        or type(extra.get("sample_plan_episodes")) is not int
        or extra["sample_plan_episodes"] != STEPS * 4
        or extra.get("sample_plan_digest") != config["training_plan_sha256"]
    ):
        raise ValueError("Official module checkpoint step/episode provenance differs")
    if any(
        metadata.get(k)
        for k in ("adaptation", "adapted_from", "support_ids", "target_support")
    ):
        raise ValueError(
            "Official module checkpoint must be an unadapted initialization"
        )
    if expected_config is not None:
        if "module_execution" in expected_config and not exact_equal(
            config.get("module_execution"), expected_config["module_execution"]
        ):
            raise ValueError("Official module checkpoint execution evidence differs")
        left = {k: v for k, v in config.items() if k != "module_execution"}
        right = {k: v for k, v in expected_config.items() if k != "module_execution"}
        if not exact_equal(left, right) or canonical_digest(left) != canonical_digest(
            right
        ):
            raise ValueError(
                "Official module checkpoint differs from the frozen training specification"
            )
    return metadata
