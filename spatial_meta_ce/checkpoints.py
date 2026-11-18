import os
import random
from pathlib import Path

import numpy as np
import torch

from .data import TRANSFORM_VERSION
from .manifests import digest
from .model import create_model


def rng_state():
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])


def save_checkpoint(
    path,
    model,
    optimizer,
    next_step,
    best_metric,
    bad_validations,
    metadata,
    extra=None,
):
    from .price_model import PriceResidualEstimator

    if isinstance(model, PriceResidualEstimator) and hasattr(model, "_pooling_config"):
        model.validate_pooling_identity()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "schema": 1,
        "model_type": model.model_type,
        "transform": TRANSFORM_VERSION,
        "model_config": model.config,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer else None,
        "next_step": next_step,
        "best_metric": best_metric,
        "bad_validations": bad_validations,
        "metadata": metadata,
        "rng": rng_state(),
        "extra": extra or {},
    }
    if model.model_type != "grouped_set_residual":
        state = {**state, "model_config_sha256": digest(model.config)}
        _validate_candidate_binding(model, state)
    temp = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, temp)
    os.replace(temp, path)


def _validate_candidate_binding(model, state):
    from .price_model import GROUP_POOLING_MODEL_TYPES

    if model.model_type == "grouped_set_residual":
        return
    if state.get("model_config_sha256") != digest(state["model_config"]):
        raise ValueError("Checkpoint model configuration hash mismatch")
    if model.config != state["model_config"]:
        raise ValueError("Checkpoint candidate config is not canonical")
    if (
        model.model_type in GROUP_POOLING_MODEL_TYPES
        and digest(model.config) != state["model_config_sha256"]
    ):
        raise ValueError("Checkpoint group pooling config is not canonical")
    if hasattr(model, "capacity"):
        from .price_capacity import validate_capacity

        if not isinstance(state["metadata"], dict):
            raise ValueError("PRICE checkpoint metadata must be a dictionary")
        hashes = state["metadata"].get("manifest_hashes")
        if hashes is None:
            raise ValueError("PRICE checkpoint is missing training cache hashes")
        validate_capacity(model.capacity, hashes)


def load_checkpoint(path, device="cpu"):
    from .price_model import GROUP_POOLING_MODEL_TYPES

    # Only use trusted locally-produced checkpoints: torch pickle can execute code.
    state = torch.load(path, map_location="cpu", weights_only=False)
    required = {
        "schema",
        "model_type",
        "transform",
        "model_config",
        "model",
        "metadata",
        "rng",
        "next_step",
        "best_metric",
        "bad_validations",
        "optimizer",
        "extra",
    }
    if not isinstance(state, dict) or not required <= state.keys():
        raise ValueError("Not a versioned spatial-meta-ce checkpoint")
    if (
        state["schema"] != 1
        or state["model_type"] not in (
            "grouped_set_residual", "sum_anchor_only_residual",
            "price_lite_residual", "price_lite_pg_residual", "price_tiny_residual",
            *GROUP_POOLING_MODEL_TYPES,
        )
        or state["transform"] != TRANSFORM_VERSION
    ):
        raise ValueError("Incompatible schema, model, or preprocessing")
    model = create_model(state["model_config"], state["model_type"]).to(device)
    _validate_candidate_binding(model, state)
    model.load_state_dict(state["model"], strict=True)
    return model, state
