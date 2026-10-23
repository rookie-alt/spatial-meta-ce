import os
import random
from pathlib import Path

import numpy as np
import torch

from .data import TRANSFORM_VERSION
from .model import SetEstimator


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
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "schema": 1,
        "model_type": "grouped_set_residual",
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
    temp = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, temp)
    os.replace(temp, path)


def load_checkpoint(path, device="cpu"):
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
        or state["model_type"] != "grouped_set_residual"
        or state["transform"] != TRANSFORM_VERSION
    ):
        raise ValueError("Incompatible schema, model, or preprocessing")
    model = SetEstimator(**state["model_config"]).to(device)
    model.load_state_dict(state["model"], strict=True)
    return model, state
