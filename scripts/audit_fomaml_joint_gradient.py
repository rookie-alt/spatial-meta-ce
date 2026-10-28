#!/usr/bin/env python3
"""Recompute the deterministic step-1 gradient audit for a fixed-budget run."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from spatial_meta_ce.data import StorePool  # noqa: E402
from spatial_meta_ce.manifests import file_hash  # noqa: E402
from spatial_meta_ce.meta import loss_value, meta_loss_terms  # noqa: E402
from spatial_meta_ce.model import SetEstimator  # noqa: E402
from spatial_meta_ce.plans import load_training_plan  # noqa: E402
from spatial_meta_ce.train import gradient_sha256  # noqa: E402


def audit_run(run_dir, plan_path, device="cpu"):
    run_dir = Path(run_dir)
    config = json.loads((run_dir / "config.json").read_text())
    training = config["training"]
    data_config = config["data"]
    audit = json.loads((run_dir / "gradient_audit.json").read_text())
    if file_hash(run_dir / "initial_state.pt") != audit["initial_state_sha256"]:
        raise ValueError("initial_state.pt does not match the audited hash")
    state = torch.load(
        run_dir / "initial_state.pt", map_location="cpu", weights_only=False
    )
    model = SetEstimator(**state["model_config"]).to(device)
    model.load_state_dict(state["state_dict"], strict=True)
    model.train()
    plan = load_training_plan(plan_path, data_config)
    episode = plan["episodes"][audit["episode_index"]]
    if (
        episode["db"] != audit["database"]
        or episode["k"] != audit["k"]
        or episode["support"] != audit["support"]
        or episode["query"] != audit["query"]
        or episode["support_hashes"] != audit["support_hashes"]
        or episode["query_hashes"] != audit["query_hashes"]
    ):
        raise ValueError("Gradient audit episode does not match the sealed plan")
    pool = StorePool(data_config["cache_root"], max_open=1)
    store = pool.get("pretrain", episode["db"])
    support = store.batch(np.asarray(episode["support"], dtype=np.int64), device)
    query = store.batch(np.asarray(episode["query"], dtype=np.int64), device)
    tau = training.get("pinball_tau")
    if training["method"] == "fomaml":
        terms = meta_loss_terms(
            model,
            support,
            query,
            training["inner_lr"],
            training["inner_steps"],
            training["adapt_scope"],
            training["loss"],
            tau=tau,
            objective=training["meta_objective"],
        )
        total = terms["total"]
        pre_loss = float(terms["pre"].detach())
        post_loss = float(terms["post"].detach())
    else:
        batch = store.batch(
            np.asarray(episode["support"] + episode["query"], dtype=np.int64), device
        )
        total = loss_value(
            model(batch),
            batch["label"],
            training["loss"],
            **({} if tau is None else {"tau": tau}),
        )
        pre_loss = None
        post_loss = float(total.detach())
    parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    gradients = torch.autograd.grad(total, parameters)
    recomputed = {
        "pre_loss": pre_loss,
        "post_loss": post_loss,
        "total_loss": float(total.detach()),
        "gradient_sha256": gradient_sha256(model, gradients),
    }
    comparisons = {
        "pre_loss": audit["pre_loss"] == recomputed["pre_loss"],
        "post_loss": audit["post_loss"] == recomputed["post_loss"],
        "total_loss": audit["total_loss"] == recomputed["total_loss"],
        "gradient_sha256": audit["gradient_sha256"] == recomputed["gradient_sha256"],
    }
    if not all(comparisons.values()):
        raise ValueError(f"Gradient audit mismatch: {comparisons}")
    return {
        "verified": True,
        "comparisons": comparisons,
        "recomputed": recomputed,
        "initial_state_sha256": audit["initial_state_sha256"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    result = audit_run(args.run_dir, args.plan, args.device)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
