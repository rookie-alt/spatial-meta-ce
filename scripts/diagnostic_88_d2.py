#!/usr/bin/env python3
"""D2 of protocol 88: how much of the head3 prediction change can a rank-4
linear function of the 64 head features express?

Read-only CPU analysis on validation databases chembl/legalacts/telstra. Loads
one checkpoint but performs no training and mutates no shared state: operator
parameters come from ``meta65_joint_runner.operator_params`` and every forward
is a ``torch.func.functional_call`` with those parameters.

Definitions from protocol 88 section 4.1, computed per (database, run):

    Delta = f_head3(q) - f_none(q)
    Z     = [head_features(q) under none, 1]            # n x 65
    c_lin = argmin_c ||Delta - Z c||^2
    R2_lin = 1 - ||Delta - Z c_lin||^2 / ||Delta||^2    # self-consistency only
    U4     = first 4 left singular vectors of Z
    R2_4   = 1 - ||Delta - U4 U4^T Delta||^2 / ||Delta||^2

The alternative basis takes the first four right singular vectors of Z, i.e.
the leading feature directions, and is reported next to the protocol figure.
Target databases are rejected before any path is constructed; hepatitis and
sakila are skipped without being read.
"""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

VALIDATION_DATABASES = ("chembl", "legalacts", "telstra")
FORBIDDEN_DATABASES = ("imdb", "stats", "genome", "ergastf1")
OPERATORS = ("none", "head3")
RUNS = tuple(range(5))
BATCH_SIZE = 512
CONSISTENCY_TOLERANCE = 1e-4
R2_LINEAR_MINIMUM = 0.95
R2_4_NOT_BLOCKED = 0.80
R2_4_INSUFFICIENT = 0.50
UNSTABLE_RANGE = 0.30
RANK = 4


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def guard_database(name):
    if name in FORBIDDEN_DATABASES:
        raise ValueError("Forbidden database name reached protocol 88 D2: %s" % name)
    if name not in VALIDATION_DATABASES:
        raise ValueError("Unexpected database in protocol 88 D2: %s" % name)


def load_inputs(run_dir, plan_path, database, run):
    """Load the frozen plan entry and the two archived prediction vectors."""
    import numpy as np

    plan = json.loads(Path(plan_path).read_text(encoding="utf-8"))
    if database not in plan or run >= len(plan[database]):
        raise ValueError("Plan entry missing: %s run %d" % (database, run))
    entry = plan[database][run]
    query = [int(value) for value in entry["query"]]
    support = [int(value) for value in entry["supports"]["20"]]
    archived = {}
    for operator in OPERATORS:
        path = Path(run_dir) / "predictions" / (
            "R_head3-%s-%s-run%d.npz" % (operator, database, run))
        with np.load(path, allow_pickle=False) as archive:
            archived[operator] = archive["log_pred"].astype(np.float64).reshape(-1)
        if len(archived[operator]) != len(query):
            raise ValueError("Archived prediction length differs from plan query: %s %s"
                             % (database, operator))
    return entry, query, support, archived


def forward_blocks(run_module, model, params, store, indices, capture_features):
    """Run one operator over every query in BATCH_SIZE chunks."""
    import numpy as np
    import torch

    from spatial_meta_ce.meta import forward

    values, features = [], []
    with torch.no_grad():
        for start in range(0, len(indices), BATCH_SIZE):
            chunk = indices[start:start + BATCH_SIZE]
            batch = run_module.certified_batch(store, chunk, "cpu")
            values.append(forward(model, params, batch).detach()
                          .double().cpu().numpy().reshape(-1))
            if capture_features:
                features.append(capture_head_features(model, params, batch)
                                .double().cpu().numpy())
    return np.concatenate(values), (np.concatenate(features) if capture_features else None)


def capture_head_features(model, params, batch):
    """Evaluate ``head[:-1](_head_input(batch))`` under the supplied parameters.

    torch 2.1 functional_call has no ``method=`` form, so the head input is
    rebuilt from the documented modules and each submodule is called with its
    own parameter slice.  No module state is written.
    """
    import torch
    from torch.func import functional_call

    def local(prefix, module):
        """Slice parameters down to the names ``module`` itself expects."""
        valid = set(module.state_dict())
        return {key[len(prefix):]: value for key, value in params.items()
                if key.startswith(prefix) and key[len(prefix):] in valid}

    if model.arch != "sum":
        raise ValueError("D2 feature capture expects the sum architecture")
    masks = batch["masks"]
    tokens = [functional_call(encoder, local("encoders.%d." % index, encoder),
                              (values.masked_fill(~mask.unsqueeze(-1), 0),), strict=True)
              for index, (encoder, values, mask) in enumerate(
                  zip(model.encoders, batch["groups"], masks))]
    pooled = []
    for token, mask in zip(tokens, masks):
        valid = mask.unsqueeze(-1)
        total = (token * valid).sum(1)
        mean = total / valid.sum(1).clamp_min(1)
        largest = token.masked_fill(~valid, -torch.inf).amax(1)
        largest = torch.where(mask.any(1, keepdim=True), largest, torch.zeros_like(largest))
        pooled.extend([mean, largest, total])
    pg = batch["pg_log"] if model.use_pg else torch.zeros_like(batch["pg_log"])
    log_cartesian = (batch["groups"][2][:, :, 0] * masks[2]).sum(1)
    head_input = torch.cat(
        [*pooled, torch.log1p(batch["counts"]), pg[:, None], log_cartesian[:, None]], dim=1)
    return functional_call(model.head[:-1], local("head.", model.head[:-1]), (head_input,), strict=True)


def solve(delta, features):
    """Return R2_linear and both rank-4 explanations in float64."""
    import numpy as np

    delta = np.asarray(delta, dtype=np.float64).reshape(-1)
    features = np.asarray(features, dtype=np.float64)
    if not np.isfinite(delta).all() or not np.isfinite(features).all():
        raise ValueError("Nonfinite delta or features")
    z = np.concatenate([features, np.ones((len(features), 1))], axis=1)
    energy = float(delta @ delta)
    if energy <= 0:
        raise ValueError("Zero delta energy")
    coefficients, *_ = np.linalg.lstsq(z, delta, rcond=None)
    residual = delta - z @ coefficients
    r2_linear = 1.0 - float(residual @ residual) / energy
    u, singular, vt = np.linalg.svd(z, full_matrices=False)
    u4 = u[:, :RANK]
    left_residual = delta - u4 @ (u4.T @ delta)
    r2_left = 1.0 - float(left_residual @ left_residual) / energy
    # Alternative basis: span of the leading RANK feature directions, i.e. the
    # first RANK right singular vectors of Z.  Fit Delta against the resulting
    # n x 4 design and take the least-squares residual.
    design = z @ vt[:RANK].T
    alt_coefficients, *_ = np.linalg.lstsq(design, delta, rcond=None)
    column_residual = delta - design @ alt_coefficients
    r2_columns = 1.0 - float(column_residual @ column_residual) / energy
    return {"R2_linear": r2_linear, "R2_4_protocol": r2_left,
            "R2_4_altbasis": r2_columns,
            "delta_norm": float(np.sqrt(energy)),
            "residual_norm_protocol": float(np.sqrt(left_residual @ left_residual)),
            "residual_norm_altbasis": float(np.sqrt(column_residual @ column_residual)),
            "singular_values_top8": [float(value) for value in singular[:8]],
            "design_columns": int(z.shape[1]), "queries": int(len(delta))}


def median(values):
    import numpy as np

    clean = [value for value in values if value is not None]
    return float(np.median(clean)) if clean else None


def decide(per_database):
    """Apply the frozen protocol-88 4.3 thresholds to protocol-basis medians."""
    medians = {db: median([cell["R2_4_protocol"] for cell in cells])
               for db, cells in per_database.items()}
    if not medians or any(value is None for value in medians.values()):
        return {"status": "inconclusive", "reason": "missing medians", "medians": medians}
    spread = max(medians.values()) - min(medians.values())
    # Guard the boundary against binary floating point: 0.90 - 0.60 is
    # 0.30000000000000004, which must not flip a boundary case to unstable.
    if spread > UNSTABLE_RANGE + 1e-12:
        status, reason = "unstable", "database range %.4f exceeds %.2f" % (
            spread, UNSTABLE_RANGE)
    elif min(medians.values()) < R2_4_INSUFFICIENT:
        status, reason = "insufficient", "at least one database median below %.2f" % (
            R2_4_INSUFFICIENT)
    elif min(medians.values()) < R2_4_NOT_BLOCKED:
        status, reason = "partial", "medians between %.2f and %.2f" % (
            R2_4_INSUFFICIENT, R2_4_NOT_BLOCKED)
    else:
        status, reason = "expressible", "every database median at or above %.2f" % (
            R2_4_NOT_BLOCKED)
    return {"status": status, "reason": reason, "medians": medians,
            "spread": spread, "thresholds": {
                "not_blocked": R2_4_NOT_BLOCKED,
                "insufficient": R2_4_INSUFFICIENT,
                "unstable_range": UNSTABLE_RANGE}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    for path in (args.run_dir, args.plan, args.checkpoint, args.cache_root, args.out):
        if any(part in FORBIDDEN_DATABASES for part in Path(path).parts):
            raise ValueError("Target database appears in a path: %s" % path)
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("D2 requires explicitly hidden CUDA")
    if args.out.exists():
        raise ValueError("Refusing to overwrite output")

    import numpy as np
    import torch

    from scripts import meta65_joint_runner as runner
    from spatial_meta_ce.checkpoints import load_checkpoint
    from spatial_meta_ce.data import FeatureStore

    torch.set_num_threads(1)
    model, state = load_checkpoint(args.checkpoint, "cpu")
    if state["next_step"] != 4000 or state["model_config"].get("arch") != "sum":
        raise ValueError("Checkpoint is not the frozen 4000-step sum64 head3 model")
    if any(not torch.isfinite(value).all() for value in state["model"].values()):
        raise FloatingPointError("Nonfinite checkpoint weights")

    report = {"schema": 1, "protocol": "88 D2", "run_dir": str(args.run_dir),
              "inputs": {"checkpoint_sha256": sha256(args.checkpoint),
                         "plan_sha256": sha256(args.plan),
                         "model_config": state["model_config"],
                         "next_step": state["next_step"],
                         "code_sha256": sha256(__file__)},
              "tolerance": CONSISTENCY_TOLERANCE,
              "databases": {}}

    for database in VALIDATION_DATABASES:
        guard_database(database)
        store = FeatureStore(Path(args.cache_root) / "pretrain" / database)
        report["databases"][database] = []
        for run in RUNS:
            _, query, support, archived = load_inputs(
                args.run_dir, args.plan, database, run)
            support_batch = runner.certified_batch(store, support, "cpu")
            params, records = {}, {}
            for operator in OPERATORS:
                variant, record = runner.operator_params(model, support_batch, operator)
                expected = {"none": (0, 0), "head3": (3, 107521)}[operator]
                if (record["steps"], record["selected_parameter_count"]) != expected:
                    raise ValueError("Unexpected adaptation record for %s" % operator)
                params[operator] = variant
                records[operator] = {key: record[key] for key in
                                     ("selected_parameter_count",
                                      "changed_parameter_count", "steps")}
            predictions, features = {}, {}
            for operator in OPERATORS:
                values, captured = forward_blocks(
                    runner, model, params[operator], store, query,
                    capture_features=(operator == "none"))
                predictions[operator] = values
                if captured is not None:
                    features[operator] = captured
            consistency = {
                "max_abs_diff_none": float(np.abs(predictions["none"] - archived["none"]).max()),
                "max_abs_diff_head3": float(np.abs(predictions["head3"] - archived["head3"]).max())}
            consistency["passed"] = (
                consistency["max_abs_diff_none"] <= CONSISTENCY_TOLERANCE
                and consistency["max_abs_diff_head3"] <= CONSISTENCY_TOLERANCE)
            solved = solve(predictions["head3"] - predictions["none"], features["none"])
            solved["run"] = run
            solved["consistency"] = consistency
            solved["operators"] = records
            report["databases"][database].append(solved)
            print("%s run%d R2_lin=%.6f R2_4=%.4f alt=%.4f diff=(%.2e,%.2e)" % (
                database, run, solved["R2_linear"], solved["R2_4_protocol"],
                solved["R2_4_altbasis"], consistency["max_abs_diff_none"],
                consistency["max_abs_diff_head3"]), flush=True)
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(
                json.dumps(report, indent=1, ensure_ascii=False, allow_nan=False),
                encoding="utf-8")
            if not consistency["passed"]:
                raise ValueError("Consistency tolerance exceeded at %s run %d; stopping"
                                 % (database, run))

    report["aggregate"] = {
        db: {"R2_linear_median": median([cell["R2_linear"] for cell in cells]),
             "R2_4_protocol_median": median([cell["R2_4_protocol"] for cell in cells]),
             "R2_4_altbasis_median": median([cell["R2_4_altbasis"] for cell in cells]),
             "delta_norm_median": median([cell["delta_norm"] for cell in cells])}
        for db, cells in report["databases"].items()}
    report["verdict"] = decide(report["databases"])
    linear = min(item["R2_linear_median"] for item in report["aggregate"].values())
    report["verification"] = {
        "R2_linear_minimum": R2_LINEAR_MINIMUM,
        "R2_linear_ok": linear >= R2_LINEAR_MINIMUM,
        "all_consistency_passed": all(
            cell["consistency"]["passed"]
            for cells in report["databases"].values() for cell in cells),
        "cells": sum(len(cells) for cells in report["databases"].values())}
    args.out.write_text(json.dumps(report, indent=1, ensure_ascii=False, allow_nan=False),
                        encoding="utf-8")
    print(json.dumps({"aggregate": report["aggregate"], "verdict": report["verdict"],
                      "verification": report["verification"]},
                     indent=1, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
