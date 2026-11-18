"""Explicit final target evaluation of the sealed experiment125 PRICE models."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import re
import sys
import time
import traceback
from pathlib import Path

from .price_meta123_runtime import (
    _cell_deadline,
    _torch_setup,
    digest,
    file_sha,
    output_path,
    read_json,
    require_supervision,
    safe_root,
    write_json,
)

DATABASES = ("imdb", "stats", "ergastf1", "genome")
SEEDS = (42, 43, 44)
ALPHA = 1e-6
INITIAL_SHA = "4f889c4df269d1bca5a50dadd83de7c801a607fe84228b6c21fb3371ae053f92"
TRAIN_BINDING_SHA = "220400d2d1a8797d3124e084caba4f1d7e1baac5ea953274884899c3206f0e17"
SEAL_SHA = "299b0cbf63f41950f54d1abc94ac0d095eaf3fa2fce553b681280d457c7159f4"
PROTOCOL_FILE = "实验文档/126-官方PRICE元学习四目标评估.md"
PROTOCOL_SHA = "50953e60708faa76b893391b0198988a09c2e45ca5f7957f0dab22d507c75bbf"
FIELDS = (
    "arm",
    "stream",
    "database",
    "k",
    "run",
    "query_id",
    "query_hash",
    "true",
    "encoded",
    "log_prediction",
)


def state_sha(model):
    value = hashlib.sha256()
    for key, tensor in model.state_dict().items():
        value.update(key.encode())
        value.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return value.hexdigest()


def build_target_model(checkpoint, counts, expected_sha, device):
    import contextlib
    import io
    import torch
    from torch import nn
    from .baselines import PriceAdapter
    from .price_meta123_model import verify_vendor
    from vendor.price_baseline.encoder import RegressionModel

    if file_sha(checkpoint) != expected_sha:
        raise ValueError("Target checkpoint SHA differs")
    if len(counts) != 4 or any(
        type(n) is not int or n < floor for n, floor in zip(counts, (12, 12, 7, 10))
    ):
        raise ValueError("Target padding may only extend the frozen training layout")
    verify_vendor()

    class TargetModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.counts = tuple(counts)
            with contextlib.redirect_stdout(io.StringIO()):
                self.price = RegressionModel(
                    *self.counts, 40, 4, 43, 512, 1024, 1, 256, 6, 8, 0.2
                )

        def forward(self, batch):
            for tensor in (*batch["groups"], batch["pg_log"], batch["counts"]):
                if not torch.isfinite(tensor).all():
                    raise FloatingPointError("Nonfinite target input")
            result = PriceAdapter.forward(self, batch)
            if not torch.isfinite(result).all():
                raise FloatingPointError("Nonfinite native target prediction")
            return result

    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if (
        not isinstance(state, dict)
        or not state
        or any(not torch.is_tensor(v) for v in state.values())
    ):
        raise ValueError("Target checkpoint is not a plain state_dict")
    for prefix in ("module.", "price."):
        if all(k.startswith(prefix) for k in state):
            state = {k[len(prefix) :]: v for k, v in state.items()}
    if any(not torch.isfinite(v).all() for v in state.values()):
        raise FloatingPointError("Nonfinite checkpoint weights")
    model = TargetModel()
    model.price.load_state_dict(state, strict=True)
    return model.to(device).eval()


def verify_csv(path, expected, cell=None):
    """Recompute standard metrics without production decoder or metric calls."""
    import numpy as np

    errors = []
    seen = set()
    with gzip.open(path, "rt", newline="") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != FIELDS:
            raise ValueError("Target prediction CSV schema changed")
        for index, row in enumerate(reader):
            if index >= len(expected) or None in row:
                raise ValueError("Target prediction CSV has extra rows")
            truth, encoded, log_prediction = (
                float(row[k]) for k in ("true", "encoded", "log_prediction")
            )
            if (
                not all(math.isfinite(v) for v in (truth, encoded, log_prediction))
                or truth < 0
            ):
                raise FloatingPointError("Nonfinite target prediction or truth")
            if (row["query_id"], row["query_hash"], truth) != tuple(
                expected[index]
            ) or row["query_hash"] in seen:
                raise ValueError("Target query identity, order or truth changed")
            if cell is not None and any(
                str(row[k]) != str(cell[k])
                for k in ("arm", "stream", "database", "k", "run")
            ):
                raise ValueError("Target cell identity changed")
            a = encoded - 1
            decoded = 0.0 if a <= math.log(2) else a + math.log1p(-math.exp(-a))
            if not math.isclose(
                decoded, max(log_prediction, 0), rel_tol=1e-10, abs_tol=1e-10
            ):
                raise ValueError("Native decoding differs from stored log prediction")
            seen.add(row["query_hash"])
            errors.append(abs(max(log_prediction, 0) - math.log(max(truth, 1))))
    if len(errors) != len(expected) or not errors:
        raise ValueError("Target prediction CSV lost queries")
    with np.errstate(over="ignore"):
        q = np.exp(np.asarray(errors, dtype=np.float64))
    if not np.isfinite(q).all():
        raise FloatingPointError("Q-error overflow; predictions cannot be clipped")
    return {
        **{
            f"q{p}": float(np.percentile(q, p, method="linear"))
            for p in (50, 90, 95, 99)
        },
        "q_mean": float(q.mean()),
        "n": len(q),
    }


def training_records(training_root):
    root = safe_root(training_root)
    binding = read_json(root / "inputs/binding.json")
    if (
        binding.get("protocol") != 125
        or binding.get("binding_sha256") != TRAIN_BINDING_SHA
        or binding.get("initial_sha256") != INITIAL_SHA
    ):
        raise ValueError("Frozen125 training binding differs")
    if (
        digest({k: v for k, v in binding.items() if k != "binding_sha256"})
        != TRAIN_BINDING_SHA
    ):
        raise ValueError("Training binding digest differs")
    if any(
        file_sha(root / "code" / name) != sha
        for name, sha in binding["code_sha256"].items()
    ):
        raise ValueError("Original125 code changed")
    seal_path = root / "outputs/seal-training/completion.json"
    seal = read_json(seal_path)
    sup = read_json(root / "outputs/seal-training.supervisor.json")
    if (
        file_sha(seal_path) != SEAL_SHA
        or sup.get("status") != "completed"
        or sup.get("completion") != seal
    ):
        raise ValueError("Training seal is not supervised or changed")
    chain = read_json(root / "evidence/continuation-completed.json")
    if (
        chain.get("status") != "completed"
        or chain.get("binding_sha256") != TRAIN_BINDING_SHA
        or len(chain.get("completed", [])) != 7
    ):
        raise ValueError("Training sequence is not complete")
    records = {
        "P-0": {
            "arm": "P",
            "stream": 0,
            "checkpoint": binding["initial_checkpoint"],
            "sha256": INITIAL_SHA,
        }
    }
    if file_sha(binding["initial_checkpoint"]) != INITIAL_SHA:
        raise ValueError("Official initialization changed")
    for arm in ("E", "M"):
        for seed in SEEDS:
            key = f"{arm}-{seed}"
            path = root / "outputs" / f"train-{key}"
            c = read_json(path / "completion.json")
            s = read_json(path.with_name(path.name + ".supervisor.json"))
            if (
                c.get("protocol") != 125
                or c.get("binding_sha256") != TRAIN_BINDING_SHA
                or c.get("updates") != 2000
                or c.get("arm") != arm
                or c.get("stream") != seed
                or c.get("alpha") != ALPHA
                or c.get("initial_sha256") != INITIAL_SHA
                or c.get("target_databases_read") != []
                or s.get("status") != "completed"
                or s.get("completion") != c
                or file_sha(path / "completion.json")
                != seal["runs"][key]["completion_sha256"]
                or file_sha(path / "last.pt") != seal["runs"][key]["checkpoint_sha256"]
                or file_sha(path / "last.pt") != c["checkpoint_sha256"]
                or file_sha(path / "training_state.pt") != c["training_state_sha256"]
            ):
                raise ValueError("Training run, state or checkpoint differs from seal")
            records[key] = {
                "arm": arm,
                "stream": seed,
                "checkpoint": str(path / "last.pt"),
                "sha256": c["checkpoint_sha256"],
            }
    return records


def code_closure(root):
    root = Path(root)
    paths = [
        root / PROTOCOL_FILE,
        root / "vendor/PROVENANCE.json",
        root / "scripts/run_price_meta126.py",
    ]
    for directory in ("spatial_meta_ce", "vendor/price_baseline"):
        paths.extend(sorted((root / directory).rglob("*.py")))
    if any(not p.is_file() or p.is_symlink() for p in paths):
        raise ValueError("Missing or linked target execution dependency")
    return {str(p.relative_to(root)): file_sha(p) for p in paths}


def bind(root, training_root, cache_root):
    from .price_meta126_plans import build_plan

    root = safe_root(root)
    closure = code_closure(root / "code")
    if closure[PROTOCOL_FILE] != PROTOCOL_SHA:
        raise ValueError("Target protocol changed")
    models = training_records(training_root)
    plan = build_plan(cache_root)
    plan_path = root / "inputs/target-plan.json"
    write_json(plan_path, plan)
    value = {
        "schema": 1,
        "protocol": 126,
        "root": str(root),
        "training_root": str(training_root),
        "training_seal_sha256": SEAL_SHA,
        "code_sha256": closure,
        "protocol_sha256": PROTOCOL_SHA,
        "models": models,
        "cache_root": str(Path(cache_root).resolve()),
        "plan": str(plan_path),
        "plan_sha256": file_sha(plan_path),
        "allowed_targets": list(DATABASES),
        "alpha": ALPHA,
        "steps": 5,
        "selection_forbidden": True,
    }
    value["binding_sha256"] = digest(value)
    write_json(root / "inputs/binding.json", value)
    return value


def load_binding(path):
    from .price_meta126_plans import verify_plan

    value = read_json(path)
    if (
        value.get("schema") != 1
        or value.get("protocol") != 126
        or value.get("binding_sha256")
        != digest({k: v for k, v in value.items() if k != "binding_sha256"})
    ):
        raise ValueError("Invalid target binding")
    root = safe_root(value["root"])
    if Path(path).resolve() != root / "inputs/binding.json" or value.get("plan") != str(
        root / "inputs/target-plan.json"
    ):
        raise ValueError("Target binding paths escaped root")
    if (
        value.get("protocol_sha256") != PROTOCOL_SHA
        or value.get("alpha") != ALPHA
        or value.get("steps") != 5
        or value.get("selection_forbidden") is not True
        or value.get("allowed_targets") != list(DATABASES)
        or value.get("training_seal_sha256") != SEAL_SHA
        or code_closure(root / "code") != value["code_sha256"]
        or value["code_sha256"].get(PROTOCOL_FILE) != PROTOCOL_SHA
        or file_sha(value["plan"]) != value["plan_sha256"]
        or training_records(value["training_root"]) != value["models"]
    ):
        raise ValueError("Frozen target execution inputs changed")
    verify_plan(read_json(value["plan"]), value["cache_root"])
    return value


def validate_authorization(auth, binding):
    if (
        auth.get("schema") != 1
        or auth.get("protocol") != 126
        or auth.get("authorized") is not True
        or auth.get("binding_sha256") != binding["binding_sha256"]
        or auth.get("root") != binding["root"]
        or auth.get("actions") != ["evaluate"]
        or auth.get("allowed_targets") != list(DATABASES)
        or auth.get("selection_forbidden") is not True
        or not auth.get("granted_by")
        or not auth.get("granted_at")
    ):
        raise ValueError("Explicit frozen four-target authorization required")
    from .price_meta123_supervisor import _validate_budget

    budget = _validate_budget(auth["budget"])
    if (
        budget["max_wall_seconds"] > 10800
        or budget["max_rss_bytes"] > 24 * 1024**3
        or not 0 < budget["max_vram_bytes"] <= 12 * 1024**3
        or budget["max_output_bytes"] > 1024**3
        or budget["poll_seconds"] > 1
    ):
        raise ValueError("Target resource budget exceeds frozen limits")
    return budget


def unit(output, model, record, db, test, indices, support, cell):
    import numpy as np
    import torch
    from .baselines import price_log_predictions
    from .metrics import summarize
    from .price_meta123_model import adapt_native, predict_native

    params, audit = None, {}
    before = state_sha(model) if model is not None else None
    device = next(model.parameters()).device if model is not None else "cpu"
    started = time.monotonic()
    if cell["k"]:
        params, audit = adapt_native(model, support, ALPHA, steps=5, meta=False)
    name = f"{cell['arm']}-{cell['stream']}-{db}-k{cell['k']}-r{cell['run']}.csv.gz"
    path = Path(output) / name
    expected, all_logs, truths = [], [], []
    with gzip.open(path, "xt", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        for offset in range(0, len(indices), 128):
            selected = indices[offset : offset + 128]
            batch = test.batch(selected, device)
            with torch.no_grad():
                if model is None:
                    raw_pg = np.asarray(test.cards[selected, 1], dtype=np.float64)
                    if not np.isfinite(raw_pg).all() or (raw_pg < 0).any():
                        raise FloatingPointError("Invalid raw PG reference")
                    logs = np.log(np.maximum(raw_pg, 1))
                    encoded = np.logaddexp(logs, 0) + 1
                else:
                    encoded = (
                        predict_native(model, batch, params)
                        .detach()
                        .cpu()
                        .double()
                        .numpy()
                    )
                    logs = price_log_predictions(encoded)
            for index, z, logp in zip(selected, encoded, logs):
                row = test.rows[index]
                truth = float(test.cards[index, 0])
                writer.writerow(
                    {
                        **cell,
                        "query_id": row["id"],
                        "query_hash": row["query_hash"],
                        "true": repr(truth),
                        "encoded": repr(float(z)),
                        "log_prediction": repr(float(logp)),
                    }
                )
                expected.append((row["id"], row["query_hash"], truth))
                all_logs.append(float(logp))
                truths.append(truth)
    metrics = verify_csv(path, expected, cell)
    online = summarize(all_logs, truths)
    for key in ("q50", "q90", "q95", "q99"):
        if not math.isclose(metrics[key], online[key], rel_tol=1e-10, abs_tol=1e-10):
            raise ValueError("Independent CSV metric disagrees")
    if model is not None and state_sha(model) != before:
        raise ValueError("Adaptation mutated shared target checkpoint")
    return {
        **cell,
        "status": "completed",
        "n": len(indices),
        "metrics": metrics,
        "lr": ALPHA if cell["k"] else 0.0,
        "population": "target_test_unique",
        "prediction_file": name,
        "prediction_sha256": file_sha(path),
        "checkpoint_sha256": record["sha256"] if record else None,
        "adaptation": audit,
        "seconds": time.monotonic() - started,
    }


def evaluate(binding, output):
    from .price_meta126_plans import TargetStores
    from .price_meta126_results import aggregate

    _torch_setup("cuda")
    plan = read_json(binding["plan"])
    stores = TargetStores(binding["cache_root"])
    units = []

    def publish(value):
        units.append(value)
        write_json(
            Path(output) / "progress.json",
            {
                "completed_cells": len(units),
                "last_cell": {
                    k: value[k] for k in ("arm", "stream", "database", "k", "run")
                },
            },
            replace=True,
        )

    for db in DATABASES:
        entry = plan["databases"][db]
        test = stores.get("test", db)
        indices = entry["test"]["indices"]
        with _cell_deadline():
            publish(
                unit(
                    output,
                    None,
                    None,
                    db,
                    test,
                    indices,
                    None,
                    {"arm": "PG", "stream": 0, "database": db, "k": 0, "run": 0},
                )
            )
        for record in binding["models"].values():
            model = build_target_model(
                record["checkpoint"],
                tuple(entry["max_counts"]),
                record["sha256"],
                "cuda",
            )
            base = {"arm": record["arm"], "stream": record["stream"], "database": db}
            specs = [(0, 0)]
            if entry["support"]["available"]:
                specs.extend((k, r) for k in (5, 20) for r in range(5))
            for k, run in specs:
                batch = None
                if k:
                    selected = entry["support"]["blocks"][run]["indices"][:k]
                    batch = stores.get("finetune", db).batch(selected, "cuda")
                with _cell_deadline():
                    publish(
                        unit(
                            output,
                            model,
                            record,
                            db,
                            test,
                            indices,
                            batch,
                            {**base, "k": k, "run": run},
                        )
                    )
            del model
    if len(units) != 242:
        raise ValueError("Incomplete target matrix")
    summary = aggregate(
        units, {db: len(plan["databases"][db]["test"]["indices"]) for db in DATABASES}
    )
    write_json(Path(output) / "units.json", units)
    write_json(Path(output) / "summary.json", summary)
    completion = {
        "schema": 1,
        "protocol": 126,
        "status": "completed",
        "binding_sha256": binding["binding_sha256"],
        "cells": 242,
        "units_sha256": file_sha(Path(output) / "units.json"),
        "summary_sha256": file_sha(Path(output) / "summary.json"),
        "target_databases_read": list(DATABASES),
        "selection_forbidden": True,
        "imdb_fewshot_status": "unavailable_missing_independent_support",
    }
    write_json(Path(output) / "completion.json", completion)
    return completion


def worker(binding_path, auth_path, output):
    require_supervision()
    binding = load_binding(binding_path)
    validate_authorization(read_json(auth_path), binding)
    output = Path(output)
    if (
        output != Path(binding["root"]) / "outputs/evaluate"
        or output.exists()
        or output.is_symlink()
    ):
        raise ValueError("Target output must be fresh and isolated")
    output.mkdir()
    try:
        return evaluate(binding, output)
    except BaseException as exc:
        write_json(
            output / "failure.json",
            {
                "type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
                "allowed_targets": list(DATABASES),
                "selection_forbidden": True,
            },
        )
        raise


def run(binding_path, auth_path):
    import fcntl
    from .price_meta123_supervisor import supervise

    binding = load_binding(binding_path)
    auth = read_json(auth_path)
    budget = validate_authorization(auth, binding)
    index, gpu_uuid = auth.get("cuda_device"), auth.get("cuda_uuid")
    if (
        type(index) is not int
        or not 0 <= index <= 7
        or not isinstance(gpu_uuid, str)
        or not re.fullmatch(r"GPU-[0-9a-fA-F-]{36}", gpu_uuid)
    ):
        raise ValueError("Exactly one authorized physical target GPU required")
    output = output_path(binding["root"], "evaluate")
    lock_path = Path(binding["root"]) / "inputs/execution.lock"
    if lock_path.is_symlink():
        raise ValueError("Execution lock is linked")
    with lock_path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return supervise(
            [
                sys.executable,
                "-B",
                str(Path(binding["root"]) / "code/scripts/run_price_meta126.py"),
                "worker",
                "--binding",
                str(Path(binding_path).resolve()),
                "--authorization",
                str(Path(auth_path).resolve()),
                "--output",
                str(output),
            ],
            output,
            cwd=Path(binding["root"]) / "code",
            budget=budget,
            env={
                "CUDA_VISIBLE_DEVICES": str(index),
                "PYTHONDONTWRITEBYTECODE": "1",
                "OMP_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
            },
            gpu_uuid=gpu_uuid,
            completion_protocol=126,
        )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    b = sub.add_parser("bind")
    for name in ("root", "training-root", "cache-root"):
        b.add_argument("--" + name, required=True)
    for command in ("run", "worker"):
        p = sub.add_parser(command)
        p.add_argument("--binding", required=True)
        p.add_argument("--authorization", required=True)
        if command == "worker":
            p.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    if args.command == "bind":
        value = bind(args.root, args.training_root, args.cache_root)
    elif args.command == "run":
        value = run(args.binding, args.authorization)
    else:
        value = worker(args.binding, args.authorization, args.output)
    print(json.dumps(value, ensure_ascii=False, allow_nan=False))
    return 0 if value.get("status", "completed") == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
