#!/usr/bin/env python3
"""Bounded 112 smoke and timing preflight. Never writes formal results."""

from __future__ import annotations

import argparse
import json
import os
import resource
import subprocess
import sys
import time
from collections import OrderedDict
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from spatial_meta_ce import train as train_lib  # noqa: E402
from spatial_meta_ce.checkpoints import load_checkpoint  # noqa: E402
from spatial_meta_ce.cli import data_settings, read_json  # noqa: E402
from spatial_meta_ce.data import StorePool  # noqa: E402
from spatial_meta_ce.manifests import file_hash  # noqa: E402
from spatial_meta_ce.meta import adapt  # noqa: E402
from spatial_meta_ce.plans import (  # noqa: E402
    EVAL_DATABASES,
    build_training_plan,
    load_source_eval_plan,
    verify_source_eval_plan_against_stores,
)
from spatial_meta_ce.source_eval import _predict  # noqa: E402
from spatial_meta_ce.train import (  # noqa: E402
    FIXED_ACCUMULATION,
    FIXED_META_BATCH,
    FIXED_MODEL,
    FIXED_QUERY_SIZE,
    FIXED_SUPPORT_SIZES,
)

MEMORY_LIMIT = 12 * 1024**3
OUTPUT_LIMIT = 2 * 1024**3
VRAM_LIMIT = 8 * 1024**3
DEFAULT_SECONDS = 30 * 60
PURPOSE = "smoke-not-research"
ARMS = {
    "A_erm": ("erm", "post"),
    "B_fomaml": ("fomaml", "post"),
    "C_joint": ("fomaml", "joint"),
}


def _set_limits():
    resource.setrlimit(resource.RLIMIT_FSIZE, (OUTPUT_LIMIT, OUTPUT_LIMIT))


def _rss_kib(pid):
    status = Path(f"/proc/{pid}/status")
    if status.is_file():
        for line in status.read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1])
        return 0
    try:
        return int(
            subprocess.check_output(
                ["ps", "-o", "rss=", "-p", str(pid)], text=True
            ).strip()
        )
    except (OSError, ValueError, subprocess.CalledProcessError):
        return 0


def _set_cuda_limit():
    if torch.cuda.is_available():
        total = torch.cuda.get_device_properties(0).total_memory
        torch.cuda.set_per_process_memory_fraction(min(1.0, VRAM_LIMIT / total))
        torch.cuda.synchronize()


def _sync(device):
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def _write(path, value):
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )


def _base_config(plan_path, steps, arm):
    config = read_json(ROOT / "configs" / "fomaml_joint_sum_4k.json")
    method, objective = ARMS[arm]
    config.update(
        method=method,
        meta_objective=objective,
        seed=42,
        steps=steps,
        fixed_budget=True,
        training_plan=str(Path(plan_path).resolve()),
        purpose=PURPOSE,
    )
    if config["model"] != FIXED_MODEL:
        raise ValueError("Smoke config does not use the frozen model")
    return config


def _build_plans(data_config, out_dir):
    plans = {}
    for steps, name in ((2, "reload"), (200, "timing")):
        plan = build_training_plan(
            data_config,
            seed=42,
            steps=steps,
            meta_batch=FIXED_META_BATCH,
            accumulation=FIXED_ACCUMULATION,
            support_sizes=list(FIXED_SUPPORT_SIZES),
            query_size=FIXED_QUERY_SIZE,
        )
        path = out_dir / "plans" / f"{name}_plan.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        _write(path, plan)
        plans[name] = str(path)
    return plans


def _reload_phase(data_config, plan_path, out_dir):
    steps = 2
    config = _base_config(plan_path, steps, "C_joint")
    output = out_dir / "reload"
    started = time.perf_counter()
    with patch.object(train_lib, "FIXED_STEPS", steps):
        result = train_lib.train(config, data_config, output, "cuda")
    _sync("cuda")
    elapsed = time.perf_counter() - started
    checkpoint = output / "last.pt"
    model, state = load_checkpoint(checkpoint, "cuda")
    metadata = state["metadata"]
    if state["next_step"] != steps or metadata["purpose"] != PURPOSE:
        raise ValueError("Reload smoke metadata mismatch")
    if metadata["fixed_steps"] != steps or (output / "best.pt").exists():
        raise ValueError("Reload smoke budget/selection mismatch")
    pool = StorePool(data_config["cache_root"], max_open=1)
    store = pool.get("pretrain", "accidents")
    with torch.no_grad():
        prediction = model(store.batch([int(store.unique_indices()[0])], "cuda"))
    if not torch.isfinite(prediction).all():
        raise ValueError("Reload smoke produced a non-finite prediction")
    return {
        "phase": "reload",
        "updates": steps,
        "seconds": elapsed,
        "counts": result["counts"],
        "checkpoint_sha256": file_hash(checkpoint),
        "initial_state_sha256": metadata["initial_state_sha256"],
        "reload_next_step": state["next_step"],
        "purpose": metadata["purpose"],
        "best_pt_exists": (output / "best.pt").exists(),
    }


def _train_phase(data_config, plan_path, arm, out_dir):
    steps = 200
    config = _base_config(plan_path, steps, arm)
    output = out_dir / "training" / arm
    started = time.perf_counter()
    with patch.object(train_lib, "FIXED_STEPS", steps):
        result = train_lib.train(config, data_config, output, "cuda")
    _sync("cuda")
    elapsed = time.perf_counter() - started
    if (output / "best.pt").exists():
        raise ValueError("Smoke training wrote best.pt")
    _, state = load_checkpoint(output / "last.pt", "cuda")
    return {
        "phase": "train",
        "arm": arm,
        "updates": steps,
        "seconds": elapsed,
        "counts": result["counts"],
        "gradient_stats": result.get("gradient_stats"),
        "checkpoint_sha256": file_hash(output / "last.pt"),
        "next_step": state["next_step"],
        "purpose": state["metadata"]["purpose"],
        "rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "cuda_max_memory_allocated": int(torch.cuda.max_memory_allocated()),
    }


def _eval_phase(data_config, source_plan_path, checkpoint, out_dir):
    plan = load_source_eval_plan(source_plan_path, data_config)
    pool = StorePool(data_config["cache_root"], max_open=len(EVAL_DATABASES))
    stores = {db: pool.get("pretrain", db) for db in EVAL_DATABASES}
    verify_source_eval_plan_against_stores(plan, stores)
    model, state = load_checkpoint(checkpoint, "cuda")
    model.eval()
    metadata = state["metadata"]
    if metadata["purpose"] != PURPOSE:
        raise ValueError("Smoke eval checkpoint has the wrong purpose")
    units = []
    for db in EVAL_DATABASES:
        store = stores[db]
        entry = plan["databases"][db]
        query = np.asarray(entry["query"], dtype=np.int64)
        params = OrderedDict(model.named_parameters())
        _sync("cuda")
        started = time.perf_counter()
        _predict(model, params, store, query, "cuda", 128)
        _sync("cuda")
        k0_seconds = time.perf_counter() - started
        support = np.asarray(
            entry["support_blocks"][0]["support"][:20], dtype=np.int64
        )
        _sync("cuda")
        adapt_started = time.perf_counter()
        adapted = adapt(
            model,
            store.batch(support, "cuda"),
            metadata["training_signature"]["inner_lr"],
            metadata["training_signature"]["inner_steps"],
            metadata["training_signature"]["adapt_scope"],
            metadata["training_signature"]["loss"],
            training=False,
        )
        _sync("cuda")
        adapt_seconds = time.perf_counter() - adapt_started
        _sync("cuda")
        predict_started = time.perf_counter()
        _predict(model, adapted, store, query, "cuda", 128)
        _sync("cuda")
        predict_seconds = time.perf_counter() - predict_started
        units.append(
            {
                "database": db,
                "k0_seconds": k0_seconds,
                "k20_adapt_seconds": adapt_seconds,
                "k20_predict_seconds": predict_seconds,
                "query_count": len(query),
                "support_count": len(support),
            }
        )
    return {
        "phase": "eval",
        "checkpoint": str(Path(checkpoint).resolve()),
        "checkpoint_sha256": file_hash(checkpoint),
        "source_eval_plan_sha256": plan["plan_sha256"],
        "units": units,
        "validation_monitor_reads": list(EVAL_DATABASES),
        "target_databases_read": [],
        "purpose": PURPOSE,
    }


def _child(args):
    _set_limits()
    _set_cuda_limit()
    torch.set_num_threads(1)
    data_config = data_settings(read_json(args.data_config), args.data_config)
    out = Path(args.out)
    if args.phase == "reload":
        result = _reload_phase(data_config, args.plan, out)
    elif args.phase == "train":
        result = _train_phase(data_config, args.plan, args.arm, out)
    elif args.phase == "eval":
        result = _eval_phase(data_config, args.plan, args.checkpoint, out)
    else:
        raise ValueError(f"Unknown phase {args.phase}")
    _write(Path(args.result), result)


def _run_child(args, env, timeout):
    command = [sys.executable, "-B", str(Path(__file__).resolve())] + args
    process = subprocess.Popen(
        command,
        cwd=ROOT,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    deadline = time.monotonic() + timeout
    peak_rss = 0
    while process.poll() is None:
        peak_rss = max(peak_rss, _rss_kib(process.pid))
        if peak_rss > MEMORY_LIMIT // 1024:
            process.kill()
            process.communicate()
            raise RuntimeError(f"Smoke child exceeded the 12 GiB RSS cap: {peak_rss} KiB")
        if time.monotonic() > deadline:
            process.kill()
            process.communicate()
            raise TimeoutError("Smoke child exceeded its wall-clock budget")
        time.sleep(0.5)
    stdout, stderr = process.communicate()
    if process.returncode != 0:
        raise RuntimeError(
            f"Smoke child failed ({process.returncode}): {stderr[-2000:]}"
        )
    return peak_rss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-config", default="configs/data.json")
    parser.add_argument("--source-plan", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-seconds", type=int, default=DEFAULT_SECONDS)
    parser.add_argument("--phase", choices=("all", "reload", "train", "eval"))
    parser.add_argument("--arm")
    parser.add_argument("--plan")
    parser.add_argument("--checkpoint")
    parser.add_argument("--result")
    args = parser.parse_args()
    if args.phase and args.phase != "all":
        _child(args)
        return
    data_config = data_settings(read_json(args.data_config), args.data_config)
    out = Path(args.out)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("Smoke output must be a new empty directory")
    out.mkdir(parents=True, exist_ok=True)
    plans = _build_plans(data_config, out)
    env = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", "1"),
        "PYTHONDONTWRITEBYTECODE": "1",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
    }
    deadline = time.monotonic() + args.max_seconds
    results = []

    def child_args(phase, result, **extra):
        values = [
            "--phase",
            phase,
            "--data-config",
            args.data_config,
            "--source-plan",
            args.source_plan,
            "--out",
            str(out),
            "--result",
            str(result),
        ]
        for key, value in extra.items():
            if value is not None:
                values.extend([f"--{key.replace('_', '-')}", str(value)])
        return values

    reload_result = out / "reload.json"
    _run_child(
        child_args("reload", reload_result, plan=plans["reload"]),
        env,
        max(1, deadline - time.monotonic()),
    )
    results.append(json.loads(reload_result.read_text()))
    for arm in ARMS:
        result = out / "training" / f"{arm}.json"
        result.parent.mkdir(parents=True, exist_ok=True)
        _run_child(
            child_args("train", result, plan=plans["timing"], arm=arm),
            env,
            max(1, deadline - time.monotonic()),
        )
        results.append(json.loads(result.read_text()))
    eval_result = out / "eval.json"
    _run_child(
        child_args(
            "eval",
            eval_result,
            plan=args.source_plan,
            checkpoint=str(out / "training" / "C_joint" / "last.pt"),
        ),
        env,
        max(1, deadline - time.monotonic()),
    )
    results.append(json.loads(eval_result.read_text()))
    protocol = {
        "schema": 1,
        "kind": "fomaml_joint_source_smoke",
        "purpose": PURPOSE,
        "device": "cuda",
        "resource_limits": {
            "vram_bytes": VRAM_LIMIT,
            "rss_bytes": MEMORY_LIMIT,
            "output_bytes": OUTPUT_LIMIT,
            "wall_seconds": args.max_seconds,
        },
        "seed": 42,
        "results": results,
        "target_databases_read": [],
        "formal_training": False,
        "validation_monitor_reads": list(EVAL_DATABASES),
        "model": FIXED_MODEL,
    }
    _write(out / "smoke_protocol.json", protocol)
    print(json.dumps(protocol, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
