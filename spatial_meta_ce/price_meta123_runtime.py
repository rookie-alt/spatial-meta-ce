"""Isolated, stage-separated execution of experiment 123. No work on import."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import os
import re
import sys
import time
import traceback
import uuid
from contextlib import contextmanager
from pathlib import Path

PROTOCOL_SHA = "9cc0a17cf5b6b97a4544965bbde5f45685a7e9f1dcdf2284b4303e548784d1fb"
INITIAL_SHA = "dee83c462944db8c5901ac32da9c844c61df6175dcb04bc50923f96babff070f"
PLAN_FILE = "实验文档/123-原版PRICE元学习增量实验方案.md"
GIB = 1024**3
MAX_PREDICTION_ROWS = 13_558_768
SEEDS = (42, 43, 44)
LRS = (1e-7, 1e-6, 1e-5)
TRAIN_STEPS = 2000
TRAIN_BATCH = 4
FIELDS = (
    "arm",
    "stream",
    "database",
    "k",
    "run",
    "lr",
    "query_id",
    "query_hash",
    "true",
    "encoded",
)


def file_sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()
    ).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value, *, replace=False):
    """Publish complete JSON atomically; evidence cannot be overwritten by default."""
    path = Path(path)
    raw = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    if path.is_symlink() or (path.exists() and not replace):
        raise FileExistsError(f"Refusing to overwrite evidence: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with temporary.open("x") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        if replace:
            os.replace(temporary, path)
        else:
            os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def safe_root(root):
    path = Path(root)
    if not path.is_absolute() or ".." in path.parts:
        raise ValueError("Isolated root must be absolute")
    if path.is_symlink() or path.resolve() != path or not path.is_dir():
        raise ValueError("Isolated root must be a real existing directory")
    if "PRICE-master" in path.parts or path == Path("/"):
        raise ValueError("PRICE master and filesystem root are read-only")
    return path


def output_path(root, name):
    root = safe_root(root)
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_.-]*", name):
        raise ValueError("Output name must be one safe relative component")
    if name in (".", "..") or (root / "outputs").is_symlink():
        raise ValueError("Unsafe output path")
    path = root / "outputs" / name
    if path.is_symlink():
        raise ValueError("Output path must not be a symlink")
    for record in (
        path,
        path.with_name(name + ".supervisor.json"),
        path.with_name(name + ".supervisor.log"),
    ):
        if record.exists() or record.is_symlink():
            raise FileExistsError(f"Output or evidence already exists: {record}")
    return path


def code_closure(code_root):
    root = Path(code_root).resolve()
    paths = [root / PLAN_FILE, root / "vendor/PROVENANCE.json"]
    for directory in ("spatial_meta_ce", "vendor/price_baseline"):
        paths.extend(sorted((root / directory).rglob("*.py")))
    paths.append(root / "scripts/run_price_meta123.py")
    if any(not path.is_file() or path.is_symlink() for path in paths):
        raise ValueError("Execution package has missing or linked source files")
    return {str(path.relative_to(root)): file_sha(path) for path in paths}


def bind(root, data_config, checkpoint, code_root):
    """Bind source files without loading caches or constructing models."""
    from .price_meta123_plans import validate_data_config

    root = safe_root(root)
    code_root = Path(code_root).resolve()
    if code_root != root / "code":
        raise ValueError("Execution code must live in this task's code directory")
    data = read_json(data_config)
    validate_data_config(data)
    closure = code_closure(code_root)
    if closure[PLAN_FILE] != PROTOCOL_SHA:
        raise ValueError("Reviewed protocol changed")
    checkpoint = Path(checkpoint).resolve()
    if file_sha(checkpoint) != INITIAL_SHA:
        raise ValueError("Initial checkpoint differs from the frozen 23-db model")
    frozen_data = root / "inputs/data_config_123.json"
    write_json(frozen_data, data)
    value = {
        "schema": 1,
        "protocol": 123,
        "root": str(root),
        "code_root": str(code_root),
        "code_sha256": closure,
        "data_config": str(frozen_data),
        "data_config_sha256": file_sha(frozen_data),
        "initial_checkpoint": str(checkpoint),
        "initial_sha256": INITIAL_SHA,
        "protocol_sha256": PROTOCOL_SHA,
        "target_databases_read": [],
        "source_pretraining_initializations": 1,
    }
    value["binding_sha256"] = digest(value)
    write_json(root / "inputs/binding.json", value)
    return value


def load_binding(path):
    value = read_json(path)
    if value.get("schema") != 1 or value.get("protocol") != 123:
        raise ValueError("Unsupported experiment binding")
    if value.get("binding_sha256") != digest(
        {k: v for k, v in value.items() if k != "binding_sha256"}
    ):
        raise ValueError("Binding SHA mismatch")
    root = safe_root(value["root"])
    if Path(path).resolve() != root / "inputs/binding.json":
        raise ValueError("Binding must belong to its isolated root")
    if Path(value["code_root"]) != root / "code":
        raise ValueError("Execution code is outside this task root")
    if code_closure(value["code_root"]) != value["code_sha256"]:
        raise ValueError("Bound code closure changed")
    if value["protocol_sha256"] != PROTOCOL_SHA:
        raise ValueError("Protocol binding changed")
    if file_sha(value["data_config"]) != value["data_config_sha256"]:
        raise ValueError("Bound data configuration changed")
    if (
        value["initial_sha256"] != INITIAL_SHA
        or file_sha(value["initial_checkpoint"]) != INITIAL_SHA
    ):
        raise ValueError("Bound initial checkpoint changed")
    from .price_meta123_plans import validate_data_config

    validate_data_config(read_json(value["data_config"]))
    return value


def validate_authorization(auth, binding, action):
    if (
        auth.get("schema") != 1
        or auth.get("protocol") != 123
        or auth.get("authorized") is not True
        or auth.get("binding_sha256") != binding["binding_sha256"]
        or auth.get("root") != binding["root"]
        or action not in auth.get("actions", [])
        or not auth.get("granted_by")
        or not auth.get("granted_at")
    ):
        raise ValueError("Missing or mismatched action authorization")
    if "budget" not in auth:
        raise ValueError("Action authorization requires a complete resource budget")
    budget = auth["budget"]
    fields = {
        "max_wall_seconds",
        "max_rss_bytes",
        "max_vram_bytes",
        "max_output_bytes",
        "poll_seconds",
    }
    if not isinstance(budget, dict) or set(budget) != fields:
        raise ValueError("Action budget schema is incomplete")
    if any(
        type(v) not in (int, float) or not math.isfinite(v) or v < 0
        for v in budget.values()
    ):
        raise ValueError("Action budget must be finite and nonnegative")
    wall_limit = (
        1800
        if action in ("probe", "confirm-alpha")
        else (7200 if action.startswith("train:") else 16 * 3600)
    )
    if (
        not 0 < budget["max_wall_seconds"] <= wall_limit
        or not 0 < budget["max_rss_bytes"] <= 24 * GIB
        or not 0 <= budget["max_vram_bytes"] <= 12 * GIB
        or not 0
        < budget["max_output_bytes"]
        <= (512 * 1024**2 if action in ("probe", "confirm-alpha") else 6 * GIB)
        or not 0 < budget["poll_seconds"] <= 1
    ):
        raise ValueError("Action budget exceeds a protocol limit")
    return budget


def require_supervision():
    if (
        os.environ.get("PRICE_META123_SUPERVISED") != "1"
        or os.environ.get("PRICE_META123_SUPERVISOR_PID") != str(os.getppid())
        or not sys.platform.startswith("linux")
    ):
        raise ValueError("Worker requires its owning Linux supervisor")


def project_budget(timing):
    names = (
        "e_update_seconds",
        "m_update_seconds",
        "calibration_cell_seconds",
        "report_cell_seconds",
        "setup_io_seconds",
        "verification_seconds",
        "bytes_per_prediction",
        "peak_rss_bytes",
        "peak_vram_bytes",
    )
    if set(timing) != set(names) or any(
        isinstance(timing[k], bool)
        or not isinstance(timing[k], (int, float))
        or not math.isfinite(timing[k])
        or timing[k] <= 0
        for k in names
    ):
        raise ValueError("Timing needs all finite positive measured quantities")
    seconds = 1.3 * (
        timing["setup_io_seconds"]
        + 6000 * timing["e_update_seconds"]
        + 6000 * timing["m_update_seconds"]
        + 192 * timing["calibration_cell_seconds"]
        + 276 * timing["report_cell_seconds"]
        + timing["verification_seconds"]
    )
    output_bytes = MAX_PREDICTION_ROWS * timing["bytes_per_prediction"] + 512 * 1024**2
    checks = {
        "total_time": seconds <= 16 * 3600,
        "e_run_time": 1.3 * 2000 * timing["e_update_seconds"] <= 7200,
        "m_run_time": 1.3 * 2000 * timing["m_update_seconds"] <= 7200,
        "cell_time": max(
            timing["calibration_cell_seconds"], timing["report_cell_seconds"]
        )
        * 1.3
        <= 120,
        "rss": timing["peak_rss_bytes"] <= 24 * GIB,
        "vram": timing["peak_vram_bytes"] <= 12 * GIB,
        "output": output_bytes <= 6 * GIB,
    }
    return {
        "projected_seconds": seconds,
        "projected_output_bytes": output_bytes,
        "max_prediction_rows": MAX_PREDICTION_ROWS,
        "checks": checks,
        "feasible": all(checks.values()),
        "purpose": "cost_projection_only_not_execution_authorization",
    }


def _finish(output, binding, action, **values):
    value = {
        "schema": 1,
        "protocol": 123,
        "status": "completed",
        "action": action,
        "binding_sha256": binding["binding_sha256"],
        "target_databases_read": [],
        **values,
    }
    write_json(Path(output) / "completion.json", value)
    return value


def _completed(binding, name, action=None):
    path = Path(binding["root"]) / "outputs" / name
    result = read_json(path / "completion.json")
    record = read_json(path.with_name(path.name + ".supervisor.json"))
    if (
        result.get("status") != "completed"
        or result.get("binding_sha256") != binding["binding_sha256"]
        or record.get("status") != "completed"
        or record.get("completion") != result
        or (action is not None and result.get("action") != action)
    ):
        raise ValueError(f"Stage is not sealed and supervised: {name}")
    return path, result


def _torch_setup(device):
    import torch

    torch.set_num_threads(1)
    if device == "cuda":
        if not os.environ.get("CUDA_VISIBLE_DEVICES", "").isdigit():
            raise ValueError("Exactly one physical GPU must be selected")
        if not torch.cuda.is_available():
            raise ValueError("Authorized CUDA device unavailable")
    elif device != "cpu":
        raise ValueError("Device must be cpu or cuda")
    return torch


def _progress(output, stage, **values):
    write_json(Path(output) / "progress.json", {"stage": stage, **values}, replace=True)


def _run_plan_stage(binding, output, kind):
    from .price_meta123_plans import make_eval_plan, make_training_plan

    data = read_json(binding["data_config"])
    if kind == "prepare-training":
        files = {}
        for seed in SEEDS:
            plan = make_training_plan(data, seed)
            path = Path(output) / f"training-{seed}.json"
            write_json(path, plan)
            files[str(seed)] = {"file": path.name, "sha256": file_sha(path)}
        return _finish(output, binding, kind, plans=files)
    plan = make_eval_plan(data)
    path = Path(output) / "source-plan.json"
    write_json(path, plan)
    return _finish(
        output, binding, kind, plan_file=path.name, plan_sha256=file_sha(path)
    )


def _plan(binding, kind, seed=None):
    from .price_meta123_plans import validate_plan

    name = "prepare-training" if kind == "training" else "prepare-source"
    path, completion = _completed(binding, name, name)
    if kind == "training":
        record = completion["plans"][str(seed)]
        plan_path, expected = path / record["file"], record["sha256"]
    else:
        plan_path = path / completion["plan_file"]
        expected = completion["plan_sha256"]
    if file_sha(plan_path) != expected:
        raise ValueError("Sealed query plan changed")
    value = read_json(plan_path)
    validate_plan(value, read_json(binding["data_config"]), kind)
    return value


def _calibration(binding):
    path, completed = _completed(binding, "calibrate-p", "calibrate-p")
    if file_sha(path / "calibration.json") != completed["calibration_sha256"]:
        raise ValueError("Calibration changed")
    result = read_json(path / "calibration.json")
    if result["lr"] not in LRS:
        raise ValueError("Calibration chose an unregistered LR")
    return result["lr"]


def _cost_gate(binding, action):
    if action in ("prepare-training", "probe"):
        return
    path, record = _completed(binding, "probe", "probe")
    if file_sha(path / "timing.json") != record["timing_sha256"]:
        raise ValueError("Sealed cost timing changed")
    timing = read_json(path / "timing.json")
    if not project_budget(timing)["feasible"]:
        raise ValueError("Measured full experiment cost exceeds the design budget")
    if action.startswith("train:") and _calibration(binding) != 1e-6:
        _completed(binding, "confirm-alpha", "confirm-alpha")


@contextmanager
def _cell_deadline(seconds=120):
    import signal

    def timed_out(_signum, _frame):
        raise TimeoutError("Evaluation cell exceeded its 120-second hard deadline")

    previous = signal.signal(signal.SIGALRM, timed_out)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def _experiment_usage(binding):
    root = Path(binding["root"]) / "outputs"
    total_seconds = 0.0
    total_bytes = 0
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError("Linked experiment output is forbidden")
        if path.is_file():
            total_bytes += path.stat().st_size
        if path.name.endswith(".supervisor.json"):
            record = read_json(path)
            if not path.name.startswith(("probe.", "prepare-training.")):
                elapsed = record.get("elapsed_seconds", record.get("elapsed", 0))
                if (
                    type(elapsed) not in (int, float)
                    or not math.isfinite(elapsed)
                    or elapsed < 0
                ):
                    raise ValueError("Invalid cumulative resource evidence")
                total_seconds += elapsed
    if total_seconds > 16 * 3600 or total_bytes > 6 * GIB:
        raise ValueError("Cumulative experiment budget exhausted")
    return total_seconds, total_bytes


def _probe(binding, output, device, *, confirm=False):
    """Training databases only: closure, optimizer round-trip and bounded timing."""
    from .price_meta123_model import (
        adapt_native,
        build_model,
        erm_loss,
        fomaml_loss,
        predict_native,
    )
    from .price_meta123_plans import SourceStores, verify_plan_stores

    torch = _torch_setup(device)
    if device != "cuda":
        raise ValueError("Actual experiment cost probes require one authorized GPU")
    plan = _plan(binding, "training", 42)
    data = read_json(binding["data_config"])
    setup_started = time.monotonic()
    stores = SourceStores(data, tuple(plan["databases"]))
    verify_plan_stores(plan, stores)
    alpha = _calibration(binding) if confirm else 1e-6
    timings = {}
    peak_vram = 0
    for arm in ("E", "M"):
        torch.manual_seed(42)
        model = build_model(binding["initial_checkpoint"], device=device)
        optimizer = torch.optim.Adam(model.parameters(), lr=2.85e-5, weight_decay=5e-5)
        torch.cuda.reset_peak_memory_stats()
        update_seconds = []
        # Two closure updates are timed separately from the 50 measured updates.
        for step in range(2 if confirm else 52):
            torch.cuda.synchronize()
            started = time.monotonic()
            optimizer.zero_grad(set_to_none=True)
            for task in plan["episodes"][step * 4 : (step + 1) * 4]:
                store = stores.get(task["db"])
                support, query = task["support"]["indices"], task["query"]["indices"]
                if arm == "E":
                    loss = erm_loss(model, store.batch(support + query, device))
                else:
                    loss, _ = fomaml_loss(
                        model,
                        store.batch(support, device),
                        store.batch(query, device),
                        alpha,
                    )
                (loss / 4).backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not torch.isfinite(norm):
                raise FloatingPointError("Nonfinite probe gradient")
            optimizer.step()
            torch.cuda.synchronize()
            elapsed = time.monotonic() - started
            if step >= 2:
                update_seconds.append(elapsed)
            _progress(output, "cost-probe", arm=arm, update=step + 1)
        path = Path(output) / f"probe-{arm}.pt"
        torch.save(
            {"model": model.price.state_dict(), "optimizer": optimizer.state_dict()},
            path,
        )
        saved = torch.load(path, map_location=device, weights_only=False)
        if any(
            not torch.equal(saved["model"][k], v)
            for k, v in model.price.state_dict().items()
        ):
            raise ValueError("Probe checkpoint round-trip failed")
        steps = {int(state["step"]) for state in saved["optimizer"]["state"].values()}
        if steps != {2 if confirm else 52}:
            raise ValueError("Probe optimizer step evidence disagrees")
        peak_vram = max(peak_vram, torch.cuda.max_memory_allocated())
        if update_seconds:
            timings[arm] = max(update_seconds)
        del optimizer, saved, model
    if confirm:
        return _finish(
            output,
            binding,
            "confirm-alpha",
            alpha=alpha,
            updates_per_arm=2,
            opened_databases=stores.opened_databases,
        )

    # Use an actual long-token training database, never source report errors.
    database = max(
        plan["databases"], key=lambda d: sum(plan["databases"][d]["max_counts"])
    )
    store = stores.get(database)
    indices = store.unique_indices().tolist()
    if len(indices) < 2100:
        raise ValueError("Training cost sample is too small for report projection")
    model = build_model(binding["initial_checkpoint"], device=device)
    support = store.batch(indices[:20], device)
    # Keep the observed setup/I/O interval conservative: it also includes the
    # short timing runs, rather than subtracting unrelated max-update estimates.
    setup_seconds = time.monotonic() - setup_started
    cell_times, written_rows, written_bytes = {}, 0, 0
    for population, selected in (
        ("calibration", indices[100:2100]),
        ("report", indices[100:]),
    ):
        torch.cuda.synchronize()
        started = time.monotonic()
        params, _ = adapt_native(model, support, alpha, meta=False)
        prediction_file = Path(output) / f"probe-{population}.csv.gz"
        with gzip.open(prediction_file, "xt", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(("query_id", "query_hash", "true", "encoded"))
            for offset in range(0, len(selected), 128):
                batch_indices = selected[offset : offset + 128]
                batch = store.batch(batch_indices, device)
                with torch.no_grad():
                    prediction = (
                        predict_native(model, batch, params)
                        .detach()
                        .cpu()
                        .double()
                        .tolist()
                    )
                for index, encoded in zip(batch_indices, prediction):
                    if not math.isfinite(encoded):
                        raise FloatingPointError("Nonfinite cost-probe prediction")
                    row = store.rows[index]
                    writer.writerow(
                        (
                            row["id"],
                            row["query_hash"],
                            repr(float(store.cards[index, 0])),
                            repr(encoded),
                        )
                    )
        torch.cuda.synchronize()
        cell_times[population] = time.monotonic() - started
        written_rows += len(selected)
        written_bytes += prediction_file.stat().st_size
    import resource

    timing = {
        "e_update_seconds": timings["E"],
        "m_update_seconds": timings["M"],
        "calibration_cell_seconds": cell_times["calibration"],
        "report_cell_seconds": cell_times["report"]
        * max(1, 47900 / (len(indices) - 100)),
        "setup_io_seconds": max(setup_seconds, 1.0),
        "verification_seconds": max(cell_times["report"] * 2, 60.0),
        "bytes_per_prediction": written_bytes / written_rows + 64,
        "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        "peak_vram_bytes": max(peak_vram, torch.cuda.max_memory_allocated()),
    }
    write_json(Path(output) / "timing.json", timing)
    write_json(Path(output) / "budget_projection.json", project_budget(timing))
    return _finish(
        output,
        binding,
        "probe",
        timing_sha256=file_sha(Path(output) / "timing.json"),
        opened_databases=stores.opened_databases,
        probe_alpha=alpha,
        updates_per_arm=52,
        research_use_forbidden=True,
    )


def _train(binding, output, arm, seed, device):
    from .price_meta123_model import build_model, erm_loss, fomaml_loss
    from .price_meta123_plans import SourceStores, verify_plan_stores

    torch = _torch_setup(device)
    if arm not in ("E", "M") or seed not in SEEDS:
        raise ValueError("Unknown training arm or stream")
    alpha = _calibration(binding)
    plan = _plan(binding, "training", seed)
    stores = SourceStores(read_json(binding["data_config"]), tuple(plan["databases"]))
    verify_plan_stores(plan, stores)
    torch.manual_seed(seed)
    model = build_model(binding["initial_checkpoint"], device=device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=2.85e-5, weight_decay=5e-5, betas=(0.9, 0.999), eps=1e-8
    )
    started = time.monotonic()
    clipped_count, presentations = 0, 0
    gradient_norms = []
    inner_audits = 0
    audit_presentations = 0
    with (Path(output) / "loss_audit.jsonl").open("x") as log:
        for step in range(TRAIN_STEPS):
            optimizer.zero_grad(set_to_none=True)
            loss_sum = 0.0
            for episode in plan["episodes"][
                step * TRAIN_BATCH : (step + 1) * TRAIN_BATCH
            ]:
                store = stores.get(episode["db"])
                support = episode["support"]["indices"]
                query = episode["query"]["indices"]
                if arm == "E":
                    loss = erm_loss(model, store.batch(support + query, device))
                    presentations += 84
                else:
                    loss, _audit = fomaml_loss(
                        model,
                        store.batch(support, device),
                        store.batch(query, device),
                        alpha,
                    )
                    presentations += 164
                    audit_presentations += (
                        _audit["support_size"] * _audit["audit_extra_forward_count"]
                    )
                    # Retain the full inner diagnostics for a fixed audit subset;
                    # this subset never affects optimization or model selection.
                    if step < 2:
                        write_json(
                            Path(output) / f"inner-{step}-{inner_audits}.json", _audit
                        )
                        inner_audits += 1
                if not torch.isfinite(loss).item():
                    raise FloatingPointError("Nonfinite outer loss")
                (loss / TRAIN_BATCH).backward()
                loss_sum += float(loss.detach()) / TRAIN_BATCH
            for parameter in model.parameters():
                if parameter.grad is None or not torch.isfinite(parameter.grad).all():
                    raise FloatingPointError("Missing or nonfinite outer gradient")
            norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
            if not math.isfinite(norm):
                raise FloatingPointError("Nonfinite outer gradient norm")
            clipped_count += norm > 1
            gradient_norms.append(norm)
            optimizer.step()
            if any(not torch.isfinite(p).all() for p in model.parameters()):
                raise FloatingPointError("Nonfinite state after optimizer update")
            log.write(
                json.dumps(
                    {
                        "update": step + 1,
                        "loss": loss_sum,
                        "gradient_norm_before_clip": norm,
                    }
                )
                + "\n"
            )
            if step % 25 == 0 or step + 1 == TRAIN_STEPS:
                log.flush()
                _progress(
                    output,
                    "training",
                    arm=arm,
                    stream=seed,
                    updates=step + 1,
                    elapsed_seconds=time.monotonic() - started,
                )
    checkpoint = Path(output) / "last.pt"
    torch.save(model.price.state_dict(), checkpoint)
    torch.save(
        {
            "optimizer": optimizer.state_dict(),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if device == "cuda" else [],
            "updates": TRAIN_STEPS,
            "resume_forbidden": True,
        },
        Path(output) / "training_state.pt",
    )
    return _finish(
        output,
        binding,
        f"train:{arm}:{seed}",
        arm=arm,
        stream=seed,
        updates=TRAIN_STEPS,
        initial_sha256=INITIAL_SHA,
        alpha=alpha,
        checkpoint_sha256=file_sha(checkpoint),
        presentations=presentations,
        audit_forward_presentations=audit_presentations,
        total_forward_presentations=presentations + audit_presentations,
        training_plan_sha256=plan["plan_sha256"],
        opened_databases=stores.opened_databases,
        validation_monitor_reads=[],
        clipped_count=int(clipped_count),
        gradient_norm_mean=sum(gradient_norms) / len(gradient_norms),
        gradient_norm_max=max(gradient_norms),
        loss_audit_sha256=file_sha(Path(output) / "loss_audit.jsonl"),
        training_state_sha256=file_sha(Path(output) / "training_state.pt"),
        wall_seconds=time.monotonic() - started,
    )


def _seal_training(binding, output):
    alpha = _calibration(binding)
    records = {}
    for arm in ("E", "M"):
        for seed in SEEDS:
            path, value = _completed(
                binding, f"train-{arm}-{seed}", f"train:{arm}:{seed}"
            )
            if (
                value.get("updates") != TRAIN_STEPS
                or value.get("arm") != arm
                or value.get("stream") != seed
                or value.get("alpha") != alpha
                or value.get("initial_sha256") != INITIAL_SHA
                or value.get("validation_monitor_reads") != []
                or value.get("opened_databases")
                != list(_plan(binding, "training", seed)["databases"])
                or value.get("training_plan_sha256")
                != _plan(binding, "training", seed)["plan_sha256"]
                or file_sha(path / "last.pt") != value.get("checkpoint_sha256")
            ):
                raise ValueError("Incomplete or mismatched training matrix")
            records[f"{arm}-{seed}"] = {
                "checkpoint_sha256": value["checkpoint_sha256"],
                "completion_sha256": file_sha(path / "completion.json"),
            }
    return _finish(output, binding, "seal-training", runs=records, alpha=alpha)


def _model_path(binding, arm, seed):
    if arm == "P":
        return binding["initial_checkpoint"], INITIAL_SHA
    _, seal = _completed(binding, "seal-training", "seal-training")
    path, record = _completed(binding, f"train-{arm}-{seed}", f"train:{arm}:{seed}")
    bound = seal["runs"][f"{arm}-{seed}"]
    if file_sha(path / "completion.json") != bound["completion_sha256"]:
        raise ValueError("Training completion changed after sealing")
    if record["checkpoint_sha256"] != bound["checkpoint_sha256"]:
        raise ValueError("Checkpoint changed after sealing")
    return path / "last.pt", record["checkpoint_sha256"]


def _source_context(binding):
    from .price_meta123_plans import SourceStores, SOURCE_DATABASES, verify_plan_stores

    plan = _plan(binding, "eval")
    stores = SourceStores(read_json(binding["data_config"]), SOURCE_DATABASES)
    verify_plan_stores(plan, stores)
    return plan, stores


def _unit(output, model, store, entry, *, arm, seed, k, run, alpha, region):
    """Write every query, then independently re-read and verify its metrics."""
    from .price_meta123_model import adapt_native, predict_native
    from .price_meta123_results import compute_unit

    import numpy as np
    import torch

    indices = entry[region]["indices"]
    device = next(model.parameters()).device
    audit = {}
    params = None
    started = time.monotonic()
    if k:
        support = entry["support_blocks"][run]["indices"][:k]
        params, audit = adapt_native(
            model, store.batch(support, device), alpha, meta=False
        )
    path = Path(output) / f"{arm}-{seed}-{store.path.name}-{k}-{run}-{alpha:.0e}.csv.gz"
    expected = []
    encoded_all, true_all = [], []
    model.eval()
    with gzip.open(path, "xt", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        for start in range(0, len(indices), 128):
            selected = indices[start : start + 128]
            batch = store.batch(selected, device)
            with torch.no_grad():
                encoded = (
                    predict_native(model, batch, params).detach().cpu().double().numpy()
                )
            if not np.isfinite(encoded).all():
                raise FloatingPointError("Nonfinite raw encoded prediction")
            for index, z in zip(selected, encoded):
                row = store.rows[index]
                truth = float(store.cards[index, 0])
                writer.writerow(
                    {
                        "arm": arm,
                        "stream": seed,
                        "database": store.path.name,
                        "k": k,
                        "run": run,
                        "lr": alpha,
                        "query_id": row["id"],
                        "query_hash": row["query_hash"],
                        "true": repr(truth),
                        "encoded": repr(float(z)),
                    }
                )
                expected.append((row["id"], row["query_hash"], truth))
                true_all.append(truth)
                encoded_all.append(float(z))
    independent = compute_unit(path, expected, population=region)
    from .baselines import price_log_predictions
    from .metrics import summarize

    online = summarize(price_log_predictions(encoded_all), true_all)
    metrics = independent["metrics"]
    for key in ("q50", "q90", "q95", "q99", "log_q95"):
        if not math.isclose(online[key], metrics[key], rel_tol=1e-10, abs_tol=1e-10):
            raise ValueError("Independent prediction metric verification failed")
    return {
        **independent,
        "prediction_file": path.name,
        "prediction_sha256": file_sha(path),
        "adaptation": audit,
        "seconds": time.monotonic() - started,
    }


def _evaluate(binding, output, stage, device):
    from .price_meta123_model import build_model
    from .price_meta123_results import aggregate_report, select_lr, make_failed_unit
    from .price_meta123_plans import SOURCE_DATABASES

    _torch_setup(device)
    plan, stores = _source_context(binding)
    if stage != "calibrate-p":
        _completed(binding, "seal-training", "seal-training")
        alpha = _calibration(binding)
    else:
        alpha = None
    if stage == "report":
        path, record = _completed(binding, "calibrate-e", "calibrate-e")
        if file_sha(path / "calibration.json") != record["calibration_sha256"]:
            raise ValueError("Strong baseline calibration changed")
        alpha_e = read_json(path / "calibration.json")["lr"]
    units = []
    models = (
        [("P", 0)]
        if stage == "calibrate-p"
        else (
            [("E", s) for s in SEEDS]
            if stage == "calibrate-e"
            else [("P", 0)] + [(a, s) for a in ("E", "M") for s in SEEDS]
        )
    )
    region = "report" if stage == "report" else "calibration"
    for arm, seed in models:
        checkpoint, sha = _model_path(binding, arm, seed)
        model = build_model(checkpoint, device=device, expected_sha=sha)
        for db in SOURCE_DATABASES:
            entry, store = plan["databases"][db], stores.get(db)
            cell_specs = [(0, 0, 0.0, arm)]
            if stage == "report":
                cell_specs += [(k, r, alpha, arm) for k in (5, 20) for r in range(5)]
                if arm == "E" and alpha_e != alpha:
                    cell_specs += [(20, r, alpha_e, "E_alt") for r in range(5)]
            else:
                cell_specs += [(20, r, lr, arm) for lr in LRS for r in range(5)]
            for k, run, lr, unit_arm in cell_specs:
                try:
                    with _cell_deadline():
                        value = _unit(
                            output,
                            model,
                            store,
                            entry,
                            arm=unit_arm,
                            seed=seed,
                            k=k,
                            run=run,
                            alpha=lr,
                            region=region,
                        )
                except FloatingPointError as exc:
                    if stage == "report" or k == 0:
                        raise
                    value = make_failed_unit(
                        unit_arm,
                        seed,
                        db,
                        k,
                        run,
                        lr,
                        region,
                        len(entry[region]["indices"]),
                        str(exc),
                    )
                units.append(value)
                _experiment_usage(binding)
                _progress(
                    output, stage, units=len(units), arm=arm, stream=seed, database=db
                )
        del model
    write_json(Path(output) / "units.json", units)
    if stage == "report":
        denominators = {
            db: entry["counts"]["report"] for db, entry in plan["databases"].items()
        }
        summary = aggregate_report(units, alpha, alpha_e, denominators=denominators)
        paired = _paired_report(output, units, alpha, alpha_e)
        write_json(Path(output) / "paired_query_diagnostics.json", paired)
        write_json(Path(output) / "decision.json", summary)
        return _finish(
            output,
            binding,
            stage,
            units=len(units),
            units_sha256=file_sha(Path(output) / "units.json"),
            decision_sha256=file_sha(Path(output) / "decision.json"),
            paired_diagnostics_sha256=file_sha(
                Path(output) / "paired_query_diagnostics.json"
            ),
            opened_databases=stores.opened_databases,
        )
    calibration = select_lr(units, "P" if stage == "calibrate-p" else "E")
    write_json(Path(output) / "calibration.json", calibration)
    return _finish(
        output,
        binding,
        stage,
        units=len(units),
        units_sha256=file_sha(Path(output) / "units.json"),
        calibration_sha256=file_sha(Path(output) / "calibration.json"),
        opened_databases=stores.opened_databases,
    )


def _paired_report(output, units, alpha, alpha_e):
    from .price_meta123_results import compare_predictions

    def key(u):
        return (u["arm"], u["stream"], u["database"], u["k"], u["run"], u["lr"])

    indexed = {key(u): u for u in units}
    diagnostics = {}
    for unit in units:
        arm, seed, db, k, run, lr = key(unit)
        if k:
            base_arm = "E" if arm == "E_alt" else arm
            references = {"self_k0": indexed[(base_arm, seed, db, 0, 0, 0.0)]}
        else:
            references = {}
        if arm == "M" and k == 20:
            strong_arm, strong_lr = (
                ("E", alpha) if alpha_e == alpha else ("E_alt", alpha_e)
            )
            references.update(
                {
                    "E20": indexed[("E", seed, db, 20, run, alpha)],
                    "E_strong20": indexed[(strong_arm, seed, db, 20, run, strong_lr)],
                    "P20": indexed[("P", 0, db, 20, run, alpha)],
                    "E0": indexed[("E", seed, db, 0, 0, 0.0)],
                    "P0": indexed[("P", 0, db, 0, 0, 0.0)],
                }
            )
        for label, reference in references.items():
            diagnostics[f"{arm}/{seed}/{db}/{k}/{run}/{label}"] = compare_predictions(
                Path(output) / unit["prediction_file"],
                Path(output) / reference["prediction_file"],
            )
    return diagnostics


def worker(binding_path, authorization_path, action, output, device):
    require_supervision()
    binding = load_binding(binding_path)
    validate_authorization(read_json(authorization_path), binding, action)
    output = Path(output)
    expected = Path(binding["root"]) / "outputs" / action.replace(":", "-")
    if output != expected or output.is_symlink() or output.exists():
        raise ValueError("Worker output is not its new isolated stage path")
    output.mkdir()
    try:
        _cost_gate(binding, action)
        if action in ("probe", "confirm-alpha"):
            return _probe(binding, output, device, confirm=action == "confirm-alpha")
        if action in ("prepare-training", "prepare-source"):
            return _run_plan_stage(binding, output, action)
        if action.startswith("train:"):
            _, arm, seed = action.split(":")
            return _train(binding, output, arm, int(seed), device)
        if action == "seal-training":
            return _seal_training(binding, output)
        if action in ("calibrate-p", "calibrate-e", "report"):
            return _evaluate(binding, output, action, device)
        raise ValueError("Unknown stage action")
    except BaseException as exc:
        write_json(
            output / "failure.json",
            {
                "action": action,
                "type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
                "target_databases_read": [],
            },
        )
        raise


def run_stage(binding_path, authorization_path, action, device):
    from .price_meta123_supervisor import supervise

    binding = load_binding(binding_path)
    auth = read_json(authorization_path)
    budget = dict(validate_authorization(auth, binding, action))
    output = output_path(binding["root"], action.replace(":", "-"))
    used_seconds, used_bytes = _experiment_usage(binding)
    if action not in ("probe", "prepare-training"):
        budget["max_wall_seconds"] = min(
            budget["max_wall_seconds"], 16 * 3600 - used_seconds
        )
        budget["max_output_bytes"] = min(
            budget["max_output_bytes"], 6 * GIB - used_bytes
        )
    if device == "cuda":
        gpu = auth.get("cuda_device")
        if type(gpu) is not int or not 0 <= gpu <= 7:
            raise ValueError("CUDA execution requires one authorized GPU index")
        env = {"CUDA_VISIBLE_DEVICES": str(gpu)}
    elif device == "cpu":
        if budget["max_vram_bytes"] != 0:
            raise ValueError("CPU action must reserve zero GPU bytes")
        env = {"CUDA_VISIBLE_DEVICES": ""}
    else:
        raise ValueError("Unknown execution device")
    for name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        env[name] = "1"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    import fcntl

    lock_path = Path(binding["root"]) / "inputs/execution.lock"
    if lock_path.is_symlink():
        raise ValueError("Execution lock must not be linked")
    with lock_path.open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError("Another stage already owns this experiment root") from exc
        result = supervise(
            [
                sys.executable,
                "-B",
                str(Path(binding["code_root"]) / "scripts/run_price_meta123.py"),
                "worker",
                "--binding",
                str(Path(binding_path).resolve()),
                "--authorization",
                str(Path(authorization_path).resolve()),
                "--action",
                action,
                "--output",
                str(output),
                "--device",
                device,
            ],
            output,
            cwd=Path(binding["code_root"]),
            budget=budget,
            env=env,
        )
        return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    binder = sub.add_parser(
        "bind", help="bind code/checkpoint/config without computation"
    )
    binder.add_argument("--root", required=True)
    binder.add_argument("--data-config", required=True)
    binder.add_argument("--checkpoint", required=True)
    for command in ("run", "worker"):
        stage = sub.add_parser(command)
        stage.add_argument("--binding", required=True)
        stage.add_argument("--authorization", required=True)
        stage.add_argument(
            "--action",
            required=True,
            choices=[
                "probe",
                "confirm-alpha",
                "prepare-training",
                "prepare-source",
                "calibrate-p",
                "seal-training",
                "calibrate-e",
                "report",
            ]
            + [f"train:{a}:{s}" for a in ("E", "M") for s in SEEDS],
        )
        stage.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
        if command == "worker":
            stage.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    if args.command == "bind":
        value = bind(
            args.root,
            args.data_config,
            args.checkpoint,
            Path(__file__).resolve().parents[1],
        )
    elif args.command == "run":
        value = run_stage(args.binding, args.authorization, args.action, args.device)
    else:
        value = worker(
            args.binding, args.authorization, args.action, args.output, args.device
        )
    print(json.dumps(value, ensure_ascii=False, allow_nan=False))
    return 0 if value.get("status", "completed") == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
