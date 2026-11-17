"""Bounded official26 PRICE continuation without validation or target selection."""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
import traceback
from pathlib import Path

from .price_meta123_runtime import (
    _torch_setup,
    digest,
    file_sha,
    output_path,
    read_json,
    require_supervision,
    safe_root,
    write_json,
)

INITIAL_SHA = "4f889c4df269d1bca5a50dadd83de7c801a607fe84228b6c21fb3371ae053f92"
PROTOCOL_FILE = "实验文档/125-官方26库划分的PRICE元学习实验.md"
PROTOCOL_SHA = "354f1d651b1617beb337468dccd73648e7c101d0564f98a04553e9528e898f92"
ALPHA = 1e-6
SEEDS = (42, 43, 44)
STEPS = 2000
META_BATCH = 4
GIB = 1024**3
ACTIONS = ("prepare-training", "probe", "seal-training") + tuple(
    f"train:{arm}:{seed}" for arm in ("E", "M") for seed in SEEDS
)


def code_closure(code_root):
    root = Path(code_root)
    paths = [root / PROTOCOL_FILE, root / "vendor/PROVENANCE.json"]
    for directory in ("spatial_meta_ce", "vendor/price_baseline"):
        paths.extend(sorted((root / directory).rglob("*.py")))
    paths.append(root / "scripts/run_price_meta125.py")
    if any(not p.is_file() or p.is_symlink() for p in paths):
        raise ValueError("Missing or linked execution dependency")
    return {str(p.relative_to(root)): file_sha(p) for p in paths}


def bind(root, data_config, checkpoint, code_root):
    from .price_meta125_plans import validate_data_config

    root = safe_root(root)
    if Path(code_root).resolve() != root / "code":
        raise ValueError("Code must live in this isolated root")
    config = read_json(data_config)
    validate_data_config(config)
    closure = code_closure(root / "code")
    if closure[PROTOCOL_FILE] != PROTOCOL_SHA:
        raise ValueError("Reviewed protocol changed")
    checkpoint = Path(checkpoint).resolve()
    if file_sha(checkpoint) != INITIAL_SHA:
        raise ValueError("Official PRICE checkpoint SHA mismatch")
    config_path = root / "inputs/data_config_125.json"
    write_json(config_path, config)
    value = {
        "schema": 1,
        "protocol": 125,
        "root": str(root),
        "code_root": str(root / "code"),
        "code_sha256": closure,
        "protocol_sha256": PROTOCOL_SHA,
        "data_config": str(config_path),
        "data_config_sha256": file_sha(config_path),
        "initial_checkpoint": str(checkpoint),
        "initial_sha256": INITIAL_SHA,
        "alpha": ALPHA,
        "validation_databases": [],
        "target_databases_read": [],
    }
    value["binding_sha256"] = digest(value)
    write_json(root / "inputs/binding.json", value)
    return value


def load_binding(path):
    from .price_meta125_plans import validate_data_config

    value = read_json(path)
    if value.get("schema") != 1 or value.get("protocol") != 125:
        raise ValueError("Not an official26 experiment125 binding")
    if value.get("binding_sha256") != digest(
        {k: v for k, v in value.items() if k != "binding_sha256"}
    ):
        raise ValueError("Binding SHA mismatch")
    root = safe_root(value["root"])
    if (
        Path(path).resolve() != root / "inputs/binding.json"
        or value.get("code_root") != str(root / "code")
        or value.get("data_config") != str(root / "inputs/data_config_125.json")
    ):
        raise ValueError("Binding is outside its isolated root")
    if (
        value.get("protocol_sha256") != PROTOCOL_SHA
        or value.get("initial_sha256") != INITIAL_SHA
        or value.get("alpha") != ALPHA
        or value.get("validation_databases") != []
        or value.get("target_databases_read") != []
        or code_closure(root / "code") != value["code_sha256"]
        or value["code_sha256"].get(PROTOCOL_FILE) != PROTOCOL_SHA
        or file_sha(value["initial_checkpoint"]) != INITIAL_SHA
        or file_sha(value["data_config"]) != value["data_config_sha256"]
    ):
        raise ValueError("Bound code, configuration or official initialization changed")
    validate_data_config(read_json(value["data_config"]))
    return value


def validate_authorization(auth, binding, action):
    if (
        action not in ACTIONS
        or auth.get("schema") != 1
        or auth.get("protocol") != 125
        or auth.get("authorized") is not True
        or auth.get("root") != binding["root"]
        or auth.get("binding_sha256") != binding["binding_sha256"]
        or auth.get("actions") != [action]
        or not auth.get("granted_by")
        or not auth.get("granted_at")
    ):
        raise ValueError("Missing or mismatched stage authorization")
    budget = auth.get("budget")
    fields = {
        "max_wall_seconds",
        "max_rss_bytes",
        "max_vram_bytes",
        "max_output_bytes",
        "poll_seconds",
    }
    if (
        not isinstance(budget, dict)
        or set(budget) != fields
        or any(
            type(v) not in (int, float) or not math.isfinite(v) or v < 0
            for v in budget.values()
        )
    ):
        raise ValueError("Invalid resource budget")
    wall = (
        600
        if action in ("prepare-training", "seal-training")
        else 1800
        if action == "probe"
        else 21600
        if action.startswith("train:M:")
        else 7200
    )
    if (
        not 0 < budget["max_wall_seconds"] <= wall
        or not 0 < budget["max_rss_bytes"] <= 24 * GIB
        or not 0 <= budget["max_vram_bytes"] <= 12 * GIB
        or not 0
        < budget["max_output_bytes"]
        <= (512 * 1024**2 if action in ("prepare-training", "probe") else 6 * GIB)
        or not 0 < budget["poll_seconds"] <= 1
    ):
        raise ValueError("Stage exceeds its frozen budget")
    return budget


def completed(binding, name):
    path = Path(binding["root"]) / "outputs" / name
    value = read_json(path / "completion.json")
    record = read_json(path.with_name(name + ".supervisor.json"))
    if (
        value.get("protocol") != 125
        or value.get("status") != "completed"
        or value.get("binding_sha256") != binding["binding_sha256"]
        or value.get("target_databases_read") != []
        or value.get("action", "").replace(":", "-") != name
        or record.get("status") != "completed"
        or record.get("completion") != value
    ):
        raise ValueError("Stage is not sealed under this binding")
    return path, value


def finish(binding, output, action, **values):
    value = {
        "schema": 1,
        "protocol": 125,
        "status": "completed",
        "action": action,
        "binding_sha256": binding["binding_sha256"],
        "target_databases_read": [],
        "alpha": ALPHA,
        **values,
    }
    write_json(Path(output) / "completion.json", value)
    return value


def project_budget(timing):
    fields = {
        "e_update_seconds",
        "m_update_seconds",
        "setup_io_seconds",
        "verification_seconds",
        "checkpoint_pair_bytes",
        "peak_rss_bytes",
        "peak_vram_bytes",
    }
    if set(timing) != fields or any(
        type(v) not in (int, float) or not math.isfinite(v) or v <= 0
        for v in timing.values()
    ):
        raise ValueError("All measured probe quantities must be finite and positive")
    seconds = 1.3 * (
        timing["setup_io_seconds"]
        + 6000 * timing["e_update_seconds"]
        + 6000 * timing["m_update_seconds"]
        + timing["verification_seconds"]
    )
    output = 3 * timing["checkpoint_pair_bytes"] + 512 * 1024**2
    checks = {
        "total_time": seconds <= 86400,
        "e_run_time": 1.3 * 2000 * timing["e_update_seconds"] <= 7200,
        "m_run_time": 1.3 * 2000 * timing["m_update_seconds"] <= 21600,
        "rss": timing["peak_rss_bytes"] <= 24 * GIB,
        "vram": timing["peak_vram_bytes"] <= 12 * GIB,
        "output": output <= 6 * GIB,
    }
    return {
        "projected_seconds": seconds,
        "projected_output_bytes": output,
        "checks": checks,
        "feasible": all(checks.values()),
        "scope": "six_training_runs_and_seal_only_no_evaluation",
    }


def cost_gate(binding):
    path, value = completed(binding, "probe")
    if file_sha(path / "timing.json") != value["timing_sha256"]:
        raise ValueError("Probe timing changed")
    sup = read_json(path.with_name("probe.supervisor.json"))
    peak = sup.get("peak_vram_bytes", 0)
    if (
        sup.get("vram_accounting") != "device_total"
        or not sup.get("gpu_uuid")
        or not 16 * 1024**2 < peak <= 12 * GIB
        or peak <= sup.get("vram_baseline_bytes", peak)
    ):
        raise ValueError("External GPU VRAM measurement is not certified")
    if not project_budget(read_json(path / "timing.json"))["feasible"]:
        raise ValueError("Measured training matrix exceeds budget")


def usage(binding):
    seconds = 0.0
    output_bytes = 0
    for path in (Path(binding["root"]) / "outputs").rglob("*"):
        if path.is_symlink():
            raise ValueError("Linked output is forbidden")
        if path.is_file():
            output_bytes += path.stat().st_size
        if path.name.endswith(".supervisor.json") and not path.name.startswith(
            ("prepare-training.", "probe.")
        ):
            elapsed = read_json(path).get("elapsed_seconds")
            if (
                type(elapsed) not in (int, float)
                or not math.isfinite(elapsed)
                or elapsed < 0
            ):
                raise ValueError("Invalid cumulative resource record")
            seconds += elapsed
    if seconds >= 86400 or output_bytes >= 6 * GIB:
        raise ValueError("Cumulative resource budget exhausted")
    return seconds, output_bytes


def training_plan(binding, seed):
    from .price_meta125_plans import validate_plan

    path, record = completed(binding, "prepare-training")
    entry = record["plans"][str(seed)]
    plan_file = path / entry["file"]
    if file_sha(plan_file) != entry["sha256"]:
        raise ValueError("Training plan changed")
    plan = read_json(plan_file)
    validate_plan(plan, read_json(binding["data_config"]))
    if plan["seed"] != seed:
        raise ValueError("Training stream differs")
    return plan


def prepare(binding, output):
    from .price_meta125_plans import make_training_plan

    plans = {}
    for seed in SEEDS:
        path = Path(output) / f"training-{seed}.json"
        write_json(path, make_training_plan(read_json(binding["data_config"]), seed))
        plans[str(seed)] = {"file": path.name, "sha256": file_sha(path)}
    return finish(binding, output, "prepare-training", plans=plans)


def train(binding, output, arm, seed, device, *, probe=False):
    from .price_meta123_model import build_model, erm_loss, fomaml_loss
    from .price_meta125_plans import TrainingStores, verify_plan_stores

    torch = _torch_setup(device)
    if device != "cuda" or arm not in ("E", "M") or seed not in SEEDS:
        raise ValueError("Training/probe requires an authorized single GPU and stream")
    plan = training_plan(binding, seed)
    stores = TrainingStores(read_json(binding["data_config"]))
    verify_plan_stores(plan, stores)
    torch.manual_seed(seed)
    model = build_model(
        binding["initial_checkpoint"], device=device, expected_sha=INITIAL_SHA
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=2.85e-5, weight_decay=5e-5)
    torch.cuda.reset_peak_memory_stats()
    limit = 52 if probe else STEPS
    started = time.monotonic()
    times = []
    with (Path(output) / f"{arm}-loss_audit.jsonl").open("x") as log:
        for step in range(limit):
            torch.cuda.synchronize()
            before = time.monotonic()
            optimizer.zero_grad(set_to_none=True)
            total = 0.0
            for task_index, episode in enumerate(
                plan["episodes"][step * 4 : (step + 1) * 4]
            ):
                store = stores.get(episode["db"])
                support = episode["support"]["indices"]
                query = episode["query"]["indices"]
                if arm == "E":
                    loss = erm_loss(model, store.batch(support + query, device))
                else:
                    loss, audit = fomaml_loss(
                        model,
                        store.batch(support, device),
                        store.batch(query, device),
                        ALPHA,
                    )
                    if step < 2:
                        write_json(
                            Path(output) / f"inner-{step}-{task_index}.json", audit
                        )
                if not torch.isfinite(loss).item():
                    raise FloatingPointError("Nonfinite outer loss")
                (loss / META_BATCH).backward()
                total += float(loss.detach()) / META_BATCH
            if any(
                p.grad is None or not torch.isfinite(p.grad).all()
                for p in model.parameters()
            ):
                raise FloatingPointError("Missing or nonfinite outer gradient")
            norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
            if not math.isfinite(norm):
                raise FloatingPointError("Nonfinite gradient norm")
            optimizer.step()
            if any(not torch.isfinite(p).all() for p in model.parameters()):
                raise FloatingPointError("Nonfinite updated weights")
            torch.cuda.synchronize()
            elapsed = time.monotonic() - before
            if step >= 2:
                times.append(elapsed)
            log.write(
                json.dumps(
                    {
                        "update": step + 1,
                        "loss": total,
                        "gradient_norm": norm,
                        "seconds": elapsed,
                    }
                )
                + "\n"
            )
            if step % 25 == 0 or step + 1 == limit:
                log.flush()
                write_json(
                    Path(output) / "progress.json",
                    {
                        "stage": "probe" if probe else "training",
                        "arm": arm,
                        "stream": seed,
                        "updates": step + 1,
                        "elapsed_seconds": time.monotonic() - started,
                    },
                    replace=True,
                )
    checkpoint = Path(output) / (f"probe-{arm}.pt" if probe else "last.pt")
    state_path = Path(output) / (
        f"probe-state-{arm}.pt" if probe else "training_state.pt"
    )
    torch.save(model.price.state_dict(), checkpoint)
    torch.save(
        {
            "optimizer": optimizer.state_dict(),
            "updates": limit,
            "resume_forbidden": True,
        },
        state_path,
    )
    saved = torch.load(checkpoint, map_location=device, weights_only=False)
    if any(not torch.equal(saved[k], v) for k, v in model.price.state_dict().items()):
        raise ValueError("Checkpoint round-trip failed")
    saved_state = torch.load(state_path, map_location="cpu", weights_only=False)
    if {int(s["step"]) for s in saved_state["optimizer"]["state"].values()} != {limit}:
        raise ValueError("Optimizer step count differs")
    result = {
        "arm": arm,
        "stream": seed,
        "updates": limit,
        "initial_sha256": INITIAL_SHA,
        "checkpoint_sha256": file_sha(checkpoint),
        "training_state_sha256": file_sha(state_path),
        "training_plan_sha256": plan["plan_sha256"],
        "opened_databases": stores.opened_databases,
        "validation_monitor_reads": [],
        "max_update_seconds": max(times),
        "checkpoint_pair_bytes": checkpoint.stat().st_size + state_path.stat().st_size,
        "allocator_peak_bytes": torch.cuda.max_memory_allocated(),
        "loss_audit_sha256": file_sha(Path(output) / f"{arm}-loss_audit.jsonl"),
        "wall_seconds": time.monotonic() - started,
    }
    if probe:
        return result
    return finish(binding, output, f"train:{arm}:{seed}", **result)


def probe(binding, output, device):
    import resource

    started = time.monotonic()
    results = {
        arm: train(binding, output, arm, 42, device, probe=True) for arm in ("E", "M")
    }
    timing = {
        "e_update_seconds": results["E"]["max_update_seconds"],
        "m_update_seconds": results["M"]["max_update_seconds"],
        "setup_io_seconds": time.monotonic() - started,
        "verification_seconds": 60.0,
        "checkpoint_pair_bytes": sum(
            r["checkpoint_pair_bytes"] for r in results.values()
        ),
        "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        "peak_vram_bytes": max(r["allocator_peak_bytes"] for r in results.values()),
    }
    write_json(Path(output) / "timing.json", timing)
    write_json(Path(output) / "budget_projection.json", project_budget(timing))
    return finish(
        binding,
        output,
        "probe",
        timing_sha256=file_sha(Path(output) / "timing.json"),
        arms=results,
        research_use_forbidden=True,
    )


def seal(binding, output):
    records = {}
    for arm in ("E", "M"):
        for seed in SEEDS:
            path, value = completed(binding, f"train-{arm}-{seed}")
            plan = training_plan(binding, seed)
            if (
                value.get("updates") != STEPS
                or value.get("arm") != arm
                or value.get("stream") != seed
                or value.get("alpha") != ALPHA
                or value.get("initial_sha256") != INITIAL_SHA
                or value.get("validation_monitor_reads") != []
                or value.get("opened_databases") != list(plan["databases"])
                or value.get("training_plan_sha256") != plan["plan_sha256"]
                or file_sha(path / "last.pt") != value.get("checkpoint_sha256")
                or file_sha(path / "training_state.pt")
                != value.get("training_state_sha256")
                or file_sha(path / f"{arm}-loss_audit.jsonl")
                != value.get("loss_audit_sha256")
            ):
                raise ValueError("Incomplete or mismatched training matrix")
            records[f"{arm}-{seed}"] = {
                "checkpoint_sha256": value["checkpoint_sha256"],
                "completion_sha256": file_sha(path / "completion.json"),
            }
    return finish(
        binding, output, "seal-training", runs=records, validation_databases=[]
    )


def worker(binding_path, authorization_path, action, output, device):
    require_supervision()
    binding = load_binding(binding_path)
    validate_authorization(read_json(authorization_path), binding, action)
    output = Path(output)
    if (
        output != Path(binding["root"]) / "outputs" / action.replace(":", "-")
        or output.exists()
        or output.is_symlink()
    ):
        raise ValueError("Worker output is not a new isolated stage")
    output.mkdir()
    try:
        if action == "prepare-training":
            return prepare(binding, output)
        if action == "probe":
            return probe(binding, output, device)
        cost_gate(binding)
        if action == "seal-training":
            return seal(binding, output)
        _, arm, seed = action.split(":")
        return train(binding, output, arm, int(seed), device)
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
    import fcntl
    from .price_meta123_supervisor import supervise

    binding = load_binding(binding_path)
    auth = read_json(authorization_path)
    budget = dict(validate_authorization(auth, binding, action))
    output = output_path(binding["root"], action.replace(":", "-"))
    gpu_uuid = None
    if device == "cuda" and action not in ("prepare-training", "seal-training"):
        index, gpu_uuid = auth.get("cuda_device"), auth.get("cuda_uuid")
        if (
            type(index) is not int
            or not 0 <= index <= 7
            or not isinstance(gpu_uuid, str)
            or not re.fullmatch(r"GPU-[0-9a-fA-F-]{36}", gpu_uuid)
            or budget["max_vram_bytes"] <= 0
        ):
            raise ValueError(
                "GPU action requires its physical index, UUID and VRAM budget"
            )
        env = {"CUDA_VISIBLE_DEVICES": str(index)}
    elif (
        device == "cpu"
        and action in ("prepare-training", "seal-training")
        and budget["max_vram_bytes"] == 0
    ):
        env = {"CUDA_VISIBLE_DEVICES": ""}
    else:
        raise ValueError("Stage/device combination is forbidden")
    env.update(
        PYTHONDONTWRITEBYTECODE="1",
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
    )
    lock_path = Path(binding["root"]) / "inputs/execution.lock"
    if lock_path.is_symlink():
        raise ValueError("Execution lock is linked")
    with lock_path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        used_seconds, used_bytes = usage(binding)
        if action == "probe":
            completed(binding, "prepare-training")
            record = read_json(
                Path(binding["root"]) / "outputs/prepare-training.supervisor.json"
            )
            budget["max_wall_seconds"] = min(
                budget["max_wall_seconds"], 1800 - record["elapsed_seconds"]
            )
            budget["max_output_bytes"] = min(
                budget["max_output_bytes"], 512 * 1024**2 - used_bytes
            )
        elif action != "prepare-training":
            budget["max_wall_seconds"] = min(
                budget["max_wall_seconds"], 86400 - used_seconds
            )
            budget["max_output_bytes"] = min(
                budget["max_output_bytes"], 6 * GIB - used_bytes
            )
        return supervise(
            [
                sys.executable,
                "-B",
                str(Path(binding["code_root"]) / "scripts/run_price_meta125.py"),
                "worker",
                "--binding",
                str(Path(binding_path).resolve()),
                "--authorization",
                str(Path(authorization_path).resolve()),
                "--action",
                action,
                "--device",
                device,
                "--output",
                str(output),
            ],
            output,
            cwd=Path(binding["code_root"]),
            budget=budget,
            env=env,
            gpu_uuid=gpu_uuid,
            completion_protocol=125,
        )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    binder = sub.add_parser("bind")
    for name in ("root", "data-config", "checkpoint"):
        binder.add_argument("--" + name, required=True)
    for name in ("run", "worker"):
        stage = sub.add_parser(name)
        for field in ("binding", "authorization"):
            stage.add_argument("--" + field, required=True)
        stage.add_argument("--action", choices=ACTIONS, required=True)
        stage.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
        if name == "worker":
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
