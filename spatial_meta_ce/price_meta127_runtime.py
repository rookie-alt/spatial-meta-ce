"""Frozen official PRICE target evaluation; full workloads include SQL overlap."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback

from .price_meta123_runtime import (
    _torch_setup,
    digest,
    file_sha,
    read_json,
    require_supervision,
    safe_root,
    write_json,
)
from .price_meta126_runtime import DATABASES, training_records

PROTOCOL_FILE = "实验文档/127-官方设置与全行口径重新评估.md"
PROTOCOL_SHA = "6fdf9419b9ee181450d9f3433e3e406c1df209cbe54a7478e32d342ecb50a0a3"
UPSTREAM = "reports/price-meta127-official-eval-20261118/upstream"
UPSTREAM_COMMIT = "487f8d8669b5592b537ce294b6902378a500c256"
ROWS = dict(zip(DATABASES, (766, 2749, 1422, 2302)))
MODEL_KEYS = ("P-0", "E-42", "E-43", "E-44", "M-42", "M-43", "M-44")
RECIPE = {
    "epochs": 100,
    "effective_batch": 15000,
    "microbatch": 512,
    "lr": 1e-5,
    "weight_decay": 5e-5,
    "optimizer": "Adam",
    "scheduler_step": "epoch",
    "scheduler": "OneCycleLR(steps_per_epoch=10,epochs=10,pct_start=.3,div_factor=2,cos)",
    "rng": 42,
    "precision": "float32",
    "topology": "single_gpu_gradient_accumulation",
    "known_support_test_overlap": True,
}
TOTAL_SECONDS = 72 * 3600
TOTAL_OUTPUT = 2 * 1024**3
FIELDS = ("model", "database", "row", "query_hash", "true", "pg", "encoded", "label")


def finetune_stages():
    return [f"finetune-{key}-{db}" for key in MODEL_KEYS for db in DATABASES[1:]]


def stages():
    return ["zero", "probe", *finetune_stages(), "report"]


def project_cost(step_seconds, *, setup_seconds):
    if (
        len(step_seconds) != 4
        or any(not math.isfinite(x) or x <= 0 for x in step_seconds)
        or not math.isfinite(setup_seconds)
        or setup_seconds < 0
    ):
        raise ValueError("Probe must contain four finite positive macro timings")
    one = (max(step_seconds) * 400 + setup_seconds + 120) * 1.3
    total = one * 21 + 1800 + 1800 + 600
    return {
        "projected_run_seconds": one,
        "projected_total_seconds": total,
        "run_limit_seconds": 8 * 3600,
        "total_limit_seconds": TOTAL_SECONDS,
        "feasible": one <= 8 * 3600 and total <= TOTAL_SECONDS,
    }


def code_closure(code):
    code = Path(code)
    paths = [
        code / PROTOCOL_FILE,
        code / "vendor/PROVENANCE.json",
        code / "scripts/run_price_meta127.py",
    ]
    for folder, pattern in (
        ("spatial_meta_ce", "*.py"),
        ("vendor/price_baseline", "*.py"),
        (UPSTREAM, "*"),
    ):
        paths += sorted(p for p in (code / folder).rglob(pattern) if p.is_file())
    if any(not p.is_file() or p.is_symlink() for p in paths):
        raise ValueError("Missing or linked official execution dependency")
    closure = {str(p.relative_to(code)): file_sha(p) for p in paths}
    manifest = read_json(code / UPSTREAM / "manifest.json")
    if (
        closure[PROTOCOL_FILE] != PROTOCOL_SHA
        or manifest["commit"] != UPSTREAM_COMMIT
        or any(
            closure.get(f"{UPSTREAM}/{p}") != s for p, s in manifest["files"].items()
        )
    ):
        raise ValueError("Official source or protocol differs")
    return closure


def bind(root, training_root, price_root, cache_root):
    import numpy as np
    from .price_meta127_native import load_raw

    root = safe_root(root)
    price, cache = Path(price_root).resolve(), Path(cache_root).resolve()
    data, test_max = {}, [0, 0, 0, 0]
    for usage in ("test", "finetune"):
        for db in DATABASES:
            path = price / "setup/features" / usage / db / "features40.pkl"
            name = f"{usage}/{db}"
            if not path.exists() and name == "finetune/imdb":
                data[name] = {"available": False, "path": str(path)}
                continue
            raw = load_raw(path, expected_rows=ROWS[db] if usage == "test" else 50000)
            workload = price / "datas/workloads" / usage / db / "workloads.sql"
            cards = []
            with workload.open() as stream:
                for line in stream:
                    if line.strip():
                        fields = line.rstrip().split("||")
                        cards.append([float(fields[1]), float(fields[2])])
            a = np.asarray(cards, dtype=np.float32)
            if (
                a.shape != (raw.rows, 2)
                or not np.array_equal(a[:, 0], raw.truth.numpy())
                or not np.array_equal(a[:, 1], raw.pg.numpy())
            ):
                raise ValueError("Raw official pickle and workload labels/order differ")
            manifest = cache / usage / db / "manifest.json"
            cached = read_json(manifest)
            source = cached["provenance"]["source"]
            if (
                len(cached["rows"]) != raw.rows
                or source["feature_sha256"] != file_sha(path)
                or source["workload_sha256"] != file_sha(workload)
                or source["database"] != db
                or source["split"] != usage
                or cached["provenance"]["excluded"]
            ):
                raise ValueError("Raw workload and identity manifest differ")
            entry = {
                "available": True,
                "path": str(path),
                "sha256": file_sha(path),
                "workload": str(workload),
                "workload_sha256": file_sha(workload),
                "manifest": str(manifest),
                "manifest_sha256": file_sha(manifest),
                "rows": raw.rows,
                "counts": list(raw.counts),
            }
            if usage == "test":
                entry["identities"] = [r["query_hash"] for r in cached["rows"]]
                test_max = [max(a, b) for a, b in zip(test_max, raw.counts)]
            else:
                hashes = set(data[f"test/{db}"]["identities"])
                entry["test_overlap_rows"] = sum(
                    r["query_hash"] in hashes for r in cached["rows"]
                )
                entry["test_overlap_unique_queries"] = len(
                    {r["query_hash"] for r in cached["rows"]} & hashes
                )
            data[name] = entry
    value = {
        "schema": 1,
        "protocol": 127,
        "root": str(root),
        "training_root": str(Path(training_root).resolve()),
        "models": training_records(training_root),
        "code_sha256": code_closure(root / "code"),
        "data": data,
        "test_counts": test_max,
        "recipe": RECIPE,
        "allowed_targets": list(DATABASES),
        "selection_forbidden": True,
    }
    value["binding_sha256"] = digest(value)
    write_json(root / "inputs/binding.json", value)
    return value


def load_binding(path):
    value = read_json(path)
    if value.get("protocol") != 127 or value.get("binding_sha256") != digest(
        {k: v for k, v in value.items() if k != "binding_sha256"}
    ):
        raise ValueError("Invalid official binding")
    root = safe_root(value["root"])
    if (
        Path(path).resolve() != root / "inputs/binding.json"
        or value["recipe"] != RECIPE
        or value["allowed_targets"] != list(DATABASES)
        or value["selection_forbidden"] is not True
        or code_closure(root / "code") != value["code_sha256"]
        or training_records(value["training_root"]) != value["models"]
    ):
        raise ValueError("Frozen official execution changed")
    for entry in value["data"].values():
        if not entry["available"]:
            if Path(entry["path"]).exists():
                raise ValueError("Previously missing official data appeared")
            continue
        for p, s in (
            ("path", "sha256"),
            ("workload", "workload_sha256"),
            ("manifest", "manifest_sha256"),
        ):
            if Path(entry[p]).is_symlink() or file_sha(entry[p]) != entry[s]:
                raise ValueError("Official data bytes changed")
    return value


def validate_authorization(auth, binding, stage):
    if (
        stage not in stages()
        or auth.get("schema") != 1
        or auth.get("protocol") != 127
        or auth.get("authorized") is not True
        or auth.get("root") != binding["root"]
        or auth.get("binding_sha256") != binding["binding_sha256"]
        or auth.get("actions") != stages()
        or auth.get("allowed_targets") != list(DATABASES)
        or auth.get("selection_forbidden") is not True
        or auth.get("recipe") != RECIPE
        or not auth.get("granted_by")
        or not auth.get("granted_at")
    ):
        raise ValueError("Explicit fixed official authorization required")


def completed(binding, stage):
    p = Path(binding["root"]) / "outputs" / stage
    c, s = (
        read_json(p / "completion.json"),
        read_json(p.with_name(p.name + ".supervisor.json")),
    )
    if (
        c.get("protocol") != 127
        or c.get("status") != "completed"
        or c.get("stage") != stage
        or c.get("binding_sha256") != binding["binding_sha256"]
        or s.get("status") != "completed"
        or s.get("completion") != c
    ):
        raise ValueError("Prior official stage is not completed")
    for name, sha in c.get("files", {}).items():
        if Path(name).name != name or file_sha(p / name) != sha:
            raise ValueError("Completed official evidence changed")
    return c, s


def finish(binding, stage, output, **extra):
    value = {
        "schema": 1,
        "protocol": 127,
        "stage": stage,
        "status": "completed",
        "binding_sha256": binding["binding_sha256"],
        "selection_forbidden": True,
        **extra,
        "files": {
            p.name: file_sha(p)
            for p in sorted(Path(output).iterdir())
            if p.is_file() and p.name != "progress.json"
        },
    }
    write_json(Path(output) / "completion.json", value)
    return value


def predict(binding, output, key, db, model, *, save_name=None):
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    from .price_meta127_metrics import summarize_official
    from .price_meta127_native import forward_batch, load_raw

    entry = binding["data"][f"test/{db}"]
    raw = load_raw(entry["path"], expected_rows=ROWS[db])
    dataset = raw.padded(binding["test_counts"])
    outputs, labels = [], []
    if model is not None:
        model.eval()
    for batch in DataLoader(dataset, batch_size=512, shuffle=False):
        with torch.no_grad():
            if model is None:
                z = torch.log(batch[2].to("cuda").float() + 1) + 1
                y = torch.log(batch[1].to("cuda").float() + 1) + 1
            else:
                z, y = forward_batch(model, batch, "cuda")
        outputs.extend(z.detach().cpu().numpy().reshape(-1).tolist())
        labels.extend(y.detach().cpu().numpy().reshape(-1).tolist())
    z, y = np.asarray(outputs, dtype=np.float32), np.asarray(labels, dtype=np.float32)
    metrics = summarize_official(z, y, true_cards=raw.truth.numpy())
    name = save_name or f"{key}-{db}.csv.gz"
    with gzip.open(Path(output) / name, "xt", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        for i in range(raw.rows):
            values = (
                key,
                db,
                i + 1,
                entry["identities"][i],
                repr(float(raw.truth[i])),
                repr(float(raw.pg[i])),
                repr(float(z[i])),
                repr(float(y[i])),
            )
            writer.writerow(dict(zip(FIELDS, values)))
    return {
        "model": key,
        "database": db,
        "n": raw.rows,
        "official": metrics,
        "prediction_file": name,
        "prediction_sha256": file_sha(Path(output) / name),
    }


def zero(binding, output):
    import torch
    from .price_meta127_native import load_model

    units = []
    for db in DATABASES:
        units.append(predict(binding, output, "PG", db, None))
        for key in MODEL_KEYS:
            rec = binding["models"][key]
            model = load_model(
                rec["checkpoint"], rec["sha256"], binding["test_counts"], "cuda"
            )
            units.append(predict(binding, output, key, db, model))
            del model
            torch.cuda.empty_cache()
    write_json(Path(output) / "units.json", units)
    return finish(
        binding, "zero", output, cells=len(units), population="official_test_all_rows"
    )


def train(binding, stage, output):
    import numpy as np
    import torch
    from .price_meta127_native import (
        create_optimizer,
        load_model,
        load_raw,
        train_epoch,
    )

    probe = stage == "probe"
    key, db = (
        ("P-0", "stats") if probe else stage.removeprefix("finetune-").rsplit("-", 1)
    )
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    started = time.monotonic()
    entry = binding["data"][f"finetune/{db}"]
    raw = load_raw(entry["path"], expected_rows=50000)
    dataset = raw.padded(raw.counts)
    record = binding["models"][key]
    model = load_model(record["checkpoint"], record["sha256"], raw.counts, "cuda")
    optimizer, scheduler = create_optimizer(model)
    setup_seconds = time.monotonic() - started
    epochs = []
    with (Path(output) / "epochs.jsonl").open("x") as log:
        for epoch in range(1 if probe else 100):
            lr = optimizer.param_groups[0]["lr"]
            stats = train_epoch(
                model, dataset, optimizer, batch_size=15000, microbatch=512
            )
            if stats["updates"] != 4 or stats["examples"] != 50000:
                raise ValueError("Official epoch lost examples or macro updates")
            scheduler.step()
            info = {"epoch": epoch + 1, "lr": lr, **stats}
            log.write(json.dumps(info, allow_nan=False) + "\n")
            log.flush()
            os.fsync(log.fileno())
            epochs.append(info)
            write_json(
                Path(output) / "progress.json",
                {
                    "stage": stage,
                    "epoch": epoch + 1,
                    "epochs": 1 if probe else 100,
                    "updates": 4 * (epoch + 1),
                    "seconds": time.monotonic() - started,
                },
                replace=True,
            )
    if probe:
        projection = project_cost(
            epochs[0]["step_seconds"], setup_seconds=setup_seconds
        )
        write_json(Path(output) / "cost.json", projection)
        return finish(
            binding,
            stage,
            output,
            research=False,
            updates=4,
            examples=50000,
            cost=projection,
        )
    torch.save(model.state_dict(), Path(output) / "last.pt")
    del model, optimizer, scheduler
    torch.cuda.empty_cache()
    adapted = load_model(
        Path(output) / "last.pt",
        file_sha(Path(output) / "last.pt"),
        binding["test_counts"],
        "cuda",
    )
    unit = predict(binding, output, key, db, adapted, save_name="predictions.csv.gz")
    write_json(Path(output) / "unit.json", unit)
    return finish(
        binding,
        stage,
        output,
        model=key,
        database=db,
        epochs=100,
        updates=400,
        examples=5000000,
        initial_sha256=record["sha256"],
        population="official_workload_with_known_overlap",
    )


def report(binding, output):
    completed(binding, "zero")
    units = list(read_json(Path(binding["root"]) / "outputs/zero/units.json"))
    for stage in finetune_stages():
        completed(binding, stage)
        units.append(read_json(Path(binding["root"]) / "outputs" / stage / "unit.json"))
    write_json(Path(output) / "units.json", units)
    return finish(
        binding,
        "report",
        output,
        cells=53,
        imdb_finetune_status="blocked_missing_official_features",
    )


def worker(binding_path, authorization, stage):
    require_supervision()
    binding = load_binding(binding_path)
    validate_authorization(read_json(authorization), binding, stage)
    output = Path(binding["root"]) / "outputs" / stage
    if output.exists() or output.is_symlink():
        raise FileExistsError("Official output must be fresh")
    output.mkdir()
    try:
        if stage != "report":
            _torch_setup("cuda")
        if stage == "zero":
            return zero(binding, output)
        if stage == "report":
            return report(binding, output)
        return train(binding, stage, output)
    except BaseException as exc:
        write_json(
            output / "failure.json",
            {
                "type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
            },
        )
        raise


def experiment_usage(binding):
    outputs = Path(binding["root"]) / "outputs"
    seconds = sum(
        float(read_json(p)["elapsed_seconds"])
        for p in outputs.glob("*.supervisor.json")
    )
    size = sum(p.stat().st_size for p in outputs.rglob("*") if p.is_file())
    return seconds, size


def run(binding_path, authorization, stage, *, deadline=None):
    import fcntl
    from .price_meta123_supervisor import supervise

    binding = load_binding(binding_path)
    auth = read_json(authorization)
    validate_authorization(auth, binding, stage)
    if stage != "zero":
        completed(binding, "zero")
    if stage in finetune_stages():
        probe, sup = completed(binding, "probe")
        if (
            not probe["cost"]["feasible"]
            or sup.get("vram_accounting") != "device_total"
            or sup["peak_vram_bytes"] <= max(16 * 1024**2, sup["vram_baseline_bytes"])
        ):
            raise ValueError("Official cost or external VRAM gate failed")
    elapsed, size = experiment_usage(binding)
    if elapsed >= TOTAL_SECONDS or size >= TOTAL_OUTPUT:
        raise ValueError("Official total resource budget exhausted")
    cap = 600 if stage == "report" else 1800 if stage in ("zero", "probe") else 8 * 3600
    volume = 256 * 1024**2 if stage in finetune_stages() else 64 * 1024**2
    budget = {
        "max_wall_seconds": min(cap, TOTAL_SECONDS - elapsed),
        "max_rss_bytes": 24 * 1024**3,
        "max_vram_bytes": 0 if stage == "report" else 12 * 1024**3,
        "max_output_bytes": min(volume, TOTAL_OUTPUT - size),
        "poll_seconds": 0.5,
    }
    if deadline is not None:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise ValueError("Official sequence deadline exhausted")
        budget["max_wall_seconds"] = min(budget["max_wall_seconds"], remaining)
    root = Path(binding["root"])
    with (root / "inputs/execution.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return supervise(
            [
                sys.executable,
                "-B",
                str(root / "code/scripts/run_price_meta127.py"),
                "worker",
                "--binding",
                str(binding_path),
                "--authorization",
                str(authorization),
                "--stage",
                stage,
            ],
            root / "outputs" / stage,
            cwd=root / "code",
            budget=budget,
            env={
                "CUDA_VISIBLE_DEVICES": str(auth["cuda_device"])
                if stage != "report"
                else "",
                "OMP_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
                "PYTHONDONTWRITEBYTECODE": "1",
            },
            gpu_uuid=auth["cuda_uuid"] if stage != "report" else None,
            completion_protocol=127,
        )


def sequence(binding_path, authorization):
    """Execute each remaining frozen cell once; never resume a partial sequence."""
    import datetime
    import fcntl

    binding = load_binding(binding_path)
    auth = read_json(authorization)
    validate_authorization(auth, binding, finetune_stages()[0])
    completed(binding, "zero")
    probe, _ = completed(binding, "probe")
    if not probe["cost"]["feasible"]:
        raise ValueError("Official sequence exceeds the fixed cost budget")
    root = Path(binding["root"])
    pending = [*finetune_stages(), "report"]
    for stage in pending:
        output = root / "outputs" / stage
        if (
            output.exists()
            or output.with_name(output.name + ".supervisor.json").exists()
        ):
            raise FileExistsError("Sequence never resumes or overwrites evidence")
    start_seconds, _ = experiment_usage(binding)
    deadline = time.monotonic() + TOTAL_SECONDS - start_seconds
    auth_sha = file_sha(authorization)
    state = {
        "schema": 1,
        "protocol": 127,
        "status": "running",
        "binding_sha256": binding["binding_sha256"],
        "pid": os.getpid(),
        "started_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "actions": pending,
        "completed": [],
        "current_stage": None,
        "authorization_sha256": auth_sha,
    }
    with (root / "inputs/sequence.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        write_json(root / "evidence/sequence-started.json", state)
        try:
            for stage in pending:
                if time.monotonic() >= deadline or file_sha(authorization) != auth_sha:
                    raise ValueError("Sequence deadline or authorization changed")
                state["current_stage"] = stage
                write_json(root / "evidence/sequence-status.json", state, replace=True)
                result = run(binding_path, authorization, stage, deadline=deadline)
                if result["status"] != "completed":
                    raise RuntimeError(
                        f"Official stage failed: {stage}: {result.get('failure')}"
                    )
                completed(binding, stage)
                state["completed"].append(stage)
                write_json(root / "evidence/sequence-status.json", state, replace=True)
            state["status"] = "completed"
            state["current_stage"] = None
            write_json(root / "evidence/sequence-completed.json", state)
        except BaseException as exc:
            state["status"] = "failed"
            state["error"] = {"type": type(exc).__name__, "message": str(exc)}
            write_json(root / "evidence/sequence-failed.json", state)
            raise
        finally:
            write_json(root / "evidence/sequence-status.json", state, replace=True)
    return state


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    b = sub.add_parser("bind")
    for name in ("root", "training-root", "price-root", "cache-root"):
        b.add_argument("--" + name, required=True)
    for command in ("run", "worker", "sequence"):
        p = sub.add_parser(command)
        p.add_argument("--binding", type=Path, required=True)
        p.add_argument("--authorization", type=Path, required=True)
        if command != "sequence":
            p.add_argument("--stage", choices=stages(), required=True)
    args = parser.parse_args(argv)
    if args.command == "bind":
        value = bind(args.root, args.training_root, args.price_root, args.cache_root)
    elif args.command == "run":
        value = run(args.binding, args.authorization, args.stage)
    elif args.command == "sequence":
        value = sequence(args.binding, args.authorization)
    else:
        value = worker(args.binding, args.authorization, args.stage)
    print(json.dumps(value, ensure_ascii=False, allow_nan=False))
    return 0 if value.get("status", "completed") == "completed" else 1
