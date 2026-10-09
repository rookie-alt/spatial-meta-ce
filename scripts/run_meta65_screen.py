#!/usr/bin/env python3
"""Frozen three-arm source-only FOMAML screen; dry-run is stdlib-only.

The caller must supervise the whole process with an external 7200-second timeout.
Elapsed checks here cover stage/chunk boundaries, not a blocked training call.
"""
import argparse
import csv
import datetime
import hashlib
import json
import math
import os
import random
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

ARMS = ("A_all5_head3", "B_head3_head3", "C_last3_last3")
VALIDATION = ("chembl", "legalacts", "telstra")
TARGETS = ("imdb", "stats", "genome", "ergastf1")
TRAIN = (
    "accidents", "airline", "baseball", "basketball", "carcinogenesis", "ccs",
    "consumer", "credit", "employee", "financial", "fnhk", "grants", "hepatitis",
    "hockey", "movielens", "sakila", "sap", "seznam", "ssb", "talkingdata",
    "tournament", "tpc_h", "tubepricing",
)
SHOTS = (5, 10, 20, 50, 100)
UNIQUE = {"chembl": 50000, "legalacts": 49504, "telstra": 50000}
SOURCE_BINDING = "3d708226437e105c88562970fdd0b2ccf711a833105b4ec8c66d899aa61f79a6"
PROTOCOL = "实验文档/48-Meta65低维适应源侧对照协议.md"
PROTOCOL_SHA = "fef38e8d18f46c4c167030628ee1dfa73dfbd9575f7b7eb76acf6e6c7844d8e4"
FROZEN = {
    "schema": 1, "experiment": "meta65-source-screen-20261009",
    "protocol_file": PROTOCOL, "protocol_sha256": PROTOCOL_SHA,
    "source_binding_sha256": SOURCE_BINDING,
    "training": {
        "method": "fomaml", "seed": 42, "steps": 4000, "meta_batch": 4,
        "accumulation": 1, "query_size": 64, "support_sizes": list(SHOTS),
        "outer_lr": 0.0003, "inner_lr": 0.001, "validate_every": 100,
        "validation_runs": 5, "validation_seed": 1729, "patience": 41,
        "min_delta": 0.0001,
    },
    "model": {"arch": "sum", "use_pg": True, "layers": 2, "heads": 4},
    "arms": {
        ARMS[0]: {"train_scope": "all", "train_steps": 5, "eval_scope": "head", "eval_steps": 3},
        ARMS[1]: {"train_scope": "head", "train_steps": 3, "eval_scope": "head", "eval_steps": 3},
        ARMS[2]: {"train_scope": "last", "train_steps": 3, "eval_scope": "last", "eval_steps": 3},
    },
    "evaluation": {"support_seed": 2718, "support_runs": 5, "shots": list(SHOTS), "batch_size": 512},
    "screen": {"min_improvement": 0.1, "max_regression": 0.1},
    "timeout_seconds": 7200,
}
CODE_FILES = (
    "scripts/run_meta65_screen.py", "scripts/run_meta65_supervised.py",
    "scripts/precision_screen_evidence.py",
    "scripts/run_s8_preflight.py", "scripts/mapping_smoke.py",
    "spatial_meta_ce/train.py", "spatial_meta_ce/meta.py", "spatial_meta_ce/data.py",
    "spatial_meta_ce/model.py", "spatial_meta_ce/checkpoints.py",
    "spatial_meta_ce/manifests.py", "spatial_meta_ce/metrics.py",
)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False, separators=(",", ":")).encode()).hexdigest()


def file_hash(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def read_json(path):
    def unique_keys(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result
    return json.loads(Path(path).read_text(), object_pairs_hook=unique_keys)


def write_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temp.replace(path)


def no_symlinks(path):
    path = Path(path).absolute()
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError(f"Symlink not allowed: {path}")
    return path.resolve()


def validate_config(config):
    # Canonical JSON also distinguishes bool/int, float/int and extra keys.
    if digest(config) != digest(FROZEN):
        raise ValueError("Frozen Meta65 config fields changed or unknown fields supplied")
    return config


def load_config(path, root=ROOT):
    config = validate_config(read_json(no_symlinks(path)))
    protocol = no_symlinks(Path(root) / config["protocol_file"])
    if not protocol.is_relative_to(Path(root).resolve()) or file_hash(protocol) != config["protocol_sha256"]:
        raise ValueError("Protocol SHA256 changed")
    return config


def validate_data(data, root=ROOT):
    splits = {"train": list(TRAIN), "validation": list(VALIDATION), "test": list(TARGETS)}
    if not isinstance(data, dict) or set(data) != {"data_root", "cache_root", "splits"}:
        raise ValueError("Only data/cache root path changes are permitted")
    if data["splits"] != splits:
        raise ValueError("Frozen exact 23/3/4 split changed; target loading forbidden")
    result = {"splits": splits}
    for key in ("data_root", "cache_root"):
        if not isinstance(data[key], str) or not data[key].strip():
            raise ValueError(f"Invalid {key}")
        path = Path(data[key]).expanduser()
        result[key] = str(no_symlinks(path if path.is_absolute() else Path(root) / path))
    return result


def validate_output(output, data, root=ROOT):
    output = no_symlinks(output)
    if output.exists():
        raise FileExistsError(f"Output must be a new directory: {output}")
    forbidden = (Path(data["cache_root"]), Path(data["data_root"]), Path("/gly/weilin/PRICE-master"))
    if any(output.is_relative_to(p.resolve()) for p in forbidden):
        raise ValueError("Output cannot be inside cache or PRICE")
    if any(p in (*TARGETS, "finetune", "test", "PRICE-master") for p in output.parts):
        raise ValueError("Target/PRICE paths are forbidden outputs")
    runs = no_symlinks(Path(root) / "runs")
    if output == runs or not output.is_relative_to(runs):
        raise ValueError("Raw predictions/checkpoints must stay under this checkout's runs directory")
    return output


def arm_config(config, arm):
    spec = config["arms"][arm]
    return {**config["training"], "model": {**config["model"], "width": 64},
            "loss": "smooth_l1", "adapt_scope": spec["train_scope"],
            "inner_steps": spec["train_steps"], "purpose": config["experiment"]}


def support_plan(db, indices, hashes, *, seed, runs, shots):
    if db not in VALIDATION:
        raise ValueError("Endpoint supports must be source validation databases")
    if (len(indices) != len(hashes) or len(set(indices)) != len(indices)
            or len(set(hashes)) != len(hashes) or not hashes
            or any(not isinstance(h, str) or not h for h in hashes)
            or not shots or sorted(set(shots)) != list(shots)
            or shots[0] < 1 or max(shots) >= len(indices) or runs < 1):
        raise ValueError("Invalid unique support/query pool")
    indices = [int(i) for i in indices]
    identities = dict(zip(indices, hashes))
    plan = []
    for run in range(runs):
        support = random.Random(f"{seed}:{db}:{run}").sample(indices, max(shots))
        excluded = set(support)
        query = [i for i in indices if i not in excluded]
        supports = {str(k): support[:k] for k in shots}
        plan.append({"database": db, "run": run, "support_max": support,
                     "supports": supports, "query": query,
                     "query_digest": digest([(i, identities[i]) for i in query]),
                     "support_digests": {str(k): digest([(i, identities[i]) for i in support[:k]]) for k in shots}})
    return plan


def decide(rows):
    expected = {(arm, db, k, run) for arm in ARMS for db in VALIDATION
                for k in SHOTS for run in range(5)}
    indexed = {}
    for row in rows:
        key = (row["arm"], row["database"], row["k"], row["run"])
        if key not in expected or key in indexed:
            raise ValueError("Unexpected or duplicate endpoint cell")
        for metric in ("all_q50", "all_q95", "four_plus_q50", "four_plus_q95"):
            value = row[metric]
            if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or value < 1:
                raise ValueError("Endpoint standard metrics must be finite and >= 1")
        for metric in ("all_log_q95", "four_plus_log_q95"):
            if not math.isfinite(row[metric]) or row[metric] < 0:
                raise ValueError("Endpoint log-Q95 must be finite and nonnegative")
        indexed[key] = row
    if set(indexed) != expected:
        raise ValueError("Endpoint requires exactly 75 cells per arm")
    for db in VALIDATION:
        for k in SHOTS:
            for run in range(5):
                reference = indexed[(ARMS[0], db, k, run)]
                for arm in ARMS[1:]:
                    actual = indexed[(arm, db, k, run)]
                    for field in ("query_digest", "support_digest", "query_count", "four_plus_count"):
                        if actual[field] != reference[field]:
                            raise ValueError(f"Endpoint pairing mismatch: {field}")
    summaries = {}
    for arm in ARMS:
        cells = [r for key, r in indexed.items() if key[0] == arm]
        groups = []
        for db in VALIDATION:
            for k in SHOTS:
                group = [indexed[(arm, db, k, run)] for run in range(5)]
                groups.append({"database": db, "k": k,
                               **{f"mean_{m}": math.fsum(r[m] for r in group) / 5
                                  for m in ("all_q50", "all_q95", "four_plus_q50", "four_plus_q95")},
                               "std_all_q95": statistics.stdev(r["all_q95"] for r in group),
                               "std_four_plus_q95": statistics.stdev(r["four_plus_q95"] for r in group)})
        summaries[arm] = {"macro_four_plus_q95": math.fsum(r["four_plus_q95"] for r in cells) / 75,
                          "macro_all_q95": math.fsum(r["all_q95"] for r in cells) / 75,
                          "macro_mean_log_q95": math.fsum(r["four_plus_log_q95"] for r in cells) / 75,
                          "macro_all_mean_log_q95": math.fsum(r["all_log_q95"] for r in cells) / 75,
                          "groups": groups}
    baseline = summaries[ARMS[0]]
    candidates = []
    for arm in ARMS[1:]:
        summary = summaries[arm]
        improved = summary["macro_four_plus_q95"] <= baseline["macro_four_plus_q95"] * (1 - FROZEN["screen"]["min_improvement"])
        regressions = []
        for group, reference in zip(summary["groups"], baseline["groups"]):
            for metric in ("all_q95", "four_plus_q95"):
                if group[f"mean_{metric}"] > reference[f"mean_{metric}"] * (1 + FROZEN["screen"]["max_regression"]):
                    regressions.append({"database": group["database"], "k": group["k"], "metric": metric})
        summary.update(relative_improvement=1 - summary["macro_four_plus_q95"] / baseline["macro_four_plus_q95"],
                       improvement_gate=improved, regressions=regressions, eligible=improved and not regressions)
        if summary["eligible"]:
            candidates.append(arm)
    adaptation_parameters = {ARMS[1]: 107521, ARMS[2]: 65}
    candidates.sort(key=lambda arm: (summaries[arm]["macro_four_plus_q95"], adaptation_parameters[arm], arm))
    return {"schema": 1, "experiment": FROZEN["experiment"],
            "status": "promising_single_seed" if candidates else "no_candidate",
            "selected_arm": candidates[0] if candidates else None, "candidates": candidates,
            "metric": "75-cell arithmetic macro standard unique-query four_plus_q95",
            "thresholds": FROZEN["screen"], "arms": summaries,
            "scope": "source-only single-seed engineering screen; not target generalization or significance"}


def check_deadline(started, timeout):
    if time.monotonic() - started >= timeout:
        raise TimeoutError("Meta65 total elapsed deadline exceeded")


def update_status(output, state, arms, started, **extra):
    write_json(output / "status.json", {"schema": 1, "experiment": FROZEN["experiment"],
               "status": state, "arms": dict(arms), "elapsed_seconds": time.monotonic() - started,
               "updated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(), **extra})


def environment(device):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    import numpy as np
    import torch
    if device not in ("cpu", "cuda", "cuda:0"):
        raise ValueError("Only cpu or single visible cuda:0 permitted")
    uuid = None
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if device != "cpu":
        if not visible or "," in visible or not (visible.isdigit() or visible.startswith("GPU-")):
            raise ValueError("CUDA requires explicit single CUDA_VISIBLE_DEVICES")
        gpu = subprocess.run(["nvidia-smi", "-i", visible,
                              "--query-gpu=uuid,memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
                             text=True, capture_output=True, check=True, timeout=10).stdout.strip().split(",")
        if len(gpu) != 3 or int(gpu[1]) > 100 or int(gpu[2]) != 0:
            raise RuntimeError("Selected GPU is not idle")
        uuid = gpu[0].strip()
        applications = subprocess.run(["nvidia-smi", "-i", visible,
                                       "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
                                      text=True, capture_output=True, check=True, timeout=10).stdout.strip()
        if applications:
            raise RuntimeError("Selected GPU already has compute processes")
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("Explicit CUDA selection is unavailable")
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    return {"python": sys.version, "executable": sys.executable, "torch": torch.__version__,
            "numpy": np.__version__, "device": device, "gpu_uuid": uuid,
            "CUDA_VISIBLE_DEVICES": visible, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
            "deterministic_algorithms": True, "external_timeout_required_seconds": 7200}


def validate_store(store, db):
    """Reject conflicting repeated identities before unique_indices removes repeats."""
    import numpy as np

    from scripts.run_s8_preflight import verify_token_duplicates
    if db not in (*TRAIN, *VALIDATION) or store.path.name != db or store.path.parent.name != "pretrain":
        raise ValueError("Only source pretrain stores permitted")
    cards, offsets = store.cards, store.offsets
    if (not len(store) or not np.isfinite(cards).all() or (cards < 0).any()
            or offsets.dtype.kind not in "iu" or (offsets < 0).any()
            or (offsets[0] != 0).any() or (offsets[1:] < offsets[:-1]).any()):
        raise ValueError("Invalid source cards/offsets")
    tables = np.diff(offsets[:, 2]).astype(np.int64)
    if (tables < 1).any():
        raise ValueError("Invalid source table counts")
    first = {}
    for i, row in enumerate(store.rows):
        identity = row.get("query_hash")
        if not isinstance(identity, str) or not identity or not isinstance(row.get("id"), str) or not row["id"].startswith(f"{db}/pretrain/"):
            raise ValueError("Invalid source identity")
        if identity in first and not np.array_equal(cards[i], cards[first[identity]]):
            raise ValueError("Duplicate identity has conflicting true/PG")
        first.setdefault(identity, i)
    verify_token_duplicates(store.path)
    unique = store.unique_indices()
    if len(unique) != store.manifest["unique_queries"] or (db in VALIDATION and len(unique) != UNIQUE[db]):
        raise ValueError("Frozen unique-query denominator changed")
    return tables


def evaluate_arm(model, stores, table_counts, plans, config, arm, output, device, deadline):
    import numpy as np
    import torch

    from spatial_meta_ce.meta import adapt, forward
    from spatial_meta_ce.metrics import summarize
    model.eval()
    spec = config["arms"][arm]
    records = []
    predictions = output / "predictions"
    predictions.mkdir()
    for db, runs in plans.items():
        if db not in VALIDATION:
            raise ValueError("Target endpoint loading forbidden")
        store = stores[db]
        for plan in runs:
            query = plan["query"]
            for k in config["evaluation"]["shots"]:
                deadline()
                started = time.monotonic()
                support = store.batch(plan["supports"][str(k)], device)
                if device != "cpu":
                    torch.cuda.synchronize()
                params = adapt(model, support, config["training"]["inner_lr"], spec["eval_steps"],
                               spec["eval_scope"], "smooth_l1", training=False)
                if device != "cpu":
                    torch.cuda.synchronize()
                adaptation_seconds = time.monotonic() - started
                started = time.monotonic()
                log_pred, true = [], []
                batch_size = config["evaluation"]["batch_size"]
                with torch.no_grad():
                    for start in range(0, len(query), batch_size):
                        deadline()
                        batch = store.batch(query[start:start + batch_size], device)
                        values = forward(model, params, batch).detach().cpu().numpy().astype(np.float64)
                        if not np.isfinite(values).all():
                            raise FloatingPointError("Nonfinite raw endpoint prediction")
                        log_pred.append(values)
                        true.append(batch["true"].cpu().numpy().astype(np.float64))
                log_pred, true = np.concatenate(log_pred), np.concatenate(true)
                prediction_seconds = time.monotonic() - started
                tables = table_counts[db][query]
                mask = tables >= 4
                if not mask.any():
                    raise ValueError("Empty four-plus endpoint bucket")
                all_metrics = summarize(log_pred, true)
                four_metrics = summarize(log_pred[mask], true[mask])
                for metrics in (all_metrics, four_metrics):
                    if not all(math.isfinite(metrics[f"q{p}"]) for p in (50, 95)):
                        raise FloatingPointError("Nonfinite standard endpoint Q-error")
                path = predictions / f"{db}-k{k}-run{plan['run']}.npz"
                np.savez(path, log_pred=log_pred, true=true, n_tables=tables,
                         query_hash=np.asarray([store.rows[i]["query_hash"] for i in query]),
                         query_id=np.asarray([store.rows[i]["id"] for i in query]))
                records.append({"arm": arm, "database": db, "k": k, "run": plan["run"],
                                "query_digest": plan["query_digest"], "support_digest": plan["support_digests"][str(k)],
                                "query_count": len(query), "four_plus_count": int(mask.sum()),
                                "all_q50": all_metrics["q50"], "all_q95": all_metrics["q95"],
                                "four_plus_q50": four_metrics["q50"], "four_plus_q95": four_metrics["q95"],
                                "all_log_q95": all_metrics["log_q95"], "four_plus_log_q95": four_metrics["log_q95"],
                                "adaptation_seconds": adaptation_seconds, "prediction_seconds": prediction_seconds,
                                "prediction_file": str(path.relative_to(output)), "prediction_sha256": file_hash(path)})
                del params, support
    return records


def write_endpoint(path, rows):
    if not rows:
        raise ValueError("Empty endpoint")
    with Path(path).open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def train_arm(config, data, output, device, deadline):
    import torch

    from scripts.mapping_smoke import state_hash
    from spatial_meta_ce.checkpoints import load_checkpoint
    from spatial_meta_ce.model import SetEstimator
    from spatial_meta_ce.train import seed_all, train
    deadline()
    seed_all(config["seed"])
    initial = SetEstimator(**config["model"])
    initial_hash = state_hash(initial)
    parameters = sum(p.numel() for p in initial.parameters())
    if parameters != 132545:
        raise ValueError("Frozen model parameter budget changed")
    del initial
    started = time.monotonic()
    summary = train(config, data, output, device=device)  # No resume; never load best.pt.
    deadline()
    model, checkpoint = load_checkpoint(output / "last.pt", device)
    extra, metadata = checkpoint["extra"], checkpoint["metadata"]
    counts = extra["counts"]
    episodes = config["steps"] * config["meta_batch"]
    if (checkpoint["next_step"] != config["steps"] or summary["last_step"] != config["steps"]
            or counts["optimizer_steps"] != config["steps"]
            or counts["query_presentations"] != episodes * config["query_size"]
            or counts["backward_calls"] != episodes * (config["inner_steps"] + 1)
            or extra["sample_plan_episodes"] != episodes
            or counts["support_presentations"] <= 0
            or counts["support_presentations"] % config["inner_steps"]
            or counts != summary["counts"]):
        raise ValueError("Frozen FOMAML training budget incomplete or inconsistent")
    signature = {k: v for k, v in config.items() if k not in ("steps", "data_config")}
    if (checkpoint["model_config"] != config["model"] or metadata["training_signature"] != signature
            or metadata["train_databases"] != list(TRAIN) or metadata["validation_databases"] != list(VALIDATION)):
        raise ValueError("Final checkpoint config/source identity changed")
    if any(not torch.isfinite(p).all() for p in model.state_dict().values()):
        raise FloatingPointError("Nonfinite final model weights")
    if state_hash(model) == initial_hash:
        raise ValueError("Optimizer did not change initial weights")
    record = {"config": config, "initialization_digest": initial_hash, "parameters": parameters,
              "metadata": metadata, "counts": counts, "optimizer_steps": counts["optimizer_steps"],
              "sample_plan_digest": extra["sample_plan_digest"], "sample_plan_episodes": episodes,
              "support_sample_count": counts["support_presentations"] // config["inner_steps"],
              "validation_plan_digest": file_hash(output / "validation_manifest.json"),
              "checkpoint_sha256": file_hash(output / "last.pt"), "checkpoint": "last.pt",
              "last_step": checkpoint["next_step"], "training_seconds": time.monotonic() - started,
              "summary": summary}
    return model, record


def verify_pairing(records):
    if not records or len(records) > len(ARMS) or [r["arm"] for r in records] != list(ARMS[:len(records)]):
        raise ValueError("Invalid partial arm matrix")
    for field in ("initialization_digest", "parameters", "sample_plan_digest", "sample_plan_episodes",
                  "support_sample_count", "validation_plan_digest", "endpoint_plan_digest", "optimizer_steps"):
        if len({digest(r[field]) for r in records}) != 1:
            raise ValueError(f"Cross-arm pairing mismatch: {field}")
    for record in records:
        if record["metadata"]["manifest_hashes"] != records[0]["metadata"]["manifest_hashes"]:
            raise ValueError("Cross-arm source manifest mismatch")


def _execute_work(config, data, output, device, deadline, arms, started, root, config_path, data_config_path):
    from scripts.precision_screen_evidence import historical_sources, source_evidence
    from spatial_meta_ce.data import FeatureStore
    deadline()
    runtime = environment(device)
    runtime.update(protocol_sha256=config["protocol_sha256"], source_binding_sha256=SOURCE_BINDING,
                   config_sha256=file_hash(config_path), data_config_sha256=file_hash(data_config_path),
                   code_sha256={p: file_hash(Path(root) / p) for p in CODE_FILES})
    write_json(output / "environment.json", runtime)
    historical = historical_sources(Path(root), list(TRAIN + VALIDATION))
    before = source_evidence(data, historical)
    write_json(output / "source_before.json", before)
    try:
        stores, table_counts = {}, {}
        for db in TRAIN + VALIDATION:
            deadline()
            store = FeatureStore(Path(data["cache_root"]) / "pretrain" / db)
            tables = validate_store(store, db)
            if db in VALIDATION:
                stores[db], table_counts[db] = store, tables
        evaluation = config["evaluation"]
        plans = {db: support_plan(db, stores[db].unique_indices().tolist(),
                                  [stores[db].rows[int(i)]["query_hash"] for i in stores[db].unique_indices()],
                                  seed=evaluation["support_seed"], runs=evaluation["support_runs"], shots=evaluation["shots"])
                 for db in VALIDATION}
        write_json(output / "endpoint_plan.json", plans)
        plan_digest = digest(plans)
        records, rows = [], []
        for arm in ARMS:
            deadline()
            arms[arm] = "running"
            update_status(output, "running", arms, started, active_arm=arm)
            arm_output = output / arm
            model, record = train_arm(arm_config(config, arm), data, arm_output, device, deadline)
            if record["metadata"]["manifest_hashes"] != {db: info["manifest_hash"] for db, info in before.items()}:
                raise ValueError("Training inputs differ from authenticated P1 source cache")
            record.update(arm=arm, endpoint_plan_digest=plan_digest)
            records.append(record)
            verify_pairing(records)
            begun = time.monotonic()
            arm_rows = evaluate_arm(model, stores, table_counts, plans, config, arm, arm_output, device, deadline)
            for row in arm_rows:
                row["prediction_file"] = f"{arm}/{row['prediction_file']}"
            record.update(endpoint_seconds=time.monotonic() - begun, endpoint_cells=len(arm_rows))
            write_json(arm_output / "metadata.json", record)
            write_endpoint(arm_output / "endpoint.csv", arm_rows)
            rows.extend(arm_rows)
            arms[arm] = "completed"
            update_status(output, "running", arms, started)
            del model
        deadline()
        decision = decide(rows)
        write_endpoint(output / "endpoint.csv", rows)
    finally:
        # Retain before/after evidence even when an arm fails; do not retry/rebuild.
        after = source_evidence(data, historical)
        write_json(output / "source_after.json", after)
        if after != before:
            raise ValueError("Source cache hashes changed during screen")
        if historical_sources(Path(root), list(TRAIN + VALIDATION)) != historical:
            raise ValueError("Historical source binding changed during screen")
        if {p: file_hash(Path(root) / p) for p in CODE_FILES} != runtime["code_sha256"]:
            raise ValueError("Runner/core source files changed during execution")
        if file_hash(config_path) != runtime["config_sha256"] or file_hash(data_config_path) != runtime["data_config_sha256"]:
            raise ValueError("Config inputs changed during execution")
        if file_hash(Path(root) / config["protocol_file"]) != config["protocol_sha256"]:
            raise ValueError("Protocol changed during execution")
    deadline()
    return {"source_rehashed_after": True, "endpoint_plan_digest": plan_digest, "decision": decision}


def execute(config, data, output, device, *, root=ROOT,
            config_path=ROOT / "configs/meta65_screen.json", data_config_path=ROOT / "configs/data.json"):
    validate_config(config)
    data = validate_data(data, root=root)
    output = validate_output(output, data, root=root)
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    arms = dict.fromkeys(ARMS, "planned")
    update_status(output, "running", arms, started)
    write_json(output / "environment.json", {"python": sys.version, "device": device,
               "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
               "runtime_preflight": "pending"})
    def deadline():
        check_deadline(started, config["timeout_seconds"])
    try:
        result = _execute_work(config, data, output, device, deadline, arms, started,
                               root, config_path, data_config_path)
        deadline()
        decision = result.pop("decision")
        update_status(output, "completed", arms, started, **result)
        # Publish only after source/code/config/protocol checks and final deadline.
        write_json(output / "decision.json", decision)
    except BaseException as error:
        for arm in arms:
            if arms[arm] == "running":
                arms[arm] = "failed"
        update_status(output, "failed", arms, started, error_type=type(error).__name__, error=str(error))
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/meta65_screen.json")
    parser.add_argument("--data-config", type=Path, default=ROOT / "configs/data.json")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    config = load_config(args.config)
    data = validate_data(read_json(no_symlinks(args.data_config)))
    if not args.execute:
        print(json.dumps({"purpose": "dry-run-not-execution", "output": str(args.out),
                          "source_only": True, "timeout_seconds": config["timeout_seconds"],
                          "matrix": [{"arm": arm, "training": arm_config(config, arm),
                                      "evaluation": {**config["evaluation"], **config["arms"][arm]}}
                                     for arm in ARMS]}, indent=2))
        return
    execute(config, data, args.out, args.device,
            config_path=args.config, data_config_path=args.data_config)


if __name__ == "__main__":
    main()
