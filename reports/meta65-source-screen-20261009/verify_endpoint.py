#!/usr/bin/env python3
"""Independent, source-only Meta65 artifact verification (stdlib + NumPy).

No runner imports, training, checkpoint deserialization, polling or remote access.
--self-test exercises synthetic helpers only; it cannot relax the formal contract.
"""
import argparse
import csv
import hashlib
import json
import math
import random
import statistics
import sys
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np

EXPERIMENT = "meta65-source-screen-20261009"
ARMS = ("A_all5_head3", "B_head3_head3", "C_last3_last3")
VAL = ("chembl", "legalacts", "telstra")
TRAIN = (
    "accidents", "airline", "baseball", "basketball", "carcinogenesis", "ccs",
    "consumer", "credit", "employee", "financial", "fnhk", "grants", "hepatitis",
    "hockey", "movielens", "sakila", "sap", "seznam", "ssb", "talkingdata",
    "tournament", "tpc_h", "tubepricing",
)
SHOTS = (5, 10, 20, 50, 100)
UNIQUE = dict(zip(VAL, (50000, 49504, 50000)))
FOUR_PLUS = dict(zip(VAL, (33947, 9178, 20311)))
SOURCE_BINDING = "3d708226437e105c88562970fdd0b2ccf711a833105b4ec8c66d899aa61f79a6"
CONFIG_SHA = "ebd5564cb66a8df03f2c78f293c4cacb493b8af1836932ce7a9cceca6f3d3c36"
DATA_SHA = "a6907d602ed92fdfe42abe2828bbbac9369be07de04241c17cf6a3bd61544283"
PROTOCOL = "实验文档/48-Meta65低维适应源侧对照协议.md"
PROTOCOL_SHA = "fef38e8d18f46c4c167030628ee1dfa73dfbd9575f7b7eb76acf6e6c7844d8e4"
CODE_FILES = (
    "scripts/run_meta65_screen.py", "scripts/run_meta65_supervised.py",
    "scripts/precision_screen_evidence.py", "scripts/run_s8_preflight.py",
    "scripts/mapping_smoke.py", "spatial_meta_ce/train.py", "spatial_meta_ce/meta.py",
    "spatial_meta_ce/data.py", "spatial_meta_ce/model.py", "spatial_meta_ce/checkpoints.py",
    "spatial_meta_ce/manifests.py", "spatial_meta_ce/metrics.py",
)
CACHE_FILES = ("manifest.json", "cards.npy", "offsets.npy", "group0.npy",
               "group1.npy", "group2.npy", "group3.npy")
METRICS = ("all_q50", "all_q95", "four_plus_q50", "four_plus_q95")
LOG_METRICS = ("all_log_q95", "four_plus_log_q95")
CSV_FIELDS = ("arm", "database", "k", "run", "query_digest", "support_digest",
              "query_count", "four_plus_count", *METRICS, *LOG_METRICS,
              "adaptation_seconds", "prediction_seconds", "prediction_file",
              "prediction_sha256")
NPZ_FIELDS = {"log_pred", "true", "n_tables", "query_hash", "query_id"}
RTOL = ATOL = 1e-12


def require(condition, message):
    if not condition:
        raise ValueError(message)


def path_checked(path, base=None):
    path = Path(path).absolute()
    require(not any(p.is_symlink() for p in (path, *path.parents)),
            f"Symlink forbidden: {path}")
    path = path.resolve()
    if base is not None:
        require(path.is_relative_to(base), f"Path escape: {path}")
    return path


def strict_json(text):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, f"Duplicate JSON key: {key}")
            result[key] = value
        return result

    def constant(value):
        raise ValueError(f"Nonfinite JSON constant: {value}")

    result = json.loads(text, object_pairs_hook=pairs, parse_constant=constant)
    # Also rejects numeric overflow such as 1e999, not just NaN tokens.
    json.dumps(result, allow_nan=False)
    return result


def digest(value, compact=True):
    options = {"separators": (",", ":")} if compact else {}
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=False,
                         allow_nan=False, **options).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def sha256(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def valid_sha(value):
    require(isinstance(value, str) and len(value) == 64
            and all(c in "0123456789abcdef" for c in value), "Invalid SHA256")
    return value


class Evidence:
    """Record byte hashes and reject concurrent changes before publishing."""
    def __init__(self):
        self.hashes = {}

    def bind(self, path, expected=None):
        path = path_checked(path)
        actual = sha256(path)
        if expected is not None:
            require(actual == valid_sha(expected), f"SHA mismatch: {path}")
        if path in self.hashes:
            require(actual == self.hashes[path], f"Evidence changed: {path}")
        self.hashes[path] = actual
        return actual

    def read(self, path):
        self.bind(path)
        return strict_json(path.read_text(encoding="utf-8"))

    def recheck(self):
        for path, expected in self.hashes.items():
            require(sha256(path_checked(path)) == expected, f"Evidence changed: {path}")


def number(value, label, minimum=0):
    require(type(value) in (int, float) and math.isfinite(value) and value >= minimum,
            f"Invalid finite {label}: {value!r}")
    return value


def equal_tree(actual, expected, label):
    """Exact structure/types; numerical summaries use the requested tolerance."""
    if isinstance(expected, dict):
        require(isinstance(actual, dict) and set(actual) == set(expected),
                f"Object fields mismatch: {label}")
        for key in expected:
            equal_tree(actual[key], expected[key], f"{label}.{key}")
    elif isinstance(expected, list):
        require(isinstance(actual, list) and len(actual) == len(expected),
                f"List mismatch: {label}")
        for i, value in enumerate(expected):
            equal_tree(actual[i], value, f"{label}[{i}]")
    elif type(expected) is float:
        number(actual, label, minimum=-sys.float_info.max)
        require(math.isclose(actual, expected, rel_tol=RTOL, abs_tol=ATOL),
                f"Numeric mismatch: {label}: {actual!r} != {expected!r}")
    else:
        require(type(actual) is type(expected) and actual == expected,
                f"Exact mismatch: {label}")


def q_metrics(log_pred, true):
    """Standard Q-error; preserve raw predictions, clamp only per convention."""
    require(log_pred.ndim == true.ndim == 1 and log_pred.shape == true.shape
            and len(true) > 0, "Invalid prediction/label dimensions")
    require(log_pred.dtype == true.dtype == np.dtype("float64"), "FP64 required")
    require(np.isfinite(log_pred).all() and np.isfinite(true).all()
            and (true >= 0).all(), "Nonfinite raw prediction or invalid label")
    log_q = np.abs(np.maximum(log_pred, 0) - np.log(np.maximum(true, 1)))
    with np.errstate(over="ignore", invalid="ignore"):
        q = np.exp(log_q)
    require(np.isfinite(q).all(), "Standard Q-error overflow; clipping forbidden")
    return {"q50": float(np.percentile(q, 50)), "q95": float(np.percentile(q, 95)),
            "log_q95": float(np.percentile(log_q, 95))}


def load_csv(path, evidence):
    evidence.bind(path)
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        require(tuple(reader.fieldnames or ()) == CSV_FIELDS, "Unexpected endpoint CSV schema")
        rows = list(reader)
    for row in rows:
        require(set(row) == set(CSV_FIELDS) and all(v is not None for v in row.values()),
                "Malformed CSV row")
        for key in ("k", "run", "query_count", "four_plus_count"):
            text = row[key]
            require(text.isdigit() and str(int(text)) == text, f"Invalid CSV integer: {key}")
            row[key] = int(text)
        for key in (*METRICS, *LOG_METRICS, "adaptation_seconds", "prediction_seconds"):
            row[key] = number(float(row[key]), key, 1 if key in METRICS else 0)
        valid_sha(row["prediction_sha256"])
    return rows


def independent_decision(rows):
    expected = {(arm, db, k, run) for arm in ARMS for db in VAL
                for k in SHOTS for run in range(5)}
    indexed = {}
    for row in rows:
        key = (row["arm"], row["database"], row["k"], row["run"])
        require(key in expected and key not in indexed, "Unexpected/duplicate endpoint cell")
        for metric in METRICS:
            number(row[metric], metric, 1)
        for metric in LOG_METRICS:
            number(row[metric], metric)
        indexed[key] = row
    require(set(indexed) == expected, "Exactly 225 endpoint cells required")
    summaries = {}
    for arm in ARMS:
        cells = [r for key, r in indexed.items() if key[0] == arm]
        groups = []
        for db in VAL:
            for k in SHOTS:
                repetitions = [indexed[arm, db, k, run] for run in range(5)]
                group = {"database": db, "k": k}
                for metric in METRICS:
                    group[f"mean_{metric}"] = math.fsum(r[metric] for r in repetitions) / 5
                for metric in ("all_q95", "four_plus_q95"):
                    group[f"std_{metric}"] = statistics.stdev(r[metric] for r in repetitions)
                groups.append(group)
        summaries[arm] = {
            "macro_four_plus_q95": math.fsum(r["four_plus_q95"] for r in cells) / 75,
            "macro_all_q95": math.fsum(r["all_q95"] for r in cells) / 75,
            "macro_mean_log_q95": math.fsum(r["four_plus_log_q95"] for r in cells) / 75,
            "macro_all_mean_log_q95": math.fsum(r["all_log_q95"] for r in cells) / 75,
            "groups": groups,
        }
    baseline = summaries[ARMS[0]]
    candidates = []
    for arm in ARMS[1:]:
        candidate = summaries[arm]
        improved = candidate["macro_four_plus_q95"] <= baseline["macro_four_plus_q95"] * 0.9
        regressions = []
        for group, reference in zip(candidate["groups"], baseline["groups"]):
            for metric in ("all_q95", "four_plus_q95"):
                if group[f"mean_{metric}"] > reference[f"mean_{metric}"] * 1.1:
                    regressions.append({"database": group["database"], "k": group["k"], "metric": metric})
        # Verify this frozen field only; no auxiliary fraction interpretation.
        candidate.update(relative_improvement=1 - candidate["macro_four_plus_q95"] / baseline["macro_four_plus_q95"],
                         improvement_gate=improved, regressions=regressions,
                         eligible=improved and not regressions)
        if candidate["eligible"]:
            candidates.append(arm)
    parameter_counts = {ARMS[1]: 107521, ARMS[2]: 65}
    candidates.sort(key=lambda arm: (summaries[arm]["macro_four_plus_q95"], parameter_counts[arm], arm))
    return {"schema": 1, "experiment": EXPERIMENT,
            "status": "promising_single_seed" if candidates else "no_candidate",
            "selected_arm": candidates[0] if candidates else None, "candidates": candidates,
            "metric": "75-cell arithmetic macro standard unique-query four_plus_q95",
            "thresholds": {"min_improvement": 0.1, "max_regression": 0.1},
            "arms": summaries,
            "scope": "source-only single-seed engineering screen; not target generalization or significance"}


def source_records(root, run, evidence):
    before = evidence.read(run / "source_before.json")
    after = evidence.read(run / "source_after.json")
    require(digest(before) == digest(after), "Source before/after changed")
    require(set(before) == set(TRAIN + VAL), "Exact 26-source coverage required")
    historical, bundle = {}, {}
    for db in TRAIN + VAL:
        folder = ("p1-mapping-sidecar-preflight-20261003" if db in VAL
                  else "p1-mapping-sidecar-train23-20261004")
        relative = f"实验文档/results/{folder}/{db}/manifest.json"
        path = path_checked(root / relative, root)
        historical[db] = evidence.read(path)
        bundle[relative] = evidence.hashes[path]
        reference, record = historical[db], before[db]
        require(reference["database"] == db and reference["split"] == "pretrain",
                f"Wrong historical source: {db}")
        require(set(record) == {"manifest_hash", "file_sha256", "rows"}
                and type(record["rows"]) is int and record["rows"] == 50000,
                f"Invalid source record: {db}")
        require(record["manifest_hash"] == reference["cache_content_hash"], "Source content SHA mismatch")
        valid_sha(record["manifest_hash"])
        require(set(record["file_sha256"]) == set(CACHE_FILES), "Incomplete source file hashes")
        for name in CACHE_FILES:
            require(valid_sha(record["file_sha256"][name]) == reference["input_sha256"][f"cache/{name}"],
                    f"Source provenance SHA mismatch: {db}/{name}")
    require(digest(bundle) == SOURCE_BINDING, "Frozen source provenance bundle changed")
    require(sum(r["rows"] for r in before.values()) == 1300000, "Source row budget mismatch")
    return before


def load_validation(cache, sources, evidence):
    stores = {}
    # Never enumerate cache: these are the only cache paths opened.
    for db in VAL:
        directory = path_checked(cache / "pretrain" / db, cache)
        for name in ("manifest.json", "cards.npy", "offsets.npy"):
            evidence.bind(directory / name, sources[db]["file_sha256"][name])
        manifest = evidence.read(directory / "manifest.json")
        payload = {k: v for k, v in manifest.items() if k != "content_hash"}
        require(digest(payload, compact=False) == manifest["content_hash"] == sources[db]["manifest_hash"],
                f"Manifest content hash mismatch: {db}")
        require(manifest["schema"] == 1 and manifest["transform"] == "price40-group-logfanout-v1"
                and not manifest.get("provenance", {}).get("excluded"),
                "Unsupported/excluded source cache")
        rows = manifest["rows"]
        cards = np.load(directory / "cards.npy", allow_pickle=False, mmap_mode="r")
        offsets = np.load(directory / "offsets.npy", allow_pickle=False, mmap_mode="r")
        require(len(rows) == 50000 and cards.shape == (50000, 2)
                and cards.dtype == np.dtype("float64"), "Cache FP64 label/row contract mismatch")
        require(np.isfinite(cards).all() and (cards >= 0).all(), "Invalid raw cache labels/PG")
        require(offsets.shape == (50001, 4) and offsets.dtype.kind in "iu"
                and (offsets >= 0).all() and (offsets[0] == 0).all()
                and (offsets[1:] >= offsets[:-1]).all(), "Invalid offsets")
        tables = np.diff(offsets[:, 2]).astype(np.int64)
        require((tables >= 1).all(), "Invalid table count")
        first, ids, unique = {}, set(), []
        for i, row in enumerate(rows):
            identity, query_id = row.get("query_hash"), row.get("id")
            require(isinstance(identity, str) and bool(identity)
                    and isinstance(query_id, str) and query_id.startswith(f"{db}/pretrain/")
                    and query_id not in ids, "Invalid/duplicate source query identity")
            ids.add(query_id)
            if identity in first:
                j = first[identity]
                require(np.array_equal(cards[i], cards[j]) and tables[i] == tables[j],
                        "Conflicting duplicate identity labels/PG/tables")
            else:
                first[identity] = i
                unique.append(i)
        require(type(manifest["unique_queries"]) is int
                and len(unique) == manifest["unique_queries"] == UNIQUE[db], "Unique denominator mismatch")
        require(int((tables[unique] >= 4).sum()) == FOUR_PLUS[db], "Unique four-plus denominator mismatch")
        stores[db] = {"rows": rows, "cards": cards, "tables": tables, "unique": unique}
    return stores


def verify_plans(plans, stores):
    require(set(plans) == set(VAL), "Only three source validation plans permitted")
    for db in VAL:
        require(isinstance(plans[db], list) and len(plans[db]) == 5, "Five support plans required")
        store = stores[db]
        identities = {i: store["rows"][i]["query_hash"] for i in store["unique"]}
        for run_number, plan in enumerate(plans[db]):
            support = random.Random(f"2718:{db}:{run_number}").sample(store["unique"], 100)
            excluded = set(support)
            query = [i for i in store["unique"] if i not in excluded]
            expected = {"database": db, "run": run_number, "support_max": support,
                        "supports": {str(k): support[:k] for k in SHOTS}, "query": query,
                        "query_digest": digest([(i, identities[i]) for i in query]),
                        "support_digests": {str(k): digest([(i, identities[i]) for i in support[:k]]) for k in SHOTS}}
            require(digest(plan) == digest(expected), f"Frozen nested/disjoint support/query plan mismatch: {db}/{run_number}")
            require(len(query) == UNIQUE[db] - 100, "Query count mismatch")


def check_arrays(arrays, store, query):
    require(set(arrays) == NPZ_FIELDS, "Unexpected NPZ arrays")
    n = len(query)
    require(all(a.ndim == 1 and a.shape == (n,) for a in arrays.values()), "NPZ length/shape mismatch")
    require(arrays["log_pred"].dtype == arrays["true"].dtype == np.dtype("float64"), "NPZ FP64 required")
    require(np.isfinite(arrays["log_pred"]).all() and np.isfinite(arrays["true"]).all(), "Nonfinite raw NPZ")
    expected_true = np.asarray(store["cards"][query, 0], dtype=np.float64)
    require(np.array_equal(arrays["true"], expected_true)
            and arrays["true"].tobytes() == expected_true.tobytes(), "Raw FP64 labels differ from source cache")
    require(arrays["n_tables"].dtype.kind in "iu"
            and np.array_equal(arrays["n_tables"], store["tables"][query]), "NPZ table counts differ from offsets")
    for name, field in (("query_hash", "query_hash"), ("query_id", "id")):
        require(arrays[name].dtype.kind == "U"
                and arrays[name].tolist() == [store["rows"][i][field] for i in query],
                f"NPZ {name} differs from source manifest/plan")


def verify_metadata(run, config, sources, plan_sha, evidence):
    records = {}
    hashes = {db: info["manifest_hash"] for db, info in sources.items()}
    for arm in ARMS:
        directory = path_checked(run / arm, run)
        record = evidence.read(directory / "metadata.json")
        expected_config = {**config["training"], "model": {**config["model"], "width": 64},
                           "loss": "smooth_l1", "adapt_scope": config["arms"][arm]["train_scope"],
                           "inner_steps": config["arms"][arm]["train_steps"], "purpose": EXPERIMENT}
        require(digest(record["config"]) == digest(expected_config), f"Training config mismatch: {arm}")
        inner_steps = expected_config["inner_steps"]
        counts = record["counts"]
        require(set(counts) == {"optimizer_steps", "query_presentations", "support_presentations", "backward_calls"}
                and all(type(v) is int for v in counts.values()), "Invalid training counts")
        require(counts["optimizer_steps"] == record["optimizer_steps"] == record["last_step"] == 4000
                and counts["query_presentations"] == 1024000
                and counts["backward_calls"] == (96000 if arm == ARMS[0] else 64000)
                and counts["support_presentations"] > 0
                and counts["support_presentations"] % inner_steps == 0
                and record["support_sample_count"] == counts["support_presentations"] // inner_steps
                and 80000 <= record["support_sample_count"] <= 1600000
                and record["sample_plan_episodes"] == 16000
                and record["parameters"] == 132545 and record["endpoint_cells"] == 75
                and record["arm"] == arm and record["endpoint_plan_digest"] == plan_sha,
                f"Incomplete/inconsistent training budget: {arm}")
        for key in ("optimizer_steps", "last_step", "parameters", "support_sample_count",
                    "sample_plan_episodes", "endpoint_cells"):
            require(type(record[key]) is int, f"Non-integer metadata field: {key}")
        summary = record["summary"]
        require(digest(summary["counts"]) == digest(counts) and summary["last_step"] == 4000
                and summary["parameters"] == 132545 and summary["purpose"] == EXPERIMENT,
                "Training summary count/config mismatch")
        metadata = record["metadata"]
        signature = {k: v for k, v in expected_config.items() if k not in ("steps", "data_config")}
        require(digest(metadata["training_signature"]) == digest(signature)
                and metadata["train_databases"] == list(TRAIN)
                and metadata["validation_databases"] == list(VAL)
                and metadata["manifest_hashes"] == hashes and metadata["method"] == "fomaml"
                and type(metadata["seed"]) is int and metadata["seed"] == 42
                and metadata["purpose"] == EXPERIMENT, "Training metadata source/config mismatch")
        for key in ("sample_plan_digest", "initialization_digest", "endpoint_plan_digest"):
            valid_sha(record[key])
        require(record["checkpoint"] == "last.pt", "Only final last.pt permitted")
        evidence.bind(directory / "last.pt", record["checkpoint_sha256"])
        evidence.bind(directory / "validation_manifest.json", record["validation_plan_digest"])
        for key in ("training_seconds", "endpoint_seconds"):
            number(record[key], key)
        number(summary["total_seconds"], "summary.total_seconds")
        require(summary["total_seconds"] <= record["training_seconds"] + ATOL,
                "Training duration order mismatch")
        records[arm] = record
    for key in ("initialization_digest", "parameters", "sample_plan_digest", "sample_plan_episodes",
                "support_sample_count", "validation_plan_digest", "endpoint_plan_digest", "optimizer_steps"):
        require(len({digest(record[key]) for record in records.values()}) == 1,
                f"Cross-arm pairing mismatch: {key}")
    return records


def verify(root, run, cache, out):
    started = time.monotonic()
    root, cache = path_checked(root), path_checked(cache)
    run = path_checked(run, root / "runs")
    out = path_checked(out)
    require(run != root / "runs" and run.is_dir(), "Expected one completed run directory")
    require(out.suffix == ".json" and not out.exists() and out.parent.is_dir(), "Output must be a new JSON in an existing directory")
    require(not out.is_relative_to(root / "runs") and not out.is_relative_to(cache)
            and not out.is_relative_to(Path("/gly/weilin/PRICE-master")), "Cannot write run/cache/PRICE inputs")
    evidence = Evidence()
    supervisor_path = run.with_name(run.name + ".supervisor.json")
    supervisor = evidence.read(supervisor_path)
    require(supervisor.get("status") == "completed" and type(supervisor.get("exit_code")) is int
            and supervisor["exit_code"] == 0 and type(supervisor.get("returncode")) is int
            and supervisor["returncode"] == 0 and supervisor.get("timed_out") is False
            and supervisor.get("decision_valid") is True and supervisor.get("timeout_seconds") == 7200
            and supervisor.get("grace_seconds") == 5
            and supervisor.get("output") == str(run)
            and not any(k in supervisor for k in ("failure_reason", "cleanup_error", "evidence_error", "interrupted_signal")),
            "Supervisor not completed/exit0/valid/non-timeout; no waiting or training")
    number(supervisor["elapsed_seconds"], "supervisor.elapsed_seconds")
    require(supervisor["elapsed_seconds"] < 7206, "Supervisor duration exceeded bounded timeout+cleanup")
    status = evidence.read(run / "status.json")
    require(status.get("schema") == 1 and status.get("experiment") == EXPERIMENT
            and status.get("status") == "completed"
            and status.get("arms") == dict.fromkeys(ARMS, "completed")
            and status.get("source_rehashed_after") is True
            and not any(k in status for k in ("error", "failure_reason", "error_type")),
            "Run is not a complete three-arm source-rehashed endpoint")
    number(status["elapsed_seconds"], "status.elapsed_seconds")
    require(status["elapsed_seconds"] < 7200
            and status["elapsed_seconds"] <= supervisor["elapsed_seconds"], "Invalid run duration")
    command = supervisor["command"]
    require(isinstance(command, list) and len(command) == 12
            and all(isinstance(v, str) for v in command)
            and command[1:3] == ["-B", str(root / "scripts/run_meta65_screen.py")]
            and command[3] == "--config" and command[5] == "--data-config"
            and command[7:9] == ["--out", str(run)] and command[9] == "--device"
            and command[10] in ("cuda", "cuda:0") and command[11] == "--execute",
            "Unexpected formal supervisor command")
    config_path, data_path = path_checked(command[4], root), path_checked(command[6], root)
    require(config_path == root / "configs/meta65_screen.json"
            and data_path == root / "configs/meta65_server_data.json",
            "Only the frozen formal config paths may be read")
    evidence.bind(config_path, CONFIG_SHA)
    evidence.bind(data_path, DATA_SHA)
    config, data = evidence.read(config_path), evidence.read(data_path)
    require(data["splits"]["train"] == list(TRAIN) and data["splits"]["validation"] == list(VAL)
            and path_checked(data["cache_root"]) == cache, "Cache/source data config mismatch")
    environment = evidence.read(run / "environment.json")
    require(set(environment["code_sha256"]) == set(CODE_FILES), "Runtime code input list mismatch")
    for name in CODE_FILES:
        evidence.bind(path_checked(root / name, root), environment["code_sha256"][name])
    evidence.bind(root / PROTOCOL, PROTOCOL_SHA)
    require(environment["config_sha256"] == CONFIG_SHA and environment["data_config_sha256"] == DATA_SHA
            and environment["protocol_sha256"] == config["protocol_sha256"] == PROTOCOL_SHA
            and environment["source_binding_sha256"] == SOURCE_BINDING
            and environment["external_timeout_required_seconds"] == 7200
            and environment["deterministic_algorithms"] is True
            and environment["OMP_NUM_THREADS"] == environment["MKL_NUM_THREADS"] == "1"
            and environment["device"] == command[10], "Frozen runtime/input mismatch")
    visible = environment["CUDA_VISIBLE_DEVICES"]
    require(isinstance(visible, str) and "," not in visible
            and (visible.isdigit() or visible.startswith("GPU-"))
            and isinstance(environment["gpu_uuid"], str) and environment["gpu_uuid"].startswith("GPU-"),
            "Expected explicitly single visible formal GPU")
    for key in ("python", "executable", "torch", "numpy"):
        require(isinstance(environment[key], str) and bool(environment[key]), f"Missing environment {key}")
    sources = source_records(root, run, evidence)
    stores = load_validation(cache, sources, evidence)
    plans = evidence.read(run / "endpoint_plan.json")
    verify_plans(plans, stores)
    plan_sha = digest(plans)
    require(status["endpoint_plan_digest"] == plan_sha, "Status endpoint plan SHA mismatch")
    records = verify_metadata(run, config, sources, plan_sha, evidence)
    rows = load_csv(run / "endpoint.csv", evidence)
    # Validate the complete matrix before opening any prediction-file path.
    independent_decision(rows)
    for arm in ARMS:
        arm_rows = load_csv(run / arm / "endpoint.csv", evidence)
        require(digest(arm_rows) == digest([r for r in rows if r["arm"] == arm]), "Per-arm endpoint differs from combined CSV")
    recomputed, prediction_hashes = [], set()
    durations = {arm: {"training_seconds": records[arm]["training_seconds"],
                       "endpoint_seconds": records[arm]["endpoint_seconds"],
                       "adaptation_seconds": [], "prediction_seconds": []} for arm in ARMS}
    for row in rows:
        arm, db, k, repetition = (row[key] for key in ("arm", "database", "k", "run"))
        plan, store = plans[db][repetition], stores[db]
        query = plan["query"]
        require(row["query_digest"] == plan["query_digest"]
                and row["support_digest"] == plan["support_digests"][str(k)]
                and row["query_count"] == len(query) == UNIQUE[db] - 100,
                "CSV query/support identity/count mismatch")
        expected_path = f"{arm}/predictions/{db}-k{k}-run{repetition}.npz"
        require(row["prediction_file"] == expected_path, "Prediction path not the frozen source cell")
        prediction_path = path_checked(run / expected_path, run)
        prediction_hashes.add(evidence.bind(prediction_path, row["prediction_sha256"]))
        with np.load(prediction_path, allow_pickle=False) as archive:
            require(len(archive.files) == len(NPZ_FIELDS) and set(archive.files) == NPZ_FIELDS, "Duplicate/unexpected NPZ entries")
            arrays = {key: archive[key] for key in NPZ_FIELDS}
        check_arrays(arrays, store, query)
        mask = arrays["n_tables"] >= 4
        require(int(mask.sum()) > 0 and row["four_plus_count"] == int(mask.sum()), "Four-plus count/coverage mismatch")
        measured = {}
        for prefix, selection in (("all", np.ones(len(query), dtype=bool)), ("four_plus", mask)):
            metrics = q_metrics(arrays["log_pred"][selection], arrays["true"][selection])
            measured.update({f"{prefix}_{name}": value for name, value in metrics.items()})
        for key, value in measured.items():
            equal_tree(row[key], value, f"CSV.{arm}/{db}/{k}/{repetition}.{key}")
        recomputed.append({**row, **measured})
        for key in ("adaptation_seconds", "prediction_seconds"):
            durations[arm][key].append(row[key])
    decision = independent_decision(recomputed)
    equal_tree(evidence.read(run / "decision.json"), decision, "decision")
    for arm in ARMS:
        for key in ("adaptation_seconds", "prediction_seconds"):
            durations[arm][key] = math.fsum(durations[arm][key])
        require(durations[arm]["adaptation_seconds"] + durations[arm]["prediction_seconds"]
                <= records[arm]["endpoint_seconds"] + ATOL, "Endpoint timing order mismatch")
    require(math.fsum(r["training_seconds"] + r["endpoint_seconds"] for r in records.values())
            <= status["elapsed_seconds"] + ATOL, "Total stage timing mismatch")
    evidence.recheck()
    result = {
        "schema": 1, "experiment": EXPERIMENT, "verification": "passed",
        "scope": "source-only single-seed artifact consistency; not significance or target generalization",
        "run": str(run), "checked225": len(recomputed), "bound_prediction_sha256_count": len(rows),
        "unique_prediction_sha256_count": len(prediction_hashes), "endpoint_plan_digest": plan_sha,
        "macros": {arm: {k: v for k, v in decision["arms"][arm].items() if k.startswith("macro_")} for arm in ARMS},
        "gates": {arm: {k: decision["arms"][arm][k] for k in ("improvement_gate", "regressions", "eligible")} for arm in ARMS[1:]},
        "decision": {k: decision[k] for k in ("status", "selected_arm", "candidates")},
        "verified_counts": {"recorded_source_databases": 26, "recorded_source_rows": 1300000,
                            "cache_databases_loaded": list(VAL), "unique_queries": UNIQUE,
                            "query_count_per_cell": {db: UNIQUE[db] - 100 for db in VAL},
                            "cells_per_arm": 75, "checkpoints_sha256_bound": 3,
                            "prediction_query_rows_verified": sum(r["query_count"] for r in recomputed),
                            "four_plus_prediction_rows_verified": sum(r["four_plus_count"] for r in recomputed),
                            "four_plus_count_per_db_run": {db: [next(r["four_plus_count"] for r in recomputed
                                if r["arm"] == ARMS[0] and r["database"] == db and r["k"] == 5
                                and r["run"] == repetition) for repetition in range(5)] for db in VAL},
                            "model_parameters": 132545, "training": {arm: {**records[arm]["counts"],
                            "normalized_support_samples": records[arm]["support_sample_count"],
                            "sample_plan_episodes": 16000} for arm in ARMS}},
        "durations": {"supervisor_seconds": supervisor["elapsed_seconds"],
                      "run_seconds": status["elapsed_seconds"], "arms": durations,
                      "verification_seconds": time.monotonic() - started},
        "limits": {"csv_rtol": RTOL, "csv_atol": ATOL, "training_timeout_seconds": 7200,
                   "macro_max_baseline_multiplier": 0.9, "per_db_k_q95_max_baseline_multiplier": 1.1,
                   "raw_predictions_unclipped": True, "query_rows_filtered": 0,
                   "targets_finetune_test_loaded": False, "checkpoint_deserialized": False,
                   "training_23_cache_and_feature_groups_rehashed": False,
                   "source26_counts_are_hash_bound_records_not_reloaded_arrays": True,
                   "counts_durations_are_consistency_checked_records_not_execution_attestation": True,
                   "no_head_freeze_required": True},
    }
    # No mkdir, overwrite, large artifact, or alternate output on failure.
    with out.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    return result


def self_test():
    class SyntheticTests(unittest.TestCase):
        def rows(self, values=(10.0, 9.0, 9.0)):
            return [{"arm": arm, "database": db, "k": k, "run": run,
                     **{key: values[i] for key in METRICS},
                     **{key: math.log(values[i]) for key in LOG_METRICS}}
                    for i, arm in enumerate(ARMS) for db in VAL for k in SHOTS for run in range(5)]

        def arrays(self):
            store = {"cards": np.array([[1.0, 2.0], [8.0, 4.0]]), "tables": np.array([2, 4]),
                     "rows": [{"query_hash": "q0", "id": "chembl/pretrain/0"},
                              {"query_hash": "q1", "id": "chembl/pretrain/1"}]}
            arrays = {"log_pred": np.array([-2.0, math.log(4)]), "true": np.array([1.0, 8.0]),
                      "n_tables": np.array([2, 4]), "query_hash": np.array(["q0", "q1"]),
                      "query_id": np.array(["chembl/pretrain/0", "chembl/pretrain/1"])}
            return store, arrays

        def test_standard_q_and_log_percentile(self):
            raw = np.array([-2.0, math.log(2), math.log(4), math.log(8)])
            original = raw.copy()
            metrics = q_metrics(raw, np.ones(4))
            self.assertAlmostEqual(metrics["q50"], 3.0)
            self.assertAlmostEqual(metrics["q95"], 7.4)
            self.assertAlmostEqual(metrics["log_q95"], 2.85 * math.log(2))
            self.assertNotAlmostEqual(metrics["log_q95"], math.log(metrics["q95"]))
            np.testing.assert_array_equal(raw, original)
            self.assertEqual(q_metrics(np.array([-3.0]), np.array([0.0]))["q95"], 1.0)

        def test_arithmetic_macro_not_logmean(self):
            rows = self.rows()
            for row in rows:
                if row["arm"] == ARMS[0] and row["database"] == VAL[0]:
                    row["four_plus_q95"] = 40.0
            decision = independent_decision(rows)
            self.assertEqual(decision["arms"][ARMS[0]]["macro_four_plus_q95"], 20.0)

        def test_inclusive_improvement_and_65_parameter_tie(self):
            decision = independent_decision(self.rows())
            self.assertEqual(decision["selected_arm"], ARMS[2])
            self.assertEqual(decision["candidates"], [ARMS[2], ARMS[1]])
            self.assertTrue(decision["arms"][ARMS[1]]["improvement_gate"])

        def test_lower_metric_precedes_parameter_count(self):
            self.assertEqual(independent_decision(self.rows((10.0, 8.0, 9.0)))["selected_arm"], ARMS[1])

        def test_each_db_k_all_and_four_gate(self):
            for metric in ("all_q95", "four_plus_q95"):
                rows = self.rows((10.0, 8.0, 10.0))
                for row in rows:
                    if row["arm"] == ARMS[1] and row["database"] == VAL[0] and row["k"] == 5:
                        row[metric] = 11.0
                self.assertTrue(independent_decision(rows)["arms"][ARMS[1]]["eligible"])
                for row in rows:
                    if row["arm"] == ARMS[1] and row["database"] == VAL[0] and row["k"] == 5:
                        row[metric] = 11.01
                candidate = independent_decision(rows)["arms"][ARMS[1]]
                self.assertTrue(candidate["improvement_gate"])
                self.assertFalse(candidate["eligible"])
                self.assertEqual(candidate["regressions"], [{"database": VAL[0], "k": 5, "metric": metric}])

        def test_no_candidate_and_missing_cell_fail(self):
            self.assertIsNone(independent_decision(self.rows((10.0, 9.01, 10.0)))["selected_arm"])
            with self.assertRaises(ValueError):
                independent_decision(self.rows()[:-1])
            with self.assertRaises(ValueError):
                independent_decision(self.rows() + self.rows()[:1])

        def test_temporary_npz_and_wrong_raw_label_fail(self):
            store, arrays = self.arrays()
            with tempfile.TemporaryDirectory(prefix="meta65-verify-", dir=str(Path(tempfile.gettempdir()).resolve())) as folder:
                path = Path(folder) / "synthetic.npz"
                np.savez(path, **arrays)
                with np.load(path, allow_pickle=False) as archive:
                    loaded = {key: archive[key] for key in archive.files}
                check_arrays(loaded, store, [0, 1])
                loaded["true"][1] = np.nextafter(8.0, 9.0)
                with self.assertRaisesRegex(ValueError, "Raw FP64 labels"):
                    check_arrays(loaded, store, [0, 1])

        def test_wrong_identity_dtype_table_length_fail(self):
            for name, value in (("query_id", np.array(["wrong", "wrong"])),
                                ("query_hash", np.array(["q1", "q0"])),
                                ("true", np.array([1.0, 8.0], dtype=np.float32)),
                                ("n_tables", np.array([2, 5])),
                                ("log_pred", np.array([0.0]))):
                store, arrays = self.arrays()
                arrays[name] = value
                with self.assertRaises(ValueError):
                    check_arrays(arrays, store, [0, 1])

        def test_nonfinite_and_overflow_rejected(self):
            for prediction in (float("nan"), float("inf"), 1000.0):
                with self.assertRaises(ValueError):
                    q_metrics(np.array([prediction]), np.ones(1))
            with self.assertRaises(ValueError):
                q_metrics(np.zeros(1), np.array([-1.0]))

        def test_incomplete_failed_timeout_supervisor_stops_before_cache(self):
            with tempfile.TemporaryDirectory(prefix="meta65-supervisor-", dir=str(Path(tempfile.gettempdir()).resolve())) as folder:
                root = Path(folder)
                run = root / "runs" / "synthetic-run"
                run.mkdir(parents=True)
                cache = root / "cache-does-not-exist"
                out = root / "must-not-be-written.json"
                supervisor_path = run.with_name(run.name + ".supervisor.json")
                valid = {"status": "completed", "returncode": 0, "exit_code": 0,
                         "timed_out": False, "decision_valid": True, "timeout_seconds": 7200,
                         "grace_seconds": 5, "output": str(run)}
                for change in ({"status": "running"}, {"status": "failed"},
                               {"timed_out": True}, {"exit_code": 1},
                               {"decision_valid": False}, {"returncode": -9}):
                    supervisor_path.write_text(json.dumps({**valid, **change}), encoding="utf-8")
                    with self.assertRaisesRegex(ValueError, "Supervisor not completed"):
                        verify(root, run, cache, out)
                    self.assertFalse(out.exists())
                    self.assertFalse(cache.exists())

        def test_strict_json_and_sha_binding(self):
            for text in ('{"a":1,"a":2}', '{"a":NaN}', '{"a":1e999}'):
                with self.assertRaises(ValueError):
                    strict_json(text)
            with tempfile.TemporaryDirectory(prefix="meta65-hash-", dir=str(Path(tempfile.gettempdir()).resolve())) as folder:
                path = Path(folder) / "synthetic.json"
                path.write_text('{"a":1}', encoding="utf-8")
                evidence = Evidence()
                self.assertEqual(evidence.read(path), {"a": 1})
                path.write_text('{"a":2}', encoding="utf-8")
                with self.assertRaises(ValueError):
                    evidence.recheck()

    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(SyntheticTests))
    return 0 if result.wasSuccessful() else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--root", type=Path)
    parser.add_argument("--run", type=Path)
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args(argv)
    if args.self_test:
        if any(getattr(args, k) is not None for k in ("root", "run", "cache", "out")):
            parser.error("--self-test is synthetic-only and cannot accompany formal input/output flags")
        return self_test()
    if any(getattr(args, k) is None for k in ("root", "run", "cache", "out")):
        parser.error("formal verification requires --root --run --cache --out")
    try:
        result = verify(args.root, args.run, args.cache, args.out)
    except (ValueError, KeyError, TypeError, OSError, OverflowError, csv.Error) as error:
        print(f"Verification failed; no result published: {error}", file=sys.stderr)
        return 1
    print(json.dumps({"verification": result["verification"], "checked225": result["checked225"],
                      "out": str(path_checked(args.out))}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
