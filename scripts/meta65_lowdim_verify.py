"""Independently certify archived Meta65 Gate 1 predictions, without Torch.

Only copied manifest/cards/offsets inputs are required. Recorded checkpoint and
feature hashes do not substitute for reloading those unavailable bytes. No
runner, model, training, or historical verifier implementation is imported.
"""
import argparse
import csv
import hashlib
import json
import math
import platform
import re
import zipfile
from pathlib import Path

import numpy as np

EXPERIMENT = "meta65-lowdim-gate1-20261018"
DATABASES = ("chembl", "legalacts", "telstra")
TARGETS = ("imdb", "stats", "genome", "ergastf1")
RUNS = tuple(range(5))
UNIQUE = {"chembl": 50000, "legalacts": 49504, "telstra": 50000}
OPERATORS = ("none", "last3_gradient_smooth_l1", "last3_gradient_mse", "last3_ridge_mse")
PAIRINGS = (
    ("last3_gradient_smooth_l1_vs_none", OPERATORS[1], "none"),
    ("last3_gradient_mse_vs_none", OPERATORS[2], "none"),
    ("last3_ridge_mse_vs_none", OPERATORS[3], "none"),
    ("ridge_vs_gradient_mse", OPERATORS[3], OPERATORS[2]),
    ("loss_within_gradient", OPERATORS[2], OPERATORS[1]),
    ("combined_solver_and_loss", OPERATORS[3], OPERATORS[1]),
)
ATTRIBUTIONS = ("solver_and_proximal_regularization", "loss", "combined_not_solver_only")
BUCKETS = ("all", "four_plus", "low4")
CHECKPOINT_SHA256 = "586bc0d0a8883c1e4cf0645c26a389d95cb5f80751df2b0f0f85c02c64add63f"
METADATA_SHA256 = "186d69d63d34c9f63f2d1d622c78816ebd9fc5e188c844b1e541dc74a3656dcc"
PLAN_SHA256 = "ebeb178de15b134722bd9fbc24ff4b4752f795a311580c1a17cb23c550bfd0f0"
REGISTRY_SHA256 = "651f70197fd33a387b5a5c363ee203de57aff6d07cb8390487ee099ee0870dd3"
RTOL = ATOL = 1e-12
MODEL = {"arch": "sum", "use_pg": True, "width": 64, "layers": 2, "heads": 4}
DEFAULT_EVIDENCE = Path(__file__).resolve().parents[1] / "reports/meta65-lowdim-gate1-20261018"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def safe_path(path):
    path = Path(path).absolute()
    require(not any(p.is_symlink() for p in (path, *path.parents)),
            f"Symlink forbidden: {path}")
    return path.resolve()


def sha256(path):
    value = hashlib.sha256()
    with safe_path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    allow_nan=False, separators=(",", ":")).encode()).hexdigest()


def read_json(path):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, f"Duplicate JSON key: {key}")
            result[key] = value
        return result

    def constant(value):
        raise ValueError(f"Nonfinite JSON value: {value}")

    return json.loads(safe_path(path).read_text(), object_pairs_hook=pairs,
                      parse_constant=constant)


class Checks:
    def __init__(self):
        self.count = 0
        self.max_abs = 0.0
        self.max_rel = 0.0

    def equal(self, actual, expected, label):
        self.count += 1
        if isinstance(expected, dict):
            require(isinstance(actual, dict) and actual.keys() == expected.keys(),
                    f"{label}: object keys differ")
            for key in expected:
                self.equal(actual[key], expected[key], f"{label}.{key}")
        elif isinstance(expected, list):
            require(isinstance(actual, list) and len(actual) == len(expected),
                    f"{label}: list length differs")
            for index, (left, right) in enumerate(zip(actual, expected)):
                self.equal(left, right, f"{label}[{index}]")
        elif isinstance(expected, float):
            self.close(actual, expected, label)
        else:
            require(type(actual) is type(expected) and actual == expected,
                    f"{label}: exact value differs")

    def close(self, actual, expected, label):
        self.count += 1
        left, right = np.asarray(actual), np.asarray(expected)
        require(left.dtype.kind in "fiu" and right.dtype.kind in "fiu"
                and left.shape == right.shape and np.isfinite(left).all()
                and np.isfinite(right).all(), f"{label}: invalid numeric values")
        delta = np.abs(left - right)
        if delta.size:
            self.max_abs = max(self.max_abs, float(np.max(delta)))
            self.max_rel = max(self.max_rel, float(np.max(delta / np.maximum(np.abs(right), ATOL))))
        require(np.allclose(left, right, rtol=RTOL, atol=ATOL), f"{label}: float mismatch")


def metric(q):
    require(len(q) > 0, "Required bucket is empty")
    return {"count": len(q), "status": "below_min_n" if len(q) < 30 else "ok",
            "q50": float(np.percentile(q, 50, method="linear")),
            "q95": float(np.percentile(q, 95, method="linear"))}


def pair_metric(left, right, left_log, right_log):
    lm, rm = metric(left), metric(right)
    delta = left_log - right_log
    return {"query_count": len(left), "status": lm["status"],
            "q50_ratio": lm["q50"] / rm["q50"], "q95_ratio": lm["q95"] / rm["q95"],
            "mean_delta_log_q": float(np.mean(delta)),
            "median_delta_log_q": float(np.median(delta)),
            "mean_delta_q": float(np.mean(left - right)),
            "improved_count": int(np.sum(left < right)),
            "worsened_count": int(np.sum(left > right)),
            "tied_count": int(np.sum(left == right)),
            "improved_fraction": float(np.mean(left < right)), "lower_q_error_is_better": True}


def index_records(records, fields, expected, label):
    require(isinstance(records, list), f"{label}: expected list")
    result = {}
    for row in records:
        key = tuple(row.get(field) for field in fields)
        require(all(type(row.get(field)) is int for field in fields if field == "run"),
                f"{label}: invalid run type")
        require(key in expected and key not in result, f"{label}: duplicate or unexpected cell/run")
        result[key] = row
    require(result.keys() == expected, f"{label}: incomplete matrix")
    return result


def check_plan(plan, rows, database, run, checks):
    require(set(plan) == {"database", "run", "support_max", "supports", "query",
                          "query_digest", "support_digests"}, "Plan schema changed")
    checks.equal(plan["database"], database, "plan database")
    checks.equal(plan["run"], run, "plan run")
    first = {}
    for i, row in enumerate(rows):
        first.setdefault(row["query_hash"], i)
    require(len(first) == UNIQUE[database], "Unique-query denominator changed")
    support = plan["support_max"]
    query = plan["query"]
    for values in (support, query):
        require(isinstance(values, list) and all(type(i) is int and 0 <= i < len(rows) for i in values)
                and len(set(values)) == len(values), "Invalid plan indices")
    require(len(support) == 100 and set(support) <= set(first.values()), "Invalid full100 support pool")
    excluded = set(support)
    checks.equal(query, [i for i in first.values() if i not in excluded], "Full query denominator")
    checks.equal(plan["supports"], {str(k): support[:k] for k in (5, 10, 20, 50, 100)},
                 "Nested support prefixes")

    def identity_digest(indices):
        return digest([(i, rows[i]["query_hash"]) for i in indices])

    checks.equal(plan["query_digest"], identity_digest(query), "query digest")
    checks.equal(plan["support_digests"], {str(k): identity_digest(support[:k])
                                         for k in (5, 10, 20, 50, 100)}, "support digests")
    return support[:20], query


def check_audit(audit, operator, support, rows, cards, checks):
    checks.equal(audit["operator"], operator, "audit operator")
    checks.equal(audit["training"], False, "audit training")
    checks.equal(audit["support_count"], 20, "audit support count")
    checks.equal(audit["feature_dim"], 64, "audit feature dim")
    method = "identity" if operator == "none" else (
        "proximal_ridge_last" if operator == OPERATORS[3] else "gradient_last")
    checks.equal(audit["method"], method, "audit method")
    checks.equal(audit["condition_number_kind"], "diagnostic_XtX_plus_lambda_I", "condition kind")
    names = audit["changed_parameter_names"]
    require(isinstance(names, list) and len(names) == len(set(names))
            and set(names) <= {"head.4.weight", "head.4.bias"}, "Unexpected changed parameters")
    for field in ("update_norm", "condition_number", "design_condition_number"):
        require(type(audit[field]) in (float, int) and math.isfinite(audit[field])
                and audit[field] >= 0, f"Invalid audit {field}")
    records = audit["support_residuals"]
    require(len(records) == 20, "Missing support residuals")
    # The residual head uses float32 log labels and PG anchors, widened before subtraction.
    targets = np.log(np.maximum(cards[support], 1)).astype(np.float32).astype(np.float64)
    targets = targets[:, 0] - targets[:, 1]
    checks.close([row["target"] for row in records], targets, "support targets from cards")
    before, after = [], []
    for row, index in zip(records, support):
        checks.equal(row["id"], rows[index]["id"], "support id")
        checks.equal(row["query_hash"], rows[index]["query_hash"], "support hash")
        checks.close(row["residual_before"], row["before"] - row["target"], "support before residual")
        checks.close(row["residual_after"], row["after"] - row["target"], "support after residual")
        before.append(row["residual_before"])
        after.append(row["residual_after"])
    for suffix, residual in (("before", np.asarray(before)), ("after", np.asarray(after))):
        checks.close(audit[f"support_mse_{suffix}"], float(np.mean(residual ** 2)), "support MSE")
        absolute = np.abs(residual)
        smooth = np.where(absolute < 1, 0.5 * residual ** 2, absolute - 0.5)
        checks.close(audit[f"support_smooth_l1_{suffix}"], float(np.mean(smooth)), "support SmoothL1")
    checks.equal(audit["support_residual_improved"],
                 audit["support_mse_after"] < audit["support_mse_before"], "support improvement flag")
    if operator == "none":
        checks.equal(names, [], "none parameters")
        checks.close(audit["update_norm"], 0.0, "none update norm")
        checks.close(before, after, "none support predictions")
    if operator == OPERATORS[3]:
        checks.close(audit["proximal_lambda"], 1.0, "ridge lambda")
        require(audit["support_residual_improved"], "Ridge support did not improve")
        for field in ("solver_condition_number", "solver_support_mse_before",
                      "solver_support_mse_after", "solver_update_norm"):
            require(type(audit[field]) in (float, int) and math.isfinite(audit[field])
                    and audit[field] >= 0, f"Invalid {field}")
    return [(row["target"], row["before"]) for row in records]


def _verify(raw_root, *, evidence_root=None):
    root = safe_path(raw_root)
    evidence = safe_path(DEFAULT_EVIDENCE if evidence_root is None else evidence_root)
    run_root, input_root = root / "run", root / "inputs"
    checks = Checks()
    artifacts = {}

    def load(path):
        artifacts[str(path.relative_to(root))] = sha256(path)
        return read_json(path)

    names = ("status.json", "inputs.json", "cells.json", "screen_plan.json",
             "macro.json", "paired_summary.json", "environment.json")
    documents = {name: load(run_root / name) for name in names}
    status, inputs = documents["status.json"], documents["inputs.json"]
    for name in ("status.json", "cells.json", "screen_plan.json", "macro.json", "paired_summary.json"):
        checks.equal(documents[name]["schema"], 1, f"{name} schema")
        checks.equal(documents[name]["experiment"], EXPERIMENT, f"{name} experiment")
    for field, value in {"status": "completed", "k": 20, "operators": list(OPERATORS),
                         "training": False, "target_read": False, "completed_cells": 60,
                         "gate": "screen_completed_not_selected", "selected_lambda": 1.0,
                         "lambda_registry_sha256": REGISTRY_SHA256}.items():
        checks.equal(status[field], value, f"status.{field}")
    checks.equal(documents["environment.json"]["training"], False, "environment.training")
    for name, expected in (("checkpoint", CHECKPOINT_SHA256), ("plan", PLAN_SHA256),
                           ("lambda_registry", REGISTRY_SHA256)):
        checks.equal(inputs[name]["sha256"], expected, f"Frozen {name} SHA256")
    registry_path = Path(__file__).resolve().parents[1] / "configs/meta65_lowdim_gate1_lambda.json"
    require(sha256(registry_path) == REGISTRY_SHA256, "Registry bytes changed")
    checks.equal(inputs["lambda_registry"]["value"], read_json(registry_path), "lambda registry")
    checks.equal(inputs["source_before"], inputs["source_after"], "Source before/after")
    require(set(inputs["source_before"]) == set(DATABASES), "Unexpected source database")
    plan_path = input_root / "endpoint_plan.json"
    require(sha256(plan_path) == PLAN_SHA256, "Frozen plan SHA256 mismatch")
    plan = load(plan_path)
    require(set(plan) == set(DATABASES), "Target or unknown plan database")
    metadata_path = input_root / "checkpoint-metadata.json"
    require(sha256(metadata_path) == METADATA_SHA256, "Frozen metadata SHA256 mismatch")
    metadata = load(metadata_path)
    checks.equal(metadata["checkpoint_sha256"], CHECKPOINT_SHA256, "checkpoint metadata hash")
    checks.equal(metadata["last_step"], 4000, "checkpoint step")
    checks.equal(metadata["config"]["seed"], 42, "checkpoint seed")
    checks.equal(metadata["config"]["model"], MODEL, "checkpoint model")
    checks.equal(metadata["metadata"]["validation_databases"], list(DATABASES), "validation split")
    train = metadata["metadata"]["train_databases"]
    require(len(train) == 23 and len(set(train)) == 23
            and not set(train) & {*DATABASES, *TARGETS}
            and not metadata["metadata"].get("adapted_database"), "Invalid checkpoint split/adaptation")
    expected_keys = {(op, db, run) for op in OPERATORS for db in DATABASES for run in RUNS}
    cells = index_records(documents["cells.json"]["cells"], ("operator", "database", "run"),
                          expected_keys, "cells")
    pair_keys = {(db, run) for db in DATABASES for run in RUNS}
    paired = documents["paired_summary.json"]
    pair_runs = index_records(paired["runs"], ("database", "run"), pair_keys, "paired runs")
    require(len(paired["contrasts"]) == 3, "Contrast metadata missing")
    for contrast, (name, left, right), attribution in zip(paired["contrasts"], PAIRINGS[3:], ATTRIBUTIONS):
        for field, value in (("name", name), ("left", left), ("right", right), ("attribution", attribution)):
            checks.equal(contrast[field], value, "contrast attribution")
    expected_npz = {f"pair-{db}-run{run}.npz" for db, run in pair_keys}
    require({p.name for p in run_root.glob("*.npz")} == expected_npz, "Pair NPZ inventory changed")
    screen = documents["screen_plan.json"]
    for field, value in (("k", 20), ("support_max", 100), ("runs", list(RUNS)),
                         ("databases", list(DATABASES)), ("plan_sha256", PLAN_SHA256)):
        checks.equal(screen[field], value, f"screen.{field}")
    require(set(screen["runs_binding"]) == set(DATABASES), "Unexpected screen database")
    all_metrics, all_pairs, state_digests = {}, {}, set()
    for database in DATABASES:
        require(len(plan[database]) == 5, "Plan requires five runs")
        source = inputs["source_before"][database]
        cache = input_root / database
        for filename in ("manifest.json", "cards.npy", "offsets.npy"):
            path = cache / filename
            actual_sha = sha256(path)
            artifacts[str(path.relative_to(root))] = actual_sha
            checks.equal(actual_sha, source["file_sha256"][filename], "Copied source SHA256")
        manifest = read_json(cache / "manifest.json")
        checks.equal(manifest["content_hash"], source["manifest_content_hash"], "Manifest content hash")
        checks.equal(manifest["provenance"]["source"]["database"], database, "Manifest source")
        checks.equal(manifest["provenance"]["source"]["split"], "pretrain", "Manifest split")
        checks.equal(manifest["provenance"]["excluded"], [], "Excluded source rows")
        rows = manifest["rows"]
        require(all(isinstance(row["query_hash"], str) and row["query_hash"]
                    and re.fullmatch(rf"{database}/pretrain/[1-9][0-9]*", row["id"]) for row in rows)
                and len({row["id"] for row in rows}) == len(rows), "Invalid source identities")
        cards = np.load(cache / "cards.npy", allow_pickle=False)
        offsets = np.load(cache / "offsets.npy", allow_pickle=False)
        require(cards.dtype == np.float64 and cards.shape == (len(rows), 2)
                and np.isfinite(cards).all() and (cards >= 0).all(), "Invalid source cards")
        require(offsets.dtype.kind in "iu" and offsets.shape == (len(rows) + 1, 4)
                and (offsets[0] == 0).all() and (np.diff(offsets, axis=0) >= 0).all(),
                "Invalid source offsets")
        tables = np.diff(offsets[:, 2])
        require((tables >= 1).all() and (tables <= np.iinfo(np.int16).max).all(), "Invalid table counts")
        bindings = index_records(screen["runs_binding"][database], ("database", "run"),
                                 {(database, r) for r in RUNS}, "screen runs")
        for run in RUNS:
            frozen = plan[database][run]
            support, query = check_plan(frozen, rows, database, run, checks)
            checks.equal(bindings[(database, run)], {
                "database": database, "run": run, "support": support, "query_count": len(query),
                "query_digest": frozen["query_digest"], "support_digest": frozen["support_digests"]["20"]
            }, "screen binding")
            path = run_root / f"pair-{database}-run{run}.npz"
            artifacts[str(path.relative_to(root))] = sha256(path)
            with np.load(path, allow_pickle=False) as archive:
                data = {key: archive[key] for key in archive.files}
                require(len(archive.files) == len(data), "Duplicate NPZ fields")
            dtypes = {"query_hash": "S64", "query_id": "S128", "true": "float64",
                      "n_tables": "int16", "four_plus": "bool", "low4": "bool"}
            for op in OPERATORS:
                dtypes.update({f"{op}_log_pred": "float64", f"{op}_q": "float64"})
            for name, _, _ in PAIRINGS:
                dtypes.update({f"{name}_delta_log_q": "float64", f"{name}_q_ratio": "float64"})
            require(data.keys() == dtypes.keys(), "Unexpected NPZ fields")
            for name, dtype in dtypes.items():
                require(data[name].dtype == np.dtype(dtype) and data[name].shape == (len(query),),
                        f"Invalid NPZ dtype/shape: {name}")
                if data[name].dtype.kind == "f":
                    require(np.isfinite(data[name]).all(), f"Nonfinite NPZ: {name}")
            expected_identity = {"query_hash": [rows[i]["query_hash"] for i in query],
                                 "query_id": [rows[i]["id"] for i in query],
                                 "true": cards[query, 0], "n_tables": tables[query]}
            for name, values in expected_identity.items():
                require(np.array_equal(data[name].astype(str) if name.startswith("query_") else data[name],
                                       values), f"NPZ identity mismatch: {name}")
            true = data["true"]
            masks = {"all": np.ones(len(query), dtype=bool), "four_plus": tables[query] >= 4,
                     "low4": (tables[query] >= 4) & (true < 100)}
            for bucket in BUCKETS[1:]:
                require(np.array_equal(data[bucket], masks[bucket]), f"NPZ mask mismatch: {bucket}")
            q, log_q, reference = {}, {}, None
            for op in OPERATORS:
                log_q[op] = np.abs(np.maximum(data[f"{op}_log_pred"], 0) - np.log(np.maximum(true, 1)))
                with np.errstate(over="raise", invalid="raise"):
                    q[op] = np.exp(log_q[op])
                checks.close(data[f"{op}_q"], q[op], "NPZ Q-error")
                cell = cells[(op, database, run)]
                for field, value in (("query_count", len(query)), ("query_digest", frozen["query_digest"]),
                                     ("support_digest", frozen["support_digests"]["20"]),
                                     ("prediction_file", path.name)):
                    checks.equal(cell[field], value, f"cell.{field}")
                state = cell["state_before"]
                require(isinstance(state, str) and re.fullmatch("[0-9a-f]{64}", state), "Invalid state digest")
                checks.equal(cell["state_after"], state, "checkpoint state after")
                checks.equal(cell["checkpoint_state_digest"], state, "checkpoint state digest")
                state_digests.add(state)
                current = check_audit(cell["support_audit"], op, support, rows, cards, checks)
                if reference is None:
                    reference = current
                checks.close(current, reference, "Shared support targets/initial predictions")
                metrics = {bucket: metric(q[op][mask]) for bucket, mask in masks.items()}
                checks.equal(cell["metrics"], metrics, "cell metrics")
                all_metrics[(op, database, run)] = metrics
            run_pairs = {}
            for name, left, right in PAIRINGS:
                checks.close(data[f"{name}_delta_log_q"], log_q[left] - log_q[right], "NPZ paired delta")
                checks.close(data[f"{name}_q_ratio"], q[left] / q[right], "NPZ paired ratio")
                run_pairs[name] = {bucket: pair_metric(q[left][mask], q[right][mask],
                                                      log_q[left][mask], log_q[right][mask])
                                   for bucket, mask in masks.items()}
            identity_digest = digest(list(zip(data["query_hash"].astype(str).tolist(),
                                             data["query_id"].astype(str).tolist())))
            checks.equal(pair_runs[(database, run)], {"database": database, "run": run,
                         "query_digest": identity_digest, "pairs": run_pairs}, "paired run")
            all_pairs[(database, run)] = run_pairs
    require(len(state_digests) == 1, "Checkpoint initialization differs across cells")
    summaries = {}
    for op in OPERATORS:
        per_database = {db: {f"{bucket}_{field}": math.fsum(
            all_metrics[(op, db, run)][bucket][field] for run in RUNS) / 5
            for bucket in BUCKETS for field in ("q50", "q95")} for db in DATABASES}
        summaries[op] = {"per_database": per_database, "macro": {
            field: math.fsum(per_database[db][field] for db in DATABASES) / 3
            for field in per_database[DATABASES[0]]}, "cell_count": 15}
    checks.equal(documents["macro.json"]["operators"], summaries, "operator macro")
    checks.equal(documents["macro.json"]["k"], 20, "macro K")
    checks.equal(documents["macro.json"]["selection"], "not_performed_by_runner", "selection")
    fields = ("q50_ratio", "q95_ratio", "mean_delta_log_q", "mean_delta_q", "improved_fraction")
    pair_macro = {name: {bucket: {f"arithmetic_mean_{field}": math.fsum(
        all_pairs[(db, run)][name][bucket][field] for db in DATABASES for run in RUNS) / 15
        for field in fields} for bucket in BUCKETS} for name, _, _ in PAIRINGS}
    checks.equal(paired["macro"], pair_macro, "pair macro")
    csv_path = run_root / "metrics.csv"
    artifacts["run/metrics.csv"] = sha256(csv_path)
    with csv_path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        require(len(reader.fieldnames) == len(set(reader.fieldnames)), "Duplicate CSV columns")
        csv_rows = list(reader)
    require(len(csv_rows) == 60, "CSV requires 60 cells")
    seen = set()
    for row in csv_rows:
        key = (row["operator"], row["database"], int(row["run"]))
        require(key in cells and key not in seen, "Duplicate or unexpected CSV cell")
        seen.add(key)
        cell = cells[key]
        expected = {"operator": key[0], "database": key[1], "run": str(key[2]),
                    "query_count": str(cell["query_count"])}
        for bucket in BUCKETS:
            for field in ("q50", "q95"):
                expected[f"{bucket}_{field}"] = cell["metrics"][bucket][field]
        expected.update({f"{b}_count": str(cell["metrics"][b]["count"]) for b in BUCKETS[1:]})
        for field in ("support_mse_before", "support_mse_after", "support_smooth_l1_before",
                      "support_smooth_l1_after", "update_norm", "condition_number", "solver_condition_number"):
            expected[field] = cell["support_audit"].get(field, "")
        require(row.keys() == expected.keys(), "CSV columns changed")
        for field, value in expected.items():
            if isinstance(value, (float, int)):
                checks.close(float(row[field]), value, f"CSV {field}")
            else:
                checks.equal(row[field], value, f"CSV {field}")
    evidence_hashes = {}
    for name in (*names[:2], "screen_plan.json", "macro.json", "paired_summary.json", "environment.json", "metrics.csv"):
        evidence_hashes[name] = sha256(evidence / name)
        if name == "metrics.csv":
            # Git normalizes CRLF; compare text without weakening field validation above.
            checks.equal(safe_path(evidence / name).read_text(), csv_path.read_text(), "Archived metrics.csv")
        else:
            require(evidence_hashes[name] == artifacts[f"run/{name}"], f"Archived evidence mismatch: {name}")
    target_hits = []
    for path in sorted((*run_root.glob("*.json"), *run_root.glob("*.csv"))):
        safe_path(path)
        if re.search(r"\b(?:imdb|stats|genome|ergastf1)\b", path.read_text(), re.IGNORECASE):
            target_hits.append(path.name)
    require(not target_hits, f"Target text hits: {target_hits}")
    require(all(sha256(root / name) == value for name, value in artifacts.items()),
            "Input artifacts changed during verification")
    return {"schema": 1, "status": "verified", "experiment": EXPERIMENT,
            "verifier_sha256": sha256(__file__), "checks_passed": checks.count,
            "check_count_unit": "recursive scalar/container and numeric-array comparisons, not independent tests",
            "environment": {"python": platform.python_version(), "numpy": np.__version__, "device": "cpu"},
            "cell_count": 60, "pair_npz_count": 15, "training": False, "target_read": False,
            "float_tolerance": {"rtol": RTOL, "atol": ATOL},
            "max_float_error": {"absolute": checks.max_abs, "relative": checks.max_rel},
            "recomputed_macro": summaries, "pair_macro": pair_macro,
            "artifact_sha256": artifacts, "target_scan": {"text_hits": [], "target_query_ids": 0},
            "evidence_mirror_checked": True, "evidence_sha256": evidence_hashes,
            "limitations": ["Checkpoint bytes and group0-3 feature arrays are not in this copy; their recorded hashes are not revalidated.",
                            "Support arithmetic is checked, not the model forward, optimizer, ridge solve, or parameter update norm.",
                            "Target isolation is an artifact/identity check, not OS-level I/O tracing.",
                            "This rebuild does not reproduce the historical verifier's implementation or assertion count."]}


def verify(raw_root, *, evidence_root=None):
    """Return a new certificate or fail closed on missing/malformed artifacts."""
    try:
        return _verify(raw_root, evidence_root=evidence_root)
    except (OSError, KeyError, TypeError, IndexError, AttributeError, OverflowError,
            FloatingPointError, zipfile.BadZipFile, EOFError) as error:
        raise ValueError(f"Invalid or missing verification artifact: {error}") from error


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", required=True, type=Path)
    parser.add_argument("--evidence-root", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        output = safe_path(args.output)
        root = safe_path(args.raw_root)
        require(not output.exists(), "Output already exists")
        require(not output.is_relative_to(root), "Output must be outside raw inputs")
        evidence = safe_path(DEFAULT_EVIDENCE if args.evidence_root is None else args.evidence_root)
        require(not output.is_relative_to(evidence), "Output must be outside evidence inputs")
        result = verify(root, evidence_root=args.evidence_root)
        with output.open("x") as stream:
            stream.write(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
        print(f"verified: {result['cell_count']} cells, {result['pair_npz_count']} NPZ -> {output}")
        return 0
    except (ValueError, OSError, KeyError, TypeError, OverflowError, FloatingPointError) as error:
        parser.exit(1, f"verification failed: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
