#!/usr/bin/env python3
"""Fixed-checkpoint evaluation only. Formal execution needs a separate approval."""
import argparse
import hashlib
import json
import os
import platform
import resource
import sys
import time
from collections import OrderedDict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DATABASES = ("chembl", "legalacts", "telstra")
CHECKPOINTS = {"A": "A_all5_head3", "B": "B_head3_head3", "C": "C_last3_last3"}
OPERATORS = ("none", "head3", "last3")
COUNTS = {"chembl": 49900, "legalacts": 49404, "telstra": 49900}
CHECKPOINT_SHA = {
    "A": "ae0c386d35ac368964b4160d634ca063d7801c65e5296ac679c459dd01951b12",
    "B": "a2d777a4dcaad6d23c3f1212929d062768f5157a3fc37bce9909903a22f18db9",
    "C": "58f2f4db6f54cd4578fd6127d5a4aa350dbdc2268a36a057b56f7161cb6097a4",
}
RETAINED = Path("/gly/weilin/spatial-meta-ce-meta65-20261009-frozen/runs/meta65-source-screen-20261009")
CACHE = Path("/gly/weilin/spatial-meta-ce/artifacts/cache/pretrain")
PLAN_SHA = "ebeb178de15b134722bd9fbc24ff4b4752f795a311580c1a17cb23c550bfd0f0"
PROTOCOL = "实验文档/56-Meta65同初始化适应算子最小受控协议.md"
PROTOCOL_SHA = "cc87ef93451e2664aba4aa51257a1cdba47c6638c004cb8e4b943773171ca4d2"
FROZEN = {
    "schema": 1, "experiment": "meta65-joint-control-20261010", "k": 20,
    "databases": list(DATABASES), "checkpoints": list(CHECKPOINTS), "operators": list(OPERATORS),
    "runs": list(range(5)), "batch_size": 512, "inner_lr": 0.001, "inner_steps": 3,
    "loss": "smooth_l1", "model": {"arch": "sum", "use_pg": True, "width": 64, "layers": 2, "heads": 4},
    "protocol_file": PROTOCOL, "protocol_sha256": PROTOCOL_SHA,
    "limits": {"total_seconds": 4800, "preparation_seconds": 300,
               "evaluation_seconds": 3600, "certification_seconds": 900,
               "rss_bytes": 4 * 1024**3, "vram_bytes": 8 * 1024**3,
               "output_bytes": 1024**3},
    "cpu_preflight": {"micro_queries": 32, "total_seconds": 900,
                      "preparation_seconds": 300, "evaluation_seconds": 120,
                      "certification_seconds": 300},
}
CODE_FILES = (
    "scripts/meta65_joint_runner.py", "scripts/meta65_joint_supervisor.py", "scripts/meta65_joint_verify.py",
    "spatial_meta_ce/data.py", "spatial_meta_ce/manifests.py", "spatial_meta_ce/mappings.py",
    "spatial_meta_ce/model.py", "spatial_meta_ce/meta.py", "spatial_meta_ce/checkpoints.py",
)
PROVENANCE = (
    "实验文档/results/meta65-source-screen-20261009/source-before.json",
    "实验文档/results/meta65-telstra-diagnosis-20261010/telstra_diagnosis.json",
    "实验文档/results/meta65-endpoint-completion-20261009/inputs.json",
)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def read_json(path):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result

    def constant(value):
        raise ValueError(f"Nonfinite JSON: {value}")

    return json.loads(Path(path).read_text(), object_pairs_hook=pairs, parse_constant=constant)


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    if temporary.exists():
        raise ValueError("Unexpected temporary output")
    temporary.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + "\n")
    temporary.replace(path)


def checked_path(path):
    path = Path(path).absolute()
    # `/var` is a system alias for `/private/var` on macOS; rejecting every
    # symlink in the parent chain would reject otherwise safe temp paths.
    if any(item.is_symlink() and item not in (Path("/var"), Path("/tmp"))
           for item in (path, *path.parents)):
        raise ValueError(f"Symlink forbidden: {path}")
    return path.resolve()


def safe_output(path, inputs):
    path = checked_path(path)
    if path.exists() or any(path.is_relative_to(checked_path(item))
                            or checked_path(item).is_relative_to(path) for item in inputs):
        raise ValueError("Output exists or overlaps frozen inputs")
    if "PRICE-master" in path.parts or path.is_relative_to(ROOT / "spatial_meta_ce"):
        raise ValueError("Read-only/source output path")
    return path


def validate_config(value):
    if digest(value) != digest(FROZEN):
        raise ValueError("Frozen configuration changed")
    return value


def source_path(database):
    if database not in DATABASES:
        raise ValueError("Only source validation databases permitted")
    return CACHE / database


def matrix():
    return [(checkpoint, operator, db, run) for checkpoint in CHECKPOINTS
            for operator in OPERATORS for db in DATABASES for run in range(5)]


def recheck(binding):
    for path, expected in binding.items():
        if sha256(checked_path(path)) != expected:
            raise ValueError(f"Input SHA changed: {path}")


def encode_identity(data):
    import numpy as np

    if set(data) != {"query_hash", "query_id", "true", "n_tables"}:
        raise ValueError("Unexpected identity fields")
    true, tables = np.asarray(data["true"]), np.asarray(data["n_tables"])
    n = len(true)
    if (not n or true.dtype != np.float64 or true.ndim != 1
            or any(np.asarray(value).shape != (n,) for value in data.values())
            or not np.isfinite(true).all() or (true < 0).any()
            or not np.isfinite(tables).all() or (tables < 1).any() or (tables > 32767).any()
            or not np.array_equal(tables, tables.astype(np.int16))):
        raise ValueError("Invalid FP64 truth or table counts")
    result = {"true": true, "n_tables": tables.astype(np.int16)}
    for key, width in (("query_hash", 64), ("query_id", 128)):
        values = [str(value).encode("utf-8") for value in data[key]]
        if any(not value or len(value) > width or b"\0" in value for value in values) or len(set(values)) != n:
            raise ValueError("Duplicate, empty or overlong identity")
        result[key] = np.asarray(values, dtype=f"S{width}")
    return result


def certify_plan(plan, data, database, run, expected_n=None):
    import numpy as np

    if database not in DATABASES or plan["database"] != database or plan["run"] != run:
        raise ValueError("Wrong plan database/run")
    n = len(data["true"])
    support, query = plan["support_max"], plan["query"]
    if (len(support) != 100 or len(set(support)) != 100 or len(set(query)) != len(query)
            or any(type(i) is not int or not 0 <= i < n for i in (*support, *query))
            or set(support) & set(query) or len(query) != (expected_n if expected_n is not None else COUNTS[database])):
        raise ValueError("Plan changed support exclusion or denominator")
    hashes = np.asarray(data["query_hash"]).astype(str)
    first = {}
    for index, value in enumerate(hashes):
        first.setdefault(value, index)
    indices = list(first.values())
    if query != [i for i in indices if i not in set(support)] or not set(support) <= set(indices):
        raise ValueError("Plan differs from complete first-unique query population")
    if set(plan["supports"]) != {"5", "10", "20", "50", "100"}:
        raise ValueError("Full nested plan required")
    for k in (5, 10, 20, 50, 100):
        if (plan["supports"][str(k)] != support[:k]
                or plan["support_digests"][str(k)] != digest([(i, hashes[i]) for i in support[:k]])):
            raise ValueError("Support prefix/digest changed")
    if plan["query_digest"] != digest([(i, hashes[i]) for i in query]):
        raise ValueError("Query digest changed")
    return {"query": query, "support": support[:20]}


def cell_metrics(log_pred, identity):
    import numpy as np

    true, tables = identity["true"], identity["n_tables"]
    log_pred = np.asarray(log_pred, dtype=np.float64)
    if log_pred.shape != true.shape or not np.isfinite(log_pred).all():
        raise ValueError("Invalid raw predictions")
    with np.errstate(over="raise", invalid="raise"):
        q = np.exp(np.abs(np.maximum(log_pred, 0) - np.log(np.maximum(true, 1))))

    def row(mask):
        n = int(mask.sum())
        return {"count": n, "status": "empty" if not n else "below_min_n" if n < 30 else "ok",
                **{f"q{p}": float(np.percentile(q[mask], p, method="linear")) if n else None
                   for p in (50, 95)}}

    card = np.searchsorted([100, 1000, 10000, 100000, 1000000], true, side="right")
    table = np.minimum(tables, 5) - 1
    labels = ("<100", "100-1e3", "1e3-1e4", "1e4-1e5", "1e5-1e6", ">=1e6")
    joint = {f"{c}|{t}": row((card == ci) & (table == ti)) for ci, c in enumerate(labels)
             for ti, t in enumerate(("1", "2", "3", "4", "5+"))}
    return {"all": row(np.ones(len(true), dtype=bool)), "4+": row(tables >= 4), "joint": joint}


def mean_metric(rows, bucket, metric):
    values = [row[bucket][metric] for row in rows]
    return sum(values) / len(values) if values and all(v is not None for v in values) else None


def verify_anchor_identity(identity, anchor):
    import numpy as np

    if set(anchor) != {"log_pred", "true", "n_tables", "query_hash", "query_id"}:
        raise ValueError("Unexpected anchor fields")
    order = np.argsort(np.asarray(anchor["query_hash"]).astype(str))
    aligned = {key: np.asarray(value)[order] for key, value in anchor.items()}
    for key in ("query_hash", "query_id", "true", "n_tables"):
        actual, expected = identity[key], aligned[key]
        if key in ("query_hash", "query_id"):
            actual, expected = np.asarray(actual).astype(str), np.asarray(expected).astype(str)
        if not np.array_equal(actual, expected) or key == "true" and expected.dtype != np.float64:
            raise ValueError(f"Regression anchor {key} differs")


def verify_anchor(log_pred, identity, anchor, positions=None):
    import numpy as np

    if set(anchor) != {"log_pred", "true", "n_tables", "query_hash", "query_id"}:
        raise ValueError("Unexpected anchor fields")
    order = np.argsort(np.asarray(anchor["query_hash"]).astype(str))
    aligned = {key: np.asarray(value)[order] for key, value in anchor.items()}
    if positions is not None:
        aligned = {key: value[positions] for key, value in aligned.items()}
    for key in ("query_hash", "query_id", "true", "n_tables"):
        actual, expected = identity[key], aligned[key]
        if key in ("query_hash", "query_id"):
            actual, expected = np.asarray(actual).astype(str), np.asarray(expected).astype(str)
        if not np.array_equal(actual, expected) or key == "true" and expected.dtype != np.float64:
            raise ValueError(f"Regression anchor {key} differs")
    if len(set(np.asarray(anchor["query_hash"]).astype(str))) != len(anchor["true"]):
        raise ValueError("Duplicate anchor queries")
    if not np.allclose(log_pred, aligned["log_pred"], rtol=1e-5, atol=1e-6):
        raise ValueError("Regression raw log_pred tolerance failed")
    expected = cell_metrics(aligned["log_pred"], identity)["all"]["q95"]
    actual = cell_metrics(log_pred, identity)["all"]["q95"]
    if not np.isclose(actual, expected, rtol=1e-4, atol=0):
        raise ValueError("Regression standard Q95 tolerance failed")


def state_digest(model):
    value = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        value.update(name.encode())
        value.update(str(tensor.dtype).encode())
        value.update(str(tuple(tensor.shape)).encode())
        value.update(tensor.detach().cpu().numpy().tobytes())
    return value.hexdigest()


def operator_params(model, support, operator):
    import torch

    from spatial_meta_ce.meta import _adaptation_names, adapt
    if operator not in OPERATORS:
        raise ValueError("Unknown operator")
    before = state_digest(model)
    original = OrderedDict(model.named_parameters())
    selected = [] if operator == "none" else _adaptation_names(model, original, operator[:-1])
    params = original if operator == "none" else adapt(
        model, support, lr=0.001, steps=3, scope=operator[:-1], loss="smooth_l1", training=False)
    changed = [name for name, tensor in original.items() if not torch.equal(tensor, params[name])]
    if state_digest(model) != before or not set(changed) <= set(selected):
        raise ValueError("Shared initialization or frozen parameters contaminated")
    record = {"selected_parameter_names": selected,
              "selected_parameter_count": sum(original[name].numel() for name in selected),
              "changed_parameter_names": changed,
              "changed_parameter_count": sum(original[name].numel() for name in changed),
              "steps": 0 if operator == "none" else 3}
    expected = {"none": 0, "head3": 107521, "last3": 65}[operator]
    if record["selected_parameter_count"] != expected:
        raise ValueError("Unexpected adaptation parameter capacity")
    return params, record


def environment(mode):
    import numpy as np
    import torch

    if mode == "cpu_preflight" and os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("CPU preflight requires explicitly hidden CUDA")
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        if os.environ.get(key) != "1":
            raise ValueError("Explicit single-thread environment required")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if mode == "formal" and (not os.environ.get("CUDA_VISIBLE_DEVICES")
                             or not torch.cuda.is_available() or torch.cuda.device_count() != 1):
        raise ValueError("Formal execution requires one explicitly selected CUDA GPU")
    return {"python": sys.version, "executable": sys.executable, "platform": platform.platform(),
            "numpy": np.__version__, "torch": torch.__version__, "cuda": torch.version.cuda,
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "threads": 1, "precision": "FP32 model/FP64 true", "deterministic_algorithms": True,
            "tf32": False, "mode": mode}


def input_binding(root=ROOT):
    old_source, diagnosis, retained = [read_json(root / p) for p in PROVENANCE]
    binding = {str(root / p): sha256(root / p) for p in (*CODE_FILES, *PROVENANCE, PROTOCOL)}
    if binding[str(root / PROTOCOL)] != PROTOCOL_SHA:
        raise ValueError("Frozen protocol changed")
    for checkpoint, arm in CHECKPOINTS.items():
        path = checked_path(RETAINED / arm / "last.pt")
        if path.stat().st_size != 2574664 or sha256(path) != CHECKPOINT_SHA[checkpoint]:
            raise ValueError("Checkpoint SHA/size changed")
        binding[str(path)] = CHECKPOINT_SHA[checkpoint]
    plan_path = checked_path(RETAINED / "endpoint_plan.json")
    if sha256(plan_path) != PLAN_SHA or retained["endpoint_plan_sha256"] != PLAN_SHA:
        raise ValueError("Plan SHA changed")
    binding[str(plan_path)] = PLAN_SHA
    for db in DATABASES:
        for name, expected in old_source[db]["file_sha256"].items():
            path = checked_path(source_path(db) / name)
            if name not in ("manifest.json", "cards.npy", "offsets.npy", *(f"group{i}.npy" for i in range(4))):
                raise ValueError("Unexpected source cache file")
            binding[str(path)] = expected
    anchors = {}
    for name, expected in diagnosis["input_sha256"].items():
        path = Path(name)
        if path.suffix == ".npz" and "-k20-run" in path.name:
            db, shot, run = path.stem.split("-")
            checkpoint = next((c for c, arm in CHECKPOINTS.items() if path.parent.parent.name == arm), None)
            if db not in DATABASES or checkpoint is None or shot != "k20":
                raise ValueError("Unexpected regression anchor")
            key = (checkpoint, db, int(run[3:]))
            if key in anchors:
                raise ValueError("Duplicate anchor")
            anchors[key] = checked_path(path)
            binding[str(path)] = expected
    if set(anchors) != {(c, db, r) for c in CHECKPOINTS for db in DATABASES for r in range(5)}:
        raise ValueError("Incomplete 45-cell regression anchors")
    recheck(binding)
    return binding, read_json(plan_path), anchors


def store_identity(store, database):
    import numpy as np

    if store.cards.dtype != np.float64 or not np.isfinite(store.cards).all() or (store.cards < 0).any():
        raise ValueError("Source cards must be finite FP64")
    tables = np.diff(store.offsets[:, 2])
    if (tables < 1).any() or store.offsets.dtype.kind not in "iu" or (np.diff(store.offsets, axis=0) < 0).any():
        raise ValueError("Invalid offsets/table counts")
    first = {}
    for index, row in enumerate(store.rows):
        if row["id"] != f"{database}/pretrain/{index+1}" or not isinstance(row["query_hash"], str):
            raise ValueError("Source manifest identity format changed")
        h = row["query_hash"]
        if h in first:
            other = first[h]
            if not np.array_equal(store.cards[index], store.cards[other]):
                raise ValueError("Conflicting duplicate source label")
            for g, values in enumerate(store.groups):
                left = values[store.offsets[index, g]:store.offsets[index+1, g]]
                right = values[store.offsets[other, g]:store.offsets[other+1, g]]
                if not np.array_equal(left, right):
                    raise ValueError("Conflicting duplicate source features")
        else:
            first[h] = index
    for group in store.groups:
        if not np.isfinite(group).all():
            raise ValueError("Nonfinite source features")
    return {"query_hash": np.array([r["query_hash"] for r in store.rows]),
            "query_id": np.array([r["id"] for r in store.rows]),
            "true": np.asarray(store.cards[:, 0]), "n_tables": tables}


def certified_batch(store, indices, device):
    import numpy as np
    import torch

    batch = store.batch(indices, device)
    expected = np.asarray(store.cards[indices, 0])
    if batch["true"].dtype != torch.float64 or not np.array_equal(batch["true"].cpu().numpy(), expected):
        raise ValueError("FP64 cards/sample/batch chain changed")
    return batch


def evaluate_cell(checkpoint_path, store, query_indices, support_indices, operator, device):
    import numpy as np
    import torch

    from spatial_meta_ce.checkpoints import load_checkpoint
    from spatial_meta_ce.meta import forward
    checkpoint = next((key for key, arm in CHECKPOINTS.items() if checkpoint_path.parent.name == arm), None)
    if checkpoint is None or sha256(checkpoint_path) != CHECKPOINT_SHA[checkpoint]:
        raise ValueError("Checkpoint changed before trusted deserialization")
    model, state = load_checkpoint(checkpoint_path, device)
    if state["next_step"] != 4000 or state["metadata"]["seed"] != 42 or model.config != FROZEN["model"]:
        raise ValueError("Checkpoint model/seed/budget changed")
    model.eval()
    before = state_digest(model)
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    support = None if operator == "none" else certified_batch(store, support_indices, device)
    params, parameter_record = operator_params(model, support, operator)
    if device == "cuda":
        torch.cuda.synchronize()
    adaptation_seconds = time.monotonic() - started
    predictions, labels = [], []
    started = time.monotonic()
    with torch.no_grad():
        for start in range(0, len(query_indices), 512):
            batch = certified_batch(store, query_indices[start:start+512], device)
            values = forward(model, params, batch).detach().cpu().numpy().astype(np.float64)
            if not np.isfinite(values).all():
                raise ValueError("Nonfinite raw predictions")
            predictions.append(values)
            labels.append(batch["true"].cpu().numpy())
    if device == "cuda":
        torch.cuda.synchronize()
    after = state_digest(model)
    if after != before:
        raise ValueError("Cell mutated shared initialization")
    record = {**parameter_record, "state_before": before, "state_after": after,
              "adaptation_seconds": adaptation_seconds,
              "prediction_seconds": time.monotonic() - started,
              "peak_vram_bytes": int(max(torch.cuda.max_memory_allocated(), torch.cuda.max_memory_reserved()))
              if device == "cuda" else 0,
              "peak_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
              * (1 if sys.platform == "darwin" else 1024),
              "rss_measurement": "process lifetime high-water mark; watchdog also samples owned process tree"}
    return np.concatenate(predictions), np.concatenate(labels), record


def worker(output, mode, config_path, root=ROOT):
    import numpy as np

    from spatial_meta_ce.data import FeatureStore
    validate_config(read_json(config_path))
    out = safe_output(output, [RETAINED, CACHE, "/gly/weilin/PRICE-master"])
    out.mkdir(parents=True)
    write_json(out / "status.json", {"status": "running", "mode": mode, "cells": 0})
    write_json(out / "phase.json", {"phase": "preparation"})
    environment_record = environment(mode)
    write_json(out / "environment.json", environment_record)
    binding, plan, anchors = input_binding(root)
    binding[str(checked_path(config_path))] = sha256(config_path)
    if set(plan) != set(DATABASES) or any(len(plan[db]) != 5 for db in DATABASES):
        raise ValueError("Full source plan is missing")
    stores, identities, certified = {}, {}, {}
    (out / "identities").mkdir()
    (out / "predictions").mkdir()
    for db in DATABASES:
        store = FeatureStore(source_path(db))
        stores[db] = store
        raw = store_identity(store, db)
        for run in range(5):
            selected = certify_plan(plan[db][run], raw, db, run)
            certified[(db, run)] = selected
            indices = np.asarray(selected["query"], dtype=np.int64)
            order = np.argsort(raw["query_hash"][indices])
            indices = indices[order]
            selected["query_sorted"] = indices.tolist()
            identity = encode_identity({key: value[indices] for key, value in raw.items()})
            identity_path = out / "identities" / f"{db}-run{run}.npz"
            np.savez(identity_path, **identity)
            identities[(db, run)] = (identity, identity_path)
            for checkpoint in CHECKPOINTS:
                with np.load(anchors[(checkpoint, db, run)], allow_pickle=False) as archive:
                    anchor = {key: archive[key] for key in archive.files}
                # Preparation certifies identity/truth/table/plan binding only.
                # Raw log_pred tolerances are checked on the exact regression
                # cells during evaluation, not over the full query population.
                verify_anchor_identity(identity, anchor)
    estimated = sum(p.stat().st_size for p in (out / "identities").glob("*.npz")) + 6714180 * 8
    if estimated > 0.30 * 1024**3:
        raise ValueError("Frozen shared identity schema exceeds capacity budget")
    write_json(out / "input-binding.json", {"schema": 1, "mode": mode, "input_sha256": binding,
                                           "code_sha256": {p: binding[str(root / p)] for p in CODE_FILES},
                                           "estimated_formal_output_bytes": estimated})
    write_json(out / "phase.json", {"phase": "evaluation"})
    cells = []
    none_checks = []
    selected_matrix = matrix() if mode == "formal" else [
        (c, "last3" if c == "C" else "head3", db, 0) for c in CHECKPOINTS for db in DATABASES]
    for checkpoint, operator, db, run in selected_matrix:
        identity, identity_path = identities[(db, run)]
        query = certified[(db, run)]["query_sorted"]
        positions = None if mode == "formal" else list(range(32))
        query = query if positions is None else query[:32]
        selected_identity = identity if positions is None else {key: value[positions] for key, value in identity.items()}
        checkpoint_path = RETAINED / CHECKPOINTS[checkpoint] / "last.pt"
        if mode == "cpu_preflight":
            none_pred, none_true, none_record = evaluate_cell(
                checkpoint_path, stores[db], query,
                certified[(db, run)]["support"], "none", "cpu")
            if none_record["selected_parameter_count"] or none_record["steps"]:
                raise ValueError("None microcheck called adaptation")
            if none_record["state_before"] != none_record["state_after"]:
                raise ValueError("None microcheck mutated initialization")
            none_checks.append({"checkpoint": checkpoint, "database": db, "run": run,
                                "operator": "none", "query_positions": positions,
                                "count": len(none_pred), "state_before": none_record["state_before"],
                                "state_after": none_record["state_after"],
                                "selected_parameter_count": none_record["selected_parameter_count"],
                                "steps": none_record["steps"]})
            if not np.array_equal(none_true, selected_identity["true"]):
                raise ValueError("None microcheck FP64 truth differs")
        pred, true, record = evaluate_cell(checkpoint_path, stores[db], query,
                                           certified[(db, run)]["support"], operator,
                                           "cuda" if mode == "formal" else "cpu")
        if not np.array_equal(true, selected_identity["true"]):
            raise ValueError("Output FP64 truth changed")
        diagonal = operator == ("last3" if checkpoint == "C" else "head3")
        if diagonal:
            with np.load(anchors[(checkpoint, db, run)], allow_pickle=False) as archive:
                verify_anchor(pred, selected_identity, {key: archive[key] for key in archive.files}, positions)
        path = out / "predictions" / f"{checkpoint}-{operator}-{db}-run{run}.npz"
        np.savez(path, log_pred=pred)
        cells.append({**record, "checkpoint": checkpoint, "operator": operator, "database": db,
                      "run": run, "k": 20, "k20": 20, "count": len(pred), "regression_anchor_checked": diagonal,
                      "identity_file": str(identity_path.relative_to(out)), "identity_sha256": sha256(identity_path),
                      "prediction_file": str(path.relative_to(out)), "prediction_sha256": sha256(path),
                      "metrics": cell_metrics(pred, selected_identity),
                      **({"query_positions": positions} if positions is not None else {})})
        write_json(out / ("cells.json" if mode == "formal" else "microcells.json"),
                   {"schema": 1, "mode": mode, "cells": cells,
                    **({"none_checks": none_checks} if mode == "cpu_preflight" else {})})
    write_json(out / "phase.json", {"phase": "certification"})
    from scripts.meta65_joint_verify import verify
    verified = verify(out)
    recheck(binding)
    write_json(out / "verification.json", verified)
    write_json(out / "status.json", {"status": "worker_complete", "mode": mode,
                                     "cells": len(cells) if mode == "formal" else 0,
                                     "micro_cells": len(cells) if mode == "cpu_preflight" else 0,
                                     "formal_result": mode == "formal"})
    return verified


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/meta65_joint_control.json")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--mode", choices=("cpu_preflight", "formal"), default="cpu_preflight")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--approval", type=Path, help="Separate user-approved formal launch contract")
    args = parser.parse_args(argv)
    validate_config(read_json(args.config))
    if args.mode == "formal":
        # No approval artifact is generated by CPU preflight or launch binding.
        # A future authorized task must bind the exact formal command and code.
        if args.approval is None:
            raise ValueError("Formal GPU matrix is not authorized by implementation/preflight")
        approval = read_json(args.approval)
        if (approval.get("permission") != "one_shot_formal_gpu_matrix"
                or approval.get("config_sha256") != sha256(args.config)
                or approval.get("code_sha256") != {p: sha256(ROOT / p) for p in CODE_FILES}
                or approval.get("protocol_sha256") != PROTOCOL_SHA
                or approval.get("output") != str(checked_path(args.out))
                or not str(approval.get("gpu_uuid", "")).startswith("GPU-")
                or approval.get("gpu_uuid") != os.environ.get("CUDA_VISIBLE_DEVICES")):
            raise ValueError("Formal approval does not bind this frozen launch")
    if not args.execute:
        print(json.dumps({"status": "prepared_not_executed", "mode": args.mode,
                          "cells": 135 if args.mode == "formal" else 0,
                          "microcells": 9 if args.mode == "cpu_preflight" else 0,
                          "formal_result": False}, indent=2))
        return
    safe_output(args.out, [RETAINED, CACHE, "/gly/weilin/PRICE-master"])
    if args.worker:
        if (os.environ.get("META65_SUPERVISED_MODE") != args.mode
                or os.environ.get("META65_SUPERVISOR_PID") != str(os.getppid())):
            raise ValueError("Worker requires its owning external supervisor")
        try:
            worker(args.out, args.mode, args.config)
        except BaseException as exc:
            if args.out.exists():
                write_json(args.out / "status.json", {"status": "failed", "mode": args.mode,
                                                     "failure": f"{type(exc).__name__}: {exc}",
                                                     "formal_result": False})
            raise
    else:
        from scripts.meta65_joint_supervisor import supervise
        command = [sys.executable, "-B", str(Path(__file__).resolve()), "--worker", "--execute",
                   "--config", str(args.config), "--out", str(args.out), "--mode", args.mode]
        if args.approval:
            command += ["--approval", str(args.approval)]
        record = supervise(command, args.out, mode=args.mode)
        print(json.dumps(record, indent=2))
        if record["status"] not in ("completed", "preflight_completed"):
            raise SystemExit(1)


if __name__ == "__main__":
    main()
