#!/usr/bin/env python3
"""Bounded CPU reanalysis of retained predictions; no checkpoint/model loading."""
import argparse
import datetime
import hashlib
import json
import subprocess
import time
from pathlib import Path

import numpy as np

SOURCE_DATABASES = ("chembl", "legalacts", "telstra")
ARMS = ("A_all5_head3", "B_head3_head3", "C_last3_last3")
SHOTS = (5, 10, 20, 50, 100)
TABLE_BUCKETS = ("1", "2", "3", "4", "5+")
TRUE_BUCKETS = ("<100", "100-1e3", "1e3-1e4", "1e4-1e5", "1e5-1e6", ">=1e6")
FIELDS = {"log_pred", "true", "n_tables", "query_hash", "query_id"}
PAIRS = {"B-A": (ARMS[0], ARMS[1]), "C-A": (ARMS[0], ARMS[2]),
         "C-B": (ARMS[1], ARMS[2])}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def metrics(q):
    return {f"q{p}": float(np.percentile(q, p, method="linear")) if len(q) else None
            for p in (50, 95)}


def align_cells(cells):
    if set(cells) != set(ARMS):
        raise ValueError("Expected exactly A/B/C")
    aligned = {}
    for arm, cell in cells.items():
        if set(cell) != FIELDS:
            raise ValueError("Unexpected prediction fields")
        if any(np.asarray(value).ndim != 1 for value in cell.values()):
            raise ValueError("Prediction fields must be vectors")
        n = len(cell["true"])
        if not n or any(len(value) != n for value in cell.values()):
            raise ValueError("Prediction lengths disagree")
        hashes = np.asarray(cell["query_hash"]).astype(str)
        ids = np.asarray(cell["query_id"]).astype(str)
        if len(set(hashes)) != n or len(set(ids)) != n:
            raise ValueError("Duplicate query identity")
        order = np.argsort(hashes)
        aligned[arm] = {key: np.asarray(value)[order] for key, value in cell.items()}
        true, tables, pred = (aligned[arm][key] for key in ("true", "n_tables", "log_pred"))
        if (not all(np.isfinite(value).all() for value in (true, tables, pred))
                or (true < 0).any() or (tables < 1).any()
                or not np.array_equal(tables, tables.astype(np.int64))):
            raise ValueError("Invalid label, table count or prediction")
        reference = aligned.get(ARMS[0])
        if reference is not None and any(not np.array_equal(aligned[arm][key], reference[key])
                                         for key in FIELDS - {"log_pred"}):
            raise ValueError("Unpaired identities, labels or table counts")
    reference = aligned[ARMS[0]]
    for cell in aligned.values():
        if any(not np.array_equal(cell[key], reference[key]) for key in FIELDS - {"log_pred"}):
            raise ValueError("Unpaired identities, labels or table counts")
    return aligned


def analyze_run(cells, min_n=30):
    cells = align_cells(cells)
    reference = cells[ARMS[0]]
    true = reference["true"].astype(np.float64)
    tables = reference["n_tables"]
    card_index = np.searchsorted([100, 1000, 10000, 100000, 1000000], true, side="right")
    table_index = np.minimum(tables, 5).astype(int) - 1
    log_q = {arm: np.abs(np.maximum(cell["log_pred"].astype(np.float64), 0)
                        - np.log(np.maximum(true, 1))) for arm, cell in cells.items()}
    with np.errstate(over="raise", invalid="raise"):
        qs = {arm: np.exp(values) for arm, values in log_q.items()}
    if not all(np.isfinite(q).all() for q in qs.values()):
        raise ValueError("Nonfinite Q-error")

    def row(mask):
        n = int(mask.sum())
        result = {"count": n, "share_of_run": n / len(true),
                  "status": "empty" if not n else "below_min_n" if n < min_n else "ok",
                  "arms": {arm: metrics(q[mask]) for arm, q in qs.items()}, "paired": {}}
        for pair, (left, right) in PAIRS.items():
            delta = log_q[right][mask] - log_q[left][mask]
            values = {"mean_delta_log_q": float(delta.mean()) if n else None,
                      "median_delta_log_q": float(np.median(delta)) if n else None,
                      "worse_count": int((delta > 0).sum()),
                      "better_count": int((delta < 0).sum()),
                      "equal_count": int((delta == 0).sum())}
            for p in (50, 95):
                a, b = result["arms"][left][f"q{p}"], result["arms"][right][f"q{p}"]
                values.update({f"q{p}_delta": b - a if n else None,
                               f"q{p}_ratio": b / a if n else None,
                               f"q{p}_log10_ratio": float(np.log10(b / a)) if n else None})
            result["paired"][pair] = values
        return result

    joint = {f"{card}|{table}": row((card_index == ci) & (table_index == ti))
             for ci, card in enumerate(TRUE_BUCKETS) for ti, table in enumerate(TABLE_BUCKETS)}
    if sum(item["count"] for item in joint.values()) != len(true):
        raise ValueError("Joint buckets lost denominator")
    marginal = {"all": row(np.ones(len(true), dtype=bool)), "4+": row(tables >= 4)}
    for ci, card in enumerate(TRUE_BUCKETS):
        marginal[f"{card}|all"] = row(card_index == ci)
        marginal[f"{card}|4+"] = row((card_index == ci) & (tables >= 4))
    marginal["<100|1-3"] = row((card_index == 0) & (tables <= 3))
    marginal[">=100|4+"] = row((card_index >= 1) & (tables >= 4))
    return {"query_count": len(true), "joint": joint, "marginal": marginal}


def aggregate_runs(rows):
    complete = all(row["count"] > 0 for row in rows)
    result = {"counts_per_run": [row["count"] for row in rows], "runs": len(rows),
              "status": "incomplete_nonempty_runs" if not complete else
                        "below_min_n" if any(row["status"] != "ok" for row in rows) else "ok",
              "arms": {}, "paired": {}}
    for arm in ARMS:
        result["arms"][arm] = {
            f"q{p}_mean": float(np.mean([row["arms"][arm][f"q{p}"] for row in rows]))
            if complete else None for p in (50, 95)}
    for pair, (left, right) in PAIRS.items():
        values = {}
        for p in (50, 95):
            a, b = (result["arms"][arm][f"q{p}_mean"] for arm in (left, right))
            values[f"q{p}_ratio_of_means"] = b / a if complete else None
            values[f"q{p}_delta_of_means"] = b - a if complete else None
        deltas = [row["paired"][pair]["q95_delta"] for row in rows]
        ratios = [row["paired"][pair]["q95_ratio"] for row in rows]
        values.update(q95_worse_runs=sum(value > 0 for value in deltas if value is not None),
                      nonempty_runs=sum(row["count"] > 0 for row in rows),
                      q95_run_ratio_range=[min(ratios), max(ratios)] if complete else None,
                      leave_one_out_positive=int(sum(np.mean(deltas[:i] + deltas[i+1:]) > 0
                                                     for i in range(len(rows))))
                      if complete and len(rows) > 1 else None)
        result["paired"][pair] = values
    return result


def recompute(binding):
    started = time.monotonic()
    expected = binding["input_sha256"]
    prediction_paths, plan_paths = {}, []
    for name in expected:
        path = Path(name)
        if path.suffix == ".npz":
            arm = path.parent.parent.name
            db, shot, run = path.stem.split("-")
            if db not in SOURCE_DATABASES or arm not in ARMS:
                raise ValueError("Non-source input rejected")
            key = (db, int(shot[1:]), int(run[3:]), arm)
            if key in prediction_paths:
                raise ValueError("Duplicate input cell")
            prediction_paths[key] = path
        elif path.name == "endpoint_plan.json":
            plan_paths.append(path)
        elif path.name not in ("cards.npy", "manifest.json", "offsets.npy") or "telstra" not in path.parts:
            raise ValueError("Unexpected retained input")
        if path.is_symlink() or any(parent.is_symlink() for parent in path.parents):
            raise ValueError("Symlink input rejected")
    required = {(db, k, run, arm) for db in SOURCE_DATABASES for k in SHOTS
                for run in range(5) for arm in ARMS}
    if set(prediction_paths) != required or len(plan_paths) != 2:
        raise ValueError("Incomplete frozen matrix")
    for path, expected_sha in expected.items():
        if sha256(path) != expected_sha:
            raise ValueError(f"Input SHA changed: {path}")
    plans = [json.loads(path.read_text()) for path in plan_paths]
    if plans[0] != plans[1] or set(plans[0]) != set(SOURCE_DATABASES):
        raise ValueError("Source plans differ")
    result = {"schema": 1, "scope": "retained_prediction_cpu_recalculation_only",
              "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "binding": binding, "metric": "standard Q50/Q95; linear percentile on Q-error",
              "aggregation": "arithmetic mean of per-run quantiles; ratio of means",
              "small_n": "n<30 flagged; empirical quantiles retained; empty=null; no dropped runs",
              "databases": {}}
    for db in SOURCE_DATABASES:
        result["databases"][db] = {}
        across_k = {}
        for k in SHOTS:
            runs, pool = [], {arm: [] for arm in ARMS}
            pooled_true, pooled_tables = [], []
            for run in range(5):
                if time.monotonic() - started > 300:
                    raise TimeoutError("CPU audit exceeded 300 seconds")
                cells = {}
                for arm in ARMS:
                    with np.load(prediction_paths[(db, k, run, arm)], allow_pickle=False) as archive:
                        cells[arm] = {key: archive[key] for key in archive.files}
                cells = align_cells(cells)
                reference = cells[ARMS[0]]
                if any(not str(value).startswith(f"{db}/pretrain/") for value in reference["query_id"]):
                    raise ValueError("Wrong source query ID")
                if len(reference["true"]) != len(plans[0][db][run]["query"]):
                    raise ValueError("Planned query denominator differs")
                fingerprint = hashlib.sha256()
                for field in sorted(FIELDS - {"log_pred"}):
                    fingerprint.update(reference[field].tobytes())
                if k == SHOTS[0]:
                    across_k[run] = fingerprint.hexdigest()
                elif across_k[run] != fingerprint.hexdigest():
                    raise ValueError("Query/label/table identity differs across K")
                runs.append(analyze_run(cells))
                pooled_true.append(reference["true"])
                pooled_tables.append(reference["n_tables"])
                for arm in ARMS:
                    pool[arm].append(cells[arm]["log_pred"])
            pooled = {arm: {"true": np.concatenate(pooled_true),
                            "n_tables": np.concatenate(pooled_tables),
                            "log_pred": np.concatenate(pool[arm]),
                            "query_hash": np.array([str(i) for i in range(sum(len(v) for v in pooled_true))]),
                            "query_id": np.array([str(i) for i in range(sum(len(v) for v in pooled_true))])}
                      for arm in ARMS}
            # These synthetic occurrence IDs are solely to distinguish retained
            # run occurrences in the explicitly labelled pooled legacy comparison.
            pooled_marginal = analyze_run(pooled)["marginal"]
            result["databases"][db][str(k)] = {
                "per_run": runs,
                "mean_of_run_quantiles": {
                    section: {bucket: aggregate_runs([row[section][bucket] for row in runs])
                              for bucket in runs[0][section]}
                    for section in ("joint", "marginal")},
                "pooled_run_occurrences_not_unique_across_runs": pooled_marginal}
    for path, expected_sha in expected.items():
        if sha256(path) != expected_sha:
            raise ValueError(f"Input SHA changed during audit: {path}")
    result["verification"] = {"input_sha_before_after": True, "input_count": len(expected),
                              "prediction_cells": len(required), "paired_identity_labels_tables": True,
                              "full_joint_denominators": True, "query_identity_across_K": True,
                              "elapsed_seconds": time.monotonic() - started}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--remote", choices=("myserver",), required=True)
    parser.add_argument("--diagnosis", type=Path, required=True)
    parser.add_argument("--decision", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError("Refusing to overwrite output")
    source = Path(__file__).read_text()
    diagnosis = json.loads(args.diagnosis.read_text())
    decision = json.loads(args.decision.read_text())
    if decision["status"] != "no_candidate" or decision["selected_arm"] is not None:
        raise ValueError("Unexpected retained decision")
    binding = {"input_sha256": diagnosis["input_sha256"], "diagnosis_sha256": sha256(args.diagnosis),
               "decision_sha256": sha256(args.decision), "protocol_sha256": sha256(args.protocol),
               "script_sha256": sha256(__file__), "baseline_commit": "a447ac1d7d1cd061beddbd72b55091d5b64d5af6"}
    payload = "__name__ = 'remote_audit'\n" + source
    payload += "\nprint(json.dumps(recompute(" + repr(binding) + "), allow_nan=False))\n"
    command = ["ssh", args.remote,
               ('env CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 '
                'timeout 300 /gly/weilin/PRICE-master/venv/bin/python -B -')]
    completed = subprocess.run(command, input=payload, text=True, capture_output=True,
                               timeout=360, check=False)
    if completed.returncode:
        raise RuntimeError(f"Read-only CPU audit failed ({completed.returncode}): "
                           f"{completed.stderr[-4000:]}")
    result = json.loads(completed.stdout)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as stream:
        json.dump(result, stream, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
        stream.write("\n")
    print(json.dumps(result["verification"], indent=2))


if __name__ == "__main__":
    main()
