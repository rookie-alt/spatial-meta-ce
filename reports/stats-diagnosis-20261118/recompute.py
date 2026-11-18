"""Recompute descriptive diagnostics from frozen predictions; never tune outputs."""

import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def load_rows(path):
    rows = {}
    with path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            key = (row["database"], row["query_hash"])
            if key in rows or row["k"] != "0" or row["run"] != "0":
                raise ValueError(f"Duplicate query or non-K0 input: {path}")
            value = (float(row["true_cardinality"]), float(row["log_prediction"]))
            if not np.isfinite(value).all() or value[0] < 0:
                raise ValueError(f"Invalid prediction row: {path}")
            rows[key] = value
    return rows


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--source-sha256", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    expected = {}
    for line in args.source_sha256.read_text().splitlines():
        sha, name = line.split(maxsplit=1)
        expected[name.strip()] = sha
    sources = {
        "pg": ("pg.csv", "target-baselines/worker/baseline/pg/predictions.csv"),
        "price": (
            "price.csv",
            "target-baselines/worker/baseline/price/predictions.csv",
        ),
    }
    for steps, prefix, folder in (
        (4000, "ours4k", "target-eval"),
        (24000, "ours", "target-eval-24k"),
    ):
        for seed in (42, 43, 44):
            sources[f"{steps}_seed{seed}"] = (
                f"{prefix}_seed{seed}.csv",
                f"{folder}/worker/eval/A1-PG-seed{seed}/predictions.csv",
            )
    datasets, bindings = {}, []
    for name, (local, remote) in sources.items():
        path = args.inputs / local
        raw = path.read_bytes()
        sha = hashlib.sha256(raw).hexdigest()
        if expected.get(remote) != sha:
            raise ValueError(f"Prediction SHA256 differs from server: {remote}")
        datasets[name] = load_rows(path)
        bindings.append({"name": name, "source": remote, "sha256": sha})
    keys = sorted(datasets["pg"])
    for name, rows in datasets.items():
        if sorted(rows) != keys or any(
            rows[key][0] != datasets["pg"][key][0] for key in keys
        ):
            raise ValueError(f"Population or truth mismatch: {name}")
    stats_keys = [key for key in keys if key[0] == "stats"]
    if len(stats_keys) != 2603:
        raise ValueError("Expected all 2603 unique STATS queries")
    truth = np.log(np.maximum([datasets["pg"][key][0] for key in stats_keys], 1))
    pg = np.array([datasets["pg"][key][1] for key in stats_keys])
    pg_error = np.abs(np.maximum(pg, 0) - truth)
    good = pg_error <= 0.5  # Descriptive band; never a deployable gate.
    summary = {}
    for name, rows in datasets.items():
        pred = np.array([rows[key][1] for key in stats_keys])
        errors = np.abs(np.maximum(pred, 0) - truth)
        delta = pred - pg
        worse = errors > pg_error + 1e-9
        signed_pg = pg - truth
        summary[name] = {
            "n": len(stats_keys),
            "q50": float(np.percentile(np.exp(errors), 50)),
            "q95": float(np.percentile(np.exp(errors), 95)),
            "log_q95": float(np.percentile(errors, 95)),
            "worse_than_pg": int(worse.sum()),
            "mean_abs_residual": float(np.abs(delta).mean()),
            "mean_signed_residual": float(delta.mean()),
            "pg_log_error_le_0_5": {
                "n": int(good.sum()),
                "worsened": int(worse[good].sum()),
                "fraction_worsened": float(worse[good].mean()),
            },
            "worsened_further_overestimate": int(
                (worse & (signed_pg > 0) & (delta > 0)).sum()
            ),
            "worsened_further_underestimate": int(
                (worse & (signed_pg < 0) & (delta < 0)).sum()
            ),
            "worsened_after_moving_toward_truth": int(
                (worse & (signed_pg * delta < 0)).sum()
            ),
        }
    result = {
        "schema": 1,
        "date": "2026-11-18",
        "kind": "descriptive_existing_prediction_audit",
        "population": "canonical unique-query; STATS test; K=0; no new inference",
        "source_root": "/gly/weilin/official-split-20261117",
        "prediction_transform": "original protocol floor max(log_prediction,0) only",
        "selection_use": "none",
        "bindings": bindings,
        "stats": summary,
        "limits": [
            "Descriptive target analysis does not identify causal training defects.",
            "PG accuracy uses true labels and cannot serve as an inference feature.",
            "No clipping, blending, threshold scan, checkpoint selection or training.",
        ],
    }
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "inputs_verified": len(bindings)}))


if __name__ == "__main__":
    main()
