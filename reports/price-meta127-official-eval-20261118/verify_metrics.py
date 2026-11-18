#!/usr/bin/env python3
"""Independent re-computation of the official 127 Q-error metrics.

The verifier deliberately shares no code with the production runtime: it
re-derives the SHA-256 digests, the upstream Q-error expression and the
percentiles on its own, so that a mistake in the production metric cannot
hide behind a matching implementation.

Usage:

    python3 verify_metrics.py --root /gly/weilin/spatial-meta-ce-127 \
        [--stage zero|all] [--output /tmp/verify_metrics.json]

Only the standard library, numpy and torch are used.  No training, no
resumption and no selection happen here: the verifier reads the frozen
evidence and reports whether it agrees with it.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

FIELDS = ("model", "database", "row", "query_hash", "true", "pg", "encoded", "label")
DATABASES = ("imdb", "stats", "ergastf1", "genome")
PERCENTILES = (50, 80, 90, 95, 99)
LABEL_TOLERANCE = 4e-6


def digest(value):
    """The canonical digest used by the 127 runtime.

    The byte layout matters: the runtime serialises with the default
    separators, ``ensure_ascii=False`` and ``allow_nan=False``, so a
    compact re-serialisation would produce a different digest.
    """
    payload = json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def file_sha(path):
    hasher = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def read_json(path):
    with open(path, "rb") as stream:
        return json.loads(stream.read())


def official_qerrors(encoded, encoded_labels):
    """The pinned upstream expression, transcribed literally."""
    preds = np.asarray(encoded, dtype=np.float32)
    labels = np.asarray(encoded_labels, dtype=np.float32)
    with np.errstate(over="ignore", under="ignore", invalid="ignore", divide="ignore"):
        preds = np.exp(preds - 1) - 1
        labels = np.exp(labels - 1) - 1
        preds = preds.astype(np.float64) + 1.0
        labels = labels.astype(np.float64) + 1.0
        q_errors = np.maximum(labels / preds, preds / labels)
        q_errors = np.nan_to_num(q_errors, nan=np.inf)
    return q_errors


def upstream_round(value):
    """``round()`` on a numpy float64, exactly like the upstream ``get_qerror``."""
    with np.errstate(over="ignore", invalid="ignore"):
        return float(np.float64(value).round(4))


def json_float(value):
    """A JSON-safe float; a non-finite value becomes ``None``."""
    value = float(value)
    return value if math.isfinite(value) else None


def percentiles(q_errors):
    result = {}
    for percentile in PERCENTILES:
        with np.errstate(invalid="ignore"):
            value = float(np.percentile(q_errors, percentile, method="linear"))
        result[f"q{percentile}"] = upstream_round(value)
    return result


def load_raw(path):
    """Read the true cardinalities and PG estimates of an official payload."""
    with open(path, "rb") as stream:
        data = pickle.load(stream)
    if (
        not isinstance(data, dict)
        or "true_cards" not in data
        or "pg_est_cards" not in data
    ):
        raise ValueError(f"Raw payload is unusable: {path}")
    truth = np.asarray(data["true_cards"], dtype=np.float32).reshape(-1)
    pg = np.asarray(data["pg_est_cards"], dtype=np.float32).reshape(-1)
    if truth.shape != pg.shape or truth.size == 0:
        raise ValueError(f"Raw payload fields are not aligned: {path}")
    return truth, pg


def encode_float32(values):
    """``torch.log(x + 1) + 1`` in float32, computed on the CPU."""
    values = torch.as_tensor(np.asarray(values, dtype=np.float32), dtype=torch.float32)
    return (torch.log(values + 1) + 1).numpy()


def read_rows(path):
    with gzip.open(path, "rt", newline="") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != FIELDS:
            raise ValueError(f"Unexpected CSV schema in {path}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"Empty prediction file: {path}")
    return rows


def check_cell(unit, stage, binding, problems):
    """Re-compute one evaluation cell and compare it with the recorded unit."""
    model = unit.get("model")
    database = unit.get("database")
    output = Path(binding["root"]) / "outputs" / stage
    name = unit.get("prediction_file")
    if not name or Path(name).name != name:
        problems.append(f"{stage}: invalid prediction file name {name!r}")
        return None
    path = output / name
    if not path.is_file():
        problems.append(f"{stage}: prediction file is missing: {path}")
        return None
    if unit.get("prediction_sha256") and file_sha(path) != unit["prediction_sha256"]:
        problems.append(f"{stage}: prediction file SHA differs: {path}")
        return None

    rows = read_rows(path)
    entry = binding["data"].get(f"test/{database}")
    if entry is None or not entry.get("available"):
        problems.append(f"{stage}: no bound test data for {database}")
        return None
    identities = entry.get("identities") or []
    if len(rows) != len(identities):
        problems.append(
            f"{stage}: row count {len(rows)} does not match the bound population "
            f"{len(identities)}"
        )
        return None

    truth, raw_pg = load_raw(entry["path"])
    if truth.shape[0] != len(rows):
        problems.append(f"{stage}: raw truth rows do not match the CSV")
        return None
    expected_label = encode_float32(truth)

    encoded = np.empty(len(rows), dtype=np.float64)
    labels = np.empty(len(rows), dtype=np.float64)
    pg_column = np.empty(len(rows), dtype=np.float64)
    for index, row in enumerate(rows):
        if row["model"] != model or row["database"] != database:
            problems.append(f"{stage}: cell identity changed at row {index + 1}")
            return None
        if int(row["row"]) != index + 1:
            problems.append(f"{stage}: row order changed at row {index + 1}")
            return None
        if row["query_hash"] != identities[index]:
            problems.append(f"{stage}: query identity changed at row {index + 1}")
            return None
        if float(row["true"]) != float(truth[index]):
            problems.append(f"{stage}: true cardinality changed at row {index + 1}")
            return None
        encoded[index] = float(row["encoded"])
        labels[index] = float(row["label"])
        pg_column[index] = float(row["pg"])

    pg_difference = np.abs(pg_column.astype(np.float32) - raw_pg)
    if float(pg_difference.max()) != 0.0:
        problems.append(
            f"{stage}: the stored PG column differs from the raw PG estimate "
            f"(max {float(pg_difference.max()):.3e})"
        )
        return None

    if model == "PG":
        expected_encoded = encode_float32(raw_pg)
        encoded_difference = np.abs(encoded.astype(np.float32) - expected_encoded)
        if float(encoded_difference.max()) > LABEL_TOLERANCE:
            problems.append(
                f"{stage}: the PG prediction is not the CPU encoding of the raw "
                f"PG estimate (max {float(encoded_difference.max()):.3e})"
            )
            return None

    label_difference = np.abs(labels.astype(np.float32) - expected_label)
    if float(label_difference.max()) > LABEL_TOLERANCE:
        problems.append(
            f"{stage}: stored labels differ from the raw truth encoding "
            f"(max {float(label_difference.max()):.3e})"
        )
        return None

    q_errors = official_qerrors(encoded, labels)
    recomputed = percentiles(q_errors)
    recorded = unit.get("official") or {}
    for key, value in recomputed.items():
        if key not in recorded:
            problems.append(f"{stage}: recorded unit lacks {key}")
            continue
        expected = recorded[key]
        if expected is None:
            if np.isfinite(value):
                problems.append(
                    f"{stage}: {key} is recorded as null but recomputes to {value}"
                )
            continue
        if not np.isfinite(expected):
            problems.append(
                f"{stage}: {key} is recorded as a non-finite value, which the "
                "JSON report cannot carry"
            )
            continue
        if not math.isclose(float(expected), float(value), rel_tol=0, abs_tol=1e-9):
            problems.append(
                f"{stage}: {key} differs, recorded {expected} != recomputed {value}"
            )

    unrounded = recorded.get("unrounded")
    if not isinstance(unrounded, dict):
        problems.append(f"{stage}: the recorded unit has no unrounded percentiles")
    else:
        for percentile in PERCENTILES:
            key = f"q{percentile}"
            with np.errstate(invalid="ignore"):
                raw_value = float(np.percentile(q_errors, percentile, method="linear"))
            expected_raw = unrounded.get(key)
            if expected_raw is None:
                if np.isfinite(raw_value):
                    problems.append(f"{stage}: the unrounded {key} is missing")
                continue
            if not np.isfinite(float(expected_raw)):
                problems.append(
                    f"{stage}: the unrounded {key} is recorded as a non-finite value"
                )
                continue
            if not math.isclose(
                float(expected_raw), raw_value, rel_tol=0, abs_tol=1e-9
            ):
                problems.append(
                    f"{stage}: the unrounded {key} differs, recorded {expected_raw} "
                    f"!= recomputed {raw_value}"
                )

    metadata = recorded.get("metadata")
    if not isinstance(metadata, dict):
        problems.append(f"{stage}: the recorded unit has no metadata")
    else:
        if metadata.get("n") != len(rows):
            problems.append(
                f"{stage}: the metadata row count {metadata.get('n')} differs from "
                f"the verified population {len(rows)}"
            )
        if not metadata.get("official_comparable", False):
            problems.append(
                f"{stage}: the recorded metadata reports a non-comparable sample"
            )
    return {
        "stage": stage,
        "model": model,
        "database": database,
        "n": len(rows),
        "recomputed": recomputed,
        "recorded": {key: recorded.get(key) for key in recomputed},
        "recomputed_unrounded": {
            f"q{percentile}": json_float(
                np.percentile(q_errors, percentile, method="linear")
            )
            for percentile in PERCENTILES
        },
        "max_label_difference": float(label_difference.max()),
        "max_pg_difference": float(pg_difference.max()),
        "q_mean": json_float(np.mean(q_errors[np.isfinite(q_errors)])),
        "n_nonfinite_q": int(np.sum(~np.isfinite(q_errors))),
    }


def collect_units(root, stage):
    """Read the finished units of one stage, if the stage is complete."""
    output = Path(root) / "outputs" / stage
    completion_path = output / "completion.json"
    supervisor_path = output.with_name(output.name + ".supervisor.json")
    if not completion_path.is_file() or not supervisor_path.is_file():
        return None, [f"{stage}: stage is not finished"]
    completion = read_json(completion_path)
    supervisor = read_json(supervisor_path)
    if (
        completion.get("status") != "completed"
        or supervisor.get("status") != "completed"
    ):
        return None, [f"{stage}: stage is not completed"]
    if supervisor.get("completion") != completion:
        return None, [f"{stage}: supervisor does not match the completion"]
    for name, sha in (completion.get("files") or {}).items():
        if Path(name).name != name or not (output / name).is_file():
            problems = [f"{stage}: completed evidence is missing: {name}"]
            return None, problems
        if file_sha(output / name) != sha:
            return None, [f"{stage}: completed evidence changed: {name}"]
    if stage in ("zero", "report"):
        units = read_json(output / "units.json")
    else:
        units = [read_json(output / "unit.json")]
    return units, []


def finished_stages(root):
    """The finished ``zero`` stage plus every finished ``finetune-*`` stage."""
    stages = []
    outputs = Path(root) / "outputs"
    if not outputs.is_dir():
        return stages
    for entry in sorted(outputs.iterdir()):
        if (
            entry.is_dir()
            and (entry.name == "zero" or entry.name.startswith("finetune-"))
            and (entry / "completion.json").is_file()
        ):
            stages.append(entry.name)
    return stages


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, help="root of the 127 run")
    parser.add_argument(
        "--stage",
        default="zero",
        choices=("zero", "all"),
        help="which completed stage to verify",
    )
    parser.add_argument("--output", default=None, help="where to write the JSON report")
    args = parser.parse_args(argv)

    root = Path(args.root).resolve()
    binding_path = root / "inputs/binding.json"
    if not binding_path.is_file():
        raise SystemExit(f"binding is missing: {binding_path}")
    binding = read_json(binding_path)
    if binding.get("protocol") != 127:
        raise SystemExit("binding is not a protocol 127 binding")
    if binding.get("binding_sha256") != digest(
        {k: v for k, v in binding.items() if k != "binding_sha256"}
    ):
        raise SystemExit("binding digest does not match its content")

    problems = []
    cells = []
    source_units = []
    stages = ["zero"] if args.stage == "zero" else finished_stages(Path(args.root))
    if not stages:
        raise SystemExit("no completed stage to verify")
    for stage in stages:
        units, issues = collect_units(root, stage)
        problems.extend(issues)
        if units is None:
            continue
        source_units.extend(units)
        for unit in units:
            result = check_cell(unit, stage, binding, problems)
            if result is not None:
                cells.append(result)

    report_checked = False
    if args.stage == "all" and (root / "outputs/report/completion.json").is_file():
        reported, issues = collect_units(root, "report")
        problems.extend(issues)
        if reported is not None:

            def identity(unit):
                return unit["model"], unit["database"], unit["prediction_file"]

            if (
                sorted(reported, key=identity) != sorted(source_units, key=identity)
                or len(reported) != 53
            ):
                problems.append("report: aggregation differs from the 53 source cells")
            else:
                report_checked = True

    report = {
        "schema": 1,
        "protocol": 127,
        "root": str(root),
        "stage": args.stage,
        "binding_sha256": binding.get("binding_sha256"),
        "cells_verified": len(cells),
        "report_checked": report_checked,
        "matrix_complete": len(cells) == 53 and report_checked,
        "problems": problems,
        "status": "verified" if not problems else "failed",
        "cells": cells,
    }
    text = json.dumps(report, indent=2, allow_nan=False, default=str)
    if args.output:
        with open(args.output, "x") as stream:
            stream.write(text + "\n")
    print(text)
    return 0 if not problems else 1


if __name__ == "__main__":
    sys.exit(main())
