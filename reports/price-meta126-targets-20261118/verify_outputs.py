"""Post-run verification independent of production metric and aggregate code."""

import csv
import gzip
import hashlib
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path("/gly/weilin/price-meta-126-targets-20261118")
DATABASES = ("imdb", "stats", "ergastf1", "genome")
FEWSHOT = DATABASES[1:]
SEEDS = (42, 43, 44)


def sha(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b""):
            value.update(block)
    return value.hexdigest()


def read(path):
    return json.loads(path.read_text())


def main():
    binding = read(ROOT / "inputs/binding.json")
    plan = read(Path(binding["plan"]))
    output = ROOT / "outputs/evaluate"
    completion = read(output / "completion.json")
    supervisor = read(ROOT / "outputs/evaluate.supervisor.json")
    units = read(output / "units.json")
    summary = read(output / "summary.json")
    assert (
        supervisor["status"] == "completed" and supervisor["completion"] == completion
    )
    assert completion["binding_sha256"] == binding["binding_sha256"]
    assert sha(output / "units.json") == completion["units_sha256"]
    assert sha(output / "summary.json") == completion["summary_sha256"]
    assert len(units) == completion["cells"] == 242
    expected_keys = set()
    for arm, seeds in (("P", (0,)), ("E", SEEDS), ("M", SEEDS), ("PG", (0,))):
        for seed in seeds:
            for db in DATABASES:
                expected_keys.add((arm, seed, db, 0, 0))
                if arm != "PG" and db in FEWSHOT:
                    expected_keys.update(
                        (arm, seed, db, k, r) for k in (5, 20) for r in range(5)
                    )
    cells = {}
    total_rows = 0
    for unit in units:
        key = tuple(unit[k] for k in ("arm", "stream", "database", "k", "run"))
        assert (
            key in expected_keys and key not in cells and unit["status"] == "completed"
        )
        identity = plan["databases"][key[2]]["test"]
        indices = identity["indices"]
        cards = np.load(
            Path(binding["cache_root"]) / "test" / key[2] / "cards.npy", mmap_mode="r"
        )
        path = output / unit["prediction_file"]
        assert sha(path) == unit["prediction_sha256"]
        errors = []
        with gzip.open(path, "rt", newline="") as stream:
            for index, row in enumerate(csv.DictReader(stream)):
                assert index < len(indices)
                assert (row["query_id"], row["query_hash"]) == (
                    identity["ids"][index],
                    identity["hashes"][index],
                )
                assert tuple(
                    row[f] for f in ("arm", "stream", "database", "k", "run")
                ) == tuple(map(str, key))
                truth, logp, encoded = (
                    float(row[k]) for k in ("true", "log_prediction", "encoded")
                )
                assert truth == float(cards[indices[index], 0])
                assert all(math.isfinite(v) for v in (truth, logp, encoded))
                a = encoded - 1
                decoded = (
                    0.0
                    if a <= math.log(2)
                    else math.log(math.expm1(a))
                    if a < 700
                    else a
                )
                assert math.isclose(max(logp, 0), decoded, rel_tol=1e-10, abs_tol=1e-10)
                if key[0] == "PG":
                    assert math.isclose(
                        logp,
                        math.log(max(float(cards[indices[index], 1]), 1)),
                        rel_tol=1e-12,
                        abs_tol=1e-12,
                    )
                errors.append(math.exp(abs(max(logp, 0) - math.log(max(truth, 1)))))
        assert len(errors) == unit["n"] == len(indices)
        values = {
            f"q{p}": float(np.quantile(errors, p / 100, method="linear"))
            for p in (50, 90, 95, 99)
        }
        assert all(
            math.isclose(values[k], unit["metrics"][k], rel_tol=1e-10, abs_tol=1e-10)
            for k in values
        )
        cells[key] = values
        total_rows += len(errors)
    assert set(cells) == expected_keys

    def qbar(arm, seed, db, k):
        seed = 0 if arm in ("P", "PG") else seed
        values = [
            cells[(arm, seed, db, k, r)]["q95"] for r in ((0,) if k == 0 else range(5))
        ]
        return sum(values) / len(values)

    ratios = {}
    for name, left, right in (
        ("M20/E20", ("M", 20), ("E", 20)),
        ("M20/P20", ("M", 20), ("P", 20)),
        ("M20/M0", ("M", 20), ("M", 0)),
        ("M20/P0", ("M", 20), ("P", 0)),
        ("M20/E0", ("M", 20), ("E", 0)),
    ):
        deltas = [
            sum(
                math.log(qbar(left[0], seed, db, left[1]))
                - math.log(qbar(right[0], seed, db, right[1]))
                for db in FEWSHOT
            )
            / 3
            for seed in SEEDS
        ]
        ratio = math.exp(sum(deltas) / 3)
        assert math.isclose(ratio, summary["ratios"][name]["ratio"], rel_tol=1e-12)
        ratios[name] = {
            "ratio": ratio,
            "percent_change": 100 * (ratio - 1),
            "improved_streams": sum(d < 0 for d in deltas),
        }
    k0_ratios = {}
    for name, left, right in (
        ("M0/P0", "M", "P"),
        ("M0/E0", "M", "E"),
        ("E0/P0", "E", "P"),
    ):
        per_db = [
            sum(qbar(left, s, db, 0) for s in SEEDS)
            / sum(qbar(right, s, db, 0) for s in SEEDS)
            for db in DATABASES
        ]
        ratio = math.exp(sum(map(math.log, per_db)) / 4)
        assert math.isclose(
            ratio, summary["k0_geometric_mean"][name]["ratio"], rel_tol=1e-12
        )
        k0_ratios[name] = ratio
    negative = sum(
        cells[("M", seed, db, 20, r)]["q95"] > qbar("M", seed, db, 0)
        for seed in SEEDS
        for db in FEWSHOT
        for r in range(5)
    )
    assert negative == summary["k0_protection"]["by_block"]["worsened"]
    value = {
        "status": "independently_verified",
        "binding_sha256": binding["binding_sha256"],
        "cells": 242,
        "prediction_rows": total_rows,
        "metric_quantiles_verified": [50, 90, 95, 99],
        "ratios": ratios,
        "k0_four_database_ratios": k0_ratios,
        "negative_transfer_blocks": negative,
        "negative_transfer_blocks_total": 45,
        "decision_reported": summary["decision"],
        "no_production_metric_or_aggregate_imports": True,
        "external_peak_vram_bytes": supervisor["peak_vram_bytes"],
        "elapsed_seconds": supervisor["elapsed_seconds"],
        "target_databases_read": list(DATABASES),
        "selection_forbidden": True,
    }
    with (ROOT / "evidence/independent-verification.json").open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(value, indent=2))


if __name__ == "__main__":
    main()
