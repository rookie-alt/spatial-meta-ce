import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

DATABASES = ("chembl", "legalacts", "telstra")
ARMS = ("sum", "true", "shuffled")
SEEDS = (42, 43, 44)
MIN_CELL_N = 100
CARDINALITY_BINS = (
    (10, 100),
    (100, 1000),
    (1000, 10000),
    (10000, 100000),
    (100000, np.inf),
)
PG_ERROR_BINS = ((-np.inf, -1), (-1, 0), (0, 1), (1, np.inf))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def q95(prediction: np.ndarray, truth: np.ndarray) -> float:
    signed = np.maximum(prediction, 0) - np.log(np.maximum(truth, 1))
    return float(np.quantile(np.exp(np.abs(signed)), 0.95))


def describe(
    mask: np.ndarray, pg: np.ndarray, truth: np.ndarray, predictions: dict
) -> dict:
    n = int(mask.sum())
    if n < MIN_CELL_N:
        return {"n": n, "status": "below_min_cell_n"}
    pg_error = pg[mask] - np.log(np.maximum(truth[mask], 1))
    true42, true43, true44 = (predictions[f"true-seed{s}"][mask] for s in SEEDS)
    shuffled43 = predictions["shuffled-seed43"][mask]
    return {
        "n": n,
        "pg_error_mean": float(pg_error.mean()),
        "pg_q95": q95(pg[mask], truth[mask]),
        "true42_q95": q95(true42, truth[mask]),
        "true43_q95": q95(true43, truth[mask]),
        "true44_q95": q95(true44, truth[mask]),
        "true42_head_mean": float(np.mean(true42 - pg[mask])),
        "true43_head_mean": float(np.mean(true43 - pg[mask])),
        "true44_head_mean": float(np.mean(true44 - pg[mask])),
        "true43_minus_42_mean": float(np.mean(true43 - true42)),
        "true43_minus_shuffled43_mean": float(np.mean(true43 - shuffled43)),
    }


def unique_cache(
    cache: Path, expected_hash: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    manifest = json.loads((cache / "manifest.json").read_text())
    payload = {key: value for key, value in manifest.items() if key != "content_hash"}
    if (
        manifest["content_hash"] != expected_hash
        or hashlib.sha256(
            json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()
        ).hexdigest()
        != expected_hash
    ):
        raise ValueError(f"Frozen source cache manifest changed: {cache.name}")
    seen = set()
    indices = []
    for i, row in enumerate(manifest["rows"]):
        if row["query_hash"] not in seen:
            indices.append(i)
            seen.add(row["query_hash"])
    offsets = np.load(cache / "offsets.npy", mmap_mode="r", allow_pickle=False)
    cards_file = np.load(cache / "cards.npy", mmap_mode="r", allow_pickle=False)
    counts = np.diff(offsets, axis=0)[indices]
    cards = cards_file[indices]
    if counts.shape != (len(indices), 4) or cards.shape != (len(indices), 2):
        raise ValueError(f"Invalid cache shape: {cache.name}")
    return (
        np.array([manifest["rows"][i]["query_hash"] for i in indices]),
        counts,
        cards,
        np.asarray(indices),
    )


def self_check() -> None:
    pg = np.log(np.array([10.0, 20.0, 40.0], dtype=np.float32)).astype(np.float64)
    truth = np.array([10.0, 20.0, 40.0])
    predictions = {
        "true-seed42": pg + 1,
        "true-seed43": pg - 1,
        "true-seed44": pg,
        "shuffled-seed43": pg + 2,
    }
    n = MIN_CELL_N
    result = describe(
        np.ones(n, dtype=bool),
        np.resize(pg, n),
        np.resize(truth, n),
        {key: np.resize(value, n) for key, value in predictions.items()},
    )
    assert result["n"] == n
    assert np.isclose(result["pg_error_mean"], 0, atol=1e-6)
    assert np.isclose(result["true43_head_mean"], -1)
    assert np.isclose(result["true43_minus_42_mean"], -2)
    assert np.isclose(result["true43_minus_shuffled43_mean"], -3)
    assert describe(
        np.array([True, False]),
        pg[:2],
        truth[:2],
        {k: v[:2] for k, v in predictions.items()},
    ) == {"n": 1, "status": "below_min_cell_n"}
    print("self-check passed")


def analyze(root: Path) -> dict:
    diagnosis_root = root / "runs/p1-mapping-ablation-20261004/diagnosis-20261004"
    cache_root = root / "artifacts/cache/pretrain"
    matrix_path = root / "runs/p1-mapping-ablation-20261004/ablation_runs.json"
    sidecar_root = root / "runs/mapping-sidecars-source26-20261004"
    overlap_path = diagnosis_root / "tail_overlap.json"
    pg_report_path = diagnosis_root / "pg_representation_verified.json"
    matrix = json.loads(matrix_path.read_text())
    overlap = json.loads(overlap_path.read_text())
    pg_report = json.loads(pg_report_path.read_text())
    if pg_report["input_sha256"]["ablation_runs.json"] != sha256(
        matrix_path
    ) or pg_report["input_sha256"]["tail_overlap.json"] != sha256(overlap_path):
        raise ValueError("Frozen PG diagnostic provenance changed")
    if {(r["arm"], r["seed"]) for r in matrix["runs"]} != {
        (a, s) for a in ARMS for s in SEEDS
    } or len(matrix["runs"]) != 9:
        raise ValueError("Frozen nine-run matrix changed")
    report = {
        "purpose": "posthoc structure-conditioned source residuals, no training or inference",
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "script_sha256": sha256(Path(__file__)),
        "training": False,
        "inference": False,
        "target_data_read": False,
        "databases": list(DATABASES),
        "minimum_cell_n": MIN_CELL_N,
        "definitions": {
            "counts": ["n_join_cols", "n_fanouts", "n_tables", "n_filter_cols"],
            "bins": "true cardinality [10,100),[100,1000),[1000,10000),[10000,100000),[100000,inf); PG signed log error <-1,[-1,0),[0,1),>=1",
            "signed_pg_error": "log(max(pg,1))-log(max(true,1))",
            "warning": "Encoded column counts are not SQL join topology, predicate semantics, or selectivity. Cells are posthoc descriptions; no causal attribution, model selection, or reweighted endpoint.",
        },
        "input_sha256": {
            "ablation_runs.json": sha256(matrix_path),
            "tail_overlap.json": sha256(overlap_path),
            "pg_representation_verified.json": sha256(pg_report_path),
        },
        "per_database": {},
    }
    for db in DATABASES:
        cache = cache_root / db
        sidecar_manifest_path = sidecar_root / db / "manifest.json"
        sidecar_manifest = json.loads(sidecar_manifest_path.read_text())
        mapped_runs = [r for r in matrix["runs"] if r["arm"] in ("true", "shuffled")]
        expected_sidecar_hashes = {
            r["mapping_provenance"][db]["manifest_hash"] for r in mapped_runs
        }
        if (
            len(expected_sidecar_hashes) != 1
            or sha256(sidecar_manifest_path) != next(iter(expected_sidecar_hashes))
            or sidecar_manifest.get("database") != db
            or sidecar_manifest.get("split") != "pretrain"
            or sidecar_manifest.get("input_sha256", {}).get("cache/offsets.npy")
            != sha256(cache / "offsets.npy")
            or sidecar_manifest.get("input_sha256", {}).get("cache/cards.npy")
            != sha256(cache / "cards.npy")
            or sidecar_manifest.get("input_sha256", {}).get("cache/manifest.json")
            != sha256(cache / "manifest.json")
        ):
            raise ValueError(f"Frozen source sidecar/cache file hashes changed: {db}")
        report["input_sha256"] = {
            **report["input_sha256"],
            f"{db}/sidecar_manifest.json": sha256(sidecar_manifest_path),
            f"{db}/offsets.npy": sha256(cache / "offsets.npy"),
            f"{db}/cards.npy": sha256(cache / "cards.npy"),
        }
        recorded_hashes = {r["manifest_hashes"][db] for r in matrix["runs"]}
        if len(recorded_hashes) != 1:
            raise ValueError(f"Unpaired frozen cache manifests: {db}")
        query_hash, counts, cards, _ = unique_cache(cache, recorded_hashes.pop())
        truth = cards[:, 0]
        pg = np.log(np.maximum(cards[:, 1], 1)).astype(np.float32).astype(np.float64)
        if not np.isfinite(pg).all() or not np.isfinite(truth).all():
            raise ValueError(f"Nonfinite cache labels or PG: {db}")
        predictions = {}
        for arm in ARMS:
            for seed in SEEDS:
                name = f"{arm}-seed{seed}"
                path = diagnosis_root / f"{name}-{db}.npz"
                if sha256(path) != overlap["npz_sha256"][path.name]:
                    raise ValueError(f"Frozen prediction hash mismatch: {name}-{db}")
                with np.load(path, allow_pickle=False) as old:
                    if (
                        not np.array_equal(query_hash, old["query_hash"])
                        or not np.array_equal(truth, old["true"])
                        or not np.array_equal(counts[:, 2], old["n_tables"])
                    ):
                        raise ValueError(
                            f"Cache/prediction pairing mismatch: {name}-{db}"
                        )
                    predictions = {**predictions, name: old["log_pred"]}
                primary = counts[:, 2] >= 4
                recorded = pg_report["per_database"][db]["endpoints"][name]["4plus"]
                if not np.isclose(
                    q95(predictions[name][primary], truth[primary]),
                    recorded["model_q95"],
                    rtol=1e-7,
                ):
                    raise ValueError(f"Frozen endpoint mismatch: {name}-{db}")
        primary = counts[:, 2] >= 4
        baseline = pg_report["per_database"][db]["baseline"]["4plus"]
        if (
            int(primary.sum()) != baseline["n"]
            or not np.isclose(
                q95(pg[primary], truth[primary]), baseline["pg_q95"], rtol=1e-7
            )
            or not np.isclose(
                np.mean(pg[primary] - np.log(np.maximum(truth[primary], 1))),
                baseline["pg_signed_error_mean"],
                atol=1e-7,
            )
        ):
            raise ValueError(f"PG baseline reconstruction mismatch: {db}")
        pg_error = pg - np.log(np.maximum(truth, 1))
        cells = {}
        for tables in np.unique(counts[primary, 2]):
            table_mask = primary & (counts[:, 2] == tables)
            for lower, upper in CARDINALITY_BINS:
                cardinality = table_mask & (truth >= lower) & (truth < upper)
                if not cardinality.any():
                    continue
                for structure, index in (("join_columns", 0), ("filter_columns", 3)):
                    values = counts[:, index]
                    for value in np.unique(values[cardinality]):
                        condition = cardinality & (values == value)
                        key = f"tables={tables}|true=[{lower},{upper})|{structure}={value}"
                        cells = {
                            **cells,
                            key: describe(condition, pg, truth, predictions),
                        }
                        for pg_low, pg_high in PG_ERROR_BINS:
                            pg_mask = (
                                condition & (pg_error >= pg_low) & (pg_error < pg_high)
                            )
                            if pg_mask.any():
                                pg_key = f"{key}|pg_error=[{pg_low},{pg_high})"
                                cells = {
                                    **cells,
                                    pg_key: describe(pg_mask, pg, truth, predictions),
                                }
        report["per_database"] = {
            **report["per_database"],
            db: {
                "queries": len(truth),
                "primary_n": int(primary.sum()),
                "primary": describe(primary, pg, truth, predictions),
                "cells": cells,
            },
        }
        print(
            f"verified {db}: {len(truth)} unique, {int(primary.sum())} primary, {len(cells)} cells",
            flush=True,
        )
    report["checks"] = {
        "source_cache_manifest_count": 3,
        "source_sidecar_manifest_hashes_verified": 3,
        "offsets_and_cards_file_hashes_verified": 6,
        "prediction_sha256_count": 27,
        "frozen_model_endpoints_verified": 27,
        "pg_q95_and_mean_verified": 3,
        "cache_query_pairing_verified": True,
    }
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument("--root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    self_check()
    if not args.self_check:
        if args.root is None or args.output is None:
            parser.error("--root and --output required")
        report = analyze(args.root)
        payload = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)
        with args.output.open("x", encoding="utf-8") as output:
            output.write(payload + "\n")
        print(f"wrote {args.output}")
