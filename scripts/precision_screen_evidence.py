import json
import math
from pathlib import Path

import numpy as np

from scripts.mapping_ablation_endpoint import bucket_q95, endpoint_rows
from spatial_meta_ce.data import FeatureStore
from spatial_meta_ce.manifests import digest, file_hash
from spatial_meta_ce.metrics import q_values

VALIDATION = ("chembl", "legalacts", "telstra")
TARGETS = frozenset(("imdb", "stats", "genome", "ergastf1"))
CACHE_FILES = (
    "manifest.json",
    "cards.npy",
    "offsets.npy",
    "group0.npy",
    "group1.npy",
    "group2.npy",
    "group3.npy",
)
COVERAGE = {
    "chembl": (50000, 33947),
    "legalacts": (49504, 9178),
    "telstra": (50000, 20311),
}
SOURCE_BINDING = "3d708226437e105c88562970fdd0b2ccf711a833105b4ec8c66d899aa61f79a6"


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def checked_path(root: Path, path: Path) -> Path:
    absolute = path.absolute()
    if any(p.is_symlink() for p in (absolute, *absolute.parents)):
        raise ValueError(f"Symlink not allowed: {path}")
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"Path escapes source root: {path}")
    return path


def historical_sources(root: Path, databases: list) -> dict:
    hashes, manifests = {}, {}
    for db in databases:
        if db in TARGETS or Path(db).name != db:
            raise ValueError("Historical sources must be source databases")
        folder = (
            "p1-mapping-sidecar-preflight-20261003"
            if db in VALIDATION
            else "p1-mapping-sidecar-train23-20261004"
        )
        relative = f"实验文档/results/{folder}/{db}/manifest.json"
        path = checked_path(root, root / relative)
        hashes[relative] = file_hash(path)
        manifests[db] = json.loads(path.read_text())
    encoded = json.dumps(
        hashes, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    import hashlib

    if hashlib.sha256(encoded.encode()).hexdigest() != SOURCE_BINDING:
        raise ValueError("Frozen P1 source manifest bundle changed")
    return manifests


def source_evidence(data: dict, historical: dict) -> dict:
    cache = Path(data["cache_root"])
    sources = data["splits"]["train"] + data["splits"]["validation"]
    if set(historical) != set(sources) or len(sources) != len(set(sources)):
        raise ValueError("Source provenance coverage changed")
    result = {}
    for db in sources:
        if db in TARGETS or Path(db).name != db:
            raise ValueError("Target or invalid source name")
        directory = checked_path(cache, cache / "pretrain" / db)
        reference = historical[db]
        if reference["database"] != db or reference["split"] != "pretrain":
            raise ValueError("Historical database identity changed")
        hashes = {}
        for name in CACHE_FILES:
            path = checked_path(cache, directory / name)
            hashes[name] = file_hash(path)
            if hashes[name] != reference["input_sha256"][f"cache/{name}"]:
                raise ValueError(f"Frozen source cache file changed: {db}/{name}")
        store = FeatureStore(directory)
        if store.manifest["content_hash"] != reference["cache_content_hash"]:
            raise ValueError(f"Frozen source manifest changed: {db}")
        if store.manifest.get("provenance", {}).get("excluded"):
            raise ValueError("Frozen source cache has exclusions")
        result[db] = {
            "manifest_hash": store.manifest["content_hash"],
            "file_sha256": hashes,
            "rows": len(store),
        }
    return result


def finite_q95(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("Q95 must be a finite numeric value")
    if not math.isfinite(value) or value < 1:
        raise ValueError("Q95 must be finite and at least one")
    return float(value)


def summarize_endpoint(rows_by_db: dict, expected_coverage: dict | None) -> dict:
    if tuple(rows_by_db) != VALIDATION:
        raise ValueError(
            "Endpoint must contain exactly the three source validation databases"
        )
    identity, per_db = [], {}
    for db, rows in rows_by_db.items():
        hashes = [r["query_hash"] for r in rows]
        if len(hashes) != len(set(hashes)):
            raise ValueError("Endpoint contains duplicate query identities")
        for row in rows:
            if not isinstance(row["n_tables"], int) or row["n_tables"] < 1:
                raise ValueError("Invalid endpoint table count")
            if not math.isfinite(row["true"]) or row["true"] < 0:
                raise ValueError("Invalid endpoint truth")
            if not math.isfinite(row["log_pred"]):
                raise FloatingPointError("Nonfinite raw endpoint prediction")
            identity.append((db, row["query_hash"], row["true"], row["n_tables"]))
        q = q_values([r["log_pred"] for r in rows], [r["true"] for r in rows])[0]
        if not np.isfinite(q).all():
            raise FloatingPointError("Endpoint Q-error overflow")
        counts = (len(rows), sum(r["n_tables"] >= 4 for r in rows))
        if expected_coverage is not None and counts != expected_coverage[db]:
            raise ValueError(f"Endpoint coverage changed: {db}: {counts}")
        per_db[db] = {
            "all_tables": bucket_q95(rows),
            "four_plus_tables": bucket_q95(rows, 4),
        }
        for bucket in per_db[db].values():
            finite_q95(bucket["q50"])
            finite_q95(bucket["q95"])
    return {
        "per_database": per_db,
        "identity_digest": digest(sorted(identity)),
        "unique_queries": len(identity),
        "macro_four_plus_q95": float(
            np.mean([v["four_plus_tables"]["q95"] for v in per_db.values()])
        ),
        "macro_all_table_q95": float(
            np.mean([v["all_tables"]["q95"] for v in per_db.values()])
        ),
    }


def endpoint(
    model, pool, device: str, expected_coverage: dict | None, smoke: bool = False
) -> tuple[dict, dict]:
    rows_by_db = {}
    for db in VALIDATION:
        store = pool.get("pretrain", db)
        if not smoke:
            rows = endpoint_rows(model, store, device)
        else:
            indices = store.unique_indices()
            large = [
                int(i)
                for i in indices
                if store.offsets[i + 1, 2] - store.offsets[i, 2] >= 4
            ][:32]
            other = [
                int(i)
                for i in indices
                if store.offsets[i + 1, 2] - store.offsets[i, 2] < 4
            ][:32]
            selected = large + other
            batch = store.batch(selected, device)
            import torch

            with torch.no_grad():
                predictions = model.eval()(batch).cpu().tolist()
            rows = [
                {
                    "query_hash": store.rows[i]["query_hash"],
                    "true": float(store.cards[i, 0]),
                    "n_tables": int(store.offsets[i + 1, 2] - store.offsets[i, 2]),
                    "log_pred": pred,
                }
                for i, pred in zip(selected, predictions)
            ]
        rows_by_db[db] = rows
    return summarize_endpoint(rows_by_db, expected_coverage), rows_by_db
