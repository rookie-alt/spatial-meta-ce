"""Doc 122 train-only PRICE padding contract; no feature/label rewriting."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .data import TRANSFORM_VERSION
from .manifests import digest, file_hash, validate_splits

TRAIN_DATABASES = (
    "accidents",
    "airline",
    "baseball",
    "basketball",
    "carcinogenesis",
    "ccs",
    "consumer",
    "credit",
    "employee",
    "financial",
    "fnhk",
    "grants",
    "hepatitis",
    "hockey",
    "movielens",
    "sakila",
    "sap",
    "seznam",
    "ssb",
    "talkingdata",
    "tournament",
    "tpc_h",
    "tubepricing",
)
SOURCE_DATABASES = ("chembl", "legalacts", "telstra")
OFFICIAL_TRAIN_DATABASES = TRAIN_DATABASES + SOURCE_DATABASES
OFFICIAL_CAPACITY_KIND = "price_train_capacity_122_official_26"
GROUP_NAMES = ("join_hist", "fanout", "table", "filter")


class UnsupportedCapacity(ValueError):
    """A query is unsupported, not truncated, filtered or resized."""

    def __init__(self, records: list[dict]) -> None:
        self.records = records
        super().__init__("unsupported: " + json.dumps(records, ensure_ascii=False))


def validate_source_splits(data_config: dict) -> None:
    validate_splits(data_config["splits"])
    splits = data_config["splits"]
    if tuple(splits["train"]) != TRAIN_DATABASES:
        raise ValueError("122 requires exactly the frozen 23 training databases")
    if tuple(splits["validation"]) != SOURCE_DATABASES:
        raise ValueError("122 requires exactly the three source validation databases")


def _sha(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(c in "0123456789abcdef" for c in value)
    )


def _counts(values: object) -> list[int]:
    if (
        not isinstance(values, list)
        or len(values) != 4
        or any(type(n) is not int or n < 0 for n in values)
    ):
        raise ValueError(
            "Invalid capacity max_counts; require four nonnegative integers"
        )
    return values


def read_source_manifest(cache_root: str | Path, db: str) -> tuple[Path, dict]:
    if db not in TRAIN_DATABASES + SOURCE_DATABASES:
        raise ValueError(f"122 rejects non-source database {db!r}")
    root = Path(cache_root).resolve()
    path = root / "pretrain" / db / "manifest.json"
    if (
        not path.resolve().is_relative_to(root)
        or path.is_symlink()
        or path.parent.is_symlink()
    ):
        raise ValueError("Source manifest path escapes the cache root")
    manifest = json.loads(path.read_text())
    payload = {k: v for k, v in manifest.items() if k != "content_hash"}
    if (
        manifest.get("schema") != 1
        or manifest.get("transform") != TRANSFORM_VERSION
        or manifest.get("content_hash") != digest(payload)
    ):
        raise ValueError(f"Source cache manifest hash/schema/transform mismatch: {db}")
    _counts(manifest.get("inspection", {}).get("max_counts"))
    return path, manifest


def build_capacity(data_config: dict) -> dict:
    """Read only the 23 complete pretrain manifests, never validation maxima."""
    validate_source_splits(data_config)
    manifests = {
        db: read_source_manifest(data_config["cache_root"], db)
        for db in TRAIN_DATABASES
    }
    maxima = {db: m["inspection"]["max_counts"] for db, (_, m) in manifests.items()}
    payload = {
        "schema": 1,
        "kind": "price_train_capacity_122",
        "split": "pretrain",
        "transform": TRANSFORM_VERSION,
        "databases": list(TRAIN_DATABASES),
        "max_counts": [max(counts[i] for counts in maxima.values()) for i in range(4)],
        "database_max_counts": maxima,
        "cache_hashes": {db: m["content_hash"] for db, (_, m) in manifests.items()},
        "manifest_sha256": {db: file_hash(path) for db, (path, _) in manifests.items()},
    }
    return {**payload, "capacity_sha256": digest(payload)}


def validate_capacity(capacity: dict, cache_hashes: dict | None = None) -> dict:
    """cache_hashes and training metadata.manifest_hashes are content_hash values.

    They are not the separate manifest-file SHA256s in manifest_sha256.
    """
    if not isinstance(capacity, dict):
        raise ValueError("Missing PRICE capacity binding")
    payload = {k: v for k, v in capacity.items() if k != "capacity_sha256"}
    if capacity.get("capacity_sha256") != digest(payload):
        raise ValueError("PRICE capacity binding hash mismatch")
    kind = capacity.get("kind")
    if kind == OFFICIAL_CAPACITY_KIND:
        expected = OFFICIAL_TRAIN_DATABASES
    elif kind == "price_train_capacity_122":
        expected = TRAIN_DATABASES
    else:
        raise ValueError("Invalid PRICE capacity source contract")
    if (
        capacity.get("schema") != 1
        or capacity.get("split") != "pretrain"
        or capacity.get("transform") != TRANSFORM_VERSION
        or capacity.get("databases") != list(expected)
    ):
        raise ValueError("Invalid PRICE capacity source contract")
    maxima = _counts(capacity.get("max_counts"))
    per_db = capacity.get("database_max_counts", {})
    if set(per_db) != set(expected):
        raise ValueError("PRICE capacity requires all training cache maxima")
    for value in per_db.values():
        _counts(value)
    if maxima != [max(v[i] for v in per_db.values()) for i in range(4)]:
        raise ValueError("PRICE capacity differs from training maxima")
    if maxima[2] == 0:
        raise ValueError("PRICE capacity requires at least one table token")
    for field in ("cache_hashes", "manifest_sha256"):
        values = capacity.get(field, {})
        if set(values) != set(expected) or not all(
            _sha(v) for v in values.values()
        ):
            raise ValueError(f"Invalid PRICE capacity {field}")
    if cache_hashes is not None and cache_hashes != capacity["cache_hashes"]:
        raise ValueError("PRICE capacity/cache hash mismatch")
    return capacity


def scan_source_capacity(data_config: dict, capacity: dict) -> dict:
    """No labels: verify all three source offset maxima before coverage claims."""
    validate_source_splits(data_config)
    validate_capacity(capacity)
    bound = capacity["max_counts"]
    records, databases = [], {}
    for db in SOURCE_DATABASES:
        path, manifest = read_source_manifest(data_config["cache_root"], db)
        maxima = manifest["inspection"]["max_counts"]
        databases[db] = {
            "cache_hash": manifest["content_hash"],
            "manifest_sha256": file_hash(path),
            "max_counts": maxima,
        }
        offsets_path = path.parent / "offsets.npy"
        if not offsets_path.resolve().is_relative_to(
            Path(data_config["cache_root"]).resolve()
        ):
            raise ValueError("Source offsets path escapes the cache root")
        offsets = np.load(offsets_path, mmap_mode="r", allow_pickle=False)
        counts = np.diff(offsets, axis=0)
        if counts.shape != (len(manifest["rows"]), 4) or (counts < 0).any():
            raise ValueError(f"Invalid source cache offsets: {db}")
        if counts.max(axis=0).tolist() != maxima:
            raise ValueError(f"Source manifest/offset maxima mismatch: {db}")
        for row, group in np.argwhere(counts > np.asarray(bound)):
            records.append(
                {
                    "database": db,
                    "query_hash": manifest["rows"][int(row)]["query_hash"],
                    "group": GROUP_NAMES[int(group)],
                    "actual": int(counts[row, group]),
                    "capacity": bound[int(group)],
                }
            )
    return {
        "capacity_sha256": capacity["capacity_sha256"],
        "databases": databases,
        "unsupported": records,
        "full_population_supported": not records,
        "target_databases_read": [],
    }
