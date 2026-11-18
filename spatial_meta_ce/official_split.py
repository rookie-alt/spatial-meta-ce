"""Official PRICE split: 26 pretraining databases, four held-out test databases.

The names mirror the read-only upstream sources: ``config/meta_config.py``
(``ExperimentConfig.train_databases``) and ``evaluate.py`` (``TEST_LIST``).
Target databases never enter training, capacity building or any selection.
"""

from __future__ import annotations

import copy

from .data import TRANSFORM_VERSION
from .manifests import digest, file_hash, validate_splits
from .price_capacity import (
    OFFICIAL_CAPACITY_KIND,
    OFFICIAL_TRAIN_DATABASES,
    SOURCE_DATABASES,
    TRAIN_DATABASES,
    read_source_manifest,
)

TEST_DATABASES = ("imdb", "stats", "genome", "ergastf1")


def validate_official_splits(splits: dict) -> None:
    validate_splits(splits, allow_empty_validation=True)
    if tuple(splits["train"]) != OFFICIAL_TRAIN_DATABASES:
        raise ValueError("Official split requires exactly the 26 training databases")
    if splits["validation"]:
        raise ValueError("Official split has no source validation database")
    if tuple(splits["test"]) != TEST_DATABASES:
        raise ValueError("Official split requires exactly the four test databases")


def official_data_config(source: dict) -> dict:
    """Retag a 23+3 protocol-122 data config as the official 26/4 split."""
    if tuple(source["splits"]["train"]) != TRAIN_DATABASES or tuple(
        source["splits"]["validation"]
    ) != SOURCE_DATABASES:
        raise ValueError("Source data config is not the frozen 23+3 split")
    result = copy.deepcopy(source)
    result["splits"] = {
        "train": list(OFFICIAL_TRAIN_DATABASES),
        "validation": [],
        "test": list(TEST_DATABASES),
    }
    validate_official_splits(result["splits"])
    return result


def build_official_capacity(data_config: dict) -> dict:
    """Capacity over all 26 training caches; never reads a test database."""
    validate_official_splits(data_config["splits"])
    manifests = {
        db: read_source_manifest(data_config["cache_root"], db)
        for db in OFFICIAL_TRAIN_DATABASES
    }
    maxima = {
        db: manifest["inspection"]["max_counts"]
        for db, (_, manifest) in manifests.items()
    }
    payload = {
        "schema": 1,
        "kind": OFFICIAL_CAPACITY_KIND,
        "split": "pretrain",
        "transform": TRANSFORM_VERSION,
        "databases": list(OFFICIAL_TRAIN_DATABASES),
        "max_counts": [max(counts[i] for counts in maxima.values()) for i in range(4)],
        "database_max_counts": maxima,
        "cache_hashes": {
            db: manifest["content_hash"] for db, (_, manifest) in manifests.items()
        },
        "manifest_sha256": {
            db: file_hash(path) for db, (path, _) in manifests.items()
        },
    }
    return {**payload, "capacity_sha256": digest(payload)}


def validate_official_capacity(capacity: dict) -> dict:
    if not isinstance(capacity, dict):
        raise ValueError("Missing official capacity binding")
    payload = {k: v for k, v in capacity.items() if k != "capacity_sha256"}
    if capacity.get("capacity_sha256") != digest(payload):
        raise ValueError("Official capacity hash mismatch")
    if (
        capacity.get("schema") != 1
        or capacity.get("kind") != OFFICIAL_CAPACITY_KIND
        or capacity.get("split") != "pretrain"
        or capacity.get("transform") != TRANSFORM_VERSION
        or capacity.get("databases") != list(OFFICIAL_TRAIN_DATABASES)
    ):
        raise ValueError("Invalid official capacity contract")
    maxima = capacity.get("max_counts")
    per_db = capacity.get("database_max_counts")
    if not isinstance(maxima, list) or len(maxima) != 4:
        raise ValueError("Official capacity requires four maxima")
    if not isinstance(per_db, dict) or set(per_db) != set(OFFICIAL_TRAIN_DATABASES):
        raise ValueError("Official capacity requires all 26 training cache maxima")
    if maxima != [max(counts[i] for counts in per_db.values()) for i in range(4)]:
        raise ValueError("Official capacity differs from training maxima")
    return capacity