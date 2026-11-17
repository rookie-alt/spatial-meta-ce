"""Frozen training-task identities for experiment 125 (official 26/0/4 split)."""

from __future__ import annotations

import json
import re
from collections import OrderedDict
from pathlib import Path

import numpy as np

from .manifests import digest

# Official PRICE ``pretrain.py`` order; 26 training libraries, no source
# validation split.  Target libraries follow the 125 protocol order.
TRAIN_DATABASES = (
    "accidents",
    "airline",
    "baseball",
    "basketball",
    "carcinogenesis",
    "ccs",
    "chembl",
    "consumer",
    "credit",
    "employee",
    "financial",
    "fnhk",
    "grants",
    "hepatitis",
    "hockey",
    "legalacts",
    "movielens",
    "sakila",
    "sap",
    "seznam",
    "ssb",
    "talkingdata",
    "telstra",
    "tournament",
    "tpc_h",
    "tubepricing",
)
TARGET_DATABASES = ("imdb", "stats", "ergastf1", "genome")
SEEDS = (42, 43, 44)
CAPACITY = (12, 12, 7, 10)
SUPPORT_SIZE = 20
QUERY_SIZE = 64
TASK_SIZE = SUPPORT_SIZE + QUERY_SIZE
STEPS = 2000
META_BATCH = 4
TASKS_PER_STREAM = STEPS * META_BATCH
RNG_SALT = 20261117
RNG_EXPERIMENT = 125
PLAN_KIND = "price_meta125_training_plan"


def validate_data_config(config):
    """Require the frozen 26-train / 0-validation / 4-test split."""
    if not isinstance(config, dict) or not isinstance(config.get("cache_root"), str):
        raise ValueError("Data config requires a cache_root")
    splits = config.get("splits", {})
    if not isinstance(splits, dict):
        raise ValueError("Data config requires a splits mapping")
    for key in ("train", "test"):
        values = splits.get(key)
        if not isinstance(values, list) or not values:
            raise ValueError(f"Invalid database list: {key}")
        if len(set(values)) != len(values):
            raise ValueError(f"Invalid database list: {key}")
        if any(
            not isinstance(db, str) or not db or "/" in db or db in (".", "..")
            for db in values
        ):
            raise ValueError("Invalid database name")
    validation = splits.get("validation", [])
    if not isinstance(validation, list):
        raise ValueError("Invalid database list: validation")
    if len(set(validation)) != len(validation):
        raise ValueError("Invalid database list: validation")
    if any(
        not isinstance(db, str) or not db or "/" in db or db in (".", "..")
        for db in validation
    ):
        raise ValueError("Invalid database name")
    for key in ("train", "test", "validation"):
        for other in ("train", "test", "validation"):
            if key == other:
                continue
            if set(splits.get(key, [])) & set(splits.get(other, [])):
                raise ValueError(f"Database leakage between {key} and {other}")
    if tuple(splits.get("train", ())) != TRAIN_DATABASES:
        raise ValueError("Experiment 125 train database order or membership changed")
    if tuple(splits.get("test", ())) != TARGET_DATABASES:
        raise ValueError("Experiment 125 test database order or membership changed")
    if validation:
        raise ValueError(
            "Experiment 125 has no independent source-validation database set"
        )


class TrainingStores:
    """Read-only role-bound stores; target libraries cannot be opened."""

    def __init__(self, config, databases=TRAIN_DATABASES):
        validate_data_config(config)
        self.databases = tuple(databases)
        names = set(self.databases)
        if (
            not names
            or len(names) != len(self.databases)
            or not names <= set(TRAIN_DATABASES)
        ):
            raise ValueError("Target databases or mixed roles are forbidden")
        self.root = Path(config["cache_root"]).resolve()
        self.opened_databases = []
        self._stores = OrderedDict()

    def get(self, db):
        if db not in self.databases:
            raise ValueError("Database is outside this stage's store whitelist")
        if db not in self._stores:
            from .data import FeatureStore

            path = self.root / "pretrain" / db
            if path.is_symlink() or path.resolve() != path:
                raise ValueError("Linked or aliased database cache is forbidden")
            required = [
                path / "manifest.json",
                path / "cards.npy",
                path / "offsets.npy",
            ]
            required += [path / f"group{i}.npy" for i in range(4)]
            if any(p.is_symlink() for p in required):
                raise ValueError("Linked cache arrays or manifests are forbidden")
            manifest = json.loads((path / "manifest.json").read_text())
            source = manifest.get("provenance", {}).get("source", {})
            if source.get("database") != db or source.get("split") != "pretrain":
                raise ValueError("Cache provenance does not match its source database")
            if any(
                not row.get("id", "").startswith(f"{db}/pretrain/")
                for row in manifest.get("rows", [])
            ):
                raise ValueError("Cache query identities refer to another database")
            self._stores[db] = FeatureStore(path)
            if db not in self.opened_databases:
                self.opened_databases.append(db)
        return self._stores[db]

    def close(self):
        self._stores.clear()


def plan_digest(plan):
    return digest({key: value for key, value in plan.items() if key != "plan_sha256"})


def _canonical(store):
    from .plans import _canonical_unique_indices

    return _canonical_unique_indices(store)


def _store_data_sha256(store):
    from .plans import _store_data_sha256 as data_sha256

    return data_sha256(store)


def _info(store, unique):
    maxima = list(store.manifest["inspection"]["max_counts"])
    if len(maxima) != 4 or any(
        type(n) is not int or n < 0 or n > c for n, c in zip(maxima, CAPACITY)
    ):
        raise ValueError("PRICE capacity is insufficient; queries may not be truncated")
    return {
        "cache_hash": store.manifest["content_hash"],
        "data_sha256": _store_data_sha256(store),
        "rows": len(store.rows),
        "unique_queries": len(unique),
        "max_counts": maxima,
    }


def _identities(store, indices):
    values = [int(i) for i in indices]
    return {
        "indices": values,
        "ids": [store.rows[i]["id"] for i in values],
        "hashes": [store.rows[i]["query_hash"] for i in values],
    }


def make_training_plan(config, seed, steps=STEPS):
    """Build the frozen 8000-task stream for one training seed."""
    validate_data_config(config)
    if seed not in SEEDS or type(steps) is not int or steps != STEPS:
        raise ValueError("Training plan requires a frozen stream and 2000 updates")
    stores = TrainingStores(config)
    unique = {db: _canonical(stores.get(db)) for db in TRAIN_DATABASES}
    metadata = {db: _info(stores.get(db), unique[db]) for db in TRAIN_DATABASES}
    rng = np.random.Generator(
        np.random.PCG64(np.random.SeedSequence([RNG_SALT, RNG_EXPERIMENT, seed]))
    )
    episodes = []
    for _ in range(steps * META_BATCH):
        db = TRAIN_DATABASES[int(rng.integers(len(TRAIN_DATABASES)))]
        indices = rng.choice(unique[db], size=TASK_SIZE, replace=False)
        store = stores.get(db)
        episodes.append(
            {
                "db": db,
                "support": _identities(store, indices[:SUPPORT_SIZE]),
                "query": _identities(store, indices[SUPPORT_SIZE:]),
            }
        )
    stores.close()
    plan = {
        "schema": 1,
        "kind": PLAN_KIND,
        "seed": seed,
        "steps": steps,
        "meta_batch": META_BATCH,
        "databases": metadata,
        "episodes": episodes,
        "numpy_version": np.__version__,
        "rng_state": rng.bit_generator.state,
        "target_databases_read": [],
    }
    plan["plan_sha256"] = plan_digest(plan)
    validate_plan(plan, config)
    return plan


def _validate_identities(identity, count, rows):
    if not isinstance(identity, dict) or set(identity) != {"indices", "ids", "hashes"}:
        raise ValueError("Query identity fields changed")
    if any(
        not isinstance(identity[k], list) or len(identity[k]) != count for k in identity
    ):
        raise ValueError("Query identity counts disagree")
    if any(type(i) is not int or not 0 <= i < rows for i in identity["indices"]):
        raise ValueError("Query index outside cache")
    if any(not isinstance(x, str) or not x for x in identity["ids"]):
        raise ValueError("Query IDs must be nonempty strings")
    if any(
        not isinstance(x, str) or not re.fullmatch(r"[0-9a-f]{64}", x)
        for x in identity["hashes"]
    ):
        raise ValueError("Invalid query hash")
    if any(len(set(identity[k])) != count for k in identity):
        raise ValueError("Duplicate query identities")


def validate_plan(plan, config):
    """Validate a sealed training plan against the frozen 125 contract."""
    validate_data_config(config)
    if not isinstance(plan, dict):
        raise ValueError("Unknown plan kind")
    if (
        plan.get("schema") != 1
        or plan.get("kind") != PLAN_KIND
        or plan.get("target_databases_read") != []
        or plan.get("plan_sha256") != plan_digest(plan)
    ):
        raise ValueError("Plan type, target exposure or content digest changed")
    if tuple(plan.get("databases", {})) != TRAIN_DATABASES:
        raise ValueError("Plan database closure or order changed")
    for db, metadata in plan["databases"].items():
        for key in ("cache_hash", "data_sha256"):
            if not re.fullmatch(r"[0-9a-f]{64}", metadata.get(key, "")):
                raise ValueError("Missing cache provenance hash")
        if (
            type(metadata.get("rows")) is not int
            or type(metadata.get("unique_queries")) is not int
            or not TASK_SIZE <= metadata["unique_queries"] <= metadata["rows"]
        ):
            raise ValueError("Invalid unique query denominator")
        if len(metadata.get("max_counts", [])) != 4 or any(
            type(n) is not int or not 0 <= n <= c
            for n, c in zip(metadata["max_counts"], CAPACITY)
        ):
            raise ValueError("Unsupported feature capacity")
    if (
        plan.get("seed") not in SEEDS
        or plan.get("steps") != STEPS
        or plan.get("meta_batch") != META_BATCH
        or len(plan.get("episodes", [])) != TASKS_PER_STREAM
    ):
        raise ValueError("Training budget or stream changed")
    for episode in plan["episodes"]:
        if episode.get("db") not in TRAIN_DATABASES:
            raise ValueError("Training episode opened a forbidden database")
        rows = plan["databases"][episode["db"]]["rows"]
        _validate_identities(episode["support"], SUPPORT_SIZE, rows)
        _validate_identities(episode["query"], QUERY_SIZE, rows)
        for key in ("indices", "hashes", "ids"):
            if set(episode["support"][key]) & set(episode["query"][key]):
                raise ValueError("Training support/query overlap")
    return plan


def verify_plan_stores(plan, stores):
    """Re-check sealed identities against the live 26-library caches."""
    validate_plan(
        plan,
        {
            "cache_root": str(stores.root),
            "splits": {
                "train": list(TRAIN_DATABASES),
                "validation": [],
                "test": list(TARGET_DATABASES),
            },
        },
    )
    for db, metadata in plan["databases"].items():
        store = stores.get(db)
        unique = _canonical(store)
        observed = _info(store, unique)
        if any(observed[k] != metadata[k] for k in observed):
            raise ValueError("Cache data or denominator changed after plan sealing")
        canonical = set(unique.tolist())
        for episode in plan["episodes"]:
            if episode["db"] != db:
                continue
            for name in ("support", "query"):
                identity = episode[name]
                if (
                    not set(identity["indices"]) <= canonical
                    or _identities(store, identity["indices"]) != identity
                ):
                    raise ValueError(
                        "Plan query identities no longer match canonical cache rows"
                    )
    return True
