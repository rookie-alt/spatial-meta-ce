"""Frozen task and source-query identities for experiment 123."""

from __future__ import annotations

import json
import re
from collections import OrderedDict
from pathlib import Path

import numpy as np

from .manifests import digest, validate_splits

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
TARGET_DATABASES = ("imdb", "stats", "genome", "ergastf1")
SEEDS = (42, 43, 44)
LRS = (1e-7, 1e-6, 1e-5)
CAPACITY = (12, 12, 7, 10)
EXPECTED_UNIQUE = {"chembl": 50000, "legalacts": 49504, "telstra": 50000}
SUPPORT_TOTAL = 100
SUPPORT_BLOCK_SIZE = 20
SUPPORT_BLOCKS = 5
CALIBRATION_SIZE = 2000
EVAL_KS = (0, 5, 20)
RUNS = (0, 1, 2, 3, 4)


def validate_data_config(config):
    if not isinstance(config, dict) or not isinstance(config.get("cache_root"), str):
        raise ValueError("Data config requires a cache_root")
    splits = config.get("splits", {})
    validate_splits(splits)
    for role, expected in (
        ("train", TRAIN_DATABASES),
        ("validation", SOURCE_DATABASES),
        ("test", TARGET_DATABASES),
    ):
        if tuple(splits[role]) != expected:
            raise ValueError(
                f"Experiment 123 {role} database order or membership changed"
            )


class SourceStores:
    """Read-only role-bound stores; targets cannot be constructed through this API."""

    def __init__(self, config, databases, *, role=None):
        validate_data_config(config)
        self.databases = tuple(databases)
        names = set(self.databases)
        inferred = (
            "train"
            if names <= set(TRAIN_DATABASES)
            else ("source" if names <= set(SOURCE_DATABASES) else None)
        )
        if not names or len(names) != len(self.databases) or inferred is None:
            raise ValueError("Target databases or mixed roles are forbidden")
        if role is not None and role != inferred:
            raise ValueError("Store role does not match the allowed databases")
        self.root = Path(config["cache_root"]).resolve()
        self.role = inferred
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


def _info(store, unique):
    from .plans import _store_data_sha256

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


def make_training_plan(config, seed, steps=2000):
    validate_data_config(config)
    if seed not in SEEDS or type(steps) is not int or steps != 2000:
        raise ValueError("Training plan requires a frozen stream and 2000 updates")
    stores = SourceStores(config, TRAIN_DATABASES)
    unique = {db: _canonical(stores.get(db)) for db in TRAIN_DATABASES}
    metadata = {db: _info(stores.get(db), unique[db]) for db in TRAIN_DATABASES}
    rng = np.random.Generator(
        np.random.PCG64(np.random.SeedSequence([20261116, 123, seed]))
    )
    episodes = []
    for _ in range(steps * 4):
        db = TRAIN_DATABASES[int(rng.integers(len(TRAIN_DATABASES)))]
        indices = rng.choice(unique[db], size=84, replace=False)
        store = stores.get(db)
        episodes.append(
            {
                "db": db,
                "support": _identities(store, indices[:20]),
                "query": _identities(store, indices[20:]),
            }
        )
    plan = {
        "schema": 1,
        "kind": "price_meta123_training_plan",
        "seed": seed,
        "steps": steps,
        "meta_batch": 4,
        "databases": metadata,
        "episodes": episodes,
        "numpy_version": np.__version__,
        "rng_state": rng.bit_generator.state,
        "target_databases_read": [],
    }
    plan["plan_sha256"] = plan_digest(plan)
    validate_plan(plan, config, "training")
    return plan


def make_eval_plan(config, expected_unique=None):
    validate_data_config(config)
    expected_unique = EXPECTED_UNIQUE if expected_unique is None else expected_unique
    stores = SourceStores(config, SOURCE_DATABASES)
    entries, states = {}, {}
    for index, db in enumerate(SOURCE_DATABASES):
        store = stores.get(db)
        unique = _canonical(store)
        if len(unique) != expected_unique[db] or len(unique) <= 2100:
            raise ValueError(
                "Source unique-query denominator differs from the frozen population"
            )
        rng = np.random.Generator(
            np.random.PCG64(np.random.SeedSequence([20261116, index]))
        )
        permutation = rng.permutation(unique)
        support = permutation[:100]
        entries[db] = {
            **_info(store, unique),
            "support": _identities(store, support),
            "support_blocks": [
                _identities(store, support[r * 20 : (r + 1) * 20]) for r in range(5)
            ],
            "calibration": _identities(store, permutation[100:2100]),
            "report": _identities(store, permutation[2100:]),
            "counts": {
                "support": 100,
                "calibration": 2000,
                "report": len(unique) - 2100,
            },
        }
        states[db] = rng.bit_generator.state
    plan = {
        "schema": 1,
        "kind": "price_meta123_eval_plan",
        "databases": entries,
        "numpy_version": np.__version__,
        "rng_state": states,
        "target_databases_read": [],
    }
    plan["plan_sha256"] = plan_digest(plan)
    validate_plan(plan, config, "eval")
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


def validate_plan(plan, config, kind):
    validate_data_config(config)
    expected_kind = (
        "price_meta123_training_plan"
        if kind == "training"
        else "price_meta123_eval_plan"
    )
    if kind not in ("training", "eval") or not isinstance(plan, dict):
        raise ValueError("Unknown plan kind")
    if (
        plan.get("schema") != 1
        or plan.get("kind") != expected_kind
        or plan.get("target_databases_read") != []
        or plan.get("plan_sha256") != plan_digest(plan)
    ):
        raise ValueError("Plan type, target exposure or content digest changed")
    expected = TRAIN_DATABASES if kind == "training" else SOURCE_DATABASES
    if tuple(plan.get("databases", {})) != expected:
        raise ValueError("Plan database closure or order changed")
    for db, metadata in plan["databases"].items():
        for key in ("cache_hash", "data_sha256"):
            if not re.fullmatch(r"[0-9a-f]{64}", metadata.get(key, "")):
                raise ValueError("Missing cache provenance hash")
        if (
            type(metadata.get("rows")) is not int
            or type(metadata.get("unique_queries")) is not int
            or not 84 <= metadata["unique_queries"] <= metadata["rows"]
        ):
            raise ValueError("Invalid unique query denominator")
        if len(metadata.get("max_counts", [])) != 4 or any(
            type(n) is not int or not 0 <= n <= c
            for n, c in zip(metadata["max_counts"], CAPACITY)
        ):
            raise ValueError("Unsupported feature capacity")
        if kind == "eval":
            count = metadata["unique_queries"]
            if count != EXPECTED_UNIQUE[db]:
                raise ValueError(
                    "Source denominator does not match the registered population"
                )
            for name, size in (
                ("support", 100),
                ("calibration", 2000),
                ("report", count - 2100),
            ):
                _validate_identities(metadata[name], size, metadata["rows"])
                if metadata["counts"][name] != size:
                    raise ValueError("Population count changed")
            combined = [
                i
                for name in ("support", "calibration", "report")
                for i in metadata[name]["hashes"]
            ]
            combined_indices = [
                i
                for name in ("support", "calibration", "report")
                for i in metadata[name]["indices"]
            ]
            if len(set(combined)) != count or len(set(combined_indices)) != count:
                raise ValueError("Support/calibration/report overlap")
            blocks = metadata["support_blocks"]
            if len(blocks) != 5:
                raise ValueError("Expected five support blocks")
            for r, block in enumerate(blocks):
                _validate_identities(block, 20, metadata["rows"])
                if any(
                    block[k] != metadata["support"][k][r * 20 : (r + 1) * 20]
                    for k in block
                ):
                    raise ValueError("Support blocks are not frozen contiguous subsets")
    if kind == "training":
        if (
            plan.get("seed") not in SEEDS
            or plan.get("steps") != 2000
            or plan.get("meta_batch") != 4
            or len(plan.get("episodes", [])) != 8000
        ):
            raise ValueError("Training budget or stream changed")
        for episode in plan["episodes"]:
            if episode.get("db") not in TRAIN_DATABASES:
                raise ValueError("Training episode opened a forbidden database")
            rows = plan["databases"][episode["db"]]["rows"]
            _validate_identities(episode["support"], 20, rows)
            _validate_identities(episode["query"], 64, rows)
            for key in ("indices", "hashes", "ids"):
                if set(episode["support"][key]) & set(episode["query"][key]):
                    raise ValueError("Training support/query overlap")
    return plan


def verify_plan_stores(plan, stores):
    validate_plan(
        plan,
        {
            "cache_root": str(stores.root),
            "splits": {
                "train": list(TRAIN_DATABASES),
                "validation": list(SOURCE_DATABASES),
                "test": list(TARGET_DATABASES),
            },
        },
        "training" if "episodes" in plan else "eval",
    )
    for db, metadata in plan["databases"].items():
        store = stores.get(db)
        unique = _canonical(store)
        observed = _info(store, unique)
        if any(observed[k] != metadata[k] for k in observed):
            raise ValueError("Cache data or denominator changed after plan sealing")
        canonical = set(unique.tolist())
        identities = (
            [
                e[name]
                for e in plan["episodes"]
                if e["db"] == db
                for name in ("support", "query")
            ]
            if "episodes" in plan
            else [metadata[name] for name in ("support", "calibration", "report")]
        )
        for identity in identities:
            if (
                not set(identity["indices"]) <= canonical
                or _identities(store, identity["indices"]) != identity
            ):
                raise ValueError(
                    "Plan query identities no longer match canonical cache rows"
                )
    return True


def write_plan(path, plan):
    from .price_meta123_runtime import write_json

    write_json(path, plan)


def load_plan(path, config, kind):
    value = json.loads(Path(path).read_text())
    return validate_plan(value, config, kind)
