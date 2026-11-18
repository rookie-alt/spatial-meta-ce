"""Frozen target-population plan for experiment 126 (four target libraries).

The plan is a data artifact, not an execution record.  It binds, per target
library, the complete unique-query test population and -- for the three
libraries that own an independent ``finetune`` cache -- the exact support
identities that every model, seed and run must share.

Protocol (doc 126):

* target order is ``imdb / stats / ergastf1 / genome``;
* the unique-query denominators are 521 / 2603 / 1258 / 1699 over raw rows
  766 / 2749 / 1422 / 2302 -- nothing is filtered by error, length or
  cardinality;
* IMDB has no independent ``finetune`` cache, so its K>0 cells do not exist.
  That is an observed property of the cache root, not a hard-coded verdict:
  the plan records the path and its existence, and a ``finetune/imdb`` cache
  that appears is refused instead of being silently used;
* support comes only from ``finetune``, is canonicalised, loses every query
  hash that also occurs in ``test``, and is then drawn without replacement by
  ``Generator(PCG64(SeedSequence([20261118, 126, db_index])))`` -- 100 queries
  split in order into five 20-query blocks;
* the PRICE layout is the elementwise maximum of the training layout
  ``(12, 12, 7, 10)`` and the library's own test and support maxima, which
  is ``(12, 14, 8, 10)`` for stats and ``(12, 12, 7, 10)`` elsewhere.

Reading a target cache is authorised here on purpose; the plan records the
authorisation instead of hiding it.
"""

from __future__ import annotations

import re
from collections import OrderedDict
from pathlib import Path

import numpy as np

from .manifests import digest

TARGET_DATABASES = ("imdb", "stats", "ergastf1", "genome")
SUPPORT_DATABASES = ("stats", "ergastf1", "genome")
ALLOWED_SPLITS = ("test", "finetune")
IMDB = "imdb"
MISSING_FINETUNE_REASON = "missing_independent_finetune_cache"

EXPECTED_UNIQUE = {"imdb": 521, "stats": 2603, "ergastf1": 1258, "genome": 1699}
EXPECTED_ROWS = {"imdb": 766, "stats": 2749, "ergastf1": 1422, "genome": 2302}
EXPECTED_MAXIMA = {
    "imdb": (12, 12, 7, 10),
    "stats": (12, 14, 8, 10),
    "ergastf1": (12, 12, 7, 10),
    "genome": (12, 12, 7, 10),
}
TRAIN_LAYOUT = (12, 12, 7, 10)
EXPECTED_SUPPORT_POOL = {"stats": 49959, "ergastf1": 49705, "genome": 49609}

SUPPORT_SIZE = 100
SUPPORT_BLOCK_SIZE = 20
SUPPORT_BLOCKS = 5
RNG_SALT = 20261118
RNG_EXPERIMENT = 126
PLAN_KIND = "price_meta126_target_plan"
SCHEMA = 1


def plan_digest(plan):
    """Content digest of a plan, excluding the digest field itself."""
    return digest({key: value for key, value in plan.items() if key != "plan_sha256"})


def _canonical(store):
    from .plans import _canonical_unique_indices

    return _canonical_unique_indices(store)


def _store_data_sha256(store):
    from .plans import _store_data_sha256 as data_sha256

    return data_sha256(store)


def _identities(store, indices):
    values = [int(i) for i in indices]
    return {
        "indices": values,
        "ids": [store.rows[i]["id"] for i in values],
        "hashes": [store.rows[i]["query_hash"] for i in values],
    }


def _same_identity(store, identity):
    return _identities(store, identity["indices"]) == {
        "indices": list(identity["indices"]),
        "ids": list(identity["ids"]),
        "hashes": list(identity["hashes"]),
    }


def _max_counts(store):
    maxima = list(store.manifest["inspection"]["max_counts"])
    if len(maxima) != 4 or any(type(n) is not int or n < 0 for n in maxima):
        raise ValueError("Cache max_counts are not four non-negative integers")
    return maxima


def _check_provenance(store, db, split):
    """Refuse a cache that is not exactly this database and split."""
    provenance = store.manifest.get("provenance", {})
    source = provenance.get("source", {})
    if source.get("database") != db or source.get("split") != split:
        raise ValueError("Cache provenance does not match its source database")
    if provenance.get("excluded"):
        raise ValueError("Cache was built with excluded queries")
    prefix = f"{db}/{split}/"
    for row in store.manifest.get("rows", []):
        if not str(row.get("id", "")).startswith(prefix):
            raise ValueError("Cache query identities refer to another database")


class TargetStores:
    """Read-only, role-bound access to the four target caches.

    Only ``test`` for every target library and ``finetune`` for the three
    libraries that own an independent support cache may be opened.  Source
    databases, the ``pretrain`` split and the IMDB ``finetune`` split are
    refused before any file is touched.
    """

    def __init__(self, cache_root):
        if not isinstance(cache_root, (str, Path)) or not str(cache_root):
            raise ValueError("TargetStores requires a cache root")
        self.root = Path(cache_root)
        self.opened = []
        self._stores = OrderedDict()

    def path_for(self, split, db):
        return self.root / split / db

    def finetune_available(self, db):
        """Whether an independent ``finetune`` cache exists on disk."""
        if db not in TARGET_DATABASES:
            raise ValueError("Database is outside the target whitelist")
        return self.path_for("finetune", db).exists()

    def get(self, split, db):
        if split not in ALLOWED_SPLITS:
            raise ValueError("Split is outside the target whitelist")
        if db not in TARGET_DATABASES:
            raise ValueError("Database is outside the target whitelist")
        if split == "finetune" and db == IMDB:
            raise ValueError(
                "IMDB has no independent finetune cache; its K>0 cells do not exist"
            )
        key = (split, db)
        if key not in self._stores:
            path = self.path_for(split, db)
            if path.is_symlink() or not path.resolve().is_relative_to(
                self.root.resolve()
            ):
                raise ValueError("Linked or aliased target cache is forbidden")
            required = [
                path / "manifest.json",
                path / "cards.npy",
                path / "offsets.npy",
            ]
            required += [path / f"group{i}.npy" for i in range(4)]
            if any(item.is_symlink() for item in required):
                raise ValueError("Linked cache arrays or manifests are forbidden")
            from .data import FeatureStore

            store = FeatureStore(path)
            _check_provenance(store, db, split)
            self._stores[key] = store
            self.opened.append(key)
        return self._stores[key]

    def close(self):
        self._stores.clear()


def _layout(train_layout, test_maxima, support_maxima):
    maxima = list(train_layout)
    for source in (test_maxima, support_maxima):
        for index, value in enumerate(source):
            maxima[index] = max(maxima[index], value)
    return maxima


def _support_pool(finetune_store, test_hashes):
    """Canonical unique support queries with every test hash removed."""
    unique = _canonical(finetune_store)
    support_hashes = {finetune_store.rows[int(i)]["query_hash"] for i in unique}
    pool = [
        int(index)
        for index in unique
        if finetune_store.rows[int(index)]["query_hash"] not in test_hashes
    ]
    overlap = len(support_hashes & set(test_hashes))
    return unique, pool, overlap


def _support_metadata(store, db, unique_count, pool, overlap):
    return {
        "cache_hash": store.manifest["content_hash"],
        "data_sha256": _store_data_sha256(store),
        "rows": len(store.rows),
        "unique_queries": int(unique_count),
        "max_counts": _max_counts(store),
        "pool_size": len(pool),
        "overlap_removed": int(overlap),
        "expected_pool_size": EXPECTED_SUPPORT_POOL.get(db),
        "bit_generator": "PCG64",
        "seed_sequence": [RNG_SALT, RNG_EXPERIMENT, TARGET_DATABASES.index(db)],
    }


def _draw_support(pool, db_index):
    rng = np.random.Generator(
        np.random.PCG64(np.random.SeedSequence([RNG_SALT, RNG_EXPERIMENT, db_index]))
    )
    return rng.choice(
        np.asarray(pool, dtype=np.int64), size=SUPPORT_SIZE, replace=False
    )


def build_plan(cache_root):
    """Build the frozen four-target plan for experiment 126."""
    stores = TargetStores(cache_root)
    databases = {}
    try:
        for db_index, db in enumerate(TARGET_DATABASES):
            test_store = stores.get("test", db)
            unique = _canonical(test_store)
            if len(unique) != EXPECTED_UNIQUE[db]:
                raise ValueError(
                    f"{db} unique-query denominator differs from the frozen population"
                )
            if len(test_store.rows) != EXPECTED_ROWS[db]:
                raise ValueError(
                    f"{db} raw row count differs from the frozen population"
                )
            test_maxima = _max_counts(test_store)
            test_hashes = {test_store.rows[int(i)]["query_hash"] for i in unique}

            support = {
                "available": False,
                "reason": None,
                "blocks": [],
                "blocks_sha256": [],
                "metadata": None,
                "cache_present": None,
                "cache_path": str(stores.path_for("finetune", db)),
            }
            support_maxima = [0, 0, 0, 0]
            if db == IMDB:
                present = stores.finetune_available(IMDB)
                support["cache_present"] = bool(present)
                if present:
                    raise ValueError(
                        "IMDB finetune cache exists on disk; the frozen protocol "
                        "records it as absent, so a supplementary protocol is "
                        "required instead of a silent few-shot switch"
                    )
                support["reason"] = MISSING_FINETUNE_REASON
            else:
                if not stores.finetune_available(db):
                    raise ValueError(f"{db} has no independent finetune cache")
                finetune_store = stores.get("finetune", db)
                unique_support, pool, overlap = _support_pool(
                    finetune_store, test_hashes
                )
                if len(pool) < SUPPORT_SIZE:
                    raise ValueError(f"{db} has too few independent support queries")
                support["cache_present"] = True
                support["available"] = True
                support["metadata"] = _support_metadata(
                    finetune_store, db, len(unique_support), pool, overlap
                )
                support_maxima = support["metadata"]["max_counts"]
                chosen = _draw_support(pool, db_index)
                support["blocks"] = [
                    {
                        **_identities(
                            finetune_store,
                            chosen[
                                r * SUPPORT_BLOCK_SIZE : (r + 1) * SUPPORT_BLOCK_SIZE
                            ],
                        ),
                    }
                    for r in range(SUPPORT_BLOCKS)
                ]
                support["blocks_sha256"] = [
                    _identity_digest(block) for block in support["blocks"]
                ]

            layout = _layout(TRAIN_LAYOUT, test_maxima, support_maxima)
            if tuple(layout) != EXPECTED_MAXIMA[db]:
                raise ValueError(
                    f"{db} layout {layout} differs from the frozen layout "
                    f"{list(EXPECTED_MAXIMA[db])}"
                )
            test_identity = {
                "indices": [int(i) for i in unique],
                "ids": [test_store.rows[int(i)]["id"] for i in unique],
                "hashes": [test_store.rows[int(i)]["query_hash"] for i in unique],
            }
            databases[db] = {
                "test": {
                    "cache_hash": test_store.manifest["content_hash"],
                    "data_sha256": _store_data_sha256(test_store),
                    "rows": len(test_store.rows),
                    "unique_queries": len(unique),
                    "max_counts": test_maxima,
                    **test_identity,
                    "identity_sha256": _identity_digest(test_identity),
                },
                "support": support,
                "max_counts": layout,
            }
    finally:
        stores.close()
    plan = {
        "schema": SCHEMA,
        "kind": PLAN_KIND,
        "databases": databases,
        "target_databases_read": list(TARGET_DATABASES),
        "caches_read": sorted(
            f"{split}/{db}"
            for split in ALLOWED_SPLITS
            for db in TARGET_DATABASES
            if not (split == "finetune" and db == IMDB)
        ),
        "numpy_version": np.__version__,
    }
    plan["plan_sha256"] = plan_digest(plan)
    validate_plan(plan)
    return plan


def _check_identity(identity, count, rows, label):
    if not isinstance(identity, dict) or set(identity) != {"indices", "ids", "hashes"}:
        raise ValueError(f"{label}: query identity fields changed")
    for key in ("indices", "ids", "hashes"):
        if not isinstance(identity[key], list) or len(identity[key]) != count:
            raise ValueError(f"{label}: query identity counts disagree")
    if any(type(i) is not int or not 0 <= i < rows for i in identity["indices"]):
        raise ValueError(f"{label}: query index outside cache")
    if any(not isinstance(x, str) or not x for x in identity["ids"]):
        raise ValueError(f"{label}: query IDs must be nonempty strings")
    if any(
        not isinstance(x, str) or not re.fullmatch(r"[0-9a-f]{64}", x)
        for x in identity["hashes"]
    ):
        raise ValueError(f"{label}: invalid query hash")
    for key in ("indices", "ids", "hashes"):
        if len(set(identity[key])) != count:
            raise ValueError(f"{label}: duplicate query identities")


def _check_identity_consistency(identity, label):
    """Index, ID and hash must describe the same rows, in the same order."""
    for position, row_id in enumerate(identity["ids"]):
        parts = row_id.rsplit("/", 1)
        if len(parts) != 2 or not parts[1].isdigit():
            raise ValueError(f"{label}: query ID is not a cache row identity")
        if int(parts[1]) != identity["indices"][position] + 1:
            raise ValueError(f"{label}: query index and ID disagree")


def _identity_digest(identity):
    """Digest of one query identity triple, so a re-sealed plan cannot swap it."""
    return digest(
        {
            "indices": list(identity["indices"]),
            "ids": list(identity["ids"]),
            "hashes": list(identity["hashes"]),
        }
    )


def validate_plan(plan):
    """Validate a sealed plan against the frozen 126 contract."""
    if not isinstance(plan, dict):
        raise ValueError("Unknown plan kind")
    if plan.get("schema") != SCHEMA or plan.get("kind") != PLAN_KIND:
        raise ValueError("Plan type changed")
    if tuple(plan.get("target_databases_read", ())) != TARGET_DATABASES:
        raise ValueError("Plan does not record the authorised target exposure")
    if plan.get("plan_sha256") != plan_digest(plan):
        raise ValueError("Plan content digest changed")
    if tuple(plan.get("databases", {})) != TARGET_DATABASES:
        raise ValueError("Plan database closure or order changed")
    for db, entry in plan["databases"].items():
        test = entry.get("test", {})
        for key in ("cache_hash", "data_sha256"):
            if not re.fullmatch(r"[0-9a-f]{64}", test.get(key, "")):
                raise ValueError(f"{db}: missing cache provenance hash")
        if (
            type(test.get("rows")) is not int
            or type(test.get("unique_queries")) is not int
            or not test["unique_queries"] <= test["rows"]
        ):
            raise ValueError(f"{db}: invalid unique query denominator")
        if (
            test["unique_queries"] != EXPECTED_UNIQUE[db]
            or test["rows"] != EXPECTED_ROWS[db]
        ):
            raise ValueError(
                f"{db}: unique-query denominator or row count differs from the "
                "frozen population"
            )
        if len(test.get("max_counts", [])) != 4 or any(
            type(n) is not int or n < 0 for n in test["max_counts"]
        ):
            raise ValueError(f"{db}: unsupported feature capacity")
        _check_identity(
            {
                "indices": test.get("indices"),
                "ids": test.get("ids"),
                "hashes": test.get("hashes"),
            },
            test["unique_queries"],
            test["rows"],
            f"{db} test",
        )
        _check_identity_consistency(
            {
                "indices": test.get("indices"),
                "ids": test.get("ids"),
                "hashes": test.get("hashes"),
            },
            f"{db} test",
        )
        if test.get("identity_sha256") != _identity_digest(
            {
                "indices": test.get("indices"),
                "ids": test.get("ids"),
                "hashes": test.get("hashes"),
            }
        ):
            raise ValueError(f"{db}: test query identities were altered")
        layout = entry.get("max_counts")
        if layout != list(EXPECTED_MAXIMA[db]) or any(
            layout[i] < TRAIN_LAYOUT[i] for i in range(4)
        ):
            raise ValueError(f"{db}: layout differs from the frozen layout")
        support = entry.get("support", {})
        if set(support) != {
            "available",
            "reason",
            "blocks",
            "blocks_sha256",
            "metadata",
            "cache_present",
            "cache_path",
        }:
            raise ValueError(f"{db}: support fields changed")
        if db == IMDB:
            if support["available"] or support["reason"] != MISSING_FINETUNE_REASON:
                raise ValueError("IMDB must not claim an independent support pool")
            if support["cache_present"] or support["blocks"]:
                raise ValueError("IMDB must not carry support identities")
            if support["metadata"] is not None:
                raise ValueError("IMDB must not carry support metadata")
            if support["blocks_sha256"] != []:
                raise ValueError("IMDB must not carry support block digests")
            continue
        if not support["available"] or support["reason"] is not None:
            raise ValueError(f"{db}: support must be available from its finetune cache")
        metadata = support["metadata"] or {}
        for key in ("cache_hash", "data_sha256"):
            if not re.fullmatch(r"[0-9a-f]{64}", metadata.get(key, "")):
                raise ValueError(f"{db}: missing support provenance hash")
        if (
            type(metadata.get("pool_size")) is not int
            or metadata["pool_size"] < SUPPORT_SIZE
            or type(metadata.get("overlap_removed")) is not int
            or metadata["overlap_removed"] < 0
        ):
            raise ValueError(f"{db}: invalid support pool record")
        blocks = support["blocks"]
        if len(blocks) != SUPPORT_BLOCKS:
            raise ValueError(f"{db}: expected five support blocks")
        flat = []
        for index, block in enumerate(blocks):
            _check_identity(
                block, SUPPORT_BLOCK_SIZE, metadata["rows"], f"{db} block {index}"
            )
            _check_identity_consistency(block, f"{db} block {index}")
            if support["blocks_sha256"][index] != _identity_digest(block):
                raise ValueError(f"{db}: support block {index} was altered")
            flat.extend(block["hashes"])
        if len(set(flat)) != SUPPORT_SIZE:
            raise ValueError(f"{db}: support blocks overlap")
        if set(flat) & set(test["hashes"]):
            raise ValueError(f"{db}: support and test share a query identity")
    return plan


def verify_plan(plan, cache_root):
    """Re-check a sealed plan against the live target caches."""
    validate_plan(plan)
    stores = TargetStores(cache_root)
    try:
        for db_index, db in enumerate(TARGET_DATABASES):
            entry = plan["databases"][db]
            test_store = stores.get("test", db)
            unique = _canonical(test_store)
            observed_test = {
                "cache_hash": test_store.manifest["content_hash"],
                "data_sha256": _store_data_sha256(test_store),
                "rows": len(test_store.rows),
                "unique_queries": len(unique),
                "max_counts": _max_counts(test_store),
            }
            recorded_test = entry["test"]
            if any(observed_test[key] != recorded_test[key] for key in observed_test):
                raise ValueError(f"{db}: test cache changed after plan sealing")
            if not _same_identity(test_store, recorded_test):
                raise ValueError(
                    f"{db}: test query identities changed after plan sealing"
                )

            support = entry["support"]
            if not support["available"]:
                if db == IMDB:
                    if stores.finetune_available(IMDB):
                        raise ValueError(
                            "IMDB finetune cache appeared after plan sealing"
                        )
                    continue
                raise ValueError(f"{db}: support must be available")

            finetune_store = stores.get("finetune", db)
            test_hashes = {test_store.rows[int(i)]["query_hash"] for i in unique}
            unique_support, pool, overlap = _support_pool(finetune_store, test_hashes)
            observed_support = _support_metadata(
                finetune_store, db, len(unique_support), pool, overlap
            )
            recorded_support = support["metadata"]
            if any(
                observed_support[key] != recorded_support[key]
                for key in (
                    "cache_hash",
                    "data_sha256",
                    "rows",
                    "unique_queries",
                    "max_counts",
                    "pool_size",
                    "overlap_removed",
                )
            ):
                raise ValueError(f"{db}: support cache changed after plan sealing")
            chosen = _draw_support(pool, db_index)
            for index in range(SUPPORT_BLOCKS):
                expected = _identities(
                    finetune_store,
                    chosen[
                        index * SUPPORT_BLOCK_SIZE : (index + 1) * SUPPORT_BLOCK_SIZE
                    ],
                )
                if expected != support["blocks"][index]:
                    raise ValueError(
                        f"{db}: support block {index} changed after plan sealing"
                    )
    finally:
        stores.close()
    return True
