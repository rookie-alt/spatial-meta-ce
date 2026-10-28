"""Deterministic source-only plans for the 112 joint-objective experiment.

The plans are data artifacts, not execution records.  They bind database
content hashes and query identities so that all three arms and all three seeds
consume the same episodes or the same evaluation population.
"""

import json
from pathlib import Path

import numpy as np

from .data import StorePool
from .manifests import digest, episode_indices, file_hash, validate_splits

TRAINING_SCHEMA = 1
TRAINING_KIND = "fomaml_joint_source_training_plan"
EVAL_SCHEMA = 1
EVAL_KIND = "fomaml_joint_source_eval_plan"
EVAL_DATABASES = ("chembl", "legalacts", "telstra")
SUPPORT_SEED = 2026
SUPPORT_BLOCKS = 5
SUPPORT_BLOCK_SIZE = 20
TRAINING_PLAN_SALT = 20261027
# Full unique-query populations (doc 81 / scripts/audit_p2_source.py).  The
# 112 evaluation Q_d is this population minus the 100 reserved support queries.
EXPECTED_UNIQUE = {"chembl": 50000, "legalacts": 49504, "telstra": 50000}
EXPECTED_QD = {
    db: count - SUPPORT_BLOCKS * SUPPORT_BLOCK_SIZE
    for db, count in EXPECTED_UNIQUE.items()
}


def _store_data_sha256(store):
    return digest(
        {
            "cards": file_hash(store.path / "cards.npy"),
            "offsets": file_hash(store.path / "offsets.npy"),
            "groups": [
                file_hash(store.path / f"group{i}.npy") for i in range(4)
            ],
        }
    )


def _is_hash(value):
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _same_row_content(store, left, right):
    if not np.array_equal(store.cards[left], store.cards[right]):
        return False
    for column, group in enumerate(store.groups):
        start_left, end_left = store.offsets[left, column], store.offsets[left + 1, column]
        start_right, end_right = (
            store.offsets[right, column],
            store.offsets[right + 1, column],
        )
        if end_left - start_left != end_right - start_right:
            return False
        if not np.array_equal(group[start_left:end_left], group[start_right:end_right]):
            return False
    return True


def _canonical_unique_indices(store):
    """One representative per hash, sorted by (query_hash, id, source_line)."""
    groups = {}
    for index, row in enumerate(store.rows):
        groups.setdefault(row["query_hash"], []).append(index)
    chosen = []
    for query_hash in sorted(groups):
        indices = sorted(
            groups[query_hash],
            key=lambda i: (store.rows[i]["id"], store.rows[i]["source_line"]),
        )
        representative = indices[0]
        for other in indices[1:]:
            if not _same_row_content(store, representative, other):
                raise ValueError(
                    f"Conflicting duplicate query hash {query_hash!r} in "
                    f"{store.path.name}"
                )
        chosen.append(representative)
    if len(chosen) != len(groups):
        raise ValueError("Unique-query canonicalization lost rows")
    return np.asarray(chosen, dtype=np.int64)


def _plan_digest(plan):
    return digest({k: v for k, v in plan.items() if k != "plan_sha256"})


def build_training_plan(
    data_config,
    *,
    seed,
    steps,
    meta_batch,
    accumulation,
    support_sizes,
    query_size,
):
    validate_splits(data_config["splits"])
    train_databases = list(data_config["splits"]["train"])
    pool = StorePool(data_config["cache_root"], max_open=len(train_databases))
    stores = {db: pool.get("pretrain", db) for db in train_databases}
    unique = {db: _canonical_unique_indices(stores[db]) for db in train_databases}
    rng = np.random.default_rng(np.random.SeedSequence([seed, TRAINING_PLAN_SALT]))
    episodes = []
    for _ in range(steps * meta_batch):
        db = str(rng.choice(train_databases))
        k = int(rng.choice(support_sizes))
        support, query = episode_indices(unique[db], k, query_size, rng)
        store = stores[db]
        episodes.append(
            {
                "db": db,
                "k": k,
                "support": [int(i) for i in support],
                "query": [int(i) for i in query],
                "support_hashes": [store.rows[int(i)]["query_hash"] for i in support],
                "query_hashes": [store.rows[int(i)]["query_hash"] for i in query],
            }
        )
    plan = {
        "schema": TRAINING_SCHEMA,
        "kind": TRAINING_KIND,
        "seed": seed,
        "steps": steps,
        "meta_batch": meta_batch,
        "accumulation": accumulation,
        "support_sizes": list(support_sizes),
        "query_size": query_size,
        "databases": {
            db: stores[db].manifest["content_hash"] for db in train_databases
        },
        "data_hashes": {db: _store_data_sha256(stores[db]) for db in train_databases},
        "episodes": episodes,
    }
    plan["plan_sha256"] = _plan_digest(plan)
    validate_training_plan(plan, data_config)
    return plan


def validate_training_plan(plan, data_config):
    validate_splits(data_config["splits"])
    if not isinstance(plan, dict) or plan.get("schema") != TRAINING_SCHEMA:
        raise ValueError("Unsupported training plan schema")
    if plan.get("kind") != TRAINING_KIND:
        raise ValueError("Unsupported training plan kind")
    expected = {
        "seed": plan.get("seed"),
        "steps": plan.get("steps"),
        "meta_batch": plan.get("meta_batch"),
        "accumulation": plan.get("accumulation"),
        "support_sizes": plan.get("support_sizes"),
        "query_size": plan.get("query_size"),
    }
    if any(value is None for value in expected.values()):
        raise ValueError("Training plan is missing fixed fields")
    if expected["steps"] < 1 or expected["meta_batch"] < 1:
        raise ValueError("Training plan has invalid step/meta_batch")
    if expected["accumulation"] < 1 or expected["query_size"] < 1:
        raise ValueError("Training plan has invalid accumulation/query_size")
    if not expected["support_sizes"] or any(k < 1 for k in expected["support_sizes"]):
        raise ValueError("Training plan has invalid support sizes")
    train_databases = list(data_config["splits"]["train"])
    if set(plan.get("databases", {})) != set(train_databases):
        raise ValueError("Training plan does not cover exactly the training databases")
    data_hashes = plan.get("data_hashes")
    if not isinstance(data_hashes, dict) or set(data_hashes) != set(train_databases):
        raise ValueError("Training plan is missing label/feature data hashes")
    if any(not _is_hash(value) for value in data_hashes.values()):
        raise ValueError("Training plan has an invalid data hash")
    episodes = plan.get("episodes")
    if not isinstance(episodes, list):
        raise ValueError("Training plan episodes must be a list")
    if len(episodes) != expected["steps"] * expected["meta_batch"]:
        raise ValueError("Training plan episode count does not match steps*meta_batch")
    for index, episode in enumerate(episodes):
        if not isinstance(episode, dict):
            raise ValueError(f"Invalid training episode {index}")
        db = episode.get("db")
        if db not in train_databases:
            raise ValueError(f"Training episode {index} uses a non-training database")
        if episode.get("k") not in expected["support_sizes"]:
            raise ValueError(f"Training episode {index} has an invalid support size")
        support = episode.get("support")
        query = episode.get("query")
        support_hashes = episode.get("support_hashes")
        query_hashes = episode.get("query_hashes")
        for name, values, size in (
            ("support", support, episode["k"]),
            ("query", query, expected["query_size"]),
        ):
            if not isinstance(values, list) or len(values) != size:
                raise ValueError(f"Training episode {index} has invalid {name} size")
            if any(
                isinstance(value, bool) or not isinstance(value, int)
                for value in values
            ):
                raise ValueError(f"Training episode {index} has non-integer {name}")
            if len(set(values)) != len(values):
                raise ValueError(f"Training episode {index} repeats {name} indices")
        if set(support) & set(query):
            raise ValueError(f"Training episode {index} overlaps support and query")
        if not isinstance(support_hashes, list) or not isinstance(query_hashes, list):
            raise ValueError(f"Training episode {index} is missing hash lists")
        if set(support_hashes) & set(query_hashes):
            raise ValueError(
                f"Training episode {index} overlaps support/query hashes"
            )
        for name, values, size in (
            ("support_hashes", support_hashes, episode["k"]),
            ("query_hashes", query_hashes, expected["query_size"]),
        ):
            if not isinstance(values, list) or len(values) != size:
                raise ValueError(f"Training episode {index} has invalid {name}")
            if any(not _is_hash(value) for value in values):
                raise ValueError(f"Training episode {index} has invalid {name} values")
        if len(set(support_hashes)) != len(support_hashes):
            raise ValueError(f"Training episode {index} repeats support_hashes")
        if len(set(query_hashes)) != len(query_hashes):
            raise ValueError(f"Training episode {index} repeats query_hashes")
    if plan.get("plan_sha256") != _plan_digest(plan):
        raise ValueError("Training plan digest mismatch")
    return plan


def validate_plan_matches_config(plan, config):
    for field in (
        "seed",
        "steps",
        "meta_batch",
        "accumulation",
        "support_sizes",
        "query_size",
    ):
        if plan.get(field) != config.get(field):
            raise ValueError(f"Training plan {field} does not match config")
    return True


def verify_plan_against_stores(plan, stores):
    for db, content_hash in plan["databases"].items():
        store = stores[db]
        if store.manifest["content_hash"] != content_hash:
            raise ValueError(f"Training plan cache hash mismatch for {db}")
        if plan["data_hashes"][db] != _store_data_sha256(store):
            raise ValueError(f"Training plan label/feature hash mismatch for {db}")
        for episode in plan["episodes"]:
            if episode["db"] != db:
                continue
            for name, hashes in (
                ("support", episode["support_hashes"]),
                ("query", episode["query_hashes"]),
            ):
                indices = episode[name]
                if len(indices) != len(hashes):
                    raise ValueError(f"Training plan {name} hash/index length mismatch")
                for index, expected in zip(indices, hashes):
                    if store.rows[index]["query_hash"] != expected:
                        raise ValueError(
                            f"Training plan {name} hash mismatch for {db} row {index}"
                        )
    return True


def load_training_plan(path, data_config):
    plan = json.loads(Path(path).read_text())
    return validate_training_plan(plan, data_config)


def build_source_eval_plan(data_config, expected_unique=EXPECTED_UNIQUE):
    validate_splits(data_config["splits"])
    if tuple(data_config["splits"]["validation"]) != EVAL_DATABASES:
        raise ValueError(
            f"Source evaluation databases must be exactly {EVAL_DATABASES}"
        )
    pool = StorePool(data_config["cache_root"], max_open=len(EVAL_DATABASES))
    databases = {}
    for db_index, db in enumerate(EVAL_DATABASES):
        store = pool.get("pretrain", db)
        unique = _canonical_unique_indices(store)
        if len(unique) <= SUPPORT_BLOCKS * SUPPORT_BLOCK_SIZE:
            raise ValueError(f"{db} has too few unique queries for the support plan")
        if expected_unique is not None and len(unique) != expected_unique[db]:
            raise ValueError(
                f"{db} unique-query population changed: {len(unique)} != "
                f"{expected_unique[db]}"
            )
        rng = np.random.default_rng(np.random.SeedSequence([SUPPORT_SEED, db_index]))
        permutation = rng.permutation(unique)
        support = permutation[: SUPPORT_BLOCKS * SUPPORT_BLOCK_SIZE]
        query = permutation[SUPPORT_BLOCKS * SUPPORT_BLOCK_SIZE :]
        blocks = []
        for block in range(SUPPORT_BLOCKS):
            indices = support[
                block * SUPPORT_BLOCK_SIZE : (block + 1) * SUPPORT_BLOCK_SIZE
            ]
            blocks.append(
                {
                    "support": [int(i) for i in indices],
                    "support_hashes": [
                        store.rows[int(i)]["query_hash"] for i in indices
                    ],
                }
            )
        databases[db] = {
            "cache_hash": store.manifest["content_hash"],
            "data_sha256": _store_data_sha256(store),
            "unique_queries": len(unique),
            "query_count": len(query),
            "support_blocks": blocks,
            "query": [int(i) for i in query],
            "query_hashes": [store.rows[int(i)]["query_hash"] for i in query],
        }
    plan = {
        "schema": EVAL_SCHEMA,
        "kind": EVAL_KIND,
        "support_seed": SUPPORT_SEED,
        "support_blocks": SUPPORT_BLOCKS,
        "support_block_size": SUPPORT_BLOCK_SIZE,
        "databases": databases,
    }
    plan["plan_sha256"] = _plan_digest(plan)
    validate_source_eval_plan(plan, data_config, expected_unique=expected_unique)
    return plan


def validate_source_eval_plan(plan, data_config, expected_unique=None):
    validate_splits(data_config["splits"])
    if not isinstance(plan, dict) or plan.get("schema") != EVAL_SCHEMA:
        raise ValueError("Unsupported source eval plan schema")
    if plan.get("kind") != EVAL_KIND:
        raise ValueError("Unsupported source eval plan kind")
    if plan.get("support_seed") != SUPPORT_SEED:
        raise ValueError("Source eval plan has an invalid support seed")
    if plan.get("support_blocks") != SUPPORT_BLOCKS:
        raise ValueError("Source eval plan has an invalid block count")
    if plan.get("support_block_size") != SUPPORT_BLOCK_SIZE:
        raise ValueError("Source eval plan has an invalid block size")
    if tuple(plan.get("databases", {})) != EVAL_DATABASES:
        raise ValueError("Source eval plan has an invalid database order")
    expected_support = SUPPORT_BLOCKS * SUPPORT_BLOCK_SIZE
    for db, entry in plan["databases"].items():
        if not isinstance(entry, dict):
            raise ValueError(f"Invalid source eval entry for {db}")
        if not _is_hash(entry.get("cache_hash")):
            raise ValueError(f"{db} source eval plan has an invalid cache hash")
        if not _is_hash(entry.get("data_sha256")):
            raise ValueError(f"{db} source eval plan has an invalid data hash")
        blocks = entry.get("support_blocks")
        query = entry.get("query")
        if not isinstance(blocks, list) or len(blocks) != SUPPORT_BLOCKS:
            raise ValueError(f"{db} source eval plan has invalid support blocks")
        support_indices = []
        support_hashes = []
        for block in blocks:
            indices = block.get("support") if isinstance(block, dict) else None
            hashes = block.get("support_hashes") if isinstance(block, dict) else None
            if not isinstance(indices, list) or len(indices) != SUPPORT_BLOCK_SIZE:
                raise ValueError(f"{db} source eval plan has invalid block size")
            if any(
                isinstance(value, bool) or not isinstance(value, int)
                for value in indices
            ):
                raise ValueError(f"{db} source eval plan has non-integer support")
            if not isinstance(hashes, list) or len(hashes) != SUPPORT_BLOCK_SIZE:
                raise ValueError(f"{db} source eval plan has invalid block hashes")
            if any(not _is_hash(value) for value in hashes):
                raise ValueError(f"{db} source eval plan has invalid block hash values")
            support_indices.extend(indices)
            support_hashes.extend(hashes)
        if len(set(support_indices)) != expected_support:
            raise ValueError(f"{db} source eval support blocks overlap")
        if len(set(support_hashes)) != expected_support:
            raise ValueError(f"{db} source eval support hashes repeat")
        if not isinstance(query, list) or not query:
            raise ValueError(f"{db} source eval plan has an empty query population")
        if any(isinstance(value, bool) or not isinstance(value, int) for value in query):
            raise ValueError(f"{db} source eval plan has non-integer query indices")
        if set(support_indices) & set(query):
            raise ValueError(f"{db} source eval support/query overlap")
        if len(set(query)) != len(query):
            raise ValueError(f"{db} source eval query repeats indices")
        query_hashes = entry.get("query_hashes")
        if not isinstance(query_hashes, list) or len(query_hashes) != len(query):
            raise ValueError(f"{db} source eval plan has invalid query hashes")
        if any(not _is_hash(value) for value in query_hashes):
            raise ValueError(f"{db} source eval plan has invalid query hashes")
        if len(set(query_hashes)) != len(query_hashes):
            raise ValueError(f"{db} source eval query hashes repeat")
        if set(support_hashes) & set(query_hashes):
            raise ValueError(f"{db} source eval support/query hash overlap")
        if entry.get("query_count") != len(query):
            raise ValueError(f"{db} source eval plan query_count mismatch")
        if entry.get("unique_queries") != len(set(support_indices) | set(query)):
            raise ValueError(f"{db} source eval plan denominator mismatch")
        if entry.get("unique_queries") != entry["query_count"] + expected_support:
            raise ValueError(f"{db} source eval plan support/Q_d denominator mismatch")
        if expected_unique is not None and entry["unique_queries"] != expected_unique[db]:
            raise ValueError(
                f"{db} unique-query population changed: {entry['unique_queries']} != "
                f"{expected_unique[db]}"
            )
    if plan.get("plan_sha256") != _plan_digest(plan):
        raise ValueError("Source eval plan digest mismatch")
    return plan


def verify_source_eval_plan_against_stores(plan, stores):
    for db, entry in plan["databases"].items():
        store = stores[db]
        if store.manifest["content_hash"] != entry["cache_hash"]:
            raise ValueError(f"Source eval cache hash mismatch for {db}")
        if entry["unique_queries"] != len(store.unique_indices()):
            raise ValueError(f"Source eval unique-query population mismatch for {db}")
        if _store_data_sha256(store) != entry["data_sha256"]:
            raise ValueError(f"Source eval label/feature hash mismatch for {db}")
        for block in entry["support_blocks"]:
            for index, expected in zip(block["support"], block["support_hashes"]):
                if store.rows[index]["query_hash"] != expected:
                    raise ValueError(f"Source eval support hash mismatch for {db}")
        for index, expected in zip(entry["query"], entry["query_hashes"]):
            if store.rows[index]["query_hash"] != expected:
                raise ValueError(f"Source eval query hash mismatch for {db}")
    return True


def load_source_eval_plan(path, data_config):
    plan = json.loads(Path(path).read_text())
    return validate_source_eval_plan(plan, data_config)
