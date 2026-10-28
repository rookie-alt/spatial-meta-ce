import hashlib
import tempfile
import unittest
from pathlib import Path

import torch

from spatial_meta_ce.data import StorePool
from spatial_meta_ce.features import write_cache
from spatial_meta_ce.manifests import digest
from spatial_meta_ce.plans import (
    EVAL_DATABASES,
    EVAL_KIND,
    EVAL_SCHEMA,
    EXPECTED_QD,
    EXPECTED_UNIQUE,
    TRAINING_KIND,
    TRAINING_SCHEMA,
    build_source_eval_plan,
    validate_source_eval_plan,
    validate_training_plan,
    verify_source_eval_plan_against_stores,
)

DATA_CONFIG = {
    "splits": {
        "train": ["a"],
        "validation": list(EVAL_DATABASES),
        "test": ["t"],
    }
}


def h(value):
    return hashlib.sha256(value.encode()).hexdigest()


def training_plan():
    episodes = []
    for step in range(2):
        episodes.append(
            {
                "db": "a",
                "k": 5,
                "support": list(range(5)),
                "query": list(range(5, 9)),
                "support_hashes": [h(f"support-{step}-{i}") for i in range(5)],
                "query_hashes": [h(f"query-{step}-{i}") for i in range(4)],
            }
        )
    plan = {
        "schema": TRAINING_SCHEMA,
        "kind": TRAINING_KIND,
        "seed": 7,
        "steps": 2,
        "meta_batch": 1,
        "accumulation": 1,
        "support_sizes": [5],
        "query_size": 4,
        "databases": {"a": h("cache")},
        "data_hashes": {"a": h("data")},
        "episodes": episodes,
    }
    plan["plan_sha256"] = digest(plan)
    return plan


def eval_plan():
    blocks = []
    for block in range(5):
        indices = list(range(block * 20, (block + 1) * 20))
        blocks.append(
            {
                "support": indices,
                "support_hashes": [h(f"support-{i}") for i in indices],
            }
        )
    query = list(range(100, 110))
    databases = {}
    for db in EVAL_DATABASES:
        databases[db] = {
            "cache_hash": h(f"{db}-cache"),
            "data_sha256": h(f"{db}-data"),
            "unique_queries": 110,
            "query_count": len(query),
            "support_blocks": blocks,
            "query": query,
            "query_hashes": [h(f"{db}-query-{i}") for i in query],
        }
    plan = {
        "schema": EVAL_SCHEMA,
        "kind": EVAL_KIND,
        "support_seed": 2026,
        "support_blocks": 5,
        "support_block_size": 20,
        "databases": databases,
    }
    plan["plan_sha256"] = digest(plan)
    return plan


class TrainingPlanTests(unittest.TestCase):
    def test_valid_training_plan_passes(self):
        plan = training_plan()
        self.assertIs(validate_training_plan(plan, DATA_CONFIG), plan)

    def test_training_plan_rejects_support_query_overlap(self):
        plan = training_plan()
        plan["episodes"][0]["query"][0] = 0
        plan["plan_sha256"] = digest(
            {key: value for key, value in plan.items() if key != "plan_sha256"}
        )
        with self.assertRaisesRegex(ValueError, "overlap"):
            validate_training_plan(plan, DATA_CONFIG)

    def test_training_plan_rejects_validation_database(self):
        plan = training_plan()
        plan["episodes"][0]["db"] = EVAL_DATABASES[0]
        plan["plan_sha256"] = digest(
            {key: value for key, value in plan.items() if key != "plan_sha256"}
        )
        with self.assertRaisesRegex(ValueError, "non-training"):
            validate_training_plan(plan, DATA_CONFIG)

    def test_training_plan_rejects_digest_mismatch(self):
        plan = training_plan()
        plan["plan_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "digest"):
            validate_training_plan(plan, DATA_CONFIG)

    def test_training_plan_rejects_invalid_data_hash(self):
        plan = training_plan()
        plan["data_hashes"]["a"] = "not-a-hash"
        plan["plan_sha256"] = digest(
            {key: value for key, value in plan.items() if key != "plan_sha256"}
        )
        with self.assertRaisesRegex(ValueError, "invalid data hash"):
            validate_training_plan(plan, DATA_CONFIG)

    def test_training_plan_rejects_support_query_hash_overlap(self):
        plan = training_plan()
        plan["episodes"][0]["query_hashes"][0] = plan["episodes"][0]["support_hashes"][0]
        plan["plan_sha256"] = digest(
            {key: value for key, value in plan.items() if key != "plan_sha256"}
        )
        with self.assertRaisesRegex(ValueError, "hash"):
            validate_training_plan(plan, DATA_CONFIG)

    def test_training_plan_rejects_repeated_query_hash(self):
        plan = training_plan()
        plan["episodes"][0]["query_hashes"][1] = plan["episodes"][0]["query_hashes"][0]
        plan["plan_sha256"] = digest(
            {key: value for key, value in plan.items() if key != "plan_sha256"}
        )
        with self.assertRaisesRegex(ValueError, "repeats query_hashes"):
            validate_training_plan(plan, DATA_CONFIG)

    def test_training_plan_rejects_wrong_episode_count(self):
        plan = training_plan()
        plan["episodes"].pop()
        plan["plan_sha256"] = digest(
            {key: value for key, value in plan.items() if key != "plan_sha256"}
        )
        with self.assertRaisesRegex(ValueError, "episode count"):
            validate_training_plan(plan, DATA_CONFIG)


class SourceEvalPlanTests(unittest.TestCase):
    def test_valid_source_eval_plan_passes(self):
        plan = eval_plan()
        self.assertIs(validate_source_eval_plan(plan, DATA_CONFIG), plan)

    def test_source_eval_rejects_support_query_overlap(self):
        plan = eval_plan()
        plan["databases"][EVAL_DATABASES[0]]["query"][0] = 0
        plan["plan_sha256"] = digest(
            {key: value for key, value in plan.items() if key != "plan_sha256"}
        )
        with self.assertRaisesRegex(ValueError, "overlap"):
            validate_source_eval_plan(plan, DATA_CONFIG)

    def test_source_eval_rejects_denominator_mismatch(self):
        plan = eval_plan()
        plan["databases"][EVAL_DATABASES[0]]["unique_queries"] = 111
        plan["plan_sha256"] = digest(
            {key: value for key, value in plan.items() if key != "plan_sha256"}
        )
        with self.assertRaisesRegex(ValueError, "denominator"):
            validate_source_eval_plan(plan, DATA_CONFIG)

    def test_source_eval_rejects_digest_mismatch(self):
        plan = eval_plan()
        plan["plan_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "digest"):
            validate_source_eval_plan(plan, DATA_CONFIG)

    def test_source_eval_rejects_repeated_query_hash(self):
        plan = eval_plan()
        entry = plan["databases"][EVAL_DATABASES[0]]
        entry["query_hashes"][1] = entry["query_hashes"][0]
        plan["plan_sha256"] = digest(
            {key: value for key, value in plan.items() if key != "plan_sha256"}
        )
        with self.assertRaisesRegex(ValueError, "query hashes repeat"):
            validate_source_eval_plan(plan, DATA_CONFIG)


class SourceEvalPlanBuilderTests(unittest.TestCase):
    def make_cache(self, root, duplicate=None):
        cache = root / "cache"
        for db in EVAL_DATABASES:
            rows = [
                {
                    "id": f"{db}/pretrain/{i}",
                    "query_hash": h(f"{db}-{i}"),
                    "source_line": i + 1,
                    "true": float(10 + i),
                    "pg": 2.0,
                }
                for i in range(130)
            ]
            if db == EVAL_DATABASES[0] and duplicate is not None:
                extra = {
                    **rows[0],
                    "id": f"{db}/pretrain/duplicate",
                    "source_line": 999,
                }
                if duplicate == "conflict":
                    extra["true"] = 999.0
                rows.append(extra)
            data = {
                "data_features": [
                    (torch.ones(40), torch.ones(40), torch.ones(4), None)
                    for _ in rows
                ],
                "true_cards": [torch.tensor(row["true"]) for row in rows],
                "pg_est_cards": [torch.tensor(2.0) for _ in rows],
                "n_join_cols": [1] * len(rows),
                "n_fanouts": [1] * len(rows),
                "n_tables": [1] * len(rows),
                "n_filter_cols": [0] * len(rows),
            }
            write_cache(cache / "pretrain" / db, rows, data, {"source": "unit-fixture"})
        return cache

    def data_config(self, root, cache):
        return {
            "cache_root": str(cache),
            "data_root": str(root / "unused"),
            "splits": {
                "train": ["a"],
                "validation": list(EVAL_DATABASES),
                "test": ["t"],
            },
        }

    def test_builder_creates_100_support_and_remaining_query_population(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache = self.make_cache(root)
            data_config = self.data_config(root, cache)
            plan = build_source_eval_plan(data_config, expected_unique=None)
            validate_source_eval_plan(plan, data_config, expected_unique=None)
            for db in EVAL_DATABASES:
                entry = plan["databases"][db]
                self.assertEqual(entry["unique_queries"], 130)
                self.assertEqual(entry["query_count"], 30)
                self.assertEqual(
                    [len(block["support"]) for block in entry["support_blocks"]],
                    [20] * 5,
                )
            pool = StorePool(str(cache), max_open=len(EVAL_DATABASES))
            stores = {db: pool.get("pretrain", db) for db in EVAL_DATABASES}
            self.assertTrue(verify_source_eval_plan_against_stores(plan, stores))
            plan["databases"][EVAL_DATABASES[0]]["data_sha256"] = h("tampered")
            with self.assertRaisesRegex(ValueError, "label/feature"):
                verify_source_eval_plan_against_stores(plan, stores)

    def test_builder_collapses_identical_duplicate_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache = self.make_cache(root, duplicate="same")
            data_config = self.data_config(root, cache)
            plan = build_source_eval_plan(data_config, expected_unique=None)
            entry = plan["databases"][EVAL_DATABASES[0]]
            self.assertEqual(entry["unique_queries"], 130)
            self.assertEqual(entry["query_count"], 30)

    def test_builder_rejects_conflicting_duplicate_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache = self.make_cache(root, duplicate="conflict")
            data_config = self.data_config(root, cache)
            with self.assertRaisesRegex(ValueError, "Conflicting duplicate"):
                build_source_eval_plan(data_config, expected_unique=None)

    def test_expected_populations_match_doc_112(self):
        self.assertEqual(
            EXPECTED_UNIQUE,
            {"chembl": 50000, "legalacts": 49504, "telstra": 50000},
        )
        self.assertEqual(
            EXPECTED_QD,
            {"chembl": 49900, "legalacts": 49404, "telstra": 49900},
        )


if __name__ == "__main__":
    unittest.main()
