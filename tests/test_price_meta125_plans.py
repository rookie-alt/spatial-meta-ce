import copy
import hashlib
import importlib
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np


class FakeStore:
    """Synthetic PRICE cache stand-in; no real cache is read in tests."""

    def __init__(self, db, rows=2108):
        self.path = Path(db)
        self.rows = [
            {
                "id": f"{db}/pretrain/{i}",
                "query_hash": hashlib.sha256(f"{db}/{i}".encode()).hexdigest(),
                "source_line": i + 1,
            }
            for i in range(rows)
        ]
        self.manifest = {
            "content_hash": "a" * 64,
            "inspection": {"max_counts": [1, 1, 1, 0]},
        }


class FakeStores:
    """Synthetic store pool with the same public surface as TrainingStores."""

    def __init__(self, config, databases=None):
        module = importlib.import_module("spatial_meta_ce.price_meta125_plans")
        self.root = Path(config["cache_root"])
        self.databases = tuple(
            module.TRAIN_DATABASES if databases is None else databases
        )
        self.opened_databases = []
        self.values = {db: FakeStore(db) for db in self.databases}

    def get(self, db):
        if db not in self.values:
            raise ValueError("Database is outside this stage's store whitelist")
        if db not in self.opened_databases:
            self.opened_databases.append(db)
        return self.values[db]

    def close(self):
        self.values.clear()


class Official26PlanGuards(unittest.TestCase):
    def module(self):
        return importlib.import_module("spatial_meta_ce.price_meta125_plans")

    def config(self):
        module = self.module()
        return {
            "cache_root": "/unused",
            "splits": {
                "train": list(module.TRAIN_DATABASES),
                "validation": [],
                "test": list(module.TARGET_DATABASES),
            },
        }

    def fake_env(self):
        module = self.module()
        return (
            mock.patch.object(module, "TrainingStores", FakeStores),
            mock.patch.object(module, "_canonical", return_value=np.arange(2108)),
            mock.patch.object(module, "_store_data_sha256", return_value="b" * 64),
        )

    def test_official26_roles_and_target_order(self):
        module = self.module()
        self.assertEqual(len(module.TRAIN_DATABASES), 26)
        self.assertEqual(
            module.TRAIN_DATABASES,
            (
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
            ),
        )
        self.assertEqual(
            module.TARGET_DATABASES, ("imdb", "stats", "ergastf1", "genome")
        )
        self.assertFalse(set(module.TRAIN_DATABASES) & set(module.TARGET_DATABASES))
        module.validate_data_config(self.config())

    def test_legacy_23_3_config_rejected(self):
        module = self.module()
        legacy_train = [
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
        ]
        legacy = {
            "cache_root": "/unused",
            "splits": {
                "train": legacy_train,
                "validation": ["chembl", "legalacts", "telstra"],
                "test": ["imdb", "stats", "genome", "ergastf1"],
            },
        }
        with self.assertRaises(ValueError):
            module.validate_data_config(legacy)
        with self.assertRaises(ValueError):
            module.validate_data_config(
                {
                    "cache_root": "/unused",
                    "splits": {
                        "train": list(module.TRAIN_DATABASES),
                        "validation": ["chembl", "legalacts", "telstra"],
                        "test": list(module.TARGET_DATABASES),
                    },
                }
            )

    def test_reordered_or_padded_roles_rejected(self):
        module = self.module()
        config = self.config()
        changed = copy.deepcopy(config)
        changed["splits"]["train"].reverse()
        with self.assertRaises(ValueError):
            module.validate_data_config(changed)
        changed = copy.deepcopy(config)
        changed["splits"]["test"].reverse()
        with self.assertRaises(ValueError):
            module.validate_data_config(changed)
        changed = copy.deepcopy(config)
        changed["splits"]["train"].append("chembl")
        with self.assertRaises(ValueError):
            module.validate_data_config(changed)
        changed = copy.deepcopy(config)
        changed["splits"]["train"][0] = "imdb"
        with self.assertRaises(ValueError):
            module.validate_data_config(changed)
        changed = copy.deepcopy(config)
        changed["splits"]["test"] = list(module.TARGET_DATABASES) + ["imdb"]
        with self.assertRaises(ValueError):
            module.validate_data_config(changed)

    def test_stores_reject_target_and_mixed_roles(self):
        module = self.module()
        for databases in (("imdb",), ("stats",), ("ergastf1",), ("genome",)):
            with self.assertRaises(ValueError):
                module.TrainingStores(self.config(), databases)
        with self.assertRaises(ValueError):
            module.TrainingStores(self.config(), ("accidents", "imdb"))
        with self.assertRaises(ValueError):
            module.TrainingStores(self.config(), ())

    def test_target_store_rejected_before_any_cache_read(self):
        module = self.module()
        with tempfile.TemporaryDirectory() as directory:
            config = self.config()
            config["cache_root"] = directory
            stores = module.TrainingStores(config)
            for db in module.TARGET_DATABASES:
                with self.assertRaises(ValueError):
                    stores.get(db)
            self.assertEqual(stores.opened_databases, [])
            self.assertFalse((Path(directory) / "pretrain").exists())
            stores.close()

    def test_legacy_config_rejected_by_stores(self):
        module = self.module()
        legacy = {
            "cache_root": "/unused",
            "splits": {
                "train": ["accidents", "chembl"],
                "validation": ["legalacts"],
                "test": ["imdb", "stats", "ergastf1", "genome"],
            },
        }
        with self.assertRaises(ValueError):
            module.TrainingStores(legacy)

    def test_plan_is_deterministic_for_same_seed(self):
        module = self.module()
        stores, canonical, sha = self.fake_env()
        with stores, canonical, sha:
            first = module.make_training_plan(self.config(), 42)
            second = module.make_training_plan(self.config(), 42)
        self.assertEqual(first, second)
        self.assertEqual(first["plan_sha256"], second["plan_sha256"])
        self.assertEqual(first["kind"], "price_meta125_training_plan")
        self.assertEqual(first["seed"], 42)
        self.assertEqual(first["steps"], 2000)
        self.assertEqual(first["meta_batch"], 4)
        self.assertEqual(len(first["episodes"]), 8000)
        self.assertEqual(first["target_databases_read"], [])
        self.assertEqual(tuple(first["databases"]), module.TRAIN_DATABASES)

    def test_seeds_produce_distinct_streams(self):
        module = self.module()
        stores, canonical, sha = self.fake_env()
        with stores, canonical, sha:
            plan42 = module.make_training_plan(self.config(), 42)
            plan43 = module.make_training_plan(self.config(), 43)
        self.assertNotEqual(plan42["plan_sha256"], plan43["plan_sha256"])
        self.assertNotEqual(
            plan42["episodes"][0]["support"]["hashes"],
            plan43["episodes"][0]["support"]["hashes"],
        )

    def test_task_identities_are_disjoint_and_sized(self):
        module = self.module()
        stores, canonical, sha = self.fake_env()
        with stores, canonical, sha:
            plan = module.make_training_plan(self.config(), 42)
        for episode in plan["episodes"][:200]:
            self.assertEqual(len(episode["support"]["indices"]), 20)
            self.assertEqual(len(episode["query"]["indices"]), 64)
            for key in ("indices", "ids", "hashes"):
                self.assertFalse(
                    set(episode["support"][key]) & set(episode["query"][key])
                )
            self.assertEqual(len(set(episode["support"]["hashes"])), 20)
            self.assertEqual(len(set(episode["query"]["hashes"])), 64)

    def test_database_choice_covers_training_libraries_equally(self):
        module = self.module()
        stores, canonical, sha = self.fake_env()
        with stores, canonical, sha:
            plan = module.make_training_plan(self.config(), 42)
        counts = {db: 0 for db in module.TRAIN_DATABASES}
        for episode in plan["episodes"]:
            counts[episode["db"]] += 1
        expected = len(plan["episodes"]) / len(module.TRAIN_DATABASES)
        deviation = 4 * np.sqrt(expected * (1 - 1 / len(module.TRAIN_DATABASES)))
        for db, count in counts.items():
            self.assertLess(abs(count - expected), deviation, db)
        self.assertEqual(set(counts), set(module.TRAIN_DATABASES))

    def test_rng_matches_frozen_seed_sequence(self):
        module = self.module()
        stores, canonical, sha = self.fake_env()
        with stores, canonical, sha:
            plan = module.make_training_plan(self.config(), 42)
        expected = np.random.Generator(
            np.random.PCG64(np.random.SeedSequence([20261117, 125, 42]))
        )
        first_db = module.TRAIN_DATABASES[int(expected.integers(26))]
        self.assertEqual(plan["episodes"][0]["db"], first_db)

    def test_support_query_overlap_tamper_rejected(self):
        module = self.module()
        stores, canonical, sha = self.fake_env()
        with stores, canonical, sha:
            plan = module.make_training_plan(self.config(), 42)
        episode = plan["episodes"][0]
        episode["query"]["indices"][0] = episode["support"]["indices"][0]
        episode["query"]["ids"][0] = episode["support"]["ids"][0]
        episode["query"]["hashes"][0] = episode["support"]["hashes"][0]
        plan["plan_sha256"] = module.plan_digest(plan)
        with self.assertRaisesRegex(ValueError, "overlap"):
            module.validate_plan(plan, self.config())

    def test_tampered_sha_rejected(self):
        module = self.module()
        stores, canonical, sha = self.fake_env()
        with stores, canonical, sha:
            plan = module.make_training_plan(self.config(), 42)
        plan["databases"]["accidents"]["data_sha256"] = "c" * 64
        with self.assertRaises(ValueError):
            module.validate_plan(plan, self.config())

    def test_forged_plan_sha_rejected(self):
        module = self.module()
        stores, canonical, sha = self.fake_env()
        with stores, canonical, sha:
            plan = module.make_training_plan(self.config(), 42)
        plan["episodes"][0]["db"] = "imdb"
        with self.assertRaises(ValueError):
            module.validate_plan(plan, self.config())

    def test_target_database_in_plan_rejected(self):
        module = self.module()
        stores, canonical, sha = self.fake_env()
        with stores, canonical, sha:
            plan = module.make_training_plan(self.config(), 42)
        plan["target_databases_read"] = ["imdb"]
        plan["plan_sha256"] = module.plan_digest(plan)
        with self.assertRaises(ValueError):
            module.validate_plan(plan, self.config())

    def test_missing_database_in_plan_rejected(self):
        module = self.module()
        stores, canonical, sha = self.fake_env()
        with stores, canonical, sha:
            plan = module.make_training_plan(self.config(), 42)
        plan["databases"].pop("chembl")
        plan["plan_sha256"] = module.plan_digest(plan)
        with self.assertRaises(ValueError):
            module.validate_plan(plan, self.config())

    def test_verify_plan_stores_detects_cache_drift(self):
        module = self.module()
        stores, canonical, sha = self.fake_env()
        with stores, canonical, sha:
            plan = module.make_training_plan(self.config(), 42)
            self.assertTrue(
                module.verify_plan_stores(plan, module.TrainingStores(self.config()))
            )
        drifted = FakeStores(self.config())
        drifted.values["accidents"].manifest["content_hash"] = "d" * 64
        with stores, canonical, sha:
            with self.assertRaises(ValueError):
                module.verify_plan_stores(plan, drifted)

    def test_verify_plan_stores_detects_identity_tamper(self):
        module = self.module()
        stores, canonical, sha = self.fake_env()
        with stores, canonical, sha:
            plan = module.make_training_plan(self.config(), 42)
        plan["episodes"][0]["support"]["hashes"][0] = "e" * 64
        plan["plan_sha256"] = module.plan_digest(plan)
        drifted = FakeStores(self.config())
        with stores, canonical, sha:
            with self.assertRaises(ValueError):
                module.verify_plan_stores(plan, drifted)

    def test_unsupported_seed_and_budget_rejected(self):
        module = self.module()
        stores, canonical, sha = self.fake_env()
        with stores, canonical, sha:
            with self.assertRaises(ValueError):
                module.make_training_plan(self.config(), 41)
            with self.assertRaises(ValueError):
                module.make_training_plan(self.config(), 42, steps=1)


if __name__ == "__main__":
    unittest.main()
