import copy
import importlib
import unittest
import hashlib
import tempfile
from pathlib import Path
from unittest import mock

import numpy as np


class PlanGuards(unittest.TestCase):
    def module(self):
        return importlib.import_module("spatial_meta_ce.price_meta123_plans")

    def config(self):
        module = self.module()
        return {
            "cache_root": "/unused",
            "splits": {
                "train": list(module.TRAIN_DATABASES),
                "validation": list(module.SOURCE_DATABASES),
                "test": list(module.TARGET_DATABASES),
            },
        }

    def test_target_train_and_reordered_roles_rejected(self):
        module = self.module()
        config = self.config()
        module.validate_data_config(config)
        for role in ("train", "validation", "test"):
            changed = copy.deepcopy(config)
            changed["splits"][role].reverse()
            with self.assertRaises(ValueError):
                module.validate_data_config(changed)
        config["splits"]["train"][0] = "imdb"
        with self.assertRaises(ValueError):
            module.validate_data_config(config)

    def test_stores_reject_target_before_feature_store_import(self):
        module = self.module()
        for databases in (("imdb",), ("accidents", "chembl")):
            with self.assertRaises(ValueError):
                module.SourceStores(self.config(), databases)

    def test_source_alias_cannot_point_to_target_cache(self):
        module = self.module()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            (root / "pretrain").mkdir()
            target = root / "test/imdb"
            target.mkdir(parents=True)
            (root / "pretrain/accidents").symlink_to(target)
            config = self.config()
            config["cache_root"] = str(root)
            stores = module.SourceStores(config, ("accidents",))
            with self.assertRaisesRegex(ValueError, "link|alias|cache"):
                stores.get("accidents")

    def test_digest_change_detected_before_store_loading(self):
        module = self.module()
        plan = {
            "schema": 1,
            "kind": "price_meta123_eval_plan",
            "databases": {},
            "target_databases_read": [],
        }
        plan["plan_sha256"] = module.plan_digest(plan)
        plan["databases"]["imdb"] = {}
        with self.assertRaises(ValueError):
            module.validate_plan(plan, self.config(), "eval")

    def fake_stores(self, module):
        class Store:
            def __init__(self, db):
                self.path = Path(db)
                self.rows = [
                    {
                        "id": f"{db}/{i}",
                        "query_hash": hashlib.sha256(f"{db}/{i}".encode()).hexdigest(),
                        "source_line": i + 1,
                    }
                    for i in range(2108)
                ]
                self.manifest = {
                    "content_hash": "a" * 64,
                    "inspection": {"max_counts": [1, 1, 1, 0]},
                }

        class Stores:
            def __init__(self, _config, databases):
                self.values = {db: Store(db) for db in databases}

            def get(self, db):
                return self.values[db]

        return Stores

    def test_eval_identity_permutation_and_overlap_refusal(self):
        module = self.module()
        counts = {db: 2108 for db in module.SOURCE_DATABASES}
        with (
            mock.patch.object(module, "SourceStores", self.fake_stores(module)),
            mock.patch.object(module, "_canonical", return_value=np.arange(2108)),
            mock.patch.object(module, "EXPECTED_UNIQUE", counts),
            mock.patch(
                "spatial_meta_ce.plans._store_data_sha256", return_value="b" * 64
            ),
        ):
            first = module.make_eval_plan(self.config())
            second = module.make_eval_plan(self.config())
        self.assertEqual(first, second)
        entry = first["databases"]["chembl"]
        expected = np.random.Generator(
            np.random.PCG64(np.random.SeedSequence([20261116, 0]))
        ).permutation(np.arange(2108))
        self.assertEqual(entry["support"]["indices"], expected[:100].tolist())
        self.assertEqual(entry["report"]["indices"], expected[2100:].tolist())
        entry["report"]["indices"][0] = entry["support"]["indices"][0]
        entry["report"]["hashes"][0] = entry["support"]["hashes"][0]
        first["plan_sha256"] = module.plan_digest(first)
        with mock.patch.object(module, "EXPECTED_UNIQUE", counts):
            with self.assertRaisesRegex(ValueError, "overlap"):
                module.validate_plan(first, self.config(), "eval")

    def test_training_is_fixed_stream_and_same_task_identity(self):
        module = self.module()
        with (
            mock.patch.object(module, "SourceStores", self.fake_stores(module)),
            mock.patch.object(module, "_canonical", return_value=np.arange(2108)),
            mock.patch(
                "spatial_meta_ce.plans._store_data_sha256", return_value="b" * 64
            ),
        ):
            plan = module.make_training_plan(self.config(), 42)
            again = module.make_training_plan(self.config(), 42)
        self.assertEqual(plan["plan_sha256"], again["plan_sha256"])
        self.assertEqual(len(plan["episodes"]), 8000)
        task = plan["episodes"][0]
        self.assertEqual(len(task["support"]["indices"]), 20)
        self.assertEqual(len(task["query"]["indices"]), 64)
        self.assertFalse(set(task["support"]["hashes"]) & set(task["query"]["hashes"]))
        with self.assertRaises(ValueError):
            module.make_training_plan(self.config(), 42, steps=1)

    def test_plan_files_never_overwrite(self):
        module = self.module()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "plan.json"
            module.write_plan(path, {"hello": "world"})
            with self.assertRaises(FileExistsError):
                module.write_plan(path, {"hello": "again"})


if __name__ == "__main__":
    unittest.main()
