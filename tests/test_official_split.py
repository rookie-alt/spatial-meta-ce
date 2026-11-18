"""Official 26/4 split: exactness, capacity and unchanged frozen guards."""

import json
import tempfile
import unittest
from pathlib import Path

from spatial_meta_ce.data import TRANSFORM_VERSION
from spatial_meta_ce.manifests import digest, validate_splits
from spatial_meta_ce.official_split import (
    OFFICIAL_TRAIN_DATABASES,
    TEST_DATABASES,
    build_official_capacity,
    official_data_config,
    validate_official_capacity,
    validate_official_splits,
)
from spatial_meta_ce.price_capacity import SOURCE_DATABASES, TRAIN_DATABASES


def _fixture(root: Path) -> dict:
    cache = root / "cache" / "pretrain"
    counts = {db: [3, 4, 3, 7] for db in OFFICIAL_TRAIN_DATABASES}
    counts["chembl"] = [11, 12, 7, 10]
    counts["sakila"] = [12, 12, 7, 10]
    for db, value in counts.items():
        location = cache / db
        location.mkdir(parents=True)
        payload = {
            "schema": 1,
            "transform": TRANSFORM_VERSION,
            "inspection": {"max_counts": value},
            "rows": [],
        }
        (location / "manifest.json").write_text(
            json.dumps({**payload, "content_hash": digest(payload)})
        )
    return {
        "data_root": str(root / "price"),
        "cache_root": str(cache.parent),
        "splits": {
            "train": list(TRAIN_DATABASES),
            "validation": list(SOURCE_DATABASES),
            "test": list(TEST_DATABASES),
        },
    }


class SpecTemplateTests(unittest.TestCase):
    def test_training_spec_is_unsigned_and_binds_the_frozen_budget(self):
        from scripts import run_price_meta_a as runner

        path = (
            Path(__file__).resolve().parents[1]
            / "reports/official-split-20261117/official_split_training_spec.json"
        )
        spec = json.loads(path.read_text())
        self.assertEqual(spec["status"], "unsigned")
        self.assertFalse(spec["runnable"])
        self.assertEqual(spec["budget_sha256"], runner.canonical_digest(spec["budget"]))
        self.assertEqual(spec["target_databases_read"], [])
        self.assertEqual(spec["seeds"], [42, 43, 44])
        for seed, record in spec["bindings"]["training_plans"].items():
            self.assertIn(int(seed), spec["seeds"])
            self.assertEqual(len(record["sha256"]), 64)
            self.assertEqual(len(record["plan_sha256"]), 64)
        authorization = json.loads(
            (
                Path(__file__).resolve().parents[1]
                / "reports/official-split-20261117"
                / "official_split_authorization_template.json"
            ).read_text()
        )
        self.assertFalse(authorization["authorized"])
        self.assertIsNone(authorization["granted_at"])
        self.assertEqual(authorization["budget_sha256"], spec["budget_sha256"])
        self.assertEqual(set(spec["bindings"]["templates"]), {"42", "43", "44"})

    def test_config_derivation_pairs_a1_and_pg_from_the_same_template(self):
        from scripts import run_official_split as official

        with tempfile.TemporaryDirectory() as directory:
            config = official_data_config(_fixture(Path(directory)))
            capacity = build_official_capacity(config)
        template = {
            "fixed_stage": "122-A1",
            "seed": 42,
            "steps": 4000,
            "model": {"arch": "price_lite", "lora": False},
            "source_preflight": "/sealed/preflight.json",
            "source_preflight_sha256": "f" * 64,
        }
        spec = {
            "preregistration": "实验文档/126-加训剂量实验预注册.md",
            "bindings": {
                "data_config": {"path": "/d/data.json", "sha256": "a" * 64},
                "training_plans": {"42": {"path": "/p/plan42.json"}},
            }
        }
        a1 = official.derive_config(template, spec, capacity, "A1", 42, 4000)
        pg = official.derive_config(template, spec, capacity, "A1-PG", 42, 24000)
        self.assertEqual(a1["fixed_stage"], "122-A1")
        self.assertEqual(pg["fixed_stage"], "122-A1-PG")
        self.assertEqual(a1["steps"], 4000)
        self.assertEqual(pg["steps"], 24000)
        self.assertEqual(pg["preregistered_step_budget"], 24000)
        self.assertNotIn("preregistered_step_budget", a1)
        self.assertNotIn("source_preflight", a1)
        self.assertEqual(a1["training_plan"], "/p/plan42.json")
        self.assertNotIn("pg_head_input", a1["model"])
        self.assertTrue(pg["model"]["pg_head_input"])

    def test_longer_step_budget_requires_preregistration(self):
        from spatial_meta_ce.train import _validate_fixed_budget

        base = {"method": "erm", "meta_objective": "post", "steps": 24000}
        with self.assertRaisesRegex(ValueError, "steps=4000"):
            _validate_fixed_budget(dict(base))
        # Passing this gate reaches the next required field, so the failure is
        # a missing key rather than a rejected step budget.
        with self.assertRaises(KeyError):
            _validate_fixed_budget(
                {**base, "preregistered_step_budget": 24000, "preregistration": "x"}
            )
        with self.assertRaisesRegex(ValueError, "preregistered_step_budget"):
            _validate_fixed_budget(
                {**base, "preregistered_step_budget": 12000, "preregistration": "x"}
            )

    def test_unsigned_authorization_is_rejected(self):
        from scripts import run_official_split as official
        from scripts.run_price_meta_a import validate_budget

        budget = validate_budget(
            {
                "total_seconds": 60.0,
                "phase_seconds": {
                    "preparation": 20.0,
                    "evaluation": 1.0,
                    "certification": 20.0,
                },
                "rss_bytes": 1024,
                "vram_bytes": 1024,
                "output_bytes": 1024,
                "poll_seconds": 0.5,
                "cpu_core_seconds": 10,
                "gpu_allocated_seconds": 10,
                "run_deadline_seconds": 10.0,
                "eval_deadline_seconds": 1.0,
            }
        )
        with self.assertRaisesRegex(ValueError, "incomplete"):
            official.validate_authorization(
                {"protocol": "122-official-26", "purpose": "official-split-training"},
                "a" * 64,
                budget,
                Path("/tmp/out"),
            )


class OfficialSplitTests(unittest.TestCase):
    def test_config_promotes_the_three_source_databases_into_training(self):
        with tempfile.TemporaryDirectory() as directory:
            source = _fixture(Path(directory))
            config = official_data_config(source)
        self.assertEqual(tuple(config["splits"]["train"]), OFFICIAL_TRAIN_DATABASES)
        self.assertEqual(config["splits"]["validation"], [])
        self.assertEqual(tuple(config["splits"]["test"]), TEST_DATABASES)
        self.assertEqual(source["splits"]["train"], list(TRAIN_DATABASES))

    def test_split_validation_rejects_wrong_lists_and_leakage(self):
        good = {
            "train": list(OFFICIAL_TRAIN_DATABASES),
            "validation": [],
            "test": list(TEST_DATABASES),
        }
        validate_official_splits(good)
        for patch in (
            {"train": list(TRAIN_DATABASES)},
            {"validation": ["chembl"]},
            {"test": ["imdb"]},
            {"train": [*OFFICIAL_TRAIN_DATABASES, "imdb"]},
        ):
            with self.subTest(patch=patch), self.assertRaises(ValueError):
                validate_official_splits({**good, **patch})
        leaked = {**good, "test": ["imdb", "stats", "genome", "sakila"]}
        with self.assertRaises(ValueError):
            validate_official_splits(leaked)

    def test_default_split_guard_still_requires_validation(self):
        with self.assertRaisesRegex(ValueError, "validation"):
            validate_splits(
                {
                    "train": list(OFFICIAL_TRAIN_DATABASES),
                    "validation": [],
                    "test": list(TEST_DATABASES),
                }
            )

    def test_capacity_covers_26_caches_and_rejects_tampering(self):
        with tempfile.TemporaryDirectory() as directory:
            config = official_data_config(_fixture(Path(directory)))
            capacity = build_official_capacity(config)
        self.assertEqual(capacity["databases"], list(OFFICIAL_TRAIN_DATABASES))
        self.assertEqual(capacity["max_counts"], [12, 12, 7, 10])
        self.assertEqual(
            set(capacity["database_max_counts"]), set(OFFICIAL_TRAIN_DATABASES)
        )
        validate_official_capacity(capacity)
        for patch in (
            {"max_counts": [1, 1, 1, 1]},
            {"databases": list(TRAIN_DATABASES)},
            {"kind": "price_train_capacity_122"},
        ):
            with self.subTest(patch=patch), self.assertRaises(ValueError):
                validate_official_capacity({**capacity, **patch})