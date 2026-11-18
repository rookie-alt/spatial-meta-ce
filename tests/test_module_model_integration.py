"""Official module dispatch and mapping scopes; synthetic CPU data only."""

import copy
import tempfile
import unittest
from pathlib import Path

import torch
from test_price_group_pooling import grouped_sample
from test_price_relation_attention import synthetic_capacity, synthetic_sample, unmapped

from spatial_meta_ce.checkpoints import load_checkpoint, save_checkpoint
from spatial_meta_ce.data import batch_samples
from spatial_meta_ce.manifests import digest
from spatial_meta_ce.model import create_model
from spatial_meta_ce.price_capacity import (
    OFFICIAL_CAPACITY_KIND,
    OFFICIAL_TRAIN_DATABASES,
)


def official_binding():
    binding = synthetic_capacity()
    binding.update(
        kind=OFFICIAL_CAPACITY_KIND,
        databases=list(OFFICIAL_TRAIN_DATABASES),
        database_max_counts={db: [4, 6, 3, 3] for db in OFFICIAL_TRAIN_DATABASES},
        cache_hashes={
            db: digest(["synthetic-cache", db]) for db in OFFICIAL_TRAIN_DATABASES
        },
        manifest_sha256={
            db: digest(["synthetic-manifest", db]) for db in OFFICIAL_TRAIN_DATABASES
        },
    )
    binding["capacity_sha256"] = digest(
        {k: v for k, v in binding.items() if k != "capacity_sha256"}
    )
    return binding


class ModuleModelIntegrationTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(42)
        self.binding = official_binding()
        self.config = {
            "arch": "price_lite",
            "capacity": self.binding,
            "pg_head_input": True,
        }

    def test_joint_registration_preserves_independent_candidate_types(self):
        expected = (
            ({}, "price_lite_pg_residual"),
            ({"group_pooling": "mean"}, "price_lite_pg_group_mean_residual"),
            ({"group_pooling": "attention"}, "price_lite_pg_group_attention_residual"),
            ({"relation_attention": True}, "price_lite_pg_relation_residual"),
            (
                {"relation_attention": True, "relation_scope": "official_26_4"},
                "price_lite_pg_official_relation_residual",
            ),
        )
        for extra, model_type in expected:
            with self.subTest(extra=extra):
                model = create_model({**self.config, **extra})
                self.assertEqual(model.model_type, model_type)
                self.assertEqual(
                    create_model(model.config, model_type).config, model.config
                )
        for extra in (
            {"group_pooling": "mean", "relation_attention": True},
            {"group_pooling": "attention", "relation_attention": True},
            {"relation_scope": "official_26_4"},
            {"relation_attention": True, "relation_scope": "unknown"},
        ):
            with self.subTest(extra=extra), self.assertRaises(ValueError):
                create_model({**self.config, **extra})

    def test_official_relation_requires_explicit_matching_database_and_split(self):
        model = create_model(
            {
                **self.config,
                "relation_attention": True,
                "relation_scope": "official_26_4",
            }
        )
        for database, split in (
            ("chembl", "pretrain"),
            ("imdb", "test"),
            ("stats", "test"),
        ):
            batch = batch_samples([synthetic_sample(database=database)])
            marked = {**batch, "mapping_scope": "official_26_4", "mapping_split": split}
            self.assertTrue(torch.equal(model(marked), marked["pg_log"]))
            for changes in (
                {"mapping_scope": None},
                {"mapping_split": None},
                {"mapping_scope": "source"},
                {"mapping_split": "finetune"},
                {"mapping_split": "pretrain" if split == "test" else "test"},
                {"databases": ["unknown"]},
            ):
                with (
                    self.subTest(database=database, changes=changes),
                    self.assertRaises(ValueError),
                ):
                    model({**marked, **changes})
        old = create_model({**self.config, "relation_attention": True})
        with self.assertRaises(ValueError):
            old(
                {
                    **batch_samples([synthetic_sample(database="stats")]),
                    "mapping_scope": "official_26_4",
                    "mapping_split": "test",
                }
            )

    def test_official_scope_rejects_nonofficial_capacity(self):
        with self.assertRaises(ValueError):
            create_model(
                {
                    **self.config,
                    "capacity": synthetic_capacity(),
                    "relation_attention": True,
                    "relation_scope": "official_26_4",
                }
            )

    def test_official_relation_live_output_and_checkpoint_roundtrip(self):
        batch = {
            **batch_samples([synthetic_sample(database="stats")]),
            "mapping_scope": "official_26_4",
            "mapping_split": "test",
        }
        torch.manual_seed(42)
        baseline = create_model(self.config)
        torch.manual_seed(42)
        model = create_model(
            {
                **self.config,
                "relation_attention": True,
                "relation_scope": "official_26_4",
            }
        )
        with torch.no_grad():
            baseline.head[-1].weight.fill_(0.1)
            model.head[-1].weight.copy_(baseline.head[-1].weight)
        self.assertTrue(torch.equal(model(batch), baseline(unmapped(batch))))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            metadata = {"manifest_hashes": self.binding["cache_hashes"]}
            save_checkpoint(path, model, None, 0, None, 0, metadata)
            restored, original = load_checkpoint(path)
            self.assertTrue(torch.equal(restored(batch), model(batch)))
            invalid = copy.deepcopy(original)
            invalid["model_config"]["relation_scope"] = "source"
            invalid["model_config_sha256"] = digest(invalid["model_config"])
            torch.save(invalid, path)
            with self.assertRaises(ValueError):
                load_checkpoint(path)

    def test_grouped_and_baseline_paths_still_use_unmapped_inputs(self):
        batch = batch_samples([grouped_sample((3, 2, 2, 3))])
        for mode in ("none", "mean", "attention"):
            model = create_model({**self.config, "group_pooling": mode})
            self.assertTrue(torch.equal(model(batch), batch["pg_log"]))
            with self.assertRaises(ValueError):
                model({**batch, "mapping": {}})


if __name__ == "__main__":
    unittest.main()
