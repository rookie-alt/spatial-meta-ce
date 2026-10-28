import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import torch

from spatial_meta_ce.checkpoints import load_checkpoint, save_checkpoint
from spatial_meta_ce.features import write_cache
from spatial_meta_ce.model import SetEstimator
from spatial_meta_ce.plans import EVAL_DATABASES, build_source_eval_plan
from spatial_meta_ce.source_eval import (
    PREDICTION_COLUMNS,
    evaluate_source,
    validate_eval_checkpoint,
)
from spatial_meta_ce.train import FIXED_MODEL, FIXED_SELECTION, FIXED_STEPS


def make_cache(root):
    cache = root / "cache"
    for split, db in [
        ("pretrain", "a"),
        *[("pretrain", db) for db in EVAL_DATABASES],
        ("test", "t"),
    ]:
        rows = [
            {
                "id": f"{db}/{split}/{i}",
                "query_hash": hashlib.sha256(f"{db}-{split}-{i}".encode()).hexdigest(),
                "source_line": i + 1,
                "true": float(20 + i),
                "pg": 2.0,
            }
            for i in range(130)
        ]
        data = {
            "data_features": [
                (torch.ones(40), torch.ones(40), torch.ones(4), None) for _ in rows
            ],
            "true_cards": [torch.tensor(row["true"]) for row in rows],
            "pg_est_cards": [torch.tensor(2.0) for _ in rows],
            "n_join_cols": [1] * len(rows),
            "n_fanouts": [1] * len(rows),
            "n_tables": [1] * len(rows),
            "n_filter_cols": [0] * len(rows),
        }
        write_cache(cache / split / db, rows, data, {"source": "unit-fixture"})
    return cache


def data_config(cache, root):
    return {
        "cache_root": str(cache),
        "data_root": str(root / "unused"),
        "splits": {
            "train": ["a"],
            "validation": list(EVAL_DATABASES),
            "test": ["t"],
        },
    }


def metadata(**overrides):
    value = {
        "method": "fomaml",
        "seed": 42,
        "train_databases": ["a"],
        "validation_databases": [],
        "manifest_hashes": {"a": "0" * 64},
        "training_signature": {
            "inner_lr": 0.001,
            "inner_steps": 5,
            "adapt_scope": "all",
            "loss": "smooth_l1",
        },
        "purpose": "research",
        "selection": FIXED_SELECTION,
        "meta_objective": "joint",
        "joint_pre_weight": 0.5,
        "fixed_budget": True,
        "fixed_steps": FIXED_STEPS,
        "selection_databases": [],
        "validation_monitor_reads": [],
        "training_plan": "plan.json",
        "training_plan_sha256": "a" * 64,
    }
    value.update(overrides)
    return value


def write_checkpoint(path, meta):
    model = SetEstimator(**FIXED_MODEL)
    save_checkpoint(
        path,
        model,
        None,
        FIXED_STEPS,
        None,
        0,
        meta,
    )


def read_rows(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


class SourceEvalTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_evaluate_source_writes_full_population_for_all_ks(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache = make_cache(root)
            config = data_config(cache, root)
            plan = build_source_eval_plan(config, expected_unique=None)
            plan_path = root / "plan.json"
            plan_path.write_text(json.dumps(plan))
            checkpoint = root / "last.pt"
            write_checkpoint(checkpoint, metadata())
            protocol = evaluate_source(
                checkpoint,
                "C_joint",
                42,
                plan_path,
                config,
                root / "eval",
                batch_size=7,
            )
            self.assertEqual(len(protocol["units"]), 33)
            self.assertEqual(protocol["target_databases_read"], [])
            rows = read_rows(root / "eval" / "predictions.csv")
            self.assertEqual(len(rows), 990)
            self.assertEqual(tuple(rows[0].keys()), PREDICTION_COLUMNS)
            for db in EVAL_DATABASES:
                k0 = [row for row in rows if row["database"] == db and row["k"] == "0"]
                self.assertEqual(len(k0), 30)
                self.assertEqual({row["run"] for row in k0}, {"0"})
                for k in ("5", "20"):
                    values = [
                        row
                        for row in rows
                        if row["database"] == db and row["k"] == k
                    ]
                    self.assertEqual(len(values), 150)
                    self.assertEqual(
                        {row["run"] for row in values},
                        {"1", "2", "3", "4", "5"},
                    )

    def test_batch_size_does_not_change_predictions(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache = make_cache(root)
            config = data_config(cache, root)
            plan = build_source_eval_plan(config, expected_unique=None)
            plan_path = root / "plan.json"
            plan_path.write_text(json.dumps(plan))
            checkpoint = root / "last.pt"
            write_checkpoint(checkpoint, metadata())
            evaluate_source(
                checkpoint,
                "C_joint",
                42,
                plan_path,
                config,
                root / "small",
                batch_size=3,
            )
            evaluate_source(
                checkpoint,
                "C_joint",
                42,
                plan_path,
                config,
                root / "large",
                batch_size=64,
            )
            small = [
                float(row["log_prediction"])
                for row in read_rows(root / "small" / "predictions.csv")
            ]
            large = [
                float(row["log_prediction"])
                for row in read_rows(root / "large" / "predictions.csv")
            ]
            self.assertEqual(len(small), len(large))
            self.assertTrue(torch.allclose(torch.tensor(small), torch.tensor(large)))

    def test_checkpoint_rejections(self):
        state = {
            "model_config": FIXED_MODEL,
            "next_step": FIXED_STEPS,
            "metadata": metadata(),
        }
        self.assertTrue(validate_eval_checkpoint(state, "C_joint", 42))
        with self.assertRaisesRegex(ValueError, "target database"):
            validate_eval_checkpoint(
                {**state, "metadata": metadata(train_databases=["imdb"])},
                "C_joint",
                42,
            )
        with self.assertRaisesRegex(ValueError, "selection"):
            validate_eval_checkpoint(
                {**state, "metadata": metadata(selection="best.pt")},
                "C_joint",
                42,
            )
        with self.assertRaisesRegex(ValueError, "objective"):
            validate_eval_checkpoint(state, "B_fomaml", 42)
        with self.assertRaisesRegex(ValueError, "4000"):
            validate_eval_checkpoint({**state, "next_step": 3999}, "C_joint", 42)
        with self.assertRaisesRegex(ValueError, "evaluation database"):
            validate_eval_checkpoint(
                {**state, "metadata": metadata(train_databases=["chembl"])},
                "C_joint",
                42,
            )
        with self.assertRaisesRegex(ValueError, "Unknown arm"):
            validate_eval_checkpoint(state, "D_unknown", 42)
        with self.assertRaisesRegex(ValueError, "training-plan hash"):
            validate_eval_checkpoint(
                {**state, "metadata": metadata(training_plan_sha256="short")},
                "C_joint",
                42,
            )
        with self.assertRaisesRegex(ValueError, "validation_monitor_reads"):
            validate_eval_checkpoint(
                {
                    **state,
                    "metadata": metadata(validation_monitor_reads=["chembl"]),
                },
                "C_joint",
                42,
            )
        with self.assertRaisesRegex(ValueError, "seed"):
            validate_eval_checkpoint(state, "C_joint", 43)
        with self.assertRaisesRegex(ValueError, "purpose"):
            validate_eval_checkpoint(
                {**state, "metadata": metadata(purpose="smoke-not-research")},
                "C_joint",
                42,
            )
        with self.assertRaisesRegex(ValueError, "fixed_steps"):
            validate_eval_checkpoint(
                {**state, "metadata": metadata(fixed_steps=3999)}, "C_joint", 42
            )
        with self.assertRaisesRegex(ValueError, "pre weight"):
            validate_eval_checkpoint(
                {**state, "metadata": metadata(joint_pre_weight=0.4)},
                "C_joint",
                42,
            )
        with self.assertRaisesRegex(ValueError, "model configuration"):
            validate_eval_checkpoint(
                {**state, "model_config": {"width": 32, "use_pg": True}},
                "C_joint",
                42,
            )

    def test_checkpoint_file_is_versioned_and_reloadable(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "last.pt"
            write_checkpoint(checkpoint, metadata())
            model, state = load_checkpoint(checkpoint)
            self.assertEqual(state["next_step"], FIXED_STEPS)
            self.assertEqual(state["metadata"]["meta_objective"], "joint")


if __name__ == "__main__":
    unittest.main()
