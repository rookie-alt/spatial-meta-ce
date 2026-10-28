import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import torch

from scripts.build_fomaml_joint_plans import build_plans
from scripts.run_fomaml_joint_source import build_runs
from spatial_meta_ce.features import write_cache
from spatial_meta_ce.plans import EVAL_DATABASES
from spatial_meta_ce.source_eval import ARMS
from spatial_meta_ce.train import FIXED_MODEL


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
            for i in range(200)
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


def data_config(cache, root, train_databases=("a",)):
    return {
        "cache_root": str(cache),
        "data_root": str(root / "unused"),
        "splits": {
            "train": list(train_databases),
            "validation": list(EVAL_DATABASES),
            "test": ["t"],
        },
    }


class PlanBuilderTests(unittest.TestCase):
    def test_build_plans_writes_three_training_plans_and_binding(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache = make_cache(root)
            config = data_config(cache, root)
            binding = build_plans(
                config,
                seeds=(42, 43, 44),
                out_dir=root / "plans",
                steps=2,
                expected_unique=None,
            )
            self.assertEqual(binding["seeds"], [42, 43, 44])
            self.assertEqual(binding["target_databases_read"], [])
            for seed in (42, 43, 44):
                path = root / "plans" / f"training_plan_seed{seed}.json"
                self.assertTrue(path.is_file())
                plan = json.loads(path.read_text())
                self.assertEqual(plan["seed"], seed)
                self.assertEqual(plan["steps"], 2)
                self.assertEqual(len(plan["episodes"]), 8)
                self.assertEqual(
                    binding["training_plans"][str(seed)]["plan_sha256"],
                    plan["plan_sha256"],
                )
            eval_plan = json.loads(
                (root / "plans" / "source_eval_plan.json").read_text()
            )
            self.assertEqual(tuple(eval_plan["databases"]), EVAL_DATABASES)
            self.assertTrue((root / "plans" / "plan_binding.json").is_file())

    def test_target_database_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache = make_cache(root)
            config = data_config(cache, root, train_databases=("imdb",))
            with self.assertRaisesRegex(ValueError, "Target databases"):
                build_plans(
                    config,
                    seeds=(42,),
                    out_dir=root / "plans",
                    steps=2,
                    expected_unique=None,
                )

    def test_seed_outside_the_preregistered_set_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache = make_cache(root)
            config = data_config(cache, root)
            with self.assertRaisesRegex(ValueError, "subset"):
                build_plans(
                    config,
                    seeds=(7,),
                    out_dir=root / "plans",
                    steps=2,
                    expected_unique=None,
                )

    def test_validation_split_must_match_the_three_source_databases(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache = make_cache(root)
            config = data_config(cache, root)
            config["splits"]["validation"] = ["chembl"]
            with self.assertRaisesRegex(ValueError, "Validation split"):
                build_plans(
                    config,
                    seeds=(42,),
                    out_dir=root / "plans",
                    steps=2,
                    expected_unique=None,
                )

    def test_build_runs_covers_nine_frozen_cells(self):
        base = json.loads(
            (Path(__file__).resolve().parents[1] / "configs" / "fomaml_joint_sum_4k.json").read_text()
        )
        runs = build_runs(base, "plans", "out", (42, 43, 44), list(ARMS))
        self.assertEqual(len(runs), 9)
        self.assertEqual(
            {(run["arm"], run["seed"]) for run in runs},
            {(arm, seed) for arm in ARMS for seed in (42, 43, 44)},
        )
        for run in runs:
            config = run["config"]
            self.assertTrue(config["fixed_budget"])
            self.assertEqual(config["model"], FIXED_MODEL)
            self.assertIn(config["seed"], (42, 43, 44))
            self.assertTrue(config["training_plan"].endswith(
                f"training_plan_seed{run['seed']}.json"
            ))
        c_joint = next(run for run in runs if run["arm"] == "C_joint")
        self.assertEqual(c_joint["config"]["method"], "fomaml")
        self.assertEqual(c_joint["config"]["meta_objective"], "joint")
        a_erm = next(run for run in runs if run["arm"] == "A_erm")
        self.assertEqual(a_erm["config"]["method"], "erm")
        self.assertEqual(a_erm["config"]["meta_objective"], "post")

    def test_build_runs_rejects_unknown_arm_or_seed(self):
        base = json.loads(
            (Path(__file__).resolve().parents[1] / "configs" / "fomaml_joint_sum_4k.json").read_text()
        )
        with self.assertRaisesRegex(ValueError, "Arms"):
            build_runs(base, "plans", "out", (42,), ["D_unknown"])
        with self.assertRaisesRegex(ValueError, "Seeds"):
            build_runs(base, "plans", "out", (7,), ["C_joint"])


if __name__ == "__main__":
    unittest.main()
