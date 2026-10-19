"""Frozen runner tests; stdlib contracts run without torch or numpy imports."""

import copy
import csv
import importlib.util
import json
import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import run_meta65_screen as screen

ROOT = Path(__file__).resolve().parents[1]


def endpoint_rows(a=100.0, b=90.0, c=95.0):
    return [
        {
            "arm": arm,
            "database": db,
            "k": k,
            "run": run,
            "all_q50": 2.0,
            "all_q95": value,
            "four_plus_q50": 2.0,
            "four_plus_q95": value,
            "all_log_q95": math.log(value),
            "four_plus_log_q95": math.log(value),
            "query_digest": f"{db}-{run}",
            "support_digest": f"{db}-{k}-{run}",
            "query_count": 500,
            "four_plus_count": 200,
        }
        for arm, value in zip(screen.ARMS, (a, b, c))
        for db in screen.VALIDATION
        for k in screen.SHOTS
        for run in range(5)
    ]


class FrozenConfigTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads((ROOT / "configs/meta65_screen.json").read_text())
        self.data = json.loads((ROOT / "configs/data.json").read_text())

    def test_actual_config_and_protocol_hash(self):
        self.assertEqual(
            screen.load_config(ROOT / "configs/meta65_screen.json"), self.config
        )
        self.assertEqual(
            screen.arm_config(self.config, screen.ARMS[2])["adapt_scope"], "last"
        )
        self.assertEqual(
            screen.arm_config(self.config, screen.ARMS[0])["loss"], "smooth_l1"
        )
        self.assertEqual(
            screen.arm_config(self.config, screen.ARMS[0])["model"]["width"], 64
        )

    def test_every_frozen_field_and_unknown_fields_are_rejected(self):
        def leaves(value, path=()):
            for key, item in value.items():
                if isinstance(item, dict):
                    yield from leaves(item, (*path, key))
                else:
                    yield (*path, key)

        for path in leaves(self.config):
            changed = copy.deepcopy(self.config)
            node = changed
            for key in path[:-1]:
                node = node[key]
            item = node[path[-1]]
            node[path[-1]] = not item if isinstance(item, bool) else None
            with self.subTest(path=path), self.assertRaises(ValueError):
                screen.validate_config(changed)
        for location in ((), ("training",), ("model",), ("arms", screen.ARMS[0])):
            changed = copy.deepcopy(self.config)
            node = changed
            for key in location:
                node = node[key]
            node["unknown"] = 1
            with self.assertRaises(ValueError):
                screen.validate_config(changed)
        changed = copy.deepcopy(self.config)
        changed["schema"] = True
        with self.assertRaises(ValueError):
            screen.validate_config(changed)

    def test_protocol_bytes_not_only_config_are_authenticated(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            file = root / self.config["protocol_file"]
            file.parent.mkdir()
            file.write_text("changed")
            config = root / "config.json"
            config.write_text(json.dumps(self.config))
            with self.assertRaisesRegex(ValueError, "[Pp]rotocol"):
                screen.load_config(config, root=root)

    def test_data_allows_only_roots_to_change(self):
        self.data["data_root"] = "/private/tmp/read-only-price"
        self.data["cache_root"] = "/private/tmp/cache"
        self.assertEqual(
            screen.validate_data(self.data)["cache_root"], "/private/tmp/cache"
        )
        for mutation in ("target", "reverse", "extra", "unknown"):
            changed = copy.deepcopy(self.data)
            if mutation == "target":
                changed["splits"]["train"][0] = "imdb"
            elif mutation == "reverse":
                changed["splits"]["validation"].reverse()
            elif mutation == "extra":
                changed["splits"]["test"].append("extra")
            else:
                changed["prepare"] = True
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                screen.validate_data(changed)

    def test_duplicate_json_keys_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "x.json"
            path.write_text('{"schema":1,"schema":1}')
            with self.assertRaises(ValueError):
                screen.read_json(path)

    def test_dry_run_needs_no_dependencies_cache_or_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "does-not-exist"
            result = subprocess.run(
                [
                    sys.executable,
                    "-S",
                    "-B",
                    str(ROOT / "scripts/run_meta65_screen.py"),
                    "--out",
                    str(output),
                ],
                capture_output=True,
                text=True,
                check=True,
            )
            matrix = json.loads(result.stdout)
            self.assertEqual(len(matrix["matrix"]), 3)
            self.assertFalse(output.exists())


class SafetyAndFailureTests(unittest.TestCase):
    def test_existing_output_even_empty_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            output = root / "runs/used"
            output.mkdir(parents=True)
            with self.assertRaises(FileExistsError):
                screen.validate_output(
                    output,
                    {
                        "cache_root": str(root / "cache"),
                        "data_root": str(root / "PRICE-master"),
                    },
                    root=root,
                )

    def test_symlink_and_protected_paths_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            (root / "runs").mkdir()
            (root / "alias").symlink_to(root / "runs", target_is_directory=True)
            data = {
                "cache_root": str(root / "cache"),
                "data_root": str(root / "PRICE-master"),
            }
            for output in (
                root / "alias/new",
                root / "cache/runs/new",
                root / "PRICE-master/runs/new",
                root / "runs/imdb/new",
                root / "runs/finetune/new",
            ):
                with self.subTest(output=output), self.assertRaises(ValueError):
                    screen.validate_output(output, data, root=root)

    def test_failure_is_recorded_once_and_retained(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            output = root / "runs/failure"
            data = screen.validate_data(
                json.loads((ROOT / "configs/data.json").read_text())
            )
            config = screen.load_config(ROOT / "configs/meta65_screen.json")
            with (
                patch.object(
                    screen,
                    "_execute_work",
                    side_effect=RuntimeError("intentional failure"),
                ) as work,
                self.assertRaisesRegex(RuntimeError, "intentional failure"),
            ):
                screen.execute(config, data, output, "cpu", root=root)
            self.assertEqual(work.call_count, 1)
            status = json.loads((output / "status.json").read_text())
            self.assertEqual(status["status"], "failed")
            self.assertIn("intentional failure", status["error"])
            self.assertEqual(status["arms"][screen.ARMS[0]], "planned")

    def test_total_deadline_not_per_arm(self):
        with (
            patch.object(screen.time, "monotonic", return_value=7210),
            self.assertRaises(TimeoutError),
        ):
            screen.check_deadline(10, 7200)


class SupportPlanTests(unittest.TestCase):
    def test_unique_nested_shared_query_and_reproducible(self):
        indices = list(range(25))
        hashes = [f"h{i}" for i in indices]
        plan = screen.support_plan(
            "chembl", indices, hashes, seed=2718, runs=5, shots=[2, 5, 10]
        )
        self.assertEqual(
            plan,
            screen.support_plan(
                "chembl", indices, hashes, seed=2718, runs=5, shots=[2, 5, 10]
            ),
        )
        self.assertNotEqual(plan[0]["support_max"], plan[1]["support_max"])
        for run in plan:
            self.assertEqual(len(run["query"]), 15)
            for k in (2, 5, 10):
                self.assertEqual(run["supports"][str(k)], run["support_max"][:k])
                self.assertFalse(set(run["supports"][str(k)]) & set(run["query"]))
            self.assertEqual(set(run["query"]) | set(run["support_max"]), set(indices))

    def test_target_and_duplicate_hash_and_small_pool_rejected(self):
        for db, indices, hashes in (
            ("imdb", list(range(101)), [str(i) for i in range(101)]),
            ("chembl", [0, 1], ["x", "x"]),
            ("chembl", [0, 1], ["x", "y"]),
        ):
            with self.assertRaises(ValueError):
                screen.support_plan(db, indices, hashes, seed=1, runs=5, shots=[5, 100])


class GateTests(unittest.TestCase):
    def test_exact_improvement_boundary_and_no_candidate(self):
        result = screen.decide(endpoint_rows())
        self.assertEqual(result["status"], "promising_single_seed")
        self.assertEqual(result["selected_arm"], screen.ARMS[1])
        self.assertEqual(
            screen.decide(endpoint_rows(b=90.00001))["status"], "no_candidate"
        )

    def test_regression_gate_is_per_database_k_and_both_metrics(self):
        for metric in ("all_q95", "four_plus_q95"):
            rows = endpoint_rows(b=80)
            for row in rows:
                if (
                    row["arm"] == screen.ARMS[1]
                    and row["database"] == "chembl"
                    and row["k"] == 5
                ):
                    row[metric] = 110
            self.assertEqual(screen.decide(rows)["selected_arm"], screen.ARMS[1])
            for row in rows:
                if (
                    row["arm"] == screen.ARMS[1]
                    and row["database"] == "chembl"
                    and row["k"] == 5
                ):
                    row[metric] = 110.00001
            self.assertEqual(screen.decide(rows)["status"], "no_candidate")

    def test_arithmetic_macro_not_geometric_pooled_or_log_quantile(self):
        rows = endpoint_rows(a=100, b=100, c=100)
        selected = [r for r in rows if r["arm"] == screen.ARMS[1]]
        selected[0]["four_plus_q95"] = 1
        for row in rows:
            if row["database"] == "chembl" and row["k"] == 5 and row["run"] == 0:
                row["query_count"] = 50000
        result = screen.decide(rows)
        self.assertAlmostEqual(
            result["arms"][screen.ARMS[1]]["macro_four_plus_q95"], (7400 + 1) / 75
        )
        self.assertNotAlmostEqual(
            result["arms"][screen.ARMS[1]]["macro_four_plus_q95"],
            math.exp(74 * math.log(100) / 75),
        )

    def test_tie_break_prefers_last_layer(self):
        self.assertEqual(
            screen.decide(endpoint_rows(b=80, c=80))["selected_arm"], screen.ARMS[2]
        )

    def test_incomplete_nonfinite_duplicate_and_unpaired_rows_fail(self):
        variants = [endpoint_rows()[:-1], endpoint_rows() + [endpoint_rows()[0]]]
        bad = endpoint_rows()
        bad[0]["four_plus_q95"] = float("inf")
        variants.append(bad)
        bad = endpoint_rows()
        bad[75]["query_digest"] = "changed"
        variants.append(bad)
        for rows in variants:
            with self.assertRaises(ValueError):
                screen.decide(rows)


class PairingTests(unittest.TestCase):
    def test_input_pairing_not_backward_cost_equality(self):
        records = [
            {
                "arm": arm,
                "initialization_digest": "init",
                "parameters": 132545,
                "sample_plan_digest": "sample",
                "sample_plan_episodes": 16000,
                "support_sample_count": 590000,
                "validation_plan_digest": "validation",
                "endpoint_plan_digest": "endpoint",
                "optimizer_steps": 4000,
                "metadata": {"manifest_hashes": {"accidents": "hash"}},
                "counts": {"backward_calls": 96000 if index == 0 else 64000},
            }
            for index, arm in enumerate(screen.ARMS)
        ]
        screen.verify_pairing(records)
        for field in (
            "initialization_digest",
            "sample_plan_digest",
            "validation_plan_digest",
            "endpoint_plan_digest",
            "support_sample_count",
            "sample_plan_episodes",
        ):
            changed = copy.deepcopy(records)
            changed[2][field] = "changed"
            with self.subTest(field=field), self.assertRaises(ValueError):
                screen.verify_pairing(changed)


@unittest.skipUnless(importlib.util.find_spec("torch"), "torch unavailable locally")
class InstanceSmokeTests(unittest.TestCase):
    def test_postcheck_and_final_deadline_failure_never_publish_decision(self):
        import numpy as np

        from scripts import precision_screen_evidence as evidence

        class Store:
            def __init__(self):
                self.rows = [{"id": str(i), "query_hash": f"h{i}"} for i in range(101)]

            def unique_indices(self):
                return np.arange(101)

        config = copy.deepcopy(screen.FROZEN)
        sources = {
            db: {"manifest_hash": "sourcehash", "rows": 101}
            for db in screen.TRAIN + screen.VALIDATION
        }

        def fake_train(_config, _data, folder, _device, _deadline):
            folder.mkdir()
            return None, {
                "initialization_digest": "init",
                "parameters": 132545,
                "sample_plan_digest": "sample",
                "sample_plan_episodes": 16000,
                "support_sample_count": 100000,
                "validation_plan_digest": "validation",
                "optimizer_steps": 4000,
                "metadata": {"manifest_hashes": {db: "sourcehash" for db in sources}},
            }

        def fake_evaluate(
            _model, _stores, _counts, _plans, _config, arm, folder, _device, _deadline
        ):
            np.savez(
                folder / "retained_raw.npz",
                log_pred=np.array([1.0]),
                true=np.array([2.0]),
            )
            return [
                {**row, "prediction_file": "retained_raw.npz"}
                for row in endpoint_rows()
                if row["arm"] == arm
            ]

        for mode in ("source", "deadline", "success"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp).resolve()
                out = root / "runs/screen"
                data = {
                    "data_root": str(root / "PRICE-master"),
                    "cache_root": str(root / "cache"),
                    "splits": {
                        "train": list(screen.TRAIN),
                        "validation": list(screen.VALIDATION),
                        "test": list(screen.TARGETS),
                    },
                }
                after = copy.deepcopy(sources)
                if mode == "source":
                    after["chembl"]["manifest_hash"] = "changed"

                def deadline(_start, _timeout, mode=mode, out=out):
                    if mode == "deadline" and (out / "source_after.json").exists():
                        raise TimeoutError("final deadline")

                with (
                    patch.object(screen, "environment", return_value={"device": "cpu"}),
                    patch.object(
                        screen, "file_hash", return_value=config["protocol_sha256"]
                    ),
                    patch.object(
                        evidence, "historical_sources", return_value={"frozen": True}
                    ),
                    patch.object(
                        evidence, "source_evidence", side_effect=[sources, after]
                    ),
                    patch("spatial_meta_ce.data.FeatureStore", return_value=Store()),
                    patch.object(
                        screen, "validate_store", return_value=np.full(101, 4)
                    ),
                    patch.object(screen, "train_arm", side_effect=fake_train),
                    patch.object(screen, "evaluate_arm", side_effect=fake_evaluate),
                    patch.object(screen, "check_deadline", side_effect=deadline),
                ):
                    if mode == "success":
                        screen.execute(config, data, out, "cpu", root=root)
                        self.assertTrue((out / "decision.json").exists())
                    else:
                        with self.assertRaises(
                            ValueError if mode == "source" else TimeoutError
                        ):
                            screen.execute(config, data, out, "cpu", root=root)
                        self.assertFalse((out / "decision.json").exists())
                        self.assertEqual(
                            json.loads((out / "status.json").read_text())["status"],
                            "failed",
                        )
                self.assertTrue((out / "endpoint.csv").exists())
                self.assertTrue((out / screen.ARMS[0] / "retained_raw.npz").exists())

    def test_source_rehashed_after_training_failure_without_retry(self):
        import numpy as np

        from scripts import precision_screen_evidence as evidence

        class Store:
            def __init__(self):
                self.rows = [{"id": str(i), "query_hash": f"h{i}"} for i in range(101)]

            def unique_indices(self):
                return np.arange(101)

        config = screen.load_config(ROOT / "configs/meta65_screen.json")
        sources = {
            db: {"manifest_hash": "sourcehash", "rows": 101}
            for db in screen.TRAIN + screen.VALIDATION
        }
        for changed in (False, True):
            after = copy.deepcopy(sources)
            if changed:
                after["chembl"]["manifest_hash"] = "modified"
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp).resolve()
                data = {
                    "data_root": str(root / "PRICE-master"),
                    "cache_root": str(root / "cache"),
                    "splits": {
                        "train": list(screen.TRAIN),
                        "validation": list(screen.VALIDATION),
                        "test": list(screen.TARGETS),
                    },
                }
                out = root / "runs/failure"
                with (
                    patch.object(screen, "environment", return_value={"device": "cpu"}),
                    patch.object(
                        screen, "file_hash", return_value=config["protocol_sha256"]
                    ),
                    patch.object(
                        evidence, "historical_sources", return_value={"frozen": True}
                    ),
                    patch.object(
                        evidence, "source_evidence", side_effect=[sources, after]
                    ) as rehash,
                    patch("spatial_meta_ce.data.FeatureStore", return_value=Store()),
                    patch.object(
                        screen, "validate_store", return_value=np.full(101, 4)
                    ),
                    patch.object(
                        screen, "train_arm", side_effect=RuntimeError("training failed")
                    ) as train,
                ):
                    expected = ValueError if changed else RuntimeError
                    with self.assertRaises(expected):
                        screen.execute(config, data, out, "cpu", root=root)
                    self.assertEqual(train.call_count, 1)
                    self.assertEqual(rehash.call_count, 2)
                status = json.loads((out / "status.json").read_text())
                self.assertEqual(status["status"], "failed")
                self.assertEqual(status["arms"][screen.ARMS[0]], "failed")
                self.assertTrue((out / "source_after.json").exists())
                self.assertTrue((out / "environment.json").exists())

    def test_real_feature_store_duplicate_conflicts_rejected(self):
        import numpy as np

        from spatial_meta_ce.data import FeatureStore
        from spatial_meta_ce.manifests import digest

        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp).resolve() / "pretrain/accidents"
            folder.mkdir(parents=True)
            manifest = {
                "schema": 1,
                "transform": "price40-group-logfanout-v1",
                "unique_queries": 3,
                "provenance": {},
                "rows": [
                    {"id": f"accidents/pretrain/{i}", "query_hash": h}
                    for i, h in enumerate(("a", "b", "a", "c"))
                ],
            }
            manifest["content_hash"] = digest(manifest)
            (folder / "manifest.json").write_text(json.dumps(manifest))
            cards = np.full((4, 2), 10.0, dtype=np.float64)
            offsets = np.arange(5, dtype=np.int64)[:, None] * np.array([[1, 1, 4, 1]])
            np.save(folder / "cards.npy", cards)
            np.save(folder / "offsets.npy", offsets)
            for group, dim in enumerate((40, 40, 4, 43)):
                np.save(
                    folder / f"group{group}.npy",
                    np.zeros((int(offsets[-1, group]), dim), np.float32),
                )
            self.assertEqual(
                screen.validate_store(FeatureStore(folder), "accidents").tolist(),
                [4] * 4,
            )
            for column in (0, 1):
                bad = cards.copy()
                bad[2, column] = 11
                np.save(folder / "cards.npy", bad)
                with (
                    self.subTest(column=column),
                    self.assertRaisesRegex(ValueError, "conflicting"),
                ):
                    screen.validate_store(FeatureStore(folder), "accidents")
            np.save(folder / "cards.npy", cards)
            features = np.zeros((4, 43), np.float32)
            features[2, 0] = 1
            np.save(folder / "group3.npy", features)
            with self.assertRaisesRegex(ValueError, "conflicting token"):
                screen.validate_store(FeatureStore(folder), "accidents")

    def test_endpoint_uses_fresh_weights_and_standard_metrics(self):
        import numpy as np
        import torch

        from spatial_meta_ce.data import batch_samples
        from spatial_meta_ce.metrics import summarize
        from spatial_meta_ce.model import SetEstimator

        class Store:
            def __init__(self):
                self.rows = [{"id": str(i), "query_hash": f"h{i}"} for i in range(12)]

            def batch(self, indices, device):
                return batch_samples(
                    [
                        {
                            "groups": [
                                torch.zeros(1, 40),
                                torch.zeros(1, 40),
                                torch.zeros(4, 4),
                                torch.zeros(1, 43),
                            ],
                            "true": 10 + i,
                            "pg": 8 + i,
                            "id": str(i),
                            "hash": f"h{i}",
                        }
                        for i in indices
                    ],
                    device,
                )

        model = SetEstimator(width=64, arch="sum", use_pg=True, layers=2, heads=4)
        initial = {n: p.detach().clone() for n, p in model.named_parameters()}
        store = Store()
        plan = screen.support_plan(
            "chembl",
            list(range(12)),
            [f"h{i}" for i in range(12)],
            seed=2718,
            runs=2,
            shots=[2, 3],
        )
        config = screen.load_config(ROOT / "configs/meta65_screen.json")
        config["evaluation"].update(shots=[2, 3], batch_size=4)
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "runs/smoke"
            out.mkdir(parents=True)
            rows = screen.evaluate_arm(
                model,
                {"chembl": store},
                {"chembl": np.full(12, 4)},
                {"chembl": plan},
                config,
                screen.ARMS[1],
                out,
                "cpu",
                lambda: None,
            )
            self.assertEqual(len(rows), 4)
            for row in rows:
                raw = np.load(out / row["prediction_file"], allow_pickle=False)
                metrics = summarize(raw["log_pred"], raw["true"])
                self.assertAlmostEqual(row["all_q95"], metrics["q95"])
                self.assertAlmostEqual(row["four_plus_q95"], metrics["q95"])
            for name, param in model.named_parameters():
                torch.testing.assert_close(param, initial[name], rtol=0, atol=0)
            screen.write_endpoint(out / "endpoint.csv", rows)
            with (out / "endpoint.csv").open() as stream:
                self.assertEqual(len(list(csv.DictReader(stream))), 4)


if __name__ == "__main__":
    unittest.main()
