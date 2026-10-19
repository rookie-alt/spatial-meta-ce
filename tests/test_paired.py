import csv
import json
import math
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

from spatial_meta_ce.evaluate import evaluate


class SyntheticStore:
    def __init__(self, rows):
        self.rows = rows
        self.cards = np.asarray([[row["true"], row["pg"]] for row in rows])
        self.manifest = {"content_hash": "synthetic-only"}

    def __len__(self):
        return len(self.rows)

    def unique_indices(self, exclude=()):
        seen = set(exclude)
        indices = []
        for i, row in enumerate(self.rows):
            if row["query_hash"] not in seen:
                indices.append(i)
                seen.add(row["query_hash"])
        return np.asarray(indices)

    def batch(self, indices, device):
        return {}  # No feature construction, training or inference in these tests.


class PairedProtocolTests(unittest.TestCase):
    def setUp(self):
        self.support = SyntheticStore(
            [
                {
                    "id": f"support-{i}",
                    "query_hash": f"support-{i}",
                    "true": 100.0,
                    "pg": 10.0,
                }
                for i in range(8)
            ]
        )
        self.test = SyntheticStore(
            [
                {"id": "row-a", "query_hash": "hash-a", "true": 100.0, "pg": 10.0},
                {
                    "id": "row-a-repeat",
                    "query_hash": "hash-a",
                    "true": 100.0,
                    "pg": 10.0,
                },
                {
                    "id": "duplicate-id",
                    "query_hash": "hash-b",
                    "true": 100.0,
                    "pg": 1000.0,
                },
                {
                    "id": "duplicate-id",
                    "query_hash": "hash-c",
                    "true": 10.0,
                    "pg": 10.0,
                },
            ]
        )
        self.config = {
            "shots": [0, 5],
            "support_runs": 2,
            "support_seed": 2026,
            "inner_lr": 0.01,
            "inner_steps": 1,
            "adapt_scope": "all",
            "loss": "smooth_l1",
            "batch_size": 8,
            "latency_repetitions": 1,
            "model": {"width": 8, "use_pg": True},
        }
        self.data_config = {
            "cache_root": "never-read",
            "splits": {
                "train": ["synthetic-train"],
                "validation": ["synthetic-validation"],
                "test": ["synthetic"],
            },
        }
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        pool = Mock()
        pool.get.side_effect = (
            lambda split, db: self.test if split == "test" else self.support
        )
        self.stack.enter_context(
            patch("spatial_meta_ce.evaluate.StorePool", return_value=pool)
        )
        self.stack.enter_context(patch("spatial_meta_ce.evaluate.seed_all"))
        model = Mock()
        model.parameters.return_value = []
        model.to.return_value = model
        self.stack.enter_context(
            patch("spatial_meta_ce.evaluate.SetEstimator", return_value=model)
        )
        self.adapt = self.stack.enter_context(patch("spatial_meta_ce.evaluate.adapt"))
        self.timing = self.stack.enter_context(
            patch("spatial_meta_ce.evaluate.latency", return_value={})
        )
        self.predict = self.stack.enter_context(
            patch(
                "spatial_meta_ce.evaluate._predict_rows",
                side_effect=self.synthetic_predictions,
            )
        )
        self.stack.enter_context(
            patch(
                "spatial_meta_ce.evaluate.forward",
                side_effect=AssertionError("Inference is forbidden"),
            )
        )

    @staticmethod
    def synthetic_predictions(
        model, params, store, indices, device, batch_size, baseline
    ):
        truth = store.cards[indices, 0]
        pred = (
            np.log(store.cards[indices, 1])
            if model is None
            else np.full(len(indices), math.log(100))
        )
        records = [
            (int(i), float(y), float(p)) for i, y, p in zip(indices, truth, pred)
        ]
        return pred, truth, records

    def write_reference(self, root):
        path = root / "pg.csv"
        with path.open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(
                [
                    "database",
                    "run",
                    "k",
                    "query_id",
                    "query_hash",
                    "true_cardinality",
                    "log_prediction",
                ]
            )
            for row in self.test.rows:
                writer.writerow(
                    [
                        "synthetic",
                        0,
                        0,
                        row["id"],
                        row["query_hash"],
                        row["true"],
                        math.log(row["pg"]),
                    ]
                )
        manifest = root / "reference.json"
        manifest.write_text(
            json.dumps({"pg_prediction_path": str(path), "metadata": {"method": "pg"}})
        )
        return manifest

    def test_evaluation_reports_unique_and_paired_metrics(self):
        with tempfile.TemporaryDirectory() as d:
            report = evaluate(
                dict(self.config, shots=[0]),
                self.data_config,
                Path(d) / "eval",
                "cpu",
                baseline="pg",
            )
            row = report["results"][0]
            self.assertIn("unique_query_metrics", row)
            self.assertIsNone(row["paired_vs_reference"])
            self.assertEqual(report["primary_metric"], "unique_query_q95")
            self.assertEqual(row["metrics"]["n"], 4)
            self.assertEqual(row["unique_query_metrics"]["n"], 3)
            self.assertAlmostEqual(row["unique_query_metrics"]["q95"], 10)
            self.adapt.assert_not_called()
            self.timing.assert_not_called()

    def test_generated_report_defines_current_operand_scale_roundoff_ties(self):
        with tempfile.TemporaryDirectory() as d:
            report = evaluate(
                dict(self.config, shots=[0]),
                self.data_config,
                Path(d) / "eval",
                "cpu",
                baseline="pg",
            )
        definitions = report["paired_metric_definitions"]
        self.assertEqual(
            definitions["fraction_ties"],
            "abs(log(Q_reference)-log(Q_model)) <= 4*float64.eps*"
            "max(abs(clamped_log_model), abs(clamped_log_reference), abs(log_true)); "
            "ties counted in neither fraction",
        )
        self.assertIn(
            "excluding machine-roundoff ties", definitions["fraction_queries_improved"]
        )
        self.assertIn(
            "excluding machine-roundoff ties", definitions["fraction_queries_worsened"]
        )

    def test_paired_manifest_is_shared_across_methods(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            evaluate(self.config, self.data_config, root / "a", "cpu", baseline="pg")
            evaluate(
                self.config,
                self.data_config,
                root / "b",
                "cpu",
                baseline="random",
                training_seed=8,
            )
            am = json.loads((root / "a" / "manifest.json").read_text())
            bm = json.loads((root / "b" / "manifest.json").read_text())
            self.assertEqual(am["episodes"], bm["episodes"])

    def test_all_rows_and_unique_pairings_bind_labels_and_pg_reference(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            reference = self.write_reference(root)
            report = evaluate(
                self.config,
                self.data_config,
                root / "eval",
                "cpu",
                baseline="random",
                reference_manifest=reference,
            )
            for row in report["results"]:
                self.assertEqual(row["reference_method"], "pg")
                if row["k"] == 0:
                    self.assertIsNone(row["paired_vs_reference"])
                    self.assertIsNone(row["paired_unique_vs_reference"])
                    continue
                paired = row["paired_vs_reference"]
                unique = row["paired_unique_vs_reference"]
                self.assertAlmostEqual(paired["mean_log_q_reduction"], math.log(10) / 2)
                self.assertEqual(
                    row["reference_comparison"], "pg_baseline_not_same_model_k0"
                )
                self.assertAlmostEqual(paired["median_log_q_reduction"], math.log(10))
                self.assertEqual(paired["fraction_queries_improved"], 0.75)
                self.assertEqual(paired["fraction_queries_worsened"], 0.25)
                self.assertAlmostEqual(paired["paired_delta_q95"], -1.35)
                self.assertAlmostEqual(unique["mean_log_q_reduction"], math.log(10) / 3)
                self.assertAlmostEqual(unique["median_log_q_reduction"], math.log(10))
                self.assertEqual(unique["fraction_queries_improved"], 2 / 3)
                self.assertEqual(unique["fraction_queries_worsened"], 1 / 3)
                self.assertAlmostEqual(unique["paired_delta_q95"], -0.9)
                self.assertAlmostEqual(row["unique_query_metrics"]["q95"], 9.1)
            aggregate = next(row for row in report["aggregate"] if row["k"] == 5)
            self.assertEqual(
                aggregate["across_support_runs"]["paired_unique"][
                    "fraction_queries_improved"
                ],
                2 / 3,
            )
            self.assertIn("paired_metric_definitions", report)

    def test_prediction_record_order_must_match_reference_for_both_scopes(self):
        for drift_length in (4, 3):
            with (
                self.subTest(drift_length=drift_length),
                tempfile.TemporaryDirectory() as d,
            ):
                root = Path(d)
                reference = self.write_reference(root)

                def changed_order(*args, drift_length=drift_length):
                    pred, truth, records = self.synthetic_predictions(*args)
                    return (
                        pred,
                        truth,
                        records[::-1] if len(records) == drift_length else records,
                    )

                self.predict.side_effect = changed_order
                with self.assertRaises(ValueError):
                    evaluate(
                        dict(self.config, shots=[5]),
                        self.data_config,
                        root / "eval",
                        "cpu",
                        baseline="random",
                        reference_manifest=reference,
                    )

    def test_prediction_labels_cannot_drift_after_reference_binding(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            reference = self.write_reference(root)

            def changed_labels(*args):
                pred, truth, records = self.synthetic_predictions(*args)
                return pred, truth + 1, records

            self.predict.side_effect = changed_labels
            with self.assertRaises(ValueError):
                evaluate(
                    dict(self.config, shots=[5]),
                    self.data_config,
                    root / "eval",
                    "cpu",
                    baseline="random",
                    reference_manifest=reference,
                )


if __name__ == "__main__":
    unittest.main()
