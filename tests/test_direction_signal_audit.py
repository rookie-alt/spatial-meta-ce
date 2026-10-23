import importlib.util
import json
import sys
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

_SOURCE = Path(__file__).resolve().parents[1] / "scripts/direction_signal_audit.py"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
_SPEC = importlib.util.spec_from_file_location("direction_signal_audit", _SOURCE)
audit = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(audit)

try:
    import sklearn  # noqa: F401

    HAVE_SKLEARN = True
except ImportError:  # pragma: no cover - server base env lacks sklearn
    HAVE_SKLEARN = False


def cache_fixture(root, rows, group_values):
    """Write a minimal valid cache; group_values maps hash -> four group blocks."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    offsets = np.zeros((len(rows) + 1, 4), dtype=np.int64)
    for index, row in enumerate(rows):
        for group in range(4):
            offsets[index + 1, group] = offsets[index, group] + len(
                group_values[row["query_hash"]][group]
            )
    for group, dim in enumerate(audit.d.GROUP_DIMS):
        blocks = [
            np.asarray(group_values[row["query_hash"]][group], dtype=np.float32)
            for row in rows
            if len(group_values[row["query_hash"]][group])
        ]
        matrix = np.concatenate(blocks) if blocks else np.zeros((0, dim), np.float32)
        np.save(root / f"group{group}.npy", matrix)
    np.save(root / "offsets.npy", offsets)
    np.save(
        root / "cards.npy",
        np.asarray([[row["true"], row["pg"]] for row in rows], dtype=np.float64),
    )
    manifest = {
        "schema": 1,
        "transform": audit.d.CACHE_TRANSFORM,
        "provenance": {"source": {"database": "synthetic"}},
        "rows": [
            {
                "id": row["id"],
                "query_hash": row["query_hash"],
                "exact_hash": row["query_hash"],
                "source_line": index + 1,
            }
            for index, row in enumerate(rows)
        ],
        "unique_queries": len({row["query_hash"] for row in rows}),
    }
    manifest["content_hash"] = audit.d.manifest_digest(manifest)
    (root / "manifest.json").write_text(json.dumps(manifest))
    return root


def make_cache(root):
    rows = [
        {"id": "db/pretrain/1", "query_hash": "h1", "true": 100.0, "pg": 10.0},
        {"id": "db/pretrain/2", "query_hash": "h2", "true": 10.0, "pg": 100.0},
    ]
    return cache_fixture(
        root,
        rows,
        {
            "h1": (
                [[0.5] * 40],
                [[1.0] * 40, [2.0] * 40],
                [[math.log(100), 1.0, 1.0, 1.0]],
                [[0.0] * 40 + [0.2, 0.4, 0.5]],
            ),
            "h2": (
                [[0.25] * 40],
                [[1.0] * 40, [1.0] * 40],
                [[math.log(1000), 0.1, 0.1, 0.1]],
                [[0.0] * 40 + [0.0, 0.5, 0.2]],
            ),
        },
    )


class LabelTests(unittest.TestCase):
    def test_direction_is_sign_of_log_residual(self):
        self.assertEqual(audit.direction_label(100.0, math.log(10.0)), 1)
        self.assertEqual(audit.direction_label(10.0, math.log(100.0)), 0)

    def test_exact_tie_is_excluded_not_guessed(self):
        self.assertIsNone(audit.direction_label(50.0, math.log(50.0)))
        self.assertIsNone(audit.direction_label(50.0, math.log(50.0) + 1e-12))

    def test_invalid_inputs_rejected(self):
        for true, pred in ((-1.0, 1.0), (float("nan"), 1.0), (10.0, float("inf"))):
            with self.subTest(true=true, pred=pred), self.assertRaises(ValueError):
                audit.direction_label(true, pred)

    def test_log_error_matches_definition(self):
        self.assertAlmostEqual(audit.log_error(3.0, 20.0), abs(3.0 - math.log(20.0)))


class BucketTests(unittest.TestCase):
    def test_table_bucket_boundaries(self):
        self.assertEqual(audit.table_bucket(1), "1")
        self.assertEqual(audit.table_bucket(2), "2-3")
        self.assertEqual(audit.table_bucket(3), "2-3")
        self.assertEqual(audit.table_bucket(4), "4-5")
        self.assertEqual(audit.table_bucket(6), "6+")


class AucTests(unittest.TestCase):
    def test_auc_matches_brute_force_definition(self):
        labels = [1, 0, 1, 1, 0, 0, 1, 0]
        scores = [0.9, 0.1, 0.8, 0.4, 0.6, 0.2, 0.7, 0.3]
        positives = [s for s, y in zip(scores, labels) if y]
        negatives = [s for s, y in zip(scores, labels) if not y]
        pairs = [
            1.0 if p > n else 0.5 if p == n else 0.0
            for p in positives
            for n in negatives
        ]
        self.assertAlmostEqual(audit.auc_score(labels, scores), sum(pairs) / len(pairs))

    def test_auc_averages_tied_scores(self):
        # One positive and one negative with the same score -> 0.5, never 1.0.
        self.assertAlmostEqual(audit.auc_score([1, 0], [0.4, 0.4]), 0.5)

    def test_auc_rejects_single_class(self):
        with self.assertRaises(ValueError):
            audit.auc_score([1, 1, 1], [0.1, 0.2, 0.3])

    def test_auc_rejects_nonfinite_scores(self):
        with self.assertRaises(ValueError):
            audit.auc_score([1, 0], [float("nan"), 0.5])

    def test_bootstrap_ci_brackets_point_estimate(self):
        rng = np.random.default_rng(0)
        scores = rng.random(400)
        labels = (scores + rng.normal(0, 0.3, 400) > 0.5).astype(int)
        point = audit.auc_score(labels, scores)
        low, high = audit.bootstrap_auc_ci(labels, scores, reps=200)
        self.assertLessEqual(low, point)
        self.assertGreaterEqual(high, point)
        self.assertLess(low, high)


class CacheTests(unittest.TestCase):
    def test_manifest_hash_is_verified(self):
        with tempfile.TemporaryDirectory() as directory:
            root = make_cache(Path(directory) / "db")
            cache = audit.load_cache(root)
            self.assertEqual(len(cache["rows"]), 2)
            manifest = json.loads((root / "manifest.json").read_text())
            manifest["unique_queries"] = 99
            (root / "manifest.json").write_text(json.dumps(manifest))
            with self.assertRaises(ValueError):
                audit.load_cache(root)

    def test_unsupported_transform_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            root = make_cache(Path(directory) / "db")
            manifest = json.loads((root / "manifest.json").read_text())
            manifest["transform"] = "other"
            manifest["content_hash"] = audit.d.manifest_digest(
                {k: v for k, v in manifest.items() if k != "content_hash"}
            )
            (root / "manifest.json").write_text(json.dumps(manifest))
            with self.assertRaises(ValueError):
                audit.load_cache(root)

    def test_feature_row_has_stable_named_shape(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = audit.load_cache(make_cache(Path(directory) / "db"))
            names, values = audit.feature_row(cache, 0)
            self.assertEqual(len(names), len(values))
            self.assertIn("n_tables", names)
            self.assertIn("pg_log", names)
            self.assertIn("table_avi_mean", names)
            self.assertIn("filter_sel_mean", names)
            self.assertEqual(names.count("join_hist_bin0"), 1)
            self.assertTrue(all(math.isfinite(value) for value in values))

    def test_feature_groups_partition_all_names(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = audit.load_cache(make_cache(Path(directory) / "db"))
            names, _ = audit.feature_row(cache, 0)
            for name in names:
                self.assertIn(audit.feature_group(name), audit.FEATURE_GROUPS)

    def test_structural_signature_groups_identical_templates(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = audit.load_cache(make_cache(Path(directory) / "db"))
            self.assertEqual(
                audit.structural_signature(cache, 0),
                audit.structural_signature(cache, 0),
            )
            self.assertNotEqual(
                audit.structural_signature(cache, 0),
                audit.structural_signature(cache, 1),
            )

    def test_unique_rows_keeps_first_occurrence(self):
        cache = {
            "rows": [{"query_hash": "a"}, {"query_hash": "b"}, {"query_hash": "a"}]
        }
        self.assertEqual(audit.unique_rows(cache).tolist(), [0, 1])


class PredictionTests(unittest.TestCase):
    def write_csv(self, path, rows):
        header = "database,run,k,query_id,query_hash,true_cardinality,log_prediction\n"
        lines = [
            f"{db},{run},{k},id/{index},{h},{t},{p}\n"
            for index, (db, run, k, h, t, p) in enumerate(rows)
        ]
        Path(path).write_text(header + "".join(lines))

    def test_path_traversal_names_refused(self):
        for name in ("..", ".", "../chembl", "chembl/../..", ""):
            with self.subTest(name=name), self.assertRaises(ValueError):
                audit.assert_source_database(name)

    def test_target_database_is_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "p.csv"
            self.write_csv(path, [("genome", 0, 0, "h", 10.0, 1.0)])
            with self.assertRaises(ValueError):
                audit.read_predictions(path)

    def test_conflicting_k0_truths_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "p.csv"
            self.write_csv(
                path,
                [
                    ("chembl", 0, 0, "h", 10.0, 1.0),
                    ("chembl", 1, 0, "h", 99.0, 1.0),
                ],
            )
            with self.assertRaises(ValueError):
                audit.read_predictions(path)

    def test_cache_csv_truth_mismatch_rejected(self):
        predictions = {
            "truths": {("db", "h1"): 100.0, ("db", "h2"): 11.0},
        }
        with self.assertRaises(ValueError):
            audit.cross_check_truths(
                "db",
                np.array(["h1", "h2"]),
                np.array([100.0, 10.0]),
                predictions,
            )

    def test_conflicting_k0_predictions_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "p.csv"
            self.write_csv(
                path,
                [
                    ("chembl", 0, 0, "h", 10.0, 1.0),
                    ("chembl", 1, 0, "h", 10.0, 2.0),
                ],
            )
            with self.assertRaises(ValueError):
                audit.read_predictions(path)

    def test_consistent_k0_is_indexed_once(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "p.csv"
            self.write_csv(
                path,
                [
                    ("chembl", 0, 0, "h", 10.0, 1.0),
                    ("chembl", 1, 0, "h", 10.0, 1.0),
                    ("chembl", 0, 20, "h", 10.0, 3.0),
                ],
            )
            result = audit.read_predictions(path)
            self.assertEqual(result["anchors"][("chembl", "h")], 1.0)
            self.assertEqual(result["adapted"][("chembl", 20)][(0, "h")], 3.0)


class CorrectionTests(unittest.TestCase):
    def test_forced_correction_flips_magnitude_and_recovers(self):
        # Anchor 0.0 vs truth log(100); adaptation moved *away* and worsened.
        truth = 100.0
        report = audit.correction_report(
            np.array([0.0]),
            np.array([-2.0]),
            np.array([truth]),
            np.array([1.0]),
            1e-9,
        )
        self.assertEqual(report["adapted"]["worsened_fraction"], 1.0)
        self.assertEqual(report["forced_direction"]["worsened_fraction"], 0.0)
        self.assertEqual(report["forced_direction"]["recovered_of_worsened"], 1.0)

    def test_gate_falls_back_to_anchor_when_direction_disagrees(self):
        truth = 100.0
        anchors = np.array([math.log(truth)])
        adapted = np.array([math.log(truth) - 2.0])
        report = audit.correction_report(
            anchors, adapted, np.array([truth]), np.array([1.0]), 1e-9
        )
        self.assertEqual(report["gate"]["worsened_fraction"], 0.0)
        self.assertAlmostEqual(report["gate"]["log_q95"], 0.0)

    def test_gate_cannot_repair_overshoot(self):
        # Correct direction but far too large: the gate accepts it unchanged,
        # so the query stays as bad as plain adaptation (no free repair).
        truth, anchor = 100.0, 0.0
        report = audit.correction_report(
            np.array([anchor]),
            np.array([20.0]),
            np.array([truth]),
            np.array([1.0]),
            1e-9,
        )
        self.assertEqual(report["gate"]["worsened_fraction"], 1.0)
        self.assertEqual(report["gate"]["recovered_of_worsened"], 0.0)

    def test_gate_never_creates_damage_on_intact_queries(self):
        # The gate only chooses between adapted and anchor, so a query whose
        # adaptation already improved cannot become worsened.
        truth = 100.0
        report = audit.correction_report(
            np.array([math.log(truth) + 1.0]),
            np.array([math.log(truth) + 0.5]),
            np.array([truth]),
            np.array([0.0]),
            1e-9,
        )
        self.assertEqual(report["gate"]["worsened_fraction"], 0.0)

    def test_oracle_direction_bounds_any_predicted_direction(self):
        truth = 100.0
        anchors = np.array([0.0, math.log(truth)])
        adapted = np.array([-2.0, math.log(truth) - 1.0])
        report = audit.correction_report(
            anchors, adapted, np.array([truth, truth]), np.array([0.0, 0.0]), 1e-9
        )
        self.assertLessEqual(
            report["oracle_direction"]["worsened_fraction"],
            report["forced_direction"]["worsened_fraction"],
        )

    def test_worsened_uses_anchor_relative_tolerance(self):
        truth = 100.0
        anchor = math.log(truth)
        report = audit.correction_report(
            np.array([anchor]),
            np.array([anchor + 1e-12]),
            np.array([truth]),
            np.array([0.0]),
            1e-9,
        )
        self.assertEqual(report["adapted"]["worsened_fraction"], 0.0)


@unittest.skipUnless(HAVE_SKLEARN, "scikit-learn is not installed")
class ModelTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.matrix = rng.normal(size=(200, 6)).astype(np.float32)
        self.labels = (
            self.matrix[:, 0] + 0.5 * self.matrix[:, 1] + rng.normal(0, 0.3, 200) > 0
        ).astype(np.int64)
        self.groups = np.repeat(np.arange(100), 2)

    def test_grouped_oof_covers_every_query(self):
        oof, folds = audit.grouped_oof(
            self.matrix, self.labels, self.groups, "hgb", n_splits=4
        )
        self.assertEqual(oof.shape, (200,))
        self.assertTrue(np.isfinite(oof).all())
        self.assertEqual(len(folds), 4)

    def test_effective_folds_respects_minority_and_groups(self):
        self.assertEqual(audit.effective_folds([1] * 100 + [0] * 3, np.arange(103)), 3)
        self.assertEqual(audit.effective_folds([1, 0, 1, 0], np.arange(4)), 2)

    def test_learning_curve_reports_growing_support(self):
        curve = audit.learning_curve(
            self.matrix, self.labels, self.groups, sizes=(20, 80), repeats=1
        )
        self.assertEqual([entry["labelled_queries"] for entry in curve], [20, 80])
        self.assertTrue(all(entry["auc_mean"] is not None for entry in curve))

    def test_grouped_oof_handles_nan_features(self):
        matrix = self.matrix.copy()
        matrix[:, 3] = np.nan
        oof, _ = audit.grouped_oof(
            matrix, self.labels, self.groups, "logistic", n_splits=4
        )
        self.assertTrue(np.isfinite(oof).all())


if __name__ == "__main__":
    unittest.main()
