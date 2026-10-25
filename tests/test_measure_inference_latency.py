import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import torch

AVAILABLE = importlib.util.find_spec("spatial_meta_ce.train") is not None
if AVAILABLE:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

    from measure_inference_latency import main, price_maxima, summarize
    from spatial_meta_ce.checkpoints import save_checkpoint
    from spatial_meta_ce.features import write_cache
    from spatial_meta_ce.model import SetEstimator

SAMPLE_ROWS = 12
MODEL_CONFIG = {"width": 8, "use_pg": True}


def write_split(cache_root, split, database, rows=SAMPLE_ROWS):
    records = [
        {
            "id": f"{database}/{split}/{i}",
            "query_hash": f"{split}-{i}",
            "true": float(20 + i),
            "pg": 2.0,
        }
        for i in range(rows)
    ]
    data = {
        "data_features": [
            (torch.ones(40), torch.ones(40), torch.ones(4), None) for _ in records
        ],
        "true_cards": [torch.tensor(r["true"]) for r in records],
        "pg_est_cards": [torch.tensor(2.0) for _ in records],
        "n_join_cols": [1] * rows,
        "n_fanouts": [1] * rows,
        "n_tables": [1] * rows,
        "n_filter_cols": [0] * rows,
    }
    write_cache(
        Path(cache_root) / split / database, records, data, {"source": "unit-fixture"}
    )


def make_checkpoint(path, metadata):
    model = SetEstimator(**MODEL_CONFIG)
    save_checkpoint(
        path,
        model,
        None,
        0,
        0.0,
        0,
        {
            "method": "unit-fixture",
            "train_databases": ["a"],
            "validation_databases": ["v"],
            **metadata,
        },
    )
    return path


def fixture(root):
    write_split(root, "test", "t")
    return make_checkpoint(root / "check.pt", {})


def argv(root, checkpoint, output, **overrides):
    values = {
        "cache_root": str(root),
        "checkpoint": str(checkpoint),
        "databases": ["t"],
        "allowed_targets": ["t"],
        "warmup": 2,
        "repeat_times": 3,
        "output": str(output),
    }
    values.update(overrides)
    args = []
    for key, value in values.items():
        flag = "--" + key.replace("_", "-")
        if isinstance(value, list):
            args += [flag] + [str(item) for item in value]
        else:
            args += [flag, str(value)]
    return args


class SummarizeTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue(AVAILABLE, "spatial_meta_ce not importable")

    def test_percentiles_and_counts(self):
        stats = summarize([1.0, 2.0, 3.0, 4.0])
        self.assertEqual(stats["n"], 4)
        self.assertAlmostEqual(stats["min"], 1.0)
        self.assertAlmostEqual(stats["max"], 4.0)
        self.assertAlmostEqual(stats["mean"], 2.5)
        self.assertGreaterEqual(stats["p99"], stats["p95"])
        self.assertGreaterEqual(stats["p95"], stats["p90"])
        self.assertGreaterEqual(stats["p90"], stats["p50"])

    def test_rejects_degenerate_samples(self):
        for bad in ([], [float("nan")], [-1.0]):
            with self.assertRaises(ValueError):
                summarize(bad)


class MaximaTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue(AVAILABLE, "spatial_meta_ce not importable")

    def test_cache_mode_is_per_database(self):
        caches = {"t": {"test": [5, 4, 3, 2]}, "u": {"test": [9, 1, 1, 1]}}
        self.assertEqual(price_maxima(caches, "t", "cache"), [5, 4, 3, 2])

    def test_pooled_test_mode_is_elementwise_max_over_test_caches(self):
        caches = {"t": {"test": [5, 4, 3, 2]}, "u": {"test": [9, 1, 1, 7]}}
        self.assertEqual(price_maxima(caches, "t", "pooled_test"), [9, 4, 3, 7])

    def test_unknown_mode_is_an_error(self):
        with self.assertRaises(ValueError):
            price_maxima({"t": {"test": [1, 1, 1, 1]}}, "t", "nonsense")


class CliTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue(AVAILABLE, "spatial_meta_ce not importable")
        torch.set_num_threads(1)

    def test_unapproved_database_is_refused(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            checkpoint = fixture(root)
            with self.assertRaises(SystemExit):
                main(argv(root, checkpoint, root / "out.json", allowed_targets=["u"]))
            self.assertFalse((root / "out.json").exists())

    def test_checkpoint_exposed_on_target_is_refused(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            write_split(root, "test", "t")
            checkpoint = make_checkpoint(
                root / "exposed.pt", {"train_databases": ["t"]}
            )
            with self.assertRaises(SystemExit):
                main(argv(root, checkpoint, root / "out.json"))

    def test_existing_output_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            checkpoint = fixture(root)
            existing = root / "out.json"
            existing.write_text("keep me")
            with self.assertRaises(SystemExit):
                main(argv(root, checkpoint, existing))
            self.assertEqual(existing.read_text(), "keep me")

    def test_writes_latency_and_protocol(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            checkpoint = fixture(root)
            output = root / "latency.json"
            self.assertEqual(main(argv(root, checkpoint, output)), 0)

            latency = json.loads(output.read_text())
            protocol = json.loads((root / "latency.protocol.json").read_text())
            ours = latency["databases"]["t"]["models"]["ours"]
            self.assertFalse(latency["includes_feature_extraction"])
            self.assertFalse(latency["includes_database_execution"])
            self.assertEqual(ours["population"]["n"], SAMPLE_ROWS - 2)
            self.assertEqual(ours["population_repeat"]["n"], SAMPLE_ROWS - 2)
            self.assertEqual(ours["single_row_repeated"]["n"], 3)
            self.assertGreaterEqual(ours["stability_relative_delta"], 0.0)
            self.assertIn("stability_ok", ours)
            self.assertEqual(latency["stability_breaches"], [])
            self.assertGreater(ours["throughput"]["rows_per_second"], 0.0)
            self.assertNotIn("price", latency["databases"]["t"]["models"])
            self.assertEqual(protocol["databases_read"], ["t"])
            self.assertEqual(protocol["threads"], 1)
            self.assertEqual(protocol["caches"]["t"]["samples"], SAMPLE_ROWS)
            self.assertEqual(len(protocol["script_sha256"]), 64)
            self.assertEqual(len(protocol["checkpoint_sha256"]), 64)
            self.assertIn("torch", protocol["environment"])
            self.assertIsNone(protocol["price"])


if __name__ == "__main__":
    unittest.main()
