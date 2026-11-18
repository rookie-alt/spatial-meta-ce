import csv
import gzip
import importlib
import tempfile
import unittest
from pathlib import Path


class TargetRuntimeTests(unittest.TestCase):
    def module(self):
        return importlib.import_module("spatial_meta_ce.price_meta126_runtime")

    def test_authorization_requires_explicit_four_targets_and_frozen_budget(self):
        m = self.module()
        binding = {"root": "/isolated", "binding_sha256": "a" * 64}
        auth = {
            "schema": 1,
            "protocol": 126,
            "authorized": True,
            **binding,
            "actions": ["evaluate"],
            "allowed_targets": list(m.DATABASES),
            "selection_forbidden": True,
            "granted_by": "user",
            "granted_at": "2026-11-18",
            "budget": {
                "max_wall_seconds": 10800,
                "max_rss_bytes": 24 * 1024**3,
                "max_vram_bytes": 12 * 1024**3,
                "max_output_bytes": 1024**3,
                "poll_seconds": 1,
            },
        }
        m.validate_authorization(auth, binding)
        for changed in (
            {**auth, "allowed_targets": ["stats"]},
            {**auth, "selection_forbidden": False},
            {**auth, "protocol": 125},
        ):
            with self.assertRaises(ValueError):
                m.validate_authorization(changed, binding)

    def test_csv_independent_metrics_reject_wrong_population_or_nonfinite(self):
        m = self.module()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pred.csv.gz"
            identity = [
                ("stats/test/0", "a" * 64, 10.0),
                ("stats/test/1", "b" * 64, 20.0),
            ]
            import math

            rows = [
                {
                    "arm": "P",
                    "stream": 0,
                    "database": "stats",
                    "k": 0,
                    "run": 0,
                    "query_id": i,
                    "query_hash": h,
                    "true": t,
                    "encoded": math.log1p(t) + 1,
                    "log_prediction": math.log(t),
                }
                for i, h, t in identity
            ]

            def write():
                with gzip.open(path, "wt", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=m.FIELDS)
                    writer.writeheader()
                    writer.writerows(rows)

            write()
            metrics = m.verify_csv(path, identity)
            self.assertAlmostEqual(metrics["q95"], 1)
            with self.assertRaises(ValueError):
                m.verify_csv(path, identity[::-1])
            rows[0]["encoded"] = float("nan")
            write()
            with self.assertRaises((ValueError, FloatingPointError)):
                m.verify_csv(path, identity)

    def test_layout_extension_preserves_weights_and_short_query_forward(self):
        m = self.module()
        import torch
        from spatial_meta_ce.price_meta123_model import NativePriceModel
        from spatial_meta_ce.data import batch_samples
        from spatial_meta_ce.price_meta123_runtime import file_sha

        torch.set_num_threads(1)
        torch.manual_seed(1)
        model = NativePriceModel().eval()
        sample = {
            "groups": [
                torch.ones(1, 40),
                torch.ones(1, 40),
                torch.ones(1, 4),
                torch.ones(1, 43),
            ],
            "true": 12.0,
            "pg": 5.0,
            "id": "sample",
        }
        batch = batch_samples([sample])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "last.pt"
            torch.save(model.price.state_dict(), path)
            a = m.build_target_model(path, (12, 12, 7, 10), file_sha(path), "cpu")
            b = m.build_target_model(path, (12, 14, 8, 10), file_sha(path), "cpu")
            self.assertEqual(m.state_sha(a), m.state_sha(b))
            with torch.no_grad():
                torch.testing.assert_close(a(batch), b(batch), rtol=2e-5, atol=2e-5)
            with self.assertRaises(ValueError):
                m.build_target_model(path, (12, 12, 7, 10), "0" * 64, "cpu")
            long_sample = dict(sample)
            long_sample["groups"] = [
                torch.ones(12, 40),
                torch.ones(14, 40),
                torch.ones(8, 4),
                torch.ones(6, 43),
            ]
            long_batch = batch_samples([long_sample])
            with self.assertRaisesRegex(ValueError, "capacity"):
                a(long_batch)
            with torch.no_grad():
                self.assertTrue(torch.isfinite(b(long_batch)).all())

    def test_pg_reference_is_recomputed_on_same_raw_truth_and_query_identity(self):
        m = self.module()
        import numpy as np
        import torch
        from spatial_meta_ce.data import batch_samples

        class Store:
            rows = [
                {"id": "imdb/test/0", "query_hash": "a" * 64},
                {"id": "imdb/test/1", "query_hash": "b" * 64},
            ]
            cards = np.asarray([[10.0, 20.0], [20.0, 10.0]])

            def batch(self, indices, device):
                return batch_samples(
                    [
                        {
                            "groups": [
                                torch.ones(1, 40),
                                torch.ones(1, 40),
                                torch.ones(1, 4),
                                torch.empty(0, 43),
                            ],
                            "true": self.cards[i, 0],
                            "pg": self.cards[i, 1],
                            "id": self.rows[i]["id"],
                        }
                        for i in indices
                    ],
                    device,
                )

        with tempfile.TemporaryDirectory() as directory:
            result = m.unit(
                directory,
                None,
                None,
                "imdb",
                Store(),
                [0, 1],
                None,
                {"arm": "PG", "stream": 0, "database": "imdb", "k": 0, "run": 0},
            )
            self.assertEqual(result["n"], 2)
            self.assertAlmostEqual(result["metrics"]["q95"], 2.0, places=5)


if __name__ == "__main__":
    unittest.main()
