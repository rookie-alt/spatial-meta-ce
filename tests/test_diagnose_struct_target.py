"""Tests for the target-anchor diagnosis (scripts/diagnose_struct_target.py).

The script's numbers are the evidence behind doc 120's conclusion, so the
by-table-count grouping and the error statistics are pinned here on a
synthetic cache.  Model loading is covered by the checkpoint tests and is not
re-exercised; ``diagnose_database`` accepts an empty checkpoint map for that.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

_SPEC = importlib.util.spec_from_file_location(
    "diagnose_struct_target", ROOT / "scripts/diagnose_struct_target.py"
)
diagnose = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(diagnose)

from spatial_meta_ce.data import GROUP_DIMS, TRANSFORM_VERSION  # noqa: E402
from spatial_meta_ce.manifests import digest  # noqa: E402


class DiagnoseTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def cache(self) -> Path:
        """Three queries with 2, 3 and 4 tables; anchor exact, 2x and exact."""
        path = self.root / "test" / "synth"
        path.mkdir(parents=True)
        offsets = np.zeros((4, 4), dtype=np.int64)
        offsets[1:, 2] = [2, 5, 9]
        np.save(path / "offsets.npy", offsets)
        for index, dim in enumerate(GROUP_DIMS):
            np.save(
                path / f"group{index}.npy",
                np.zeros((int(offsets[-1, index]), dim), dtype=np.float32),
            )
        np.save(
            path / "cards.npy",
            np.asarray([[10.0, 5.0], [100.0, 50.0], [1000.0, 500.0]]),
        )
        manifest = {
            "schema": 1,
            "transform": TRANSFORM_VERSION,
            "rows": [{"id": i, "query_hash": f"h{i}"} for i in range(3)],
            "unique_queries": 3,
        }
        manifest["content_hash"] = digest(manifest)
        (path / "manifest.json").write_text(json.dumps(manifest))
        np.save(
            path / "struct_log.npy",
            np.asarray([np.log(10.0), np.log(200.0), np.log(1000.0)], dtype=np.float64),
        )
        (path / "struct_manifest.json").write_text(
            json.dumps(
                {
                    "cache_content_hash": manifest["content_hash"],
                    "rows": 3,
                    "bins": 160,
                }
            )
        )
        return path

    def test_grouping_and_error_statistics(self):
        from spatial_meta_ce.data import FeatureStore

        store = FeatureStore(self.cache())
        report = diagnose.diagnose_database(store, {})

        self.assertEqual(report["queries"], 3)
        self.assertEqual(report["seeds"], {})
        by_tables = report["anchor_by_table_count"]
        self.assertEqual(sorted(by_tables), [2, 3, 4])
        self.assertEqual([by_tables[n]["count"] for n in (2, 3, 4)], [1, 1, 1])
        self.assertAlmostEqual(by_tables[2]["median_error"], 0.0, places=6)
        # batch() hands struct_log to the model as float32, so compare at 6 dp.
        self.assertAlmostEqual(by_tables[3]["median_error"], np.log(2.0), places=6)
        self.assertAlmostEqual(by_tables[4]["median_error"], 0.0, places=6)
        # np.percentile(..., 95) on n=3 interpolates at index 1.9 -> 0.9 * log(2).
        self.assertAlmostEqual(report["anchor_q95_error"], 0.9 * np.log(2.0), places=6)
        self.assertAlmostEqual(report["anchor_median_error"], 0.0, places=6)
        self.assertEqual(report["sidecar"]["struct_manifest"]["bins"], 160)
        self.assertEqual(
            report["sidecar"]["cache_content_hash"],
            store.manifest["content_hash"],
        )
        self.assertEqual(len(report["sidecar"]["struct_log_sha256"]), 64)

    def test_main_rejects_path_like_database_name(self):
        with self.assertRaises(SystemExit):
            diagnose.main(
                [
                    "--cache-root",
                    str(self.root),
                    "--databases",
                    "../synth",
                    "--checkpoint",
                    "42=missing.pt",
                ]
            )

    def test_main_rejects_malformed_checkpoint(self):
        with self.assertRaises(SystemExit):
            diagnose.main(
                [
                    "--cache-root",
                    str(self.root),
                    "--databases",
                    "synth",
                    "--checkpoint",
                    "not-a-seed-path",
                ]
            )

    def test_assert_struct_checkpoint_accepts_struct_without_pg(self):
        state = {
            "model_config": {"anchor": "struct", "use_pg": False},
            "metadata": {
                "train_databases": ["accidents"],
                "validation_databases": ["chembl"],
            },
        }
        diagnose.assert_struct_checkpoint(state, diagnose.TARGET_DATABASES)

    def test_assert_struct_checkpoint_rejects_pg_anchor(self):
        state = {
            "model_config": {"width": 64, "use_pg": True},
            "metadata": {
                "train_databases": ["accidents"],
                "validation_databases": ["chembl"],
            },
        }
        with self.assertRaises(ValueError):
            diagnose.assert_struct_checkpoint(state, diagnose.TARGET_DATABASES)

    def test_assert_struct_checkpoint_rejects_target_exposure(self):
        state = {
            "model_config": {"anchor": "struct", "use_pg": False},
            "metadata": {
                "train_databases": ["imdb"],
                "validation_databases": ["chembl"],
            },
        }
        with self.assertRaises(ValueError):
            diagnose.assert_struct_checkpoint(state, diagnose.TARGET_DATABASES)

    def test_assert_struct_checkpoint_rejects_adapted_checkpoint(self):
        state = {
            "model_config": {"anchor": "struct", "use_pg": False},
            "metadata": {
                "train_databases": ["accidents"],
                "validation_databases": ["chembl"],
                "adapted_database": "imdb",
            },
        }
        with self.assertRaises(ValueError):
            diagnose.assert_struct_checkpoint(state, diagnose.TARGET_DATABASES)


if __name__ == "__main__":
    unittest.main()
