"""Tests for the doc-118 structural anchor: batch plumbing and zero-init identity."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from spatial_meta_ce.data import GROUP_DIMS, FeatureStore, batch_samples  # noqa: E402
from spatial_meta_ce.model import SetEstimator  # noqa: E402


def sample(struct_log=None):
    groups = [np.zeros((0, dim), dtype=np.float32) for dim in GROUP_DIMS]
    value = {"groups": groups, "true": 10.0, "pg": 5.0, "id": "q"}
    if struct_log is not None:
        value["struct_log"] = struct_log
    return value


class StructBatchTests(unittest.TestCase):
    def test_struct_log_is_carried_through(self):
        batch = batch_samples([sample(2.5), sample(3.5)])
        self.assertIn("struct_log", batch)
        self.assertEqual(batch["struct_log"].tolist(), [2.5, 3.5])

    def test_absent_struct_log_is_absent(self):
        batch = batch_samples([sample(), sample()])
        self.assertNotIn("struct_log", batch)

    def test_mixed_presence_is_rejected(self):
        with self.assertRaises(ValueError):
            batch_samples([sample(1.0), sample()])

    def test_nonfinite_is_rejected(self):
        with self.assertRaises(ValueError):
            batch_samples([sample(float("nan"))])


class StructAnchorTests(unittest.TestCase):
    def test_anchor_returns_struct_log(self):
        model = SetEstimator(width=8, arch="sum", use_pg=False, anchor="struct")
        batch = batch_samples([sample(2.5)])
        self.assertAlmostEqual(model.output_anchor(batch).item(), 2.5, places=6)

    def test_struct_anchor_requires_the_sidecar(self):
        model = SetEstimator(width=8, arch="sum", use_pg=False, anchor="struct")
        batch = batch_samples([sample()])
        with self.assertRaises(ValueError):
            model.output_anchor(batch)

    def test_untrained_model_equals_the_anchor(self):
        model = SetEstimator(width=8, arch="sum", use_pg=False, anchor="struct")
        model.eval()
        batch = batch_samples([sample(2.5), sample(-1.0)])
        output = model(batch)
        self.assertEqual(output.tolist(), batch["struct_log"].tolist())

    def test_pg_anchor_is_unchanged_by_default(self):
        model = SetEstimator(width=8, arch="sum", use_pg=True)
        batch = batch_samples([sample(2.5)])
        self.assertEqual(model.output_anchor(batch).tolist(), batch["pg_log"].tolist())
        self.assertNotIn("anchor", model.config)

    def test_unknown_anchor_is_rejected(self):
        with self.assertRaises(ValueError):
            SetEstimator(anchor="oracle")

    def test_struct_anchor_rejects_pg_input(self):
        # use_pg defaults to True; the structural arm must be PG-free.
        with self.assertRaises(ValueError):
            SetEstimator(width=8, arch="sum", anchor="struct")

    def test_config_round_trips(self):
        model = SetEstimator(width=8, arch="sum", use_pg=False, anchor="struct")
        self.assertEqual(model.config["anchor"], "struct")
        clone = SetEstimator(**model.config)
        self.assertEqual(clone.config, model.config)


class FeatureStoreSidecarTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def cache(self, rows=2, struct=True, hash_mismatch=False, length=None):
        from spatial_meta_ce.data import TRANSFORM_VERSION
        from spatial_meta_ce.manifests import digest

        path = self.root / "pretrain" / "synth"
        path.mkdir(parents=True)
        np.save(path / "offsets.npy", np.zeros((rows + 1, 4), dtype=np.int64))
        for index, dim in enumerate(GROUP_DIMS):
            np.save(path / f"group{index}.npy", np.zeros((0, dim), dtype=np.float32))
        np.save(path / "cards.npy", np.asarray([[10.0, 5.0]] * rows))
        rowlist = [{"id": i, "query_hash": f"h{i}"} for i in range(rows)]
        manifest = {
            "schema": 1,
            "transform": TRANSFORM_VERSION,
            "rows": rowlist,
            "unique_queries": rows,
        }
        manifest["content_hash"] = digest(manifest)
        (path / "manifest.json").write_text(json.dumps(manifest))
        if struct:
            np.save(path / "struct_log.npy", np.asarray([1.5] * (length or rows)))
            (path / "struct_manifest.json").write_text(
                json.dumps(
                    {
                        "cache_content_hash": "mismatch"
                        if hash_mismatch
                        else manifest["content_hash"],
                        "rows": length or rows,
                    }
                )
            )
        return path

    def test_loads_sidecar(self):
        store = FeatureStore(self.cache())
        self.assertIsNotNone(store.struct_log)
        self.assertEqual(store[1]["struct_log"], 1.5)

    def test_absent_sidecar_leaves_none(self):
        store = FeatureStore(self.cache(struct=False))
        self.assertIsNone(store.struct_log)
        self.assertNotIn("struct_log", store[0])

    def test_hash_mismatch_is_rejected(self):
        with self.assertRaises(ValueError):
            FeatureStore(self.cache(hash_mismatch=True))

    def test_length_mismatch_is_rejected(self):
        with self.assertRaises(ValueError):
            FeatureStore(self.cache(length=1))

    def test_missing_manifest_is_rejected(self):
        path = self.cache()
        (path / "struct_manifest.json").unlink()
        with self.assertRaises(ValueError):
            FeatureStore(path)


if __name__ == "__main__":
    unittest.main()
