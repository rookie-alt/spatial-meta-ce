import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from spatial_meta_ce.data import FeatureStore
from spatial_meta_ce.features import prepare_database, write_cache
from spatial_meta_ce.manifests import digest, file_hash

ARRAY_FILES = (
    "group0.npy",
    "group1.npy",
    "group2.npy",
    "group3.npy",
    "offsets.npy",
    "cards.npy",
)


class CacheIntegrityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "price"
        self.cache_root = self.root / "cache"
        self.path = self.cache_root / "pretrain" / "synthetic"
        workload = self.source / "datas/workloads/pretrain/synthetic/workloads.sql"
        workload.parent.mkdir(parents=True)
        workload.write_text("select count(*) from a where a.x=1||10||2\n")
        extractor = (
            Path(__file__).resolve().parents[1]
            / "vendor/price_features/features_tool.py"
        )
        self.provenance = {
            "source": {
                "workload_sha256": file_hash(workload),
                "feature_sha256": None,
                "database": "synthetic",
                "split": "pretrain",
                "sqlglot": "18.17.0",
                "statistics": {},
                "extractor_sha256": file_hash(extractor),
            },
            "excluded": [],
        }
        self.rows = [
            {"id": str(i), "query_hash": f"hash-{i}", "true": 10.0, "pg": 2.0}
            for i in range(2)
        ]
        self.data = {
            "data_features": [
                (np.ones(40), np.ones(40), np.ones(4), np.ones(43)) for _ in self.rows
            ],
            "true_cards": [10.0, 10.0],
            "pg_est_cards": [2.0, 2.0],
            "n_join_cols": [1, 1],
            "n_fanouts": [1, 1],
            "n_tables": [1, 1],
            "n_filter_cols": [1, 1],
        }

    def create(self, excluded=()):
        return write_cache(
            self.path,
            self.rows,
            self.data,
            {**self.provenance, "excluded": list(excluded)},
        )

    def rewrite_manifest(self, manifest):
        manifest["content_hash"] = digest(
            {k: v for k, v in manifest.items() if k != "content_hash"}
        )
        (self.path / "manifest.json").write_text(json.dumps(manifest))

    def prepare(self, allow_exclusions=False):
        return prepare_database(
            self.source,
            self.cache_root,
            "synthetic",
            "pretrain",
            allow_exclusions=allow_exclusions,
        )

    def test_new_manifest_binds_every_array_byte_and_loads(self):
        manifest = self.create()
        self.assertEqual(manifest["schema"], 2)
        self.assertEqual(
            manifest["array_sha256"],
            {name: file_hash(self.path / name) for name in ARRAY_FILES},
        )
        store = FeatureStore(self.path)
        self.assertTrue(store.array_hashes_verified)
        self.assertEqual(store[0]["true"], 10.0)

    def test_content_hash_changes_when_only_cardinality_changes(self):
        rows = [
            *self.rows,
            {**self.rows[0], "id": "2", "query_hash": "hash-2", "true": 1000.0},
        ]
        data = {key: [*values, values[0]] for key, values in self.data.items()}
        data["true_cards"] = [10.0, 10.0, 1000.0]
        first = write_cache(self.path, rows, data, self.provenance)
        rows = [rows[0], {**rows[1], "true": 20.0}, rows[2]]
        data = {**data, "true_cards": [10.0, 20.0, 1000.0]}
        second = write_cache(self.root / "second", rows, data, self.provenance)
        self.assertEqual(first["inspection"], second["inspection"])
        self.assertNotEqual(first["content_hash"], second["content_hash"])

    def test_each_same_shape_array_mutation_is_rejected(self):
        self.create()
        for name in ARRAY_FILES:
            with self.subTest(name=name):
                path = self.path / name
                original = path.read_bytes()
                values = np.load(path)
                values.flat[0] += 1
                np.save(path, values)
                with self.assertRaisesRegex(ValueError, "array.*hash"):
                    FeatureStore(self.path)
                path.write_bytes(original)

    def test_missing_and_truncated_arrays_are_rejected(self):
        self.create()
        path = self.path / "cards.npy"
        original = path.read_bytes()
        for content in (None, original[:20]):
            with self.subTest(content=content):
                if content is None:
                    path.unlink()
                else:
                    path.write_bytes(content)
                with self.assertRaisesRegex(ValueError, "array"):
                    FeatureStore(self.path)

    def test_v2_requires_complete_valid_array_hashes_even_with_valid_manifest_hash(
        self,
    ):
        manifest = self.create()
        for hashes in (
            None,
            {},
            {"cards.npy": "0" * 64},
            {**manifest.get("array_sha256", {}), "extra.npy": "0" * 64},
            {name: "not-a-sha256" for name in ARRAY_FILES},
        ):
            with self.subTest(hashes=hashes):
                self.rewrite_manifest({**manifest, "array_sha256": hashes})
                with self.assertRaisesRegex(ValueError, "array.*hash"):
                    FeatureStore(self.path)

    def test_legacy_cache_is_explicitly_unverified_and_is_not_rewritten(self):
        manifest = self.create()
        manifest["schema"] = 1
        manifest.pop("array_sha256", None)
        self.rewrite_manifest(manifest)
        before = {p.name: p.read_bytes() for p in self.path.iterdir()}
        with self.assertWarnsRegex(RuntimeWarning, "Legacy.*array"):
            store = FeatureStore(self.path)
        self.assertFalse(store.array_hashes_verified)
        self.assertEqual(store[0]["true"], 10.0)
        self.assertEqual(before, {p.name: p.read_bytes() for p in self.path.iterdir()})

    def test_reuse_verifies_arrays_instead_of_only_source_hashes(self):
        self.create()
        good = self.prepare()
        self.assertTrue(good["cached"])
        self.assertTrue(good["array_hashes_verified"])
        self.assertEqual(good["excluded"], 0)
        values = np.load(self.path / "cards.npy")
        values[0, 0] = 1000.0
        np.save(self.path / "cards.npy", values)
        with self.assertRaisesRegex(ValueError, "array.*hash"):
            self.prepare()

    def test_strict_reuse_refuses_excluded_queries_and_opt_in_discloses_count(self):
        manifest = self.create(
            excluded=[{"id": "missing-row", "error": "missing statistics"}]
        )
        before = (self.path / "manifest.json").read_bytes()
        with self.assertRaisesRegex(ValueError, "exclud"):
            self.prepare()
        reused = self.prepare(allow_exclusions=True)
        self.assertTrue(reused["cached"])
        self.assertEqual(reused["excluded"], 1)
        self.assertEqual(reused["content_hash"], manifest["content_hash"])
        self.assertEqual(before, (self.path / "manifest.json").read_bytes())

    def test_legacy_reuse_also_enforces_exclusion_policy(self):
        manifest = self.create(
            excluded=[{"id": "excluded", "error": "unsupported SQL"}]
        )
        manifest["schema"] = 1
        manifest.pop("array_sha256")
        self.rewrite_manifest(manifest)
        with (
            self.assertWarns(RuntimeWarning),
            self.assertRaisesRegex(ValueError, "exclud"),
        ):
            self.prepare()
        with self.assertWarns(RuntimeWarning):
            reused = self.prepare(allow_exclusions=True)
        self.assertFalse(reused["array_hashes_verified"])
        self.assertEqual(reused["excluded"], 1)

    def test_reuse_still_rejects_changed_source_and_manifest(self):
        manifest = self.create()
        self.rewrite_manifest(
            {**manifest, "provenance": {"source": {}, "excluded": []}}
        )
        with self.assertRaisesRegex(ValueError, "Source changed"):
            self.prepare()
        self.rewrite_manifest(manifest)
        (self.path / "manifest.json").write_text(
            json.dumps({**manifest, "unique_queries": 999})
        )
        with self.assertRaisesRegex(ValueError, "manifest.*hash"):
            self.prepare()


if __name__ == "__main__":
    unittest.main()
