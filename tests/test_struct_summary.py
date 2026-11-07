"""Tests for the doc-118 structural summary (exact re-assembly and shared bins)."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))

import struct_summary as ss  # noqa: E402
from spatial_meta_ce import struct_summary as core  # noqa: E402
import tree_identity as ti  # noqa: E402
from g1_oracle import Predicate  # noqa: E402


def _write_dataset(root: Path, db: str, tables: dict[str, pd.DataFrame]) -> Path:
    dataset = root / db
    dataset.mkdir(parents=True)
    lines = []
    for name, frame in tables.items():
        frame.to_csv(dataset / f"{name}.csv", sep="|", index=False)
        lines.append(f"CREATE TABLE {name} (")
        lines.append(
            ",\n".join(
                f"  {column} {kind}" for column, kind in frame.attrs["kinds"].items()
            )
        )
        lines.append(");")
        lines.append(
            f"\\copy {name} from './{name}.csv' csv delimiter '|' null 'NULL' header;"
        )
    (dataset / f"postgres_create_{db}.sql").write_text("\n".join(lines) + "\n")
    return dataset


def _frame(data: dict[str, list], kinds: dict[str, str]) -> pd.DataFrame:
    frame = pd.DataFrame(data)
    frame.attrs["kinds"] = kinds
    return frame


def _plan(db: str, spec: ti.TreeSpec, truth: float) -> dict:
    return {
        "databases": {
            db: {
                "queries": [
                    {
                        "query_hash": "q1",
                        "sql": "select count(*) from t1 a join t2 b on a.id = b.ref",
                        "true": truth,
                        "pg": 1.0,
                        "spec": ti._json_spec(spec),
                    }
                ]
            }
        }
    }


class RebucketTests(unittest.TestCase):
    def hist(self, edges, counts, kind="int"):
        return core.ColumnHist(
            np.asarray(edges, float), np.asarray(counts, float), 10, 10, kind
        )

    def test_identity_when_edges_match(self):
        hist = self.hist([0, 1, 2, 3], [2, 3, 5])
        np.testing.assert_allclose(
            hist.rebucket(np.asarray([0, 1, 2, 3], float)), [2, 3, 5]
        )

    def test_splits_by_overlap_fraction(self):
        # Continuous domains split geometrically; integers split by key count.
        hist = self.hist([0, 10], [10], kind="float")
        np.testing.assert_allclose(hist.rebucket(np.asarray([0, 5, 10], float)), [5, 5])

    def test_empty_hist_gives_zeros(self):
        hist = self.hist([0, 1], [0])
        np.testing.assert_allclose(hist.rebucket(np.asarray([0, 1], float)), [0])

    def test_full_range_contained_in_wider_target(self):
        hist = self.hist([0, 10], [10])
        np.testing.assert_allclose(
            hist.rebucket(np.asarray([0, 20, 40], float)), [10, 0]
        )

    def test_integer_source_bin_keeps_its_own_key(self):
        # A source bin [0.622, 1.244) holds only integer key 1.  Geometric
        # splitting would give bin0 (key 0) 0.378/0.622 of the mass; integer
        # accounting must put all of it on key 1.
        hist = self.hist([0.622, 1.244], [7])
        np.testing.assert_allclose(hist.rebucket(np.asarray([0, 1, 2], float)), [0, 7])


class BucketJoinSumTests(unittest.TestCase):
    def hist(self, edges, counts, kind="int"):
        return core.ColumnHist(
            np.asarray(edges, float), np.asarray(counts, float), 100, 100, kind
        )

    def test_aligned_uniform_edges_are_exact(self):
        # 200 rows over 100 integer keys in four bins of 25 values each; uniform
        # density gives 2 rows per key.  The last bin is closed on the right, so
        # it holds 26 keys and the estimate lands just under the exact 400.
        left = self.hist([0, 25, 50, 75, 100], [50, 50, 50, 50])
        right = self.hist([0, 25, 50, 75, 100], [50, 50, 50, 50])
        exact = 4 * (50 / 25) * (50 / 25) * 25
        self.assertAlmostEqual(
            core.bucket_join_sum([left, right], bins=4), exact, delta=5
        )

    def test_disjoint_domains_contribute_zero(self):
        left = self.hist([0, 10], [10])
        right = self.hist([100, 110], [10])
        self.assertEqual(core.bucket_join_sum([left, right], bins=4), 0.0)

    def test_narrow_domain_uses_uniform_assumption(self):
        # One stored bin covering two integer keys: the summary cannot tell the
        # rows apart, so each key gets half of 7 rows and half of 9 rows.
        left = self.hist([5, 6], [7])
        right = self.hist([5, 6], [9])
        self.assertAlmostEqual(
            core.bucket_join_sum([left, right], bins=4), 7 * 9 / 2, places=6
        )

    def test_constant_columns_multiply_exactly(self):
        left = self.hist([5, 5], [7])
        right = self.hist([5, 5], [9])
        self.assertAlmostEqual(
            core.bucket_join_sum([left, right], bins=4), 63.0, places=6
        )

    def test_null_only_member_makes_the_join_zero(self):
        left = self.hist([0, 10], [10])
        empty = core.ColumnHist(np.asarray([0.0, 1.0]), np.asarray([0.0]), 10, 0, "int")
        # A join variable with a NULL-only member joins nothing; this matches
        # tree_identity.score_query_tree on the same shape.
        self.assertEqual(core.bucket_join_sum([left, empty], bins=4), 0.0)

    def test_constant_column_outside_shared_domain_joins_nothing(self):
        left = self.hist([0, 10], [10])
        stray = self.hist([100, 100], [5])
        self.assertEqual(core.bucket_join_sum([left, stray], bins=4), 0.0)

    def test_mixed_join_key_kinds_raise(self):
        as_int = self.hist([0, 10], [10], kind="int")
        as_str = self.hist([0, 10], [10], kind="str")
        with self.assertRaises(ValueError):
            core.bucket_join_sum([as_int, as_str], bins=4)

    def test_integer_width_does_not_depend_on_domain_magnitude(self):
        hist = core.ColumnHist(
            np.asarray([0.0, 4e8, 8e8]), np.asarray([1.0, 1.0]), 2, 2, "int"
        )
        widths = core._bin_widths(hist.edges, "int")
        np.testing.assert_allclose(widths, [4e8, 4e8])


class EqualWidthHistTests(unittest.TestCase):
    def test_matches_numpy_histogram(self):
        series = pd.Series([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype="Int64")
        hist = core.equal_width_hist(series, bins=5, kind="int")
        counts, edges = np.histogram(series.to_numpy(dtype=float), bins=5)
        np.testing.assert_allclose(hist.counts, counts)
        np.testing.assert_allclose(hist.edges, edges)

    def test_nulls_excluded_from_counts(self):
        series = pd.Series([1, None, 3], dtype="Int64")
        hist = core.equal_width_hist(series, bins=2, kind="int")
        self.assertEqual(hist.counts.sum(), 2)
        self.assertEqual(hist.nonnull, 2)
        self.assertEqual(hist.total, 3)

    def test_constant_column_single_bin(self):
        hist = core.equal_width_hist(pd.Series([7, 7, 7], dtype="Int64"), kind="int")
        self.assertEqual(hist.counts.sum(), 3)
        self.assertEqual(len(hist.edges), 2)


class SummaryRoundTripTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        t1 = _frame(
            {"id": [1, 2, 3, 4], "val": [1, 0, 1, 0]},
            {"id": "integer", "val": "integer"},
        )
        t2 = _frame(
            {"ref": [1, 1, 2, 4], "val": [0, 1, 1, 1]},
            {"ref": "integer", "val": "integer"},
        )
        self.dataset = _write_dataset(self.root, "synth", {"t1": t1, "t2": t2})
        self.spec = ti.TreeSpec(
            tables=("a", "b"),
            names=("t1", "t2"),
            predicates=(("a", "id", "b", "ref"),),
            filters=(Predicate("a", "val", "=", "1", False),),
        )

    def tearDown(self):
        self.tmp.cleanup()

    def test_build_and_verify_are_bit_identical(self):
        plan = _plan("synth", self.spec, truth=3.0)
        out = self.root / "summary"
        ss.build(plan, self.root, out)
        manifest = json.loads((out / "manifest.json").read_text())
        self.assertIn("synth", manifest["databases"])
        result = ss.verify(plan, self.root, out)
        self.assertTrue(result["bit_identical"], result["mismatches"])

    def test_manifest_hash_rejects_tampering(self):
        plan = _plan("synth", self.spec, truth=3.0)
        out = self.root / "summary"
        ss.build(plan, self.root, out)
        manifest = json.loads((out / "manifest.json").read_text())
        manifest["bins"] = 999
        (out / "manifest.json").write_text(json.dumps(manifest))
        with self.assertRaises(ValueError):
            ss._load_summary(out)


class BinnedEstimatorTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        keys = list(range(1, 101))
        t1 = _frame({"id": keys, "val": [1] * 100}, {"id": "integer", "val": "integer"})
        t2 = _frame(
            {"ref": keys, "val": [1] * 100}, {"ref": "integer", "val": "integer"}
        )
        _write_dataset(self.root, "synth", {"t1": t1, "t2": t2})
        kinds, files = ss.parse_ddl(self.root / "synth" / "postgres_create_synth.sql")
        self.tables = ss.load_tables(self.root / "synth", kinds, files)
        self.frames = {"a": self.tables["t1"], "b": self.tables["t2"]}
        self.spec = ti.TreeSpec(
            tables=("a", "b"),
            names=("t1", "t2"),
            predicates=(("a", "id", "b", "ref"),),
            filters=(),
        )

    def tearDown(self):
        self.tmp.cleanup()

    def test_binned_matches_exact_when_domains_align(self):
        exact = ti.score_query_tree(self.spec, self.tables, 100.0)["c_uncond"]
        binned = ss.binned_c_uncond(self.spec, self.frames, bins=40)
        # Integer bins coarser than one key add discretisation noise; 5% is the
        # tolerance the shared-bin design accepts on a perfectly uniform table.
        self.assertLess(abs(binned / exact - 1.0), 0.05)

    def test_empty_filter_result_is_zero(self):
        spec = ti.TreeSpec(
            tables=self.spec.tables,
            names=self.spec.names,
            predicates=self.spec.predicates,
            filters=(Predicate("a", "val", "=", "0", False),),
        )
        self.assertEqual(ss.binned_c_uncond(spec, self.frames, bins=40), 0.0)

    def test_cached_path_matches_uncached(self):
        # `queries` uses the cached path; `bineval` uses binned_c_uncond. They
        # must agree exactly, or the anchor and the reported Q95 diverge.
        for bins in (40, 160):
            direct = ss.binned_c_uncond(self.spec, self.frames, bins)
            cached = ss._cached_estimate(self.spec, self.tables, bins, {}, {})
            self.assertAlmostEqual(direct, cached, places=9)


class RepeatedMiddleColumnTests(unittest.TestCase):
    """Two predicates on the same hub column must keep one shared variable."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        hub = _frame(
            {"k": [1, 2, 3], "val": [1, 1, 1]}, {"k": "integer", "val": "integer"}
        )
        left = _frame(
            {"k": [1, 2, 3], "val": [1, 1, 1]}, {"k": "integer", "val": "integer"}
        )
        right = _frame(
            {"k": [1, 2, 3], "val": [1, 1, 1]}, {"k": "integer", "val": "integer"}
        )
        _write_dataset(self.root, "synth", {"hub": hub, "left": left, "right": right})
        self.kinds, self.files = ss.parse_ddl(
            self.root / "synth" / "postgres_create_synth.sql"
        )
        self.frames = ss.load_tables(self.root / "synth", self.kinds, self.files)
        self.tables = self.frames
        self.frames = {
            "h": self.tables["hub"],
            "l": self.tables["left"],
            "r": self.tables["right"],
        }
        self.spec = ti.TreeSpec(
            tables=("h", "l", "r"),
            names=("hub", "left", "right"),
            predicates=(("h", "k", "l", "k"), ("h", "k", "r", "k")),
            filters=(),
        )

    def tearDown(self):
        self.tmp.cleanup()

    def test_binned_matches_exact_for_repeated_middle_column(self):
        exact = ti.score_query_tree(self.spec, self.tables, 3.0)["c_uncond"]
        binned = ss.binned_c_uncond(self.spec, self.frames, bins=40)
        self.assertAlmostEqual(binned / exact, 1.0, places=2)


class VariableClassDriftTests(unittest.TestCase):
    """The frozen evidence script and the package core must agree exactly."""

    def test_repeated_middle_column_matches_package_core(self):
        spec = ti.TreeSpec(
            tables=("h", "l", "r"),
            names=("hub", "left", "right"),
            predicates=(("h", "k", "l", "k"), ("h", "k", "r", "k")),
            filters=(),
        )
        frozen = ti.variable_classes(spec)
        packed = core.variable_classes(spec.predicates, spec.tables)
        self.assertEqual(frozen, packed)

    def test_star_matches_package_core(self):
        spec = ti.TreeSpec(
            tables=("a", "b", "c"),
            names=("t1", "t2", "t3"),
            predicates=(("a", "id", "b", "ref"), ("a", "id", "c", "rid")),
            filters=(),
        )
        self.assertEqual(
            ti.variable_classes(spec),
            core.variable_classes(spec.predicates, spec.tables),
        )


if __name__ == "__main__":
    unittest.main()
