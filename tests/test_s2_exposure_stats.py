"""Synthetic tests for the source-only S2 statistics; no experiment inputs."""

import unittest

import numpy as np

from scripts.s2_exposure_stats import build_tables, weighted_quantile


def fixture(size=210):
    caches = {
        db: {
            "true": np.full(size, 20.0, dtype=np.float64),
            "pg": np.full(size, 10.0, dtype=np.float64),
            "n_tables": np.full(size, 5),
            "query_hash": [str(i) for i in range(size)],
            "raw_rows": size + 2,
            "indices": np.arange(size),
        }
        for db in ["a", "b", "v"]
    }
    state = {
        db: {
            "support": np.ones(size, dtype=np.int64),
            "query": np.ones(size, dtype=np.int64),
            "mass": np.full(size, 1 / size),
            "episodes": 2,
            "k_counts": {5: 2},
        }
        for db in ["a", "b"]
    }
    replay = {
        "prefixes": {
            step: {
                "databases": state,
                "digest": "fixture",
                "episodes": 4,
                "presentations": 4 * size,
            }
            for step in [1000, 2000, 4000]
        },
        "digest": "fixture",
    }
    return caches, replay


def row(rows, **keys):
    return next(r for r in rows if all(r[k] == v for k, v in keys.items()))


class StatisticsTests(unittest.TestCase):
    def test_fixed_boundaries_and_complete_axes(self):
        caches, replay = fixture(14)
        values = [
            0,
            0.5,
            9.9,
            10,
            99,
            100,
            999,
            1000,
            9999,
            10000,
            99999,
            100000,
            1e9,
            1,
        ]
        caches["a"] = {
            **caches["a"],
            "true": np.array(values),
            "n_tables": np.array([1, 2, 3, 4, 5, 6, 7] * 2),
        }
        tables = build_tables(caches, replay, ["a", "b"], ["v"])
        selected = [r for r in tables["pool"] if r["database"] == "a"]
        for axis, length in [("cardinality", 7), ("n_tables", 4), ("cross", 28)]:
            axis_rows = [r for r in selected if r["axis"] == axis]
            self.assertEqual(len(axis_rows), length)
            self.assertEqual(sum(r["pool_unique"] for r in axis_rows), 14)
            self.assertAlmostEqual(sum(r["share"] for r in axis_rows), 1)
        self.assertEqual(
            row(selected, axis="cardinality", bucket="(0,10)")["pool_unique"], 3
        )

    def test_sparse_unique_cannot_be_inflated_by_repeats(self):
        caches, replay = fixture()
        for prefix in replay["prefixes"].values():
            for db in ["a", "b"]:
                prefix["databases"][db] = {
                    **prefix["databases"][db],
                    "support": np.r_[100000, np.zeros(209, dtype=int)],
                    "query": np.zeros(210, dtype=int),
                    "mass": np.r_[1.0, np.zeros(209)],
                }
        tables = build_tables(caches, replay, ["a", "b"], ["v"])
        record = row(
            tables["residuals"],
            database="train_pooled",
            step=4000,
            axis="all",
            population="combined_presentations",
        )
        self.assertEqual(record["observed_unique"], 2)
        self.assertEqual(record["status"], "sparse")
        self.assertIsNone(record["r_ref_mean"])
        self.assertEqual(record["weight_mass"], 200000)

    def test_fp32_rounding_does_not_change_reference_direction(self):
        caches, replay = fixture()
        caches["a"] = {
            **caches["a"],
            "true": np.full(210, 1e8 + 1),
            "pg": np.full(210, 1e8),
        }
        result = build_tables(caches, replay, ["a", "b"], ["v"])
        r = row(result["residuals"], database="a", population="pool_unique", axis="all")
        self.assertGreater(r["r_ref_mean"], 0)
        self.assertEqual(r["r_fp32_mean"], 0)
        self.assertEqual(r["positive_share"], 1)
        self.assertEqual(r["zero_share"], 0)

    def test_weighted_quantiles_inverse_ecdf(self):
        values, weights = np.array([0.0, 10.0, 20.0]), np.array([1.0, 8.0, 1.0])
        np.testing.assert_equal(
            weighted_quantile(values, weights, [0.1, 0.5, 0.9]), [0, 10, 10]
        )
        np.testing.assert_equal(
            weighted_quantile(values, np.array([0.0, 0.0, 2.0]), [0, 1]), [20, 20]
        )
        with self.assertRaises(ValueError):
            weighted_quantile(values, np.zeros(3), [0.5])

    def test_zero_denominator_and_comparison_conservation(self):
        caches, replay = fixture()
        tables = build_tables(caches, replay, ["a", "b"], ["v"])
        zero = row(
            tables["comparison"],
            step=4000,
            axis="cardinality",
            bucket="0",
            source_population="nominal_mass",
            validation_database="v",
        )
        self.assertEqual(zero["ratio"], "undefined")
        for population in [
            "train_db_equal_pool",
            "train_pooled_pool",
            "combined_presentations",
            "nominal_mass",
        ]:
            for axis in ["cardinality", "n_tables", "cross"]:
                rows = [
                    r
                    for r in tables["comparison"]
                    if r["step"] == 4000
                    and r["source_population"] == population
                    and r["validation_database"] == "v"
                    and r["axis"] == axis
                ]
                self.assertAlmostEqual(sum(r["source_share"] for r in rows), 1)
                self.assertAlmostEqual(sum(r["validation_share"] for r in rows), 1)
                self.assertAlmostEqual(rows[0]["axis_tv"], 0)
        for axis in ["cardinality", "n_tables", "cross"]:
            rows = [
                r
                for r in tables["exposure"]
                if r["step"] == 4000
                and r["database"] == "train_pooled"
                and r["axis"] == axis
            ]
            self.assertEqual(sum(r["combined_presentations"] for r in rows), 840)
            self.assertAlmostEqual(sum(r["nominal_mass"] for r in rows), 2)

    def test_cross_database_hashes_are_not_deduplicated(self):
        caches, replay = fixture()
        tables = build_tables(caches, replay, ["a", "b"], ["v"])
        r = row(tables["pool"], database="train_pooled", axis="all")
        self.assertEqual(r["pool_unique"], 420)
        self.assertEqual(r["raw_rows"], 424)
        self.assertEqual(r["source_db_coverage"], 2)
        summary = tables["distribution_summary"]["prefixes"]["4000"]["a"]
        self.assertEqual(summary["repeat_counts"]["2"], 210)

    def test_equal_database_pool_and_unweighted_quantiles(self):
        caches, replay = fixture()
        caches["a"] = {**caches["a"], "pg": np.ones(210)}
        result = build_tables(caches, replay, ["a", "b"], ["v"])
        r = row(result["residuals"], database="train_db_equal_pool", axis="all")
        self.assertEqual(r["quantile_method"], "inverse_ecdf")
        self.assertAlmostEqual(r["weight_mass"], 2)
        r = row(result["residuals"], database="v", axis="all")
        self.assertEqual(r["quantile_method"], "linear")
        self.assertAlmostEqual(r["r_ref_median"], np.log(2))

    def test_unequal_database_sizes_keep_equal_and_pooled_estimands_separate(self):
        caches, replay = fixture(210)
        caches["a"] = {
            "true": np.zeros(420),
            "pg": np.ones(420),
            "n_tables": np.ones(420, dtype=int),
            "query_hash": [str(i) for i in range(420)],
            "raw_rows": 420,
            "indices": np.arange(420),
        }
        for prefix in replay["prefixes"].values():
            prefix["databases"]["a"] = {
                "support": np.ones(420, dtype=int),
                "query": np.ones(420, dtype=int),
                "mass": np.full(420, 1 / 420),
                "episodes": 2,
                "k_counts": {5: 2},
            }
        result = build_tables(caches, replay, ["a", "b"], ["v"])
        equal = row(
            result["pool"],
            database="train_db_equal_pool",
            axis="cardinality",
            bucket="0",
        )
        pooled = row(
            result["pool"], database="train_pooled", axis="cardinality", bucket="0"
        )
        self.assertAlmostEqual(equal["share"], 0.5)
        self.assertAlmostEqual(pooled["share"], 2 / 3)
        equal_residual = row(
            result["residuals"], database="train_db_equal_pool", axis="all"
        )
        self.assertAlmostEqual(equal_residual["r_ref_mean"], np.log(2) / 2)
        pooled_residual = row(
            result["residuals"], database="train_pooled", axis="all", step=0
        )
        self.assertAlmostEqual(pooled_residual["r_ref_mean"], np.log(2) / 3)

    def test_small_pool_remains_sparse_and_empty_exposure_is_undefined(self):
        caches, replay = fixture(2)
        for prefix in replay["prefixes"].values():
            for db in ["a", "b"]:
                prefix["databases"][db] = {
                    **prefix["databases"][db],
                    "support": np.zeros(2, dtype=int),
                    "query": np.zeros(2, dtype=int),
                    "mass": np.zeros(2),
                }
        result = build_tables(caches, replay, ["a", "b"], ["v"])
        r = row(result["residuals"], database="a", axis="all", step=0)
        self.assertEqual(r["status"], "sparse")
        self.assertIsNone(r["positive_share"])
        c = row(
            result["comparison"],
            step=1000,
            source_population="nominal_mass",
            validation_database="v",
            axis="cardinality",
            bucket="[10,100)",
        )
        self.assertEqual(c["source_share"], "undefined")
        self.assertEqual(c["axis_tv"], "undefined")

    def test_unweighted_linear_and_weighted_ecdf_are_distinct(self):
        caches, replay = fixture(200)
        caches["a"] = {
            **caches["a"],
            "true": np.r_[np.ones(100), np.full(100, 100.0)],
            "pg": np.ones(200),
        }
        result = build_tables(caches, replay, ["a", "b"], ["v"])
        plain = row(
            result["residuals"], database="a", population="pool_unique", axis="all"
        )
        weighted = row(
            result["residuals"],
            database="a",
            population="combined_presentations",
            axis="all",
            step=4000,
        )
        self.assertEqual(plain["status"], "ok")
        self.assertAlmostEqual(plain["r_ref_median"], np.log(100) / 2)
        self.assertEqual(weighted["r_ref_median"], 0)

    def test_invalid_inputs_fail_instead_of_dropping_rows(self):
        caches, replay = fixture()
        for change in [
            {"true": np.full(210, np.nan)},
            {"pg": np.full(210, -1.0)},
            {"n_tables": np.full(210, 1.5)},
        ]:
            with self.subTest(change=list(change)), self.assertRaises(ValueError):
                build_tables(
                    {**caches, "a": {**caches["a"], **change}},
                    replay,
                    ["a", "b"],
                    ["v"],
                )
        with self.assertRaises(ValueError):
            build_tables(caches, replay, ["a", "v"], ["v"])
        with self.assertRaises(ValueError):
            build_tables(caches, {"prefixes": {}}, ["a", "b"], ["v"])
        for weights, probabilities in [
            (np.array([-1.0, 1.0]), [0.5]),
            (np.ones(2), [2.0]),
        ]:
            with self.assertRaises(ValueError):
                weighted_quantile(np.arange(2.0), weights, probabilities)
        replay["prefixes"][1000]["databases"]["a"]["mass"] = np.full(210, -1.0)
        with self.assertRaises(ValueError):
            build_tables(caches, replay, ["a", "b"], ["v"])

    def test_exposure_reports_global_and_within_database_mass_shares(self):
        caches, replay = fixture()
        result = build_tables(caches, replay, ["a", "b"], ["v"])
        r = row(result["exposure"], database="a", axis="all", step=4000)
        self.assertAlmostEqual(r["nominal_mass_share"], 1)
        self.assertAlmostEqual(r["nominal_mass_global_share"], 0.5)
        self.assertAlmostEqual(r["combined_presentations_global_share"], 0.5)

    def test_zero_source_nonzero_validation_is_not_smoothed(self):
        caches, replay = fixture()
        caches = {
            db: {**cache, "true": np.zeros(210)} if db in ["a", "b"] else cache
            for db, cache in caches.items()
        }
        result = build_tables(caches, replay, ["a", "b"], ["v"])
        r = row(
            result["comparison"],
            step=4000,
            axis="cardinality",
            bucket="[10,100)",
            source_population="combined_presentations",
            validation_database="v",
        )
        self.assertEqual(r["ratio"], 0)
        self.assertEqual(r["difference"], -1)
        self.assertEqual(r["source_db_coverage"], 0)
        self.assertEqual(r["axis_tv"], 1)

    def test_inputs_remain_unchanged(self):
        import copy
        import pickle

        caches, replay = fixture()
        before = pickle.dumps(copy.deepcopy((caches, replay)))
        build_tables(caches, replay, ["a", "b"], ["v"])
        self.assertEqual(before, pickle.dumps((caches, replay)))


if __name__ == "__main__":
    unittest.main()
