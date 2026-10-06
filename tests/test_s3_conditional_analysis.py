"""Synthetic protocol tests: no file access, models, or training."""

from __future__ import annotations

import copy
import unittest

import numpy as np

from scripts.s3_conditional_analysis import _group, iter_analysis


def fixture(sizes=(400, 400, 400, 400, 400)):
    names = ["t1", "t2", "v1", "v2", "v3"]
    caches = {}
    for name, n in zip(names, sizes):
        caches[name] = {
            "true": np.full(n, 10.0),
            "pg": np.full(n, 10.0),
            "n_tables": np.ones(n, dtype=np.int64),
            "indices": np.arange(n),
            "query_hash": [str(i) for i in range(n)],
            "raw_rows": n,
            "token_counts": np.zeros((n, 4), dtype=np.int64),
        }
    replay = {
        "prefixes": {
            4000: {
                "databases": {
                    db: {
                        "support": np.ones(sizes[i], dtype=np.int64),
                        "query": np.ones(sizes[i], dtype=np.int64),
                        "mass": np.ones(sizes[i], dtype=np.float64),
                    }
                    for i, db in enumerate(names[:2])
                }
            }
        }
    }
    return caches, replay, names[:2], names[2:]


def all_chunk(args):
    return next(iter_analysis(*args))


def row(chunk, kind, db, population, condition="2,0,0,0,0"):
    return next(
        r
        for r in chunk[kind]
        if r["database"] == db
        and r["population"] == population
        and r["condition"] == condition
    )


def alignment(chunk, source="train_pooled_pool", target="v1"):
    return next(
        r
        for r in chunk["alignment"]
        if r["source_population"] == source and r["validation_database"] == target
    )


class ConditionalAnalysisTests(unittest.TestCase):
    def test_million_fractional_weights_match_independent_fp64_sum(self):
        n = 1_000_000
        weights = np.full(n, 0.001, dtype=np.float64)
        ref = np.ones(n, dtype=np.float64)
        data = {"t": {"n": n, "ref": ref, "fp32": ref}}
        selected = {"t": np.arange(n)}
        ids = {"t": np.ones(n, dtype=np.int64)}
        replay = {"t": {"nominal_mass": weights, "combined_presentations": np.ones(n)}}
        group = _group(data, selected, ids, ("t",), "nominal_mass", replay, 3)
        expected = np.sum(weights, dtype=np.float64)
        self.assertLessEqual(abs(group["mass"][1] - expected), 1e-8)
        self.assertLessEqual(abs(group["sums"][1] - np.sum(weights * ref)), 1e-8)
        np.testing.assert_array_equal(group["mass"][[0, 2]], [0, 0])
        np.testing.assert_array_equal(group["sums"][[0, 2]], [0, 0])
        np.testing.assert_array_equal(group["counts"], [0, n, 0])

    def test_group_segment_sum_handles_empty_and_unsorted_missing_keys(self):
        data = {
            "t": {"n": 4, "ref": np.array([1.0, -2.0, 3.0, -4.0]), "fp32": np.zeros(4)}
        }
        replay = {
            "t": {
                "nominal_mass": np.array([0.1, 0.2, 0.3, 0.4]),
                "combined_presentations": np.ones(4),
            }
        }
        ids = {"t": np.array([3, 1, 3, 1])}
        group = _group(
            data, {"t": np.arange(4)}, ids, ("t",), "nominal_mass", replay, 5
        )
        np.testing.assert_allclose(group["mass"], [0, 0.6, 0, 0.4, 0])
        np.testing.assert_allclose(group["sums"], [0, -2.0, 0, 1.0, 0])
        for nkeys in (0, 5):
            empty = _group(
                data,
                {"t": np.empty(0, dtype=int)},
                {"t": np.empty(0, dtype=int)},
                ("t",),
                "nominal_mass",
                replay,
                nkeys,
            )
            for key in ("mass", "sums", "counts", "observed"):
                np.testing.assert_array_equal(empty[key], np.zeros(nkeys))

    def test_pg_boundaries_and_complete_union(self):
        args = fixture((7, 7, 7, 7, 7))
        args[0]["t1"]["pg"] = np.array([0, 1, 10, 100, 1000, 10000, 100000.0])
        args[0]["v1"]["token_counts"][:, 0] = 8
        chunk = all_chunk(args)
        rows = [r for r in chunk["conditions"] if r["database"] == "v2"]
        self.assertEqual(len(rows), 8)
        self.assertEqual(
            [r["condition"] for r in rows],
            [
                "0,0,0,0,0",
                "1,0,0,0,0",
                "2,0,0,0,0",
                "2,8,0,0,0",
                "3,0,0,0,0",
                "4,0,0,0,0",
                "5,0,0,0,0",
                "6,0,0,0,0",
            ],
        )
        self.assertEqual(rows[0]["pool_unique"], 0)
        self.assertEqual(rows[-1]["pg_bucket"], "[100000,inf)")

    def test_199_200_threshold_and_exposure_observed(self):
        args = fixture((199, 200, 200, 200, 200))
        args[1]["prefixes"][4000]["databases"]["t2"]["support"][:1] = 0
        args[1]["prefixes"][4000]["databases"]["t2"]["query"][:1] = 0
        chunk = all_chunk(args)
        self.assertEqual(
            row(chunk, "residuals", "t1", "pool_unique")["status"], "sparse"
        )
        self.assertEqual(row(chunk, "residuals", "t2", "pool_unique")["status"], "ok")
        exposure = row(chunk, "residuals", "t2", "combined_presentations")
        self.assertEqual(exposure["observed_unique"], 199)
        self.assertIsNone(exposure["r_ref_mean"])

    def test_empty_common_and_noncommon_are_retained(self):
        args = fixture()
        for db in args[3]:
            args[0][db]["token_counts"][:, 0] = 1
        chunk = all_chunk(args)
        result = alignment(chunk)
        self.assertEqual(result["status"], "no_common_support")
        self.assertIsNone(result["d_signed"])
        self.assertEqual(result["full_train_mean"], 0)
        self.assertEqual(
            len(
                [
                    r
                    for r in chunk["conditions"]
                    if r["database"] == "train_pooled"
                    and r["population"] == "pool_unique"
                ]
            ),
            2,
        )

    def test_equal_weights_use_full_database_denominators(self):
        args = fixture((200, 600, 400, 400, 400))
        args[0]["t1"]["true"][:] = 100
        args[0]["t2"]["token_counts"][200:, 0] = 1
        chunk = all_chunk(args)
        pooled = row(chunk, "residuals", "train_pooled", "pool_unique")
        equal = row(chunk, "residuals", "train_db_equal_pool", "db_equal_pool")
        self.assertAlmostEqual(pooled["r_ref_mean"], np.log(10) / 2)
        self.assertAlmostEqual(equal["r_ref_mean"], np.log(10) * 3 / 4)
        self.assertAlmostEqual(equal["weight_mass"], 2 / 3)
        self.assertEqual(pooled["quantile_method"], "linear")
        self.assertEqual(equal["quantile_method"], "inverse_ecdf")

    def test_signed_cancellation_and_label_independent_conditions_weights(self):
        args = fixture()
        for cache in args[0].values():
            cache["token_counts"][200:, 0] = 1
        for db in args[3]:
            args[0][db]["true"][:200] = 100
            args[0][db]["true"][200:] = 1
        before = all_chunk(args)
        result = alignment(before)
        self.assertAlmostEqual(result["d_signed"], 0, places=14)
        self.assertAlmostEqual(result["d_absolute"], np.log(10))
        args[0]["v1"]["true"][:] = 10000
        after = all_chunk(args)
        self.assertEqual(before["conditions"], after["conditions"])
        self.assertEqual(
            alignment(before)["validation_common_weight_mass"],
            alignment(after)["validation_common_weight_mass"],
        )

    def test_missing_validation_guard(self):
        args = fixture()
        args[0]["v3"]["token_counts"][:, 0] = 1
        result = alignment(all_chunk(args), target="validation_db_equal_pool")
        self.assertEqual(result["status"], "incomplete_validation_support")
        self.assertIsNone(result["common_train_mean"])
        self.assertIsNotNone(result["full_validation_mean"])
        self.assertEqual(result["validation_db_coverage"], 2)

    def test_zero_exposure_preserves_pool_database_coverage(self):
        args = fixture()
        state = args[1]["prefixes"][4000]["databases"]["t1"]
        for values in state.values():
            values[:] = 0
        chunk = all_chunk(args)
        actual = row(chunk, "conditions", "t1", "nominal_mass")
        self.assertEqual(actual["source_db_coverage"], 1)
        self.assertEqual(actual["observed_unique"], 0)
        self.assertEqual(actual["share"], "undefined")
        self.assertEqual(alignment(chunk, "nominal_mass")["source_db_coverage"], 2)

    def test_quantiles_directions_and_fp32_definition(self):
        args = fixture((200, 200, 200, 200, 200))
        true = np.exp(np.linspace(0, 9, 200))
        args[0]["t1"]["true"] = true
        state = args[1]["prefixes"][4000]["databases"]["t1"]
        state["support"] = np.arange(1, 201)
        state["query"][:] = 0
        state["mass"] = np.arange(1, 201, dtype=np.float64) / 20100
        chunk = all_chunk(args)
        linear = row(chunk, "residuals", "t1", "pool_unique")
        weighted = row(chunk, "residuals", "t1", "combined_presentations")
        values = np.log(true) - np.log(10)
        weights = state["support"]
        self.assertAlmostEqual(linear["r_ref_p10"], np.quantile(values, 0.1))
        index = np.searchsorted(np.cumsum(weights), 0.5 * weights.sum())
        self.assertAlmostEqual(weighted["r_ref_median"], values[index])
        self.assertAlmostEqual(
            weighted["positive_share"], weights[values > 0].sum() / weights.sum()
        )
        fp32 = np.log(true).astype(np.float32).astype(np.float64) - float(
            np.float32(np.log(10))
        )
        self.assertAlmostEqual(
            weighted["r_fp32_mean"], np.average(fp32, weights=weights)
        )

    def test_full_means_include_noncommon_and_mass_coverage(self):
        args = fixture()
        for db in args[2]:
            args[0][db]["token_counts"][200:, 0] = 2
            args[0][db]["true"][200:] = 100
        result = alignment(all_chunk(args))
        self.assertEqual(result["status"], "ok")
        self.assertAlmostEqual(result["full_train_mean"], np.log(10) / 2)
        self.assertAlmostEqual(result["common_train_mean"], 0)
        self.assertEqual(result["source_unique_coverage"], 0.5)
        self.assertEqual(result["source_mass_coverage"], 0.5)
        self.assertEqual(result["d_absolute"], 0)

    def test_validation_guard_requires_one_sufficient_individual_cell(self):
        args = fixture()
        for cache in args[0].values():
            cache["token_counts"][200:, 0] = 1
        args[0]["v3"]["token_counts"][100:300, 0] = 1
        args[0]["v3"]["token_counts"][300:, 0] = 0
        # v3 has 200 each now; change to 100+100 common and 200 unmatched.
        args[0]["v3"]["token_counts"][200:, 0] = 2
        args[0]["v3"]["token_counts"][:100, 0] = 0
        args[0]["v3"]["token_counts"][100:200, 0] = 1
        result = alignment(all_chunk(args), target="validation_db_equal_pool")
        self.assertEqual(
            result["validation_db_coverage_json"], '{"v1":400,"v2":400,"v3":200}'
        )
        self.assertEqual(result["status"], "incomplete_validation_support")

    def test_rejects_invalid_input_and_uses_only_final_prefix(self):
        for key, bad in (
            ("true", np.full(400, np.nan)),
            ("token_counts", np.full((400, 4), -1)),
            ("n_tables", np.full(400, 1.5)),
        ):
            args = fixture()
            args[0]["t1"][key] = bad
            with self.assertRaises(ValueError):
                all_chunk(args)
        args = fixture()
        args[1]["prefixes"][4000]["databases"]["t1"]["support"][0] = -1
        with self.assertRaises(ValueError):
            all_chunk(args)
        args = fixture()
        with self.assertRaises(ValueError):
            all_chunk((args[0], args[1], [], args[3]))
        args[1]["prefixes"][1000] = {"databases": {}}
        self.assertEqual(alignment(all_chunk(args))["status"], "ok")

    def test_streams_40_layers_without_mutating_inputs(self):
        args = fixture()
        original = copy.deepcopy(args)
        count = 0
        for chunk in iter_analysis(*args):
            self.assertEqual(
                set(chunk), {"axis", "bucket", "conditions", "residuals", "alignment"}
            )
            count += 1
        self.assertEqual(count, 40)
        for db in args[0]:
            for key, value in args[0][db].items():
                if isinstance(value, np.ndarray):
                    np.testing.assert_array_equal(value, original[0][db][key])
                else:
                    self.assertEqual(value, original[0][db][key])
        for db in args[2]:
            for key in ("support", "query", "mass"):
                np.testing.assert_array_equal(
                    args[1]["prefixes"][4000]["databases"][db][key],
                    original[1]["prefixes"][4000]["databases"][db][key],
                )


if __name__ == "__main__":
    unittest.main()
