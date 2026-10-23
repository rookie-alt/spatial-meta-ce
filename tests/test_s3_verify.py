"""Independent fixtures and corruption tests for the streaming S3 verifier."""

from __future__ import annotations

import copy
import importlib.util
import json
import os
from pathlib import Path
import unittest

import numpy as np

_PATH = Path(__file__).resolve().parents[1] / "scripts" / "s3_verify.py"
_SPEC = importlib.util.spec_from_file_location("s3_verify", _PATH)
verify = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(verify)
TRAIN = ["a", "b"]
VALIDATION = ["c", "d", "e"]
CARD = (
    "0",
    "(0,10)",
    "[10,100)",
    "[100,1000)",
    "[1000,10000)",
    "[10000,100000)",
    "[100000,inf)",
)
TABLE = ("1-3", "4", "5", "6+")
LAYERS = (
    [("all", "all")]
    + [("cardinality", c) for c in CARD]
    + [("n_tables", t) for t in TABLE]
    + [("cross", f"{c}|{t}") for c in CARD for t in TABLE]
)
ACTIVE = {
    ("all", "all"),
    ("cardinality", "[10,100)"),
    ("n_tables", "5"),
    ("cross", "[10,100)|5"),
}


def inputs() -> tuple:
    caches = {
        db: {
            "true": np.full(200, 10.0),
            "pg": np.full(200, 10.0),
            "n_tables": np.full(200, 5),
            "indices": np.arange(200),
            "query_hash": [f"hash{i}" for i in range(200)],
            "raw_rows": 200,
            "token_counts": np.tile([0, 0, 1, 0], (200, 1)),
        }
        for db in TRAIN + VALIDATION
    }
    states = {
        db: {
            "support": np.full(200, 2),
            "query": np.ones(200, dtype=int),
            "mass": np.full(200, 10.0),
        }
        for db in TRAIN
    }
    replay = {"prefixes": {4000: {"databases": states, "presentations": 1200}}}
    return caches, replay


def fixture(axis: str, bucket: str) -> dict:
    """Hand-calculated constant-residual tables, no verifier or producer helpers."""
    active = (axis, bucket) in ACTIVE
    rows, residuals, alignment = [], [], []
    specs = [(db, "pool_unique", 200, 200.0, 1, True) for db in TRAIN + VALIDATION]
    specs += [
        ("train_pooled", "pool_unique", 400, 400.0, 2, True),
        ("train_db_equal_pool", "db_equal_pool", 400, 1.0, 2, False),
        ("validation_db_equal_pool", "db_equal_pool", 600, 1.0, 3, False),
    ]
    specs += [
        (
            db,
            pop,
            400 if db == "train_pooled" else 200,
            (1200.0 if db == "train_pooled" else 600.0)
            if pop == "combined_presentations"
            else (4000.0 if db == "train_pooled" else 2000.0),
            2 if db == "train_pooled" else 1,
            False,
        )
        for pop in ("combined_presentations", "nominal_mass")
        for db in TRAIN + ["train_pooled"]
    ]
    if active:
        for db, pop, n, mass, coverage, linear in specs:
            row = dict(
                database=db,
                population=pop,
                axis=axis,
                bucket=bucket,
                condition="2,0,0,1,0",
                pg_bucket="[10,100)",
                n_join_column_tokens=0,
                n_fanout_tokens=0,
                n_table_tokens=1,
                n_filter_tokens=0,
                pool_unique=n,
                observed_unique=n,
                group_pool_unique=n,
                weight_mass=mass,
                group_weight_mass=mass,
                share=1.0,
                source_db_coverage=coverage,
            )
            rows.append(row)
            statistics = {
                f"r_{name}_{stat}": 0.0
                for name in ("ref", "fp32")
                for stat in ("mean", "median", "p10", "p90")
            }
            residuals.append(
                {
                    **row,
                    "status": "ok",
                    "quantile_method": "linear" if linear else "inverse_ecdf",
                    **statistics,
                    "positive_share": 0.0,
                    "negative_share": 0.0,
                    "zero_share": 1.0,
                }
            )
    for source, full_mass in (
        ("train_pooled_pool", 400.0),
        ("train_db_equal_pool", 1.0),
        ("combined_presentations", 1200.0),
        ("nominal_mass", 4000.0),
    ):
        for target in VALIDATION + ["validation_db_equal_pool"]:
            equal = target == "validation_db_equal_pool"
            nval, mval = (600, 1.0) if equal else (200, 200.0)
            count = 400 if active else 0
            val = nval if active else 0
            mass = full_mass if active else 0.0
            vmass = mval if active else 0.0
            coverage = {
                db: 200 if active else 0 for db in VALIDATION if equal or db == target
            }
            alignment.append(
                dict(
                    axis=axis,
                    bucket=bucket,
                    source_population=source,
                    validation_database=target,
                    status="ok" if active else "no_common_support",
                    common_conditions=1 if active else 0,
                    source_pool_unique=count,
                    validation_pool_unique=val,
                    source_observed_unique=count,
                    source_weight_mass=mass,
                    validation_weight_mass=vmass,
                    source_common_unique=count,
                    validation_common_unique=val,
                    source_common_observed_unique=count,
                    source_common_weight_mass=mass,
                    validation_common_weight_mass=vmass,
                    source_unique_coverage=1.0 if active else "undefined",
                    validation_unique_coverage=1.0 if active else "undefined",
                    source_mass_coverage=1.0 if active else "undefined",
                    validation_mass_coverage=1.0 if active else "undefined",
                    source_db_coverage=2 if active else 0,
                    validation_db_coverage=(3 if equal else 1) if active else 0,
                    validation_db_coverage_json=json.dumps(coverage),
                    **{
                        k: 0.0 if active else None
                        for k in (
                            "full_train_mean",
                            "full_validation_mean",
                            "full_mean_difference",
                            "common_train_mean",
                            "common_validation_mean",
                            "common_mean_difference",
                            "d_signed",
                            "d_absolute",
                        )
                    },
                )
            )
    return dict(
        axis=axis,
        bucket=bucket,
        conditions=rows,
        residuals=residuals,
        alignment=alignment,
    )


class VerificationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.caches, self.replay = inputs()

    def checker(self) -> object:
        return verify.Verification(self.caches, self.replay, TRAIN, VALIDATION)

    def test_full_stream_independent_fixture_and_immutable_inputs(self) -> None:
        before = copy.deepcopy((self.caches, self.replay))
        checker = self.checker()
        for layer in reversed(LAYERS):
            chunk = fixture(*layer)
            original = copy.deepcopy(chunk)
            checker.check(chunk)
            self.assertEqual(chunk, original)
        result = checker.finish()
        self.assertEqual(result["groups"], 40)
        self.assertEqual(
            result["rows"], {"conditions": 56, "residuals": 56, "alignment": 640}
        )
        for db in self.caches:
            for key, value in self.caches[db].items():
                if isinstance(value, np.ndarray):
                    np.testing.assert_array_equal(value, before[0][db][key])
                else:
                    self.assertEqual(value, before[0][db][key])
        for db in TRAIN:
            for key in ("support", "query", "mass"):
                np.testing.assert_array_equal(
                    self.replay["prefixes"][4000]["databases"][db][key],
                    before[1]["prefixes"][4000]["databases"][db][key],
                )

    def test_every_field_corruption_rejected(self) -> None:
        base = fixture("all", "all")
        for table in ("conditions", "residuals", "alignment"):
            for field, value in base[table][0].items():
                with self.subTest(table=table, field=field):
                    chunk = copy.deepcopy(base)
                    replacement = (
                        value + 1 if isinstance(value, (int, float)) else "corrupt"
                    )
                    chunk[table][0][field] = replacement
                    with self.assertRaisesRegex(ValueError, table):
                        self.checker().check(chunk)

    def test_missing_duplicate_extra_columns_nan_and_integer_types(self) -> None:
        for table in ("conditions", "residuals", "alignment"):
            for mode in ("missing", "duplicate", "extra", "nan", "float_count"):
                with self.subTest(table=table, mode=mode):
                    chunk = fixture("all", "all")
                    if mode == "missing":
                        chunk[table].pop()
                    elif mode == "duplicate":
                        chunk[table].append(copy.deepcopy(chunk[table][0]))
                    elif mode == "extra":
                        chunk[table][0]["extra"] = 1
                    elif mode == "nan":
                        chunk[table][0][
                            "source_weight_mass"
                            if table == "alignment"
                            else "weight_mass"
                        ] = float("nan")
                    else:
                        field = (
                            "common_conditions"
                            if table == "alignment"
                            else "pool_unique"
                        )
                        chunk[table][0][field] = float(chunk[table][0][field])
                    with self.assertRaises(ValueError):
                        self.checker().check(chunk)

    def test_layers_guard_and_json_structural_comparison(self) -> None:
        checker = self.checker()
        chunk = fixture("all", "all")
        for row in chunk["alignment"]:
            row["validation_db_coverage_json"] = json.dumps(
                dict(
                    reversed(
                        list(json.loads(row["validation_db_coverage_json"]).items())
                    )
                ),
                indent=2,
            )
        checker.check(chunk)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            checker.check(chunk)
        with self.assertRaisesRegex(ValueError, "40"):
            checker.finish()
        for change in ({"axis": "bad"}, {"bucket": "bad"}, {"extra": 1}):
            with self.assertRaises(ValueError):
                self.checker().check({**fixture("all", "all"), **change})

    def test_input_guards(self) -> None:
        mutations = [
            ("pg", np.full(200, np.nan)),
            ("true", np.full(200, -1)),
            ("n_tables", np.full(200, 1.5)),
            ("token_counts", np.ones((200, 4), dtype=float)),
            ("token_counts", np.full((200, 4), -1)),
            ("indices", np.zeros(200, dtype=int)),
            ("raw_rows", 199),
            ("query_hash", ["duplicate"] * 200),
        ]
        for key, value in mutations:
            with self.subTest(key=key):
                caches = copy.deepcopy(self.caches)
                caches["a"][key] = value
                with self.assertRaises(ValueError):
                    verify.Verification(caches, self.replay, TRAIN, VALIDATION)
        with self.assertRaises(ValueError):
            verify.Verification(self.caches, self.replay, ["a"], VALIDATION)
        with self.assertRaises(ValueError):
            verify.Verification(self.caches, {}, TRAIN, VALIDATION)
        for key, value in (
            ("support", np.ones(200)),
            ("mass", np.zeros(200)),
            ("mass", np.full(200, np.nan)),
            ("query", np.full(200, -1)),
        ):
            replay = copy.deepcopy(self.replay)
            replay["prefixes"][4000]["databases"]["a"][key] = value
            with self.assertRaises(ValueError):
                verify.Verification(self.caches, replay, TRAIN, VALIDATION)
        replay = copy.deepcopy(self.replay)
        replay["prefixes"][4000]["presentations"] += 1
        with self.assertRaisesRegex(ValueError, "conservation"):
            verify.Verification(self.caches, replay, TRAIN, VALIDATION)

    def test_pg_boundaries_and_local_identity(self) -> None:
        values = np.array(
            [
                0,
                1,
                np.nextafter(10.0, 0),
                10,
                99,
                100,
                999,
                1000,
                9999,
                10000,
                99999,
                100000,
            ]
        )
        np.testing.assert_array_equal(
            verify._classification(values), [0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6]
        )
        # Same hashes in different databases must not collapse the combined pool.
        self.assertEqual(len(self.checker().data["a"]["ref"]), 200)

    def test_independent_linear_and_weighted_quantiles_direction_reference(
        self,
    ) -> None:
        values = np.array([-3.0, 1.0, 2.0, 9.0])
        linear = verify._statistics(values, np.ones(4), True)
        self.assertAlmostEqual(linear["median"], 1.5)
        self.assertAlmostEqual(linear["p10"], -1.8)
        self.assertAlmostEqual(linear["p90"], 6.9)
        weighted = verify._statistics(values, np.array([1.0, 0.0, 8.0, 1.0]), False)
        self.assertEqual(
            weighted, {"mean": 2.2, "median": 2.0, "p10": -3.0, "p90": 2.0}
        )
        part = (np.full(200, 1e-12), np.zeros(200), np.ones(200))
        cell = verify._cell([part], True)
        self.assertEqual(
            cell["directions"],
            {"positive_share": 1.0, "negative_share": 0.0, "zero_share": 0.0},
        )
        sparse = verify._cell([(part[0][:199], part[1][:199], part[2][:199])], True)
        self.assertIsNone(sparse["stats"])

    def test_alignment_standardization_cancellation_and_macro_guard(self) -> None:
        checker = self.checker()
        keys = [(2, 0, 0, 1, 0), (2, 1, 0, 1, 0)]

        def summary(n: int, mass: float, mean: float, observed: int = 200) -> dict:
            return dict(unique=n, observed=observed, mass=mass, sum=mass * mean)

        source = {keys[0]: summary(200, 100.0, 0.0), keys[1]: summary(200, 900.0, 0.0)}
        target = {keys[0]: summary(200, 100.0, -1.0), keys[1]: summary(200, 300.0, 1.0)}
        views = {
            (db, "pool_unique"): ({key: summary(200, 200.0, 0.0) for key in keys}, True)
            for db in VALIDATION
        }
        groups = {db: {key: np.arange(200) for key in keys} for db in TRAIN}
        row = checker._alignment(
            "all", "all", "nominal_mass", "c", source, target, views, groups
        )
        self.assertEqual(row["d_signed"], 0.5)
        self.assertEqual(row["d_absolute"], 1.0)
        self.assertEqual(row["common_mean_difference"], 0.5)
        # Full means remain available while all three target libraries are required.
        views[("e", "pool_unique")] = (
            {key: summary(99, 99.0, 0.0, 99) for key in keys},
            True,
        )
        row = checker._alignment(
            "all",
            "all",
            "nominal_mass",
            "validation_db_equal_pool",
            source,
            target,
            views,
            groups,
        )
        self.assertEqual(row["status"], "incomplete_validation_support")
        self.assertIsNone(row["d_signed"])
        self.assertIsNone(row["common_train_mean"])
        self.assertEqual(row["full_validation_mean"], 0.5)
        # 199 sampled unique cannot borrow repeated presentations to pass M.
        source = {key: summary(200, 1000.0, 0.0, 199) for key in keys}
        row = checker._alignment(
            "all", "all", "combined_presentations", "c", source, target, views, groups
        )
        self.assertEqual(row["status"], "no_common_support")
        self.assertEqual(row["source_common_unique"], 0)
        self.assertEqual(row["full_train_mean"], 0.0)

    def test_pool_database_coverage_includes_unsampled_pool_members(self) -> None:
        part = (np.zeros(200), np.zeros(200), np.zeros(200))
        cell = verify._cell([part], False)
        self.assertEqual(cell["coverage"], 1)
        self.assertEqual(cell["unique"], 200)
        self.assertEqual(cell["observed"], 0)
        self.assertIsNone(cell["stats"])

    def test_mass_accumulation_tolerance_and_row_tolerance(self) -> None:
        weights = np.full(1_000_000, 0.0001)
        sequential = float(
            np.bincount(np.zeros(len(weights), dtype=int), weights=weights)[0]
        )
        independent = float(weights.sum())
        self.assertGreater(abs(sequential - independent), verify.ROW_ATOL)
        verify._compare(sequential, independent, "row.weight_mass")
        verify._compare(sequential, independent, "row.source_common_weight_mass")
        with self.assertRaises(ValueError):
            verify._compare(independent + 2e-8, independent, "row.group_weight_mass")
        with self.assertRaises(ValueError):
            verify._compare(0.5 + 2e-10, 0.5, "row.share")

    def test_producer_integration_mixed_labels_and_sparse_macro(self) -> None:
        path = Path(
            os.environ.get(
                "S3_PRODUCER_PATH", str(_PATH.with_name("s3_conditional_analysis.py"))
            )
        )
        if not path.is_file():
            self.skipTest("producer not installed in this isolated worktree")
        spec = importlib.util.spec_from_file_location("s3_producer_integration", path)
        producer = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(producer)
        rng = np.random.default_rng(753)
        for variant in range(3):
            caches, states = {}, {}
            for index, db in enumerate(TRAIN + VALIDATION):
                n = 1200 + index * 137
                condition = rng.integers(0, 4, n)
                pg = np.choose(condition, [0.0, 10.0, 100000.0, 1.000000000001])
                true = np.choose(
                    rng.integers(0, 7, n),
                    [0.0, 9.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0],
                )
                if variant == 1:
                    true = np.full(n, 10.0)
                if variant == 2 and db == "e":
                    condition = np.full(n, 8)
                    pg = np.full(n, 10.0)
                caches[db] = dict(
                    true=true,
                    pg=pg,
                    n_tables=rng.integers(1, 9, n),
                    indices=np.arange(n),
                    raw_rows=n,
                    query_hash=[f"h{i}" for i in range(n)],
                    token_counts=np.column_stack(
                        (
                            condition,
                            condition % 2,
                            np.ones(n, dtype=int),
                            np.zeros(n, dtype=int),
                        )
                    ),
                )
                if db in TRAIN:
                    support = rng.integers(0, 4, n)
                    query = rng.integers(0, 3, n)
                    if variant == 1 and db == "b":
                        support = np.where(condition == 0, 0, support)
                        query = np.where(condition == 0, 0, query)
                    states[db] = dict(
                        support=support,
                        query=query,
                        mass=(support + query) * rng.uniform(0.5, 2.0, n),
                    )
            mass_total = sum(float(s["mass"].sum()) for s in states.values())
            states = {
                db: {**state, "mass": state["mass"] * (4000 / mass_total)}
                for db, state in states.items()
            }
            presentations = sum(
                int((s["support"] + s["query"]).sum()) for s in states.values()
            )
            replay = {
                "prefixes": {
                    4000: {"databases": states, "presentations": presentations}
                }
            }
            checker = verify.Verification(caches, replay, TRAIN, VALIDATION)
            for chunk in producer.iter_analysis(caches, replay, TRAIN, VALIDATION):
                checker.check(chunk)
            self.assertEqual(checker.finish()["groups"], 40)


if __name__ == "__main__":
    unittest.main()
