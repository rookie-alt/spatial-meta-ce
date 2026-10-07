"""Synthetic tests only; no experimental results or source caches are read."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts import s7_source_diagnosis as core


class S7Tests(unittest.TestCase):
    def fixture(self, n=400):
        y = np.arange(n, dtype=float) % 6
        return {
            "query_hash": [f"q{i}" for i in range(n)],
            "true": np.exp(y),
            "pg": np.exp(y + 0.5),
            "n_tables": np.full(n, 4, dtype=np.int64),
            "A": y + np.linspace(-2, 2, n),
            "B": y + np.linspace(-1, 3, n),
        }

    def test_standard_q_floor_and_raw_error_are_separate(self):
        result = core.metrics(np.array([-3.0, 2.0]), np.array([0.0, np.exp(2)]))
        self.assertEqual(result["q95"], 1.0)
        self.assertEqual(result["raw_log_mae"], 1.5)
        self.assertEqual(result["n"], 2)

    def test_sparse_group_only_counts(self):
        self.assertEqual(
            core.pair(self.fixture(5), np.ones(5, dtype=bool)),
            {"n": 5, "status": "below_min_cell_n"},
        )

    def test_tail_masks_partition_and_ties_not_forced_five_percent(self):
        d = self.fixture()
        tails = core.tail_masks(d, np.ones(400, dtype=bool))
        self.assertTrue(np.all(sum(tails.values()) == 1))
        report = core.legalacts_report(d)
        self.assertEqual(sum(r["n"] for r in report["tail_groups"]), 400)
        d["A"][:] = 1
        d["B"][:] = 1
        d["true"][:] = 1
        tails = core.tail_masks(d, np.ones(400, dtype=bool))
        self.assertEqual(int(tails["persistent"].sum()), 400)

    def test_legalacts_all_and_four_plus_denominators(self):
        d = self.fixture()
        d["n_tables"][:20] = 2
        r = core.legalacts_report(d)
        self.assertEqual(r["all"]["n"], 400)
        self.assertEqual(r["four_plus"]["n"], 380)
        self.assertEqual(sum(x["n"] for x in r["tail_groups"]), 380)

    def test_composition_small_tail_counts_no_percentile(self):
        d = self.fixture()
        r = core.composition(d, np.arange(400) < 10)
        self.assertEqual(sum(r["cardinality_counts"].values()), 10)
        self.assertEqual(sum(r["b_error_direction_counts"].values()), 10)
        self.assertEqual(r["stats"]["status"], "below_min_cell_n")

    def test_nonfinite_negative_shape_and_overflow_rejected(self):
        for pred, true in [
            ([np.nan], [1]),
            ([1], [-1]),
            ([800], [1]),
            ([1, 2], [1]),
            ([1], [np.inf]),
        ]:
            with self.subTest(pred=pred, true=true), self.assertRaises(ValueError):
                core.metrics(np.array(pred), np.array(true))

    def test_pair_identity_and_duplicate_rejected(self):
        for defect in ["duplicate", "tables", "shape", "infinite"]:
            d = self.fixture()
            if defect == "duplicate":
                d["query_hash"][1] = d["query_hash"][0]
            if defect == "tables":
                d["n_tables"][0] = 0
            if defect == "shape":
                d["B"] = d["B"][:1]
            if defect == "infinite":
                d["pg"][0] = np.inf
            with self.subTest(defect=defect), self.assertRaises(ValueError):
                core.validate(d)

    def test_sampling_identity_only_order_invariant(self):
        hashes = ["c", "b", "a", "d", "e"]
        chosen = core.deterministic_sample(
            "accidents", "seen", hashes, [0, 1, 2, 3, 4], 3
        )
        reverse = hashes[::-1]
        chosen2 = core.deterministic_sample(
            "accidents", "seen", reverse, [0, 1, 2, 3, 4], 3
        )
        self.assertEqual([hashes[i] for i in chosen], [reverse[i] for i in chosen2])
        self.assertEqual(len(chosen), 3)
        self.assertEqual(len(set(chosen)), 3)

    def test_sampling_empty_full_and_target_guard(self):
        self.assertEqual(
            core.deterministic_sample("accidents", "unseen", ["x"], [], 2), []
        )
        self.assertEqual(
            core.deterministic_sample("accidents", "seen", ["x"], [0], 2), [0]
        )
        for db in ["stats", "imdb", "../accidents", "chembl"]:
            with self.assertRaises(ValueError):
                core.deterministic_sample(db, "seen", ["x"], [0], 1)
        with self.assertRaises(ValueError):
            core.deterministic_sample("accidents", "seen", ["x"], [0], 0)

    def test_seen_membership_from_both_roles(self):
        roles = {"support": np.array([1, 0, 0]), "query": np.array([0, 1, 0])}
        seen, unseen = core.memberships(roles)
        np.testing.assert_array_equal(seen, [0, 1])
        np.testing.assert_array_equal(unseen, [2])
        with self.assertRaises(ValueError):
            core.memberships({"support": np.array([-1]), "query": np.array([0])})

    def test_q95_boundary_neighborhood_tie_invariant(self):
        d = self.fixture()
        r = core.legalacts_report(d)
        self.assertEqual(r["q95_neighborhood"]["n"], 8)
        np.testing.assert_equal(r["four_plus"]["n"], 400)

    def test_empty_cell_counts_no_missing_queries(self):
        d = self.fixture()
        r = core.legalacts_report(d)
        self.assertTrue(any(x["n"] == 0 for x in r["strata"]))
        card_cells = [x for x in r["strata"] if x["axis"] == "true"]
        self.assertEqual(sum(x["n"] for x in card_cells), 400)

    def test_json_roundtrip_and_nan_forbidden(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "result.json"
            core.write_json(p, {"n": 1})
            self.assertTrue(p.exists())
            with self.assertRaises(ValueError):
                core.write_json(p, {"x": np.nan})


class IndependentS7Tests(unittest.TestCase):
    def test_manual_percentile_and_metric_reference(self):
        from scripts.s7_independent_verify import metric_values, percentile

        self.assertEqual(percentile([4, 1, 3, 2], 50), 2.5)
        with self.assertRaises(ValueError):
            percentile([], 95)
        d = S7Tests().fixture()
        a = metric_values(d["A"], d["true"])
        b = core.metrics(d["A"], d["true"])
        for k in a:
            self.assertAlmostEqual(a[k], b[k], places=10)

    def test_independent_pair_rejects_tampered_statistics(self):
        from scripts.s7_independent_verify import pair_check

        d = S7Tests().fixture()
        mask = np.ones(400, dtype=bool)
        r = core.pair(d, mask)
        pair_check(r, d, mask)
        r["A"]["q95"] += 0.1
        with self.assertRaises(ValueError):
            pair_check(r, d, mask)

    def test_endpoint_loader_identity_values_and_missing_rows(self):
        import json

        from scripts.run_s7_diagnosis import endpoint_data

        c = {
            "legalacts": {
                "indices": np.array([0, 1]),
                "query_hash": ["x", "y"],
                "true": np.array([10.0, 0.0]),
                "pg": np.array([1.0, 0.0]),
                "n_tables": np.array([4, 4]),
            }
        }
        base = {
            "database": "legalacts",
            "n_tables": 4,
            "log_pred": 1.0,
            "pg_reference_log": 0.0,
            "pg_input_log": 0.0,
        }
        rows = [
            {**base, "query_hash": "x", "true": 10.0},
            {**base, "query_hash": "y", "true": 0.0},
        ]
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "rows.jsonl"

            def put(values):
                p.write_text("\n".join(json.dumps(v) for v in values))

            put(rows)
            np.testing.assert_array_equal(endpoint_data(p, c)["legalacts"], [1.0, 1.0])
            for values in [
                rows[:1],
                rows + rows[:1],
                [{**rows[0], "database": "imdb"}, rows[1]],
                [{**rows[0], "true": 11.0}, rows[1]],
                [{**rows[0], "pg_input_log": 0.1}, rows[1]],
            ]:
                put(values)
                with self.assertRaises(ValueError):
                    endpoint_data(p, c)


class S7VerificationIntegrationTests(unittest.TestCase):
    def make(self, root):
        import json

        from scripts.run_s7_diagnosis import cohort_report
        from scripts.s2_source_replay import digest

        data = S7Tests().fixture()
        chosen = core.deterministic_sample(
            "accidents", "seen", data["query_hash"], range(400), 2048
        )
        caches = {
            db: {k: v for k, v in data.items() if k not in ("A", "B")}
            for db in ("accidents", "chembl", "legalacts", "telstra")
        }
        for c in caches.values():
            c["indices"] = np.arange(400)
        states = {
            "accidents": {
                "support": np.ones(400, dtype=np.int64),
                "query": np.zeros(400, dtype=np.int64),
            }
        }
        samples, predictions = [], []
        for i in chosen:
            r = {
                "database": "accidents",
                "cohort": "seen",
                "unique_position": i,
                "raw_index": i,
                "query_hash": data["query_hash"][i],
                "presentations": 1,
                **{k: float(data[k][i]) for k in ("true", "pg", "n_tables")},
            }
            samples.append(r)
            predictions.append(
                {
                    k: v
                    for k, v in r.items()
                    if k not in ("unique_position", "raw_index", "presentations")
                }
                | {a: float(data[a][i]) for a in ("A", "B")}
            )
        for name, rows in [
            ("sample-manifest.jsonl", samples),
            ("source-predictions.jsonl", predictions),
        ]:
            (root / name).write_text("".join(json.dumps(r) + "\n" for r in rows))
        plan = {
            "cohorts": [
                {
                    "database": "accidents",
                    "cohort": "seen",
                    "pool_unique": 400,
                    "sample_unique": 400,
                    "selected_identity_digest": digest(
                        [data["query_hash"][i] for i in chosen]
                    ),
                },
                {
                    "database": "accidents",
                    "cohort": "unseen",
                    "pool_unique": 0,
                    "sample_unique": 0,
                },
            ]
        }
        core.write_json(root / "sample-plan.json", plan)
        hist = root / "history"
        for arm, name in [("A", "A_mean"), ("B", "B_mean_tail10")]:
            (hist / name).mkdir(parents=True)
            with (hist / name / "endpoint_rows.jsonl").open("w") as stream:
                for db in ("chembl", "legalacts", "telstra"):
                    for i in range(400):
                        r = {
                            "database": db,
                            "query_hash": data["query_hash"][i],
                            "true": float(data["true"][i]),
                            "n_tables": 4,
                            "pg_reference_log": float(np.log(data["pg"][i])),
                            "log_pred": float(data[arm][i]),
                        }
                        stream.write(json.dumps(r) + "\n")
        gen = {
            "rows": [
                {
                    "database": "accidents",
                    "cohort": "seen",
                    "statistics": cohort_report(data),
                },
                {
                    "database": "accidents",
                    "cohort": "unseen",
                    "statistics": [
                        {"bucket": "all", "n": 0, "status": "below_min_cell_n"}
                    ],
                },
            ]
        }
        gen["rows"].extend(
            {
                "database": db,
                "cohort": "source_validation_full",
                "statistics": cohort_report(data),
            }
            for db in ("chembl", "legalacts", "telstra")
        )
        core.write_json(root / "generalization.json", gen)
        core.write_json(root / "legalacts.json", core.legalacts_report(data))
        return hist, caches, states

    def test_full_independent_verification_and_tampering(self):
        import json

        from scripts.s7_independent_verify import verify

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            hist, caches, states = self.make(root)
            result = verify(root, root, hist, caches, states)
            self.assertTrue(result["passed"])
            self.assertEqual(result["sample_queries"], 400)
            p = root / "source-predictions.jsonl"
            text = p.read_text()
            rows = [json.loads(x) for x in text.splitlines()]
            rows[0]["true"] += 1
            p.write_text("".join(json.dumps(r) + "\n" for r in rows))
            with self.assertRaisesRegex(ValueError, "true/PG/table"):
                verify(root, root, hist, caches, states)
            p.write_text(text)
            p2 = root / "generalization.json"
            gen = json.loads(p2.read_text())
            gen["rows"][0]["statistics"][0]["A"]["q95"] += 1
            core.write_json(p2, gen)
            with self.assertRaises(ValueError):
                verify(root, root, hist, caches, states)


class ArchiveOverviewTests(unittest.TestCase):
    def test_descriptive_overview_and_cohort_completeness(self):
        from copy import deepcopy

        from scripts.s7_archive_report import overview

        cell = {
            "bucket": "all",
            "status": "descriptive",
            "A": {"q95": 2.0},
            "B": {"q95": 1.0},
            "q95_relative_change_b_vs_a": -0.5,
        }
        rows = [
            {
                "database": db,
                "cohort": cohort,
                "statistics": [
                    deepcopy(cell),
                    {**deepcopy(cell), "bucket": "four_plus"},
                ],
            }
            for db, cohort in [
                ("accidents", "seen"),
                ("accidents", "unseen"),
                ("chembl", "source_validation_full"),
                ("legalacts", "source_validation_full"),
                ("telstra", "source_validation_full"),
            ]
        ]
        result = overview(rows, ["accidents"])
        self.assertEqual(
            result["within_database_ratios"][0][
                "median_within_database_q95_unseen_div_seen"
            ],
            1.0,
        )
        for malformed in [rows[:-1], rows + rows[:1]]:
            with self.assertRaisesRegex(ValueError, "cohort"):
                overview(malformed, ["accidents"])


if __name__ == "__main__":
    unittest.main()
