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
        import hashlib
        import json

        from scripts.run_s7_diagnosis import cohort_report
        from scripts.s2_source_replay import digest, replay

        data = S7Tests().fixture()
        train = list(core.TRAIN)
        validation = ("chembl", "legalacts", "telstra")
        caches = {
            db: {k: v for k, v in data.items() if k not in ("A", "B")}
            for db in (*train, *validation)
        }
        for c in caches.values():
            c["indices"] = np.arange(400)
        config = {
            "seed": 42,
            "steps": 160,
            "meta_batch": 1,
            "accumulation": 1,
            "query_size": 2,
            "support_sizes": [1, 2],
        }
        sampled = replay(caches, train, config, prefixes=(config["steps"],))
        snapshot = sampled["prefixes"][config["steps"]]
        states = snapshot["databases"]
        (root / "configs").mkdir()
        core.write_json(root / "configs/data.json", {"splits": {"train": train}})
        samples, predictions, entries, sources = [], [], [], {}
        for db in train:
            directory = root / "artifacts/cache/pretrain" / db
            directory.mkdir(parents=True)
            content_hash = digest({"database": db})
            core.write_json(directory / "manifest.json", {"content_hash": content_hash})
            sources[db] = {
                "manifest_hash": content_hash,
                "file_sha256": {
                    "manifest.json": hashlib.sha256(
                        (directory / "manifest.json").read_bytes()
                    ).hexdigest()
                },
            }
            seen, unseen = core.memberships(states[db])
            counts = states[db]["support"] + states[db]["query"]
            for cohort, candidates in (
                ("seen", seen.tolist()),
                ("unseen", unseen.tolist()),
            ):
                chosen = core.deterministic_sample(
                    db, cohort, data["query_hash"], candidates, 2048
                )
                masks = core.cardinality_masks(data["true"])
                entries.append(
                    {
                        "database": db,
                        "cohort": cohort,
                        "pool_unique": len(candidates),
                        "sample_unique": len(chosen),
                        "manifest_hash": content_hash,
                        "selected_identity_digest": digest(
                            [data["query_hash"][i] for i in chosen]
                        ),
                        "sample_presentations": int(counts[chosen].sum()),
                        "pool_cardinality_counts": {
                            k: int(m[candidates].sum()) for k, m in masks.items()
                        },
                        "sample_cardinality_counts": {
                            k: int(m[chosen].sum()) for k, m in masks.items()
                        },
                        "pool_four_plus": len(candidates),
                        "sample_four_plus": len(chosen),
                    }
                )
                for i in chosen:
                    r = {
                        "database": db,
                        "cohort": cohort,
                        "unique_position": i,
                        "raw_index": i,
                        "query_hash": data["query_hash"][i],
                        "presentations": int(counts[i]),
                        **{k: float(data[k][i]) for k in ("true", "pg", "n_tables")},
                    }
                    samples.append(r)
                    predictions.append(
                        {
                            k: v
                            for k, v in r.items()
                            if k
                            not in ("unique_position", "raw_index", "presentations")
                        }
                        | {arm: float(data[arm][i]) for arm in ("A", "B")}
                    )
        for name, rows in (
            ("sample-manifest.jsonl", samples),
            ("source-predictions.jsonl", predictions),
        ):
            (root / name).write_text("".join(json.dumps(r) + "\n" for r in rows))
        replay_digest = sampled["digest"]
        core.write_json(
            root / "sample-plan.json",
            {
                "cohorts": entries,
                "sample_digest": replay_digest,
                "episodes": snapshot["episodes"],
                "presentations": snapshot["presentations"],
                "selected_rows_sha256": hashlib.sha256(
                    (root / "sample-manifest.jsonl").read_bytes()
                ).hexdigest(),
            },
        )
        hist = root / "history"
        hist.mkdir()
        core.write_json(hist / "preflight.json", {"sources": sources})
        core.write_json(
            hist / "summary.json",
            {
                "runs": [
                    {
                        "sample_plan_digest": replay_digest,
                        "sample_plan_episodes": snapshot["episodes"],
                        "config": config,
                        "counts": {
                            "support_presentations": 0,
                            "query_presentations": snapshot["presentations"],
                        },
                    }
                ]
            },
        )
        for arm, name in (("A", "A_mean"), ("B", "B_mean_tail10")):
            (hist / name).mkdir()
            with (hist / name / "endpoint_rows.jsonl").open("w") as stream:
                for db in validation:
                    for i in range(400):
                        pg_log = float(np.log(data["pg"][i]))
                        r = {
                            "database": db,
                            "query_hash": data["query_hash"][i],
                            "true": float(data["true"][i]),
                            "n_tables": 4,
                            "pg_reference_log": pg_log,
                            "pg_input_log": float(np.float32(pg_log)),
                            "log_pred": float(data[arm][i]),
                        }
                        stream.write(json.dumps(r) + "\n")
        gen = {"rows": []}
        for db in train:
            seen, unseen = core.memberships(states[db])
            for cohort, members in (("seen", seen), ("unseen", unseen)):
                member = {
                    k: ([v[i] for i in members] if k == "query_hash" else v[members])
                    for k, v in data.items()
                }
                gen["rows"].append(
                    {
                        "database": db,
                        "cohort": cohort,
                        "statistics": cohort_report(member)
                        if len(members)
                        else [{"bucket": "all", "n": 0, "status": "below_min_cell_n"}],
                    }
                )
        gen["rows"].extend(
            {
                "database": db,
                "cohort": "source_validation_full",
                "statistics": cohort_report(data),
            }
            for db in validation
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
            self.assertEqual(result["sample_queries"], 400 * len(core.TRAIN))
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
            next(row for row in gen["rows"] if row["database"] == "chembl")[
                "statistics"
            ][0]["A"]["q95"] += 1
            core.write_json(p2, gen)
            with self.assertRaises(ValueError):
                verify(root, root, hist, caches, states)

    def test_cohort_sets_must_be_complete_unique_and_source_only(self):
        import json

        from scripts.s7_independent_verify import verify

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            hist, caches, states = self.make(root)
            for filename, mutations in [
                (
                    "sample-plan.json",
                    [
                        lambda d: d.update(cohorts=[]),
                        lambda d: d["cohorts"].append(d["cohorts"][0]),
                        lambda d: d["cohorts"][0].update(database="imdb"),
                    ],
                ),
                (
                    "generalization.json",
                    [
                        lambda d: d.update(rows=[]),
                        lambda d: d["rows"].append(d["rows"][0]),
                        lambda d: d["rows"].append(
                            {
                                "database": "imdb",
                                "cohort": "seen",
                                "statistics": [
                                    {
                                        "bucket": "all",
                                        "n": 0,
                                        "status": "below_min_cell_n",
                                    }
                                ],
                            }
                        ),
                        lambda d: d["rows"][0]["statistics"].append(
                            d["rows"][0]["statistics"][0]
                        ),
                    ],
                ),
            ]:
                path = root / filename
                original = path.read_text()
                for number, mutate in enumerate(mutations):
                    with self.subTest(file=filename, mutation=number):
                        value = json.loads(original)
                        mutate(value)
                        core.write_json(path, value)
                        with self.assertRaises(ValueError):
                            verify(root, root, hist, caches, states)
                        path.write_text(original)

    def test_validation_endpoint_cache_binding_and_complete_identities(self):
        import json

        from scripts.s7_independent_verify import verify

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            hist, caches, states = self.make(root)
            for arm in ("A_mean", "B_mean_tail10"):
                path = hist / arm / "endpoint_rows.jsonl"
                original = path.read_text()
                mutations = [
                    lambda rows: rows[0].update(true=-123),
                    lambda rows: rows[0].update(n_tables=0),
                    lambda rows: rows[0].update(pg_reference_log=9),
                    lambda rows: rows[0].update(pg_input_log=9),
                    lambda rows: rows.append(rows[0]),
                    lambda rows: rows.pop(),
                    lambda rows: rows.append({**rows[0], "database": "imdb"}),
                    lambda rows: rows[0].update(log_pred=float("nan")),
                ]
                for number, mutate in enumerate(mutations):
                    with self.subTest(arm=arm, mutation=number):
                        rows = [json.loads(line) for line in original.splitlines()]
                        mutate(rows)
                        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
                        with self.assertRaises(ValueError):
                            verify(root, root, hist, caches, states)
                        path.write_text(original)

    def test_legalacts_report_sets_thresholds_and_composition_are_bound(self):
        import json

        from scripts.s7_independent_verify import verify

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            hist, caches, states = self.make(root)
            path = root / "legalacts.json"
            original = path.read_text()
            mutations = [
                lambda d: d.update(tail_groups=[]),
                lambda d: d["tail_groups"].append(d["tail_groups"][0]),
                lambda d: d.update(strata=[]),
                lambda d: d["strata"].append(d["strata"][0]),
                lambda d: d.update(thresholds={"A_q95": -1, "B_q95": -1}),
                lambda d: d.update(table_counts={"999": 999}),
            ]
            for number, mutate in enumerate(mutations):
                with self.subTest(mutation=number):
                    value = json.loads(original)
                    mutate(value)
                    core.write_json(path, value)
                    with self.assertRaises(ValueError):
                        verify(root, root, hist, caches, states)
                    path.write_text(original)

    def test_sample_plan_composition_exposure_and_bindings_are_verified(self):
        import json

        from scripts.s7_independent_verify import verify

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            hist, caches, states = self.make(root)
            path = root / "sample-plan.json"
            original = path.read_text()
            changes = {
                "pool_cardinality_counts": {},
                "sample_cardinality_counts": {},
                "pool_four_plus": 0,
                "sample_four_plus": 0,
                "sample_presentations": -1,
                "selected_identity_digest": "WRONG",
                "manifest_hash": "WRONG",
            }
            for field, value in changes.items():
                with self.subTest(field=field):
                    d = json.loads(original)
                    d["cohorts"][0][field] = value
                    core.write_json(path, d)
                    with self.assertRaises(ValueError):
                        verify(root, root, hist, caches, states)
                    path.write_text(original)
            for field, value in {
                "selected_rows_sha256": "WRONG",
                "sample_digest": "WRONG",
                "episodes": -1,
                "presentations": -1,
            }.items():
                with self.subTest(field=field):
                    d = json.loads(original)
                    d[field] = value
                    core.write_json(path, d)
                    with self.assertRaises(ValueError):
                        verify(root, root, hist, caches, states)
                    path.write_text(original)

    def test_role_swaps_and_joint_preserving_redistribution_are_rejected(self):
        from copy import deepcopy

        from scripts.s7_independent_verify import verify

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            hist, caches, states = self.make(root)
            self.assertTrue(verify(root, root, hist, caches, states)["passed"])
            altered = deepcopy(states)
            for row in altered.values():
                row["support"], row["query"] = row["query"], row["support"]
            with self.assertRaisesRegex(ValueError, "role"):
                verify(root, root, hist, caches, altered)
            altered = deepcopy(states)
            for row in altered.values():
                row["query"] += row["support"]
                row["support"][:] = 0
            with self.assertRaisesRegex(ValueError, "role"):
                verify(root, root, hist, caches, altered)

    def test_independent_role_replay_rejects_unsigned_order_and_boolean_accumulation(
        self,
    ):
        import json
        from copy import deepcopy

        from scripts.s7_independent_verify import independent_roles

        with tempfile.TemporaryDirectory() as tmp:
            hist, caches, _ = self.make(Path(tmp))
            config = json.loads((hist / "summary.json").read_text())["runs"][0][
                "config"
            ]
            malformed = deepcopy(caches)
            malformed["accidents"]["indices"] = malformed["accidents"][
                "indices"
            ].astype(np.uint64)[::-1]
            with self.assertRaisesRegex(ValueError, "pool"):
                independent_roles(malformed, config)
            with self.assertRaisesRegex(ValueError, "support"):
                independent_roles(caches, {**config, "accumulation": True})

    def test_sparse_pairs_reject_nonfinite_and_overflow_before_count_only_return(self):
        from scripts.s7_independent_verify import pair_check

        record = {"n": 1, "status": "below_min_cell_n"}
        data = S7Tests().fixture(1)
        mask = np.ones(1, dtype=bool)
        for arm in ("A", "B"):
            for invalid in (np.nan, np.inf, -np.inf, 2000.0, -1e200):
                with self.subTest(arm=arm, invalid=invalid):
                    data[arm][:] = invalid
                    with self.assertRaisesRegex(ValueError, "finite|overflow"):
                        pair_check(record, data, mask)
                data[arm][:] = 0
        pair_check(record, data, mask)
        pair_check(
            {"n": 0, "status": "below_min_cell_n"}, data, np.zeros(1, dtype=bool)
        )


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

    def test_empty_and_sparse_cohorts_have_no_invented_median(self):
        from scripts.s7_archive_report import overview

        rows = [
            {
                "database": db,
                "cohort": cohort,
                "statistics": [{"bucket": "all", "n": 0, "status": "below_min_cell_n"}],
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
        for item in result["within_database_ratios"]:
            self.assertEqual(item["database_count"], 0)
            self.assertIsNone(item["median_within_database_q95_unseen_div_seen"])
            self.assertEqual(item["status"], "no_eligible_databases")
        for item in result["b_vs_a_direction_counts"]:
            self.assertEqual(item["database_count"], 0)
            self.assertIsNone(item["median_relative_q95_change_b_vs_a"])
        rows[0]["statistics"][0]["n"] = 1
        with self.assertRaises(ValueError):
            overview(rows, ["accidents"])

    def test_archive_rejects_aliases_without_modifying_raw_evidence(self):
        import hashlib
        import json
        import os

        from scripts.s7_archive_report import FILES, main

        for link_kind in ("symlink", "hardlink", "none", "during_write"):
            with (
                self.subTest(link_kind=link_kind),
                tempfile.TemporaryDirectory() as tmp,
            ):
                root = Path(tmp)
                fixture = root / "fixture"
                fixture.mkdir()
                S7VerificationIntegrationTests().make(fixture)
                output = root / "实验文档/results/s7-fit-generalization-20261007"
                reports = root / "reports/s7-fit-generalization-20261007"
                output.mkdir(parents=True)
                reports.mkdir(parents=True)
                (root / "configs").mkdir(exist_ok=True)
                core.write_json(
                    root / "configs/data.json",
                    {
                        "splits": {
                            "train": [
                                r["database"]
                                for r in json.loads(
                                    (fixture / "sample-plan.json").read_text()
                                )["cohorts"]
                                if r["cohort"] == "seen"
                            ]
                        }
                    },
                )
                for name in (
                    "sample-plan.json",
                    "generalization.json",
                    "legalacts.json",
                ):
                    (output / name).write_bytes((fixture / name).read_bytes())
                plan = json.loads((output / "sample-plan.json").read_text())
                core.write_json(
                    output / "summary.json",
                    {
                        "status": "completed",
                        "training": False,
                        "target_data_read": False,
                        "adaptation": False,
                        "independent_verification_passed": True,
                        "s6_decision": "no_candidate",
                        "inference_queries": sum(
                            r["sample_unique"] for r in plan["cohorts"]
                        ),
                    },
                )
                core.write_json(output / "protocol.json", {})
                core.write_json(output / "verification.json", {"passed": True})
                hashes = {
                    name: hashlib.sha256((output / name).read_bytes()).hexdigest()
                    for name in FILES[1:]
                }
                core.write_json(
                    output / "status.json",
                    {"status": "completed", "output_sha256": hashes},
                )
                hashes["status.json"] = hashlib.sha256(
                    (output / "status.json").read_bytes()
                ).hexdigest()
                core.write_json(reports / "server-archive-sha256.json", hashes)
                source = output / "generalization.json"
                alias = output / "overview.json"
                if link_kind == "symlink":
                    alias.symlink_to(source)
                elif link_kind == "hardlink":
                    os.link(source, alias)
                before = source.read_bytes()
                if link_kind in ("symlink", "hardlink"):
                    with self.assertRaisesRegex(ValueError, "alias|link"):
                        main(root)
                elif link_kind == "none":
                    import contextlib
                    import io

                    with contextlib.redirect_stdout(io.StringIO()):
                        main(root)
                    self.assertTrue(
                        json.loads((reports / "archive-verification.json").read_text())[
                            "passed"
                        ]
                    )
                else:
                    from unittest.mock import patch

                    write = Path.write_text

                    def mutate_after_write(
                        path,
                        text,
                        *args,
                        write=write,
                        alias=alias,
                        source=source,
                        **kwargs,
                    ):
                        result = write(path, text, *args, **kwargs)
                        if path == alias:
                            write(source, "CORRUPTED")
                        return result

                    with (
                        patch.object(Path, "write_text", mutate_after_write),
                        self.assertRaisesRegex(ValueError, "Raw evidence changed"),
                    ):
                        main(root)
                    self.assertFalse((reports / "archive-verification.json").exists())
                    source.write_bytes(before)
                self.assertEqual(source.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
