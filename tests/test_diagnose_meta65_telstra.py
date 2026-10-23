import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from scripts import diagnose_meta65_telstra as diagnosis


class QErrorTests(unittest.TestCase):
    def test_q_error_clips_prediction_and_truth_to_one(self):
        log_pred = np.asarray([0.0, np.log(10.0), np.log(1.0), np.log(100.0)])
        true = np.asarray([1.0, 1.0, 10.0, 10.0])
        q, log_q = diagnosis.q_error(log_pred, true)
        np.testing.assert_allclose(q, [1.0, 10.0, 10.0, 10.0])
        np.testing.assert_allclose(
            log_q, [0.0, np.log(10.0), np.log(10.0), np.log(10.0)]
        )

    def test_log_q95_is_exp_of_log_space_percentile(self):
        log_pred = np.asarray([0.0, np.log(2.0), np.log(4.0), np.log(8.0)])
        true = np.ones(4)
        report = diagnosis.quantile_metrics(log_pred, true)
        expected = float(np.exp(np.percentile(np.log([1.0, 2.0, 4.0, 8.0]), 95)))
        self.assertEqual(report["log_q95"], expected)
        self.assertNotEqual(
            report["log_q95"],
            float(
                np.percentile(np.exp([0.0, np.log(2.0), np.log(4.0), np.log(8.0)]), 95)
            ),
        )


class BucketTests(unittest.TestCase):
    def test_d1_bucket_boundaries_and_shares(self):
        tables = np.asarray([2, 2, 3, 4, 5, 6, 7], dtype=np.int64)
        rows = diagnosis.table_bucket_rows(tables, min_n=1)
        self.assertEqual(rows["2"]["count"], 2)
        self.assertAlmostEqual(rows["2"]["share"], 2 / 7)
        self.assertEqual(rows["3"]["count"], 1)
        self.assertEqual(rows["4"]["count"], 1)
        self.assertEqual(rows["5+"]["count"], 3)
        self.assertAlmostEqual(sum(row["share"] for row in rows.values()), 1.0)

    def test_d1_assigns_one_table_queries_to_two_bucket(self):
        rows = diagnosis.table_bucket_rows(np.asarray([1, 2, 3]), min_n=1)
        self.assertEqual(rows["2"]["count"], 2)
        self.assertEqual(rows["3"]["count"], 1)

    def test_d2_fixed_true_bucket_boundaries(self):
        true = np.asarray(
            [
                1.0,
                99.0,
                100.0,
                999.0,
                1000.0,
                9999.0,
                10000.0,
                99999.0,
                100000.0,
                999999.0,
                1000000.0,
            ]
        )
        rows = diagnosis.true_bucket_rows(true, min_n=1)
        self.assertEqual(
            [rows[key]["count"] for key in diagnosis.TRUE_BUCKET_LABELS],
            [2, 2, 2, 2, 2, 1],
        )
        self.assertAlmostEqual(sum(row["share"] for row in rows.values()), 1.0)

    def test_below_min_bucket_reports_only_count_and_status(self):
        rows = diagnosis.true_bucket_rows(np.asarray([1.0]), min_n=2)
        self.assertEqual(rows["<100"], {"count": 1, "status": "below_min_n"})

    def test_quadruplet_never_reports_only_percent(self):
        values = diagnosis.four_tuple(2.0, 3.0)
        self.assertEqual(set(values), {"A", "C_minus_A", "C_over_A", "log10_C_over_A"})
        self.assertEqual(values["A"], 2.0)
        self.assertEqual(values["C_minus_A"], 1.0)
        self.assertEqual(values["C_over_A"], 1.5)
        self.assertAlmostEqual(values["log10_C_over_A"], np.log10(1.5))

    def test_direction_labels_use_a_and_optional_pg(self):
        a_log_pred = np.asarray([0.0, 3.0, 1.0, 1.0])
        true = np.asarray([1.0, 1.0, 10.0, 10.0])
        self.assertEqual(
            diagnosis.direction_labels(a_log_pred, true).tolist(),
            ["over", "over", "under", "under"],
        )
        pg = np.asarray([2.0, 10.0, 1.0, 100.0])
        self.assertEqual(
            diagnosis.pg_direction_labels(pg, true).tolist(),
            ["over", "over", "under", "over"],
        )

    def test_paired_delta_uses_intersection_and_reports_hashes(self):
        left = np.asarray([1.0, 2.0, 3.0])
        right = np.asarray([2.0, 1.0, 3.0])
        result = diagnosis.delta_summary(
            left, right, hashes=["h1", "h2", "h3"], ids=["i1", "i2", "i3"], top_n=2
        )
        self.assertEqual(result["count"], 3)
        self.assertEqual(
            [item["query_hash"] for item in result["largest"]], ["h1", "h2"]
        )
        self.assertAlmostEqual(result["mean"], 0.0)
        self.assertAlmostEqual(result["p50"], 0.0)

    def test_d4_requires_identical_run_query_sets(self):
        first = [("h1", "i1"), ("h2", "i2")]
        second = [("h1", "i1"), ("h2", "i2")]
        third = [("h2", "i2"), ("h1", "i1")]
        self.assertEqual(
            set(diagnosis.verify_query_sets([first, second], "run")), {"h1", "h2"}
        )
        # Reordering is allowed; losing or adding an identity is not.
        self.assertEqual(
            set(diagnosis.verify_query_sets([first, third], "run")), {"h1", "h2"}
        )
        fourth = [("h1", "i1"), ("h3", "i3")]
        with self.assertRaisesRegex(ValueError, "query set"):
            diagnosis.verify_query_sets([first, fourth], "run")

    def test_d4_aligns_each_run_by_hash_not_position(self):
        cells = {}
        for k in diagnosis.SHOTS:
            for arm in diagnosis.ARMS:
                for run in range(diagnosis.RUNS):
                    cells[(arm, "telstra", k, run)] = {
                        "query_hash": np.asarray(["h2", "h1"], dtype=str),
                        "query_id": np.asarray([f"i2-{run}", f"i1-{run}"], dtype=str),
                        "log_pred": np.asarray([np.log(10.0), np.log(10.0)]),
                        "true": np.asarray([10.0, 10.0]),
                        "n_tables": np.asarray([4, 4]),
                    }
        result = diagnosis._d4(cells, "telstra")["5"]
        # Runs use disjoint query sets by design, so pairing happens on the
        # intersection of identities common to every arm and run.
        self.assertEqual(result["query_count"], 2)
        self.assertEqual(
            result["query_set_basis"],
            "intersection of identities present in all arms and all runs",
        )
        self.assertIn(result["C_minus_A"]["largest"][0]["query_hash"], {"h1", "h2"})

    def test_d5_leave_one_out_reports_sign_and_order_flips(self):
        per_k = {
            5: {"A": [10.0, 11.0], "C": [12.0, 13.0]},
            10: {"A": [20.0, 21.0], "C": [24.0, 25.0]},
        }
        result = diagnosis.leave_one_out_stability(per_k)
        self.assertEqual(result["sign_flips"], [])
        self.assertEqual(len(result["leave_one_out"]), 4)
        self.assertFalse(result["k_order_changed"])

    def test_d5_real_arm_keys_do_not_silently_skip_leave_one_out(self):
        cells = {
            (arm, "telstra", k, run): {
                "n_tables": np.asarray([4, 5]),
                "true": np.ones(2),
                "log_pred": np.full(2, float(index + 1)),
            }
            for index, arm in enumerate(diagnosis.ARMS)
            for k in diagnosis.SHOTS
            for run in range(diagnosis.RUNS)
        }
        result = diagnosis._d5(cells, "telstra")["leave_one_out"]
        self.assertEqual(len(result["leave_one_out"]), 25)
        self.assertTrue(
            all(row["continued_positive"] for row in result["leave_one_out"])
        )

    def test_d5_missing_keys_fail_instead_of_claiming_stability(self):
        with self.assertRaises(ValueError):
            diagnosis.leave_one_out_stability({5: {"wrong": [1, 2]}})

    def test_d6_top_five_percent_log_contribution(self):
        log_q = np.asarray([0.0] * 19 + [10.0])
        result = diagnosis.top_log_contribution(log_q, fraction=0.05)
        self.assertEqual(result["selected_count"], 1)
        self.assertEqual(result["contribution_ratio"], 1.0)
        self.assertEqual(result["tail_count_share"], 0.05)

    def test_pg_anchor_verifies_manifest_row_mapping_and_values(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = [
                {"id": "telstra/pretrain/1", "query_hash": "h0"},
                {"id": "telstra/pretrain/2", "query_hash": "h1"},
                {"id": "telstra/pretrain/3", "query_hash": "h1"},
            ]
            manifest = {"rows": rows}
            (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            np.save(
                root / "cards.npy", np.asarray([[1.0, 2.0], [3.0, 2.0], [9.0, 9.0]])
            )
            np.save(
                root / "offsets.npy",
                np.asarray([[0, 0, 2, 0], [0, 0, 4, 0], [0, 0, 7, 0], [0, 0, 10, 0]]),
            )
            status, details, pg = diagnosis.anchor_pg(
                root,
                query_id=["telstra/pretrain/1", "telstra/pretrain/2"],
                query_hash=["h0", "h1"],
                true=np.asarray([1.0, 3.0]),
                n_tables=np.asarray([2, 3]),
            )
            self.assertEqual(status, "verified")
            self.assertEqual(details["rows"], [0, 1])
            np.testing.assert_allclose(pg, [2.0, 2.0])

    def test_pg_anchor_failure_is_explicit_and_never_used(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "manifest.json").write_text(
                json.dumps(
                    {"rows": [{"id": "telstra/pretrain/0", "query_hash": "h0"}]}
                ),
                encoding="utf-8",
            )
            np.save(root / "cards.npy", np.asarray([[2.0, 2.0]]))
            np.save(root / "offsets.npy", np.asarray([[0, 0, 2, 0], [0, 0, 3, 0]]))
            status, details, pg = diagnosis.anchor_pg(
                root,
                query_id=["telstra/pretrain/0"],
                query_hash=["h0"],
                true=np.asarray([1.0]),
                n_tables=np.asarray([2]),
            )
            self.assertEqual(status, "failed")
            self.assertIsNone(pg)
            self.assertIn("error", details)

    def test_output_guards_reject_gpu_target_and_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            input_dir = root / "input"
            cache_dir = root / "cache"
            input_dir.mkdir()
            cache_dir.mkdir()
            existing = root / "out.json"
            existing.write_text("{}")
            with self.assertRaisesRegex(ValueError, "cpu"):
                diagnosis.prepare_outputs(
                    SimpleNamespace(
                        device="cuda",
                        out=str(root / "new.json"),
                        csv_dir=None,
                        old_run=str(input_dir),
                        completion_run=str(input_dir),
                        cache=None,
                    )
                )
            with self.assertRaisesRegex(ValueError, "target"):
                diagnosis.prepare_outputs(
                    SimpleNamespace(
                        device="cpu",
                        out=str(input_dir / "new.json"),
                        csv_dir=None,
                        old_run=str(input_dir),
                        completion_run=str(input_dir),
                        cache=None,
                    )
                )
            with self.assertRaisesRegex(ValueError, "overwrite"):
                diagnosis.prepare_outputs(
                    SimpleNamespace(
                        device="cpu",
                        out=str(existing),
                        csv_dir=None,
                        old_run=str(input_dir),
                        completion_run=str(input_dir),
                        cache=None,
                    )
                )
            with self.assertRaisesRegex(ValueError, "input directory|cache"):
                diagnosis.prepare_outputs(
                    SimpleNamespace(
                        device="cpu",
                        out=str(cache_dir / "new.json"),
                        csv_dir=None,
                        old_run=str(input_dir),
                        completion_run=str(input_dir),
                        cache=str(cache_dir),
                    )
                )


class EndToEndSyntheticTests(unittest.TestCase):
    def _write_npz(
        self, path, query_hash, log_pred, true, tables, db=None, query_id=None
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        if query_id is None:
            query_id = [f"id-{h}" for h in query_hash]
        np.savez(
            path,
            log_pred=np.asarray(log_pred, dtype=np.float64),
            true=np.asarray(true, dtype=np.float64),
            n_tables=np.asarray(tables, dtype=np.int64),
            query_hash=np.asarray(query_hash, dtype=str),
            query_id=np.asarray(query_id, dtype=str),
        )

    def test_execute_reads_only_source_databases_and_writes_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            old = root / "old"
            completion = root / "completion"
            protocol = root / "protocol.md"
            protocol.write_text("protocol", encoding="utf-8")
            for db in diagnosis.SOURCE_DATABASES:
                hashes = [f"{db}-h{i}" for i in range(6)]
                for arm in ("A_all5_head3", "B_head3_head3"):
                    for k in diagnosis.SHOTS:
                        for run in range(5):
                            self._write_npz(
                                old / arm / "predictions" / f"{db}-k{k}-run{run}.npz",
                                hashes,
                                np.log(np.arange(1, 7)),
                                np.arange(1, 7),
                                np.asarray([2, 2, 3, 4, 5, 6]),
                                query_id=[f"{db}/pretrain/{i}" for i in range(6)],
                            )
                for k in diagnosis.SHOTS:
                    for run in range(5):
                        self._write_npz(
                            completion
                            / "C_last3_last3"
                            / "predictions"
                            / f"{db}-k{k}-run{run}.npz",
                            hashes,
                            np.log(np.arange(1, 7)),
                            np.arange(1, 7),
                            np.asarray([2, 2, 3, 4, 5, 6]),
                            query_id=[f"{db}/pretrain/{i}" for i in range(6)],
                        )
            plan = {}
            for db in diagnosis.SOURCE_DATABASES:
                plan[db] = [{"run": run, "query": list(range(6))} for run in range(5)]
            (old / "endpoint_plan.json").write_text(json.dumps(plan), encoding="utf-8")
            (completion / "endpoint_plan.json").write_text(
                json.dumps(plan), encoding="utf-8"
            )
            args = SimpleNamespace(
                old_run=str(old),
                completion_run=str(completion),
                cache=None,
                out=str(root / "out.json"),
                csv_dir=str(root / "csv"),
                device="cpu",
                execute=True,
                seed=42,
            )
            result = diagnosis.execute(args, protocol_path=protocol)
            self.assertEqual(result["scope"], "descriptive_diagnosis_only")
            self.assertEqual(result["primary_analysis_database"], "telstra")
            self.assertEqual(set(result["databases"]), set(diagnosis.SOURCE_DATABASES))
            self.assertIn("input_sha256", result)
            self.assertIn("protocol_sha256", result)
            self.assertEqual(result["pg_anchor_status"], "not_attempted")


class NoHeavyDependencyTests(unittest.TestCase):
    def test_diagnosis_module_does_not_import_torch(self):
        text = Path(diagnosis.__file__).read_text(encoding="utf-8")
        self.assertNotIn("import torch", text)
        self.assertNotIn("from torch", text)


if __name__ == "__main__":
    unittest.main()
