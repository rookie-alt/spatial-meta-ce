import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts import s1_source_analysis as s1


class S1AnalysisTests(unittest.TestCase):
    def fixture(self, n=1000):
        true = np.geomspace(10, 1000000, n)
        tables = np.where(np.arange(n) % 2 == 0, 4, 6)
        pg = np.log(true) + 0.5
        a = np.log(true) + np.linspace(-2, 2, n)
        return {
            "true": true,
            "n_tables": tables,
            "pg_reference_log": pg,
            "pg_input_log": pg.astype(np.float32).astype(np.float64),
            "predictions": {
                s1.ARMS[0]: a,
                s1.ARMS[1]: a + 0.2,
                s1.ARMS[2]: a - 0.1,
                s1.ARMS[3]: a + 0.3,
            },
        }

    def test_raw_and_official_metrics_are_separate(self):
        result = s1.metrics(np.array([-2.0, 0.0, 2.0]), np.ones(3))
        self.assertAlmostEqual(result["raw_log_mse"], 8 / 3)
        self.assertAlmostEqual(result["raw_log_mae"], 4 / 3)
        self.assertAlmostEqual(result["official_log_mae"], 2 / 3)
        self.assertEqual(result["q50"], 1.0)

    def test_nonfinite_or_overflow_is_rejected(self):
        for value in [float("nan"), float("inf"), 1000.0]:
            with self.assertRaises((ValueError, FloatingPointError)):
                s1.metrics(np.array([value]), np.array([1.0]))

    def test_cardinality_and_table_bins_partition_all_rows(self):
        true = np.array(
            [0.0, 0.5, 9.0, 10.0, 99.0, 100.0, 999.0, 1000.0, 10000.0, 100000.0]
        )
        np.testing.assert_array_equal(
            sum(s1.cardinality_masks(true).values()), np.ones(len(true), dtype=int)
        )
        tables = np.array([1, 3, 4, 5, 6, 7])
        np.testing.assert_array_equal(
            sum(s1.table_masks(tables).values()), np.ones(len(tables), dtype=int)
        )

    def test_tail_ties_are_preserved_and_query_sets_paired(self):
        true = np.ones(100)
        a = np.zeros(100)
        a[-10:] = np.log(10)
        b = np.zeros(100)
        b[-5:] = np.log(10)
        b[:5] = np.log(10)
        result = s1.tail_migration(a, b, true)
        self.assertEqual(result["reference_tail_n"], 10)
        self.assertEqual(result["left_reference_tail_n"], 5)
        self.assertEqual(result["entered_tail_n"], 5)
        self.assertEqual(result["overlap_n"], 5)
        self.assertAlmostEqual(result["reference_tail_fraction"], 0.1)

    def test_sparse_cells_remain_counted_without_detailed_metrics(self):
        result = s1.analyze_database(self.fixture(100))
        self.assertEqual(result["strata"]["all:tables=4"]["n"], 50)
        self.assertEqual(result["strata"]["all:tables=4"]["status"], "below_min_cell_n")
        self.assertNotIn("arms", result["strata"]["all:tables=4"])
        self.assertEqual(result["global_metrics"]["all_tables"][s1.ARMS[0]]["n"], 100)

    def test_q99_suppressed_for_cells_below_1000(self):
        data = self.fixture(500)
        cell = s1.describe_cell(data, np.ones(500, dtype=bool))
        self.assertNotIn("q99", cell["arms"][s1.ARMS[0]])
        self.assertEqual(cell["arms"][s1.ARMS[0]]["n"], 500)

    def test_factor_pairs_and_pg_residual_accounting(self):
        result = s1.analyze_database(self.fixture())
        self.assertEqual(set(result["factor_pairs"]), {"B-A", "D-C", "C-A", "D-B"})
        for values in result["pg_residual"].values():
            for arm in s1.ARMS:
                self.assertLess(values["arms"][arm]["reconstruction_max_abs"], 1e-12)
                self.assertAlmostEqual(
                    sum(
                        values["arms"][arm][k]
                        for k in (
                            "improved_vs_pg_input_fraction",
                            "worsened_vs_pg_input_fraction",
                            "tied_vs_pg_input_fraction",
                        )
                    ),
                    1.0,
                )

    def test_global_and_primary_coverage(self):
        result = s1.analyze_database(self.fixture())
        self.assertEqual(result["global_metrics"]["all_tables"][s1.ARMS[0]]["n"], 1000)
        self.assertEqual(
            result["global_metrics"]["four_plus_tables"][s1.ARMS[0]]["n"], 1000
        )
        self.assertIn("B-A", result["tail_migration"]["four_plus_tables"])

    def test_pair_sign_and_tie_reference_are_explicit(self):
        pair = s1.paired(
            np.array([0.0, 1.0, 2.0]), np.array([1.0, 1.0, 0.0]), np.ones(3)
        )
        self.assertAlmostEqual(pair["improved_vs_reference_fraction"], 1 / 3)
        self.assertAlmostEqual(pair["worsened_vs_reference_fraction"], 1 / 3)
        self.assertAlmostEqual(pair["tied_vs_reference_fraction"], 1 / 3)

    def test_loader_aligns_shuffled_rows_and_rejects_identity_damage(self):
        import hashlib
        import json

        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "rows.jsonl"
            caches = {
                db: {**self.fixture(4), "query_hash": [f"h{i}" for i in range(4)]}
                for db in s1.DATABASES
            }
            rows = [
                {
                    "database": db,
                    "query_hash": h,
                    "true": float(data["true"][i]),
                    "n_tables": int(data["n_tables"][i]),
                    "log_pred": float(data["predictions"][s1.ARMS[0]][i]),
                }
                for db, data in caches.items()
                for i, h in enumerate(data["query_hash"])
            ]

            def save(values):
                path.write_text("".join(json.dumps(row) + "\n" for row in values))
                return hashlib.sha256(path.read_bytes()).hexdigest()

            result = s1.read_predictions(path, caches, save(rows[::-1]))
            np.testing.assert_array_equal(
                result["chembl"], caches["chembl"]["predictions"][s1.ARMS[0]]
            )
            for changed in [
                rows[:-1],
                rows + [rows[0]],
                [{**rows[0], "database": "stats"}] + rows[1:],
                [{**rows[0], "true": 123.0}] + rows[1:],
                [{**rows[0], "n_tables": 5}] + rows[1:],
                [{**rows[0], "query_hash": "unknown"}] + rows[1:],
            ]:
                with self.assertRaises(ValueError):
                    s1.read_predictions(path, caches, save(changed))
            with self.assertRaises(ValueError):
                s1.read_predictions(path, caches, "bad")

    def test_cache_loader_rejects_modified_arrays_and_bad_manifest(self):
        import json

        from spatial_meta_ce.manifests import file_hash
        from tests.test_mapping_ablation import build_store

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            build_store(root, "chembl", n_tables=4)
            path = root / "cache/pretrain/chembl"
            manifest = json.loads((path / "manifest.json").read_text())
            frozen = {
                "manifest_hash": manifest["content_hash"],
                "file_sha256": {p.name: file_hash(p) for p in path.iterdir()},
            }
            loaded = s1.read_cache(path, frozen)
            self.assertEqual(len(loaded["true"]), 8)
            np.testing.assert_array_equal(
                loaded["pg_input_log"],
                loaded["pg_reference_log"].astype(np.float32).astype(float),
            )
            bad = {**frozen, "manifest_hash": "wrong"}
            with self.assertRaises(ValueError):
                s1.read_cache(path, bad)
            with (path / "cards.npy").open("ab") as stream:
                stream.write(b"changed")
            with self.assertRaises(ValueError):
                s1.read_cache(path, frozen)

    def test_pg_nonfinite_and_sparse_boundary(self):
        data = self.fixture(200)
        self.assertEqual(
            s1.describe_cell(data, np.ones(200, dtype=bool))["status"], "descriptive"
        )
        data["pg_input_log"][0] = float("nan")
        with self.assertRaises(ValueError):
            s1.analyze_database(data)

    def test_runner_frozen_inputs_and_no_target_paths(self):
        from scripts import run_s1_diagnosis as runner

        contract = runner.load_contract(runner.ROOT)
        self.assertEqual(contract["summary"]["decision"], "no_candidate")
        self.assertEqual(set(contract["runs"]), set(s1.ARMS))
        from unittest.mock import patch

        with (
            patch.object(runner, "file_hash", return_value="wrong"),
            self.assertRaises(ValueError),
        ):
            runner.load_contract(runner.ROOT)

    def test_supervisor_timeout_reaps_own_child(self):
        import subprocess
        from unittest.mock import Mock, patch

        from scripts import run_s1_diagnosis as runner

        with tempfile.TemporaryDirectory() as temp:
            child = Mock(pid=123)
            child.wait.side_effect = [subprocess.TimeoutExpired("s1", 1), 0]
            with (
                patch.object(runner.subprocess, "Popen", return_value=child),
                patch.object(
                    runner.os, "killpg", side_effect=[None, ProcessLookupError]
                ) as kill,
                self.assertRaises(TimeoutError),
            ):
                runner.supervise(["test"], Path(temp), timeout=1)
            self.assertEqual(kill.call_args_list[0].args, (123, runner.signal.SIGTERM))

    def test_integration_reads_only_approved_arrays_and_exports(self):
        import copy
        import json
        from unittest.mock import patch

        from scripts import run_s1_diagnosis as runner
        from spatial_meta_ce.manifests import file_hash
        from tests.test_mapping_ablation import build_store

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            contract = copy.deepcopy(runner.load_contract(runner.ROOT))
            rows_by_arm = {a: [] for a in s1.ARMS}
            for db in s1.DATABASES:
                build_store(root / "artifacts", db, n_rows=400, n_tables=4)
                path = root / "artifacts/cache/pretrain" / db
                manifest = json.loads((path / "manifest.json").read_text())
                contract["preflight"]["sources"][db] = {
                    "manifest_hash": manifest["content_hash"],
                    "file_sha256": {p.name: file_hash(p) for p in path.iterdir()},
                }
                data = s1.read_cache(path, contract["preflight"]["sources"][db])
                for p in path.glob("group*.npy"):
                    p.unlink()
                for index, arm in enumerate(s1.ARMS):
                    pred = np.log(data["true"]) + 0.1 * (index + 1)
                    metric = s1.metrics(pred, data["true"])
                    contract["runs"][arm]["endpoint"]["per_database"][db] = {
                        b: {"queries": 400, "q50": metric["q50"], "q95": metric["q95"]}
                        for b in ("all_tables", "four_plus_tables")
                    }
                    rows_by_arm[arm].extend(
                        {
                            "database": db,
                            "query_hash": h,
                            "true": float(t),
                            "n_tables": int(n),
                            "log_pred": float(p),
                        }
                        for h, t, n, p in zip(
                            data["query_hash"], data["true"], data["n_tables"], pred
                        )
                    )
            for arm, rows in rows_by_arm.items():
                path = root / "runs" / runner.FORMAL_NAME / arm / "endpoint_rows.jsonl"
                path.parent.mkdir(parents=True)
                path.write_text("".join(json.dumps(r) + "\n" for r in rows))
                contract["runs"][arm]["endpoint_rows_sha256"] = file_hash(path)
                contract["runs"][arm]["endpoint"]["macro_four_plus_q95"] = np.exp(
                    0.1 * (s1.ARMS.index(arm) + 1)
                )
            (root / "scripts").mkdir()
            (root / "scripts/s1_source_analysis.py").write_bytes(
                Path(s1.__file__).read_bytes()
            )
            output = root / "runs/s1-fixture"
            output.mkdir()
            with (
                patch.object(runner, "load_contract", return_value=contract),
                patch.object(runner, "snapshot", return_value={}),
                patch.object(
                    runner, "COVERAGE", {db: (400, 400) for db in s1.DATABASES}
                ),
            ):
                report = runner.analyze(root, output)
            self.assertEqual(
                report["status"], "analysis_complete_requires_supervisor_verification"
            )
            for name in (
                "summary.json",
                "global.csv",
                "strata.csv",
                "tails.csv",
                "output-manifest.json",
            ):
                self.assertTrue((output / name).exists())

    def test_tail_sample_gates_include_459_row_fixed_tail(self):
        true = np.ones(9178)
        a = np.linspace(0, 3, 9178)
        result = s1.tail_migration(a, a + 0.1, true)
        fixed = result["fixed_reference_tail_comparison"]
        self.assertEqual(fixed["n"], 459)
        self.assertNotIn("q99", fixed["candidate"])
        small = s1.tail_migration(a[:100], a[:100] + 0.1, true[:100])
        self.assertEqual(
            small["fixed_reference_tail_comparison"]["status"], "below_min_cell_n"
        )

    def test_supervisor_verifies_exports_before_completed(self):
        import json
        from unittest.mock import Mock, patch

        from scripts import run_s1_diagnosis as runner

        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp).resolve()
            runner.update_status(output, "outputs_ready")
            files = {}
            for name in ("summary.json", "global.csv", "strata.csv", "tails.csv"):
                (output / name).write_text("{}")
                files[name] = runner.file_hash(output / name)
            runner.write_json(output / "output-manifest.json", files)
            child = Mock()
            child.wait.return_value = 0
            with patch.object(runner.subprocess, "Popen", return_value=child):
                runner.supervise(["test"], output)
            self.assertEqual(
                json.loads((output / "status.json").read_text())["status"], "completed"
            )

    def test_modules_do_not_import_torch(self):
        import subprocess
        import sys

        code = 'import sys; import scripts.run_s1_diagnosis, scripts.s1_source_analysis; assert "torch" not in sys.modules'
        subprocess.run([sys.executable, "-B", "-c", code], check=True)

    def test_prediction_read_rechecks_hash(self):
        import json
        from unittest.mock import patch

        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "rows.jsonl"
            caches = {
                db: {**self.fixture(1), "query_hash": ["h"]} for db in s1.DATABASES
            }
            rows = [
                {
                    "database": db,
                    "query_hash": "h",
                    "true": float(d["true"][0]),
                    "n_tables": int(d["n_tables"][0]),
                    "log_pred": 1.0,
                }
                for db, d in caches.items()
            ]
            path.write_text("".join(json.dumps(r) + "\n" for r in rows))
            with (
                patch.object(s1, "file_hash", side_effect=["same", "changed"]),
                self.assertRaises(ValueError),
            ):
                s1.read_predictions(path, caches, "same")

    def test_pg_conditioned_cells_apply_q99_gate(self):
        report = s1.analyze_database(self.fixture(500))
        group = report["pg_conditioned"]["all_tables:pg_over"]
        self.assertEqual(group["n"], 500)
        self.assertNotIn("q99", group["pg_reference"])
        self.assertNotIn(
            "q99",
            group["arms"][s1.ARMS[0]]["official_comparison_vs_pg_reference"][
                "candidate"
            ],
        )

    def test_pg_direction_uses_reference_not_float32_rounding(self):
        data = self.fixture()
        data["pg_reference_log"] = np.log(data["true"])
        data["pg_input_log"] = (
            data["pg_reference_log"].astype(np.float32).astype(np.float64)
        )
        report = s1.analyze_database(data)
        self.assertEqual(report["pg_conditioned"]["all_tables:pg_exact"]["n"], 1000)
        self.assertEqual(report["pg_conditioned"]["all_tables:pg_under"]["n"], 0)
        self.assertEqual(report["pg_conditioned"]["all_tables:pg_over"]["n"], 0)

    def test_worker_lifecycle_without_resource_side_effects(self):
        import os
        from unittest.mock import Mock, patch

        from scripts import run_s1_diagnosis as runner

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            out = root / "runs/s1"
            out.mkdir(parents=True)
            runner.update_status(out, "planned")
            runner.write_json(
                out / "supervision.json", {"pid": os.getppid(), "token": "trusted"}
            )
            env = {
                **{k: "1" for k in runner.THREAD_ENV},
                "CUDA_VISIBLE_DEVICES": "",
                "S1_PARENT": str(os.getppid()),
                "S1_TOKEN": "trusted",
            }
            lib = Mock()
            lib.prctl.return_value = 0
            with (
                patch.dict(os.environ, env),
                patch.object(runner.sys, "platform", "linux"),
                patch.object(runner.ctypes, "CDLL", return_value=lib),
                patch.object(runner.resource, "setrlimit") as limits,
                patch.object(runner, "analyze", return_value={}),
            ):
                runner.worker(root, out)
            limits.assert_called_once_with(
                runner.resource.RLIMIT_AS,
                (runner.WORKER_MEMORY_BYTES, runner.WORKER_MEMORY_BYTES),
            )
            self.assertEqual(
                __import__("json").loads((out / "status.json").read_text())["status"],
                "outputs_ready",
            )

    def test_worker_rejects_parent_platform_and_ownership(self):
        import os
        from unittest.mock import patch

        from scripts import run_s1_diagnosis as runner

        with (
            patch.object(runner.sys, "platform", "darwin"),
            self.assertRaises(RuntimeError),
        ):
            runner.worker(Path("/unused"), Path("/unused"))
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            out = root / "runs/s1"
            out.mkdir(parents=True)
            runner.update_status(out, "planned")
            runner.write_json(out / "supervision.json", {"pid": 0, "token": "wrong"})
            env = {
                **{k: "1" for k in runner.THREAD_ENV},
                "CUDA_VISIBLE_DEVICES": "",
                "S1_PARENT": "0",
            }
            with (
                patch.dict(os.environ, env),
                patch.object(runner.sys, "platform", "linux"),
                self.assertRaises(ValueError),
            ):
                runner.worker(root, out)

    def test_budget_exhausted_never_launches(self):
        from unittest.mock import patch

        from scripts import run_s1_diagnosis as runner

        with tempfile.TemporaryDirectory() as temp:
            with (
                patch.object(runner.subprocess, "Popen") as start,
                self.assertRaises(TimeoutError),
            ):
                runner.supervise(["test"], Path(temp), timeout=0)
            start.assert_not_called()
        with self.assertRaises(TimeoutError):
            runner.deadline_expired(None, None)

    def test_parent_cli_dry_run(self):
        import contextlib
        import io
        from unittest.mock import patch

        from scripts import run_s1_diagnosis as runner

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            with (
                patch.object(runner, "ROOT", root),
                patch.object(runner, "load_contract", return_value={}),
                patch.object(
                    runner.sys, "argv", ["s1", "--output", str(root / "runs/new")]
                ),
                patch.object(runner.sys, "platform", "darwin"),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                runner.main()
            self.assertFalse((root / "runs/new").exists())

    def test_output_is_new_and_does_not_overlap_inputs(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            (root / "runs").mkdir()
            s1.validate_output(root / "runs/new", root)
            for path in (
                root / "outside",
                root / "runs",
                root / "runs/precision-screen-20261006/new",
            ):
                with self.assertRaises(ValueError):
                    s1.validate_output(path, root)
            (root / "runs/old").mkdir()
            with self.assertRaises(FileExistsError):
                s1.validate_output(root / "runs/old", root)
            (root / "runs/link").symlink_to(root / "outside")
            with self.assertRaises(ValueError):
                s1.validate_output(root / "runs/link/new", root)


if __name__ == "__main__":
    unittest.main()
