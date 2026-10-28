import csv
import math
import tempfile
import unittest
from pathlib import Path

from spatial_meta_ce.plans import EVAL_DATABASES
from spatial_meta_ce.source_eval import ARMS
from spatial_meta_ce.source_summary import summarize_predictions, write_summary

QUERIES = 6
TRUE = 100.0


def log_for(q_value):
    return math.log(TRUE) + math.log(q_value)


def write_predictions(path, values):
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            (
                "arm",
                "seed",
                "database",
                "k",
                "run",
                "query_id",
                "query_hash",
                "true_cardinality",
                "log_prediction",
            )
        )
        for arm in ARMS:
            for seed in (42, 43, 44):
                for db in EVAL_DATABASES:
                    units = [(0, 0)]
                    for k in (5, 20):
                        units.extend((k, run) for run in range(1, 6))
                    for k, run in units:
                        q_value = values[arm][k]
                        if isinstance(q_value, dict):
                            q_value = q_value[db]
                        for index in range(QUERIES):
                            writer.writerow(
                                (
                                    arm,
                                    seed,
                                    db,
                                    k,
                                    run,
                                    f"{db}/q{index}",
                                    f"{db}-hash-{index}",
                                    format(TRUE, ".17g"),
                                    format(log_for(q_value), ".17g"),
                                )
                            )


def base_values():
    return {
        "A_erm": {0: 10.0, 5: 10.0, 20: 10.0},
        "B_fomaml": {0: 10.0, 5: 9.5, 20: 9.0},
        "C_joint": {0: 10.0, 5: 9.0, 20: 8.0},
    }


class SourceSummaryTests(unittest.TestCase):
    def test_promising_matrix_passes_all_sections(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "predictions.csv"
            write_predictions(path, base_values())
            summary = summarize_predictions(path, expected_qd=None)
            decision = summary["decision"]
            self.assertEqual(decision["decision"], "promising_source_screen")
            self.assertTrue(decision["six_one"]["passed"])
            self.assertTrue(decision["six_two"]["passed"])
            self.assertTrue(decision["six_three"]["passed"])
            self.assertAlmostEqual(
                decision["six_one"]["g_c_b_k20"]["mean"], 1 - 8 / 9, places=12
            )
            self.assertAlmostEqual(
                decision["six_two"]["g_self_c_k20"]["mean"], 0.2, places=12
            )
            self.assertAlmostEqual(
                decision["six_three"]["g_c_a_k20"]["mean"], 0.2, places=12
            )
            self.assertAlmostEqual(
                decision["g_self"]["A_erm"]["mean"], 0.0, places=12
            )
            self.assertAlmostEqual(
                decision["six_three"]["cross_ratio_c20_a0"]["42"], 0.8, places=12
            )
            self.assertAlmostEqual(
                summary["unit_metrics"]["C_joint"]["42"]["chembl"]["20"]["1"]["q95"],
                8.0,
                places=12,
            )
            self.assertEqual(
                summary["run_q95_sd"]["C_joint"]["42"]["chembl"]["20"], 0.0
            )
            self.assertIn("C_over_B", decision["ratios_q95"])
            write_summary(root / "out", summary)
            self.assertTrue((root / "out" / "summary.json").is_file())
            self.assertTrue((root / "out" / "decision.json").is_file())

    def test_no_candidate_when_joint_does_not_beat_post(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            values = base_values()
            values["C_joint"][20] = 10.0
            path = root / "predictions.csv"
            write_predictions(path, values)
            summary = summarize_predictions(path, expected_qd=None)
            self.assertEqual(summary["decision"]["decision"], "no_candidate")
            self.assertFalse(summary["decision"]["six_one"]["passed"])

    def test_per_database_guard_fails_when_one_database_regresses(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            values = base_values()
            values["C_joint"][20] = {
                "chembl": 10.0,
                "legalacts": 7.5,
                "telstra": 7.5,
            }
            path = root / "predictions.csv"
            write_predictions(path, values)
            summary = summarize_predictions(path, expected_qd=None)
            six_one = summary["decision"]["six_one"]
            self.assertTrue(six_one["g_c_b_k20"]["mean"] >= 0.05)
            self.assertFalse(six_one["passed"])
            self.assertTrue(any("chembl" in reason for reason in six_one["failures"]))

    def test_partial_signal_when_self_adaptation_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            values = base_values()
            values["C_joint"] = {0: 8.5, 5: 8.5, 20: 8.5}
            values["B_fomaml"] = {0: 10.0, 5: 9.5, 20: 9.0}
            path = root / "predictions.csv"
            write_predictions(path, values)
            summary = summarize_predictions(path, expected_qd=None)
            self.assertTrue(summary["decision"]["six_one"]["passed"])
            self.assertFalse(summary["decision"]["six_two"]["passed"])
            self.assertTrue(summary["decision"]["six_three"]["passed"])
            self.assertEqual(
                summary["decision"]["decision"], "partial_source_signal"
            )

    def test_missing_unit_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "predictions.csv"
            write_predictions(path, base_values())
            lines = path.read_text().splitlines()
            filtered = [
                line for line in lines if not line.startswith("C_joint,42,chembl,20,5,")
            ]
            path.write_text("\n".join(filtered) + "\n")
            with self.assertRaisesRegex(ValueError, "Missing prediction unit"):
                summarize_predictions(path, expected_qd=None)

    def test_k0_with_nonzero_run_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "predictions.csv"
            write_predictions(path, base_values())
            lines = path.read_text().splitlines()
            lines.append(
                "C_joint,42,chembl,0,1,chembl/q0,chembl-hash-0,100,2.302585092994046"
            )
            path.write_text("\n".join(lines) + "\n")
            with self.assertRaisesRegex(ValueError, "K0 predictions must use run=0"):
                summarize_predictions(path, expected_qd=None)

    def test_expected_qd_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "predictions.csv"
            write_predictions(path, base_values())
            with self.assertRaisesRegex(ValueError, "Q_d row count"):
                summarize_predictions(
                    path,
                    expected_qd={"chembl": 7, "legalacts": 6, "telstra": 6},
                )


if __name__ == "__main__":
    unittest.main()
