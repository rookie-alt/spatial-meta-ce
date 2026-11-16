import csv
import importlib
import math
import tempfile
import unittest
from pathlib import Path


def completed(arm, seed, db, k, run, lr, q95):
    return {
        "arm": arm,
        "stream": seed,
        "database": db,
        "k": k,
        "run": run,
        "lr": lr,
        "population": "report",
        "status": "completed",
        "n": 2,
        "metrics": {"q50": q95 / 2, "q95": q95},
    }


class ResultTests(unittest.TestCase):
    def module(self):
        return importlib.import_module("spatial_meta_ce.price_meta123_results")

    def test_raw_native_csv_recomputed_and_identity_mismatch_refused(self):
        module = self.module()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "predictions.csv"
            with path.open("w", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(module.FIELDS)
                for index, (true, pred) in enumerate(((2, 4), (4, 4))):
                    writer.writerow(
                        (
                            "P",
                            0,
                            "chembl",
                            0,
                            0,
                            0.0,
                            f"q{index}",
                            str(index) * 64,
                            true,
                            math.log1p(pred) + 1,
                        )
                    )
            expected = [("q0", "0" * 64, 2.0), ("q1", "1" * 64, 4.0)]
            unit = module.compute_unit(path, expected)
            self.assertAlmostEqual(unit["metrics"]["q95"], 1.95)
            with self.assertRaises(ValueError):
                module.compute_unit(path, expected[::-1])

    def test_lr_selection_uses_mean_support_then_log_and_rejects_incomplete(self):
        module = self.module()
        units = []
        for db in ("chembl", "legalacts", "telstra"):
            baseline = completed("P", 0, db, 0, 0, 0.0, 10)
            baseline["population"] = "calibration"
            units.append(baseline)
            for lr, score in ((1e-7, 20), (1e-6, 10), (1e-5, 30)):
                for run in range(5):
                    row = completed("P", 0, db, 20, run, lr, score)
                    row["population"] = "calibration"
                    units.append(row)
        self.assertEqual(module.select_lr(units, "P")["lr"], 1e-6)
        with self.assertRaises(ValueError):
            module.select_lr(units[:-1], "P")
        with self.assertRaises(ValueError):
            module.select_lr([u for u in units if u["k"] != 0], "P")

    def matrix(self, q_m20=7.0):
        units = []
        for arm, seeds in (("P", (0,)), ("E", (42, 43, 44)), ("M", (42, 43, 44))):
            for seed in seeds:
                for db in ("chembl", "legalacts", "telstra"):
                    for k in (0, 5, 20):
                        q = 10.0 if k == 0 else (q_m20 if arm == "M" else 9.0)
                        for run in range(1) if k == 0 else range(5):
                            units.append(
                                completed(
                                    arm, seed, db, k, run, 0.0 if k == 0 else 1e-6, q
                                )
                            )
        return units

    def test_complete_matrix_passes_and_missing_duplicate_denominator_fail(self):
        module = self.module()
        units = self.matrix()
        decision = module.aggregate_report(units, 1e-6, 1e-6)
        self.assertEqual(decision["decision"], "promising_source_screen")
        for changed in (units[:-1], units + [units[0]]):
            with self.assertRaises(ValueError):
                module.aggregate_report(changed, 1e-6, 1e-6)
        units[-1]["n"] = 3
        with self.assertRaises(ValueError):
            module.aggregate_report(units, 1e-6, 1e-6)

    def test_no_gain_is_not_invalid_or_promising(self):
        module = self.module()
        decision = module.aggregate_report(self.matrix(12.0), 1e-6, 1e-6)
        self.assertEqual(decision["decision"], "no_source_gain")

    def test_paired_raw_predictions_measure_query_worsening(self):
        module = self.module()
        with tempfile.TemporaryDirectory() as directory:
            paths = [Path(directory) / f"{side}.csv" for side in ("a", "b")]
            for path, pred in zip(paths, (4, 2)):
                with path.open("w", newline="") as stream:
                    writer = csv.writer(stream)
                    writer.writerow(module.FIELDS)
                    writer.writerow(
                        (
                            "P",
                            0,
                            "chembl",
                            0,
                            0,
                            0.0,
                            "q0",
                            "0" * 64,
                            2,
                            math.log1p(pred) + 1,
                        )
                    )
            value = module.compare_predictions(*paths)
            self.assertEqual(value["n"], 1)
            self.assertEqual(value["fraction_worsened"], 1.0)
            self.assertAlmostEqual(value["mean_delta_log_error"], math.log(2))

    def test_failing_candidate_cannot_be_dropped_and_every_candidate_failure_stops(
        self,
    ):
        module = self.module()
        units = []
        for db in module.DATABASES:
            baseline = completed("P", 0, db, 0, 0, 0.0, 10)
            baseline["population"] = "calibration"
            units.append(baseline)
            for lr in module.LRS:
                for run in range(5):
                    units.append(
                        module.make_failed_unit(
                            "P", 0, db, 20, run, lr, "calibration", 2000, "nonfinite"
                        )
                    )
        with self.assertRaisesRegex(ValueError, "rejected"):
            module.select_lr(units, "P")

    def test_strong_baseline_and_per_database_guards_are_enforced(self):
        module = self.module()
        units = self.matrix()
        for seed in (42, 43, 44):
            for db in module.DATABASES:
                for run in range(5):
                    units.append(completed("E_alt", seed, db, 20, run, 1e-7, 6.0))
        self.assertFalse(
            module.aggregate_report(units, 1e-6, 1e-7)["gates"]["relative_gain"]
        )


if __name__ == "__main__":
    unittest.main()
