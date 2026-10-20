import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

_SOURCE = Path(__file__).resolve().parents[1] / "scripts/three_seed_audit_summary.py"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
_SPEC = importlib.util.spec_from_file_location("three_seed_audit_summary", _SOURCE)
summary_module = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(summary_module)

DATABASES = ("chembl", "legalacts", "telstra")
MODELS = ("hgb", "logistic")
GROUPS = (
    "in_database",
    "near_duplicate",
    "lodo",
    "lodo_accuracy",
    "lodo_majority",
)


def uniform(value):
    return {model: {database: value for database in DATABASES} for model in MODELS}


def audit_payload(in_database, lodo, target_reads=None):
    """Build a synthetic per-seed audit.json payload."""
    databases = {}
    lodo_block = {}
    for database in DATABASES:
        databases[database] = {
            "in_database": {
                model: {"auc_pooled_oof": in_database[model][database]}
                for model in MODELS
            },
            # The real audit runs the near-duplicate sensitivity pass for the
            # gradient-boosting model only.
            "near_duplicate_grouped_cv": {
                "hgb": {"auc_pooled_oof": in_database["hgb"][database]}
            },
        }
        lodo_block[database] = {
            model: {
                "auc": lodo[model][database],
                "accuracy": 0.5,
                "majority_accuracy": 0.5,
            }
            for model in MODELS
        }
    return {
        "target_databases_read": [] if target_reads is None else target_reads,
        "databases": databases,
        "leave_one_database_out": lodo_block,
        "inputs": {"predictions": {"sha256": "0" * 64}},
        "verdict": {"verdict": "x"},
    }


def write_audit(directory, name, payload):
    path = Path(directory) / name
    path.write_text(json.dumps(payload))
    return path


def make_seed(in_database=0.92, lodo=0.55, telstra=0.40):
    """One seed's parsed metrics, in the shape judge()/summarize() expect."""
    in_db = {db: in_database for db in DATABASES}
    lodo_block = {db: lodo for db in DATABASES}
    lodo_block["telstra"] = telstra
    seed = {}
    for group in GROUPS:
        seed[group] = {}
        for model in MODELS:
            source = in_db if group in ("in_database", "near_duplicate") else lodo_block
            if group == "near_duplicate" and model != "hgb":
                continue
            seed[group][model] = dict(source)
    return seed


class MeanSdTests(unittest.TestCase):
    def test_mean_and_sample_sd(self):
        stats = summary_module._mean_sd([0.9, 0.8, 0.7])
        self.assertAlmostEqual(stats["mean"], 0.8)
        # Sample SD (ddof=1) of 0.9/0.8/0.7 is 0.1.
        self.assertAlmostEqual(stats["std"], 0.1)
        self.assertAlmostEqual(stats["min"], 0.7)
        self.assertAlmostEqual(stats["max"], 0.9)
        self.assertEqual(stats["n"], 3)

    def test_single_seed_has_zero_sd(self):
        stats = summary_module._mean_sd([0.9167])
        self.assertAlmostEqual(stats["mean"], 0.9167)
        self.assertEqual(stats["std"], 0.0)

    def test_population_sd_is_not_used(self):
        # Population SD of 0.9/0.8/0.7 is 0.0816; the sample SD is 0.1.
        population = (sum((v - 0.8) ** 2 for v in (0.9, 0.8, 0.7)) / 3) ** 0.5
        self.assertNotAlmostEqual(
            summary_module._mean_sd([0.9, 0.8, 0.7])["std"], population
        )


class ReadAuditTests(unittest.TestCase):
    def test_metrics_are_extracted_per_database_and_model(self):
        with tempfile.TemporaryDirectory() as directory:
            payload = audit_payload(
                in_database={
                    "hgb": {db: 0.9 for db in DATABASES},
                    "logistic": {db: 0.8 for db in DATABASES},
                },
                lodo={
                    "hgb": {db: 0.4 for db in DATABASES},
                    "logistic": {db: 0.3 for db in DATABASES},
                },
            )
            path = write_audit(directory, "audit.json", payload)
            parsed = summary_module.read_audit(path)
        self.assertAlmostEqual(parsed["in_database"]["hgb"]["chembl"], 0.9)
        self.assertAlmostEqual(parsed["in_database"]["logistic"]["telstra"], 0.8)
        self.assertAlmostEqual(parsed["lodo"]["hgb"]["legalacts"], 0.4)
        self.assertAlmostEqual(parsed["near_duplicate"]["hgb"]["chembl"], 0.9)

    def test_audit_that_read_a_target_database_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            payload = audit_payload(uniform(0.9), uniform(0.4), target_reads=["stats"])
            path = write_audit(directory, "audit.json", payload)
            with self.assertRaises(ValueError):
                summary_module.read_audit(path)


class SummarizeTests(unittest.TestCase):
    def test_summary_reports_mean_and_sd_over_seeds(self):
        seeds = {"42": make_seed(in_database=0.90), "43": make_seed(in_database=0.92)}
        result = summary_module.summarize(seeds)
        stats = result["in_database"]["hgb"]["chembl"]
        self.assertAlmostEqual(stats["mean"], 0.91)
        self.assertAlmostEqual(stats["std"], 0.02 / (2**0.5))
        self.assertEqual(stats["n"], 2)

    def test_unmeasured_model_is_null_not_zero(self):
        seeds = {"42": make_seed(), "43": make_seed()}
        result = summary_module.summarize(seeds)
        # The near-duplicate pass is gradient boosting only.
        self.assertIsNone(result["near_duplicate"]["logistic"]["chembl"])
        self.assertEqual(result["near_duplicate"]["hgb"]["chembl"]["n"], 2)


class JudgeTests(unittest.TestCase):
    def test_all_thresholds_met_is_confirmed(self):
        verdict = summary_module.judge(
            {"42": make_seed(), "43": make_seed(), "44": make_seed()}
        )
        self.assertTrue(verdict["all_pass"])
        self.assertEqual(verdict["conclusion"], "confirmed")

    def test_low_in_database_auc_fails(self):
        seeds = {
            "42": make_seed(),
            "43": make_seed(),
            "44": make_seed(),
        }
        seeds["44"]["in_database"]["hgb"]["chembl"] = 0.80
        verdict = summary_module.judge(seeds)
        self.assertFalse(verdict["all_pass"])
        self.assertEqual(verdict["conclusion"], "inconclusive")
        self.assertFalse(
            verdict["in_database_auc_min_over_seeds_and_databases"]["pass"]
        )

    def test_high_lodo_auc_fails(self):
        seeds = {
            "42": make_seed(),
            "43": make_seed(),
            "44": make_seed(),
        }
        seeds["43"]["lodo"]["hgb"]["chembl"] = 0.70
        verdict = summary_module.judge(seeds)
        self.assertFalse(verdict["all_pass"])
        self.assertFalse(verdict["lodo_auc_max_over_seeds_and_databases"]["pass"])

    def test_telstra_direction_needs_two_of_three_seeds(self):
        seeds = {
            "42": make_seed(),
            "43": make_seed(telstra=0.55),
            "44": make_seed(telstra=0.55),
        }
        verdict = summary_module.judge(seeds)
        self.assertFalse(verdict["telstra_lodo_below_half"]["pass"])
        self.assertEqual(verdict["telstra_lodo_below_half"]["seeds"], ["42"])
        self.assertEqual(verdict["conclusion"], "inconclusive")

    def test_two_of_three_seeds_below_half_is_enough(self):
        seeds = {
            "42": make_seed(),
            "43": make_seed(),
            "44": make_seed(telstra=0.55),
        }
        verdict = summary_module.judge(seeds)
        self.assertTrue(verdict["telstra_lodo_below_half"]["pass"])
        self.assertTrue(verdict["all_pass"])


class WriteJsonTests(unittest.TestCase):
    def test_write_is_atomic_and_json_safe(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nested" / "s.json"
            summary_module.write_json(path, {"a": 1})
            self.assertEqual(json.loads(path.read_text()), {"a": 1})
            self.assertFalse(Path(str(path) + ".tmp").exists())
            with self.assertRaises(ValueError):
                summary_module.write_json(path, {"a": float("nan")})


if __name__ == "__main__":
    unittest.main()
