import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from spatial_meta_ce.model import SetEstimator

from scripts import tail_aligned_screen_evidence as evidence

# scripts/ is not a package, so gate_usage is loaded by path.
_SOURCE = Path(__file__).resolve().parents[1] / "scripts/gate_usage.py"
_SPEC = importlib.util.spec_from_file_location("gate_usage", _SOURCE)
gate_usage = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(gate_usage)

VALIDATION = ("chembl", "legalacts", "telstra")


def endpoint(macro, per_db):
    return {
        "macro_four_plus_q95": macro,
        "per_database": {
            db: {
                "all_tables": {"q95": per_db[db][0]},
                "four_plus_tables": {"q95": per_db[db][1]},
            }
            for db in VALIDATION
        },
    }


def baseline():
    return endpoint(44.0, {db: (60.0, 44.0) for db in VALIDATION})


def arm(macro_scale=1.0, db_scale=None):
    per_db = {
        db: (60.0 * (db_scale or {}).get(db, 1.0), 44.0 * (db_scale or {}).get(db, 1.0))
        for db in VALIDATION
    }
    return endpoint(44.0 * macro_scale, per_db)


class ScreenDecisionTests(unittest.TestCase):
    """Doc 93 §0/§3 freeze: macro >= 10% and every per-db guard <= 1.10x."""

    def test_candidate_requires_both_macro_and_guard(self):
        records = {"baseline": baseline(), "A": arm(0.85), "B": arm(0.95)}
        decision = evidence.choose_candidates(records, "baseline")
        self.assertEqual(decision["decision"], "promising_single_seed")
        self.assertEqual(decision["candidates"], ["A"])
        self.assertAlmostEqual(decision["primary_improvement"]["A"], 0.15, places=6)

    def test_guard_violation_blocks_an_otherwise_large_gain(self):
        records = {"baseline": baseline(), "A": arm(0.80, {"legalacts": 1.2})}
        self.assertEqual(
            evidence.choose_candidates(records, "baseline")["candidates"], []
        )

    def test_exact_thresholds_are_inclusive(self):
        records = {"baseline": baseline(), "A": arm(0.90, {"chembl": 1.10})}
        self.assertEqual(
            evidence.choose_candidates(records, "baseline")["candidates"], ["A"]
        )

    def test_just_missing_the_line_is_not_a_candidate(self):
        records = {"baseline": baseline(), "A": arm(0.9001)}
        self.assertEqual(
            evidence.choose_candidates(records, "baseline")["decision"], "no_candidate"
        )

    def test_baseline_must_be_present_and_named(self):
        with self.assertRaises(ValueError):
            evidence.choose_candidates({"A": arm()}, "baseline")
        with self.assertRaises(ValueError):
            evidence.choose_candidates({"baseline": baseline()}, "A")

    def test_nonfinite_metrics_rejected(self):
        bad = endpoint(float("nan"), {db: (60.0, 44.0) for db in VALIDATION})
        with self.assertRaises(ValueError):
            evidence.choose_candidates({"baseline": baseline(), "A": bad}, "baseline")

    def test_candidates_are_ranked_by_macro_then_arm_name(self):
        records = {
            "baseline": baseline(),
            "A": arm(0.80),
            "B": arm(0.80),
            "C": arm(0.75),
        }
        self.assertEqual(
            evidence.choose_candidates(records, "baseline")["candidates"],
            ["C", "A", "B"],
        )


class GateUsageTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(3)

    def test_requires_a_gated_model(self):
        with self.assertRaises(ValueError):
            gate_usage.gate_report(SetEstimator(), _Pool(), "cpu")

    def test_report_bounds_and_shapes(self):
        model = SetEstimator(gate="sigmoid")
        with torch.no_grad():
            model.gate[-1].bias.fill_(0.5)
        report = gate_usage.gate_report(model, _Pool(), "cpu")
        self.assertEqual(report["queries"], 6)
        for key in ("mean", "p05", "p50", "p95", "min", "max", "fraction_below_0_1"):
            self.assertIn(key, report)
            self.assertTrue(0.0 <= report[key] <= 1.0, key)
        # bias=0.5 puts sigmoid above 0.5; the gate weight adds its own offset.
        self.assertGreater(report["min"], 0.5)
        self.assertLess(report["max"], 1.0)

    def test_report_includes_per_database_breakdown(self):
        report = gate_usage.gate_report(SetEstimator(gate="sigmoid"), _Pool(), "cpu")
        self.assertEqual(set(report["per_database"]), set(VALIDATION))
        for db in VALIDATION:
            self.assertEqual(report["per_database"][db]["queries"], 2)

    def test_constant_residual_gate_still_measured(self):
        # A gate that never varies must still report a finite, bounded distribution.
        model = SetEstimator(gate="sigmoid")
        with torch.no_grad():
            model.gate[-1].weight.zero_()
            model.gate[-1].bias.fill_(-30.0)
        report = gate_usage.gate_report(model, _Pool(), "cpu")
        self.assertLess(report["max"], 1e-6)


class _FakeStore:
    """Two unique queries per database, three-table joins."""

    def __init__(self):
        self.rows = [
            {"id": f"{db}/{i}", "query_hash": f"{db}-{i}"}
            for db in VALIDATION
            for i in range(2)
        ]
        self.cards = [[10.0 + i, 2.0] for i in range(len(self.rows))]

    def unique_indices(self):
        import numpy as np

        return np.arange(len(self.rows), dtype=np.int64)

    def batch(self, indices, device):
        from spatial_meta_ce.data import batch_samples

        samples = [
            {
                "groups": [
                    torch.ones(2, 40),
                    torch.ones(2, 40),
                    torch.ones(3, 4),
                    torch.ones(0, 43),
                ],
                "true": float(self.cards[int(i)][0]),
                "pg": float(self.cards[int(i)][1]),
                "id": self.rows[int(i)]["id"],
                "hash": self.rows[int(i)]["query_hash"],
            }
            for i in indices
        ]
        return batch_samples(samples, device)


class _FakeDbStore(_FakeStore):
    """Same fixture narrowed to one database, so pools match a real StorePool."""

    def __init__(self, db):
        self.rows = [{"id": f"{db}/{i}", "query_hash": f"{db}-{i}"} for i in range(2)]
        self.cards = [[10.0 + i, 2.0] for i in range(2)]


class _Pool(_FakeStore):
    def get(self, split, db):
        return _FakeDbStore(db)


class ProtocolFileTests(unittest.TestCase):
    def test_arm_matrix_is_frozen_and_documented(self):
        root = Path(__file__).resolve().parents[1]
        config = json.loads((root / "configs/tail_aligned_screen.json").read_text())
        self.assertEqual(config["schema"], 1)
        self.assertEqual(
            set(config["arms"]),
            {"baseline", "log_l1", "pinball_08", "pinball_09", "gate_log_l1"},
        )
        self.assertEqual(config["baseline"], "baseline")
        self.assertEqual(
            config["guard"], {"macro_improvement": 0.10, "per_database": 1.10}
        )
        for name, arm_config in config["arms"].items():
            if arm_config["loss"] == "pinball_tau":
                self.assertIn(arm_config["pinball_tau"], (0.8, 0.9), name)
            else:
                self.assertNotIn("pinball_tau", arm_config, name)
        self.assertEqual(config["arms"]["gate_log_l1"]["model"]["gate"], "sigmoid")
        for name, arm_config in config["arms"].items():
            if name != "gate_log_l1":
                self.assertNotIn("gate", arm_config.get("model", {}), name)
        self.assertNotIn("stats", json.dumps(config["arms"]))
        self.assertNotIn("genome", json.dumps(config["arms"]))

    def test_write_json_is_atomic_and_rejects_nan(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "x.json"
            evidence.write_json(path, {"a": 1})
            self.assertEqual(json.loads(path.read_text()), {"a": 1})
            self.assertFalse((Path(d) / "x.json.tmp").exists())
            with self.assertRaises(ValueError):
                evidence.write_json(path, {"a": float("nan")})


class SupervisionTests(unittest.TestCase):
    def test_supervise_kills_a_hanging_child(self):
        import sys

        with tempfile.TemporaryDirectory() as d:
            output = Path(d)
            with self.assertRaises(Exception):
                evidence.supervise(
                    [sys.executable, "-c", "import time; time.sleep(30)"],
                    output,
                    timeout=1,
                    grace=1,
                )
            status = json.loads((output / "status.json").read_text())
            self.assertEqual(status["status"], "failed")
            self.assertIn("timed out", status["error"].lower())

    def test_supervise_records_child_failure(self):
        import sys

        with tempfile.TemporaryDirectory() as d:
            output = Path(d)
            with self.assertRaises(Exception):
                evidence.supervise(
                    [sys.executable, "-c", "raise SystemExit(3)"],
                    output,
                    timeout=30,
                    grace=1,
                )
            status = json.loads((output / "status.json").read_text())
            self.assertEqual(status["status"], "failed")

    def test_supervise_passes_a_clean_child(self):
        import sys

        with tempfile.TemporaryDirectory() as d, patch.object(evidence, "_log"):
            evidence.supervise(
                [sys.executable, "-c", "print('ok')"], Path(d), timeout=30, grace=1
            )


if __name__ == "__main__":
    unittest.main()
