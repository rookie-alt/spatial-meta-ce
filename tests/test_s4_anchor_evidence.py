import copy
import unittest
from unittest.mock import patch

import numpy as np

from scripts import s4_anchor_evidence as ev


HASH = "a" * 64
FROZEN_PREFLIGHT = {
    "python": "3.11.4 fixture", "torch": "2.1.0+cu121", "numpy": "1.26.1",
    "sqlglot": "18.17.0", "gpu_uuid": "GPU-fixture", "config_sha256": HASH,
    "code_sha256": {"scripts/s4_anchor_evidence.py": HASH},
    "sources": {db: {"manifest_hash": HASH, "rows": 3,
        "file_sha256": {name: HASH for name in ("manifest.json", "cards.npy", "offsets.npy", "group0.npy", "group1.npy", "group2.npy", "group3.npy")}}
        for db in ev.SOURCES},
}


def rows() -> dict:
    return {
        db: [
            {
                "query_hash": f"q{i}",
                "true": float(t),
                "n_tables": n,
                "log_pred": p,
                "pg_input_log": float(np.float32(np.log(max(t, 1)))),
                "pg_reference_log": float(np.log(max(t, 1))),
            }
            for i, (t, n, p) in enumerate(((0, 1, -2), (10, 4, 2.0), (100, 5, 5.0)))
        ]
        for db in ev.DATABASES
    }


def endpoint(q: float = 10.0, formal: bool = True) -> dict:
    return {
        "per_database": {
            db: {
                b: {
                    "queries": ev.COVERAGE[db][i] if formal else (3, 2)[i],
                    "q50": 1.0,
                    "q95": q,
                }
                for i, b in enumerate(ev.BUCKETS)
            }
            for db in ev.DATABASES
        },
        "identity_digest": HASH,
        "pg_identity_digest": HASH,
        "unique_queries": 149504 if formal else 9,
        "macro_four_plus_q95": q,
        "macro_all_table_q95": q,
    }


def record(arm: str = "A_anchor1", formal: bool = True) -> dict:
    steps = 4000 if formal else 2
    return {
        "arm": arm,
        "config": {
            **copy.deepcopy(ev.FROZEN_CONFIG),
            "steps": steps,
            **({"support_sizes": [5], "query_size": 8, "validate_every": 1, "validation_runs": 1} if not formal else {}),
            "purpose": "research" if formal else "gpu-smoke-not-research",
            "model": {**ev.FROZEN_MODEL, "pg_output_scale": float(arm == "A_anchor1")},
        },
        "optimizer_steps": steps,
        "counts": {
            "optimizer_steps": steps,
            "backward_calls": steps * 4,
            "support_presentations": 0,
            "query_presentations": 1621915 if formal else 104,
        },
        "sample_plan_episodes": steps * 4,
        "sample_plan_digest": ev.SAMPLE_DIGEST if formal else HASH,
        "validation_plan_hash": ev.VALIDATION_HASH if formal else HASH,
        "initialization_hash": ev.INITIAL_HASH,
        "parameters": 132545,
        "manifest_hashes": {
            db: FROZEN_PREFLIGHT["sources"][db]["manifest_hash"] for db in ev.SOURCES
        },
        "checkpoint_sha256": HASH,
        "endpoint_rows_sha256": HASH,
        "checkpoint_checks": {
            "roundtrip_exact": True,
            "optimizer_changed_weights": True,
            "checkpoint_steps": steps,
            "cpu_device_max_abs_difference": 0.0,
        },
        "endpoint": endpoint(formal=formal),
    }


def preflight() -> dict:
    return {**copy.deepcopy(FROZEN_PREFLIGHT), "protocol_sha256": ev.PROTOCOL_HASH}


class EvidenceTests(unittest.TestCase):
    def setUp(self) -> None:
        patcher = patch.object(ev, "SOURCE_HASH", ev._digest(FROZEN_PREFLIGHT["sources"]))
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_formal_run_passes(self) -> None:
        self.assertTrue(ev.verify_run(record(), preflight(), True)["passed"])

    def test_smoke_budget(self) -> None:
        self.assertTrue(
            ev.verify_run(record(formal=False), preflight(), False)["passed"]
        )
        with self.assertRaises(ValueError):
            ev.verify_run(record(formal=False), preflight(), True)

    def test_frozen_fields(self) -> None:
        for path, value in (
            ("steps", 3999),
            ("outer_lr", 0.001),
            ("loss", "mse"),
            ("seed", True),
        ):
            r = record()
            r["config"][path] = value
            with self.assertRaises(ValueError):
                ev.verify_run(r, preflight(), True)

    def test_scale_rejected(self) -> None:
        for scale in (True, 0.5, float("nan"), 0.0):
            r = record()
            r["config"]["model"]["pg_output_scale"] = scale
            with self.assertRaises(ValueError):
                ev.verify_run(r, preflight(), True)

    def test_frozen_hashes_and_counts(self) -> None:
        for key in (
            "sample_plan_digest",
            "initialization_hash",
            "validation_plan_hash",
        ):
            r = {**record(), key: HASH}
            with self.assertRaises(ValueError):
                ev.verify_run(r, preflight(), True)
        for key in (
            "query_presentations",
            "backward_calls",
            "optimizer_steps",
            "support_presentations",
        ):
            r = record()
            r["counts"][key] += 1
            with self.assertRaises(ValueError):
                ev.verify_run(r, preflight(), True)

    def test_source_coverage_and_protocol(self) -> None:
        p = preflight()
        p["sources"].pop("chembl")
        with self.assertRaises(ValueError):
            ev.verify_run(record(), p, True)
        with self.assertRaises(ValueError):
            ev.verify_run(record(), {**preflight(), "protocol_sha256": HASH}, True)

    def test_checkpoint_checks(self) -> None:
        r = record()
        r["checkpoint_checks"]["roundtrip_exact"] = False
        with self.assertRaises(ValueError):
            ev.verify_run(r, preflight(), True)

    def test_pair_passes(self) -> None:
        self.assertTrue(ev.verify_pair([record(), record("B_anchor0")], True)["passed"])

    def test_pair_rejects_missing_duplicate_and_changed_identity(self) -> None:
        for pair in (
            [record()],
            [record(), record()],
            [record(), {**record("B_anchor0"), "sample_plan_digest": HASH}],
        ):
            with self.assertRaises(ValueError):
                ev.verify_pair(pair, True)
        b = record("B_anchor0")
        b["endpoint"]["pg_identity_digest"] = "b" * 64
        with self.assertRaises(ValueError):
            ev.verify_pair([record(), b], True)

    def test_choose_exact_boundary(self) -> None:
        a, b = record(), record("B_anchor0")
        b["endpoint"] = endpoint(9.0)
        self.assertEqual(ev.choose([a, b])["decision"], "promising_single_seed")
        self.assertAlmostEqual(ev.choose([a, b])["relative_improvement"], 0.1)

    def test_choose_guardrail_boundary(self) -> None:
        a, b = record(), record("B_anchor0")
        b["endpoint"] = endpoint(9.0)
        b["endpoint"]["per_database"]["chembl"]["all_tables"]["q95"] = 11.0
        b["endpoint"]["macro_all_table_q95"] = (11 + 9 + 9) / 3
        self.assertEqual(ev.choose([a, b])["decision"], "promising_single_seed")
        b["endpoint"]["per_database"]["chembl"]["all_tables"]["q95"] = float(np.nextafter(11.0, np.inf))
        b["endpoint"]["macro_all_table_q95"] = (
            b["endpoint"]["per_database"]["chembl"]["all_tables"]["q95"] + 18
        ) / 3
        self.assertEqual(ev.choose([a, b])["decision"], "no_candidate")

    def test_choose_invalid_metrics(self) -> None:
        for value in (True, float("inf"), 0.9):
            b = record("B_anchor0")
            b["endpoint"]["per_database"]["chembl"]["all_tables"]["q50"] = value
            with self.assertRaises(ValueError):
                ev.choose([record(), b])

    def test_baseline_reproduction(self) -> None:
        old = {**record(), "arm": "A_w64_smooth"}
        self.assertTrue(ev.verify_baseline(record(), {"runs": [old]})["passed"])
        r = record()
        r["endpoint"] = endpoint(10.1)
        with self.assertRaisesRegex(ValueError, "reproduction_blocked"):
            ev.verify_baseline(r, {"runs": [old]})

    def test_rows_metrics_independent(self) -> None:
        output = ev.analyze_rows(rows(), 1)
        cell = next(
            c
            for c in output
            if c["database"] == "chembl" and c["bucket"] == "all_tables"
        )
        self.assertEqual(cell["n"], 3)
        self.assertAlmostEqual(cell["q50"], np.exp(abs(2 - np.log(10))))
        self.assertAlmostEqual(
            cell["raw_log_mae"], (2 + abs(2 - np.log(10)) + abs(5 - np.log(100))) / 3
        )
        self.assertLess(cell["identity_max_abs_error"], 1e-12)

    def test_strata_minimum(self) -> None:
        output = ev.analyze_rows(rows(), 0)
        self.assertEqual(len(output), 3 * (2 + 7 + 4 + 28 + 6))
        cells = [c for c in output if c["bucket"] == "true=0"]
        self.assertTrue(
            all(c["status"] == "below_min_cell_n" and "q95" not in c for c in cells)
        )

    def test_fp32_direction_is_not_reference_direction(self) -> None:
        rr = rows()
        for db in rr:
            rr[db] = [
                {
                    "query_hash": "z",
                    "true": 10.0,
                    "n_tables": 4,
                    "log_pred": 0.0,
                    "pg_reference_log": float(np.log(10.0)),
                    "pg_input_log": float(np.float32(np.log(10.0))),
                }
            ]
        cell = ev.analyze_rows(rr, 0)[0]
        self.assertEqual(cell["pg_reference_exact_fraction"], 1.0)
        self.assertNotEqual(cell["pg_input_exact_fraction"], 1.0)
        self.assertEqual(cell["head_output_mode"], "direct_log_prediction")
        self.assertEqual(cell["head_output_mean"], 0.0)

    def test_bad_rows_fail_closed(self) -> None:
        for key, value in (
            ("true", -1),
            ("true", True),
            ("n_tables", True),
            ("log_pred", float("nan")),
            ("pg_input_log", float("inf")),
            ("database", "imdb"),
        ):
            rr = rows()
            rr["chembl"][0][key] = value
            with self.assertRaises(ValueError):
                ev.analyze_rows(rr, 1)
        rr = rows()
        rr["chembl"].append(rr["chembl"][0])
        with self.assertRaises(ValueError):
            ev.analyze_rows(rr, 1)

    def test_q_overflow_and_missing_bucket(self) -> None:
        rr = rows()
        rr["chembl"][0]["log_pred"] = 1000.0
        with self.assertRaises(ValueError):
            ev.analyze_rows(rr, 1)
        rr = rows()
        rr["chembl"] = rr["chembl"][:1]
        with self.assertRaises(ValueError):
            ev.analyze_rows(rr, 1)

    def test_verify_rows_and_order_pairing(self) -> None:
        rr = rows()
        e = ev.verify_rows(rr, 1, None, formal=False)["endpoint"]
        self.assertTrue(ev.verify_rows(rr, 1, e, formal=False)["passed"])
        b = copy.deepcopy(rr)
        b["chembl"] = b["chembl"][::-1]
        self.assertNotEqual(
            ev.verify_rows(b, 0, None, False)["endpoint"]["pg_identity_digest"],
            e["pg_identity_digest"],
        )
        with self.assertRaises(ValueError):
            ev.verify_rows(rr, 1, e, formal=True)

    def test_verify_rows_rejects_forged_endpoint(self) -> None:
        e = ev.verify_rows(rows(), 1, None, False)["endpoint"]
        e["per_database"]["chembl"]["all_tables"]["q95"] += 1
        with self.assertRaises(ValueError):
            ev.verify_rows(rows(), 1, e, False)


if __name__ == "__main__":
    unittest.main()
