"""Independent S6 fixtures and fail-closed evidence gates."""

import copy
import unittest
from unittest.mock import patch

from scripts import s4_anchor_evidence as s4
from scripts import s6_tail_evidence as s6

H = "a" * 64


def endpoint(value=10.0, formal=True):
    per = {
        db: {
            bucket: {
                "queries": s4.COVERAGE[db][i] if formal else 2,
                "q50": value,
                "q95": value,
            }
            for i, bucket in enumerate(s4.BUCKETS)
        }
        for db in s4.DATABASES
    }
    return {
        "per_database": per,
        "identity_digest": H,
        "pg_identity_digest": H,
        "unique_queries": sum(x["all_tables"]["queries"] for x in per.values()),
        "macro_all_table_q95": value,
        "macro_four_plus_q95": value,
    }


def record(arm="A_mean", formal=True):
    steps = 4000 if formal else 2
    config = {
        **s4.FROZEN_CONFIG,
        "tail_risk_mode": "mean" if arm == "A_mean" else "mean_tail10",
        "model": {**s4.FROZEN_MODEL, "pg_output_scale": 1.0},
    }
    if not formal:
        config = {
            **config,
            "steps": 2,
            "support_sizes": [5],
            "query_size": 8,
            "validate_every": 1,
            "validation_runs": 1,
            "purpose": "gpu-smoke-not-research",
        }
    counts = {
        "optimizer_steps": steps,
        "backward_calls": steps * 4,
        "support_presentations": 0,
        "query_presentations": 1621915 if formal else 104,
    }
    end = endpoint(formal=formal)
    initial = endpoint(formal=formal)
    return {
        "arm": arm,
        "config": config,
        "counts": counts,
        "optimizer_steps": steps,
        "sample_plan_episodes": steps * 4,
        "sample_plan_digest": s4.SAMPLE_DIGEST,
        "validation_plan_hash": s4.VALIDATION_HASH,
        "initialization_hash": s4.INITIAL_HASH,
        "parameters": 132545,
        "manifest_hashes": dict.fromkeys(s4.SOURCES, H),
        "checkpoint_sha256": H,
        "endpoint_rows_sha256": H,
        "initial_endpoint_rows_sha256": H,
        "gpu_uuid": "GPU-fixture",
        "checkpoint_checks": {
            "roundtrip_exact": True,
            "optimizer_changed_weights": True,
            "checkpoint_steps": steps,
            "cpu_device_max_abs_difference": 0,
        },
        "endpoint": end,
        "initial_endpoint": initial,
        "pg_identity_digest": H,
        "initial_pg_identity_digest": H,
        "rows_verification": {
            "passed": True,
            "formal": formal,
            "endpoint": copy.deepcopy(end),
        },
        "initial_rows_verification": {
            "passed": True,
            "formal": formal,
            "endpoint": copy.deepcopy(initial),
        },
        "loss_verification": {
            "passed": True,
            "mode": config["tail_risk_mode"],
            "episodes": steps * 4,
            "presentations": counts["query_presentations"],
            "sample_plan_digest": s4.SAMPLE_DIGEST,
            "snapshot_rows": 16 if formal else 8,
            "audit_sha256": H,
            "snapshots_sha256": H,
        },
        "summary": {
            "loss_evidence": {
                "audit_rows": steps * 4,
                "snapshot_rows": 16 if formal else 8,
                "mode": config["tail_risk_mode"],
            },
            "last_step": steps,
            "counts": copy.deepcopy(counts),
            "parameters": 132545,
            "gradient_stats": {
                "count": steps,
                "total_norm": 4.0 * steps,
                "max_norm": 4.0,
                "clipped_count": steps,
            },
        },
    }


def preflight():
    return {
        "protocol_sha256": s6.PROTOCOL_HASH,
        "sources": {db: {"manifest_hash": H} for db in s4.SOURCES},
        "torch": "2.1.0+cu121",
        "numpy": "1.26.1",
        "sqlglot": "18.17.0",
        "python": "3.11.4 fixture",
        "gpu_uuid": "GPU-fixture",
        "config_sha256": H,
        "code_sha256": {"fixture": H},
    }


class Gates(unittest.TestCase):
    def test_valid_formal_and_smoke(self):
        p = preflight()
        with patch.object(s6, "SOURCE_HASH", s4._digest(p["sources"])):
            for formal in (True, False):
                self.assertTrue(
                    s6.verify_run(record(formal=formal), p, formal)["passed"]
                )

    def test_bad_arm_config_sha_and_evidence(self):
        p = preflight()
        changes = [
            ("arm", "A_uniform"),
            ("checkpoint_sha256", "bad"),
            ("gpu_uuid", "GPU-other"),
            ("initialization_hash", H),
            ("rows_verification", {"passed": False}),
            ("sample_plan_digest", H),
            ("initial_endpoint_rows_sha256", None),
        ]
        with patch.object(s6, "SOURCE_HASH", s4._digest(p["sources"])):
            for key, value in changes:
                with self.subTest(key=key), self.assertRaises(ValueError):
                    s6.verify_run({**record(), key: value}, p, True)
            for key, value in (
                ("purpose", "production"),
                ("tail_risk_mode", "mean_tail10"),
                ("steps", 3999),
                ("seed", True),
            ):
                r = record()
                r = {**r, "config": {**r["config"], key: value}}
                with self.subTest(key=key), self.assertRaises(ValueError):
                    s6.verify_run(r, p, True)
            for key in ("protocol_sha256", "config_sha256", "torch", "python"):
                with self.assertRaises(ValueError):
                    s6.verify_run(record(), {**p, key: "bad"}, True)
        with self.assertRaises(ValueError):
            s6.verify_run(record(), p, True)

    def test_pair_scale_and_initial_endpoint(self):
        a, b = record(), record("B_mean_tail10")
        self.assertTrue(s6.verify_pair([a, b], True)["passed"])
        config = {
            **b["config"],
            "model": {**b["config"]["model"], "pg_output_scale": 0.0},
        }
        with self.assertRaises(ValueError):
            s6.verify_pair([a, {**b, "config": config}], True)
        initial = endpoint(11)
        with self.assertRaises(ValueError):
            s6.verify_pair(
                [
                    a,
                    {
                        **b,
                        "initial_endpoint": initial,
                        "initial_rows_verification": {
                            **b["initial_rows_verification"],
                            "endpoint": initial,
                        },
                    },
                ],
                True,
            )

    def test_baseline(self):
        old = {"runs": [{"arm": "A_uniform", "endpoint": endpoint()}]}
        self.assertTrue(s6.verify_baseline(record(), old)["passed"])
        for history in (
            {"runs": []},
            {"runs": [{"arm": "bad", "endpoint": endpoint()}]},
            {"runs": [{"arm": "A_uniform", "endpoint": endpoint(10.01)}]},
        ):
            with self.assertRaisesRegex(ValueError, "reproduction_blocked"):
                s6.verify_baseline(record(), history)

    def test_threshold_boundaries_and_mandatory(self):
        a, b = record(), record("B_mean_tail10")
        for q50, q95, decision in (
            (11.0, 11.0, "promising_single_seed"),
            (11.0, 11.000001, "no_candidate"),
            (11.000001, 11.000001, "no_candidate"),
        ):
            end = endpoint(9)
            cell = end["per_database"]["chembl"]["all_tables"]
            per = {
                **end["per_database"],
                "chembl": {
                    **end["per_database"]["chembl"],
                    "all_tables": {**cell, "q50": q50, "q95": q95},
                },
            }
            end = {**end, "per_database": per, "macro_all_table_q95": (q95 + 18) / 3}
            candidate = {
                **b,
                "endpoint": end,
                "rows_verification": {**b["rows_verification"], "endpoint": end},
            }
            self.assertEqual(s6.choose([a, candidate])["decision"], decision)
        for value, decision in (
            (9.0, "promising_single_seed"),
            (9.000001, "no_candidate"),
        ):
            end = endpoint(value)
            candidate = {
                **b,
                "endpoint": end,
                "rows_verification": {**b["rows_verification"], "endpoint": end},
            }
            self.assertEqual(s6.choose([a, candidate])["decision"], decision)
        for q in ("q50", "q95"):
            end = b["endpoint"]
            per = end["per_database"]
            buckets = per["telstra"]
            cell = buckets["four_plus_tables"]
            missing = {
                **b,
                "endpoint": {
                    **end,
                    "per_database": {
                        **per,
                        "telstra": {
                            **buckets,
                            "four_plus_tables": {
                                k: v for k, v in cell.items() if k != q
                            },
                        },
                    },
                },
            }
            with self.assertRaises(ValueError):
                s6.choose([a, missing])
        with self.assertRaises(ValueError):
            s6.choose([a])

    def test_counts_gradients_and_rows_fail_closed(self):
        a, b = record(), record("B_mean_tail10")
        bad = [
            {**a, "counts": {**a["counts"], "query_presentations": 1621915.0}},
            {
                **a,
                "initial_rows_verification": {
                    **a["initial_rows_verification"],
                    "formal": False,
                },
            },
            {**a, "endpoint": {**a["endpoint"], "pg_identity_digest": "b" * 64}},
        ]
        for key, value in (
            ("max_norm", float("nan")),
            ("count", 0),
            ("clipped_count", 4001),
            ("total_norm", -1),
        ):
            bad = bad + [
                {
                    **a,
                    "summary": {
                        **a["summary"],
                        "gradient_stats": {
                            **a["summary"]["gradient_stats"],
                            key: value,
                        },
                    },
                }
            ]
        for r in bad:
            with self.assertRaises(ValueError):
                s6.verify_pair([r, b], True)


class LossAuthentication(unittest.TestCase):
    def test_missing_evidence_blocks_selection(self):
        a, b = record(), record("B_mean_tail10")
        for key in ("loss_verification", "summary"):
            incomplete = {k: v for k, v in b.items() if k != key}
            with self.subTest(key=key), self.assertRaises(ValueError):
                s6.choose([a, incomplete])
        for key in ("loss_evidence", "gradient_stats"):
            incomplete = {
                **b,
                "summary": {k: v for k, v in b["summary"].items() if k != key},
            }
            with self.subTest(key=key), self.assertRaises(ValueError):
                s6.choose([a, incomplete])

    def test_loss_verification_exact_counts_mode_hashes(self):
        a, b = record(), record("B_mean_tail10")
        changes = (
            ("passed", False),
            ("passed", 1),
            ("passed", "true"),
            ("mode", "mean"),
            ("episodes", 15999),
            ("episodes", 16000.0),
            ("presentations", 1621914),
            ("presentations", True),
            ("sample_plan_digest", "b" * 64),
            ("audit_sha256", "bad"),
            ("snapshots_sha256", None),
            ("snapshot_rows", 8),
            ("snapshot_rows", 16.0),
        )
        for key, value in changes:
            candidate = {
                **b,
                "loss_verification": {**b["loss_verification"], key: value},
            }
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                s6.choose([a, candidate])
        for key in b["loss_verification"]:
            candidate = {
                **b,
                "loss_verification": {
                    k: v for k, v in b["loss_verification"].items() if k != key
                },
            }
            with self.subTest(missing=key), self.assertRaises(ValueError):
                s6.choose([a, candidate])

    def test_summary_loss_counts_and_mode(self):
        a, b = record(), record("B_mean_tail10")
        for key, value in (
            ("mode", "mean"),
            ("audit_rows", 15999),
            ("audit_rows", 16000.0),
            ("snapshot_rows", 8),
        ):
            candidate = {
                **b,
                "summary": {
                    **b["summary"],
                    "loss_evidence": {**b["summary"]["loss_evidence"], key: value},
                },
            }
            with self.subTest(key=key), self.assertRaises(ValueError):
                s6.choose([a, candidate])
        for key in b["summary"]["loss_evidence"]:
            candidate = {
                **b,
                "summary": {
                    **b["summary"],
                    "loss_evidence": {
                        k: v
                        for k, v in b["summary"]["loss_evidence"].items()
                        if k != key
                    },
                },
            }
            with self.subTest(missing=key), self.assertRaises(ValueError):
                s6.choose([a, candidate])

    def test_pair_loss_tracks_may_differ(self):
        a, b = record(), record("B_mean_tail10")
        b = {
            **b,
            "loss_verification": {
                **b["loss_verification"],
                "audit_sha256": "b" * 64,
                "snapshots_sha256": "c" * 64,
                "extra_diagnostic": 0.8,
            },
        }
        self.assertTrue(s6.verify_pair([b, a], True)["passed"])

    def test_s5_weight_and_quantization_fields_forbidden(self):
        a, b = record(), record("B_mean_tail10")
        for key, value in (
            ("complex_query_weight", 1),
            ("complex_query_weight", 2),
            ("precision", "fp16"),
            ("quantization", 8),
            ("data_config", "configs/other.json"),
            ("tail_risk_mode", True),
        ):
            changed = {**b, "config": {**b["config"], key: value}}
            with self.subTest(key=key), self.assertRaises(ValueError):
                s6.verify_pair([a, changed], True)
        for key, value in (
            ("width", 128),
            ("use_pg", False),
            ("pg_output_scale", True),
        ):
            changed = {
                **b,
                "config": {
                    **b["config"],
                    "model": {**b["config"]["model"], key: value},
                },
            }
            with self.subTest(model=key), self.assertRaises(ValueError):
                s6.verify_pair([a, changed], True)

    def test_smoke_purpose_and_snapshot_budget(self):
        a, b = record(formal=False), record("B_mean_tail10", False)
        self.assertTrue(s6.verify_pair([a, b], False)["passed"])
        for key, value in (
            ("purpose", "research"),
            ("steps", 4000),
            ("support_sizes", [10]),
            ("query_size", 64),
            ("validate_every", 100),
            ("validation_runs", 5),
        ):
            changed = {**b, "config": {**b["config"], key: value}}
            with self.subTest(key=key), self.assertRaises(ValueError):
                s6.verify_pair([a, changed], False)
        for key, value in (
            ("snapshot_rows", 16),
            ("episodes", 16000),
            ("presentations", 1621915),
        ):
            changed = {**b, "loss_verification": {**b["loss_verification"], key: value}}
            with self.subTest(key=key), self.assertRaises(ValueError):
                s6.verify_pair([a, changed], False)
        with self.assertRaises(ValueError):
            s6.choose([a, b])

    def test_initialization_identity_and_flow_must_match(self):
        a, b = record(), record("B_mean_tail10")
        for key in (
            "initialization_hash",
            "validation_plan_hash",
            "sample_plan_digest",
            "pg_identity_digest",
            "initial_pg_identity_digest",
        ):
            with self.subTest(key=key), self.assertRaises(ValueError):
                s6.verify_pair([a, {**b, key: "b" * 64}], True)
        with self.assertRaises(ValueError):
            s6.verify_pair([a, a], True)
        with self.assertRaises(ValueError):
            s6.verify_pair([a, b], 1)

    def test_baseline_all_fourteen_metrics_and_tolerance(self):
        a = record()
        old = {"runs": [{"arm": "A_uniform", "endpoint": endpoint(10.00005)}]}
        self.assertTrue(s6.verify_baseline(a, old)["passed"])
        for db in s4.DATABASES:
            for bucket in s4.BUCKETS:
                for q in ("q50", "q95"):
                    historical = endpoint()
                    cell = historical["per_database"][db][bucket]
                    historical = {
                        **historical,
                        "per_database": {
                            **historical["per_database"],
                            db: {
                                **historical["per_database"][db],
                                bucket: {**cell, q: 9.99 if q == "q50" else 10.01},
                            },
                        },
                    }
                    if q == "q95":
                        name = (
                            "macro_all_table_q95"
                            if bucket == "all_tables"
                            else "macro_four_plus_q95"
                        )
                        historical = {**historical, name: 10 + 0.01 / 3}
                    history = {"runs": [{"arm": "A_uniform", "endpoint": historical}]}
                    with self.subTest(db=db, bucket=bucket, q=q):
                        with self.assertRaisesRegex(ValueError, "reproduction_blocked"):
                            s6.verify_baseline(a, history)
        with self.assertRaisesRegex(ValueError, "reproduction_blocked"):
            s6.verify_baseline(record("B_mean_tail10"), old)

    def test_q50_guard_independent_of_q95(self):
        a, b = record(), record("B_mean_tail10")
        for value, passed in ((2.2, True), (2.200001, False)):

            def with_q50(r, median, q95):
                end = endpoint(q95)
                end = {
                    **end,
                    "per_database": {
                        db: {
                            bucket: {**cell, "q50": median}
                            for bucket, cell in buckets.items()
                        }
                        for db, buckets in end["per_database"].items()
                    },
                }
                return {
                    **r,
                    "endpoint": end,
                    "rows_verification": {**r["rows_verification"], "endpoint": end},
                }

            result = s6.choose([with_q50(a, 2, 10), with_q50(b, value, 9)])
            self.assertIs(result["q50_guardrail"], passed)
            self.assertTrue(result["q95_guardrail"])


if __name__ == "__main__":
    unittest.main()
