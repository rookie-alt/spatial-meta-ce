"""S5 evidence gates and independently hand-accounted replay fixtures."""

import copy
import hashlib
import json
import unittest
from unittest.mock import patch

import numpy as np

from scripts import s4_anchor_evidence as s4
from scripts import s5_weight_evidence as s5

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


def record(arm="A_uniform", formal=True):
    steps = 4000 if formal else 2
    config = {
        **s4.FROZEN_CONFIG,
        "complex_query_weight": 1 if arm == "A_uniform" else 2,
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
        "summary": {
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
        "protocol_sha256": s5.PROTOCOL_HASH,
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
        with patch.object(s5, "SOURCE_HASH", s4._digest(p["sources"])):
            for formal in (True, False):
                self.assertTrue(
                    s5.verify_run(record(formal=formal), p, formal)["passed"]
                )

    def test_bad_arm_config_sha_and_evidence(self):
        p = preflight()
        changes = [
            ("arm", "A_anchor1"),
            ("checkpoint_sha256", "bad"),
            ("gpu_uuid", "GPU-other"),
            ("initialization_hash", H),
            ("rows_verification", {"passed": False}),
            ("sample_plan_digest", H),
            ("initial_endpoint_rows_sha256", None),
        ]
        with patch.object(s5, "SOURCE_HASH", s4._digest(p["sources"])):
            for key, value in changes:
                with self.subTest(key=key), self.assertRaises(ValueError):
                    s5.verify_run({**record(), key: value}, p, True)
            for key, value in (
                ("purpose", "production"),
                ("complex_query_weight", 2),
                ("steps", 3999),
                ("seed", True),
            ):
                r = record()
                r = {**r, "config": {**r["config"], key: value}}
                with self.subTest(key=key), self.assertRaises(ValueError):
                    s5.verify_run(r, p, True)
            for key in ("protocol_sha256", "config_sha256", "torch", "python"):
                with self.assertRaises(ValueError):
                    s5.verify_run(record(), {**p, key: "bad"}, True)
        with self.assertRaises(ValueError):
            s5.verify_run(record(), p, True)

    def test_pair_scale_and_initial_endpoint(self):
        a, b = record(), record("B_complex2")
        self.assertTrue(s5.verify_pair([a, b], True)["passed"])
        config = {
            **b["config"],
            "model": {**b["config"]["model"], "pg_output_scale": 0.0},
        }
        with self.assertRaises(ValueError):
            s5.verify_pair([a, {**b, "config": config}], True)
        initial = endpoint(11)
        with self.assertRaises(ValueError):
            s5.verify_pair(
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
        old = {"runs": [{"arm": "A_anchor1", "endpoint": endpoint()}]}
        self.assertTrue(s5.verify_baseline(record(), old)["passed"])
        for history in (
            {"runs": []},
            {"runs": [{"arm": "bad", "endpoint": endpoint()}]},
            {"runs": [{"arm": "A_anchor1", "endpoint": endpoint(10.01)}]},
        ):
            with self.assertRaisesRegex(ValueError, "reproduction_blocked"):
                s5.verify_baseline(record(), history)

    def test_threshold_boundaries_and_mandatory(self):
        a, b = record(), record("B_complex2")
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
            self.assertEqual(s5.choose([a, candidate])["decision"], decision)
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
            self.assertEqual(s5.choose([a, candidate])["decision"], decision)
        for q in ("q50", "q95"):
            missing = copy.deepcopy(b)
            del missing["endpoint"]["per_database"]["telstra"]["four_plus_tables"][q]
            with self.assertRaises(ValueError):
                s5.choose([a, missing])
        with self.assertRaises(ValueError):
            s5.choose([a])

    def test_counts_gradients_and_rows_fail_closed(self):
        a, b = record(), record("B_complex2")
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
            bad += [
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
                s5.verify_pair([r, b], True)


class Flow(unittest.TestCase):
    def fixture(self, homogeneous=False):
        stores = {
            db: {
                "query_hashes": [f"{db}-{i}" for i in range(8)],
                "unique": np.arange(8),
                "tables": np.array([1] * 8 if homogeneous else [1, 4] * 4),
            }
            for db in reversed(s5.TRAIN)
        }
        config = {
            "seed": 42,
            "steps": 3,
            "meta_batch": 4,
            "accumulation": 1,
            "support_sizes": [1, 2],
            "query_size": 2,
        }
        return stores, config

    def test_independent_small_replay(self):
        stores, config = self.fixture()
        rng = np.random.default_rng(43)
        previous = hashlib.sha256(b"").hexdigest()
        expected = {db: [0, 0, 0, 0.0, 0.0, 0.0, 0.0] for db in s5.TRAIN}
        for _ in range(12):
            db = str(rng.choice(s5.TRAIN))
            k = int(rng.choice([1, 2]))
            selected = rng.choice(8, k + 2, replace=False)
            n = k + 2
            high = sum(int(i % 2 == 1) for i in selected)
            low = n - high
            old = expected[db]
            increments = [
                int(0 < high < n),
                low,
                high,
                sum(1 / (4 * n) for i in selected if i % 2 == 0),
                sum(1 / (4 * n) for i in selected if i % 2 == 1),
                sum(1 / (4 * (low + 2 * high)) for i in selected if i % 2 == 0),
                sum(2 / (4 * (low + 2 * high)) for i in selected if i % 2 == 1),
            ]
            expected = {**expected, db: [a + b for a, b in zip(old, increments)]}
            episode = {
                "db": db,
                "support": [stores[db]["query_hashes"][i] for i in selected[:k]],
                "query": [stores[db]["query_hashes"][i] for i in selected[k:]],
            }
            previous = hashlib.sha256(
                json.dumps(
                    {"previous": previous, "episode": episode},
                    sort_keys=True,
                    ensure_ascii=False,
                ).encode()
            ).hexdigest()
        out = s5.audit_flow(stores, config)
        self.assertEqual(out["sample_plan_digest"], previous)
        self.assertEqual(out["episodes"], 12)
        self.assertEqual(
            out["presentations"], sum(v[1] + v[2] for v in expected.values())
        )
        for db, values in expected.items():
            for key, value in zip(
                (
                    "mixed_episodes",
                    "low_presented",
                    "high_presented",
                    "a_low_mass",
                    "a_high_mass",
                    "b_low_mass",
                    "b_high_mass",
                ),
                values,
            ):
                self.assertAlmostEqual(out["per_database"][db][key], value, places=12)
        self.assertAlmostEqual(out["nominal_mass_A"], 3)
        self.assertAlmostEqual(out["nominal_mass_B"], 3)
        self.assertEqual(
            out, s5.audit_flow(dict(reversed(list(stores.items()))), config)
        )
        changed = {
            db: {
                **store,
                "query_hashes": [h + "changed" for h in store["query_hashes"]],
            }
            for db, store in stores.items()
        }
        self.assertNotEqual(
            previous, s5.audit_flow(changed, config)["sample_plan_digest"]
        )

    def test_no_effective_intervention_and_invalid(self):
        stores, config = self.fixture(True)
        with self.assertRaisesRegex(ValueError, "no_effective_intervention"):
            s5.audit_flow(stores, config)
        high = {db: {**s, "tables": np.full(8, 4)} for db, s in stores.items()}
        with self.assertRaisesRegex(ValueError, "no_effective_intervention"):
            s5.audit_flow(high, config)
        for key, value in (
            ("steps", 0),
            ("meta_batch", True),
            ("query_size", float("inf")),
        ):
            with self.assertRaises(ValueError):
                s5.audit_flow(stores, {**config, key: value})
        with self.assertRaises(ValueError):
            s5.audit_flow({**stores, "imdb": stores[s5.TRAIN[0]]}, config)
        broken = {
            **stores,
            s5.TRAIN[0]: {**stores[s5.TRAIN[0]], "unique": np.array([0, 0])},
        }
        with self.assertRaises(ValueError):
            s5.audit_flow(broken, config)


class RoundoffAndRawRows(unittest.TestCase):
    def test_independent_roundoff_tolerance(self):
        a, b = record(), record("B_complex2")
        verified = endpoint(10.000000000000002)
        changed = {
            **a,
            "rows_verification": {**a["rows_verification"], "endpoint": verified},
        }
        self.assertTrue(s5.verify_pair([changed, b], True)["passed"])

    def test_real_smoke_rows_and_duplicate_raw_pool(self):
        rows = {
            db: [
                {
                    "query_hash": f"{db}-{i}",
                    "n_tables": t,
                    "true": 10.0,
                    "log_pred": 2.0,
                    "pg_input_log": 2.0,
                    "pg_reference_log": 2.0,
                }
                for i, t in enumerate((1, 4))
            ]
            for db in s4.DATABASES
        }
        check = s5.verify_rows(rows, 1.0, None, False)
        r = record(formal=False)
        r = {
            **r,
            "endpoint": check["endpoint"],
            "initial_endpoint": check["endpoint"],
            "rows_verification": check,
            "initial_rows_verification": check,
            "pg_identity_digest": check["endpoint"]["pg_identity_digest"],
            "initial_pg_identity_digest": check["endpoint"]["pg_identity_digest"],
        }
        self.assertTrue(
            s5.verify_pair(
                [
                    r,
                    {
                        **r,
                        "arm": "B_complex2",
                        "config": {**r["config"], "complex_query_weight": 2},
                    },
                ],
                False,
            )["passed"]
        )
        stores, config = Flow().fixture()
        duplicates = {
            db: {
                "query_hashes": [h for h in st["query_hashes"] for _ in range(2)],
                "unique": np.arange(0, 16, 2),
                "tables": np.repeat(st["tables"], 2),
            }
            for db, st in stores.items()
        }
        self.assertEqual(
            s5.audit_flow(stores, config)["sample_plan_digest"],
            s5.audit_flow(duplicates, config)["sample_plan_digest"],
        )


class AdditionalGuardrails(unittest.TestCase):
    def test_q50_guardrail_independent_of_q95(self):
        a, b = record(), record("B_complex2")

        def with_q50(r, value, q95):
            end = endpoint(q95)
            per = {
                db: {bucket: {**cell, "q50": value} for bucket, cell in buckets.items()}
                for db, buckets in end["per_database"].items()
            }
            end = {**end, "per_database": per}
            return {
                **r,
                "endpoint": end,
                "rows_verification": {**r["rows_verification"], "endpoint": end},
            }

        a = with_q50(a, 2, 10)
        result = s5.choose([a, with_q50(b, 2.2, 9)])
        self.assertTrue(result["q50_guardrail"])
        result = s5.choose([a, with_q50(b, 2.200001, 9)])
        self.assertFalse(result["q50_guardrail"])
        self.assertTrue(result["q95_guardrail"])
        self.assertEqual(result["decision"], "no_candidate")

    def test_formal_fixture_cannot_approximate_frozen_digest(self):
        stores = {
            db: {
                "query_hashes": [f"fixture-{db}-{i}" for i in range(180)],
                "unique": np.arange(180),
                "tables": np.array([1, 4] * 90),
            }
            for db in s5.TRAIN
        }
        with self.assertRaisesRegex(ValueError, "Frozen formal sample plan mismatch"):
            s5.audit_flow(stores, record()["config"])


if __name__ == "__main__":
    unittest.main()
