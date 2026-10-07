"""Synthetic evidence only: independent smoke authentication, never research."""

import ast
import copy
import hashlib
import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.s8_independent_verify import _quantile, verify_smoke

TRAIN = (
    "airline",
    "baseball",
    "basketball",
    "carcinogenesis",
    "ccs",
    "consumer",
    "credit",
    "financial",
    "fnhk",
    "hepatitis",
    "movielens",
    "sakila",
    "sap",
    "seznam",
    "ssb",
    "talkingdata",
    "tournament",
    "tpc_h",
    "tubepricing",
)
HOLDOUT = ("accidents", "grants", "employee", "hockey")
VALIDATION = ("chembl", "legalacts", "telstra")
ARMS = ("A_h_adam", "B_h_ls", "C_x_adam", "D_x_ls", "F0")


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def select(db, hashes, prefix, n):
    return sorted(
        range(len(hashes)),
        key=lambda i: (
            hashlib.sha256(f"{prefix}|{db}|{hashes[i]}".encode()).hexdigest(),
            hashes[i],
        ),
    )[:n]


def percentile(values, fraction):
    ordered = sorted(values)
    rank = fraction * (len(ordered) - 1)
    lo, hi = math.floor(rank), math.ceil(rank)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (rank - lo)


def metric_rows(endpoints):
    rows = []
    for db in (*TRAIN, *VALIDATION, *HOLDOUT):
        cohort = "P" if db in TRAIN else "V" if db in VALIDATION else "H"
        group = [r for r in endpoints if r["database"] == db]
        for bucket in ("all", "four_plus"):
            cell = [r for r in group if bucket == "all" or r["n_tables"] >= 4]
            row = {
                "database": db,
                "cohort": cohort,
                "bucket": bucket,
                "n": len(cell),
                "status": "descriptive" if len(cell) >= 200 else "below_min_cell_n",
            }
            if len(cell) >= 200:
                row["arms"] = {}
                for arm in ARMS:
                    errors = [
                        r["predictions"][arm] - math.log(max(r["true"], 1))
                        for r in cell
                    ]
                    q = [
                        math.exp(
                            abs(
                                max(r["predictions"][arm], 0)
                                - math.log(max(r["true"], 1))
                            )
                        )
                        for r in cell
                    ]
                    row["arms"][arm] = {
                        "q50": percentile(q, 0.5),
                        "q95": percentile(q, 0.95),
                        "raw_log_mae": sum(map(abs, errors)) / len(errors),
                        "raw_log_mse": sum(e * e for e in errors) / len(errors),
                    }
            rows.append(row)
    return {"rows": rows}


def fixture():
    caches, panels, endpoints = {}, {}, []
    for db in (*TRAIN, *VALIDATION, *HOLDOUT):
        n = 1224 if db in TRAIN else 320 if db in VALIDATION else 64
        hashes = [f"{db}-q{i:04d}" for i in range(n)]
        caches[db] = {
            "indices": np.arange(n, dtype=np.int64) * 3 + 7,
            "query_hash": hashes,
            "true": np.arange(n, dtype=float) % 37,
            "pg": np.arange(n, dtype=float) % 23,
            "n_tables": np.full(n, 2 if db == "accidents" else 4, dtype=np.int64),
        }
        prefix, count, cohort = (
            ("s8-train-panel-v1", 1024, "P")
            if db in TRAIN
            else (
                ("s8-bridge-v1", 256, "V")
                if db in VALIDATION
                else ("s8-smoke-h-v1", 32, "H")
            )
        )
        positions = select(db, hashes, prefix, count)
        if db in TRAIN:
            panels[db] = positions
        for i in positions:
            raw = math.log(max(caches[db]["true"][i], 1)) + (i % 11 - 5) / 7
            endpoints.append(
                {
                    "database": db,
                    "cohort": cohort,
                    "query_hash": hashes[i],
                    "true": float(caches[db]["true"][i]),
                    "pg": float(caches[db]["pg"][i]),
                    "n_tables": int(caches[db]["n_tables"][i]),
                    "predictions": {arm: raw + j / 10 for j, arm in enumerate(ARMS)},
                }
            )
    bindings = {
        "schema": 1,
        "mode": "smoke",
        "purpose": "s8-engineering-smoke-not-research",
        "seed": 42,
        "train_steps": 2,
        "meta_batch": 4,
        "query_size": 64,
        "support_sizes": [5, 10, 20, 50, 100],
        "train_databases": list(TRAIN),
        "holdout_databases": list(HOLDOUT),
        "validation_databases": list(VALIDATION),
        "resourcefixedfields": {
            "batch_size": 128,
            "threads": 4,
            "probe_fit_per_database": 64,
            "probe_adam_steps": 2,
        },
        "panel_identity_digest": digest(
            {db: [caches[db]["query_hash"][i] for i in panels[db]] for db in TRAIN}
        ),
    }
    episodes, rng = [], np.random.default_rng(43)
    for step in range(1, 3):
        for episode in range(4):
            db = str(rng.choice(TRAIN))
            k = int(rng.choice(bindings["support_sizes"]))
            cache = caches[db]
            raw = rng.choice(
                np.sort(cache["indices"][panels[db]]), size=k + 64, replace=False
            )
            lookup = {int(r): i for i, r in enumerate(cache["indices"])}
            positions = [lookup[int(r)] for r in raw]
            episodes.append(
                {
                    "step": step,
                    "episode": len(episodes) + 1,
                    "database": db,
                    "support_positions": positions[:k],
                    "query_positions": positions[k:],
                    "support_hashes": [cache["query_hash"][i] for i in positions[:k]],
                    "query_hashes": [cache["query_hash"][i] for i in positions[k:]],
                }
            )
    return bindings, panels, caches, episodes, endpoints, metric_rows(endpoints)


class IndependentSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.evidence = fixture()

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        (
            self.bindings,
            self.panels,
            self.caches,
            self.episodes,
            self.endpoints,
            self.metrics,
        ) = copy.deepcopy(self.evidence)
        self.nomination = b'{"purpose":"smoke-not-research","nominee":null}\n'
        self.events = [
            {
                "event": "nomination_sealed",
                "file_sha256": hashlib.sha256(self.nomination).hexdigest(),
            },
            {
                "event": "H_semantic_load",
                "database": "accidents",
                "hash": "synthetic-source-hash",
            },
        ]

    def write(self):
        for name, rows in (
            ("episodes", self.episodes),
            ("endpoints", self.endpoints),
            ("events", self.events),
        ):
            (self.root / f"{name}.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in rows)
            )
        (self.root / "metrics.json").write_text(json.dumps(self.metrics))
        (self.root / "nomination.json").write_bytes(self.nomination)

    def verify(self):
        self.write()
        return verify_smoke(self.root, self.bindings, self.panels, self.caches)

    def reject(self, pattern=None):
        with self.assertRaisesRegex(ValueError, pattern or "."):
            self.verify()

    def test_valid_complete_smoke_and_limits(self):
        result = self.verify()
        self.assertTrue(result["passed"])
        self.assertEqual(result["purpose"], "smoke-not-research")
        self.assertEqual(result["query_count"], 19456 + 3 * 256 + 4 * 32)
        self.assertEqual(result["verified_counts"]["episodes"], 8)
        self.assertEqual(result["verified_counts"]["metric_cells"], 52)
        self.assertFalse(result["full_linear_optimality_verified"])
        self.assertFalse(result["actual_H_access_order_verified"])
        self.assertEqual(result["units"]["raw_log_mae"], "natural_log_cardinality")

    def test_smoke_binding_and_split_mutations(self):
        for key, value in (
            ("schema", 2),
            ("mode", "formal"),
            ("purpose", "research"),
            ("seed", 43),
            ("train_steps", 4000),
            ("meta_batch", 5),
            ("query_size", 32),
            ("support_sizes", [5]),
            ("train_databases", list(TRAIN[::-1])),
            ("holdout_databases", ["imdb"]),
            ("validation_databases", ["stats"]),
            ("panel_identity_digest", "bad"),
        ):
            with self.subTest(key=key):
                self.bindings = copy.deepcopy(self.evidence[0])
                self.bindings[key] = value
                self.reject("binding|split|digest")

    def test_runner_H_nomination_hash_and_metrics_purpose(self):
        self.events[1] = {
            "event": "H_semantic_load",
            "database": "accidents",
            "nomination_sha256": hashlib.sha256(self.nomination).hexdigest(),
        }
        self.metrics["purpose"] = "smoke-not-research"
        self.assertTrue(self.verify()["passed"])
        self.events[1]["nomination_sha256"] = "0" * 64
        self.reject("event|nomination")

    def test_metrics_top_level_purpose_never_research(self):
        self.metrics["purpose"] = "s8-research"
        self.reject("metric|purpose")

    def test_resource_bindings_and_H_event_metadata(self):
        for key, bad in (
            ("threads", 1),
            ("batch_size", 64),
            ("probe_fit_per_database", 1024),
            ("probe_adam_steps", 1000),
        ):
            with self.subTest(key=key):
                self.bindings = copy.deepcopy(self.evidence[0])
                self.bindings["resourcefixedfields"][key] = bad
                self.reject("resource|binding")
        self.bindings = copy.deepcopy(self.evidence[0])
        for key in ("database", "hash"):
            with self.subTest(key=key):
                self.events = [
                    {
                        "event": "nomination_sealed",
                        "file_sha256": hashlib.sha256(self.nomination).hexdigest(),
                    },
                    {
                        "event": "H_semantic_load",
                        "database": "accidents",
                        "hash": "synthetic-source-hash",
                    },
                ]
                del self.events[1][key]
                self.reject("event")

    def test_large_integer_source_value_is_not_rounded_into_match(self):
        row = self.endpoints[-1]
        db, h = row["database"], row["query_hash"]
        i = self.caches[db]["query_hash"].index(h)
        self.caches[db]["pg"] = self.caches[db]["pg"].astype(np.int64)
        self.caches[db]["pg"][i] = 2**53 + 1
        row["pg"] = 2**53
        self.reject("endpoint")

    def test_minimum_cell_n_200_boundary(self):
        db = TRAIN[0]
        selected = set(self.panels[db][:200])
        self.caches[db]["n_tables"][:] = 1
        self.caches[db]["n_tables"][list(selected)] = 4
        for row in self.endpoints:
            if row["database"] == db:
                i = self.caches[db]["query_hash"].index(row["query_hash"])
                row["n_tables"] = int(self.caches[db]["n_tables"][i])
        self.metrics = metric_rows(self.endpoints)
        cell = next(
            r
            for r in self.metrics["rows"]
            if r["database"] == db and r["bucket"] == "four_plus"
        )
        self.assertEqual(cell["n"], 200)
        self.assertEqual(cell["status"], "descriptive")
        self.assertTrue(self.verify()["passed"])
        i = self.panels[db][199]
        self.caches[db]["n_tables"][i] = 1
        next(
            r
            for r in self.endpoints
            if r["database"] == db
            and r["query_hash"] == self.caches[db]["query_hash"][i]
        )["n_tables"] = 1
        self.metrics = metric_rows(self.endpoints)
        cell = next(
            r
            for r in self.metrics["rows"]
            if r["database"] == db and r["bucket"] == "four_plus"
        )
        self.assertEqual(cell["n"], 199)
        self.assertNotIn("arms", cell)
        self.assertTrue(self.verify()["passed"])

    def test_panel_sha_order_is_not_trusted(self):
        self.panels[TRAIN[0]][:2] = self.panels[TRAIN[0]][:2][::-1]
        self.reject("panel")

    def test_pool_at_least_1224_is_not_mocked(self):
        db = TRAIN[0]
        for key in self.caches[db]:
            self.caches[db][key] = self.caches[db][key][:1223]
        self.reject("1224|pool")

    def test_source_cache_invalid_or_extra(self):
        for change in (
            "extra",
            "duplicate_hash",
            "duplicate_raw",
            "shape",
            "negative_true",
            "nan_pg",
            "tables",
            "raw_type",
        ):
            with self.subTest(change=change):
                self.caches = copy.deepcopy(self.evidence[2])
                c = self.caches[TRAIN[0]]
                if change == "extra":
                    self.caches["imdb"] = c
                elif change == "duplicate_hash":
                    c["query_hash"][1] = c["query_hash"][0]
                elif change == "duplicate_raw":
                    c["indices"][1] = c["indices"][0]
                elif change == "shape":
                    c["true"] = c["true"][:, None]
                elif change == "negative_true":
                    c["true"][0] = -1
                elif change == "nan_pg":
                    c["pg"][0] = float("nan")
                elif change == "tables":
                    c["n_tables"][0] = 0
                else:
                    c["indices"] = c["indices"].astype(float)
                self.reject("cache|source|indices")

    def test_episode_exact_replay_and_no_leakage(self):
        for change in (
            "missing",
            "duplicate",
            "reorder",
            "Q",
            "H",
            "V",
            "target",
            "hash",
            "raw_instead_position",
            "step",
        ):
            with self.subTest(change=change):
                self.episodes = copy.deepcopy(self.evidence[3])
                r = self.episodes[0]
                if change == "missing":
                    self.episodes.pop()
                elif change == "duplicate":
                    self.episodes.append(r)
                elif change == "reorder":
                    self.episodes.reverse()
                elif change == "Q":
                    r["support_positions"][0] = next(
                        i for i in range(1224) if i not in self.panels[r["database"]]
                    )
                elif change in ("H", "V", "target"):
                    r["database"] = {
                        "H": HOLDOUT[0],
                        "V": VALIDATION[0],
                        "target": "imdb",
                    }[change]
                elif change == "hash":
                    r["query_hashes"][0] = "changed"
                elif change == "raw_instead_position":
                    r["support_positions"][0] = int(
                        self.caches[r["database"]]["indices"][r["support_positions"][0]]
                    )
                else:
                    r["step"] = 0
                self.reject("episode")

    def test_raw_sampling_order_differs_from_panel_sha_order(self):
        rng = np.random.default_rng(43)
        for i in range(8):
            db = str(rng.choice(TRAIN))
            k = int(rng.choice(self.bindings["support_sizes"]))
            cache = self.caches[db]
            pool = cache["indices"][self.panels[db]]
            self.assertFalse(np.array_equal(pool, np.sort(pool)))
            legacy_raw = rng.choice(pool, size=k + 64, replace=False)
            lookup = {int(r): pos for pos, r in enumerate(cache["indices"])}
            positions = [lookup[int(r)] for r in legacy_raw]
            row = self.episodes[i]
            row["support_positions"], row["query_positions"] = (
                positions[:k],
                positions[k:],
            )
            row["support_hashes"] = [cache["query_hash"][pos] for pos in positions[:k]]
            row["query_hashes"] = [cache["query_hash"][pos] for pos in positions[k:]]
        self.reject("episode")

    def test_episode_counters_are_global_one_based(self):
        self.assertEqual([r["step"] for r in self.episodes], [1] * 4 + [2] * 4)
        self.assertEqual([r["episode"] for r in self.episodes], list(range(1, 9)))
        self.episodes[4]["episode"] = 1
        self.reject("episode")

    def test_endpoint_denominator_and_values(self):
        for change in (
            "missingP",
            "missingH",
            "duplicate",
            "extra",
            "cohort",
            "true",
            "pg",
            "n_tables",
            "missing_arm",
        ):
            with self.subTest(change=change):
                self.endpoints = copy.deepcopy(self.evidence[4])
                r = self.endpoints[0]
                if change == "missingP":
                    self.endpoints.pop(0)
                elif change == "missingH":
                    self.endpoints.pop()
                elif change == "duplicate":
                    self.endpoints.append(r)
                elif change == "extra":
                    r["query_hash"] = "not-selected"
                elif change == "cohort":
                    r["cohort"] = "Q"
                elif change in ("true", "pg", "n_tables"):
                    r[change] += 1
                else:
                    r["predictions"].pop("F0")
                self.reject("endpoint|prediction")

    def test_nonfinite_and_overflow_even_below_200(self):
        for bad in (float("nan"), float("inf"), 710.0, -1e200):
            with self.subTest(bad=bad):
                self.endpoints = copy.deepcopy(self.evidence[4])
                self.endpoints[-1]["predictions"]["F0"] = bad
                self.reject("finite|overflow|numeric")

    def test_empty_and_small_cells_have_no_metrics(self):
        r = next(
            r
            for r in self.metrics["rows"]
            if r["database"] == "accidents" and r["bucket"] == "four_plus"
        )
        self.assertEqual(r["n"], 0)
        r["arms"] = {
            arm: {"q50": 1, "q95": 1, "raw_log_mae": 0, "raw_log_mse": 0}
            for arm in ARMS
        }
        self.reject("metric")

    def test_metrics_complete_unique_and_independently_recomputed(self):
        for change in (
            "missing",
            "duplicate",
            "extra",
            "count",
            "status",
            "percentile",
            "raw",
            "nan",
            "small_detail",
        ):
            with self.subTest(change=change):
                self.metrics = copy.deepcopy(self.evidence[5])
                r = self.metrics["rows"][0]
                if change == "missing":
                    self.metrics["rows"].pop()
                elif change == "duplicate":
                    self.metrics["rows"].append(r)
                elif change == "extra":
                    r["bucket"] = "pooled"
                elif change == "count":
                    r["n"] -= 1
                elif change == "status":
                    r["status"] = "research"
                elif change == "percentile":
                    r["arms"]["A_h_adam"]["q95"] += 0.1
                elif change == "raw":
                    r["arms"]["F0"]["raw_log_mse"] += 0.1
                elif change == "nan":
                    r["arms"]["F0"]["q50"] = float("nan")
                else:
                    self.metrics["rows"][-1]["q95"] = 1
                self.reject("metric|finite")

    def test_nomination_and_events_bound_and_ordered(self):
        for change in (
            "reverse",
            "missingH",
            "missingSeal",
            "duplicateSeal",
            "sha",
            "mutated_file",
        ):
            with self.subTest(change=change):
                self.events = [
                    {
                        "event": "nomination_sealed",
                        "file_sha256": hashlib.sha256(self.nomination).hexdigest(),
                    },
                    {
                        "event": "H_semantic_load",
                        "database": "accidents",
                        "hash": "synthetic-source-hash",
                    },
                ]
                if change == "reverse":
                    self.events.reverse()
                elif change == "missingH":
                    self.events.pop()
                elif change == "missingSeal":
                    self.events.pop(0)
                elif change == "duplicateSeal":
                    self.events.insert(0, self.events[0])
                elif change == "sha":
                    self.events[0]["file_sha256"] = "bad"
                else:
                    self.nomination += b" "
                self.reject("nomination|event")

    def test_required_artifacts_and_malformed_json_raise(self):
        for name in (
            "episodes.jsonl",
            "endpoints.jsonl",
            "metrics.json",
            "nomination.json",
            "events.jsonl",
        ):
            with self.subTest(name=name):
                self.write()
                (self.root / name).unlink()
                with self.assertRaises((ValueError, FileNotFoundError)):
                    verify_smoke(self.root, self.bindings, self.panels, self.caches)
        self.write()
        (self.root / "events.jsonl").write_text(
            '{"event":"nomination_sealed","event":"H_semantic_load"}\n'
        )
        with self.assertRaises(ValueError):
            verify_smoke(self.root, self.bindings, self.panels, self.caches)

    def test_sorted_linear_interpolation_known_values(self):
        self.assertEqual(_quantile([9.0, 1.0, 5.0, 3.0], 0.5), 4.0)
        self.assertAlmostEqual(_quantile([9.0, 1.0, 5.0, 3.0], 0.95), 8.4)
        self.assertEqual(_quantile([7.0], 0.95), 7.0)

    def test_no_production_imports_or_percentile_reuse(self):
        source = (
            Path(__file__).resolve().parents[1] / "scripts" / "s8_independent_verify.py"
        )
        tree = ast.parse(source.read_text())
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imports.append(node.module)
        self.assertEqual(set(imports), {"hashlib", "json", "math", "pathlib", "numpy"})
        self.assertNotIn("np.percentile", source.read_text())
        self.assertNotIn("np.quantile", source.read_text())

    def test_SHA_selection_does_not_read_labels(self):
        # Labels/PG/tables change while the fixed identity-only panel stays put.
        for cache in self.caches.values():
            cache["true"][:] = 7
            cache["pg"][:] = 3
            cache["n_tables"][:] = 1
        for row in self.endpoints:
            row["true"], row["pg"], row["n_tables"] = 7, 3, 1
        self.metrics = metric_rows(self.endpoints)
        self.assertTrue(self.verify()["passed"])

    def test_verifier_does_not_modify_artifacts(self):
        self.write()
        before = {p.name: p.read_bytes() for p in self.root.iterdir()}
        verify_smoke(self.root, self.bindings, self.panels, self.caches)
        self.assertEqual(before, {p.name: p.read_bytes() for p in self.root.iterdir()})


if __name__ == "__main__":
    unittest.main()
