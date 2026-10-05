import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.arch_select import checkpoint_pool
from scripts.mapping_ablation_endpoint import (
    bucket_q95,
    paired_comparison,
    plan_digest,
    run_endpoint,
    sidecar_risks,
)
from scripts.run_mapping_ablation import (
    ARMS,
    EXPECTED_ARCH,
    EXPECTED_MODE,
    SEEDS,
    configure_determinism,
    initial_state,
    load_run_config,
    run_all,
    run_matrix,
    verify_pairing,
)
from spatial_meta_ce import data as data_module
from spatial_meta_ce.checkpoints import load_checkpoint
from spatial_meta_ce.cli import read_json as cli_read_json
from spatial_meta_ce.data import StorePool, batch_samples
from spatial_meta_ce.features import write_cache
from spatial_meta_ce.manifests import file_hash
from spatial_meta_ce.model import SetEstimator
from spatial_meta_ce.train import train


def row_groups(n_tables: int, filters: int = 2):
    g = torch.Generator().manual_seed(n_tables)
    return [
        torch.rand(n_tables, 40, generator=g),
        torch.rand(2 * (n_tables - 1), 40, generator=g),
        torch.rand(n_tables, 4, generator=g),
        torch.rand(filters, 43, generator=g),
    ]


def row_mapping(n_tables: int, filters: int = 2):
    return {
        "join_column_tables": list(range(n_tables)),
        "filter_column_tables": list(range(filters)) if filters else [],
        "fanout_join_columns": [v for i in range(1, n_tables) for v in (0, i)],
        "counts": [n_tables, 2 * (n_tables - 1), n_tables, filters],
    }


def shuffled(mapping: dict) -> dict:
    fanout = mapping["fanout_join_columns"]
    even, odd = fanout[0::2], fanout[1::2]
    return {
        **mapping,
        "join_column_tables": sorted(mapping["join_column_tables"], reverse=True),
        "filter_column_tables": sorted(mapping["filter_column_tables"], reverse=True),
        "fanout_join_columns": [
            v for pair in zip(reversed(even), reversed(odd)) for v in pair
        ],
    }


def build_store(root: Path, db: str, n_rows: int = 8, n_tables: int = 3) -> None:
    groups = row_groups(n_tables)
    rows = [
        {
            "id": f"{db}/pretrain/{i}",
            "query_hash": f"{db}-h{i}",
            "exact_hash": f"{db}-x{i}",
            "source_line": i + 1,
            "true": 10.0 + i,
            "pg": 2.0,
        }
        for i in range(n_rows)
    ]
    original = {
        "data_features": [groups for _ in rows],
        "true_cards": [torch.tensor(r["true"]) for r in rows],
        "pg_est_cards": [torch.tensor(2.0) for _ in rows],
        "n_join_cols": [n_tables] * n_rows,
        "n_fanouts": [2 * (n_tables - 1)] * n_rows,
        "n_tables": [n_tables] * n_rows,
        "n_filter_cols": [2] * n_rows,
    }
    cache = root / "cache" / "pretrain" / db
    write_cache(cache, rows, original, {"source": "ablation-fixture"})
    out = root / "sidecars" / db
    out.mkdir(parents=True)
    mapping = row_mapping(n_tables)
    records = [
        {
            **r,
            "source_lines": [r["source_line"]],
            "mapping": mapping,
            "shuffled_mapping": shuffled(mapping),
        }
        for r in rows
    ]
    (out / "mapping.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
    manifest = {
        "schema": 1,
        "database": db,
        "split": "pretrain",
        "shuffle_seed": 42,
        "cache_content_hash": json.loads((cache / "manifest.json").read_text())[
            "content_hash"
        ],
        "jsonl_sha256": file_hash(out / "mapping.jsonl"),
        "input_sha256": {f"cache/{p.name}": file_hash(p) for p in cache.iterdir()},
    }
    (out / "manifest.json").write_text(json.dumps(manifest))


def build_fixture(root: Path, train: list, validation: list, n_tables: int = 3) -> dict:
    for db in train + validation:
        build_store(root, db, n_tables=n_tables)
    return {
        "cache_root": str(root / "cache"),
        "data_root": str(root / "unused"),
        "splits": {"train": train, "validation": validation, "test": ["stats"]},
    }


def tiny_config(root: Path, steps: int = 2) -> dict:
    return {
        "model": {"width": 8, "use_pg": True, "arch": "sum_map"},
        "mapping": {"root": str(root / "sidecars"), "mode": "mapping"},
        "method": "erm",
        "seed": 7,
        "steps": steps,
        "meta_batch": 2,
        "accumulation": 1,
        "outer_lr": 0.001,
        "inner_lr": 0.01,
        "inner_steps": 1,
        "adapt_scope": "all",
        "support_sizes": [1],
        "query_size": 2,
        "loss": "smooth_l1",
        "validate_every": 1,
        "validation_runs": 1,
        "validation_seed": 1729,
        "patience": 20,
        "min_delta": 0.0,
        "purpose": "synthetic-test",
    }


def fake_record(seed: int, arm: str, parameters: int = 100) -> dict:
    return {
        "arm": arm,
        "seed": seed,
        "sample_plan_digest": f"d{seed}",
        "sample_plan_episodes": 8,
        "counts": {"optimizer_steps": 4000},
        "validation_plan_hash": f"v{seed}",
        "optimizer_steps": 4000,
        "initialization_hash": f"i{seed}",
        "parameters": parameters,
        "manifest_hashes": {"a": f"c{seed}", "v": f"c{seed}"},
        "seconds": 1.0,
        "peak_memory_bytes": 1024,
        "mapping_provenance": {
            d: {"manifest_hash": f"m{d}", "jsonl_hash": f"j{d}", "mode": arm}
            for d in ("a", "v")
        },
        "train_databases": {"a"},
        "validation_databases": {"v"},
    }


def fake_matrix() -> list:
    return [fake_record(seed, arm) for seed in SEEDS for arm in ARMS]


class ProtocolTests(unittest.TestCase):
    def test_matrix_is_nine_runs_in_seed_major_order(self):
        matrix = run_matrix("/tmp/ablation")
        self.assertEqual(len(matrix), 9)
        self.assertEqual([m["seed"] for m in matrix], [s for s in SEEDS for _ in ARMS])
        self.assertEqual([m["arm"] for m in matrix], [a for _ in SEEDS for a in ARMS])
        self.assertEqual(matrix[0]["output"], "/tmp/ablation/sum-seed42")

    def test_frozen_configs_declare_arm_identity_and_budget(self):
        for arm in ARMS:
            with self.subTest(arm=arm):
                config = load_run_config(arm, 43)
                self.assertEqual(config["seed"], 43)
                self.assertEqual(config["model"]["arch"], EXPECTED_ARCH[arm])
                self.assertEqual(
                    (config.get("mapping") or {}).get("mode"), EXPECTED_MODE[arm]
                )
                self.assertEqual(
                    (config["steps"], config["patience"], config["method"]),
                    (4000, 41, "erm"),
                )

    def test_tampered_frozen_config_is_rejected(self):
        original = cli_read_json
        for change in (
            lambda c: {**c, "model": {**c["model"], "arch": "attn"}},
            lambda c: {**c, "steps": 100},
            lambda c: {**c, "mapping": {**c["mapping"], "mode": "bad"}},
            lambda c: {**c, "validate_every": 50},
            lambda c: {**c, "validation_seed": 1},
            lambda c: {**c, "mapping": {**c["mapping"], "root": "/elsewhere"}},
        ):
            with (
                self.subTest(change=change),
                patch(
                    "scripts.run_mapping_ablation.read_json",
                    side_effect=lambda path, change=change: change(original(path)),
                ),self.assertRaises(ValueError)
            ):
                load_run_config("true", 42)

    def test_initial_state_is_seed_deterministic_and_arm_specific(self):
        config = load_run_config("true", 42)
        first = initial_state(config)
        self.assertEqual(first, initial_state(config))
        self.assertNotEqual(first, initial_state(load_run_config("true", 43)))
        self.assertEqual(
            first["initialization_hash"],
            initial_state(load_run_config("shuffled", 42))["initialization_hash"],
        )
        self.assertEqual(
            first["parameters"],
            initial_state(load_run_config("shuffled", 42))["parameters"],
        )
        self.assertNotEqual(
            first["initialization_hash"],
            initial_state(load_run_config("sum", 42))["initialization_hash"],
        )

    def test_pairing_accepts_complete_matrix_and_reports_capacity(self):
        verdict = verify_pairing(fake_matrix())
        self.assertEqual(verdict["seeds"], list(SEEDS))
        self.assertEqual(verdict["parameters"][42], {"sum": 100, "sum_map": 100})

    def test_pairing_rejects_every_violation(self):
        cases = {
            "digest": ("sum", lambda r: {**r, "sample_plan_digest": "x"}),
            "counts": ("sum", lambda r: {**r, "counts": {"optimizer_steps": 3999}}),
            "plan": ("sum", lambda r: {**r, "validation_plan_hash": "x"}),
            "steps": ("sum", lambda r: {**r, "optimizer_steps": 3999}),
            "init": ("shuffled", lambda r: {**r, "initialization_hash": "x"}),
            "capacity": ("shuffled", lambda r: {**r, "parameters": 101}),
            "sidecar": (
                "shuffled",
                lambda r: {
                    **r,
                    "mapping_provenance": {
                        d: {**v, "manifest_hash": "x"}
                        for d, v in r["mapping_provenance"].items()
                    },
                },
            ),
            "target": ("sum", lambda r: {**r, "train_databases": {"stats"}}),
            "cache": (
                "sum",
                lambda r: {**r, "manifest_hashes": {"a": "x", "v": f"c{r['seed']}"}},
            ),
        }
        for name, (arm, change) in cases.items():
            with self.subTest(name=name):
                records = [
                    change(r) if (r["seed"], r["arm"]) == (42, arm) else r
                    for r in fake_matrix()
                ]
                with self.assertRaises(ValueError):
                    verify_pairing(records)
        with self.assertRaises(ValueError):
            verify_pairing(fake_matrix()[:-1])
        with self.assertRaises(ValueError):
            verify_pairing([r for r in fake_matrix() if r["arm"] != "sum"])

    def test_gpu_occupancy_is_checked_once_not_once_per_run(self):
        seen = []

        def fake_run_one(spec, data, device, gpu_uuid):
            seen.append(gpu_uuid)
            return fake_record(spec["seed"], spec["arm"])

        with (
            tempfile.TemporaryDirectory() as d,
            patch("scripts.run_mapping_ablation.run_one", side_effect=fake_run_one),
            patch(
                "scripts.run_mapping_ablation.available_gpu", return_value="uuid-x"
            ) as guard,
        ):
            report = run_all(Path(d), {}, "cuda:0", "4")
        self.assertEqual(guard.call_count, 1)
        self.assertEqual(len(seen), 9)
        self.assertEqual(report["pairing"]["seeds"], list(SEEDS))
    def test_determinism_sets_cublas_workspace_before_first_use(self):
        with patch.dict(os.environ):
            os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
            configure_determinism()
            self.assertEqual(os.environ["CUBLAS_WORKSPACE_CONFIG"], ":4096:8")
            torch.use_deterministic_algorithms(False)


class ResidentStoreTests(unittest.TestCase):
    @staticmethod
    def counting_calls():
        calls = []
        original = data_module.FeatureStore

        def counting(*args, **kwargs):
            calls.append(args[0] if args else kwargs.get("path"))
            return original(*args, **kwargs)

        return calls, counting

    def test_pool_reopens_nothing_when_capacity_covers_all_sources(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            build_fixture(root, ["a", "b", "c", "d", "e"], ["v"])
            calls, counting = self.counting_calls()
            with patch.object(data_module, "FeatureStore", counting):
                pool = StorePool(
                    root / "cache", max_open=6, mapping_root=root / "sidecars"
                )
                for _ in range(3):
                    for db in ("a", "b", "c", "d", "e", "v"):
                        pool.get("pretrain", db)
            self.assertEqual(len(calls), 6)

    def test_training_does_not_reopen_mapped_source_stores(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            data = build_fixture(root, ["a", "b", "c", "d", "e"], ["v"])
            calls, counting = self.counting_calls()
            with patch.object(data_module, "FeatureStore", counting):
                train(tiny_config(root, steps=12), data, root / "run")
            self.assertEqual(len(calls), 6)

    def test_unmapped_batch_has_no_mapping_and_sum_predicts(self):
        plain = {
            "groups": row_groups(3),
            "true": 10.0,
            "pg": 2.0,
            "id": "x",
            "hash": "h",
        }
        batch = batch_samples([plain])
        self.assertNotIn("mapping", batch)
        self.assertTrue(torch.isfinite(SetEstimator(arch="sum", width=8)(batch)).all())


class EndpointTests(unittest.TestCase):
    @staticmethod
    def rows(values):
        return [
            {
                "query_hash": f"h{i}",
                "n_tables": n,
                "log_pred": float(p),
                "true": float(t),
            }
            for i, (n, p, t) in enumerate(values)
        ]

    def test_bucket_respects_table_threshold(self):
        rows = self.rows([(2, 1.0, 10.0), (5, 1.0, 10.0), (7, 1.0, 10.0)])
        self.assertEqual(bucket_q95(rows)["queries"], 3)
        self.assertEqual(bucket_q95(rows, 4)["queries"], 2)
        with self.assertRaises(ValueError):
            bucket_q95(rows, 20)

    def test_paired_comparison_detects_per_query_improvement(self):
        left = self.rows([(5, 4.0, 100.0), (6, 2.0, 10.0), (2, 9.0, 1.0)])
        right = self.rows([(5, 0.0, 100.0), (6, 2.0, 10.0), (2, 9.0, 1.0)])
        result = paired_comparison(left, right, "true", "shuffled")
        self.assertEqual(result["queries"], 2)
        self.assertLess(result["left_q95"], result["right_q95"])
        self.assertEqual(result["left_better_fraction"], 0.5)
        self.assertLess(result["mean_log_q_delta"], 0)
        with self.assertRaises(ValueError):
            paired_comparison(left, right[:-1], "true", "shuffled")
        with self.assertRaises(ValueError):
            paired_comparison(
                self.rows([(2, 1.0, 1.0)]), self.rows([(2, 1.0, 1.0)]), "a", "b"
            )

    def test_plan_digest_binds_query_identity_and_shape(self):
        rows = self.rows([(5, 1.0, 1.0), (4, 1.0, 1.0)])
        self.assertEqual(plan_digest(rows), plan_digest(list(reversed(rows))))
        self.assertNotEqual(plan_digest(rows), plan_digest(rows[:-1]))
        changed = [dict(r) for r in rows]
        changed[0]["n_tables"] = 9
        self.assertNotEqual(plan_digest(rows), plan_digest(changed))

    def test_sidecar_risks_are_reported_verbatim(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            for db in ("v1", "v2"):
                out = root / db
                out.mkdir()
                (out / "summary.json").write_text(
                    json.dumps(
                        {
                            "database": db,
                            "rows": 50000,
                            "unique_queries": 49504,
                            "duplicate_rows": 496,
                            "changed_rows": 39468,
                            "changed_fraction": 0.78936,
                            "unchanged_rows": 10532,
                            "unchangeable_rows": 0,
                            "random_unchanged_rows": 10532,
                            "sql_flags": {"multiple_filters": 8406},
                        }
                    )
                )
            report = sidecar_risks(str(root), ["v1", "v2"])
            self.assertEqual(report[0]["sql_flags"], {"multiple_filters": 8406})
            self.assertEqual(report[1]["changed_fraction"], 0.78936)
            self.assertEqual([r["database"] for r in report], ["v1", "v2"])
            with self.assertRaises(ValueError):
                sidecar_risks(str(root), ["stats"])

    def test_run_endpoint_on_trained_fixture_is_finite_and_stable(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            data = build_fixture(root, ["t"], ["v"], n_tables=5)
            train(tiny_config(root, steps=2), data, root / "run")
            model, state = load_checkpoint(root / "run" / "last.pt")
            result = run_endpoint(model, checkpoint_pool(state, data), ["v"], "cpu")
            self.assertEqual(set(result["per_database"]), {"v"})
            self.assertEqual(
                result["per_database"]["v"]["four_plus_tables"]["queries"], 8
            )
            self.assertTrue(np.isfinite(result["macro_four_plus_q95"]))
            self.assertEqual(result["unique_queries"], 8)
            again = run_endpoint(model, checkpoint_pool(state, data), ["v"], "cpu")
            self.assertEqual(result["plan_digest"], again["plan_digest"])
            self.assertEqual(
                result["macro_four_plus_q95"], again["macro_four_plus_q95"]
            )


if __name__ == "__main__":
    unittest.main()
