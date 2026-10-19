import contextlib
import copy
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from spatial_meta_ce.checkpoints import load_checkpoint, save_checkpoint
from spatial_meta_ce.data import FeatureStore, StorePool, batch_samples
from spatial_meta_ce.features import write_cache
from spatial_meta_ce.manifests import file_hash
from spatial_meta_ce.meta import meta_loss
from spatial_meta_ce.model import SetEstimator
from spatial_meta_ce.train import train


def mapping(n: int = 3, empty: bool = False) -> dict:
    return {
        "join_column_tables": list(range(n)),
        "filter_column_tables": [] if empty else [0, n - 1],
        "fanout_join_columns": [v for i in range(1, n) for v in (0, i)],
        "counts": [n, 2 * (n - 1), n, 0 if empty else 2],
    }


def sample(n: int = 3, empty: bool = False, mapped: bool = True) -> dict:
    g = torch.Generator().manual_seed(n)
    result = {
        "groups": [
            torch.rand(n, 40, generator=g),
            torch.rand(2 * (n - 1), 40, generator=g),
            torch.rand(n, 4, generator=g),
            torch.rand(0 if empty else 2, 43, generator=g),
        ],
        "true": 10.0,
        "pg": 2.0,
        "id": f"a/pretrain/{n}",
        "hash": f"h{n}",
    }
    return (
        {**result, "mapping": mapping(n, empty), "mapping_mode": "mapping"}
        if mapped
        else result
    )


def fixture(root: Path) -> tuple[dict, dict]:
    for db in ("a", "v"):
        s = sample()
        rows = [
            {
                "id": f"{db}/pretrain/{i}",
                "query_hash": f"h{i}",
                "exact_hash": f"exact{i}",
                "source_line": i + 1,
                "true": 10.0 + i,
                "pg": 2.0,
            }
            for i in range(16)
        ]
        original = {
            "data_features": [s["groups"] for _ in rows],
            "true_cards": [torch.tensor(r["true"]) for r in rows],
            "pg_est_cards": [torch.tensor(2.0) for _ in rows],
            "n_join_cols": [3] * 16,
            "n_fanouts": [4] * 16,
            "n_tables": [3] * 16,
            "n_filter_cols": [2] * 16,
        }
        cache = root / "cache/pretrain" / db
        write_cache(cache, rows, original, {"source": "fixture"})
        out = root / "sidecars" / db
        out.mkdir(parents=True)
        m = mapping()
        shuffled = {
            **m,
            "join_column_tables": [2, 0, 1],
            "filter_column_tables": [2, 0],
            "fanout_join_columns": [0, 2, 0, 1],
        }
        records = [
            {
                **r,
                "source_lines": [r["source_line"]],
                "mapping": m,
                "shuffled_mapping": shuffled,
            }
            for r in rows
        ]
        (out / "mapping.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in records)
        )
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
    dc = {
        "cache_root": str(root / "cache"),
        "data_root": str(root / "unused"),
        "splits": {"train": ["a"], "validation": ["v"], "test": ["stats"]},
    }
    cfg = {
        "model": {"width": 8, "use_pg": True, "arch": "sum_map"},
        "mapping": {"root": str(root / "sidecars"), "mode": "mapping"},
        "method": "erm",
        "seed": 7,
        "steps": 2,
        "meta_batch": 1,
        "accumulation": 1,
        "outer_lr": 0.001,
        "inner_lr": 0.01,
        "inner_steps": 1,
        "adapt_scope": "all",
        "support_sizes": [2],
        "query_size": 3,
        "loss": "smooth_l1",
        "validate_every": 1,
        "validation_runs": 1,
        "validation_seed": 1729,
        "patience": 20,
        "min_delta": 0.0,
        "purpose": "synthetic-test",
    }
    return cfg, dc


class MappingModelTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.set_num_threads(1)
        torch.manual_seed(7)

    def live(self) -> SetEstimator:
        model = SetEstimator(arch="sum_map", width=8).eval()
        nn.init.normal_(model.head[-1].weight, std=0.1)
        return model

    def test_batch_padding_and_no_numeric_changes(self) -> None:
        a, b = sample(), sample(2, empty=True)
        batch = batch_samples([a, b])
        plain = batch_samples(
            [
                {k: v for k, v in s.items() if k not in ("mapping", "mapping_mode")}
                for s in (a, b)
            ]
        )
        for key in ("groups", "masks"):
            self.assertTrue(
                all(torch.equal(x, y) for x, y in zip(batch[key], plain[key]))
            )
        self.assertEqual(
            batch["mapping"]["join_column_tables"].tolist(), [[0, 1, 2], [0, 1, -1]]
        )
        self.assertEqual(
            batch["mapping"]["fanout_join_columns"].tolist(),
            [[0, 1, 0, 2], [0, 1, -1, -1]],
        )
        self.assertEqual(
            batch["mapping"]["filter_column_tables"].tolist(), [[0, 2], [-1, -1]]
        )

    def test_mixed_mapping_presence_modes_or_bad_indices_rejected(self) -> None:
        for samples in (
            [sample(), sample(mapped=False)],
            [sample(), {**sample(), "mapping_mode": "shuffled_mapping"}],
        ):
            with self.assertRaises(ValueError):
                batch_samples(samples)
        for bad in ([0, 1, 3], [0, 1, -1], [0, 1, 1.5], [0, 1, True], [0]):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                batch_samples(
                    [{**sample(), "mapping": {**mapping(), "join_column_tables": bad}}]
                )

    def test_missing_mapping_and_unmapped_arch_do_not_silently_fallback(self) -> None:
        with self.assertRaises(ValueError):
            SetEstimator(arch="sum_map")(batch_samples([sample(mapped=False)]))
        with self.assertRaises(ValueError):
            SetEstimator(arch="sum")(batch_samples([sample()]))

    def test_initial_prediction_is_pg_and_checkpoint_roundtrip(self) -> None:
        batch = batch_samples([sample(), sample(2, empty=True)])
        model = SetEstimator(arch="sum_map", width=8)
        self.assertTrue(torch.equal(model(batch), batch["pg_log"]))
        model = self.live()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            save_checkpoint(path, model, None, 1, 0.0, 0, {}, {})
            loaded, _ = load_checkpoint(path)
            self.assertEqual(loaded.config, model.config)
            self.assertTrue(torch.equal(model(batch), loaded(batch)))

    def test_each_relation_changes_prediction_and_gradients_are_finite(self) -> None:
        model, s = self.live(), sample()
        first = model(batch_samples([s]))
        for key, value in (
            ("join_column_tables", [2, 0, 1]),
            ("filter_column_tables", [2, 0]),
            ("fanout_join_columns", [0, 2, 0, 1]),
        ):
            changed = {**s, "mapping": {**s["mapping"], key: value}}
            self.assertFalse(
                torch.allclose(first, model(batch_samples([changed])), atol=1e-7), key
            )
        first.sum().backward()
        self.assertTrue(
            all(
                p.grad is not None and torch.isfinite(p.grad).all()
                for p in model.parameters()
            )
        )

    def test_table_join_filter_and_pair_reordering_is_invariant(self) -> None:
        model, s = self.live(), sample()
        expected = model(batch_samples([s]))
        # Reverse tables/join columns/filters; permute fanout as whole directed pairs.
        groups = [
            s["groups"][0].flip(0),
            s["groups"][1].reshape(2, 2, 40).flip(0).reshape(4, 40),
            s["groups"][2].flip(0),
            s["groups"][3].flip(0),
        ]
        remapped = {
            "join_column_tables": [0, 1, 2],
            "filter_column_tables": [0, 2],
            "fanout_join_columns": [2, 0, 2, 1],
            "counts": [3, 4, 3, 2],
        }
        self.assertTrue(
            torch.allclose(
                expected,
                model(batch_samples([{**s, "groups": groups, "mapping": remapped}])),
                atol=1e-6,
            )
        )

    def test_zero_join_fanout_and_filter_support_meta_gradients(self) -> None:
        s = sample(1, empty=True)
        single = {
            **s,
            "groups": [
                torch.empty(0, 40),
                torch.empty(0, 40),
                s["groups"][2],
                torch.empty(0, 43),
            ],
            "mapping": {
                "join_column_tables": [],
                "filter_column_tables": [],
                "fanout_join_columns": [],
                "counts": [0, 0, 1, 0],
            },
        }
        model = self.live()
        batch = batch_samples([single])
        self.assertTrue(torch.isfinite(model(batch)).all())
        meta_loss(model, batch, batch, 0.001, 1).backward()
        self.assertTrue(
            all(
                p.grad is not None and torch.isfinite(p.grad).all()
                for p in model.parameters()
            )
        )

    def test_fanout_fusion_consumes_both_table_encodings(self) -> None:
        model, s = self.live(), sample()
        batch = batch_samples([s])
        captured = []
        handle = model.fanout_fusion.register_forward_pre_hook(
            lambda module, args: captured.append(args[0].detach())
        )
        try:
            model(batch)
        finally:
            handle.remove()
        self.assertEqual(captured[0].shape[-1], 5 * model.config["width"])
        tables = model.encoders[2](batch["groups"][2])
        self.assertTrue(torch.equal(captured[0][:, :, -16:-8], tables[:, [0, 1, 0, 2]]))
        self.assertTrue(torch.equal(captured[0][:, :, -8:], tables[:, [1, 0, 2, 0]]))

    def test_mapping_initialization_is_identical_for_both_modes(self) -> None:
        torch.manual_seed(42)
        first = SetEstimator(arch="sum_map", width=8)
        torch.manual_seed(42)
        second = SetEstimator(arch="sum_map", width=8)
        self.assertTrue(
            all(
                torch.equal(v, second.state_dict()[k])
                for k, v in first.state_dict().items()
            )
        )

    def test_padding_batch_composition_and_inputs_are_immutable(self) -> None:
        model, s = self.live(), sample(2, empty=True)
        batch = batch_samples([s, sample()])
        original = copy.deepcopy(batch)
        expected = model(batch).detach()
        self.assertTrue(
            torch.allclose(expected[:1], model(batch_samples([s])), atol=1e-6)
        )
        self.assertTrue(
            all(torch.equal(a, b) for a, b in zip(batch["groups"], original["groups"]))
        )
        dirty = {
            **batch,
            "groups": [
                g.masked_fill(~m.unsqueeze(-1), 1e8)
                for g, m in zip(batch["groups"], batch["masks"])
            ],
        }
        self.assertTrue(torch.allclose(expected, model(dirty), atol=1e-6))


class MappingIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.set_num_threads(1)

    def rewrite_records(self, path: Path, transform) -> None:
        records = [
            json.loads(line)
            for line in (path / "mapping.jsonl").read_text().splitlines()
        ]
        records = transform(records)
        (path / "mapping.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in records)
        )
        manifest = json.loads((path / "manifest.json").read_text())
        (path / "manifest.json").write_text(
            json.dumps({**manifest, "jsonl_sha256": file_hash(path / "mapping.jsonl")})
        )

    def test_full_chain_and_shuffled_mode_return_indices_only(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            cfg, dc = fixture(Path(d))
            plain = FeatureStore(Path(dc["cache_root"]) / "pretrain/a")
            true = FeatureStore(
                plain.path,
                mapping_path=Path(cfg["mapping"]["root"]) / "a",
                mapping_mode="mapping",
            )
            shuffled = FeatureStore(
                plain.path,
                mapping_path=Path(cfg["mapping"]["root"]) / "a",
                mapping_mode="shuffled_mapping",
            )
            self.assertEqual(
                set(true[0]["mapping"]),
                {
                    "counts",
                    "join_column_tables",
                    "filter_column_tables",
                    "fanout_join_columns",
                },
            )
            self.assertNotEqual(true[0]["mapping"], shuffled[0]["mapping"])
            self.assertTrue(
                all(
                    torch.equal(a, b)
                    for a, b in zip(
                        plain.batch([0])["groups"], true.batch([0])["groups"]
                    )
                )
            )
            self.assertTrue(
                bool(
                    torch.isfinite(SetEstimator(arch="sum_map")(true.batch([0]))).all()
                )
            )
            with self.assertRaises(KeyError):
                true.mapping["missing"]

    def test_tampered_jsonl_cache_or_manifest_rejected(self) -> None:
        for kind in ("jsonl", "cache", "manifest"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as d:
                root = Path(d)
                fixture(root)
                path, cache = root / "sidecars/a", root / "cache/pretrain/a"
                if kind == "jsonl":
                    (path / "mapping.jsonl").write_text("{}\n")
                elif kind == "cache":
                    cards = np.load(cache / "cards.npy")
                    np.save(cache / "cards.npy", cards + 1)
                else:
                    m = json.loads((path / "manifest.json").read_text())
                    (path / "manifest.json").write_text(
                        json.dumps({**m, "database": "stats"})
                    )
                with self.assertRaises(ValueError):
                    FeatureStore(cache, mapping_path=path, mapping_mode="mapping")

    def test_duplicate_missing_hash_source_lines_cards_counts_and_indices_rejected(
        self,
    ) -> None:
        changes = [
            lambda r: r + [r[0]],
            lambda r: r[1:],
            lambda r: [{**r[0], "source_lines": [999]}, *r[1:]],
            lambda r: [{**r[0], "true": 99.0}, *r[1:]],
            lambda r: [
                {**r[0], "mapping": {**r[0]["mapping"], "counts": [3, 4, 4, 2]}},
                *r[1:],
            ],
            lambda r: [
                {
                    **r[0],
                    "mapping": {**r[0]["mapping"], "fanout_join_columns": [0, 3, 0, 1]},
                },
                *r[1:],
            ],
        ]
        for change in changes:
            with self.subTest(change=change), tempfile.TemporaryDirectory() as d:
                root = Path(d)
                fixture(root)
                self.rewrite_records(root / "sidecars/a", change)
                with self.assertRaises(ValueError):
                    FeatureStore(
                        root / "cache/pretrain/a",
                        mapping_path=root / "sidecars/a",
                        mapping_mode="mapping",
                    )

    def test_train_pairing_resume_and_source_validation(self) -> None:
        with (
            tempfile.TemporaryDirectory() as d,
            contextlib.redirect_stdout(io.StringIO()),
        ):
            root = Path(d)
            cfg, dc = fixture(root)
            calls = []
            original = StorePool.get

            def tracked(pool, split, db):
                calls.append((split, db))
                return original(pool, split, db)

            with patch.object(StorePool, "get", tracked):
                train(cfg, dc, root / "true")
                train(
                    {**cfg, "mapping": {**cfg["mapping"], "mode": "shuffled_mapping"}},
                    dc,
                    root / "shuffled",
                )
            _, a = load_checkpoint(root / "true/last.pt")
            _, b = load_checkpoint(root / "shuffled/last.pt")
            self.assertEqual(
                a["extra"]["sample_plan_digest"], b["extra"]["sample_plan_digest"]
            )
            self.assertEqual(a["extra"]["counts"], b["extra"]["counts"])
            self.assertNotIn("sample_plan", a["extra"])
            self.assertEqual(a["extra"]["sample_plan_episodes"], 2)
            plain_cfg = {k: v for k, v in cfg.items() if k != "mapping"}
            train(
                {**plain_cfg, "model": {**cfg["model"], "arch": "sum"}},
                dc,
                root / "plain",
            )
            _, plain = load_checkpoint(root / "plain/last.pt")
            self.assertEqual(
                a["extra"]["sample_plan_digest"], plain["extra"]["sample_plan_digest"]
            )
            self.assertEqual(a["extra"]["counts"], plain["extra"]["counts"])
            self.assertEqual(
                json.loads((root / "true/validation_manifest.json").read_text()),
                json.loads((root / "shuffled/validation_manifest.json").read_text()),
            )
            self.assertEqual(set(calls), {("pretrain", "a"), ("pretrain", "v")})
            self.assertEqual(set(a["metadata"]["mapping_provenance"]), {"a", "v"})
            train({**cfg, "steps": 1}, dc, root / "resumed")
            train(cfg, dc, root / "resumed", resume=root / "resumed/last.pt")
            _, c = load_checkpoint(root / "resumed/last.pt")
            self.assertEqual(
                a["extra"]["sample_plan_digest"], c["extra"]["sample_plan_digest"]
            )
            self.assertTrue(
                all(torch.equal(v, c["model"][k]) for k, v in a["model"].items())
            )
            with self.assertRaises(ValueError):
                train(
                    {**cfg, "mapping": {**cfg["mapping"], "mode": "shuffled_mapping"}},
                    dc,
                    root / "resumed",
                    resume=root / "resumed/last.pt",
                )
            # Same config but provenance changed after checkpoint creation.
            p = root / "sidecars/a/manifest.json"
            p.write_text(json.dumps({**json.loads(p.read_text()), "shuffle_seed": 43}))
            with self.assertRaises(ValueError):
                train(cfg, dc, root / "resumed", resume=root / "resumed/last.pt")

    def test_smoke_configuration_and_pairing_are_fixed(self) -> None:
        from scripts.mapping_smoke import smoke_config, verify_arms

        cfg = smoke_config({"model": {"width": 8}}, "/sidecars", "mapping")
        self.assertEqual(
            (cfg["seed"], cfg["steps"], cfg["meta_batch"], cfg["query_size"]),
            (42, 20, 4, 64),
        )
        self.assertEqual(cfg["purpose"], "smoke-only; not research evidence")
        arm = {
            "optimizer_steps": 20,
            "sample_plan_digest": "same",
            "sample_plan_episodes": 80,
            "counts": {"optimizer_steps": 20},
            "initialization_hash": "same",
            "parameters": 100,
        }
        verify_arms({"sum": arm, "mapping": arm, "shuffled_mapping": arm})
        for key, value in (
            ("sample_plan_digest", "different"),
            ("optimizer_steps", 19),
            ("initialization_hash", "different"),
            ("parameters", 101),
        ):
            with self.subTest(key=key), self.assertRaises(ValueError):
                verify_arms(
                    {
                        "sum": arm,
                        "mapping": arm,
                        "shuffled_mapping": {**arm, key: value},
                    }
                )

    def test_smoke_run_passes_source_only_valid_splits(self) -> None:
        from scripts import mapping_smoke
        from spatial_meta_ce.manifests import validate_splits

        def fake_arm(config: dict, data: dict, output: Path) -> dict:
            validate_splits(data["splits"])
            self.assertEqual(data["splits"]["train"], ["chembl"])
            self.assertEqual(data["splits"]["validation"], ["legalacts", "telstra"])
            return {
                "optimizer_steps": 20,
                "counts": {"optimizer_steps": 20},
                "sample_plan_digest": "same",
                "sample_plan_episodes": 80,
                "initialization_hash": "same",
                "parameters": 100,
                "validation_plan_hash": "same",
            }

        with tempfile.TemporaryDirectory() as d, contextlib.ExitStack() as stack:
            stack.enter_context(patch.dict("os.environ", {"CUDA_VISIBLE_DEVICES": "4"}))
            stack.enter_context(
                patch.object(mapping_smoke, "available_gpu", return_value="uuid")
            )
            for name, value in (
                ("is_available", True),
                ("device_count", 1),
                ("get_device_name", "mocked"),
            ):
                stack.enter_context(patch.object(torch.cuda, name, return_value=value))
            stack.enter_context(patch.object(torch.cuda, "empty_cache"))
            stack.enter_context(patch.object(torch, "use_deterministic_algorithms"))
            stack.enter_context(
                patch.object(mapping_smoke, "run_arm", side_effect=fake_arm)
            )
            result = mapping_smoke.run(
                "/cache", "/sidecars", str(Path(d) / "smoke"), "4"
            )
            self.assertEqual(result["status"], "passed")
            self.assertEqual(result["target_databases_opened"], [])

    def test_legacy_unmapped_checkpoint_resumes_with_reconstructed_digest(self) -> None:
        with (
            tempfile.TemporaryDirectory() as d,
            contextlib.redirect_stdout(io.StringIO()),
        ):
            root = Path(d)
            cfg, dc = fixture(root)
            cfg = {
                **{k: v for k, v in cfg.items() if k != "mapping"},
                "model": {"width": 8, "use_pg": True, "arch": "sum"},
            }
            train(cfg, dc, root / "full")
            train({**cfg, "steps": 1}, dc, root / "legacy")
            _, state = load_checkpoint(root / "legacy/last.pt")
            legacy = {
                **state,
                "metadata": {
                    k: v
                    for k, v in state["metadata"].items()
                    if k != "mapping_provenance"
                },
                "extra": {
                    k: v
                    for k, v in state["extra"].items()
                    if not k.startswith("sample_plan")
                },
            }
            torch.save(legacy, root / "legacy/last.pt")
            train(cfg, dc, root / "legacy", resume=root / "legacy/last.pt")
            _, expected = load_checkpoint(root / "full/last.pt")
            _, actual = load_checkpoint(root / "legacy/last.pt")
            self.assertEqual(
                expected["extra"]["sample_plan_digest"],
                actual["extra"]["sample_plan_digest"],
            )
            self.assertEqual(
                expected["extra"]["sample_plan_episodes"],
                actual["extra"]["sample_plan_episodes"],
            )
            self.assertTrue(
                all(
                    torch.equal(v, actual["model"][k])
                    for k, v in expected["model"].items()
                )
            )

    def test_arch_selection_uses_checkpoint_mapping_mode(self) -> None:
        from scripts.arch_select import checkpoint_pool, evaluate_checkpoint
        from spatial_meta_ce.train import validation_plan

        with (
            tempfile.TemporaryDirectory() as d,
            contextlib.redirect_stdout(io.StringIO()),
        ):
            root = Path(d)
            cfg, dc = fixture(root)
            train(cfg, dc, root / "trained")
            model, state = load_checkpoint(root / "trained/last.pt")
            pool = checkpoint_pool(state, dc)
            episodes = validation_plan(pool, ["v"], cfg)
            result = evaluate_checkpoint(model, cfg, pool, episodes, "cpu")
            self.assertTrue(np.isfinite(result["macro_log_q95"]))
            self.assertEqual(pool.get("pretrain", "v").mapping.mode, "mapping")
            from scripts.arch_select import main

            config_path, report_path = (
                root / "selection.json",
                root / "selection-result.json",
            )
            config_path.write_text(json.dumps({**cfg, **dc}))
            with patch(
                "sys.argv",
                [
                    "arch_select",
                    "--config",
                    str(config_path),
                    "--checkpoints",
                    f"mapped={root}/trained/last.pt",
                    "--output",
                    str(report_path),
                ],
            ):
                main()
            self.assertTrue(
                np.isfinite(
                    json.loads(report_path.read_text())["results"]["mapped"][
                        "macro_log_q95"
                    ]
                )
            )
            for key in ("manifest_hash", "jsonl_hash", "mode"):
                bad = {
                    **state,
                    "metadata": {
                        **state["metadata"],
                        "mapping_provenance": {
                            **state["metadata"]["mapping_provenance"],
                            "v": {
                                **state["metadata"]["mapping_provenance"]["v"],
                                key: "changed",
                            },
                        },
                    },
                }
                with self.subTest(key=key), self.assertRaises(ValueError):
                    checkpoint_pool(bad, dc)
            with self.assertRaises(ValueError):
                checkpoint_pool(
                    {
                        **state,
                        "metadata": {**state["metadata"], "mapping_provenance": {}},
                    },
                    dc,
                )
            ep = episodes[0]
            wrong = {**ep, "query_hashes": ["wrong"] * len(ep["query"])}
            with self.assertRaises(ValueError):
                evaluate_checkpoint(model, cfg, pool, [wrong], "cpu")
            for metadata, data in (
                ({**state["metadata"], "adapted_database": "stats"}, dc),
                (
                    state["metadata"],
                    {**dc, "splits": {**dc["splits"], "validation": ["a"]}},
                ),
            ):
                with self.assertRaises(ValueError):
                    checkpoint_pool({**state, "metadata": metadata}, data)

    def test_mapping_config_is_required_and_target_reads_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            cfg, dc = fixture(root)
            for bad in (
                {k: v for k, v in cfg.items() if k != "mapping"},
                {**cfg, "model": {"arch": "sum"}},
                {**cfg, "mapping": {**cfg["mapping"], "mode": "bad"}},
            ):
                with self.assertRaises(ValueError):
                    train(bad, dc, root / "bad")
            pool = StorePool(
                dc["cache_root"],
                mapping_root=cfg["mapping"]["root"],
                mapping_mode="mapping",
            )
            with self.assertRaises(ValueError):
                pool.get("test", "stats")
            with patch("spatial_meta_ce.data.FeatureStore") as opened:
                for db in ("../stats", "/tmp/stats"):
                    with self.subTest(db=db), self.assertRaises(ValueError):
                        pool.get("pretrain", db)
                opened.assert_not_called()


if __name__ == "__main__":
    unittest.main()
