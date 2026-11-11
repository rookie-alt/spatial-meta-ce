"""Doc 122 contracts; all data are synthetic, not research evidence."""

import copy
import json
import tempfile
import unittest
from collections import OrderedDict
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from spatial_meta_ce.checkpoints import load_checkpoint, save_checkpoint
from spatial_meta_ce.data import TRANSFORM_VERSION, batch_samples
from spatial_meta_ce.manifests import digest
from spatial_meta_ce.meta import _adaptation_names, adapt, forward, meta_loss
from spatial_meta_ce.model import SetEstimator, create_model
from spatial_meta_ce.price_capacity import (
    TRAIN_DATABASES,
    UnsupportedCapacity,
    build_capacity,
    validate_capacity,
)


def sample(n=2, true=50.0):
    return {
        "groups": [
            torch.rand(n, 40),
            torch.rand(n, 40) * 10,
            torch.rand(n, 4),
            torch.rand(n, 43),
        ],
        "true": true,
        "pg": 2.0,
        "id": f"synthetic/pretrain/{n}",
        "hash": digest([n, true]),
        "database": "synthetic",
    }


def capacity(max_counts=(4, 4, 4, 4)):
    """Generate a binding via the real reader, using synthetic manifests only."""
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        for db in TRAIN_DATABASES:
            payload = {
                "schema": 1,
                "transform": TRANSFORM_VERSION,
                "inspection": {"max_counts": list(max_counts)},
                "rows": [{"id": f"{db}/pretrain/1", "query_hash": digest(db)}],
                "provenance": {"purpose": "smoke-not-research"},
            }
            entry = {**payload, "content_hash": digest(payload)}
            path = root / "pretrain" / db
            path.mkdir(parents=True)
            (path / "manifest.json").write_text(json.dumps(entry))
        config = {
            "cache_root": str(root),
            "splits": {
                "train": list(TRAIN_DATABASES),
                "validation": ["chembl", "legalacts", "telstra"],
                "test": ["imdb", "stats", "genome", "ergastf1"],
            },
        }
        return build_capacity(config)


class PriceModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.binding = capacity()

    def setUp(self):
        torch.manual_seed(42)

    def model(self, arch="price_lite", lora=False):
        return create_model({"arch": arch, "capacity": self.binding, "lora": lora})

    def live_head(self, model):
        # At a zero output head, trunk/LoRA gradients are mathematically zero.
        nn.init.normal_(model.head[-1].weight, std=0.1)
        return model

    def test_parameter_counts_and_exact_pg_zero_step(self):
        batch = batch_samples([sample(1), sample(4)])
        for arch in ("price_lite", "price_tiny"):
            with self.subTest(arch=arch):
                model = self.model(arch)
                self.assertTrue(torch.equal(model(batch), batch["pg_log"]))
                params = adapt(model, batch, steps=0, scope="all")
                self.assertTrue(
                    torch.equal(forward(model, params, batch), batch["pg_log"])
                )
                self.assertFalse(
                    any("final_linear" in n for n, _ in model.named_parameters())
                )
                self.assertEqual(model.head[-1].in_features, 64)
                self.assertEqual(model.head[0].out_features, 128)
        self.assertLess(
            sum(p.numel() for p in self.model("price_tiny").parameters()),
            sum(p.numel() for p in self.model().parameters()),
        )

    def test_only_anchor_reads_pg_and_sum_reference_still_reads_head_pg(self):
        batch = batch_samples([sample(2)])
        changed = {**batch, "pg_log": batch["pg_log"] + 10}
        for model in (
            self.model(),
            self.model("price_tiny"),
            create_model({"arch": "sum_anchor_only"}),
        ):
            features = model._head_input(batch)
            self.assertTrue(torch.equal(features, model._head_input(changed)))
            self.live_head(model)
            self.assertTrue(
                torch.allclose(model(changed) - model(batch), torch.tensor([10.0]))
            )
        old = SetEstimator(arch="sum")
        self.assertFalse(torch.equal(old._head_input(batch), old._head_input(changed)))
        self.assertEqual(sum(p.numel() for p in old.parameters()), 132545)
        anchor = create_model({"arch": "sum_anchor_only"})
        self.assertEqual(sum(p.numel() for p in anchor.parameters()), 132417)
        self.assertEqual(old.head[0].in_features - anchor.head[0].in_features, 1)

    def test_price_padding_masks_raw_fanout_and_log_counts(self):
        model = self.live_head(self.model())
        single = sample(1)
        batch = batch_samples([single, sample(4)])
        vector, mask = model.price_inputs(batch)
        shaped = vector[:, : 4 * 40].reshape(2, 4, 40)
        self.assertTrue(torch.equal(shaped[0, 1:], torch.full((3, 40), -1000.0)))
        fanout = vector[:, 4 * 40 : 8 * 40].reshape(2, 4, 40)
        self.assertTrue(torch.allclose(fanout[0, 0], single["groups"][1][0], atol=2e-6))
        tables = vector[:, 8 * 40 : 8 * 40 + 16].reshape(2, 4, 4)
        self.assertTrue(torch.equal(tables[0, 1:, 0], torch.zeros(3)))
        self.assertEqual(mask[0].sum().item(), 5)
        with patch.object(
            model.len_net, "forward", wraps=model.len_net.forward
        ) as call:
            model(batch)
            self.assertTrue(
                torch.equal(call.call_args.args[0], torch.log1p(batch["counts"]))
            )
        self.assertTrue(
            torch.allclose(model(batch_samples([single])), model(batch)[:1], atol=1e-6)
        )
        dirty = {
            **batch,
            "groups": [
                g.masked_fill(~m[..., None], float("inf"))
                for g, m in zip(batch["groups"], batch["masks"])
            ],
        }
        self.assertTrue(torch.equal(model(batch), model(dirty)))

    def test_capacity_equal_passes_and_overflow_is_explicit(self):
        model = self.model()
        self.assertTrue(
            torch.equal(
                model(batch_samples([sample(4)])), torch.log(torch.tensor([2.0]))
            )
        )
        with self.assertRaisesRegex(
            UnsupportedCapacity, "unsupported.*synthetic.*join_hist.*5.*4"
        ):
            model(batch_samples([sample(5)]))
        invalid = {**self.binding, "max_counts": [5, 4, 4, 4]}
        with self.assertRaisesRegex(ValueError, "capacity.*hash"):
            validate_capacity(invalid)
        empty = sample(1)
        empty["groups"] = [
            torch.empty(0, 40),
            torch.empty(0, 40),
            torch.rand(1, 4),
            torch.empty(0, 43),
        ]
        self.assertTrue(torch.isfinite(model(batch_samples([empty]))).all())

    def test_lora_is_registered_qv_only_and_functionally_replaceable(self):
        model = self.live_head(self.model(lora=True))
        params = OrderedDict(model.named_parameters())
        names = _adaptation_names(model, params, "lora")
        self.assertEqual(sum(params[n].numel() for n in names), 10240)
        self.assertTrue(all(isinstance(params[n], nn.Parameter) for n in names))
        self.assertTrue(
            all(
                "encoder.blocks." in n and (".Query." in n or ".Value." in n)
                for n in names
            )
        )
        self.assertTrue(
            all(
                torch.count_nonzero(params[n]) == 0
                for n in names
                if n.endswith("lora_B")
            )
        )
        batch = batch_samples([sample(2), sample(3)])
        changed = OrderedDict(
            (n, p + 0.3 if n in names and n.endswith("lora_B") else p)
            for n, p in params.items()
        )
        self.assertFalse(torch.equal(forward(model, changed, batch), model(batch)))
        self.assertTrue(all(p.requires_grad for p in model.parameters()))
        clone = self.model(lora=True)
        self.assertTrue(
            all(
                torch.equal(params[n], dict(clone.named_parameters())[n]) for n in names
            )
        )

    def test_lora_zero_head_and_b_zero_gradient_mathematics(self):
        from spatial_meta_ce.price_model import LoRALinear

        generator = torch.Generator().manual_seed(42)
        layer = LoRALinear(nn.Linear(64, 16), generator)
        x = torch.rand(2, 64)
        zero = layer(x)
        params = dict(layer.named_parameters())
        from torch.func import functional_call

        changed = {**params, "lora_A": params["lora_A"] + 1}
        self.assertTrue(
            torch.equal(zero, functional_call(layer, changed, (x,), strict=True))
        )
        b = torch.full_like(params["lora_B"], 0.2)
        changed = {**params, "lora_B": b}
        expected = nn.functional.linear(
            x, layer.weight, layer.bias
        ) + nn.functional.linear(nn.functional.linear(x, layer.lora_A), b)
        self.assertTrue(
            torch.equal(expected, functional_call(layer, changed, (x,), strict=True))
        )
        zero.sum().backward()
        self.assertEqual(torch.count_nonzero(layer.lora_A.grad).item(), 0)
        self.assertGreater(layer.lora_B.grad.abs().sum().item(), 0)
        model = self.model(lora=True)
        self.assertEqual(torch.count_nonzero(model.head[-1].weight).item(), 0)
        self.assertEqual(torch.count_nonzero(model.head[-1].bias).item(), 0)
        batch = batch_samples([sample(2)])
        model(batch).sum().backward()
        upstream = [p for n, p in model.named_parameters() if not n.startswith("head.")]
        self.assertTrue(
            all(
                p.grad is not None and torch.count_nonzero(p.grad) == 0
                for p in upstream
            )
        )

    def test_lora_support_changes_only_fast_ab_and_outer_reaches_all_groups(self):
        model = self.live_head(self.model(lora=True))
        support = batch_samples([sample(2, 200), sample(3, 400)])
        query = batch_samples([sample(1, 300), sample(4, 500)])
        base = {n: p.detach().clone() for n, p in model.named_parameters()}
        first = adapt(model, support, lr=0.1, steps=1, scope="lora")
        fast = adapt(model, support, lr=0.1, steps=5, scope="lora")
        for name, param in model.named_parameters():
            self.assertTrue(torch.equal(param, base[name]))
            if "lora_" not in name:
                self.assertTrue(torch.equal(fast[name], base[name]))
        self.assertTrue(
            any(not torch.equal(fast[n], base[n]) for n in base if n.endswith("lora_A"))
        )
        self.assertTrue(
            any(
                not torch.equal(first[n], base[n]) for n in base if n.endswith("lora_B")
            )
        )
        loss = meta_loss(model, support, query, 0.1, 5, scope="lora")
        loss.backward()
        for group in ("base", "head", "lora_A", "lora_B"):
            selected = [
                p
                for n, p in model.named_parameters()
                if (
                    (group == "base" and not n.startswith("head.") and "lora_" not in n)
                    or (group == "head" and n.startswith("head."))
                    or n.endswith(group)
                )
            ]
            self.assertTrue(
                all(
                    p.grad is not None and torch.isfinite(p.grad).all()
                    for p in selected
                )
            )
            self.assertGreater(sum(float(p.grad.abs().sum()) for p in selected), 0)
        meta_loss(
            model,
            batch_samples([sample(1, 999)]),
            batch_samples([sample(3, 888)]),
            0.05,
            2,
            scope="lora",
        ).backward()
        second = adapt(model, support, lr=0.1, steps=5, scope="lora")
        self.assertTrue(all(torch.equal(fast[n], second[n]) for n in fast))

    def test_scope_mode_isolation_and_nonfinite_fail_closed(self):
        model = self.live_head(self.model(lora=True))
        support = batch_samples([sample(1)])
        before = {n: p.clone() for n, p in model.named_parameters()}
        for scope in ("all", "full", "head", "lora"):
            names = _adaptation_names(model, dict(model.named_parameters()), scope)
            self.assertTrue(names)
            if scope == "head":
                self.assertTrue(all(n.startswith("head.") for n in names))
            model.train()
            model.filter_encoder.eval()  # Preserve even heterogeneous caller modes.
            original_modes = {n: m.training for n, m in model.named_modules()}
            original_grad = {n: p.requires_grad for n, p in model.named_parameters()}
            modes = []
            hook = model.scale_encoder.register_forward_pre_hook(
                lambda module, _: modes.append(module.training)
            )
            first = adapt(model, support, steps=2, scope=scope, training=False)
            hook.remove()
            self.assertEqual(modes, [False, False])
            self.assertTrue(model.training)
            self.assertEqual(
                original_modes, {n: m.training for n, m in model.named_modules()}
            )
            self.assertEqual(
                original_grad, {n: p.requires_grad for n, p in model.named_parameters()}
            )
            again = adapt(model, support, steps=2, scope=scope, training=False)
            self.assertTrue(all(torch.equal(first[n], again[n]) for n in first))
        self.assertTrue(
            all(torch.equal(p, before[n]) for n, p in model.named_parameters())
        )
        with self.assertRaisesRegex(ValueError, "LoRA|lora"):
            adapt(SetEstimator(), support, scope="lora")
        invalid = {**support, "label": torch.full_like(support["label"], float("nan"))}
        with self.assertRaises(FloatingPointError):
            adapt(model, invalid, scope="lora")

    def test_meta_support_and_query_use_same_training_mode(self):
        model = self.live_head(self.model(lora=True)).eval()
        batch = batch_samples([sample(2)])
        modes = []
        hook = model.scale_encoder.register_forward_pre_hook(
            lambda module, _: modes.append(module.training)
        )
        meta_loss(model, batch, batch, 0.001, 1, scope="lora").backward()
        hook.remove()
        self.assertEqual(modes, [True, True])
        self.assertFalse(model.training)
        from spatial_meta_ce.meta import meta_loss_terms

        modes = []
        hook = model.scale_encoder.register_forward_pre_hook(
            lambda module, _: modes.append(module.training)
        )
        meta_loss_terms(model, batch, batch, 0.001, 1, scope="lora")["total"].backward()
        hook.remove()
        self.assertEqual(modes, [True, True, True])
        self.assertFalse(model.training)

    def test_candidate_uses_log_cardinality_never_price_label_conversion(self):
        batches = batch_samples([sample(1, 0), sample(2, 1), sample(3, 20)])
        self.assertTrue(
            np.array_equal(
                batches["label"].numpy(), np.log([1, 1, 20]).astype(np.float32)
            )
        )
        model = self.model(lora=True)
        with (
            patch(
                "spatial_meta_ce.baselines.price_support",
                side_effect=AssertionError("wrong label"),
            ),
            patch(
                "spatial_meta_ce.baselines.price_log_predictions",
                side_effect=AssertionError("wrong decode"),
            ),
        ):
            meta_loss(model, batches, batches, 0.001, 1, scope="all").backward()

    def test_candidate_checkpoint_malformed_metadata_is_a_value_error(self):
        model = self.model()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "candidate.pt"
            save_checkpoint(
                path,
                model,
                None,
                0,
                None,
                0,
                {"manifest_hashes": self.binding["cache_hashes"]},
            )
            original = torch.load(path, map_location="cpu", weights_only=False)
            for value in (None, [], "malformed"):
                invalid = {**original, "metadata": value}
                torch.save(invalid, path)
                with (
                    self.subTest(value=value),
                    self.assertRaisesRegex(ValueError, "metadata"),
                ):
                    load_checkpoint(path)

    def test_new_and_legacy_checkpoint_roundtrip_and_rejections(self):
        batch = batch_samples([sample(2)])
        models = [
            SetEstimator(),
            SetEstimator(arch="sum"),
            create_model({"arch": "sum_anchor_only"}),
            self.model(),
            self.model("price_tiny"),
            self.model(lora=True),
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.pt"
            for model in models:
                self.live_head(model)
                meta = (
                    {"manifest_hashes": self.binding["cache_hashes"]}
                    if hasattr(model, "capacity")
                    else {}
                )
                save_checkpoint(path, model, None, 0, None, 0, meta)
                loaded, state = load_checkpoint(path)
                self.assertTrue(torch.equal(model(batch), loaded(batch)))
                self.assertEqual(loaded.config, model.config)
                broken = copy.deepcopy(state)
                broken["model_type"] = "unknown"
                torch.save(broken, path)
                with self.assertRaises(ValueError):
                    load_checkpoint(path)
            model = self.model()
            save_checkpoint(
                path,
                model,
                None,
                0,
                None,
                0,
                {"manifest_hashes": self.binding["cache_hashes"]},
            )
            original = torch.load(path, weights_only=False)
            for change in ("type", "config", "capacity", "cache"):
                state = copy.deepcopy(original)
                if change == "type":
                    state["model_type"] = "grouped_set_residual"
                elif change == "config":
                    state["model_config"]["d_model"] = 32
                elif change == "capacity":
                    state["model_config"]["capacity"]["max_counts"][0] += 1
                else:
                    state["metadata"]["manifest_hashes"][TRAIN_DATABASES[0]] = "0" * 64
                torch.save(state, path)
                with self.assertRaises(ValueError):
                    load_checkpoint(path)
            legacy = SetEstimator(arch="sum")
            save_checkpoint(path, legacy, None, 0, None, 0, {})
            state = torch.load(path, weights_only=False)
            self.assertEqual(state["model_type"], "grouped_set_residual")
            state["model"]["unexpected"] = torch.ones(1)
            torch.save(state, path)
            with self.assertRaises(RuntimeError):
                load_checkpoint(path)


if __name__ == "__main__":
    unittest.main()
