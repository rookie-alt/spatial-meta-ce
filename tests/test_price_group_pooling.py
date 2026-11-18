"""A1-PG group pooling contracts; synthetic CPU smoke, not research evidence."""

import copy
import json
import tempfile
import unittest
from collections import OrderedDict
from pathlib import Path
from unittest.mock import patch

import torch
from test_price_model import capacity, sample
from torch import nn

from spatial_meta_ce.checkpoints import load_checkpoint, save_checkpoint
from spatial_meta_ce.data import GROUP_DIMS, batch_samples
from spatial_meta_ce.manifests import digest
from spatial_meta_ce.meta import adapt, forward, meta_loss
from spatial_meta_ce.model import create_model
from spatial_meta_ce.price_model import PriceResidualEstimator
from spatial_meta_ce.price_protocol import a_model_config, exploratory_model_config
from spatial_meta_ce.train import _validate_fixed_budget, train

ROOT = Path(__file__).resolve().parents[1]
MODEL_TYPES = {
    "mean": "price_lite_pg_group_mean_residual",
    "attention": "price_lite_pg_group_attention_residual",
}


def grouped_sample(counts, true=50.0):
    result = sample(1, true)
    result["groups"] = [
        torch.rand(n, dim) * (10 if i == 1 else 1)
        for i, (n, dim) in enumerate(zip(counts, GROUP_DIMS))
    ]
    result["hash"] = digest([list(counts), true])
    return result


class PriceGroupPoolingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.binding = capacity((3, 2, 2, 4))

    def setUp(self):
        torch.manual_seed(42)

    def model(self, mode="attention", binding=None):
        return create_model(
            {
                "arch": "price_lite",
                "capacity": self.binding if binding is None else binding,
                "pg_head_input": True,
                "group_pooling": mode,
            }
        )

    @staticmethod
    def live_head(model):
        # The zero residual output deliberately gives the trunk zero gradient.
        nn.init.normal_(model.head[-1].weight, std=0.1)
        return model

    def test_explicit_modes_have_distinct_canonical_identities_and_dimensions(self):
        types = {"price_lite_pg_residual"}
        for mode, model_type in MODEL_TYPES.items():
            with self.subTest(mode=mode):
                model = self.model(mode)
                self.assertEqual(model.model_type, model_type)
                self.assertEqual(model.config["group_pooling"], mode)
                self.assertEqual(model.head[0].in_features, 82 + 4 * 64)
                rebuilt = create_model(model.config, model_type)
                self.assertEqual(rebuilt.config, model.config)
                types.add(model.model_type)
                batch = batch_samples([grouped_sample((2, 1, 2, 3))])
                self.assertTrue(torch.equal(model(batch), batch["pg_log"]))
                self.assertTrue(
                    torch.equal(model.output_anchor(batch), batch["pg_log"])
                )
        self.assertEqual(len(types), 3)

    def test_pooling_flag_and_pg_flag_validation_are_strict(self):
        for mode in (None, False, True, 0, 1, [], {}, "", "Mean", "mean+attention"):
            with self.subTest(mode=mode), self.assertRaisesRegex(ValueError, "pooling"):
                self.model(mode)
        for changes in (
            {"pg_head_input": False},
            {"pg_head_input": 1},
            {"pg_head_input": "true"},
            {"arch": "price_tiny"},
        ):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                PriceResidualEstimator(
                    **{
                        "capacity": self.binding,
                        "pg_head_input": True,
                        "group_pooling": "mean",
                        **changes,
                    }
                )
        for arch in ("set", "sum", "sum_anchor_only"):
            with self.subTest(arch=arch), self.assertRaises(ValueError):
                create_model({"arch": arch, "group_pooling": "mean"})

    def test_none_preserves_config_state_rng_and_baseline_forward(self):
        batch = batch_samples([grouped_sample((2, 1, 2, 3))])
        for arch, pg in (
            ("price_lite", False),
            ("price_lite", True),
            ("price_tiny", False),
        ):
            with self.subTest(arch=arch, pg=pg):
                config = {"arch": arch, "capacity": self.binding, "pg_head_input": pg}
                torch.manual_seed(42)
                original = create_model(config)
                original_rng = torch.get_rng_state().clone()
                torch.manual_seed(42)
                disabled = create_model({**config, "group_pooling": "none"})
                self.assertTrue(torch.equal(original_rng, torch.get_rng_state()))
                self.assertEqual(original.config, disabled.config)
                self.assertNotIn("group_pooling", disabled.config)
                self.assertEqual(
                    original.state_dict().keys(), disabled.state_dict().keys()
                )
                for name, parameter in original.state_dict().items():
                    self.assertTrue(
                        torch.equal(parameter, disabled.state_dict()[name]), name
                    )
                self.live_head(original)
                disabled.load_state_dict(original.state_dict(), strict=True)
                self.assertTrue(torch.equal(original(batch), disabled(batch)))
                x, mask = original.price_inputs(batch)
                scale = original.scale_encoder(
                    original.scale_embedding(x), mask[:, : 1 + sum(original.counts[:2])]
                )
                cls = original.filter_encoder(
                    original.filter_embedding(scale, x), mask
                )[:, 0]
                tables = batch["groups"][2].masked_fill(
                    ~batch["masks"][2][..., None], 0
                )
                expected = torch.cat(
                    [
                        cls,
                        original.len_net(torch.log1p(batch["counts"])),
                        tables[:, :, 0].sum(1, keepdim=True),
                        *([batch["pg_log"][:, None]] if pg else []),
                    ],
                    dim=1,
                )
                self.assertTrue(torch.equal(original._head_input(batch), expected))

    def test_pooling_keeps_shared_initialization_and_appends_features(self):
        batch = batch_samples([grouped_sample((2, 1, 2, 3))])
        for mode in MODEL_TYPES:
            torch.manual_seed(42)
            baseline = create_model(exploratory_model_config(self.binding))
            torch.manual_seed(42)
            candidate = self.model(mode)
            for name, parameter in baseline.state_dict().items():
                actual = candidate.state_dict()[name]
                if name == "head.0.weight":
                    actual = actual[:, :82]
                self.assertTrue(torch.equal(parameter, actual), (mode, name))
            self.assertTrue(
                torch.equal(
                    candidate._head_input(batch)[:, :82], baseline._head_input(batch)
                )
            )
            self.assertGreater(candidate.head[0].weight[:, 82:].abs().sum().item(), 0)
            extra = sum(p.numel() for p in candidate.parameters()) - sum(
                p.numel() for p in baseline.parameters()
            )
            self.assertEqual(extra, 32768 + (4228 if mode == "attention" else 0))

    def test_fixed_group_offsets_use_final_encoder_tokens_and_valid_counts(self):
        batch = batch_samples(
            [
                grouped_sample((2, 1, 0, 3)),
                grouped_sample((1, 0, 2, 1)),
            ]
        )
        for mode in MODEL_TYPES:
            with self.subTest(mode=mode):
                model = self.model(mode)
                _, mask = model.price_inputs(batch)
                tokens = torch.full((2, 12, 64), float("inf"))
                tokens[:, 0] = -17
                expected = []
                offset = 1
                for group, width in enumerate(model.counts):
                    rows = []
                    for row in range(2):
                        n = int(batch["counts"][row, group])
                        for index in range(n):
                            tokens[row, offset + index] = 10 * group + index + 1
                        rows.append(
                            torch.full((64,), 10 * group + (n + 1) / 2)
                            if n
                            else torch.zeros(64)
                        )
                    expected.append(torch.stack(rows))
                    offset += width
                if mode == "attention":
                    for scorer in model.group_scorers:
                        for parameter in scorer.parameters():
                            nn.init.zeros_(parameter)
                pooled = model._pool_groups(tokens, mask)
                self.assertTrue(torch.equal(pooled, torch.cat(expected, dim=1)))
                with patch.object(model.filter_encoder, "forward", return_value=tokens):
                    features = model._head_input(batch)
                self.assertTrue(torch.equal(features[:, :64], tokens[:, 0]))
                self.assertTrue(torch.equal(features[:, 82:], pooled))

    def test_attention_is_masked_softmax_of_independent_group_mlp_scores(self):
        model = self.model()
        self.assertEqual(len(model.group_scorers), 4)
        for scorer in model.group_scorers:
            self.assertEqual(scorer[0].in_features, 64)
            self.assertEqual(scorer[0].out_features, 16)
            self.assertEqual(scorer[-1].in_features, 16)
            self.assertEqual(scorer[-1].out_features, 1)
        self.assertEqual(sum(p.numel() for p in model.group_scorers.parameters()), 4228)
        batch = batch_samples(
            [grouped_sample((2, 1, 2, 3)), grouped_sample((1, 0, 1, 2))]
        )
        _, mask = model.price_inputs(batch)
        tokens = torch.randn(2, mask.shape[1], 64).masked_fill(
            ~mask[..., None], float("inf")
        )
        pooled = model._pool_groups(tokens, mask)
        offset = 1
        for group, width in enumerate(model.counts):
            for row in range(2):
                valid = tokens[row, offset : offset + width][
                    mask[row, offset : offset + width]
                ]
                expected = torch.zeros(64)
                if len(valid):
                    weights = model.group_scorers[group](valid).squeeze(-1).softmax(0)
                    expected = (valid * weights[:, None]).sum(0)
                torch.testing.assert_close(
                    pooled[row, group * 64 : (group + 1) * 64],
                    expected,
                    rtol=1e-6,
                    atol=1e-6,
                )
            offset += width

    def test_raw_padding_batch_invariance_and_nonzero_encoder_padding(self):
        one = grouped_sample((1, 0, 1, 0))
        batch = batch_samples([one, grouped_sample((3, 2, 2, 4))])
        for mode in MODEL_TYPES:
            model = self.live_head(self.model(mode)).eval()
            values = []
            hook = model.filter_encoder.register_forward_hook(
                lambda _, args, out, values=values: values.append(out.detach())
            )
            clean = model(batch)
            hook.remove()
            _, mask = model.price_inputs(batch)
            self.assertGreater(torch.count_nonzero(values[0][~mask]).item(), 0)
            for value in (12345.0, float("inf"), float("nan")):
                dirty = {
                    **batch,
                    "groups": [
                        g.masked_fill(~m[..., None], value)
                        for g, m in zip(batch["groups"], batch["masks"])
                    ],
                }
                self.assertTrue(torch.equal(model(dirty), clean), (mode, value))
            torch.testing.assert_close(
                model(batch_samples([one])), clean[:1], rtol=1e-6, atol=1e-6
            )
            padded = {
                **batch,
                "groups": [
                    nn.functional.pad(g, (0, 0, 0, 3), value=float("inf"))
                    for g in batch["groups"]
                ],
                "masks": [
                    nn.functional.pad(m, (0, 3), value=False) for m in batch["masks"]
                ],
            }
            self.assertTrue(torch.equal(model(padded), clean))

    def test_empty_rows_are_exact_zero_and_backward_is_finite(self):
        batch = batch_samples(
            [grouped_sample((0, 0, 0, 0)), grouped_sample((2, 1, 2, 3))]
        )
        for mode in MODEL_TYPES:
            model = self.model(mode)
            _, mask = model.price_inputs(batch)
            tokens = (
                torch.randn(2, mask.shape[1], 64)
                .masked_fill(~mask[..., None], float("inf"))
                .requires_grad_()
            )
            pooled = model._pool_groups(tokens, mask)
            self.assertTrue(torch.equal(pooled[0], torch.zeros(256)))
            pooled.square().sum().backward()
            self.assertTrue(torch.isfinite(tokens.grad).all())
            self.assertEqual(torch.count_nonzero(tokens.grad[~mask]).item(), 0)
            self.assertTrue(torch.isfinite(model(batch)).all())

    def test_global_zero_capacity_groups_have_differentiable_zero(self):
        binding = capacity((0, 0, 2, 0))
        batch = batch_samples([grouped_sample((0, 0, 1, 0))])
        for mode in MODEL_TYPES:
            with self.subTest(mode=mode):
                model = self.live_head(self.model(mode, binding))
                features = model._head_input(batch)
                for group in (0, 1, 3):
                    self.assertTrue(
                        torch.equal(
                            features[:, 82 + group * 64 : 82 + (group + 1) * 64],
                            torch.zeros(1, 64),
                        )
                    )
                model(batch).sum().backward()
                self.assertTrue(
                    all(
                        p.grad is not None and torch.isfinite(p.grad).all()
                        for p in model.parameters()
                    )
                )
                if mode == "attention":
                    for group in (0, 1, 3):
                        self.assertTrue(
                            all(
                                torch.count_nonzero(p.grad) == 0
                                for p in model.group_scorers[group].parameters()
                            )
                        )
                fast = adapt(model, batch, steps=1, scope="all")
                self.assertTrue(torch.isfinite(forward(model, fast, batch)).all())

    def test_live_head_reaches_all_scorers_and_strict_functional_adaptation(self):
        model = self.live_head(self.model())
        batch = batch_samples(
            [grouped_sample((3, 2, 2, 4), 200), grouped_sample((2, 2, 2, 3), 400)]
        )
        before = {n: p.detach().clone() for n, p in model.named_parameters()}
        params = OrderedDict(model.named_parameters())
        self.assertTrue(torch.equal(model(batch), forward(model, params, batch)))
        with self.assertRaisesRegex(RuntimeError, "Missing key"):
            forward(model, OrderedDict(list(params.items())[1:]), batch)
        model(batch).square().mean().backward()
        for scorer in model.group_scorers:
            self.assertTrue(
                all(
                    p.grad is not None and torch.isfinite(p.grad).all()
                    for p in scorer.parameters()
                )
            )
            self.assertGreater(scorer[0].weight.grad.abs().sum().item(), 0)
            self.assertGreater(scorer[-1].weight.grad.abs().sum().item(), 0)
        model.zero_grad(set_to_none=True)
        fast = adapt(model, batch, steps=1, lr=0.01, scope="all")
        self.assertTrue(
            any(
                not torch.equal(fast[n], before[n])
                for n in fast
                if n.startswith("group_scorers.")
            )
        )
        head = adapt(model, batch, steps=1, scope="head")
        self.assertTrue(
            all(
                torch.equal(head[n], before[n])
                for n in head
                if not n.startswith("head.")
            )
        )
        self.assertTrue(
            all(torch.equal(p, before[n]) for n, p in model.named_parameters())
        )
        meta_loss(model, batch, batch, 0.001, 1, scope="all").backward()
        self.assertTrue(
            all(
                p.grad is not None and torch.isfinite(p.grad).all()
                for p in model.parameters()
            )
        )

    def test_mean_supports_strict_functional_adaptation(self):
        model = self.live_head(self.model("mean"))
        batch = batch_samples([grouped_sample((2, 1, 2, 3), 200)])
        fast = adapt(model, batch, steps=1, scope="all")
        self.assertTrue(torch.isfinite(forward(model, fast, batch)).all())

    def test_counts_masks_and_mapping_fail_closed(self):
        batch = batch_samples([grouped_sample((2, 1, 2, 3))])
        model = self.model()
        for changes in (
            {"counts": batch["counts"] + 0.5},
            {"counts": batch["counts"] + 10},
            {"counts": torch.full_like(batch["counts"], float("nan"))},
            {"masks": [~m for m in batch["masks"]]},
            {"masks": [m.float() for m in batch["masks"]]},
            {"groups": batch["groups"][:3]},
            {"mapping": {}},
        ):
            with self.subTest(keys=list(changes)), self.assertRaises(ValueError):
                model({**batch, **changes})

    def test_checkpoint_roundtrip_and_identity_config_hash_rejections(self):
        batch = batch_samples([grouped_sample((2, 1, 2, 3))])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "synthetic.pt"
            for mode, model_type in MODEL_TYPES.items():
                model = self.live_head(self.model(mode))
                save_checkpoint(
                    path,
                    model,
                    None,
                    1,
                    None,
                    0,
                    {"manifest_hashes": self.binding["cache_hashes"]},
                )
                restored, original = load_checkpoint(path)
                self.assertEqual(restored.config, model.config)
                self.assertEqual(restored.model_type, model_type)
                self.assertTrue(torch.equal(restored(batch), model(batch)))
                self.assertEqual(original["model_config_sha256"], digest(model.config))
                for change in (
                    "baseline_type",
                    "other_pool_type",
                    "missing_mode",
                    "changed_mode",
                    "hash",
                    "missing_hash",
                    "pg",
                    "unknown_config",
                    "explicit_none",
                    "cache",
                    "weights",
                ):
                    invalid = copy.deepcopy(original)
                    if change == "baseline_type":
                        invalid["model_type"] = "price_lite_pg_residual"
                    elif change == "other_pool_type":
                        invalid["model_type"] = MODEL_TYPES[
                            "mean" if mode == "attention" else "attention"
                        ]
                    elif change == "missing_mode":
                        invalid["model_config"].pop("group_pooling")
                    elif change == "changed_mode":
                        invalid["model_config"]["group_pooling"] = (
                            "mean" if mode == "attention" else "attention"
                        )
                    elif change == "hash":
                        invalid["model_config_sha256"] = "0" * 64
                    elif change == "missing_hash":
                        invalid.pop("model_config_sha256")
                    elif change == "pg":
                        invalid["model_config"]["pg_head_input"] = False
                    elif change == "unknown_config":
                        invalid["model_config"]["unexpected"] = 1
                    elif change == "explicit_none":
                        invalid["model_config"]["group_pooling"] = "none"
                    elif change == "cache":
                        invalid["metadata"]["manifest_hashes"][
                            next(iter(self.binding["cache_hashes"]))
                        ] = "0" * 64
                    else:
                        invalid["model"].pop("head.0.weight")
                    if change not in ("hash", "missing_hash"):
                        invalid["model_config_sha256"] = digest(invalid["model_config"])
                    torch.save(invalid, path)
                    with (
                        self.subTest(mode=mode, change=change),
                        self.assertRaises((ValueError, RuntimeError)),
                    ):
                        load_checkpoint(path)
                # A direct dispatch cannot relabel a candidate as baseline.
                with self.assertRaisesRegex(ValueError, "model_type"):
                    create_model(model.config, "price_lite_pg_residual")

    def test_checkpoint_rejects_numeric_aliases_and_reloads_adapted_weights(self):
        batch = batch_samples([grouped_sample((2, 1, 2, 3), 200)])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "adapted.pt"
            for mode in MODEL_TYPES:
                model = self.live_head(self.model(mode))
                fast = adapt(model, batch, steps=2, scope="all")
                model.load_state_dict(fast, strict=True)
                save_checkpoint(
                    path, model, None, 2, None, 0,
                    {"manifest_hashes": self.binding["cache_hashes"]},
                )
                restored, original = load_checkpoint(path)
                self.assertEqual(digest(restored.config), original["model_config_sha256"])
                self.assertTrue(torch.equal(restored(batch), model(batch)))
                for field, alias in (("d_model", 64.0), ("dropout", 0)):
                    invalid = copy.deepcopy(original)
                    invalid["model_config"][field] = alias
                    invalid["model_config_sha256"] = digest(invalid["model_config"])
                    torch.save(invalid, path)
                    with (
                        self.subTest(mode=mode, field=field),
                        self.assertRaisesRegex(ValueError, "canonical"),
                    ):
                        load_checkpoint(path)

    def test_checkpoint_save_rejects_mutated_candidate_identity_or_config(self):
        with tempfile.TemporaryDirectory() as directory:
            for change in (
                "type",
                "legacy_type",
                "mode",
                "config",
                "pg",
                "pg_as_int",
                "lora_as_int",
                "extra_key",
            ):
                model = self.model()
                if change == "type":
                    model.model_type = "price_lite_pg_residual"
                elif change == "legacy_type":
                    model.model_type = "grouped_set_residual"
                elif change == "mode":
                    model.group_pooling = "mean"
                elif change == "config":
                    model.config["group_pooling"] = "mean"
                elif change == "pg":
                    model.config["pg_head_input"] = False
                elif change == "pg_as_int":
                    model.config["pg_head_input"] = 1
                elif change == "lora_as_int":
                    model.config["lora"] = 0
                else:
                    model.config["extra"] = True
                with (
                    self.subTest(change=change),
                    self.assertRaisesRegex(ValueError, "pooling|candidate|canonical"),
                ):
                    save_checkpoint(
                        Path(directory) / "bad.pt",
                        model,
                        None,
                        0,
                        None,
                        0,
                        {"manifest_hashes": self.binding["cache_hashes"]},
                    )

    def test_container_aliases_rejected_before_checkpoint_or_directory_creation(self):
        for mode in MODEL_TYPES:
            for lora in (False, True):
                with tempfile.TemporaryDirectory() as directory:
                    root = Path(directory)
                    path = root / "existing.pt"
                    metadata = {"manifest_hashes": self.binding["cache_hashes"]}
                    config = {**self.model(mode).config, "lora": lora}
                    save_checkpoint(path, create_model(config), None, 0, None, 0, metadata)
                    original_bytes = path.read_bytes()
                    for location in ("config", "capacity"):
                        for field in ("max_counts", "databases", "database_max_counts"):
                            model = create_model(config)
                            binding = model.config["capacity"] if location == "config" else model.capacity
                            original_digest = digest(binding)
                            if field == "database_max_counts":
                                database = next(iter(binding[field]))
                                binding[field][database] = tuple(binding[field][database])
                            else:
                                binding[field] = tuple(binding[field])
                            self.assertEqual(digest(binding), original_digest)
                            fresh = root / f"new-{location}-{field}" / "candidate.pt"
                            for target in (path, fresh):
                                with (
                                    self.subTest(mode=mode, lora=lora, location=location, field=field, target=target.name),
                                    self.assertRaisesRegex(ValueError, "canonical|identity|pooling"),
                                ):
                                    save_checkpoint(target, model, None, 0, None, 0, metadata)
                                self.assertEqual(path.read_bytes(), original_bytes)
                                self.assertFalse(target.with_suffix(".pt.tmp").exists())
                                self.assertFalse(fresh.parent.exists())

    def test_old_122_and_pg_training_authorizations_reject_pooling_before_cache_io(
        self,
    ):
        base = json.loads((ROOT / "configs/fomaml_joint_sum_4k.json").read_text())
        data = json.loads((ROOT / "configs/data.json").read_text())
        for mode in MODEL_TYPES:
            for stage in ("122-A1", "122-A1-PG"):
                canonical = (
                    a_model_config("A1", self.binding)
                    if stage == "122-A1"
                    else exploratory_model_config(self.binding)
                )
                old = {
                    **base,
                    "method": "erm",
                    "meta_objective": "post",
                    "fixed_stage": stage,
                    "model": canonical,
                }
                _validate_fixed_budget(old)
                changed = {
                    **old,
                    "model": {
                        **canonical,
                        "group_pooling": mode,
                        "pg_head_input": True,
                    },
                }
                with (
                    self.subTest(mode=mode, stage=stage),
                    self.assertRaisesRegex(ValueError, "frozen model"),
                ):
                    _validate_fixed_budget(changed)
                with (
                    patch(
                        "spatial_meta_ce.train.StorePool",
                        side_effect=AssertionError("cache accessed"),
                    ),
                    self.assertRaisesRegex(ValueError, "frozen model"),
                ):
                    train(changed, data, "unused")


if __name__ == "__main__":
    unittest.main()
