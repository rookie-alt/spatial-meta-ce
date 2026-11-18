"""Relation-attention contracts on synthetic data: smoke-not-research only."""

import copy
import itertools
import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn
from torch.func import functional_call

from spatial_meta_ce.checkpoints import load_checkpoint, save_checkpoint
from spatial_meta_ce.data import GROUP_DIMS, TRANSFORM_VERSION, batch_samples
from spatial_meta_ce.manifests import digest
from spatial_meta_ce.meta import _adaptation_names, adapt, forward, meta_loss
from spatial_meta_ce.model import create_model
from spatial_meta_ce.price_capacity import TRAIN_DATABASES
from spatial_meta_ce.price_protocol import fixed_model_scope
from vendor.price_baseline.module import Head

MODEL_TYPE = "price_lite_pg_relation_residual"


def synthetic_capacity(max_counts=(4, 6, 3, 3)):
    """No cache access: all hashes and maxima describe a synthetic fixture."""
    payload = {
        "schema": 1,
        "kind": "price_train_capacity_122",
        "split": "pretrain",
        "transform": TRANSFORM_VERSION,
        "databases": list(TRAIN_DATABASES),
        "max_counts": list(max_counts),
        "database_max_counts": {db: list(max_counts) for db in TRAIN_DATABASES},
        "cache_hashes": {db: digest(["synthetic-cache", db]) for db in TRAIN_DATABASES},
        "manifest_sha256": {
            db: digest(["synthetic-manifest", db]) for db in TRAIN_DATABASES
        },
    }
    return {**payload, "capacity_sha256": digest(payload)}


def synthetic_sample(singleton=False, database="synthetic"):
    counts = [0, 0, 1, 0] if singleton else [3, 4, 3, 2]
    generator = torch.Generator().manual_seed(18 if singleton else 19)
    mapping = {
        "counts": counts,
        "join_column_tables": [] if singleton else [0, 1, 2],
        "filter_column_tables": [] if singleton else [0, 2],
        "fanout_join_columns": [] if singleton else [0, 1, 1, 2],
    }
    return {
        "groups": [
            torch.rand(n, dim, generator=generator)
            for n, dim in zip(counts, GROUP_DIMS)
        ],
        "mapping": mapping,
        "mapping_mode": "mapping",
        "true": 50.0,
        "pg": 2.0,
        "database": database,
        "id": f"{database}/pretrain/{singleton}",
        "hash": digest(["synthetic", singleton]),
    }


def unmapped(batch):
    return {k: v for k, v in batch.items() if k not in ("mapping", "mapping_mode")}


class RelationAttentionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(42)
        self.capacity = synthetic_capacity()
        self.batch = batch_samples([synthetic_sample(), synthetic_sample(True)])

    def model(self, *, lora=False, capacity=None, relation=True):
        config = {
            "arch": "price_lite",
            "capacity": capacity or self.capacity,
            "pg_head_input": True,
            "lora": lora,
        }
        if relation:
            config["relation_attention"] = True
        return create_model(config)

    @staticmethod
    def live_head(model):
        with torch.no_grad():
            nn.init.normal_(model.head[-1].weight, std=0.1)
        return model

    @staticmethod
    def set_bias(model):
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                if name.endswith("relation_bias"):
                    parameter.copy_(torch.linspace(-0.7, 0.7, 8))

    def test_opt_in_identity_and_preserved_default_state(self):
        baseline = self.model(relation=False)
        torch.manual_seed(42)
        disabled = create_model({**baseline.config, "relation_attention": False})
        self.assertNotIn("relation_attention", baseline.config)
        self.assertEqual(disabled.config, baseline.config)
        self.assertEqual(disabled.model_type, "price_lite_pg_residual")
        self.assertEqual(baseline.state_dict().keys(), disabled.state_dict().keys())
        for name, value in baseline.state_dict().items():
            self.assertTrue(torch.equal(value, disabled.state_dict()[name]), name)
        candidate = self.model()
        self.assertEqual(candidate.model_type, MODEL_TYPE)
        self.assertIs(candidate.config["relation_attention"], True)
        self.assertTrue(torch.equal(candidate(self.batch), self.batch["pg_log"]))
        for config in (
            {"relation_attention": 1},
            {"relation_attention": "mapping"},
            {"pg_head_input": False, "relation_attention": True},
            {"arch": "price_tiny", "pg_head_input": False, "relation_attention": True},
        ):
            with self.subTest(config=config), self.assertRaises(ValueError):
                create_model({**baseline.config, **config})

    def test_zero_bias_same_initialization_and_live_output_match_baseline(self):
        for lora in (False, True):
            with self.subTest(lora=lora):
                torch.manual_seed(42)
                baseline = self.live_head(self.model(lora=lora, relation=False))
                torch.manual_seed(42)
                candidate = self.live_head(self.model(lora=lora))
                common = baseline.state_dict()
                for name, value in common.items():
                    self.assertTrue(
                        torch.equal(value, candidate.state_dict()[name]), name
                    )
                biases = [
                    p
                    for n, p in candidate.named_parameters()
                    if n.endswith("relation_bias")
                ]
                self.assertEqual(len(biases), 16)
                self.assertTrue(all(torch.count_nonzero(p) == 0 for p in biases))
                plain = unmapped(self.batch)
                self.assertFalse(torch.equal(baseline(plain), plain["pg_log"]))
                self.assertTrue(torch.equal(baseline(plain), candidate(self.batch)))
                self.assertTrue(
                    torch.equal(
                        baseline._head_input(plain), candidate._head_input(self.batch)
                    )
                )

    def test_exact_relations_offsets_directions_and_padding(self):
        from spatial_meta_ce.price_relations import build_relation_matrices

        sample = synthetic_sample()
        sample["groups"] = [g[:n] for g, n in zip(sample["groups"], (2, 2, 2, 1))]
        sample["mapping"] = {
            "counts": [2, 2, 2, 1],
            "join_column_tables": [0, 1],
            "filter_column_tables": [0],
            "fanout_join_columns": [0, 1],
        }
        batch = batch_samples([sample])
        scale, full = build_relation_matrices(batch, (3, 4, 3, 2))
        expected = torch.zeros(1, 13, 13, dtype=torch.long)
        # Capacity offsets: CLS=0, join=1, fanout=4, table=8, filter=11.
        for members in ((1, 4, 8, 11), (2, 5, 9)):
            for source, destination in itertools.permutations(members, 2):
                expected[0, source, destination] = 7
        for source, destination in ((1, 8), (2, 9), (11, 8)):
            expected[0, source, destination] = 2
            expected[0, destination, source] = 3
        for source, destination in ((4, 1), (5, 2)):
            expected[0, source, destination] = 4
            expected[0, destination, source] = 5
        expected[0, 4, 5] = expected[0, 5, 4] = 6
        for token in (0, 1, 2, 4, 5, 8, 9, 11):
            expected[0, token, token] = 1
        self.assertTrue(torch.equal(full, expected))
        self.assertTrue(torch.equal(scale, expected[:, :8, :8]))
        self.assertTrue(
            torch.equal(full, build_relation_matrices(batch, (3, 4, 3, 2))[1])
        )
        self.assertEqual(full.device, batch["counts"].device)

    def test_precedence_same_endpoint_fanout_pair_over_same_table(self):
        from spatial_meta_ce.price_relations import build_relation_matrices

        batch = copy.deepcopy(self.batch)
        # Independent directional shuffling may give both endpoints the same column.
        batch["mapping"]["fanout_join_columns"][0, 1] = 0
        _, full = build_relation_matrices(batch, self.capacity["max_counts"])
        self.assertEqual(full[0, 5, 6].item(), 6)
        self.assertEqual(full[0, 6, 5].item(), 6)
        self.assertEqual(full[0, 5, 1].item(), 4)
        self.assertEqual(full[0, 1, 5].item(), 5)
        self.assertEqual(full[0, 1, 11].item(), 2)
        self.assertEqual(full[0, 1, 1].item(), 1)

    def test_bias_is_added_before_key_mask_and_softmax_with_gradient(self):
        from spatial_meta_ce.price_relations import RelationHead

        original = Head(2, 2, 0.0)
        with torch.no_grad():
            original.Key.weight.zero_()
            original.Key.bias.zero_()
            original.Query.weight.zero_()
            original.Query.bias.zero_()
            original.Value.weight.copy_(torch.eye(2))
            original.Value.bias.zero_()
        head = RelationHead(original)
        inputs = torch.tensor([[[1.0, 0.0], [0.0, 1.0], [100.0, 100.0]]])
        mask = torch.tensor([[True, True, False]])
        relations = torch.tensor([[[1, 0, 2], [0, 1, 2], [0, 0, 0]]])
        self.assertTrue(
            torch.equal(head(inputs, mask, relations), original(inputs, mask))
        )
        with torch.no_grad():
            head.relation_bias[1] = torch.log(torch.tensor(3.0))
            head.relation_bias[2] = 100.0
        weights = head.attention_weights(inputs, mask, relations)
        torch.testing.assert_close(weights[0, 0], torch.tensor([0.75, 0.25, 0.0]))
        output = head(inputs, mask, relations)
        torch.testing.assert_close(output[0, 0], torch.tensor([0.75, 0.25]))
        output[0, 0, 0].backward()
        self.assertTrue(torch.isfinite(head.relation_bias.grad).all())
        self.assertGreater(head.relation_bias.grad[1].abs().item(), 0)

    def test_live_bias_changes_output_and_receives_trunk_gradient(self):
        model = self.live_head(self.model())
        before = model(self.batch).detach().clone()
        self.set_bias(model)
        changed = model(self.batch)
        self.assertGreater((changed - before).abs().max().item(), 1e-6)
        (changed - self.batch["label"]).square().mean().backward()
        grads = [
            p.grad for n, p in model.named_parameters() if n.endswith("relation_bias")
        ]
        self.assertEqual(len(grads), 16)
        self.assertTrue(all(g is not None and torch.isfinite(g).all() for g in grads))
        self.assertGreater(sum(g.abs().sum().item() for g in grads), 0)
        self.assertTrue(
            torch.equal(model.output_anchor(self.batch), self.batch["pg_log"])
        )

    def test_true_shuffled_use_identical_model_and_parameter_count(self):
        from spatial_meta_ce.price_relations import build_relation_matrices

        shuffled = copy.deepcopy(self.batch)
        shuffled["mapping_mode"] = "shuffled_mapping"
        shuffled["mapping"]["join_column_tables"][0] = torch.tensor([2, 0, 1])
        shuffled["mapping"]["filter_column_tables"][0] = torch.tensor([2, 0])
        shuffled["mapping"]["fanout_join_columns"][0] = torch.tensor([1, 2, 0, 1])
        true_relations = build_relation_matrices(
            self.batch, self.capacity["max_counts"]
        )
        shuffled_relations = build_relation_matrices(
            shuffled, self.capacity["max_counts"]
        )
        self.assertTrue(
            all(
                not torch.equal(a, b)
                for a, b in zip(true_relations, shuffled_relations)
            )
        )
        torch.manual_seed(42)
        true_model = self.live_head(self.model())
        torch.manual_seed(42)
        shuffled_model = self.live_head(self.model())
        self.assertEqual(true_model.config, shuffled_model.config)
        self.assertEqual(sum(p.numel() for p in true_model.parameters()), 294161)
        self.assertEqual(sum(p.numel() for p in shuffled_model.parameters()), 294161)
        self.assertEqual(
            sum(p.numel() for p in self.model(relation=False).parameters()), 294033
        )
        for name, value in true_model.state_dict().items():
            self.assertTrue(torch.equal(value, shuffled_model.state_dict()[name]))
        self.set_bias(true_model)
        self.set_bias(shuffled_model)
        self.assertFalse(torch.equal(true_model(self.batch), shuffled_model(shuffled)))

    def test_relations_ignore_features_pg_labels_and_database_names(self):
        from spatial_meta_ce.price_relations import build_relation_matrices

        changed = copy.deepcopy(self.batch)
        changed["groups"] = [g * 100 for g in changed["groups"]]
        changed["pg_log"] += 50
        changed["label"] *= -10
        changed["true"] += 999
        changed["databases"] = ["other-source"] * 2
        expected = build_relation_matrices(self.batch, self.capacity["max_counts"])
        actual = build_relation_matrices(changed, self.capacity["max_counts"])
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(expected, actual)))

    def test_missing_invalid_mixed_mapping_modes_and_presence_fail(self):
        model = self.model()
        invalids = [
            unmapped(self.batch),
            {**self.batch, "mapping": None},
            {**self.batch, "mapping_mode": None},
            {**self.batch, "mapping_mode": "unknown"},
            {**self.batch, "mapping_mode": ["mapping", "shuffled_mapping"]},
            {k: v for k, v in self.batch.items() if k != "mapping_mode"},
        ]
        for key in self.batch["mapping"]:
            mapping = dict(self.batch["mapping"])
            del mapping[key]
            invalids.append({**self.batch, "mapping": mapping})
        for bad in invalids:
            with self.subTest(keys=list(bad)), self.assertRaises(ValueError):
                model(bad)
        with self.assertRaises(ValueError):
            batch_samples([synthetic_sample(), {**synthetic_sample(), "mapping": None}])
        with self.assertRaises(ValueError):
            batch_samples(
                [
                    synthetic_sample(),
                    {**synthetic_sample(), "mapping_mode": "shuffled_mapping"},
                ]
            )
        with self.assertRaises(ValueError):
            self.model(relation=False)(self.batch)

    def test_mapping_dtype_shape_device_padding_and_range_fail(self):
        model = self.model()
        bads = []
        for key in self.batch["mapping"]:
            for change in (
                "float",
                "int32",
                "shape",
                "batch",
                "device",
                "padding",
                "presence",
                "range",
            ):
                bad = copy.deepcopy(self.batch)
                values = bad["mapping"][key]
                if change == "float":
                    bad["mapping"][key] = values.float()
                elif change == "int32":
                    bad["mapping"][key] = values.int()
                elif change == "shape":
                    bad["mapping"][key] = values[:, :-1]
                elif change == "batch":
                    bad["mapping"][key] = values[:1]
                elif change == "device":
                    bad["mapping"][key] = values.to("meta")
                elif change == "padding":
                    values[1, 0] = -2
                elif change == "presence":
                    values[0, 0] = -1
                else:
                    values[0, 0] = 99
                bads.append((key, change, bad))
        for key, change, bad in bads:
            with self.subTest(key=key, change=change), self.assertRaises(ValueError):
                model(bad)
        bad = copy.deepcopy(self.batch)
        bad["mapping"]["join_column_tables"][1, 0] = 1
        bad["counts"][1, 0] = 1
        bad["masks"][0][1, 0] = True
        with self.assertRaises(ValueError):
            model(bad)  # Index 1 is inside capacity but is a padded table in row 1.

    def test_bad_group_batch_shape_device_prefix_and_counts_fail(self):
        model = self.model()
        bads = []
        for change in (
            "count_batch",
            "fractional",
            "negative",
            "nonfinite",
            "mask_prefix",
            "mask_dtype",
            "group_batch",
            "group_device",
            "mask_device",
            "no_table",
            "odd_fanout",
        ):
            bad = copy.deepcopy(self.batch)
            if change == "count_batch":
                bad["counts"] = bad["counts"][:1]
            elif change == "fractional":
                bad["counts"][0, 0] = 1.5
            elif change == "negative":
                bad["counts"][0, 0] = -1
            elif change == "nonfinite":
                bad["counts"][0, 0] = float("nan")
            elif change == "mask_prefix":
                bad["masks"][0][1, :2] = torch.tensor([False, True])
            elif change == "mask_dtype":
                bad["masks"][0] = bad["masks"][0].float()
            elif change == "group_batch":
                bad["groups"][0] = bad["groups"][0][:1]
            elif change == "group_device":
                bad["groups"][0] = bad["groups"][0].to("meta")
            elif change == "mask_device":
                bad["masks"][0] = bad["masks"][0].to("meta")
            elif change == "no_table":
                bad["counts"][1, 2] = 0
                bad["masks"][2][1] = False
            else:
                bad["counts"][0, 1] = 3
                bad["masks"][1][0, 3] = False
                bad["mapping"]["fanout_join_columns"][0, 3] = -1
            bads.append((change, bad))
        for change, bad in bads:
            with self.subTest(change=change), self.assertRaises(ValueError):
                model(bad)

    def test_fanout_cannot_point_to_padded_join_or_have_incomplete_pair(self):
        model = self.model()
        bad = copy.deepcopy(self.batch)
        bad["counts"][0, 0] = 2
        bad["masks"][0][0, 2] = False
        bad["mapping"]["join_column_tables"][0, 2] = -1
        with self.assertRaises(ValueError):
            model(bad)  # fanout endpoint 2 now points to a masked join token.
        bad = copy.deepcopy(self.batch)
        bad["mapping"]["fanout_join_columns"][0, 1] = -1
        with self.assertRaises(ValueError):
            model(bad)

    def test_direct_batch_api_rejects_targets_and_missing_identity(self):
        model = self.model()
        for database in ("imdb", "stats", "genome", "ergastf1", "STATS"):
            bad = {**self.batch, "databases": ["synthetic", database]}
            with (
                self.subTest(database=database),
                self.assertRaisesRegex(ValueError, "source|target"),
            ):
                model(bad)
        for identity in (None, ["synthetic"], ["synthetic", None]):
            with self.subTest(identity=identity), self.assertRaises(ValueError):
                model({**self.batch, "databases": identity})
        bad = {k: v for k, v in self.batch.items() if k != "databases"}
        with self.assertRaises(ValueError):
            model(bad)
        with self.assertRaises(ValueError):
            model(batch_samples([synthetic_sample(database="stats")]))

    def test_arbitrary_masked_padding_is_inert_and_batch_composition_invariant(self):
        model = self.live_head(self.model())
        self.set_bias(model)
        expected = model(self.batch)
        bad_padding = copy.deepcopy(self.batch)
        for group, mask in zip(bad_padding["groups"], bad_padding["masks"]):
            group[~mask] = float("nan")
        self.assertTrue(torch.equal(model(bad_padding), expected))
        for group, mask in zip(bad_padding["groups"], bad_padding["masks"]):
            group[~mask] = float("inf")
        self.assertTrue(torch.equal(model(bad_padding), expected))
        singleton = batch_samples([synthetic_sample(True)])
        torch.testing.assert_close(model(singleton), expected[1:])

    def test_empty_and_zero_capacity_groups_finite_with_strict_full_adaptation(self):
        for maxima in ((4, 6, 3, 3), (0, 0, 1, 0), (3, 0, 1, 0), (0, 2, 1, 2)):
            with self.subTest(maxima=maxima):
                model = self.live_head(self.model(capacity=synthetic_capacity(maxima)))
                batch = batch_samples([synthetic_sample(True)])
                # Also exercise genuinely zero-width inputs, without a dummy token.
                for i in (0, 1, 3):
                    batch["groups"][i] = batch["groups"][i][:, :0]
                    batch["masks"][i] = batch["masks"][i][:, :0]
                for key in batch["mapping"]:
                    batch["mapping"][key] = batch["mapping"][key][:, :0]
                self.assertTrue(torch.isfinite(model(batch)).all())
                fast = adapt(model, batch, lr=0.001, steps=1, scope="all")
                self.assertTrue(torch.isfinite(forward(model, fast, batch)).all())

    def test_strict_functional_call_bias_parameters_and_pg_anchor(self):
        model = self.live_head(self.model())
        params = dict(model.named_parameters())
        expected = model(self.batch)
        self.assertTrue(
            torch.equal(
                functional_call(model, params, (self.batch,), strict=True), expected
            )
        )
        changed = dict(params)
        name = "scale_encoder.blocks.0.attention.heads.0.relation_bias"
        changed[name] = params[name] + torch.linspace(-2.0, 2.0, 8)
        self.assertFalse(
            torch.equal(
                functional_call(model, changed, (self.batch,), strict=True), expected
            )
        )
        del changed[name]
        with self.assertRaises(RuntimeError):
            functional_call(model, changed, (self.batch,), strict=True)
        pg_changed = {**self.batch, "pg_log": self.batch["pg_log"] + 3}
        torch.testing.assert_close(model(pg_changed) - expected, torch.full((2,), 3.0))

    def test_all_head_last_and_lora_adapt_scopes_and_outer_backward(self):
        model = self.live_head(self.model(lora=True))
        self.set_bias(model)
        before = {n: p.detach().clone() for n, p in model.named_parameters()}
        for scope in ("all", "head", "last", "lora"):
            with self.subTest(scope=scope):
                fast = adapt(
                    model, self.batch, lr=0.01, steps=2, scope=scope, training=False
                )
                selected = _adaptation_names(
                    model, dict(model.named_parameters()), scope
                )
                self.assertTrue(
                    any(not torch.equal(fast[n], before[n]) for n in selected)
                )
                self.assertTrue(
                    all(torch.equal(p, before[n]) for n, p in model.named_parameters())
                )
                self.assertTrue(torch.isfinite(forward(model, fast, self.batch)).all())
                for name in fast.keys() - set(selected):
                    self.assertTrue(torch.equal(fast[name], before[name]), name)
                if scope == "all":
                    self.assertTrue(
                        any(
                            not torch.equal(fast[n], before[n])
                            for n in selected
                            if n.endswith("relation_bias")
                        )
                    )
                model.zero_grad(set_to_none=True)
                meta_loss(
                    model, self.batch, self.batch, 0.001, 1, scope=scope
                ).backward()
                self.assertTrue(
                    all(
                        p.grad is not None and torch.isfinite(p.grad).all()
                        for p in model.parameters()
                    )
                )
        lora_extra = sum(
            p.numel()
            for n, p in model.named_parameters()
            if n.endswith(("lora_A", "lora_B"))
        )
        self.assertEqual(lora_extra, 10240)
        self.assertEqual(sum(p.numel() for p in model.parameters()), 304401)

    def test_checkpoint_roundtrip_strict_type_hash_config_and_state_rejection(self):
        for lora in (False, True):
            with self.subTest(lora=lora), tempfile.TemporaryDirectory() as directory:
                model = self.live_head(self.model(lora=lora))
                self.set_bias(model)
                path = Path(directory) / "relation.pt"
                metadata = {
                    "manifest_hashes": self.capacity["cache_hashes"],
                    "purpose": "smoke-not-research",
                }
                save_checkpoint(path, model, None, 0, None, 0, metadata)
                loaded, original = load_checkpoint(path)
                self.assertEqual(loaded.model_type, MODEL_TYPE)
                self.assertEqual(original["model_config_sha256"], digest(model.config))
                self.assertTrue(torch.equal(loaded(self.batch), model(self.batch)))
                for change in (
                    "type",
                    "grouped_type",
                    "config_hash",
                    "flag",
                    "noncanonical",
                    "state",
                    "cache",
                ):
                    state = copy.deepcopy(original)
                    if change == "type":
                        state["model_type"] = "price_lite_pg_residual"
                    elif change == "grouped_type":
                        state["model_type"] = "price_lite_pg_group_pool_residual"
                    elif change == "config_hash":
                        state["model_config_sha256"] = "0" * 64
                    elif change == "flag":
                        state["model_config"]["relation_attention"] = False
                        state["model_config_sha256"] = digest(state["model_config"])
                    elif change == "noncanonical":
                        del state["model_config"]["d_model"]
                        state["model_config_sha256"] = digest(state["model_config"])
                    elif change == "state":
                        del state["model"][
                            "scale_encoder.blocks.0.attention.heads.0.relation_bias"
                        ]
                    else:
                        state["metadata"]["manifest_hashes"][TRAIN_DATABASES[0]] = (
                            "0" * 64
                        )
                    torch.save(state, path)
                    with (
                        self.subTest(change=change),
                        self.assertRaises((ValueError, RuntimeError)),
                    ):
                        load_checkpoint(path)

    def test_checkpoint_numeric_aliases_rejected_and_adapted_weights_reload(self):
        for lora in (False, True):
            with self.subTest(lora=lora), tempfile.TemporaryDirectory() as directory:
                model = self.live_head(self.model(lora=lora))
                fast = adapt(model, self.batch, steps=2, scope="all")
                model.load_state_dict(fast, strict=True)
                path = Path(directory) / "adapted.pt"
                save_checkpoint(
                    path, model, None, 2, None, 0,
                    {"manifest_hashes": self.capacity["cache_hashes"]},
                )
                restored, original = load_checkpoint(path)
                self.assertEqual(digest(restored.config), original["model_config_sha256"])
                self.assertTrue(torch.equal(restored(self.batch), model(self.batch)))
                for field, alias in (("d_model", 64.0), ("dropout", 0), ("dropout", False)):
                    invalid = copy.deepcopy(original)
                    invalid["model_config"][field] = alias
                    invalid["model_config_sha256"] = digest(invalid["model_config"])
                    torch.save(invalid, path)
                    with (
                        self.subTest(lora=lora, field=field, alias=alias),
                        self.assertRaisesRegex(ValueError, "canonical"),
                    ):
                        load_checkpoint(path)

    def test_save_rejects_mutated_identity_before_replacing_existing_checkpoint(self):
        changes = (
            "relation_config", "pg_config", "dimension", "lora_config", "type",
            "legacy_type", "relation_runtime", "pg_runtime", "capacity", "counts",
            "state_shape", "state_key", "numeric_alias",
        )
        for lora in (False, True):
            with self.subTest(lora=lora), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "existing.pt"
                metadata = {"manifest_hashes": self.capacity["cache_hashes"]}
                save_checkpoint(path, self.model(lora=lora), None, 0, None, 0, metadata)
                original_bytes = path.read_bytes()
                for change in changes:
                    model = self.model(lora=lora)
                    if change == "relation_config":
                        model.config["relation_attention"] = False
                    elif change == "pg_config":
                        model.config["pg_head_input"] = False
                    elif change == "dimension":
                        model.config["d_model"] = 32
                    elif change == "lora_config":
                        model.config["lora"] = not lora
                    elif change == "type":
                        model.model_type = "price_lite_pg_residual"
                    elif change == "legacy_type":
                        model.model_type = "grouped_set_residual"
                    elif change == "relation_runtime":
                        model.relation_attention = False
                    elif change == "pg_runtime":
                        model.pg_head_input = False
                    elif change == "capacity":
                        model.capacity["max_counts"][0] += 1
                    elif change == "counts":
                        model.counts = (1, 2, 3, 4)
                    elif change == "state_shape":
                        model.head[-1] = nn.Linear(64, 2)
                    elif change == "state_key":
                        del model.head[-1].bias
                    else:
                        model.config["dropout"] = False
                    with (
                        self.subTest(lora=lora, change=change),
                        self.assertRaisesRegex(ValueError, "canonical|identity"),
                    ):
                        save_checkpoint(path, model, None, 0, None, 0, metadata)
                    self.assertEqual(path.read_bytes(), original_bytes)
                    self.assertFalse(path.with_suffix(".pt.tmp").exists())

    def test_container_aliases_rejected_before_checkpoint_or_directory_creation(self):
        for lora in (False, True):
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                path = root / "existing.pt"
                metadata = {"manifest_hashes": self.capacity["cache_hashes"]}
                save_checkpoint(path, self.model(lora=lora), None, 0, None, 0, metadata)
                original_bytes = path.read_bytes()
                for location in ("config", "capacity"):
                    for field in ("max_counts", "databases", "database_max_counts"):
                        model = self.model(lora=lora)
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
                                self.subTest(lora=lora, location=location, field=field, target=target.name),
                                self.assertRaisesRegex(ValueError, "canonical|identity"),
                            ):
                                save_checkpoint(target, model, None, 0, None, 0, metadata)
                            self.assertEqual(path.read_bytes(), original_bytes)
                            self.assertFalse(target.with_suffix(".pt.tmp").exists())
                            self.assertFalse(fresh.parent.exists())

    def test_model_type_dispatch_rejects_baseline_and_unknown_feature_flags(self):
        candidate = self.model()
        for model_type in (
            "price_lite_pg_residual",
            "grouped_set_residual",
            "price_lite_pg_group_pool_residual",
        ):
            with self.subTest(model_type=model_type), self.assertRaises(ValueError):
                create_model(candidate.config, model_type)
        with self.assertRaises(ValueError):
            create_model({**candidate.config, "group_pooling": "attention"})
        with self.assertRaises(ValueError):
            create_model({**candidate.config, "film": True})

    def test_frozen_training_model_scope_does_not_authorize_candidate(self):
        config = {
            "fixed_stage": "122-A1-PG",
            "method": "erm",
            "meta_objective": "post",
            "model": self.model().config,
        }
        frozen, _ = fixed_model_scope(config, {}, "all")
        self.assertNotEqual(frozen, config["model"])
        self.assertNotIn("relation_attention", frozen)


if __name__ == "__main__":
    unittest.main()
