import copy
import tempfile
import unittest
from pathlib import Path

import torch

from spatial_meta_ce.checkpoints import load_checkpoint, save_checkpoint
from spatial_meta_ce.data import batch_samples
from spatial_meta_ce.model import SetEstimator
from scripts.mapping_smoke import state_hash


def synthetic_batch():
    return batch_samples(
        [
            {
                "groups": [
                    torch.rand(n, 40),
                    torch.rand(2 * n, 40),
                    torch.rand(n, 4),
                    torch.rand(n, 43),
                ],
                "true": 10.0,
                "pg": pg,
                "id": str(i),
                "hash": str(i),
            }
            for i, (n, pg) in enumerate([(2, 20.0), (4, 1000.0), (1, 0.0)])
        ]
    )


class FrozenSumBeforeS4(torch.nn.Module):
    # Independent reference to the sum formula before the output-anchor intervention.
    def __init__(self, use_pg=True):
        super().__init__()
        self.use_pg = use_pg
        self.encoders = torch.nn.ModuleList(
            [
                torch.nn.Sequential(
                    torch.nn.Linear(dim, 64),
                    torch.nn.ReLU(),
                    torch.nn.Linear(64, 64),
                    torch.nn.ReLU(),
                )
                for dim in (40, 40, 4, 43)
            ]
        )
        self.head = torch.nn.Sequential(
            torch.nn.Linear(774, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 1),
        )
        torch.nn.init.zeros_(self.head[-1].weight)
        torch.nn.init.zeros_(self.head[-1].bias)

    def forward(self, batch):
        pooled = []
        for encoder, x, mask in zip(self.encoders, batch["groups"], batch["masks"]):
            t = encoder(x.masked_fill(~mask.unsqueeze(-1), 0))
            valid = mask.unsqueeze(-1)
            total = (t * valid).sum(1)
            mean = total / valid.sum(1).clamp_min(1)
            largest = t.masked_fill(~valid, -torch.inf).amax(1)
            largest = torch.where(
                mask.any(1, keepdim=True), largest, torch.zeros_like(largest)
            )
            pooled.extend([mean, largest, total])
        pg = batch["pg_log"] if self.use_pg else torch.zeros_like(batch["pg_log"])
        log_cartesian = (batch["groups"][2][:, :, 0] * batch["masks"][2]).sum(1)
        x = torch.cat(
            [
                *pooled,
                torch.log1p(batch["counts"]),
                pg[:, None],
                log_cartesian[:, None],
            ],
            dim=1,
        )
        return pg + self.head(x).squeeze(-1)


class PGAnchorModelTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(42)

    def test_initialization_hash_and_capacity_unchanged(self):
        a = SetEstimator(arch="sum", pg_output_scale=1.0)
        rng_a = torch.get_rng_state().clone()
        torch.manual_seed(42)
        b = SetEstimator(arch="sum", pg_output_scale=0.0)
        self.assertEqual(state_hash(a), state_hash(b))
        torch.testing.assert_close(rng_a, torch.get_rng_state(), rtol=0, atol=0)
        torch.manual_seed(42)
        legacy = FrozenSumBeforeS4()
        self.assertEqual(state_hash(a), state_hash(legacy))
        torch.testing.assert_close(rng_a, torch.get_rng_state(), rtol=0, atol=0)
        if torch.__version__ == "2.1.0+cu121":
            self.assertEqual(
                state_hash(a),
                "41e594b03ef19c56c3f236237da850929f2ec5c919702b44402dd9b2864d284a",
            )
        self.assertEqual(sum(p.numel() for p in a.parameters()), 132545)
        self.assertNotIn("pg_output_scale", a.config)
        self.assertEqual(b.config["pg_output_scale"], 0.0)
        batch = synthetic_batch()
        torch.testing.assert_close(a(batch), batch["pg_log"], rtol=0, atol=0)
        torch.testing.assert_close(
            b(batch), torch.zeros_like(batch["pg_log"]), rtol=0, atol=0
        )

    def test_same_inputs_live_head_and_forward_difference(self):
        batch = synthetic_batch()
        a = SetEstimator(arch="sum", pg_output_scale=1.0)
        torch.nn.init.normal_(a.head[-1].weight, std=0.01)
        b = SetEstimator(arch="sum", pg_output_scale=0.0)
        b.load_state_dict(a.state_dict())
        inputs = []
        ha = a.head[0].register_forward_pre_hook(
            lambda m, x: inputs.append(x[0].detach().clone())
        )
        hb = b.head[0].register_forward_pre_hook(
            lambda m, x: inputs.append(x[0].detach().clone())
        )
        try:
            pa, pb = a(batch), b(batch)
        finally:
            ha.remove()
            hb.remove()
        torch.testing.assert_close(inputs[0], inputs[1], rtol=0, atol=0)
        torch.testing.assert_close(inputs[0][:, -2], batch["pg_log"], rtol=0, atol=0)
        torch.testing.assert_close(pa - pb, batch["pg_log"], rtol=1e-6, atol=1e-6)

    def test_legacy_forward_and_gradient_identical(self):
        batch = synthetic_batch()
        legacy = FrozenSumBeforeS4()
        torch.nn.init.normal_(legacy.head[-1].weight, std=0.01)
        explicit = SetEstimator(arch="sum", pg_output_scale=1.0)
        explicit.load_state_dict(legacy.state_dict())
        torch.testing.assert_close(legacy(batch), explicit(batch), rtol=0, atol=0)
        for m in (legacy, explicit):
            m.zero_grad()
            torch.nn.functional.smooth_l1_loss(m(batch), batch["label"]).backward()
        for (na, a), (nb, b) in zip(
            legacy.named_parameters(), explicit.named_parameters()
        ):
            self.assertEqual(na, nb)
            torch.testing.assert_close(a.grad, b.grad, rtol=0, atol=0)

    def test_zero_anchor_roundtrip_and_legacy_checkpoint(self):
        batch = synthetic_batch()
        for scale in (0.0, 1.0):
            model = SetEstimator(arch="sum", pg_output_scale=scale)
            with tempfile.TemporaryDirectory() as d:
                path = Path(d) / "last.pt"
                save_checkpoint(path, model, None, 0, 1.0, 0, {"purpose": "synthetic"})
                loaded, state = load_checkpoint(path)
                self.assertEqual(loaded.pg_output_scale, scale)
                self.assertEqual(loaded.config, model.config)
                torch.testing.assert_close(model(batch), loaded(batch), rtol=0, atol=0)
                if scale == 0.0:
                    changed = copy.deepcopy(model.config)
                    changed["pg_output_scale"] = 1.0
                    wrong = SetEstimator(**changed)
                    wrong.load_state_dict(model.state_dict())
                    self.assertFalse(torch.equal(wrong(batch), model(batch)))
                else:
                    self.assertNotIn("pg_output_scale", state["model_config"])

    def test_invalid_scale_and_no_pg_compatibility(self):
        for invalid in (-1, 0.5, 2, float("nan"), float("inf"), True, False, "0", None):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                SetEstimator(pg_output_scale=invalid)
        batch = synthetic_batch()
        for scale in (0.0, 1.0):
            model = SetEstimator(arch="sum", use_pg=False, pg_output_scale=scale)
            torch.testing.assert_close(
                model(batch), torch.zeros_like(batch["pg_log"]), rtol=0, atol=0
            )
            torch.nn.init.normal_(model.head[-1].weight, std=0.01)
            legacy = FrozenSumBeforeS4(use_pg=False)
            legacy.load_state_dict(model.state_dict())
            torch.testing.assert_close(model(batch), legacy(batch), rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
