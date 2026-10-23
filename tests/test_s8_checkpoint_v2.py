"""Local synthetic checkpoint states; no optimizer steps or source training."""

import tempfile
import unittest
from pathlib import Path

import torch
from test_s8_state_v2 import v2_bindings

from scripts import run_s8_smoke as runner
from spatial_meta_ce.model import SetEstimator


class CheckpointV2Tests(unittest.TestCase):
    def test_v2_roundtrip_binds_policy_version_and_acceptance_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = SetEstimator(**runner.MODEL).eval()
            optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
            # Explicit fake endpoint states test serialization validation only.
            # They are not evidence that any real F0 update was performed.
            for parameter in model.parameters():
                optimizer.state[parameter] = {
                    "step": torch.tensor(2.0),
                    "exp_avg": torch.zeros_like(parameter),
                    "exp_avg_sq": torch.zeros_like(parameter),
                }
            bindings = v2_bindings()
            path = Path(tmp).resolve() / "last.pt"
            sha = runner.checkpoint_write(
                path,
                model,
                optimizer,
                bindings,
                {"optimizer_steps": 2, "backward_calls": 8},
                "d" * 64,
            )
            restored, state = runner.checkpoint_reload(path, sha, bindings)
            self.assertEqual(runner.tensor_hash(restored), runner.tensor_hash(model))
            self.assertEqual(state["bindings"], bindings)
            for key, value in [
                ("acceptance_protocol_sha256", "e" * 64),
                ("numerical_acceptance_policy", "unknown"),
                ("schema", 1),
            ]:
                with self.subTest(key=key), self.assertRaises(ValueError):
                    runner.checkpoint_reload(path, sha, {**bindings, key: value})
            with self.assertRaises(FileExistsError):
                runner.checkpoint_write(
                    path,
                    model,
                    optimizer,
                    bindings,
                    {"optimizer_steps": 2, "backward_calls": 8},
                    "d" * 64,
                )


if __name__ == "__main__":
    unittest.main()
