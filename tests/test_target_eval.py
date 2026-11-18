"""D-stage zero-shot evaluation: guards, bindings and unsigned rejection."""

import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import run_target_eval_d as runner
from spatial_meta_ce import target_eval
from spatial_meta_ce.official_split import OFFICIAL_TRAIN_DATABASES, TEST_DATABASES


class TargetEvalGuardTests(unittest.TestCase):
    def _state(self, train):
        return {
            "model_type": "grouped_set_residual",
            "metadata": {"train_databases": list(train), "validation_databases": []},
            "model_config": {},
        }

    def test_checkpoint_trained_on_a_test_database_is_refused(self):
        state = self._state([*OFFICIAL_TRAIN_DATABASES[:-1], TEST_DATABASES[0]])
        with (
            patch.object(target_eval, "load_checkpoint", return_value=(None, state)),
            self.assertRaisesRegex(ValueError, "test database"),
        ):
            target_eval.evaluate_target_zero_shot(
                "checkpoint.pt", "A1", 42, {"cache_root": "."}, "/tmp/never"
            )

    def test_adapted_or_non_official_checkpoints_are_refused(self):
        state = self._state(OFFICIAL_TRAIN_DATABASES)
        state["metadata"]["adapted_database"] = "stats"
        with (
            patch.object(target_eval, "load_checkpoint", return_value=(None, state)),
            self.assertRaisesRegex(ValueError, "adapted"),
        ):
            target_eval.evaluate_target_zero_shot(
                "checkpoint.pt", "A1", 42, {"cache_root": "."}, "/tmp/never"
            )
        short = self._state(OFFICIAL_TRAIN_DATABASES[:-1])
        with (
            patch.object(target_eval, "load_checkpoint", return_value=(None, short)),
            self.assertRaisesRegex(ValueError, "official 26"),
        ):
            target_eval.evaluate_target_zero_shot(
                "checkpoint.pt", "A1", 42, {"cache_root": "."}, "/tmp/never"
            )

    def test_spec_must_be_k0_on_exactly_four_test_databases(self):
        good = {
            "protocol": "122-official-26",
            "purpose": "target-zero-shot-d",
            "arms": ["A1", "A1-PG"],
            "seeds": [42, 43, 44],
            "shots": [0],
            "databases": list(TEST_DATABASES),
            "batch_size": 128,
        }
        for patch_value in (
            {"shots": [5]},
            {"databases": list(TEST_DATABASES[:3])},
            {"batch_size": 64},
        ):
            with self.subTest(patch=patch_value), self.assertRaises(ValueError):
                runner.verify_spec({**good, **patch_value})

    def test_spec_arms_and_seeds_stay_inside_the_allowlist(self):
        good = {
            "protocol": "122-official-26",
            "purpose": "target-zero-shot-d",
            "arms": ["A1-PG"],
            "seeds": [42, 43, 44],
            "shots": [0],
            "databases": list(TEST_DATABASES),
            "batch_size": 128,
        }
        for patch_value in (
            {"arms": ["A9"]},
            {"arms": []},
            {"arms": ["A1", "A1"]},
            {"seeds": [42, 45]},
            {"seeds": []},
        ):
            with self.subTest(patch=patch_value), self.assertRaisesRegex(
                ValueError, "frozen D plan"
            ):
                runner.verify_spec({**good, **patch_value})

    def test_unsigned_evaluation_authorization_is_rejected(self):
        from scripts.run_price_meta_a import validate_budget

        budget = validate_budget(
            {
                "total_seconds": 60.0,
                "phase_seconds": {
                    "preparation": 20.0,
                    "evaluation": 1.0,
                    "certification": 20.0,
                },
                "rss_bytes": 1024,
                "vram_bytes": 1024,
                "output_bytes": 1024,
                "poll_seconds": 0.5,
                "cpu_core_seconds": 10,
                "gpu_allocated_seconds": 10,
                "run_deadline_seconds": 10.0,
                "eval_deadline_seconds": 1.0,
            }
        )
        with self.assertRaisesRegex(ValueError, "unsigned"):
            runner.validate_authorization(
                {"protocol": "122-official-26", "purpose": "target-zero-shot-d"},
                "a" * 64,
                budget,
                Path("/tmp/never"),
            )