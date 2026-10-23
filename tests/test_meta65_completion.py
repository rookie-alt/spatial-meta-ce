import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import complete_meta65_endpoint as completion


class CompletionTests(unittest.TestCase):
    def test_exact_checkpoint_plan_and_budget_contract(self):
        config = copy.deepcopy(completion.FROZEN)
        completion.validate_contract(config)
        for key, value in [
            ("checkpoint_sha256", "0" * 64),
            ("endpoint_plan_sha256", "0" * 64),
            ("timeout_seconds", 7200),
            ("mode", "training"),
            ("unknown", True),
        ]:
            bad = {**config, key: value}
            with self.subTest(key=key), self.assertRaises(ValueError):
                completion.validate_contract(bad)

    def test_combined_matrix_has_explicit_origin_and_exact_counts(self):
        rows = {
            arm: [{"arm": arm, "prediction_file": f"{arm}/predictions/file.npz"}]
            for arm in completion.original.ARMS
        }
        with self.assertRaises(ValueError):
            completion.combine_rows(rows)
        rows = {arm: [dict(row) for _ in range(75)] for arm, [row] in rows.items()}
        result = completion.combine_rows(rows)
        self.assertEqual(len(result), 225)
        self.assertEqual({r["source_run"] for r in result[:150]}, {"retained"})
        self.assertEqual({r["source_run"] for r in result[150:]}, {"completion"})
        self.assertNotIn("source_run", rows[completion.original.ARMS[0]][0])

    def test_retained_hash_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            (root / "input.json").write_text("{}")
            with self.assertRaisesRegex(ValueError, "Retained input"):
                completion.check_retained(root, {"input.json": "0" * 64})

    def test_failure_removes_normal_candidate_path_without_deleting_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            (out / "decision.json").write_text('{"status":"promising_single_seed"}')
            completion.record_failure(out, RuntimeError("postcheck failed"))
            self.assertFalse((out / "decision.json").exists())
            self.assertTrue((out / "decision.invalid.json").exists())
            self.assertEqual(
                completion.original.read_json(out / "status.json")["status"], "failed"
            )

    def test_main_dry_run_never_evaluates_or_supervises(self):
        with (
            patch.object(
                completion,
                "load_contract",
                return_value=copy.deepcopy(completion.FROZEN),
            ),
            patch.object(completion, "supervise") as supervised,
            patch.object(completion, "work") as work,
        ):
            completion.main(["--out", "runs/not-executed"])
            supervised.assert_not_called()
            work.assert_not_called()

    def test_old_deployment_rejected_before_supervisor(self):
        with (
            patch.object(
                completion,
                "load_contract",
                return_value=copy.deepcopy(completion.FROZEN),
            ),
            patch.object(
                completion, "preflight_output", side_effect=ValueError("isolated")
            ),
            patch.object(completion, "supervise") as supervised,
        ):
            with self.assertRaisesRegex(ValueError, "isolated"):
                completion.main(["--execute"])
            supervised.assert_not_called()
        with self.assertRaisesRegex(ValueError, "isolated"):
            completion.preflight_output(
                completion.FROZEN,
                Path(completion.FROZEN["retained_run"]) / "new",
                root=Path(completion.FROZEN["retained_root"]),
            )

    def test_changed_data_path_rejected_before_creation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            (root / "configs").mkdir()
            source = completion.ROOT / "configs"
            (root / "configs/meta65_screen.json").write_bytes(
                (source / "meta65_screen.json").read_bytes()
            )
            data = completion.original.read_json(source / "meta65_server_data.json")
            data["cache_root"] = str(root / "another-cache")
            (root / "configs/meta65_server_data.json").write_text(json.dumps(data))
            out = root / "runs/new"
            with self.assertRaisesRegex(ValueError, "frozen data/model config"):
                completion.preflight_output(completion.FROZEN, out, root)
            self.assertFalse(out.exists())

    def test_cache_change_during_verifier_blocks_normal_decision(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory).resolve()
            (out / "decision.provisional.json").write_text("{}")
            with patch(
                "scripts.precision_screen_evidence.source_evidence",
                return_value={"changed": True},
            ):
                with self.assertRaisesRegex(
                    ValueError, "independent verification"
                ) as error:
                    completion.final_source_check(
                        {}, {}, {"original": True}, out, lambda: None
                    )
                completion.record_failure(out, error.exception)
            self.assertFalse((out / "decision.json").exists())
            self.assertTrue((out / "decision.provisional.json").exists())
            self.assertEqual(
                completion.original.read_json(out / "status.json")["status"], "failed"
            )


if __name__ == "__main__":
    unittest.main()
