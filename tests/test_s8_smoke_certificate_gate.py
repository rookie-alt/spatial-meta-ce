"""Synthetic execute certificate gates; no real training, sources or checkpoints."""

import io
import json
import tempfile
import unittest
from contextlib import ExitStack, redirect_stdout
from pathlib import Path
from unittest.mock import Mock, call, patch, sentinel

import numpy as np
import torch

from scripts import run_s8_smoke as runner


class SmokeCertificateGateTests(unittest.TestCase):
    @staticmethod
    def synthetic_design(features):
        columns = features.shape[1]
        normalizer = {
            "mean": np.zeros(columns),
            "std": np.ones(columns),
            "scale": np.ones(columns),
            "zero_mask": np.zeros(columns, dtype=bool),
            "tiny_mask": np.zeros(columns, dtype=bool),
        }
        design = np.column_stack((features.astype(np.float64), np.ones(len(features))))
        return design, normalizer

    @staticmethod
    def synthetic_adam(design, _target, *, steps):
        return {"theta": np.zeros(design.shape[1]), "objective": 1.0, "updates": steps}

    @staticmethod
    def synthetic_ls(design, _target):
        return {"theta": np.zeros(design.shape[1]), "objective": 0.0}

    def check_rejection(self, *, resource):
        error = (
            "Resource-profile fixed-cutoff solver numerical failure"
            if resource else "Small-matrix smoke probe numerical verification failed"
        )
        features = {
            "h": np.zeros((1024, 2), dtype=np.float32),
            "x": np.zeros((1024, 3), dtype=np.float32),
            "label": np.ones(1024, dtype=np.float32),
            "pg_log": np.zeros(1024, dtype=np.float32),
        }
        model = Mock(return_value=torch.zeros(128, dtype=torch.float32))
        store = Mock(spec=["batch", "get", "allow_holdout"])
        store.batch.return_value = sentinel.synthetic_batch
        store.get.side_effect = AssertionError("No downstream V/H semantic loading")
        store.allow_holdout.side_effect = AssertionError("No holdout authorization")
        mocks = {
            "runtime_threadpools": Mock(return_value=[{"user_api": "blas", "num_threads": 4}]),
            "SourceStore": Mock(return_value=store),
            "panel_indices": Mock(return_value=(list(range(1024)), [])),
            "train_f0": Mock(return_value=(model, sentinel.optimizer, {}, "a" * 64, "b" * 64)),
            "checkpoint_write": Mock(return_value="c" * 64),
            "checkpoint_reload": Mock(return_value=(model, {})),
            "capture": Mock(return_value=features),
            "residual_targets": Mock(side_effect=lambda labels, _pg: np.zeros(len(labels))),
            "standardize_fit": Mock(side_effect=self.synthetic_design),
            "solve_adam": Mock(side_effect=self.synthetic_adam),
            "solve_ls": Mock(side_effect=self.synthetic_ls),
            "certify_pair": Mock(side_effect=[{"pass": True}, {"pass": False}]
                                 if resource else [{"pass": False}]),
            "standardize_apply": Mock(side_effect=AssertionError("No endpoint prediction")),
            "metrics_rows": Mock(side_effect=AssertionError("No endpoint metrics")),
        }
        with tempfile.TemporaryDirectory(prefix="s8-certificate-gate-", dir="/private/tmp") as tmp:
            root = Path(tmp).resolve()
            output = root / "runs/s8-g1-smoke-certificate-test"
            sources = {db: {} for db in runner.TRAIN}
            with ExitStack() as stack:
                stack.enter_context(patch.multiple(runner, DATA_ROOT=root, **mocks))
                stack.enter_context(patch.object(runner.np, "__version__", "1.26.1"))
                stack.enter_context(patch.object(runner.torch, "__version__", "2.1.0+cu121"))
                stack.enter_context(patch.object(runner.sys, "version_info", (3, 11, 4)))
                stack.enter_context(patch.object(runner.torch, "set_num_threads"))
                stack.enter_context(patch.object(runner.torch, "set_num_interop_threads"))
                preflight = stack.enter_context(patch.object(
                    runner.source, "preflight", return_value={"input_sha256": {}}))
                safe_binary = stack.enter_context(patch.object(
                    runner.source, "safe_binary",
                    return_value=io.BytesIO(json.dumps({"sources": sources}).encode())))
                safe_hash = stack.enter_context(patch.object(
                    runner.source, "safe_hash", return_value="d" * 64))
                metadata = stack.enter_context(patch.object(
                    runner.source, "source_identity_metadata",
                    side_effect=lambda path, _source, **_kwargs: {
                        "query_hash": [f"synthetic-{path.name}-{i}" for i in range(1024)]
                    }))
                stack.enter_context(patch("spatial_meta_ce.s8_state.signature"))
                seal = stack.enter_context(patch(
                    "spatial_meta_ce.s8_state.seal_nomination",
                    side_effect=AssertionError("No nomination seal")))
                verify = stack.enter_context(patch(
                    "scripts.s8_independent_verify.verify_smoke",
                    side_effect=AssertionError("No completed endpoint verification")))
                stack.enter_context(redirect_stdout(io.StringIO()))
                with self.assertRaisesRegex(ValueError, error):
                    runner.execute(root, output)

                status = json.loads((output / "status.json").read_text(encoding="utf-8"))
                self.assertEqual(status["status"], "failed")
                self.assertEqual(status["purpose"], "s8-engineering-smoke-not-research")
                self.assertIs(status["formal_ready"], False)
                self.assertEqual(status["error"], error)
                self.assertGreaterEqual(status["seconds"], 0)
                events = [json.loads(line) for line in
                          (output / "events.jsonl").read_text(encoding="utf-8").splitlines()]
                self.assertEqual([row["event"] for row in events],
                                 ["source_authentication_complete", "F0_checkpoint_reloaded"])
                for name in ("nomination.json", "probe-smoke.json", "endpoints.jsonl",
                             "metrics.json", "verification.json", "summary.json", "last.pt"):
                    self.assertFalse((output / name).exists(), name)
                saved_probe_files = {p.name for p in output.iterdir()
                                     if p.name.endswith(("-theta.npy", "-normalizer.npz"))}
                # Full-P refusal retains only the already accepted small h artifacts.
                self.assertEqual(saved_probe_files, {
                    "A_h_adam-theta.npy", "B_h_ls-theta.npy", "h-normalizer.npz"
                } if resource else set())
                self.assertTrue(all((output / f"P-{name}.npy").is_file() for name in features))
                expected_rows = [1216, 19456] if resource else [1216]
                for name in ("standardize_fit", "solve_ls", "certify_pair"):
                    self.assertEqual([call.args[0].shape[0] for call in mocks[name].call_args_list],
                                     expected_rows, name)
                mocks["solve_adam"].assert_called_once()
                self.assertEqual(mocks["solve_adam"].call_args.kwargs, {"steps": 2})
                mocks["train_f0"].assert_called_once()
                mocks["checkpoint_write"].assert_called_once()
                mocks["checkpoint_reload"].assert_called_once()
                self.assertEqual([call.args[2] for call in mocks["capture"].call_args_list],
                                 list(runner.TRAIN))
                self.assertEqual(metadata.call_count, len(runner.TRAIN))
                self.assertEqual([call.args[0].name for call in metadata.call_args_list],
                                 list(runner.TRAIN))
                preflight.assert_called_once()
                safe_binary.assert_called_once_with(root / "runs/s6-mean-tail-20261007/preflight.json")
                self.assertEqual(safe_hash.call_args_list, [
                    call(runner.ACCEPTANCE_PROTOCOL),
                    call(runner.ROOT / '实验文档/27-S8G1运行器与真实smoke协议.md'),
                ])
                store.batch.assert_called_once_with(runner.TRAIN[0], list(range(128)))
                store.get.assert_not_called()
                store.allow_holdout.assert_not_called()
                seal.assert_not_called()
                verify.assert_not_called()
                mocks["standardize_apply"].assert_not_called()
                mocks["metrics_rows"].assert_not_called()

    def test_small_panel_certificate_refusal_leaves_failed_without_probe_artifacts(self):
        self.check_rejection(resource=False)

    def test_full_p_certificate_refusal_stops_before_nomination_or_holdout(self):
        self.check_rejection(resource=True)


if __name__ == "__main__":
    unittest.main()
