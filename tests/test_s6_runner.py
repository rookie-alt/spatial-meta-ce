import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import run_s6_tail as s


class S6RunnerTests(unittest.TestCase):
    def test_plan_only_changes_reduction(self):
        p = s.load_plan()
        a, b = (p["arms"][x] for x in s.ARMS)
        self.assertEqual(a["tail_risk_mode"], "mean")
        self.assertEqual(b["tail_risk_mode"], "mean_tail10")
        self.assertEqual(
            {k: v for k, v in a.items() if k != "tail_risk_mode"},
            {k: v for k, v in b.items() if k != "tail_risk_mode"},
        )
        self.assertNotIn("complex_query_weight", a)
        self.assertEqual(a["model"]["pg_output_scale"], 1)
        self.assertEqual(a["steps"], 4000)

    def test_matrix_and_protocol_tampering(self):
        raw = json.loads(s.CONFIG.read_text())
        with tempfile.TemporaryDirectory() as tmp:
            for key in ("model", "arms", "training", "timeout_seconds"):
                p = Path(tmp) / "bad.json"
                p.write_text(json.dumps({**raw, key: {}}))
                with self.assertRaises(ValueError):
                    s.load_plan(p)
        with patch.object(s, "file_hash", return_value="0" * 64):
            with self.assertRaises(ValueError):
                s.load_plan()

    def test_source_mismatch_blocks_before_forward(self):
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.object(s.anchor, "gpu_gate"),
            patch.object(s, "source_evidence", return_value={}),
            patch.object(s.anchor, "run_one") as run,
        ):
            with self.assertRaises(ValueError):
                s.execute_arms(
                    s.load_plan()["arms"],
                    {},
                    Path(tmp),
                    {"gpu_uuid": "GPU-one", "sources": {"old": 1}},
                    {},
                    "3",
                )
            run.assert_not_called()
            self.assertEqual(
                json.loads((Path(tmp) / "status.json").read_text())["status"], "failed"
            )

    def test_incomplete_outputs_and_expired_deadline(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            s.write_json(root / "output-manifest.json", {})
            with self.assertRaises(ValueError):
                s.verify_outputs(root)
            s.status(root, "planned", dict.fromkeys(s.ARMS, "not_started"))
            with patch.object(s.subprocess, "Popen") as proc:
                with self.assertRaises(TimeoutError):
                    s.supervise(["unused"], root, s.time.monotonic() - 1)
                proc.assert_not_called()
            self.assertNotEqual(
                json.loads((root / "status.json").read_text())["status"], "completed"
            )

    def test_worker_owner_required_before_preflight(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            s.write_json(root / "supervision.json", {"pid": 0, "token": "fixture"})
            with (
                patch.object(s, "checked_path", side_effect=lambda r, p: p),
                patch.object(s, "preflight") as p,
            ):
                with self.assertRaises(ValueError):
                    s.worker(root, "3")
                p.assert_not_called()

    def test_loss_stores_only_training_caches(self):
        import numpy as np
        from unittest.mock import Mock

        p = s.load_plan()
        pool = Mock()
        store = Mock(
            rows=[{"query_hash": "one"}, {"query_hash": "two"}],
            cards=np.asarray([[3.0, 2.0], [0.0, 1.0]]),
            offsets=np.asarray([[0, 0, 0], [0, 0, 3], [0, 0, 7]]),
        )
        store.unique_indices.return_value = np.asarray([0, 1])
        pool.get.return_value = store
        with patch.object(s, "StorePool", return_value=pool):
            value = s.loss_stores({"splits": p["splits"], "cache_root": "unused"})
        self.assertEqual(set(value), set(p["splits"]["train"]))
        self.assertEqual(
            [x.args for x in pool.get.call_args_list],
            [("pretrain", db) for db in p["splits"]["train"]],
        )
        for v in value.values():
            np.testing.assert_array_equal(v["tables"], [3, 4])
            np.testing.assert_array_equal(
                v["labels"], np.log([3, 1]).astype(np.float32).astype(np.float64)
            )


if __name__ == "__main__":
    unittest.main()
