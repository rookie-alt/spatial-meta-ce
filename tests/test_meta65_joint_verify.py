import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts import meta65_joint_verify as verifier

DATABASES = ("chembl", "legalacts", "telstra")
CHECKPOINTS = ("A", "B", "C")
OPERATORS = ("none", "head3", "last3")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def expected_metric(count, q50=1.0, q95=1.0, status="ok"):
    return {"count": count, "q50": q50, "q95": q95, "status": status}


def metrics_for_all_four_plus(count=32):
    joint = {}
    cards = ("<100", "100-1e3", "1e3-1e4", "1e4-1e5", "1e5-1e6", ">=1e6")
    tables = ("1", "2", "3", "4", "5+")
    for card in cards:
        for table in tables:
            nonempty = card == "<100" and table == "4"
            joint[f"{card}|{table}"] = expected_metric(
                count if nonempty else 0,
                status="ok" if nonempty and count >= 30 else "empty",
                q50=1.0 if nonempty else None,
                q95=1.0 if nonempty else None,
            )
    return {
        "all": expected_metric(count),
        "4+": expected_metric(count),
        "joint": joint,
    }


class Meta65VerifierTests(unittest.TestCase):
    def test_linear_quantile_and_aggregate_are_standalone(self):
        self.assertEqual(verifier.quantile([9.0, 1.0, 5.0, 3.0], 0.5), 4.0)
        self.assertAlmostEqual(verifier.quantile([1.0, 2.0, 4.0, 8.0], 0.95), 7.4)
        self.assertIsNone(verifier.quantile([], 0.5))
        self.assertEqual(verifier.aggregate([]), {
            "count": 0, "q50": None, "q95": None, "status": "empty"
        })
        self.assertEqual(verifier.aggregate([1.0, 2.0]), {
            "count": 2, "q50": 1.5, "q95": 1.95, "status": "below_min_n"
        })

    def test_joint_masks_cover_complete_four_bucket_projection(self):
        true = np.array([1, 99, 100, 999, 1000, 999999, 1000000], dtype=np.float64)
        tables = np.array([1, 3, 4, 5, 2, 4, 7], dtype=np.int16)
        masks = verifier.joint_masks(true, tables)
        self.assertEqual(set(masks), {
            f"{card}|{table}"
            for card in ("<100", "100-1e3", "1e3-1e4", "1e4-1e5", "1e5-1e6", ">=1e6")
            for table in ("1", "2", "3", "4", "5+")
        })
        self.assertEqual(sum(int(mask.sum()) for mask in masks.values()), len(true))
        self.assertEqual(masks["<100|1"].tolist(), [True, False, False, False, False, False, False])
        self.assertEqual(masks[">=1e6|5+"].tolist(), [False, False, False, False, False, False, True])

    def test_cpu_preflight_verifies_micro_matrix_metrics_and_does_not_write_summary(self):
        with tempfile.TemporaryDirectory() as directory:
            root = self._write_cpu_output(Path(directory))
            result = verifier.verify(root)
            self.assertEqual(result["status"], "preflight_verified")
            self.assertFalse(result["formal_result"])
            self.assertEqual(result["cells_verified"], 9)
            self.assertTrue(result["all_metrics_verified"])
            self.assertIsNone(result["summary"])
            self.assertFalse((root / "summary.json").exists())

    def test_cpu_rejects_input_hash_change_and_prediction_float32(self):
        with tempfile.TemporaryDirectory() as directory:
            root = self._write_cpu_output(Path(directory))
            source = root / "source3.json"
            source.write_text("changed", encoding="utf-8")
            with self.assertRaises(ValueError):
                verifier.verify(root)

        with tempfile.TemporaryDirectory() as directory:
            root = self._write_cpu_output(Path(directory))
            prediction = root / "predictions" / "A-head3-chembl.npz"
            np.savez(prediction, log_pred=np.zeros(32, dtype=np.float32))
            self._refresh_cell_hash(root, prediction)
            with self.assertRaises(ValueError):
                verifier.verify(root)

    def test_cpu_rejects_escape_duplicate_and_nonfinite_values(self):
        with tempfile.TemporaryDirectory() as directory:
            root = self._write_cpu_output(Path(directory))
            cells_path = root / "microcells.json"
            payload = json.loads(cells_path.read_text(encoding="utf-8"))
            payload["cells"][0]["prediction_file"] = "../escape.npz"
            cells_path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaises(ValueError):
                verifier.verify(root)

        with tempfile.TemporaryDirectory() as directory:
            root = self._write_cpu_output(Path(directory))
            cells_path = root / "microcells.json"
            payload = json.loads(cells_path.read_text(encoding="utf-8"))
            payload["cells"].append(dict(payload["cells"][0]))
            cells_path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaises(ValueError):
                verifier.verify(root)

        with tempfile.TemporaryDirectory() as directory:
            root = self._write_cpu_output(Path(directory))
            prediction = root / "predictions" / "A-head3-chembl.npz"
            np.savez(prediction, log_pred=np.full(32, np.nan, dtype=np.float64))
            self._refresh_cell_hash(root, prediction)
            with self.assertRaises(ValueError):
                verifier.verify(root)

    def test_rejects_incomplete_code_binding_and_inconsistent_initialization(self):
        with tempfile.TemporaryDirectory() as directory:
            root = self._write_cpu_output(Path(directory))
            binding_path = root / "input-binding.json"
            binding = json.loads(binding_path.read_text())
            binding["code_sha256"] = {}
            binding_path.write_text(json.dumps(binding))
            with self.assertRaises(ValueError):
                verifier.verify(root)

        with tempfile.TemporaryDirectory() as directory:
            root = self._write_cpu_output(Path(directory))
            cells_path = root / "microcells.json"
            payload = json.loads(cells_path.read_text())
            payload["cells"][1]["state_before"] = "different"
            payload["cells"][1]["state_after"] = "different"
            cells_path.write_text(json.dumps(payload))
            with self.assertRaises(ValueError):
                verifier.verify(root)

    def test_formal_mode_requires_exact_135_cell_matrix_and_denominators(self):
        with tempfile.TemporaryDirectory() as directory:
            root = self._write_cpu_output(Path(directory))
            cells = json.loads((root / "microcells.json").read_text(encoding="utf-8"))
            cells["mode"] = "formal"
            (root / "cells.json").write_text(json.dumps(cells), encoding="utf-8")
            (root / "microcells.json").unlink()
            binding = json.loads((root / "input-binding.json").read_text(encoding="utf-8"))
            binding["mode"] = "formal"
            (root / "input-binding.json").write_text(json.dumps(binding), encoding="utf-8")
            with self.assertRaises(ValueError):
                verifier.verify(root)

    def _write_cpu_output(self, root):
        (root / "identities").mkdir(parents=True)
        (root / "predictions").mkdir()
        source = root / "source3.json"
        source.write_text("source3", encoding="utf-8")
        cells = []
        for database in DATABASES:
            identity_file = root / "identities" / f"{database}-run0.npz"
            hashes = np.asarray([f"{database}-{i:03d}" for i in range(32)], dtype="S64")
            ids = np.asarray([f"{database}/q/{i}" for i in range(32)], dtype="S128")
            true = np.arange(1, 33, dtype=np.float64)
            tables = np.full(32, 4, dtype=np.int16)
            np.savez(identity_file, query_hash=hashes, query_id=ids, true=true, n_tables=tables)
            for checkpoint, operator in (("A", "head3"), ("B", "head3"), ("C", "last3")):
                prediction_file = root / "predictions" / f"{checkpoint}-{operator}-{database}.npz"
                np.savez(prediction_file, log_pred=np.log(true))
                cells.append({
                    "checkpoint": checkpoint,
                    "operator": operator,
                    "database": database,
                    "run": 0,
                    "k": 20,
                    "k20": 20,
                    "identity_file": str(identity_file.relative_to(root)),
                    "identity_sha256": sha256(identity_file),
                    "prediction_file": str(prediction_file.relative_to(root)),
                    "prediction_sha256": sha256(prediction_file),
                    "count": 32,
                    "query_positions": list(range(32)),
                    "metrics": metrics_for_all_four_plus(),
                    "state_before": f"state-{checkpoint}",
                    "state_after": f"state-{checkpoint}",
                    "selected_parameter_names": [operator],
                    "selected_parameter_count": {"head3": 107521, "last3": 65}[operator],
                    "changed_parameter_names": [operator],
                    "changed_parameter_count": {"head3": 107521, "last3": 65}[operator],
                    "steps": 3,
                    "adaptation_seconds": 0.1,
                    "prediction_seconds": 0.2,
                })
        code = {name: "0" * 64 for name in (
            "scripts/meta65_joint_runner.py", "scripts/meta65_joint_supervisor.py",
            "scripts/meta65_joint_verify.py", "spatial_meta_ce/data.py",
            "spatial_meta_ce/manifests.py", "spatial_meta_ce/mappings.py",
            "spatial_meta_ce/model.py", "spatial_meta_ce/meta.py",
            "spatial_meta_ce/checkpoints.py")}
        binding = {"schema": 1, "mode": "cpu_preflight", "input_sha256": {str(source): sha256(source)},
                   "code_sha256": code, "estimated_formal_output_bytes": 0}
        (root / "input-binding.json").write_text(json.dumps(binding), encoding="utf-8")
        (root / "microcells.json").write_text(
            json.dumps({"schema": 1, "mode": "cpu_preflight", "cells": cells}),
            encoding="utf-8",
        )
        return root

    def _refresh_cell_hash(self, root, prediction):
        cells_path = root / "microcells.json"
        payload = json.loads(cells_path.read_text(encoding="utf-8"))
        for cell in payload["cells"]:
            if cell["prediction_file"] == str(prediction.relative_to(root)):
                cell["prediction_sha256"] = sha256(prediction)
        cells_path.write_text(json.dumps(payload), encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
