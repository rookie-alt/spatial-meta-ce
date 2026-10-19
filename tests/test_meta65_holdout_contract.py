import copy
import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts import meta65_holdout_contract as contract


class HoldoutContractTests(unittest.TestCase):
    def data(self):
        return {
            "data_root": "/tmp/source-data",
            "cache_root": "/tmp/source-cache",
            "splits": {
                "train": list(contract.TRAIN),
                "validation": list(contract.VALIDATION),
                "holdout": list(contract.HOLDOUT),
                "test": list(contract.TARGETS),
            },
        }

    def test_split_closure_and_training_projection(self):
        data = contract.validate_data(self.data(), root=Path("/"))
        self.assertEqual(len(data["splits"]["train"]), 21)
        self.assertEqual(len(data["splits"]["validation"]), 3)
        self.assertEqual(data["splits"]["holdout"], ["hepatitis", "sakila"])
        training = contract.training_data(data)
        self.assertEqual(set(training["splits"]), {"train", "validation", "test"})
        self.assertFalse(set(training["splits"]["train"]) & set(contract.HOLDOUT))

    def test_split_leakage_and_unknown_key_are_rejected(self):
        bad = self.data()
        bad["splits"]["train"] = [*bad["splits"]["train"], "sakila"]
        with self.assertRaises(ValueError):
            contract.validate_data(bad, root=Path("/"))
        extra = self.data()
        extra["extra"] = True
        with self.assertRaises(ValueError):
            contract.validate_data(extra, root=Path("/"))

    def test_frozen_matrix_counts(self):
        self.assertEqual(len(contract.matrix("formal")), 50)
        self.assertEqual(len(contract.matrix("cpu_preflight")), 10)
        self.assertEqual({cell[0] for cell in contract.matrix("formal")}, {"R_head3"})
        self.assertEqual(
            {cell[1] for cell in contract.matrix("formal")}, {"none", "head3"}
        )

    def test_deterministic_plan_rejects_tampering(self):
        indices = list(range(130))
        hashes = [f"h{index:03d}" for index in indices]
        plan = contract.build_run_plan("sakila", indices, hashes, seed=2718, run=0)
        self.assertEqual(len(plan["support_max"]), 100)
        self.assertEqual(len(plan["query"]), 30)
        self.assertFalse(set(plan["support_max"]) & set(plan["query"]))
        self.assertEqual(
            contract.certify_run_plan(plan, indices, hashes, "sakila", 0),
            {"query": plan["query"], "support": plan["support_max"][:20]},
        )
        bad = copy.deepcopy(plan)
        bad["query_digest"] = "0" * 64
        with self.assertRaises(ValueError):
            contract.certify_run_plan(bad, indices, hashes, "sakila", 0)

    def test_metric_buckets_are_independent_and_standard(self):
        true = np.array([50.0, 200.0, 50.0, 200.0, 500.0])
        tables = np.array([4, 4, 2, 5, 4], dtype=np.int16)
        pred = np.log(true) + np.array([0.1, -0.2, 0.3, 0.4, -0.1])
        metrics = contract.metric_buckets(pred, true, tables)
        self.assertEqual(metrics["4+"]["count"], 4)
        self.assertEqual(metrics["low4"]["count"], 1)
        self.assertEqual(metrics["all"]["status"], "below_min_n")

    def test_safe_output_requires_isolated_runs_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            output = contract.safe_output(root / "runs" / "preflight", [], root=root)
            self.assertEqual(output, root / "runs" / "preflight")
            with self.assertRaises(ValueError):
                contract.safe_output(root / "outside", [], root=root)


if __name__ == "__main__":
    unittest.main()
