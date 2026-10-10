import copy
import hashlib
import json
import tempfile
import unittest
from collections import OrderedDict
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from scripts import meta65_joint_runner as runner
from spatial_meta_ce.data import batch_samples
from spatial_meta_ce.model import SetEstimator


def identity(n=120):
    return {"query_hash": np.array([f"h{i:03}" for i in range(n)]),
            "query_id": np.array([f"telstra/pretrain/{i+1}" for i in range(n)]),
            "true": np.arange(n, dtype=np.float64) + 16777217,
            "n_tables": np.full(n, 4, dtype=np.int16)}


def plan_for(data):
    indices = list(range(len(data["true"])))
    support = indices[:100]
    query = indices[100:]
    values = {"database": "telstra", "run": 0, "support_max": support,
              "supports": {str(k): support[:k] for k in (5, 10, 20, 50, 100)},
              "query": query,
              "query_digest": runner.digest([(i, str(data["query_hash"][i])) for i in query]),
              "support_digests": {str(k): runner.digest([(i, str(data["query_hash"][i]))
                                                       for i in support[:k]])
                                  for k in (5, 10, 20, 50, 100)}}
    return values


class JointRunnerTests(unittest.TestCase):
    def test_fixed_matrix_and_target_rejection(self):
        self.assertEqual(len(runner.matrix()), 135)
        with self.assertRaises(ValueError):
            runner.source_path("imdb")
        with self.assertRaises(ValueError):
            runner.validate_config({**runner.FROZEN, "k": 5})

    def test_full_plan_preserves_100_exclusion_and_digests(self):
        data = identity()
        plan = plan_for(data)
        result = runner.certify_plan(plan, data, "telstra", 0, expected_n=20)
        self.assertEqual(result["support"], list(range(20)))
        self.assertEqual(result["query"], list(range(100, 120)))
        for mutator in (lambda p: p["query"].append(99),
                        lambda p: p["supports"].update({"20": list(range(1, 21))}),
                        lambda p: p.update(query_digest="bad")):
            bad = copy.deepcopy(plan)
            mutator(bad)
            with self.assertRaises(ValueError):
                runner.certify_plan(bad, data, "telstra", 0, expected_n=20)

    def test_identity_requires_fp64_and_no_truncation_or_duplicate(self):
        data = identity()
        for key, value in (("true", data["true"].astype(np.float32)),
                           ("query_hash", np.array(["a" * 65] * 120)),
                           ("n_tables", np.zeros(120))):
            bad = {**data, key: value}
            with self.assertRaises(ValueError):
                runner.encode_identity(bad)
        result = runner.encode_identity(data)
        np.testing.assert_array_equal(result["true"], data["true"])
        self.assertEqual(result["query_hash"].dtype, np.dtype("S64"))

    def test_metrics_linear_standard_and_empty_bucket_null(self):
        data = {key: value[:4] for key, value in identity().items()}
        data["true"] = np.ones(4)
        pred = np.log([1, 2, 4, 8])
        result = runner.cell_metrics(pred, data)
        self.assertAlmostEqual(result["all"]["q95"], float(np.percentile([1, 2, 4, 8], 95)))
        self.assertIsNone(result["joint"][">=1e6|5+"]["q95"])
        self.assertEqual(sum(r["count"] for r in result["joint"].values()), 4)

    def test_anchor_exact_identity_truth_and_fixed_prediction_tolerance(self):
        data = {key: value[:4] for key, value in identity().items()}
        pred = np.log(data["true"]) + 0.1
        anchor = {**data, "log_pred": pred.copy()}
        runner.verify_anchor(pred, data, anchor)
        bad = {**anchor, "true": anchor["true"] + 0.01}
        with self.assertRaises(ValueError):
            runner.verify_anchor(pred, data, bad)
        with self.assertRaises(ValueError):
            runner.verify_anchor(pred + 0.1, data, anchor)

    def test_none_never_calls_adapt_and_selected_names(self):
        model = SetEstimator(arch="sum", use_pg=True)
        before = runner.state_digest(model)
        with patch("spatial_meta_ce.meta.adapt", side_effect=AssertionError("none called adapt")):
            params, record = runner.operator_params(model, None, "none")
        self.assertEqual(record["selected_parameter_count"], 0)
        self.assertEqual(record["selected_parameter_names"], [])
        self.assertEqual(list(params), list(OrderedDict(model.named_parameters())))
        self.assertEqual(runner.state_digest(model), before)

    def test_head_last_isolate_initialization_and_frozen_parameters(self):
        torch.manual_seed(42)
        model = SetEstimator(arch="sum", use_pg=True)
        support = batch_samples([{"groups": [np.ones((1, dim), dtype=np.float32)
                                            for dim in (40, 40, 4, 43)],
                                  "true": 16777217.25, "pg": 2.0,
                                  "id": "telstra/pretrain/1"}])
        before = runner.state_digest(model)
        for operator, expected in (("head3", 107521), ("last3", 65)):
            first, record = runner.operator_params(model, support, operator)
            second, again = runner.operator_params(model, support, operator)
            self.assertEqual(record["selected_parameter_count"], expected)
            self.assertEqual(record, again)
            self.assertTrue(record["changed_parameter_names"])
            self.assertEqual(runner.state_digest(model), before)
            for name in first:
                self.assertTrue(torch.equal(first[name], second[name]))
                if name not in record["selected_parameter_names"]:
                    self.assertTrue(torch.equal(first[name], dict(model.named_parameters())[name]))

    def test_parent_symlink_is_rejected_and_formal_needs_approval(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            (root / "original").mkdir()
            (root / "link").symlink_to(root / "original")
            with self.assertRaises(ValueError):
                runner.checked_path(root / "link" / "new")
            with self.assertRaisesRegex(ValueError, "not authorized"):
                runner.main(["--out", str(root / "output"), "--mode", "formal", "--execute"])

    def test_sha_mutation_and_output_overwrite_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory) / "input"
            p.write_text("original")
            binding = {str(p): hashlib.sha256(p.read_bytes()).hexdigest()}
            runner.recheck(binding)
            p.write_text("changed")
            with self.assertRaises(ValueError):
                runner.recheck(binding)
            with self.assertRaises(ValueError):
                runner.safe_output(p, [p])

    def test_json_strict_and_empty_run_not_dropped(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory) / "data.json"
            p.write_text('{"a":1,"a":2}')
            with self.assertRaises(ValueError):
                runner.read_json(p)
        rows = [{"all": {"q95": 1}}, {"all": {"q95": None}}]
        self.assertIsNone(runner.mean_metric(rows, "all", "q95"))
        self.assertEqual(json.loads(json.dumps(runner.FROZEN)), runner.FROZEN)
