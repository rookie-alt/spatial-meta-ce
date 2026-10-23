"""The probe's fallback feature path must equal the model's own head input.

Server checkpoints predate ``SetEstimator.head_features()``, so the probe
rebuilds the pooled vector by hand.  If that rebuild ever drifts from the
model's real path, every probe number becomes meaningless — hence this test
compares the two on a small random batch.
"""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

import numpy as np
import torch

from spatial_meta_ce.model import GROUP_DIMS, SetEstimator

ROOT = Path(__file__).resolve().parent.parent


def _load_probe():
    spec = importlib.util.spec_from_file_location(
        "db_identity_probe", ROOT / "scripts" / "db_identity_probe.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _batch(batch_size=5, n_tables=4, n_filters=3, n_joins=2):
    # group order follows GROUP_DIMS: joins, fanouts, tables, filters
    widths = (n_joins, n_joins, n_tables, n_filters)
    return {
        "groups": [
            torch.rand(batch_size, width, dim) for width, dim in zip(widths, GROUP_DIMS)
        ],
        "masks": [
            torch.ones(batch_size, n_joins, dtype=torch.bool),
            torch.ones(batch_size, n_joins, dtype=torch.bool),
            torch.ones(batch_size, n_tables, dtype=torch.bool),
            torch.ones(batch_size, n_filters, dtype=torch.bool),
        ],
        # one count per group, as batch_samples produces: (batch, 4)
        "counts": torch.rand(batch_size, len(GROUP_DIMS)) * 10,
        "pg_log": torch.rand(batch_size) * 3,
    }


class ProbeFeaturePathTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.probe = _load_probe()

    def test_fallback_matches_head_input_with_pg(self):
        torch.manual_seed(0)
        model = SetEstimator(use_pg=True, arch="set")
        model.eval()
        batch = _batch()
        with torch.no_grad():
            expected = model._head_input(batch)
            actual = self.probe._pooled_features(model, batch)
        self.assertEqual(expected.shape, actual.shape)
        np.testing.assert_allclose(
            expected.numpy(), actual.numpy(), rtol=1e-6, atol=1e-6
        )

    def test_fallback_matches_head_input_without_pg(self):
        torch.manual_seed(1)
        model = SetEstimator(use_pg=False, arch="set")
        model.eval()
        batch = _batch()
        with torch.no_grad():
            expected = model._head_input(batch)
            actual = self.probe._pooled_features(model, batch)
        np.testing.assert_allclose(
            expected.numpy(), actual.numpy(), rtol=1e-6, atol=1e-6
        )


if __name__ == "__main__":
    unittest.main()
