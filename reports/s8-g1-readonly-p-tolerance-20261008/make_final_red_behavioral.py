"""Behavioural RED: new bound API present, but certify_pair reverted to old
fixed 1e-12 allclose. Proves the strong rank-deficiency case genuinely fails
under the pre-revision gate. No production file is modified (in-memory patch)."""
import importlib
import sys
import unittest
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from spatial_meta_ce import s8_probe as probe

_real = probe.certify_pair


def _old_tol_certify(design, target, adam, ls):
    cert = _real(design, target, adam, ls)
    u, singular, _ = np.linalg.svd(probe._array64(design, 2, "d"), full_matrices=False)
    rank = int(np.count_nonzero(singular > probe.RCOND * singular[0]))
    projection = u[:, :rank] @ (u[:, :rank].T @ target)
    theta = ls["theta"]
    reference = np.linalg.lstsq(design, target, rcond=probe.RCOND)[0]
    cert["retained_subspace_pass"] = bool(np.allclose(design @ theta, projection, rtol=1e-12, atol=1e-12))
    cert["reference_retained_subspace_pass"] = bool(np.allclose(design @ reference, projection, rtol=1e-12, atol=1e-12))
    return cert


probe.certify_pair = _old_tol_certify
module = importlib.import_module("tests.test_s8_retained_tolerance")
names = {"test_strongly_rank_deficient_pair_passes",
         "test_residual_at_bound_passes",
         "test_bound_never_tighter_than_frozen_floor"}
suite = unittest.TestSuite()
for name in list(unittest.defaultTestLoader.getTestCaseNames(module.RetainedToleranceTests)):
    if name in names:
        suite.addTest(module.RetainedToleranceTests(name))
res = unittest.TextTestRunner(verbosity=2).run(suite)
raise SystemExit(0 if res.wasSuccessful() else 1)
