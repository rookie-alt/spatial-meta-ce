"""Reproduce RED for the FINAL test file names by simulating pre-revision code.

Two independent pre-revision conditions are exercised, both bound to the tests
as committed:
  (A) the new bound functions are absent  -> tests referencing them error;
  (B) the retained check is the old fixed 1e-12 allclose -> the strong
      rank-deficiency test fails.
No production file is modified; the module object is patched in memory only.
"""
import importlib
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from spatial_meta_ce import s8_probe as probe

# (B) restore the old fixed-tolerance retained behaviour inside certify_pair.
import numpy as np


def _old_certify(design, target, adam, ls):
    cert = probe.certify_pair(design, target, adam, ls)
    u, singular, _ = np.linalg.svd(probe._array64(design, 2, "d"), full_matrices=False)
    rank = int(np.count_nonzero(singular > probe.RCOND * singular[0]))
    projection = u[:, :rank] @ (u[:, :rank].T @ target)
    theta = probe.solve_ls(design, target)["theta"]
    reference = np.linalg.lstsq(design, target, rcond=probe.RCOND)[0]
    cert["retained_subspace_pass"] = bool(np.allclose(design @ theta, projection, rtol=1e-12, atol=1e-12))
    cert["reference_retained_subspace_pass"] = bool(np.allclose(design @ reference, projection, rtol=1e-12, atol=1e-12))
    return cert


probe.certify_pair = _old_certify
# (A) remove the new bound API so tests that reference it error.
for name in ("retained_projection_bound", "retained_projection_check"):
    if hasattr(probe, name):
        delattr(probe, name)

module = importlib.import_module("tests.test_s8_retained_tolerance")
suite = unittest.defaultTestLoader.loadTestsFromModule(module)
result = unittest.TextTestRunner(verbosity=2).run(suite)
raise SystemExit(0 if result.wasSuccessful() else 1)
