"""Tests for the target-sidecar preflight (scripts/check_target_sidecars.py).

The script only reports, but its two branches (sidecar present / absent) decide
whether a target evaluation is allowed to start, so both are pinned here.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

_SPEC = importlib.util.spec_from_file_location(
    "check_target_sidecars", ROOT / "scripts/check_target_sidecars.py"
)
sidecars = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(sidecars)

from spatial_meta_ce.data import GROUP_DIMS, TRANSFORM_VERSION  # noqa: E402
from spatial_meta_ce.manifests import digest  # noqa: E402


class CheckTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def cache(self, rows: int = 2, struct: bool = True) -> Path:
        path = self.root / "test" / "synth"
        path.mkdir(parents=True)
        np.save(path / "offsets.npy", np.zeros((rows + 1, 4), dtype=np.int64))
        for index, dim in enumerate(GROUP_DIMS):
            np.save(path / f"group{index}.npy", np.zeros((0, dim), dtype=np.float32))
        np.save(path / "cards.npy", np.asarray([[10.0, 5.0]] * rows))
        manifest = {
            "schema": 1,
            "transform": TRANSFORM_VERSION,
            "rows": [{"id": i, "query_hash": f"h{i}"} for i in range(rows)],
            "unique_queries": rows,
        }
        manifest["content_hash"] = digest(manifest)
        (path / "manifest.json").write_text(json.dumps(manifest))
        if struct:
            np.save(path / "struct_log.npy", np.asarray([1.5] * rows))
            (path / "struct_manifest.json").write_text(
                json.dumps(
                    {
                        "cache_content_hash": manifest["content_hash"],
                        "rows": rows,
                        "bins": 160,
                        "target_database": True,
                        "failure_count": 0,
                    }
                )
            )
        return path

    def test_present_sidecar_passes_and_reports_manifest_fields(self):
        self.cache()
        record, ok = sidecars.check(self.root, "test", "synth")
        self.assertTrue(ok)
        self.assertEqual(record["rows"], 2)
        self.assertTrue(record["batch_has_struct_log"])
        self.assertTrue(record["target_database"])
        self.assertEqual(record["bins"], 160)
        self.assertEqual(record["failure_count"], 0)
        self.assertAlmostEqual(record["anchor_median_log"], 1.5)

    def test_absent_sidecar_fails(self):
        self.cache(struct=False)
        record, ok = sidecars.check(self.root, "test", "synth")
        self.assertFalse(ok)
        self.assertEqual(record, {})

    def test_main_exit_code_reflects_failure(self):
        self.cache(struct=False)
        code = sidecars.main(["--cache-root", str(self.root), "--databases", "synth"])
        self.assertEqual(code, 1)

    def test_main_rejects_path_like_database_name(self):
        with self.assertRaises(SystemExit):
            sidecars.main(["--cache-root", str(self.root), "--databases", "../synth"])


if __name__ == "__main__":
    unittest.main()
