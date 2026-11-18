import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

import test_official_mappings as official  # noqa: E402

from scripts.prepare_official_module_inputs import (  # noqa: E402
    InputPreparationError,
    _input_paths,
    prepare_inputs,
)
from scripts.prepare_official_module_inputs_parallel import (  # noqa: E402
    prepare_inputs_parallel,
)
from spatial_meta_ce.manifests import file_hash  # noqa: E402
from spatial_meta_ce.official_mappings import verify_input_report  # noqa: E402
from spatial_meta_ce.official_split import TEST_DATABASES  # noqa: E402
from spatial_meta_ce.price_capacity import OFFICIAL_TRAIN_DATABASES  # noqa: E402

PARALLEL_SOURCE = "scripts/prepare_official_module_inputs_parallel.py"
ORIGINAL_CODE = {
    "scripts/prepare_official_module_inputs.py",
    "spatial_meta_ce/official_mappings.py",
    "scripts/audit_input_mapping.py",
    "scripts/build_mapping_sidecar.py",
    "spatial_meta_ce/data.py",
    "spatial_meta_ce/features.py",
    "spatial_meta_ce/manifests.py",
    "spatial_meta_ce/mappings.py",
    "spatial_meta_ce/official_split.py",
    "spatial_meta_ce/price_capacity.py",
    "vendor/price_features/features_tool.py",
}


def make_sources(root):
    data = None
    for database in OFFICIAL_TRAIN_DATABASES:
        data, rows, paths = official.fixture(root, db=database)
        official.sidecar(root / "legacy", rows, paths, official=False)
    return data


class ParallelOfficialPreparationTests(unittest.TestCase):
    def test_parallel_sidecars_match_serial_with_fixed_report_closure(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            data = make_sources(root)
            serial = prepare_inputs(data, root / "serial", root / "legacy")
            parallel = prepare_inputs_parallel(
                data,
                root / "parallel",
                root / "legacy",
                workers=2,
            )

            self.assertEqual(set(parallel) - set(serial), {"preparation_executor"})
            self.assertEqual(
                set(parallel["code_sha256"]), ORIGINAL_CODE | {PARALLEL_SOURCE}
            )
            self.assertEqual(
                parallel["preparation_executor"],
                {
                    "kind": "cpu-process-pool",
                    "workers": 2,
                    "source_sha256": file_hash(ROOT / PARALLEL_SOURCE),
                },
            )
            self.assertEqual(parallel["databases"], serial["databases"])
            self.assertEqual(parallel["target_data_read"], [])
            self.assertEqual(parallel["target_files_read"], [])
            self.assertEqual(parallel["failed_databases"], [])
            self.assertEqual(
                list(parallel["databases"]["pretrain"]),
                list(OFFICIAL_TRAIN_DATABASES),
            )
            for database in OFFICIAL_TRAIN_DATABASES:
                for name in ("manifest.json", "mapping.jsonl"):
                    serial_path = root / "serial/pretrain" / database / name
                    parallel_path = root / "parallel/pretrain" / database / name
                    self.assertEqual(
                        parallel_path.read_bytes(),
                        serial_path.read_bytes(),
                        f"{database}/{name}",
                    )

            verify_input_report(parallel, splits=("pretrain",))
            for executor in (
                None,
                {**parallel["preparation_executor"], "workers": True},
                {**parallel["preparation_executor"], "workers": 5},
                {**parallel["preparation_executor"], "source_sha256": "0" * 64},
            ):
                bad = copy.deepcopy(parallel)
                bad["preparation_executor"] = executor
                with self.subTest(executor=executor), self.assertRaises(ValueError):
                    verify_input_report(bad, splits=("pretrain",))
            bad = copy.deepcopy(parallel)
            bad.pop("preparation_executor")
            with self.assertRaises(ValueError):
                verify_input_report(bad, splits=("pretrain",))

    def test_explicit_test_preparation_records_worker_failures_and_rows(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            data = make_sources(root)
            for database in TEST_DATABASES:
                sql = official.SQL
                if database == TEST_DATABASES[1]:
                    sql = "select count(*) from alpha a join beta b on a.id=b.aid"
                official.fixture(root, "test", database, sql=sql)

            broken = (
                root
                / "price/datas/workloads/pretrain"
                / OFFICIAL_TRAIN_DATABASES[0]
                / "workloads.sql"
            )
            broken.write_text(broken.read_text().replace("|| 10", "|| 11"))

            with self.assertRaises(InputPreparationError) as raised:
                prepare_inputs_parallel(
                    data,
                    root / "parallel",
                    root / "legacy",
                    include_test=True,
                    workers=2,
                )
            report = raised.exception.report
            self.assertEqual(
                report["failed_databases"],
                [
                    f"pretrain/{OFFICIAL_TRAIN_DATABASES[0]}",
                    f"test/{TEST_DATABASES[1]}",
                ],
            )
            self.assertEqual(
                list(report["databases"]["pretrain"]),
                list(OFFICIAL_TRAIN_DATABASES),
            )
            self.assertEqual(list(report["databases"]["test"]), list(TEST_DATABASES))
            failed_training = report["databases"]["pretrain"][
                OFFICIAL_TRAIN_DATABASES[0]
            ]
            self.assertEqual(failed_training["state"], "failed")
            self.assertIn("provenance", failed_training["error"])
            self.assertFalse(failed_training["unsupported_count_complete"])
            self.assertEqual(
                sum(
                    entry["state"] == "completed"
                    for entry in report["databases"]["pretrain"].values()
                ),
                25,
            )
            failed_target = report["databases"]["test"][TEST_DATABASES[1]]
            self.assertEqual(failed_target["state"], "failed")
            self.assertEqual(failed_target["unsupported_count"], 2)
            self.assertTrue(failed_target["unsupported_count_complete"])
            self.assertEqual(report["unsupported_count"], 2)
            self.assertEqual(
                sum(
                    entry["state"] == "completed"
                    for entry in report["databases"]["test"].values()
                ),
                3,
            )
            expected_target_files = []
            for database in TEST_DATABASES:
                paths = _input_paths(data, "test", database)
                expected_target_files.extend(
                    str(path) for name, path in paths.items() if name != "extractor"
                )
            self.assertEqual(report["target_files_read"], expected_target_files)
            on_disk = json.loads(
                (root / "parallel/official_module_inputs.json").read_text()
            )
            self.assertEqual(on_disk, report)

    def test_workers_are_strict_and_output_protection_leaves_bytes_untouched(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            data = official.config(root)
            for workers in (True, 0, 5, -1, 1.0, "2"):
                with self.subTest(workers=workers), self.assertRaises(ValueError):
                    prepare_inputs_parallel(data, root / "new/output", workers=workers)
            self.assertFalse((root / "new").exists())

            existing = root / "existing"
            existing.mkdir()
            sentinel = existing / "sentinel.txt"
            sentinel.write_text("keep")
            with self.assertRaises(FileExistsError):
                prepare_inputs_parallel(data, existing, workers=2)
            self.assertEqual(sentinel.read_text(), "keep")

            target = root / "target"
            target.mkdir()
            alias = root / "alias"
            alias.symlink_to(target, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "symlink"):
                prepare_inputs_parallel(data, alias / "output", workers=2)
            self.assertEqual(list(target.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
