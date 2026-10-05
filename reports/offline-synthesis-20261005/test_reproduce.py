"""Evidence-export regression checks. All mutations use isolated temporary copies."""
import contextlib
import csv
import importlib.util
import io
import json
import shutil
import tempfile
import unittest
from pathlib import Path

BASE = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location('offline_reproduce', Path(__file__).with_name('reproduce.py'))
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
SOURCE_NAMES = json.loads((BASE / '实验文档/results/offline-synthesis-20261005/summary.json').read_text())['source_sha256']


class ReproduceTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name)
        for name in SOURCE_NAMES:
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(BASE / name, path)
        old = MODULE.ROOT, MODULE.RESULTS, MODULE.OUT, MODULE.DOCUMENT
        self.addCleanup(self.restore, old)
        MODULE.ROOT = self.root
        MODULE.RESULTS = self.root / '实验文档/results'
        MODULE.OUT = MODULE.RESULTS / 'offline-synthesis-20261005'
        MODULE.DOCUMENT = self.root / '实验文档/report.md'

    @staticmethod
    def restore(old):
        MODULE.ROOT, MODULE.RESULTS, MODULE.OUT, MODULE.DOCUMENT = old

    def mutate(self, name, transform):
        path = MODULE.RESULTS / name
        data = json.loads(path.read_text())
        transform(data)
        path.write_text(json.dumps(data))

    def run_export(self):
        with contextlib.redirect_stdout(io.StringIO()):
            MODULE.main()

    def test_complete_export_and_deterministic_regeneration(self):
        self.run_export()
        summary = json.loads((MODULE.OUT / 'summary.json').read_text())
        self.assertEqual(len(summary['target_aggregate']), 108)
        self.assertEqual(len(summary['validation']), 14)
        self.assertEqual(sum(r['support_rows_checked'] for r in summary['validation']), 1260)
        genome = next(r for r in summary['target_aggregate']
                      if r['method'] == 'FOMAML-set' and r['database'] == 'genome' and r['k'] == 0)
        self.assertAlmostEqual(genome['q95_mean'], 248.688, places=3)
        price = next(r for r in summary['target_aggregate'] if r['method'] == 'PRICE')
        self.assertIsNone(price['q95_between_training_seed_sample_std'])
        self.assertEqual(price['training_seed_replicates'], 0)
        self.assertEqual(summary['methodology']['paired_reference'], 'pg, not model zero-shot')
        self.assertAlmostEqual(summary['p1']['three_seed_arithmetic_means']['sum'], 43.3766, places=4)
        self.assertEqual(len(set(summary['p1']['training_sample_plan_digest_by_seed'].values())), 3)
        with (MODULE.OUT / 'per-initialization.csv').open() as f:
            self.assertEqual(len(list(csv.DictReader(f))), 252)
        before = {p.name: p.read_bytes() for p in MODULE.OUT.iterdir()}
        report_before = MODULE.DOCUMENT.read_bytes()
        self.run_export()
        self.assertEqual(before, {p.name: p.read_bytes() for p in MODULE.OUT.iterdir()})
        self.assertEqual(report_before, MODULE.DOCUMENT.read_bytes())

    def test_changed_record_without_matching_aggregate_fails(self):
        self.mutate('frozen3-20260927/s42-summary.json',
                    lambda d: d['results'][0]['unique_query_metrics'].__setitem__('q95', 999.0))
        with self.assertRaises(ValueError):
            self.run_export()
        self.assertFalse(MODULE.OUT.exists())

    def test_support_identity_mismatch_fails(self):
        self.mutate('erm-s42-summary.json',
                    lambda d: d['results'][1]['support_ids'].__setitem__(0, 'wrong-support-id'))
        with self.assertRaises(AssertionError):
            self.run_export()

    def test_reference_cannot_be_relabelled_zero_shot(self):
        self.mutate('frozen3-20260927/s42-summary.json',
                    lambda d: d['results'][1].__setitem__('reference_method', 'zero_shot'))
        with self.assertRaises(AssertionError):
            self.run_export()

    def test_source_split_mismatch_fails(self):
        self.mutate('erm-s42-summary.json',
                    lambda d: d['metadata']['train_databases'].append('stats'))
        with self.assertRaises(AssertionError):
            self.run_export()

    def test_p0_geometric_ratio_drift_fails(self):
        self.mutate('p0-fixed-budget-20261001/decision.json',
                    lambda d: d.__setitem__('primary_macro_ratio', 1.0))
        with self.assertRaises(ValueError):
            self.run_export()

    def test_p1_within_seed_sampling_mismatch_fails(self):
        self.mutate('p1-mapping-ablation-20261004/ablation_runs.json',
                    lambda d: d['runs'][1].__setitem__('sample_plan_digest', 'different'))
        with self.assertRaises(AssertionError):
            self.run_export()

    def test_duplicate_episode_cannot_pass_count_check(self):
        self.mutate('erm-s42-summary.json',
                    lambda d: d['results'].__setitem__(1, d['results'][0]))
        with self.assertRaises(AssertionError):
            self.run_export()


if __name__ == '__main__':
    unittest.main()
