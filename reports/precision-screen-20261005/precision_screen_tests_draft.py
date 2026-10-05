import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from scripts import run_precision_screen as screen
from tests.test_mapping_ablation import build_store


class PrecisionScreenTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_frozen_four_arm_plan_and_no_adaptation(self):
        plan = screen.load_plan()
        self.assertEqual(list(plan['arms']), list(screen.ARMS))
        self.assertEqual([plan['arms'][a]['model']['width'] for a in screen.ARMS], [64, 64, 128, 128])
        self.assertEqual([plan['arms'][a]['loss'] for a in screen.ARMS], ['smooth_l1', 'mse'] * 2)
        for cfg in plan['arms'].values():
            self.assertEqual((cfg['method'], cfg['seed'], cfg['steps'], cfg['inner_steps']), ('erm', 42, 4000, 0))
            self.assertEqual(cfg['patience'], 41)

    def test_changed_frozen_config_is_rejected(self):
        contract = screen.read_json(screen.CONFIG)
        contract['training']['steps'] = 3
        with patch.object(screen, 'read_json', return_value=contract), self.assertRaises(ValueError):
            screen.load_plan()

    def test_protocol_hash_change_is_rejected(self):
        with patch.object(screen, 'file_hash', return_value='bad'), self.assertRaises(ValueError):
            screen.load_plan()

    def test_source_split_rejects_target_or_missing_source(self):
        plan = screen.load_plan()
        data = screen.data_settings(plan['arms'][screen.ARMS[0]], screen.CONFIG)
        screen.validate_data(data, plan)
        for field, value in [('train', 'stats'), ('validation', 'genome')]:
            bad = copy.deepcopy(data)
            bad['splits'][field][0] = value
            with self.assertRaises(ValueError):
                screen.validate_data(bad, plan)

    def test_output_rejects_old_run_or_symlink_to_original(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / 'runs').mkdir()
            data = {'data_root': str(root / 'source'), 'cache_root': str(root / 'cache')}
            screen.validate_output(root / 'runs/new', data, root)
            (root / 'runs/old').mkdir()
            with self.assertRaises(FileExistsError):
                screen.validate_output(root / 'runs/old', data, root)
            with self.assertRaises(ValueError):
                screen.validate_output(root / 'source/new', data, root)
            (root / 'source').mkdir()
            (root / 'runs/link').symlink_to(root / 'source', target_is_directory=True)
            with self.assertRaises(ValueError):
                screen.validate_output(root / 'runs/link/new', data, root)

    def test_cpu_four_arm_smoke_has_shared_sampling_and_same_width_initialization(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for db in ['train', 'chembl', 'legalacts', 'telstra']:
                build_store(root, db, n_rows=10, n_tables=4)
            data = {'data_root': str(root / 'source'), 'cache_root': str(root / 'cache'),
                    'splits': {'train': ['train'], 'validation': ['chembl', 'legalacts', 'telstra'],
                               'test': ['imdb', 'stats', 'genome', 'ergastf1']}}
            configs = screen.load_plan()['arms']
            records = []
            for arm in screen.ARMS:
                cfg = {**configs[arm], 'steps': 2, 'meta_batch': 1, 'support_sizes': [2],
                       'query_size': 3, 'validate_every': 1, 'validation_runs': 1,
                       'purpose': 'synthetic-smoke-not-research'}
                records.append(screen.run_one(arm, cfg, data, root / arm, 'cpu',
                                               expected_coverage=None))
            screen.verify_group(records, steps=2)
            self.assertEqual(len({r['sample_plan_digest'] for r in records}), 1)
            self.assertEqual(records[0]['initialization_hash'], records[1]['initialization_hash'])
            self.assertEqual(records[2]['initialization_hash'], records[3]['initialization_hash'])
            self.assertEqual([r['parameters'] for r in records], [132545, 132545, 288641, 288641])
            bad = copy.deepcopy(records)
            bad[1]['endpoint']['identity_digest'] = 'different'
            with self.assertRaises(ValueError):
                screen.verify_group(bad, steps=2)

    def fixture_records(self):
        return [{'arm': a, 'parameters': 1 if i < 2 else 2,
                 'endpoint': {'macro_four_plus_q95': 100., 'per_database': {
                     db: {'all_tables': {'q95': 100.}, 'four_plus_tables': {'q95': 100.}}
                     for db in ['chembl', 'legalacts', 'telstra']}}}
                for i, a in enumerate(screen.ARMS)]

    def test_promising_rule_uses_both_primary_and_each_database_guards(self):
        records = self.fixture_records()
        self.assertEqual(screen.choose_candidates(records)['decision'], 'no_candidate')
        records[1]['endpoint']['macro_four_plus_q95'] = 90.
        self.assertEqual(screen.choose_candidates(records)['candidates'], [screen.ARMS[1]])
        records[1]['endpoint']['per_database']['telstra']['all_tables']['q95'] = 110.01
        self.assertEqual(screen.choose_candidates(records)['decision'], 'no_candidate')

    def test_nonfinite_result_or_incomplete_matrix_is_rejected(self):
        records = self.fixture_records()
        records[1]['endpoint']['macro_four_plus_q95'] = float('nan')
        with self.assertRaises(ValueError):
            screen.choose_candidates(records)
        with self.assertRaises(ValueError):
            screen.choose_candidates(records[:2])
        with self.assertRaises(ValueError):
            screen.choose_candidates([records[0]] * 4)

    def test_gpu_gate_rejects_cpu_or_wrong_visible_device(self):
        with patch.dict('os.environ', {'CUDA_VISIBLE_DEVICES': '4'}):
            with self.assertRaises(ValueError):
                screen.formal_gpu('cpu', '4')
            with self.assertRaises(ValueError):
                screen.formal_gpu('cuda', '5')
            with patch.object(screen, 'available_gpu', side_effect=RuntimeError('occupied')):
                with self.assertRaises(RuntimeError):
                    screen.formal_gpu('cuda', '4')

    def test_supervisor_timeout_terminates_only_own_child_group(self):
        with tempfile.TemporaryDirectory() as temp:
            child = unittest.mock.Mock()
            child.pid = 12345
            child.wait.side_effect = [screen.subprocess.TimeoutExpired('test', 1), 0]
            with patch.object(screen.subprocess, 'Popen', return_value=child), \
                    patch.object(screen.os, 'killpg') as kill:
                with self.assertRaises(TimeoutError):
                    screen.supervise(['synthetic-command'], Path(temp), timeout=1)
            kill.assert_called_once_with(12345, screen.signal.SIGTERM)
            status = json.loads((Path(temp) / 'status.json').read_text())
            self.assertEqual(status['status'], 'partial')

    def test_failed_arm_records_failure_and_stops_following_arms(self):
        with tempfile.TemporaryDirectory() as temp:
            configs = screen.load_plan()['arms']
            with patch.object(screen, 'run_one', side_effect=FloatingPointError('Nonfinite loss')) as run:
                with self.assertRaises(FloatingPointError):
                    screen.execute_arms(configs, {}, Path(temp), 'cpu', expected_coverage=None)
            self.assertEqual(run.call_count, 1)
            d = json.loads((Path(temp) / 'status.json').read_text())
            self.assertEqual(d['status'], 'failed')
            self.assertEqual(d['arms'][screen.ARMS[0]], 'failed')
            self.assertEqual(d['arms'][screen.ARMS[1]], 'planned')


if __name__ == '__main__':
    unittest.main()
