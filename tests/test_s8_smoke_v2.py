"""All run execution/solver/source/checkpoint operations mocked: never trains."""

import hashlib
import io
import json
import tempfile
import unittest
from contextlib import ExitStack, redirect_stdout
from pathlib import Path
from unittest.mock import Mock, patch, sentinel

import numpy as np
import torch

from scripts import run_s8_smoke as runner

POLICY = 's8-g1-resource-engineering-v2'
STRICT = {'status': 'unknown',
          'scopes': ['true_rank', 'rigorous_retained_subspace', 'strict_linear_optimality'],
          'reason': 'no_verified_arithmetic_certificate'}
FLAGS = ('metadata_pass', 'normal_residual_pass', 'objective_pass', 'reference_objective_pass',
         'reference_prediction_pass', 'reference_spectrum_pass', 'svd_backend_pass',
         'retained_subspace_pass', 'reference_retained_subspace_pass', 'observed_rank_agreement')


class SmokeV2Tests(unittest.TestCase):
    def run_mocked(self, failure=None):
        with tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
            root = Path(tmp).resolve()
            output = root / 'runs/s8-g1-smoke-synthetic-v2'
            sources = {db: {} for db in runner.TRAIN}
            caches = {db: {'query_hash': [f'{db}-{i}' for i in range(1024)],
                          'indices': np.arange(1024), 'true': np.ones(1024),
                          'pg': np.ones(1024), 'n_tables': np.ones(1024, dtype=int)}
                      for db in (*runner.TRAIN, *runner.VALIDATION, *runner.HELD_OUT)}
            model = Mock(return_value=torch.zeros(128))
            store = Mock(spec=['batch', 'get', 'allow_holdout'])
            store.batch.return_value = sentinel.synthetic_batch
            store.get.side_effect = lambda db: caches[db]
            captures = []
            predictions_from = []

            def capture(_model, _store, db, indices):
                captures.append(db)
                n = len(indices)
                return {'h': np.zeros((n, 2), dtype=np.float32),
                        'x': np.zeros((n, 3), dtype=np.float32),
                        'label': np.zeros(n, dtype=np.float32),
                        'pg_log': np.zeros(n, dtype=np.float32),
                        'raw_log_pred': np.zeros(n, dtype=np.float32)}

            def standardize(features):
                n, p = features.shape
                norm = {'mean': np.zeros(p), 'std': np.ones(p), 'scale': np.ones(p),
                        'zero_mask': np.zeros(p, dtype=bool), 'tiny_mask': np.zeros(p, dtype=bool)}
                return np.column_stack((features.astype(np.float64), np.ones(n))), norm

            def solve(design, _target):
                # This sentinel theta MUST be discarded, never used for endpoints.
                coefficient = 999.0 if len(design) == 19456 else 0.0
                return {'theta': np.full(design.shape[1], coefficient), 'objective': 0., 'rank': 1}

            def certificate(design, _target, _adam, _ls):
                resource = len(design) == 19456
                state = 'unknown' if resource else 'stable'
                eng = not (failure == 'resource_engineering' and resource)
                old = not resource and failure != 'small_old_gate'
                cert = {key: True for key in FLAGS}
                cert.update({'engineering_verified': eng, 'pass': old, 'solver_verified': old,
                             'rank_stability': state, 'strict_certificate': dict(STRICT),
                             'observed_ranks': {'reported_ls': 1, 'svd': 1, 'reference_ls': 1},
                             'rank_limited': True, 'numerical_rank_limited': True,
                             'fullrank_numeric_solver': False, 'full_linear_optimality': False,
                             'full_linear_optimality_verified': False})
                if not eng:
                    cert['reference_prediction_pass'] = False
                if failure == 'small_old_gate' and not resource:
                    cert['rank_stability'] = 'unknown'
                return cert

            def apply(features, norm):
                predictions_from.append(len(features))
                return np.column_stack((features.astype(np.float64), np.ones(len(features))))

            def safe_hash(path):
                path = Path(path)
                return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else 'd' * 64

            for name, value in {'DATA_ROOT': root, 'runtime_threadpools': Mock(return_value=[
                    {'user_api': 'blas', 'num_threads': 4}]), 'SourceStore': Mock(return_value=store),
                    'panel_indices': Mock(return_value=(list(range(1024)), [])),
                    'train_f0': Mock(return_value=(model, sentinel.optimizer, {}, 'a' * 64, 'b' * 64)),
                    'checkpoint_write': Mock(return_value='c' * 64),
                    'checkpoint_reload': Mock(return_value=(model, {})),
                    'capture': Mock(side_effect=capture), 'standardize_fit': Mock(side_effect=standardize),
                    'solve_adam': Mock(side_effect=lambda design, target, **kwargs: {
                        'theta': np.zeros(design.shape[1]), 'objective': 0., 'updates': 2}),
                    'solve_ls': Mock(side_effect=solve), 'certify_pair': Mock(side_effect=certificate),
                    'standardize_apply': Mock(side_effect=apply)}.items():
                stack.enter_context(patch.object(runner, name, value))
            stack.enter_context(patch.object(runner.np, '__version__', '1.26.1'))
            stack.enter_context(patch.object(runner.torch, '__version__', '2.1.0+cu121'))
            stack.enter_context(patch.object(runner.sys, 'version_info', (3, 11, 4)))
            stack.enter_context(patch.object(runner.torch, 'set_num_threads'))
            stack.enter_context(patch.object(runner.torch, 'set_num_interop_threads'))
            stack.enter_context(patch.object(runner.np, 'save'))
            stack.enter_context(patch.object(runner.np, 'savez'))
            stack.enter_context(patch.object(runner.source, 'preflight', return_value={'input_sha256': {}}))
            stack.enter_context(patch.object(runner.source, 'safe_binary', side_effect=lambda path: io.BytesIO(
                json.dumps({'sources': sources}).encode())))
            stack.enter_context(patch.object(runner.source, 'safe_hash', side_effect=safe_hash))
            stack.enter_context(patch.object(runner.source, 'source_identity_metadata',
                                            side_effect=lambda path, *args, **kwargs: caches[path.name]))
            # Signature itself is tested separately; this fixture tests runner gating and data flow.
            stack.enter_context(patch('spatial_meta_ce.s8_state.signature'))
            stack.enter_context(patch('spatial_meta_ce.s8_state.seal_nomination', return_value={'smoke': True}))
            verify = stack.enter_context(patch('scripts.s8_independent_verify.verify_smoke',
                return_value={'passed': True, 'engineering_verified': True, 'strict_certificate_status': 'unknown'}))
            stack.enter_context(redirect_stdout(io.StringIO()))
            if failure:
                with self.assertRaisesRegex(ValueError, 'Small-matrix' if failure == 'small_old_gate' else 'Resource-profile'):
                    runner.execute(root, output)
                status = json.loads((output / 'status.json').read_text())
                self.assertEqual(status['status'], 'failed')
                self.assertFalse(status['formal_ready'])
                store.get.assert_not_called()
                store.allow_holdout.assert_not_called()
                verify.assert_not_called()
                self.assertFalse((output / 'summary.json').exists())
                return status
            summary = runner.execute(root, output)
            profile = json.loads((output / 'probe-smoke.json').read_text())
            bindings = json.loads((output / 'contract.json').read_text())
            status = json.loads((output / 'status.json').read_text())
            self.assertEqual(summary['status'], 'completed_engineering_smoke')
            self.assertEqual(status['status'], summary['status'])
            self.assertTrue(summary['engineering_verified'])
            self.assertEqual(summary['strict_certificate_status'], 'unknown')
            self.assertFalse(summary['formal_ready'])
            self.assertEqual(summary['candidate_decision'], 'smoke_not_selection')
            self.assertEqual(bindings['schema'], 2)
            self.assertEqual(bindings['numerical_acceptance_policy'], POLICY)
            self.assertEqual(profile['acceptance_protocol_sha256'], bindings['acceptance_protocol_sha256'])
            for feature in ('h', 'x'):
                res = profile['resource_profile'][feature]
                self.assertFalse(res['certificate']['pass'])
                self.assertTrue(res['certificate']['engineering_verified'])
                self.assertEqual(res['stage_status'], 'resource_profiled_with_rank_stability_unknown')
                self.assertTrue(res['not_used_for_predictions_or_selection'])
                self.assertEqual(res['seconds'], res['standardization_seconds'] + res['solver_certification_seconds'])
                self.assertEqual(res['linalg_calls'], 3)
                for stage in ('probe', 'resource'):
                    self.assertTrue((output / f'{feature}-{stage}-certificate.json').exists())
            endpoints = [json.loads(line) for line in (output / 'endpoints.jsonl').read_text().splitlines()]
            self.assertEqual(len(endpoints), 20352)
            self.assertTrue(all(value == 0 for row in endpoints for value in row['predictions'].values()))
            self.assertEqual(len(predictions_from), 52)
            store.allow_holdout.assert_called_once()
            verify.assert_called_once()
            return summary

    def test_approved_resource_unknown_continues_without_resource_coefficients(self):
        self.run_mocked()

    def test_small_panel_old_pass_false_still_stops_even_if_engineering_true(self):
        self.run_mocked('small_old_gate')

    def test_resource_engineering_false_is_failed_not_scientific_unknown(self):
        self.run_mocked('resource_engineering')

    def test_plan_declares_smoke_only_policy_without_execution(self):
        plan = runner.smoke_plan()
        self.assertEqual(plan['status'], 'planned')
        self.assertEqual(plan['numerical_acceptance_policy'], POLICY)
        self.assertFalse(plan['formal_ready'])


if __name__ == '__main__':
    unittest.main()
