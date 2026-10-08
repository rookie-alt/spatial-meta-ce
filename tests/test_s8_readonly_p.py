"""Local synthetic fixtures ONLY; no server, old P, smoke, H or checkpoint I/O."""

import copy
import importlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

try:
    ro = importlib.import_module('scripts.verify_s8_readonly_p')
except ModuleNotFoundError:
    ro = None


class ReadOnlyPTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(ro, 'RED: independent read-only P entrypoint is missing')
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()
        self.old = self.root / 'old-failed'
        self.old.mkdir()
        rng = np.random.default_rng(73)
        self.rows = len(ro.TRAIN) * 4
        self.arrays = {
            'P-h.npy': rng.normal(size=(self.rows, 3)).astype('float32'),
            'P-x.npy': rng.normal(size=(self.rows, 5)).astype('float32'),
            'P-label.npy': rng.normal(size=self.rows).astype('float32'),
            'P-pg_log.npy': rng.normal(size=self.rows).astype('float32'),
        }
        for name, value in self.arrays.items():
            np.save(self.old / name, value, allow_pickle=False)
        self.panels = {db: [0, 1, 2, 3] for db in ro.TRAIN}
        source_map = {'/historic/T-only-source/manifest.json': 'a' * 64}
        self.metadata = {
            'status.json': {'status': 'failed', 'formal_ready': False},
            'contract.json': {'schema': 1, 'mode': 'smoke',
                              'train_databases': list(ro.TRAIN),
                              'panel_identity_digest': 'b' * 64,
                              'source_binding_sha256': ro.digest(source_map),
                              'protocol_sha256': 'c' * 64},
            'panels.json': self.panels,
            'preflight.json': {'input_sha256': source_map, 'panel_unique': self.rows,
                               'split': {'train': list(ro.TRAIN)}},
        }
        for name, value in self.metadata.items():
            (self.old / name).write_text(json.dumps(value))
        files = {name: {'sha256': ro.sha256(self.old / name)}
                 for name in (*ro.ARRAYS, *ro.METADATA)}
        for name, value in self.arrays.items():
            files[name].update(shape=list(value.shape), dtype='float32')
        self.contract = {
            'schema': 1, 'entrypoint': ro.ENTRYPOINT, 'evidence_kind': 'synthetic_fixture',
            'purpose': 'readonly_saved_P_resource_engineering_not_G1',
            'input_root': str(self.old), 'files': files,
            'panel': {'train_databases': list(ro.TRAIN), 'rows_per_database': 4,
                      'small_rows_per_database': 2, 'panel_identity_digest': 'b' * 64,
                      'source_binding_sha256': ro.digest(source_map),
                      'original_protocol_sha256': 'c' * 64,
                      'identity_scope': 'historical_sha_bound_panel_attestation'},
            'normalizer_scope': 'P_only_recomputed_per_panel',
            'numerical_policy': 's8-g1-resource-engineering-v2',
            'budget': dict(ro.BUDGET), 'source_sha256': {},
        }
        self.before = {name: ro.sha256(self.old / name) for name in files}

    def execute(self, contract=None, probe=None):
        output = self.root / 'new-proof'
        output.mkdir()
        return ro.check_saved_p(contract or self.contract, output, probe=probe)

    def mutate_file(self, name, value):
        if name.endswith('.npy'):
            np.save(self.old / name, value, allow_pickle=False)
        else:
            (self.old / name).write_text(json.dumps(value))
        self.contract['files'][name]['sha256'] = ro.sha256(self.old / name)

    def test_synthetic_success_is_not_real_p_or_g1(self):
        result = self.execute()
        self.assertEqual(result['status'], 'verified_readonly_P_engineering')
        self.assertEqual(result['evidence_kind'], 'synthetic_fixture')
        self.assertFalse(result['formal_ready'])
        self.assertEqual(result['original_G1_status'], 'failed')
        for feature in ('h', 'x'):
            full = result['features'][feature]['full_P_resource']
            self.assertTrue(full['engineering_verified'])
            self.assertIn(full['rank_stability'], ('stable', 'unknown'))
            self.assertEqual(full['strict_certificate']['status'], 'unknown')
            self.assertTrue(full['coefficients_discarded'])
            self.assertNotIn('theta', full)
            self.assertEqual(result['features'][feature]['small_P_zero_reference']['claim'],
                             'diagnostic_not_original_Adam_probe_gate')
        self.assertEqual(self.before, {n: ro.sha256(self.old / n) for n in self.before})
        self.assertEqual(set(result['input_sha256_before']), set(ro.ARRAYS + ro.METADATA))

    def test_non_readonly_entrypoint_rejected_before_input_io(self):
        self.contract['entrypoint'] = 'scripts/run_s8_smoke.py'
        with patch.object(ro, 'read_bound', side_effect=AssertionError('unexpected I/O')), self.assertRaises(ValueError):
            self.execute()

    def test_training_calls_absent(self):
        text = Path(ro.__file__).read_text()
        for forbidden in ('solve_adam(', 'train_f0(', 'torch.load(', '.backward(',
                          'run_s8_smoke import', 's8_state import'):
            self.assertNotIn(forbidden, text)

    def test_h_target_checkpoint_extra_files_rejected(self):
        for name in ('H-h.npy', 'imdb.npy', 'last.pt', '../P-h.npy', '/tmp/P-x.npy'):
            with self.subTest(name=name):
                contract = copy.deepcopy(self.contract)
                contract['files'][name] = {'sha256': 'd' * 64}
                with self.assertRaises(ValueError):
                    ro.validate_contract(contract)

    def test_access_guard_rejects_h_target_checkpoint_and_writes(self):
        for name in ('H-h.npy', 'imdb.npy', 'last.pt', 'model.py'):
            with self.subTest(name=name), self.assertRaises(PermissionError):
                ro.authorize_data_open(self.old / name, self.contract, False)
        with self.assertRaises(PermissionError):
            ro.authorize_data_open(self.old / 'P-h.npy', self.contract, True)

    def test_sha_mismatch_fails_and_keeps_new_failure(self):
        self.contract['files']['P-h.npy']['sha256'] = '0' * 64
        result = self.execute()
        self.assertEqual(result['status'], 'failed')
        self.assertEqual(result['stage'], 'source_validation')
        self.assertEqual(result['features']['h']['full_P_resource']['status'], 'not_run')
        self.assertTrue((self.root / 'new-proof/worker-proof.json').is_file())

    def test_dtype_mismatch_not_silently_converted(self):
        self.mutate_file('P-label.npy', self.arrays['P-label.npy'].astype('float64'))
        self.assertEqual(self.execute()['status'], 'failed')

    def test_shape_mismatch_fails_closed(self):
        self.mutate_file('P-x.npy', self.arrays['P-x.npy'][:-1])
        self.assertEqual(self.execute()['status'], 'failed')

    def test_nonfinite_input_fails_closed(self):
        value = self.arrays['P-h.npy'].copy()
        value[0, 0] = np.nan
        self.mutate_file('P-h.npy', value)
        self.assertEqual(self.execute()['status'], 'failed')

    def test_identity_mismatch_fails_closed(self):
        value = copy.deepcopy(self.metadata['contract.json'])
        value['panel_identity_digest'] = 'e' * 64
        self.mutate_file('contract.json', value)
        self.assertEqual(self.execute()['status'], 'failed')

    def test_target_database_identity_fails_closed(self):
        value = copy.deepcopy(self.metadata['contract.json'])
        value['train_databases'][0] = 'imdb'
        self.mutate_file('contract.json', value)
        self.assertEqual(self.execute()['status'], 'failed')

    def test_duplicate_panel_positions_rejected(self):
        self.panels[ro.TRAIN[0]] = [0, 0, 2, 3]
        self.mutate_file('panels.json', self.panels)
        self.assertEqual(self.execute()['status'], 'failed')

    def test_source_binding_mismatch_rejected(self):
        value = copy.deepcopy(self.metadata['preflight.json'])
        value['input_sha256']['/historic/T-only-source/manifest.json'] = 'f' * 64
        self.mutate_file('preflight.json', value)
        self.assertEqual(self.execute()['status'], 'failed')

    def test_symlink_input_rejected(self):
        path = self.old / 'P-h.npy'
        moved = self.root / 'moved.npy'
        path.rename(moved)
        path.symlink_to(moved)
        self.assertEqual(self.execute()['status'], 'failed')

    def test_old_failed_output_and_existing_output_rejected(self):
        for path in (self.old, self.old / 'proof', self.root):
            with self.subTest(path=path), self.assertRaises(ValueError):
                ro.create_output(path, self.contract)
        self.assertEqual(self.before, {n: ro.sha256(self.old / n) for n in self.before})

    def test_resource_unknown_rank_allowed_only_when_engineering_true(self):
        from spatial_meta_ce import s8_probe
        class Probe:
            standardize_fit = staticmethod(s8_probe.standardize_fit)
            residual_targets = staticmethod(s8_probe.residual_targets)
            solve_ls = staticmethod(s8_probe.solve_ls)
            @staticmethod
            def certify_pair(*args):
                record = s8_probe.certify_pair(*args)
                record.update(engineering_verified=True, rank_stability='unknown', rank_interval_margin=0, fullrank_numeric_solver=False,
                              **{'pass': False, 'solver_verified': False})
                return record
        result = self.execute(probe=Probe)
        self.assertEqual(result['status'], 'verified_readonly_P_engineering')
        self.assertEqual(result['features']['x']['full_P_resource']['rank_stability'], 'unknown')

    def test_solver_exception_marks_started_stage_failed_not_not_run(self):
        from spatial_meta_ce import s8_probe
        with patch.object(s8_probe, 'solve_ls', side_effect=ValueError('numeric backend rejected')):
            proof = self.execute(probe=s8_probe)
        self.assertEqual(proof['status'], 'failed')
        row = proof['features']['h']['small_P_zero_reference']
        self.assertEqual(row['status'], 'failed')
        self.assertTrue(row['stage_started'])
        self.assertIsNone(row['engineering_verified'])
        self.assertEqual(proof['features']['x']['full_P_resource']['status'], 'not_run')

    def test_retained_projection_diagnostic_metrics_and_directions(self):
        design = np.eye(3, dtype=np.float64)
        target = np.array([1.0, 2.0, 3.0])
        ls = {'theta': np.array([1.0, 2.0, 3.0 + 1e-10]),
              'rank': 3, 'singular_values': np.array([1.0, 1.0, 1.0])}
        reference = np.array([1.0, 2.0, 3.0 + 2e-10])
        result = ro.retained_projection_diagnostic(design, target, ls, reference)
        self.assertEqual(result['shape'], [3, 3])
        self.assertEqual(result['dtype'], 'float64')
        self.assertEqual(result['rank'], 3)
        self.assertEqual(result['rtol'], 1e-12)
        self.assertEqual(result['atol'], 1e-12)
        self.assertEqual(result['numpy'], np.__version__)
        for name, expected in [('reported', 1e-10), ('reference', 2e-10)]:
            row = result[name]
            self.assertAlmostEqual(row['max_abs_residual'], expected, delta=1e-24)
            self.assertAlmostEqual(row['threshold_at_max'], 4e-12, delta=1e-24)
            self.assertAlmostEqual(row['ratio_to_threshold'], expected / 4e-12, delta=1e-9)
            self.assertGreater(row['residual_norm'], 0)
            self.assertEqual(row['max_row_index'], 2)
            top = row['top_singular_directions'][0]
            self.assertEqual(top['u_direction_index'], 2)
            self.assertAlmostEqual(top['residual_coordinate'], expected, delta=1e-24)
            self.assertAlmostEqual(top['projection_coordinate'], 3.0, delta=1e-24)
            self.assertAlmostEqual(top['threshold'], 4e-12, delta=1e-24)
            self.assertGreater(top['ratio'], 1.0)
        self.assertNotIn('theta', result)
        self.assertNotIn('reference_theta', result)

    def test_failed_retained_stage_records_diagnostic_without_gate_change(self):
        from spatial_meta_ce import s8_probe
        certificate = s8_probe.certify_pair(
            np.eye(3), np.ones(3), {'theta': np.zeros(3), 'objective': 1.0, 'updates': 0},
            {'theta': np.zeros(3), 'objective': 1.0, 'rank': 3,
             'singular_values': np.ones(3), 'normalized_normal_residual': 0.0})
        certificate.update(engineering_verified=False, retained_subspace_pass=False,
                            reference_retained_subspace_pass=False, rank_stability='stable',
                            strict_certificate={'status': 'unknown'}, **{'pass': False, 'solver_verified': False})
        with patch.object(s8_probe, 'certify_pair', return_value=certificate):
            result = self.execute(probe=s8_probe)
        self.assertEqual(result['status'], 'failed')
        row = result['features']['h']['small_P_zero_reference']
        self.assertFalse(row['engineering_verified'])
        diagnostic = row['retained_subspace_diagnostic']
        self.assertEqual(diagnostic['gate_effect'], 'diagnostic_only_no_gate_change')
        self.assertIn('reported', diagnostic)
        self.assertIn('reference', diagnostic)
        self.assertIn('top_singular_directions', diagnostic['reported'])
        stage = json.loads((self.root/'new-proof/h-small_P_zero_reference.json').read_text())
        self.assertIn('retained_subspace_diagnostic', stage)
        self.assertNotIn('theta', stage)
        self.assertNotIn('reference_theta', stage)

    def test_engineering_failure_preserved_stops_before_x(self):
        from spatial_meta_ce import s8_probe
        with patch.object(s8_probe, 'certify_pair', return_value={
                'engineering_verified': False, 'rank_stability': 'unknown',
                'strict_certificate': {'status': 'unknown'}, 'pass': False}):
            result = self.execute(probe=s8_probe)
        self.assertEqual(result['status'], 'failed')
        self.assertEqual(result['features']['x']['full_P_resource']['status'], 'not_run')
        self.assertTrue((self.root / 'new-proof/h-small_P_zero_reference.json').is_file())

    def test_source_changed_during_solve_failure(self):
        from spatial_meta_ce import s8_probe
        original = s8_probe.solve_ls
        def change(*args):
            result = original(*args)
            (self.old / 'status.json').write_text('{}')
            return result
        with patch.object(s8_probe, 'solve_ls', side_effect=change):
            self.assertEqual(self.execute(probe=s8_probe)['status'], 'failed')

    def test_contract_budget_cannot_be_relaxed(self):
        for key, value in (('wall_seconds', 901), ('threads', 5),
                           ('address_space_bytes', 13 * 1024 ** 3), ('cuda_visible_devices', '0')):
            contract = copy.deepcopy(self.contract)
            contract['budget'][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                ro.validate_contract(contract)

    def test_output_schema_not_run_not_false_success(self):
        result = ro.initial_proof(self.contract)
        for feature in ('h', 'x'):
            for panel in ('small_P_zero_reference', 'full_P_resource'):
                row = result['features'][feature][panel]
                self.assertEqual(row['status'], 'not_run')
                self.assertIsNone(row['engineering_verified'])
                self.assertEqual(row['rank_stability'], 'not_run')
                self.assertEqual(row['strict_certificate']['status'], 'unknown')
        self.assertEqual(result['real_smoke_executed'], False)

    def test_real_contract_cannot_use_synthetic_shapes(self):
        self.contract['evidence_kind'] = 'real_saved_P'
        with self.assertRaises(ValueError):
            ro.validate_contract(self.contract)

    def test_schema_accepts_not_run_and_synthetic_success(self):
        from jsonschema import Draft202012Validator
        schema = json.loads((ro.ROOT / 'reports/s8-g1-readonly-p-20261007/proof.schema.json').read_text())
        validator = Draft202012Validator(schema)
        validator.validate(ro.initial_proof(self.contract))
        proof = self.execute()
        validator.validate(proof)
        proof['features']['x']['full_P_resource']['engineering_verified'] = None
        self.assertTrue(list(validator.iter_errors(proof)))

    def test_contract_data_path_rejected_before_bytes(self):
        with patch.object(ro, 'read_regular', side_effect=AssertionError('must not open data')), self.assertRaises(SystemExit):
            ro.main(['--contract', str(self.old/'P-h.npy')])

    def test_complete_direct_worker_invocation_rejected(self):
        c = json.loads((ro.ROOT/'reports/s8-g1-readonly-p-20261007/execution-contract.json').read_text())
        c.update(expected_output_root=ro.REAL_OUTPUT_ROOT, expected_package_root=ro.REAL_PACKAGE_ROOT)
        payload = json.dumps(c).encode()
        import hashlib
        pin = hashlib.sha256(payload).hexdigest()
        package = Path(ro.REAL_PACKAGE_ROOT)
        manifest = package/ro.CONTRACT_RELATIVE
        with patch.object(ro, 'ROOT', package), patch.object(ro.sys, 'platform', 'linux'), patch.object(ro, 'read_regular', return_value=payload), patch.object(ro, 'check_sources'), patch.dict(os.environ, CUDA_VISIBLE_DEVICES='', S8_READONLY_PARENT_PID=str(os.getppid())), patch.object(ro, 'worker', side_effect=AssertionError('must not enter worker')), self.assertRaises(SystemExit):
            ro.main(['--contract', str(manifest), '--contract-sha256', pin,
                     '--output', ro.REAL_INPUT_ROOT, '--_worker'])

    def test_parent_directory_swap_between_check_and_read_rejected(self):
        old_check = ro.no_symlink
        swapped = False
        def swap(path):
            nonlocal swapped
            result = old_check(path)
            if Path(path) == self.old/'P-h.npy' and not swapped:
                self.old.rename(self.root/'original')
                self.old.symlink_to(self.root/'forbidden', target_is_directory=True)
                swapped = True
            return result
        forbidden = self.root/'forbidden'
        forbidden.mkdir()
        (forbidden/'P-h.npy').write_bytes(b'forbidden H bytes')
        with patch.object(ro, 'no_symlink', side_effect=swap), self.assertRaises((ValueError, OSError)):
            ro.read_regular(self.old/'P-h.npy')

    def test_output_directory_swap_between_check_and_write_rejected(self):
        out = self.root/'proof-race'
        out.mkdir()
        old_check = ro.no_symlink
        def swap(path):
            result = old_check(path)
            out.rename(self.root/'proof-original')
            out.symlink_to(self.old, target_is_directory=True)
            return result
        with patch.object(ro, 'no_symlink', side_effect=swap), self.assertRaises((ValueError, OSError)):
            ro.write_json(out/'forbidden.json', {})
        self.assertFalse((self.old/'forbidden.json').exists())

    def test_success_proof_rejects_use_shape_normalizer_and_numeric_conflicts(self):
        proof = self.execute()
        changes = [('resource_only', False), ('not_used_for_predictions_or_selection', False),
                   ('reference_kind', 'Adam'), ('rows', -1), ('columns', -1), ('linalg_calls', 4),
                   ('normalizer_source', None), ('matrix_bytes', 1)]
        for field, value in changes:
            altered = copy.deepcopy(proof)
            altered['features']['x']['full_P_resource'][field] = value
            with self.subTest(field=field):
                self.assertFalse(ro.proof_complete(altered, self.contract))
        for field in ('spectrum_gamma', 'normalized_normal_residual', 'objective_ls'):
            altered = copy.deepcopy(proof)
            del altered['features']['h']['full_P_resource']['certificate'][field]
            with self.subTest(field=field):
                self.assertFalse(ro.proof_complete(altered, self.contract))

    def test_partial_stage_records_after_child_kill_are_not_not_run(self):
        proof = self.execute()
        out = self.root/'new-proof'
        (out/'worker-proof.json').unlink()
        recovered = ro.collect_worker_proof(self.contract, out)
        self.assertEqual(recovered['status'], 'failed')
        self.assertFalse(recovered['original_inputs_unchanged'])
        self.assertTrue(recovered['partial_records_recovered'])
        self.assertEqual(recovered['features']['h']['full_P_resource'], proof['features']['h']['full_P_resource'])
        self.assertNotEqual(recovered['features']['h']['full_P_resource']['status'], 'not_run')
        self.assertFalse(ro.proof_complete(recovered, self.contract))

    def test_truncated_worker_proof_recovers_stages_but_never_success(self):
        proof = self.execute()
        out = self.root/'new-proof'
        (out/'worker-proof.json').write_text('{truncated')
        recovered = ro.collect_worker_proof(self.contract, out)
        self.assertEqual(recovered['status'], 'failed')
        self.assertIn('worker_record_error', recovered)
        self.assertEqual(recovered['features']['x']['full_P_resource'], proof['features']['x']['full_P_resource'])
        self.assertFalse(ro.proof_complete(recovered, self.contract))

    def test_partial_recovery_rejects_parseable_unbound_stage_records(self):
        proof = self.execute()
        out = self.root/'new-proof'
        (out/'worker-proof.json').unlink()
        original = proof['features']['x']['full_P_resource']
        records = [{'status': 'verified', 'engineering_verified': True}]
        for key, value in [('rows', -1), ('resource_only', False), ('normalizer_source', None)]:
            row = copy.deepcopy(original)
            row[key] = value
            records.append(row)
        row = copy.deepcopy(original)
        row['normalizer_source']['input_sha256'] = '0' * 64
        records.append(row)
        schema = json.loads((ro.ROOT/'reports/s8-g1-readonly-p-20261007/proof.schema.json').read_text())
        path = out/'x-full_P_resource.json'
        for i, row in enumerate(records):
            with self.subTest(case=i):
                path.write_text(json.dumps(row))
                before = ro.sha256(path)
                recovered = ro.collect_worker_proof(self.contract, out)
                failed = recovered['features']['x']['full_P_resource']
                self.assertEqual(recovered['status'], 'failed')
                self.assertEqual(failed['status'], 'failed')
                self.assertTrue(failed['record_unverified'])
                self.assertFalse(failed['engineering_verified'])
                self.assertEqual(ro.sha256(path), before)
                self.assertTrue(ro.schema_accepts(recovered, schema))

    def test_parent_source_sha_failure_keeps_failed_schema_proof(self):
        import hashlib
        c = json.loads((ro.ROOT/'reports/s8-g1-readonly-p-20261007/execution-contract.json').read_text())
        c.update(expected_output_root=ro.REAL_OUTPUT_ROOT, expected_package_root=ro.REAL_PACKAGE_ROOT)
        payload = json.dumps(c).encode()
        pin = hashlib.sha256(payload).hexdigest()
        package = Path(ro.REAL_PACKAGE_ROOT)
        manifest = package/ro.CONTRACT_RELATIVE
        out = self.root/'parent-source-failure'
        out.mkdir()
        with patch.object(ro, 'ROOT', package), patch.object(ro.sys, 'platform', 'linux'), patch.object(ro, 'read_regular', return_value=payload), patch.object(ro, 'check_sources', side_effect=ValueError('source SHA mismatch')), patch.object(ro, 'create_output', return_value=out), patch.object(ro, 'install_resource_limits'), patch.dict(os.environ, CUDA_VISIBLE_DEVICES=''), patch.object(ro.subprocess, 'Popen', side_effect=AssertionError('must not launch worker')):
            code = ro.main(['--contract', str(manifest), '--contract-sha256', pin,
                            '--output', ro.REAL_OUTPUT_ROOT, '--execute'])
        self.assertEqual(code, 1)
        proof = json.loads((out/'proof.json').read_text())
        self.assertEqual(proof['status'], 'failed')
        self.assertEqual(proof['stage'], 'parent_startup_or_finalize')
        self.assertEqual(proof['features']['x']['full_P_resource']['status'], 'not_run')

    def test_source_sha_mismatch_prevents_import(self):
        self.contract['source_sha256'] = {'spatial_meta_ce/s8_probe.py': '0' * 64}
        self.assertEqual(self.execute()['status'], 'failed')

    def test_default_cli_is_plan_only_no_saved_p_reads(self):
        manifest = self.root / ro.CONTRACT_RELATIVE
        manifest.parent.mkdir(parents=True)
        manifest.write_text(json.dumps(self.contract))
        with patch.object(ro, 'ROOT', self.root), patch.object(ro, 'read_bound', side_effect=AssertionError('saved P read')):
            self.assertEqual(ro.main(['--contract', str(manifest)]), 0)

    def test_direct_worker_cli_denied(self):
        with patch.dict(os.environ, {}, clear=True), self.assertRaises(SystemExit):
            ro.main(['--_worker'])


    def test_fifo_input_is_rejected_without_blocking(self):
        fifo = self.root / 'fifo'
        os.mkfifo(fifo)
        with self.assertRaises(ValueError):
            ro.read_regular(fifo)

    def test_parent_directory_symlink_rejected(self):
        alias = self.root / 'alias'
        alias.symlink_to(self.old, target_is_directory=True)
        self.contract['input_root'] = str(alias)
        self.assertEqual(self.execute()['status'], 'failed')

    def test_protected_price_and_old_code_output_rejected(self):
        for path in ('/gly/weilin/PRICE-master/new-proof',
                     '/gly/weilin/spatial-meta-ce/runs/s8-g1-execution-code-20261007/new-proof'):
            with self.subTest(path=path), self.assertRaises(ValueError):
                ro.create_output(Path(path), self.contract)

    def test_untrusted_contract_sha_rejected_before_input_read(self):
        # Mutated manifest cannot replace the trusted command-line SHA.
        manifest = self.root / 'contract.json'
        contract = copy.deepcopy(self.contract)
        contract['evidence_kind'] = 'real_saved_P'
        contract['input_root'] = ro.REAL_INPUT_ROOT
        contract['expected_output_root'] = ro.REAL_OUTPUT_ROOT
        contract['expected_package_root'] = ro.REAL_PACKAGE_ROOT
        contract['expected_runtime'] = ro.EXPECTED_RUNTIME
        contract['panel'].update(rows_per_database=1024, small_rows_per_database=64)
        for name, shape in {'P-h.npy': [19456, 64], 'P-x.npy': [19456, 774],
                            'P-label.npy': [19456], 'P-pg_log.npy': [19456]}.items():
            contract['files'][name]['shape'] = shape
        contract['source_sha256'] = {name: 'a' * 64 for name in ro.SOURCE_FILES}
        manifest.write_text(json.dumps(contract))
        with patch.object(ro.sys, 'platform', 'linux'), patch.dict(os.environ, CUDA_VISIBLE_DEVICES=''), patch.object(ro, 'read_bound', side_effect=AssertionError('saved P read')), self.assertRaises(SystemExit):
            ro.main(['--contract', str(manifest), '--contract-sha256', 'b' * 64,
                     '--output', ro.REAL_OUTPUT_ROOT, '--execute'])

    def test_real_backend_version_family_and_origin_mismatch_stops(self):
        contract = copy.deepcopy(self.contract)
        contract['evidence_kind'] = 'real_saved_P'
        pools = [{'filepath': '/gly/weilin/PRICE-master/venv/lib/libopenblas64_p.so', 'num_threads': 4},
                 {'filepath': '/gly/weilin/PRICE-master/venv/lib/libgomp.so', 'num_threads': 4}]
        ro.check_runtime_binding(contract, '1.26.1', '2.1.0+cu121', pools)
        for numpy_version, torch_version, value in [('2.0', '2.1.0+cu121', pools),
                ('1.26.1', '2.3', pools), ('1.26.1', '2.1.0+cu121', [{'filepath': '/tmp/libopenblas64_.so', 'num_threads': 4}])]:
            with self.subTest(numpy_version=numpy_version, torch_version=torch_version), self.assertRaises(ValueError):
                ro.check_runtime_binding(contract, numpy_version, torch_version, value)

    def test_parent_and_worker_runtime_limits_are_bounded(self):
        with patch.object(ro.os, 'sched_setaffinity', create=True) as affinity, patch.object(ro.os, 'sched_getaffinity', return_value=set(range(8)), create=True), patch.object(ro.resource, 'setrlimit') as limit, patch.object(ro.resource, 'getrlimit', return_value=(ro.MEMORY, ro.MEMORY)):
            ro.install_resource_limits()
        affinity.assert_called_once_with(0, {0, 1, 2, 3})
        limit.assert_called_once_with(ro.resource.RLIMIT_AS, (ro.MEMORY, ro.MEMORY))

    def test_worker_limit_failure_leaves_evidence(self):
        out = self.root / 'failed-limit'
        out.mkdir()
        with patch.object(ro.os, 'sched_setaffinity', create=True), patch.object(ro.os, 'sched_getaffinity', return_value={0}, create=True), patch.object(ro.resource, 'setrlimit', side_effect=ValueError('limit unavailable')):
            code = ro.worker(self.contract, out, self.root / 'contract.json')
        self.assertEqual(code, 1)
        self.assertEqual(json.loads((out / 'worker-error.json').read_text())['status'], 'failed')

    def test_read_guard_actual_denials_in_isolated_process(self):
        manifest = self.root / 'contract.json'
        manifest.write_text(json.dumps(self.contract))
        out = self.root / 'guard-output'
        out.mkdir()
        code = """
import json, pathlib, sys
from scripts import verify_s8_readonly_p as ro
contract_path=pathlib.Path(sys.argv[1]); out=pathlib.Path(sys.argv[2])
c=json.loads(contract_path.read_text()); old=pathlib.Path(c['input_root'])
ro.install_read_guard(c, out, contract_path)
for operation in [lambda: open(old/'last.pt','rb'), lambda: open(old/'H.npy','rb'),
                  lambda: open(old/'imdb.npy','rb'), lambda: open(old/'status.json','w'),
                  lambda: __import__('os').unlink(old/'status.json'),
                  lambda: __import__('socket').socket().connect(('127.0.0.1',1)),
                  lambda: __import__('subprocess').run(['echo','forbidden'])]:
    try: operation()
    except PermissionError: pass
    else: raise AssertionError('guard allowed a forbidden operation')
"""
        proc = subprocess.run([sys.executable, '-B', '-c', code, str(manifest), str(out)],
                              cwd=ro.ROOT, capture_output=True, text=True, timeout=20, check=False)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(ro.sha256(self.old/'status.json'), self.before['status.json'])

    def test_guarded_local_numeric_import_and_synthetic_ls(self):
        manifest = self.root/'fixture-contract.json'
        manifest.write_text(json.dumps(self.contract))
        out = self.root/'guarded-ls'
        out.mkdir()
        code = """
import json, pathlib, sys
from scripts import verify_s8_readonly_p as ro
manifest=pathlib.Path(sys.argv[1]); out=pathlib.Path(sys.argv[2]); c=json.loads(manifest.read_text())
ro.install_read_guard(c,out,manifest)
from spatial_meta_ce import s8_probe
with ro.computation_guard():
    result=ro.check_saved_p(c,out,probe=s8_probe)
assert result['status']=='verified_readonly_P_engineering', result
"""
        proc = subprocess.run([sys.executable, '-B', '-c', code, str(manifest), str(out)],
                              cwd=ro.ROOT, capture_output=True, text=True, timeout=20, check=False)
        self.assertEqual(proc.returncode, 0, proc.stderr)

    def test_computation_guard_rejects_adam_and_checkpoint_loader(self):
        calls = []
        def solve_adam():
            calls.append('bad')
        with ro.computation_guard(), self.assertRaises(PermissionError):
            solve_adam()
        self.assertEqual(calls, [])

    def test_proof_validator_rejects_missing_and_conflicting_success(self):
        proof = self.execute()
        self.assertTrue(ro.proof_complete(proof, self.contract))
        for key, value in [('formal_ready', True), ('H_read', True), ('Adam_updates', 1),
                           ('original_inputs_unchanged', False)]:
            altered = copy.deepcopy(proof)
            altered[key] = value
            with self.subTest(key=key):
                self.assertFalse(ro.proof_complete(altered, self.contract))
        altered = copy.deepcopy(proof)
        del altered['features']['x']
        self.assertFalse(ro.proof_complete(altered, self.contract))


class MockHandshakeTests(unittest.TestCase):
    """Mock syscall regressions ONLY; Linux integration is a separate test module."""
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.package = Path(self.tmp.name).resolve()/'package'
        self.package.mkdir()
        self.parent_cwd = self.package
        self.relative_contract = getattr(ro, 'CONTRACT_RELATIVE',
                                        'reports/s8-g1-readonly-p-20261007/execution-contract.json')
        self.contract_path = self.package/self.relative_contract
        self.contract_path.parent.mkdir(parents=True)
        self.contract_path.write_text('{}')
        self.output = self.package/'new-proof'
        self.output.mkdir()
        self.entry = self.package/ro.ENTRYPOINT
        self.entry.parent.mkdir()
        self.entry.write_text('# synthetic fixture parent identity')
        self.pin = 'a'*64
        self.command = [sys.executable, '-B', str(self.entry), '--contract',
                        str(self.contract_path), '--contract-sha256', self.pin,
                        '--output', str(self.output), '--execute']
        self.ticket = {'parent_pid': 7777, 'contract_sha256': self.pin,
                       'output': str(self.output), 'package': str(self.package)}
        self.read_pipe = 'pipe:[1234]'
        self.owner_pipe = self.read_pipe

    def call_handshake(self):
        import stat
        from types import SimpleNamespace
        args = SimpleNamespace(contract=self.contract_path, output=self.output,
                               _worker_fd=71, _parent_write_fd=72)
        real_readlink = ro.os.readlink
        def readlink(path):
            paths = {'/proc/7777/cwd': str(self.parent_cwd), '/proc/self/fd/71': self.read_pipe,
                     '/proc/7777/fd/72': self.owner_pipe}
            return paths[str(path)] if str(path) in paths else real_readlink(path)
        with patch.object(ro, 'ROOT', self.package), patch.object(ro.os, 'getppid', return_value=7777), patch.object(ro, 'read_regular', return_value=('\0'.join(self.command)+'\0').encode()), patch.object(ro.os, 'readlink', side_effect=readlink), patch.object(ro.os, 'fstat', return_value=SimpleNamespace(st_mode=stat.S_IFIFO)), patch.object(ro.os, 'read', return_value=json.dumps(self.ticket).encode()), patch.object(ro.os, 'close'), patch.object(ro.os, 'set_blocking'):
            ro.verify_worker_handshake(args, {'expected_output_root': str(self.output)}, self.pin)

    def test_relative_contract_same_frozen_file_passes(self):
        self.command[4] = self.relative_contract
        self.call_handshake()

    def test_absolute_contract_call_still_passes(self):
        self.call_handshake()

    def test_relative_contract_uses_parent_cwd_not_child_cwd(self):
        self.command[4] = self.relative_contract
        with patch.object(ro.os, 'getcwd', return_value=str(self.package/'different-child-cwd')):
            self.call_handshake()

    def test_wrong_parent_cwd_contract_rejected(self):
        self.parent_cwd = self.package/'wrong-parent-cwd'
        self.parent_cwd.mkdir()
        self.command[4] = self.relative_contract
        with self.assertRaises(ValueError):
            self.call_handshake()

    def test_same_basename_wrong_contract_file_rejected(self):
        other = self.package/'elsewhere'/self.contract_path.name
        other.parent.mkdir()
        other.write_text('{}')
        self.command[4] = str(other)
        with self.assertRaises(ValueError):
            self.call_handshake()

    def test_parent_contract_symlink_alias_rejected(self):
        alias = self.package/'contract-alias.json'
        alias.symlink_to(self.contract_path)
        self.command[4] = str(alias)
        with self.assertRaises(ValueError):
            self.call_handshake()

    def test_wrong_parent_contract_SHA_rejected(self):
        self.command[6] = 'b'*64
        with self.assertRaises(ValueError):
            self.call_handshake()

    def test_wrong_parent_output_rejected(self):
        self.command[8] = str(self.package/'wrong-output')
        with self.assertRaises(ValueError):
            self.call_handshake()

    def test_forged_parent_identity_rejected(self):
        self.command[2] = str(self.package/'untrusted-parent.py')
        with self.assertRaises(ValueError):
            self.call_handshake()

    def test_forged_pipe_owner_rejected(self):
        self.owner_pipe = 'pipe:[9999]'
        with self.assertRaises(ValueError):
            self.call_handshake()

    def test_wrong_ticket_contract_SHA_rejected(self):
        self.ticket['contract_sha256'] = 'b'*64
        with self.assertRaises(ValueError):
            self.call_handshake()

    def test_linux_suite_TERM_handler_unwinds_owned_process_cleanup(self):
        import ast
        import signal
        import time
        envelope = ro.ROOT/'reports/s8-g1-readonly-p-handshake-v2-20261008/linux-synthetic-once.py'
        tree = ast.parse(envelope.read_text())
        handlers = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                    and node.name == 'interrupt_suite']
        self.assertEqual(len(handlers), 1, 'RED: suite has no TERM cleanup bridge')
        handler = ast.unparse(handlers[0])
        driver = handler + """
import json, os, signal, subprocess, sys
signal.signal(signal.SIGTERM, interrupt_suite)
child=subprocess.Popen([sys.executable,'-B','-c','import time; time.sleep(10)'],start_new_session=True)
print('ready',flush=True)
error=None
try:
    child.communicate(timeout=5)
except InterruptedError as exc:
    error=str(exc)
finally:
    try: os.killpg(child.pid,signal.SIGKILL)
    except ProcessLookupError: pass
    child.wait(timeout=1)
print(json.dumps({'interrupted': error is not None,'child_reaped':child.poll() is not None,'returncode':child.returncode}),flush=True)
"""
        proc = subprocess.Popen([sys.executable, '-B', '-c', driver],
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                                start_new_session=True)
        try:
            self.assertEqual(proc.stdout.readline().strip(), 'ready')
            time.sleep(0.05)
            proc.send_signal(signal.SIGTERM)
            output, errors = proc.communicate(timeout=3)
            self.assertEqual(proc.returncode, 0, errors)
            record = json.loads(output)
            self.assertTrue(record['interrupted'])
            self.assertTrue(record['child_reaped'])
            self.assertEqual(record['returncode'], -signal.SIGKILL)
        finally:
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.communicate(timeout=1)

    def test_missing_parent_execute_still_rejected(self):
        self.command.remove('--execute')
        with self.assertRaises(ValueError):
            self.call_handshake()


class SupervisorTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(ro, 'RED: read-only supervisor is missing')
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.out = Path(self.tmp.name).resolve()

    def supervise(self, command, **kwargs):
        with (self.out / 'stdout.log').open('wb') as stdout, (self.out / 'stderr.log').open('wb') as stderr:
            child = subprocess.Popen([sys.executable, '-B', '-c', command],
                                     stdout=stdout, stderr=stderr, start_new_session=True)
            if sys.platform != 'linux' and 'sample_process' not in kwargs:
                # macOS VSZ is not Linux RLIMIT_AS; actual exit/signal/timeout tested, counters mocked.
                # Specific resource-failure tests keep their outer mock.
                if not hasattr(ro.sample_process, 'mock_calls'):
                    with patch.object(ro, 'sample_process', return_value={'rss_bytes': 1, 'address_space_bytes': 1}):
                        return ro.supervise(child, self.out, **kwargs)
            return ro.supervise(child, self.out, **kwargs)

    def test_timeout_kills_and_records_no_retry(self):
        result = self.supervise('import time; time.sleep(10)', wall_seconds=0.1)
        self.assertEqual(result['failure_reason'], 'wall_timeout')
        self.assertNotEqual(result['returncode'], 0)
        self.assertEqual(result['attempts'], 1)
        self.assertTrue((self.out / 'supervisor.json').is_file())

    def test_abnormal_exit_records_failure(self):
        result = self.supervise('raise SystemExit(7)')
        self.assertEqual(result['returncode'], 7)
        self.assertEqual(result['failure_reason'], 'child_exit')

    def test_signal_exit_records_signal(self):
        result = self.supervise('import os, signal; os.kill(os.getpid(), signal.SIGKILL)')
        self.assertEqual(result['signal'], 9)
        self.assertEqual(result['failure_reason'], 'child_signal')

    def test_rss_exceeded_stops_and_labels_sample_not_hard_cap(self):
        with patch.object(ro, 'sample_process', return_value={'rss_bytes': ro.MEMORY + 1,
                                                            'address_space_bytes': 1}):
            result = self.supervise('import time; time.sleep(10)')
        self.assertEqual(result['failure_reason'], 'rss_stop_threshold')
        self.assertEqual(result['rss_measurement'], 'sampled_not_hard_cap')

    def test_address_space_exceeded_stops(self):
        with patch.object(ro, 'sample_process', return_value={'rss_bytes': 1,
                                                            'address_space_bytes': ro.MEMORY + 1}):
            result = self.supervise('import time; time.sleep(10)')
        self.assertEqual(result['failure_reason'], 'address_space_budget')

    def test_missing_resource_observation_fails_closed(self):
        with patch.object(ro, 'sample_process', side_effect=ValueError('counter missing')):
            result = self.supervise('import time; time.sleep(10)')
        self.assertEqual(result['failure_reason'], 'resource_observation_failed')

    def test_supervisor_sigterm_cleans_owned_child_and_keeps_evidence(self):
        import signal
        import time
        code = """
import pathlib, subprocess, sys
from scripts import verify_s8_readonly_p as ro
out=pathlib.Path(sys.argv[1])
child=subprocess.Popen([sys.executable,'-B','-c','import time; time.sleep(10)'],start_new_session=True)
(out/'child-pid').write_text(str(child.pid))
ro.sample_process=lambda pid: {'rss_bytes':1,'address_space_bytes':1}
(out/'ready').write_text('ready')
ro.supervise(child,out)
"""
        proc = subprocess.Popen([sys.executable, '-B', '-c', code, str(self.out)], cwd=ro.ROOT)
        try:
            deadline = time.monotonic() + 3
            while not (self.out/'ready').exists() and proc.poll() is None and time.monotonic() < deadline:
                time.sleep(0.01)
            # Wait until supervisor has entered its sampling loop.
            time.sleep(0.1)
            proc.send_signal(signal.SIGTERM)
            proc.wait(timeout=3)
            self.assertTrue((self.out/'supervisor.json').is_file(), 'SIGTERM must leave evidence')
            result = json.loads((self.out/'supervisor.json').read_text())
            self.assertEqual(result['failure_reason'], 'supervisor_interrupted')
            self.assertEqual(result['signal'], 9)
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait()
            # Only the local test-owned PID; not a historical/server process.
            if (self.out/'child-pid').exists():
                pid = int((self.out/'child-pid').read_text())
                try:
                    os.kill(pid, 0)
                except ProcessLookupError:
                    pass
                else:
                    os.kill(pid, signal.SIGKILL)

    def test_success_without_worker_proof_is_failure(self):
        result = self.supervise('pass')
        self.assertEqual(result['failure_reason'], 'missing_worker_proof')


if __name__ == '__main__':
    unittest.main()
