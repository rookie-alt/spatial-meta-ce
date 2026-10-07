"""Synthetic S8 state/signature checks; no model, predictions or training."""

import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from spatial_meta_ce.s8_state import (
    H4,
    T19,
    V3,
    require_nomination,
    seal_nomination,
    select_candidate,
    signature,
)

ARMS = ('A_h_adam', 'B_h_ls', 'C_x_adam', 'D_x_ls', 'F0')


def bindings(mode='smoke'):
    result = {
        'schema': 1,
        'purpose': 's8-engineering-smoke-not-research' if mode == 'smoke' else 's8-research',
        'mode': mode,
        'protocol_sha256': '1' * 64,
        'source_binding_sha256': '2' * 64,
        'panel_identity_digest': '3' * 64,
        'train_databases': list(T19), 'holdout_databases': list(H4),
        'validation_databases': list(V3),
        'model_config': {'arch': 'sum', 'width': 64, 'use_pg': True,
                         'pg_output_scale': 1.0, 'layers': 2, 'heads': 4},
        'seed': 42, 'train_steps': 2 if mode == 'smoke' else 4000,
        'meta_batch': 4, 'query_size': 64, 'support_sizes': [5, 10, 20, 50, 100],
    }
    if mode == 'smoke':
        result['resourcefixedfields'] = {
            'batch_size': 128, 'threads': 4, 'probe_fit_per_database': 64,
            'probe_adam_steps': 2, 'timeout_seconds': 1800,
            'address_space_bytes': 12884901888,
        }
    return result


def metrics():
    return {arm: {db: {'all': {'n': 1000, 'q50': 10.0, 'q95': 100.0},
                       'four_plus': {'n': 200, 'q50': 10.0, 'q95': 80.0}}
                  for db in V3} for arm in ARMS}


def set_candidate(data, arm, q95=72.0):
    for db in V3:
        data[arm][db]['four_plus']['q95'] = q95


def canonical(data):
    return json.dumps(data, sort_keys=True, separators=(',', ':'),
                      ensure_ascii=False, allow_nan=False).encode('utf-8')


def file_sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class SignatureTests(unittest.TestCase):
    def test_signature_is_canonical_and_binds_auxiliary_fields(self):
        data = bindings()
        original = copy.deepcopy(data)
        expected = hashlib.sha256(canonical(data)).hexdigest()
        self.assertEqual(signature(data), expected)
        self.assertEqual(signature(dict(reversed(list(data.items())))), expected)
        data['runtime'] = {'python': 'synthetic-only', 'batch_size': 128}
        self.assertNotEqual(signature(data), expected)
        self.assertEqual(original, bindings())

    def test_literal_partitions_match_frozen_contract(self):
        self.assertEqual(H4, ('accidents', 'grants', 'employee', 'hockey'))
        self.assertEqual(V3, ('chembl', 'legalacts', 'telstra'))
        self.assertEqual(T19, ('airline', 'baseball', 'basketball', 'carcinogenesis',
                              'ccs', 'consumer', 'credit', 'financial', 'fnhk',
                              'hepatitis', 'movielens', 'sakila', 'sap', 'seznam',
                              'ssb', 'talkingdata', 'tournament', 'tpc_h', 'tubepricing'))

    def test_all_required_fields_are_mandatory(self):
        for key in bindings():
            data = bindings()
            del data[key]
            with self.subTest(key=key), self.assertRaises(ValueError):
                signature(data)

    def test_unknown_target_heldout_and_changed_order_rejected(self):
        for key in ('train_databases', 'holdout_databases', 'validation_databases'):
            for replacement in ('unknown', 'imdb', 'stats', 'genome', 'ergastf1'):
                data = bindings()
                data[key][0] = replacement
                with self.subTest(key=key, db=replacement), self.assertRaises(ValueError):
                    signature(data)
            data = bindings()
            data[key].reverse()
            with self.subTest(key=key), self.assertRaises(ValueError):
                signature(data)
        for db in (*H4, *V3):
            data = bindings()
            data['train_databases'][0] = db
            with self.subTest(db=db), self.assertRaises(ValueError):
                signature(data)

    def test_unsafe_auxiliary_database_and_non_json_values_rejected(self):
        for extra in ({'database': 'unknown'}, {'checkpoint': '/runs/imdb/last.pt'},
                      {'database': 'imdb'}, {'runtime': {'databases': ['unknown']}},
                      {'extra': float('nan')}, {'extra': float('inf')},
                      {'extra': {1: 'non-string-key'}}, {'extra': (1, 2)}):
            with self.subTest(extra=extra), self.assertRaises(ValueError):
                signature({**bindings(), **extra})

    def test_mode_purpose_steps_and_frozen_controls(self):
        self.assertEqual(len(signature(bindings('formal'))), 64)
        for key, value in (('schema', True), ('mode', 'development'),
                           ('purpose', 's8-research'), ('train_steps', 4000),
                           ('train_steps', True), ('seed', 43), ('seed', 42.0),
                           ('meta_batch', 1), ('query_size', 32),
                           ('support_sizes', [5, 10]), ('support_sizes', [True, 10, 20, 50, 100])):
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                signature({**bindings(), key: value})
        for steps in (2, 3999, 4001):
            with self.subTest(formal_steps=steps), self.assertRaises(ValueError):
                signature({**bindings('formal'), 'train_steps': steps})
        with self.assertRaises(ValueError):
            signature({**bindings('formal'), 'purpose': bindings()['purpose']})

    def test_fixed_model_and_no_unsafe_overrides(self):
        for key, value in (('arch', 'set'), ('width', 128), ('width', 64.0),
                           ('use_pg', False), ('use_pg', 1), ('pg_output_scale', 0),
                           ('pg_output_scale', True), ('layers', 3), ('heads', 8),
                           ('unexpected', 1)):
            data = bindings()
            data['model_config'][key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                signature(data)
        for key in ('arch', 'width', 'use_pg', 'pg_output_scale'):
            data = bindings()
            del data['model_config'][key]
            with self.subTest(key=key), self.assertRaises(ValueError):
                signature(data)

    def test_smoke_resource_budget_is_required_exact_and_signed(self):
        resources = bindings()['resourcefixedfields']
        for key, value in resources.items():
            for bad in (value + 1, True, float(value)):
                data = bindings()
                data['resourcefixedfields'][key] = bad
                with self.subTest(key=key, value=bad), self.assertRaises(ValueError):
                    signature(data)
            data = bindings()
            del data['resourcefixedfields'][key]
            with self.subTest(missing=key), self.assertRaises(ValueError):
                signature(data)
        data = bindings()
        data['resourcefixedfields']['unexpected'] = 1
        with self.assertRaises(ValueError):
            signature(data)

    def test_cyclic_or_unencodable_metadata_is_engineering_error(self):
        data = bindings()
        data['extra'] = data
        with self.assertRaises(ValueError):
            signature(data)
        with self.assertRaises(ValueError):
            signature({**bindings(), 'extra': '\ud800'})

    def test_bad_sha_and_type_rejected(self):
        for key in ('protocol_sha256', 'source_binding_sha256', 'panel_identity_digest'):
            for value in ('', 'g' * 64, 'A' * 64, '1' * 63, None):
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    signature({**bindings(), key: value})
        for bad in (None, [], 'bindings'):
            with self.assertRaises(ValueError):
                signature(bad)


class SelectionTests(unittest.TestCase):
    def test_no_candidate_is_a_valid_development_decision(self):
        before = metrics()
        self.assertEqual(select_candidate(before)['decision'], 'no_development_candidate')
        self.assertIsNone(select_candidate(before)['candidate'])
        self.assertEqual(before, metrics())

    def test_threshold_equality_and_tie_order(self):
        data = metrics()
        for arm in ARMS[1:4]:
            set_candidate(data, arm)
            for db in V3:
                data[arm][db]['all'].update(q50=11.0, q95=110.0)
                data[arm][db]['four_plus']['q50'] = 11.0
        selected = select_candidate(data)
        self.assertEqual(selected['candidate'], 'B_h_ls')
        self.assertEqual(selected['decision'], 'development_promising')
        data['B_h_ls']['chembl']['all']['q50'] = 11.000001
        self.assertEqual(select_candidate(data)['candidate'], 'C_x_adam')
        set_candidate(data, 'D_x_ls', 71.0)
        self.assertEqual(select_candidate(data)['candidate'], 'D_x_ls')

    def test_macro_not_pooled_or_each_database_gate(self):
        data = metrics()
        for db, n, value in zip(V3, (200, 400, 800), (88.0, 70.0, 58.0)):
            for arm in ARMS:
                data[arm][db]['four_plus']['n'] = n
            data['B_h_ls'][db]['four_plus']['q95'] = value
        self.assertEqual(select_candidate(data)['candidate'], 'B_h_ls')

    def test_both_references_and_each_all_four_plus_quantile_protect(self):
        for cohort in ('all', 'four_plus'):
            for quantile in ('q50', 'q95'):
                for reference in ('A_h_adam', 'F0'):
                    data = metrics()
                    set_candidate(data, 'B_h_ls')
                    # Reduce either reference; macro still passes for all-quantile cases.
                    data[reference]['chembl'][cohort][quantile] = 1.0 if quantile == 'q50' else 10.0
                    if cohort == 'four_plus' and quantile == 'q95':
                        for db in V3[1:]:
                            data['B_h_ls'][db]['four_plus']['q95'] = 20.0
                    with self.subTest(cohort=cohort, quantile=quantile, ref=reference):
                        self.assertIsNone(select_candidate(data)['candidate'])
        for reference in ('A_h_adam', 'F0'):
            data = metrics()
            set_candidate(data, 'B_h_ls', 72.000001)
            with self.subTest(reference=reference):
                self.assertIsNone(select_candidate(data)['candidate'])
            set_candidate(data, 'B_h_ls', 70.0)
            for db in V3:
                data[reference][db]['four_plus']['q95'] = 70.0
            self.assertIsNone(select_candidate(data)['candidate'])

    def test_missing_extra_arms_database_cohort_or_quantile_rejected(self):
        for path in (('F0',), ('C_x_adam',), ('B_h_ls', 'chembl'),
                     ('A_h_adam', 'telstra', 'four_plus'),
                     ('D_x_ls', 'legalacts', 'all', 'q95')):
            data = metrics()
            node = data
            for key in path[:-1]:
                node = node[key]
            del node[path[-1]]
            with self.subTest(path=path), self.assertRaises(ValueError):
                select_candidate(data)
        for data in (None, [], {**metrics(), 'PG': metrics()['F0']},
                     {**metrics(), 'H': {}}):
            with self.assertRaises(ValueError):
                select_candidate(data)
        data = metrics()
        data['F0']['imdb'] = data['F0']['chembl']
        with self.assertRaises(ValueError):
            select_candidate(data)

    def test_sparse_nonfinite_invalid_counts_and_quantiles_rejected(self):
        for field, value in (('n', 199), ('n', True), ('n', 200.0), ('n', -1),
                             ('q50', float('nan')), ('q95', float('inf')),
                             ('q50', 0.0), ('q95', -1.0), ('q50', True),
                             ('q95', '100'), ('q50', 101.0)):
            data = metrics()
            data['B_h_ls']['chembl']['all'][field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                select_candidate(data)
        data = metrics()
        for arm in ARMS:
            data[arm]['chembl']['four_plus']['n'] = 1001
        with self.assertRaises(ValueError):
            select_candidate(data)

    def test_overflowing_numeric_summary_is_engineering_error(self):
        data = metrics()
        data['B_h_ls']['chembl']['all']['q95'] = 10 ** 1000
        with self.assertRaises(ValueError):
            select_candidate(data)

    def test_sparse_four_plus_and_unpaired_counts_rejected(self):
        for cohort in ('all', 'four_plus'):
            data = metrics()
            data['D_x_ls']['telstra'][cohort]['n'] += 1
            with self.subTest(cohort=cohort), self.assertRaises(ValueError):
                select_candidate(data)
        data = metrics()
        for arm in ARMS:
            data[arm]['chembl']['four_plus']['n'] = 199
        with self.assertRaises(ValueError):
            select_candidate(data)


class NominationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir='/private/tmp')
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.path = self.root / 'nomination.json'

    def test_smoke_never_selects_or_reads_metrics(self):
        class ForbiddenMetrics(dict):
            def __iter__(self):
                raise AssertionError('smoke must not inspect V metrics')

        with patch('spatial_meta_ce.s8_state.select_candidate', side_effect=AssertionError):
            payload = seal_nomination(self.path, ForbiddenMetrics(), bindings())
        self.assertEqual(payload['decision'], 'smoke_not_selection')
        self.assertIsNone(payload['candidate'])
        self.assertIsNone(payload['metrics'])
        self.assertEqual(self.path.read_bytes(), canonical(payload))
        self.assertEqual(require_nomination(self.path, file_sha(self.path), bindings()), payload)
        self.assertEqual(payload['bindings_sha256'], signature(bindings()))
        body = {k: v for k, v in payload.items() if k != 'body_digest'}
        self.assertEqual(payload['body_digest'], hashlib.sha256(canonical(body)).hexdigest())

    def test_formal_selection_and_no_candidate_remain_distinct_from_smoke(self):
        data = metrics()
        set_candidate(data, 'D_x_ls', 71.0)
        payload = seal_nomination(self.path, data, bindings('formal'))
        self.assertEqual(payload['candidate'], 'D_x_ls')
        self.assertEqual(require_nomination(self.path, file_sha(self.path), bindings('formal')), payload)
        empty = seal_nomination(self.root / 'none.json', metrics(), bindings('formal'))
        self.assertEqual(empty['decision'], 'no_development_candidate')
        self.assertIsNone(empty['candidate'])

    def test_missing_formal_metrics_and_invalid_bindings_leave_no_seal(self):
        for data, meta in ((None, bindings('formal')), (metrics(), {**bindings(), 'seed': 43})):
            with self.assertRaises(ValueError):
                seal_nomination(self.path, data, meta)
            self.assertFalse(self.path.exists())

    def test_overwrite_rejected_and_original_bytes_unchanged(self):
        seal_nomination(self.path, None, bindings())
        original = self.path.read_bytes()
        with self.assertRaises(ValueError):
            seal_nomination(self.path, None, bindings())
        self.assertEqual(self.path.read_bytes(), original)

    def test_symlink_final_parent_ancestor_and_dotdot_rejected(self):
        real = self.root / 'real'
        real.mkdir()
        nested = real / 'nested'
        nested.mkdir()
        link = self.root / 'link'
        link.symlink_to(real, target_is_directory=True)
        file_link = self.root / 'file-link'
        file_link.symlink_to(self.path)
        for path in (link / 'seal.json', link / 'nested' / 'seal.json', file_link,
                     real / '..' / 'unsafe.json'):
            with self.subTest(path=path), self.assertRaises(ValueError):
                seal_nomination(path, None, bindings())
        self.assertFalse(self.path.exists())
        seal_nomination(real / 'safe.json', None, bindings())
        for path in (link / 'safe.json',):
            with self.assertRaises(ValueError):
                require_nomination(path, file_sha(real / 'safe.json'), bindings())
        file_link.unlink()
        file_link.symlink_to(real / 'safe.json')
        with self.assertRaises(ValueError):
            require_nomination(file_link, file_sha(real / 'safe.json'), bindings())

    def test_missing_seal_and_bad_file_sha_block_h(self):
        with self.assertRaises(ValueError):
            require_nomination(self.path, '0' * 64, bindings())
        seal_nomination(self.path, None, bindings())
        for sha in ('0' * 64, '', None):
            with self.assertRaises(ValueError):
                require_nomination(self.path, sha, bindings())

    def test_modified_source_protocol_panel_auxiliary_and_mode_block_h(self):
        seal_nomination(self.path, None, bindings())
        sha = file_sha(self.path)
        for key in ('protocol_sha256', 'source_binding_sha256', 'panel_identity_digest'):
            with self.subTest(key=key), self.assertRaises(ValueError):
                require_nomination(self.path, sha, {**bindings(), key: '4' * 64})
        with self.assertRaises(ValueError):
            require_nomination(self.path, sha, {**bindings(), 'runtime': {'batch_size': 128}})
        with self.assertRaises(ValueError):
            require_nomination(self.path, sha, bindings('formal'))

    def test_tamper_and_recomputed_file_hash_do_not_bypass_body_digest(self):
        seal_nomination(self.path, None, bindings())
        original_sha = file_sha(self.path)
        payload = json.loads(self.path.read_bytes())
        payload['candidate'] = 'B_h_ls'
        self.path.write_bytes(canonical(payload))
        for sha in (original_sha, file_sha(self.path)):
            with self.assertRaises(ValueError):
                require_nomination(self.path, sha, bindings())

    def test_even_recomputed_body_digest_rejects_wrong_decision_or_binding(self):
        payload = seal_nomination(self.path, None, bindings())
        for change in ({'decision': 'no_development_candidate'}, {'bindings_sha256': '4' * 64},
                       {'candidate': 'B_h_ls'}, {'metrics': metrics()}, {'schema': 2},
                       {'unexpected': 'field'}):
            data = {**payload, **change}
            body = {k: v for k, v in data.items() if k != 'body_digest'}
            data['body_digest'] = hashlib.sha256(canonical(body)).hexdigest()
            self.path.write_bytes(canonical(data))
            with self.subTest(change=change), self.assertRaises(ValueError):
                require_nomination(self.path, file_sha(self.path), bindings())

    def test_formal_gate_recomputed_on_read(self):
        data = metrics()
        set_candidate(data, 'B_h_ls')
        payload = seal_nomination(self.path, data, bindings('formal'))
        payload['candidate'] = 'D_x_ls'
        body = {k: v for k, v in payload.items() if k != 'body_digest'}
        payload['body_digest'] = hashlib.sha256(canonical(body)).hexdigest()
        self.path.write_bytes(canonical(payload))
        with self.assertRaises(ValueError):
            require_nomination(self.path, file_sha(self.path), bindings('formal'))

    def test_invalid_path_type_is_engineering_error(self):
        for path in (None, 1):
            with self.subTest(path=path), self.assertRaises(ValueError):
                seal_nomination(path, None, bindings())
            with self.subTest(path=path), self.assertRaises(ValueError):
                require_nomination(path, '1' * 64, bindings())

    def test_parent_swap_cannot_redirect_creation(self):
        # Swap a checked parent to a symlink immediately before the final open.
        import os

        parent = self.root / 'parent'
        parent.mkdir()
        destination = self.root / 'destination'
        destination.mkdir()
        moved = self.root / 'moved'
        original_open = os.open

        def swap_on_create(path, flags, *args, **kwargs):
            if flags & os.O_CREAT:
                parent.rename(moved)
                parent.symlink_to(destination, target_is_directory=True)
            return original_open(path, flags, *args, **kwargs)

        with patch('spatial_meta_ce.s8_state.os.open', side_effect=swap_on_create):
            seal_nomination(parent / 'nomination.json', None, bindings())
        self.assertFalse((destination / 'nomination.json').exists())
        self.assertTrue((moved / 'nomination.json').is_file())
        # Stale pathname is not an authenticated seal path after this race.
        with self.assertRaises(ValueError):
            require_nomination(parent / 'nomination.json',
                               file_sha(moved / 'nomination.json'), bindings())

    def test_noncanonical_duplicate_keys_and_nonfinite_json_block_h(self):
        payload = seal_nomination(self.path, None, bindings())
        for raw in (json.dumps(payload, indent=2).encode(), b'{"schema":1,"schema":1}',
                    b'{"extra":NaN}', b'not-json'):
            self.path.write_bytes(raw)
            with self.subTest(raw=raw[:20]), self.assertRaises(ValueError):
                require_nomination(self.path, file_sha(self.path), bindings())


if __name__ == '__main__':
    unittest.main()
