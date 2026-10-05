import contextlib
import io
import json
import pickle
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from scripts.audit_input_mapping import (
    ROOT,
    audit_database,
    feature_groups_match,
    main,
    query_layout,
    source_databases,
)
from spatial_meta_ce.features import write_cache
from spatial_meta_ce.manifests import file_hash, read_workload


class InputAuditTests(unittest.TestCase):
    def test_token_order_and_fanout_direction(self):
        layout = query_layout(
            'select count(*) from beta b, alpha a '
            'where a.id = b.aid and b.v > 2 and a.x < 9'
        )
        self.assertEqual(layout['tables'], ['b', 'a'])
        self.assertEqual(layout['join_columns'], ['b.aid', 'a.id'])
        self.assertEqual(layout['join_column_tables'], [0, 1])
        self.assertEqual(layout['filter_columns'], ['b.v', 'a.x'])
        self.assertEqual(layout['filter_column_tables'], [0, 1])
        self.assertEqual(layout['fanout_join_columns'], [1, 0])
        self.assertEqual(layout['counts'], [2, 2, 2, 2])
        self.assertTrue(layout['mapping_recoverable'])

    def test_repeated_join_column_uses_same_histogram_token(self):
        layout = query_layout(
            'select count(*) from a, b, c where a.id=b.id and a.id=c.id'
        )
        self.assertEqual(layout['join_columns'], ['a.id', 'b.id', 'c.id'])
        self.assertEqual(layout['fanout_join_columns'], [0, 1, 0, 2])
        self.assertEqual(layout['counts'], [3, 4, 3, 0])

    def test_join_filter_overlap_is_reported_without_inventing_filter_token(self):
        layout = query_layout(
            'select count(*) from a, b where a.id=b.id and a.id>3'
        )
        self.assertIn('join_filter_overlap', layout['flags'])
        self.assertEqual(layout['filter_columns'], [])
        self.assertEqual(layout['counts'], [2, 2, 2, 0])
        self.assertTrue(layout['mapping_recoverable'])

    def test_on_join_and_missing_where_are_reported(self):
        layout = query_layout('select count(*) from a join b on a.id=b.id')
        self.assertIn('join_on', layout['flags'])
        self.assertIn('missing_where', layout['flags'])
        self.assertFalse(layout['mapping_recoverable'])

    def test_complex_predicate_types_are_reported(self):
        layout = query_layout(
            'select count(*) from a, b where a.id=b.id and '
            '(a.x between 1 and 3 or b.x in (2,4))'
        )
        self.assertTrue({'or', 'between', 'in'} <= set(layout['flags']))
        self.assertTrue({'Between', 'In'} <= set(layout['predicate_types']))

    def test_negated_and_unsupported_filters_are_flagged(self):
        layout = query_layout(
            "select count(*) from a,b where a.id=b.id and "
            "not a.x like 'A%' and a.y is null and b.x!=3"
        )
        self.assertTrue({'not', 'like', 'is', 'neq'} <= set(layout['flags']))

    def test_non_equi_and_nested_query_are_not_certified(self):
        for sql, flag in (
            ('select count(*) from a,b where a.id>b.id', 'non_equi_join'),
            (('select count(*) from a,b where a.id=b.id and a.x in '
             '(select c.x from c where c.v=1)'), 'subquery'),
        ):
            with self.subTest(flag=flag):
                self.assertIn(flag, query_layout(sql)['flags'])
                self.assertFalse(query_layout(sql)['mapping_recoverable'])

    def test_self_join_aliases_are_local_and_tree_is_connected(self):
        layout = query_layout(
            'select count(*) from users a, users b where a.id=b.parent_id'
        )
        self.assertIn('self_join', layout['flags'])
        self.assertEqual(layout['table_names'], ['users', 'users'])
        self.assertEqual(layout['join_column_tables'], [0, 1])
        self.assertTrue(layout['mapping_recoverable'])
        cycle = query_layout(
            'select count(*) from a,b,c,d where a.id=b.id and '
            'b.id=c.id and c.id=a.id'
        )
        self.assertIn('not_join_tree', cycle['flags'])
        self.assertFalse(cycle['mapping_recoverable'])

    def test_unqualified_columns_are_not_positionally_guessed(self):
        layout = query_layout('select count(*) from a,b where a.id=b.id and x>1')
        self.assertIn('unqualified_column', layout['flags'])
        self.assertFalse(layout['mapping_recoverable'])

    def test_literal_left_and_multiple_filters_are_visible(self):
        layout = query_layout(
            'select count(*) from a,b where a.id=b.id and 1<a.x and a.x<5'
        )
        self.assertIn('literal_left', layout['flags'])
        self.assertIn('multiple_filters', layout['flags'])

    def test_target_or_unknown_database_is_rejected(self):
        data = {'splits': {'train': ['a'], 'validation': ['v'], 'test': ['stats']}}
        self.assertEqual(source_databases(data, None), ['a', 'v'])
        for selected in (['stats'], ['unknown'], ['a', 'a']):
            with self.subTest(selected=selected), self.assertRaises(ValueError):
                source_databases(data, selected)

    def test_constant_equal_column_is_not_a_valid_join(self):
        layout = query_layout('select count(*) from a,b where 2=b.id')
        self.assertIn('malformed_legacy_join', layout['flags'])
        self.assertFalse(layout['mapping_recoverable'])

    def test_reverse_join_flips_fanout_histogram_indices(self):
        a = query_layout('select count(*) from a,b where a.id=b.id')
        b = query_layout('select count(*) from a,b where b.id=a.id')
        self.assertEqual(a['join_columns'], b['join_columns'])
        self.assertEqual(a['fanout_join_columns'], list(reversed(b['fanout_join_columns'])))

    def test_feature_comparison_checks_shape_values_and_empty_groups(self):
        groups = [np.zeros((1, 40)), np.ones((2, 40)), np.zeros((2, 4)), None]
        cached = [np.zeros((1, 40)), np.ones((2, 40)), np.zeros((2, 4)), np.zeros((0, 43))]
        self.assertTrue(feature_groups_match(groups, cached))
        self.assertFalse(feature_groups_match(groups, [*cached[:3], np.ones((1, 43))]))
        self.assertFalse(feature_groups_match(groups, [cached[0], cached[1] + 1, *cached[2:]]))


def audit_fixture(root: Path) -> tuple[dict, dict, list[dict]]:
    source = root / 'source'
    workload = source / 'datas/workloads/pretrain/a/workloads.sql'
    feature = source / 'setup/features/pretrain/a/features40.pkl'
    stats = source / 'datas/statistics/pretrain/a'
    for path in (workload.parent, feature.parent, stats):
        path.mkdir(parents=True)
    workload.write_text('select count(*) from a,b where a.id=b.id || 10 || 2\n' * 2)
    rows = read_workload(workload, 'a', 'pretrain')
    groups = (torch.arange(80, dtype=torch.float32),
              torch.arange(80, dtype=torch.float32) + 5,
              torch.arange(8, dtype=torch.float32), None)
    original = {'data_features': [groups, groups],
                'true_cards': [torch.tensor(10.)] * 2,
                'pg_est_cards': [torch.tensor(2.)] * 2,
                'n_join_cols': [2] * 2, 'n_fanouts': [2] * 2,
                'n_tables': [2] * 2, 'n_filter_cols': [0] * 2}
    with feature.open('wb') as stream:
        pickle.dump(original, stream)
    provenance = {'source': {'workload_sha256': file_hash(workload),
                            'feature_sha256': file_hash(feature),
                            'extractor_sha256': file_hash(ROOT / 'vendor/price_features/features_tool.py'),
                            'statistics': {}}, 'regenerated': False}
    cache = root / 'cache'
    write_cache(cache / 'pretrain/a', rows, original, provenance)
    data = {'data_root': str(source), 'cache_root': str(cache),
            'splits': {'train': ['a'], 'validation': ['v'], 'test': ['stats']}}
    return data, original, rows


class InputAuditIntegrationTests(unittest.TestCase):
    def test_full_recreation_preserves_repeated_rows_and_inputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data, original, rows = audit_fixture(root)
            output = root / 'audit'
            output.mkdir()
            inputs = {p: file_hash(p) for parent in ('source', 'cache')
                      for p in (root / parent).rglob('*') if p.is_file()}
            result = (original['data_features'][0], 2, 2, 2, 0)
            with patch('scripts.audit_input_mapping.extractor') as factory:
                factory.return_value.create_sql_features.return_value = result
                report = audit_database(data, 'a', output, recreate=True)
            self.assertTrue(report['alignment_passed'])
            self.assertEqual(report['totals']['checked_rows'], 2)
            self.assertEqual(report['totals']['recreated_rows'], 2)
            self.assertEqual(report['unique_queries'], 1)
            self.assertEqual(factory.return_value.create_sql_features.call_count, 2)
            self.assertEqual(report['mapping_examples'][0]['fanout_join_columns'], [0, 1])
            self.assertEqual(inputs, {p: file_hash(p) for p in inputs})
            self.assertEqual(report['by_tables']['2']['rows'], len(rows))

    def test_every_row_recreation_mismatch_is_counted(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data, original, _ = audit_fixture(root)
            output = root / 'audit'
            output.mkdir()
            bad = [original['data_features'][0][0] + 1, *original['data_features'][0][1:]]
            with patch('scripts.audit_input_mapping.extractor') as factory:
                factory.return_value.create_sql_features.return_value = (bad, 2, 2, 2, 0)
                report = audit_database(data, 'a', output, recreate=True)
            self.assertFalse(report['alignment_passed'])
            self.assertEqual(report['totals']['recreate_cache_features'], 2)
            self.assertEqual(report['totals']['recreate_source_features'], 2)
            self.assertEqual(report['by_predicate']['EQ']['errors']['recreate_cache_features'], 2)

    def test_cache_and_source_corruption_are_reported(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data, _, _ = audit_fixture(root)
            cache = root / 'cache/pretrain/a'
            cards = np.load(cache / 'cards.npy')
            np.save(cache / 'cards.npy', cards + 1)
            groups = np.load(cache / 'group0.npy')
            np.save(cache / 'group0.npy', groups + 1)
            offsets = np.load(cache / 'offsets.npy')
            offsets[1, 2] = 1
            np.save(cache / 'offsets.npy', offsets)
            output = root / 'audit'
            output.mkdir()
            report = audit_database(data, 'a', output, recreate=False)
            self.assertFalse(report['alignment_passed'])
            for key in ('cache_cards', 'source_cache_features',
                        'source_cache_counts', 'sql_cache_counts'):
                self.assertEqual(report['totals'][key], 2)

    def test_recreation_failure_does_not_drop_rows(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data, _, _ = audit_fixture(root)
            output = root / 'audit'
            output.mkdir()
            with patch('scripts.audit_input_mapping.extractor') as factory:
                factory.return_value.create_sql_features.side_effect = ValueError('fixture')
                report = audit_database(data, 'a', output, recreate=True)
            self.assertFalse(report['alignment_passed'])
            self.assertEqual(report['totals']['checked_rows'], 2)
            self.assertEqual(report['totals']['recreate_failed'], 2)
            self.assertEqual(report['totals']['mapping_recoverable'], 0)

    def test_mapping_denominator_is_independent_of_recreation_failure(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data, _, _ = audit_fixture(root)
            output = root / 'audit'
            output.mkdir()
            with patch('scripts.audit_input_mapping.extractor') as factory:
                factory.return_value.create_sql_features.side_effect = ValueError('fixture')
                report = audit_database(data, 'a', output, recreate=True)
            self.assertEqual(report['totals']['sql_mapping_recoverable'], 2)
            self.assertEqual(report['totals']['mapping_recoverable'], 0)
            self.assertEqual(report['by_tables']['2']['sql_mapping_recoverable'], 2)

    def test_partial_audit_cannot_claim_full_alignment(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data, _, _ = audit_fixture(root)
            output = root / 'audit'
            output.mkdir()
            report = audit_database(data, 'a', output, recreate=False)
            self.assertFalse(report['alignment_passed'])
            self.assertTrue(report['stored_alignment_passed'])

    def test_cli_reports_source_failure_and_rejects_overwrite(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data, _, _ = audit_fixture(root)
            config = root / 'data.json'
            config.write_text(json.dumps(data))
            args = ['audit', '--recreate', '--config', str(config), '--databases', 'a',
                    '--output', str(root / 'report')]
            with patch.object(sys, 'argv', args), \
                    patch('scripts.audit_input_mapping.extractor') as factory, \
                    contextlib.redirect_stdout(io.StringIO()):
                factory.return_value.create_sql_features.return_value = (
                    torch.arange(80, dtype=torch.float32),
                    torch.arange(80, dtype=torch.float32) + 5,
                    torch.arange(8, dtype=torch.float32), None), 2, 2, 2, 0
                main()
            summary = json.loads((root / 'report/summary.json').read_text())
            self.assertEqual(summary['failed_databases'], [])
            with patch.object(sys, 'argv', args), self.assertRaises(FileExistsError):
                main()
            with patch.object(sys, 'argv', [*args[:-1], str(root / 'failed')]), \
                    patch('scripts.audit_input_mapping.audit_database', side_effect=ValueError('fixture failure')), \
                    contextlib.redirect_stdout(io.StringIO()), self.assertRaises(SystemExit):
                main()
            failed = json.loads((root / 'failed/summary.json').read_text())
            self.assertEqual(failed['failed_databases'], ['a'])


if __name__ == '__main__':
    unittest.main()
