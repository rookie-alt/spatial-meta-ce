import csv
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from spatial_meta_ce.data import FeatureStore
from spatial_meta_ce.evaluate import (
    _pairwise_metrics,
    _predict_rows,
    _reference_predictions,
)
from spatial_meta_ce.features import write_cache
from spatial_meta_ce.metrics import summarize


class PairedMetricsTests(unittest.TestCase):
    def paired(self, pred, reference, true, **overrides):
        keys = [('row-' + str(i), 'query-' + str(i)) for i in range(len(true))]
        binding = {'reference_true': true, 'query_keys': keys, 'reference_query_keys': keys}
        binding.update(overrides)
        return _pairwise_metrics(pred, reference, true, **binding)

    def test_true_100_pg_10_model_100_is_improved(self):
        actual = self.paired([math.log(100)], [math.log(10)], [100])
        self.assertAlmostEqual(actual['mean_log_q_reduction'], math.log(10))
        self.assertAlmostEqual(actual['median_log_q_reduction'], math.log(10))
        self.assertEqual(actual['fraction_queries_improved'], 1)
        self.assertEqual(actual['fraction_queries_worsened'], 0)
        self.assertAlmostEqual(actual['paired_delta_q95'], -9)

    def test_under_over_cross_truth_and_equal_errors(self):
        # Independent cardinality-ratio oracle, not q_values/summarize.
        for pred, ref, true in [(10, 1, 100), (1000, 10000, 100),
                                (200, 10, 100), (1000, 10, 100),
                                (100, 100, 100), (1, 10, 100)]:
            with self.subTest(pred=pred, reference=ref, true=true):
                model_q = max(pred / true, true / pred)
                ref_q = max(ref / true, true / ref)
                actual = self.paired([math.log(pred)], [math.log(ref)], [true])
                self.assertAlmostEqual(actual['mean_log_q_reduction'],
                                       math.log(ref_q) - math.log(model_q))
                self.assertEqual(actual['fraction_queries_improved'], float(model_q < ref_q))
                self.assertEqual(actual['fraction_queries_worsened'], float(model_q > ref_q))
                self.assertAlmostEqual(actual['paired_delta_q95'], model_q - ref_q)

    def test_mean_median_and_q95_use_different_scales(self):
        # model Q=[1,2,4,8], reference Q=[8,4,2,1].
        # log reductions=[log8,log2,-log2,-log8]; each Q95=7.4.
        actual = self.paired(np.log([100, 50, 400, 800]),
                             np.log([800, 25, 200, 100]), [100] * 4)
        self.assertAlmostEqual(actual['mean_log_q_reduction'], 0)
        self.assertAlmostEqual(actual['median_log_q_reduction'], 0)
        self.assertEqual(actual['fraction_queries_improved'], 0.5)
        self.assertEqual(actual['fraction_queries_worsened'], 0.5)
        self.assertAlmostEqual(actual['paired_delta_q95'], 0)

    def test_equal_near_truth_under_over_error_is_neutral_in_both_directions(self):
        for pred, reference in ((125, 80), (80, 125)):
            with self.subTest(pred=pred, reference=reference):
                actual = self.paired([math.log(pred)], [math.log(reference)], [100])
                self.assertEqual(actual['fraction_queries_improved'], 0.0)
                self.assertEqual(actual['fraction_queries_worsened'], 0.0)
                self.assertAlmostEqual(actual['mean_log_q_reduction'], 0.0)

    def test_q95_is_percentile_of_q_not_exp_percentile_log_q(self):
        # Linear interpolation: Q=[1,100] -> Q95=95.05, not 100**0.95.
        actual = self.paired([0, math.log(100)], [0, 0], [1, 1])
        self.assertAlmostEqual(actual['paired_delta_q95'], 94.05)

    def test_zero_labels_and_negative_log_predictions_follow_standard_clamps(self):
        actual = self.paired([-5, 0, math.log(10)], [0, math.log(10), -9], [0, 0, 0])
        self.assertAlmostEqual(actual['mean_log_q_reduction'], 0)
        self.assertAlmostEqual(actual['median_log_q_reduction'], 0)
        self.assertEqual(actual['fraction_queries_improved'], 1 / 3)
        self.assertEqual(actual['fraction_queries_worsened'], 1 / 3)
        self.assertAlmostEqual(actual['paired_delta_q95'], 0)

    def test_small_real_improvements_are_not_rounded_to_ties(self):
        actual = self.paired([math.log(1000) - 1e-10], [math.log(1000)], [100])
        self.assertEqual(actual['fraction_queries_improved'], 1)
        self.assertEqual(actual['fraction_queries_worsened'], 0)

    def test_nonrepresentable_q_errors_fail_without_clipping(self):
        with self.assertRaises(ValueError):
            self.paired([1000], [1000], [1])

    def test_invalid_arrays_fail_closed(self):
        cases = [([], [], []), ([0], [0, 0], [1]), ([0], [0], [1, 1]),
                 ([math.nan], [0], [1]), ([0], [math.inf], [1]),
                 ([0], [0], [math.nan]), ([0], [0], [math.inf]),
                 ([0], [0], [-1]), ([[0]], [[0]], [1])]
        for pred, ref, true in cases:
            with self.subTest(pred=pred, reference=ref, true=true), self.assertRaises(ValueError):
                self.paired(pred, ref, true)

    def test_label_and_query_bindings_must_match_before_clamping(self):
        for changes in [{'reference_true': [0]}, {'reference_true': [math.nan]},
                        {'reference_true': []}, {'query_keys': []},
                        {'reference_query_keys': [('other-row', 'query-0')]},
                        {'reference_query_keys': [('row-0', 'other-query')]}]:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.paired([0], [0], [1], **changes)

    def test_empty_or_non_string_query_identity_is_rejected(self):
        for keys in [[('', 'query-0')], [('row-0', '')], [('row-0', '   ')],
                     [(math.nan, 'query-0')], [('row-0', math.inf)], ['row-0']]:
            with self.subTest(keys=keys), self.assertRaises(ValueError):
                self.paired([0], [0], [1], query_keys=keys, reference_query_keys=keys)

    def test_no_reference_remains_none(self):
        self.assertIsNone(self.paired([0], None, [1]))

    def test_summarize_unique_q95_is_unchanged(self):
        # Explicit unique positions [0,2,3], ordinary Q=[1,2,100].
        prediction = np.log([100, 100, 50, 10000])
        truth = np.asarray([100] * 4)
        actual = summarize(prediction[[0, 2, 3]], truth[[0, 2, 3]])
        self.assertEqual(actual['n'], 3)
        self.assertAlmostEqual(actual['q95'], 90.2)
        self.assertAlmostEqual(actual['log_q95'], 0.1 * math.log(2) + 0.9 * math.log(100))
        self.assertNotAlmostEqual(actual['q95'], math.exp(actual['log_q95']))


class RawLabelPrecisionTests(unittest.TestCase):
    def test_synthetic_store_batch_and_pg_row_collection_preserve_double_labels(self):
        # First label cannot be represented exactly in float32. No model is used.
        truth = [16777217., 100.125]
        rows = [{'id': f'row-{i}', 'query_hash': f'hash-{i}', 'true': y, 'pg': 1.}
                for i, y in enumerate(truth)]
        features = {
            'data_features': [(torch.ones(40), torch.ones(40), torch.ones(4), None) for _ in rows],
            'true_cards': [torch.tensor(y, dtype=torch.float32) for y in truth],
            'pg_est_cards': [torch.tensor(1.) for _ in rows],
            'n_join_cols': [1, 1], 'n_fanouts': [1, 1], 'n_tables': [1, 1], 'n_filter_cols': [0, 0],
        }
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'test' / 'synthetic'
            write_cache(path, rows, features, {'source': 'synthetic-unit-only'})
            store = FeatureStore(path)
            self.assertEqual(store.cards.dtype, np.float64)
            np.testing.assert_array_equal(store.cards[:, 0], truth)
            batch = store.batch([0, 1], 'cpu')
            self.assertEqual(batch['true'].dtype, torch.float64)
            self.assertEqual(batch['true'].tolist(), truth)
            self.assertNotEqual(float(np.float32(truth[0])), truth[0])
            with patch('spatial_meta_ce.evaluate.forward',
                       side_effect=AssertionError('Model inference is forbidden')):
                _, collected_truth, records = _predict_rows(None, None, store, [0, 1], 'cpu', 1, 'pg')
            self.assertEqual(collected_truth.dtype, np.float64)
            np.testing.assert_array_equal(collected_truth, truth)
            self.assertEqual([record[1] for record in records], truth)


class ReferenceBindingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / 'reference.csv'
        self.store = SimpleNamespace(
            rows=[{'id': 'duplicate-id', 'query_hash': 'hash-a'},
                  {'id': 'duplicate-id', 'query_hash': 'hash-b'}],
            cards=np.asarray([[100., 10.], [20., 5.]]))
        self.rows = [
            {'database': 'synthetic', 'run': 0, 'k': 0, 'query_id': 'duplicate-id',
             'query_hash': 'hash-a', 'true_cardinality': 100., 'log_prediction': math.log(10)},
            {'database': 'synthetic', 'run': 0, 'k': 0, 'query_id': 'duplicate-id',
             'query_hash': 'hash-b', 'true_cardinality': 20., 'log_prediction': math.log(5)},
        ]
        self.metadata = {'pg_prediction_path': str(self.path), 'metadata': {'method': 'pg'}}

    def load(self, rows=None, indices=(0, 1), fields=None):
        rows = self.rows if rows is None else rows
        fields = list(self.rows[0]) if fields is None else fields
        with self.path.open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(rows)
        return _reference_predictions(self.metadata, 'synthetic', self.store, indices)

    def test_duplicate_ids_keep_distinct_hashes_and_requested_order(self):
        prediction, truth, keys = self.load(indices=(1, 0))
        np.testing.assert_allclose(prediction, [math.log(5), math.log(10)])
        np.testing.assert_array_equal(truth, [20., 100.])
        self.assertEqual(keys, [('duplicate-id', 'hash-b'), ('duplicate-id', 'hash-a')])

    def test_identical_k0_repeats_are_allowed_but_conflicts_are_not(self):
        repeats = [dict(row, run=1) for row in self.rows]
        prediction, _, _ = self.load(self.rows + repeats)
        np.testing.assert_allclose(prediction, [math.log(10), math.log(5)])
        for field, value in [('log_prediction', math.log(11)), ('true_cardinality', 101.)]:
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.load(self.rows + [dict(self.rows[0], run=1, **{field: value})])

    def test_identity_label_and_coverage_mismatch_fail_closed(self):
        for field, value in [('query_id', 'other-id'), ('query_hash', 'other-hash'),
                             ('query_hash', ''), ('true_cardinality', 101.)]:
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.load([dict(self.rows[0], **{field: value}), self.rows[1]])
        with self.assertRaises(ValueError):
            self.load(self.rows[:1])

    def test_invalid_and_incomplete_csv_fail_closed(self):
        for field, value in [('true_cardinality', math.nan), ('true_cardinality', math.inf),
                             ('true_cardinality', -1), ('log_prediction', math.nan),
                             ('log_prediction', math.inf)]:
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.load([dict(self.rows[0], **{field: value}), self.rows[1]])
        for field in ('query_hash', 'true_cardinality'):
            with self.subTest(missing_field=field), self.assertRaises(ValueError):
                self.load(fields=[name for name in self.rows[0] if name != field])

    def test_only_requested_database_k0_rows_are_used(self):
        extras = [dict(self.rows[0], database='other', log_prediction='not-used'),
                  dict(self.rows[0], k=5, log_prediction='not-used')]
        prediction, _, _ = self.load(self.rows + extras)
        np.testing.assert_allclose(prediction, [math.log(10), math.log(5)])

    def test_pg_method_declaration_is_required(self):
        for declaration in (None, {}, {'method': None}, {'method': ''}):
            with self.subTest(declaration=declaration):
                self.metadata['metadata'] = declaration
                with self.assertRaises(ValueError):
                    self.load()
        self.metadata.pop('metadata')
        with self.assertRaises(ValueError):
            self.load()

    def test_blank_reference_identity_is_rejected_even_if_store_matches(self):
        self.store.rows[0]['query_hash'] = '   '
        self.rows[0]['query_hash'] = '   '
        with self.assertRaises(ValueError):
            self.load()

    def test_pg_path_cannot_be_mislabeled_as_same_model_k0(self):
        self.metadata['metadata']['method'] = 'fomaml'
        with self.assertRaises(ValueError):
            self.load()


class PairedConsumerLabelTests(unittest.TestCase):
    def test_pg_consumers_label_fraction_as_pg_comparison_not_adaptation(self):
        root = Path(__file__).resolve().parents[1]
        for name in ('eval_frozen.py', 'price_fewshot.py'):
            with self.subTest(consumer=name):
                source = (root / 'scripts' / name).read_text()
                self.assertIn('improved_vs_pg={fi:.3f}', source)
                self.assertNotIn(' improved={fi:.3f}', source)


if __name__ == '__main__':
    unittest.main()
