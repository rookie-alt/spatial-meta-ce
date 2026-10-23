"""Synthetic independent cardinality-ratio oracle; no data/model execution."""
import hashlib
import json
import math
import statistics
import sys
from decimal import Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from spatial_meta_ce.evaluate import _pairwise_metrics


def ordinary_q(prediction, truth):
    prediction = max(Decimal(str(prediction)), Decimal(1))
    truth = max(Decimal(str(truth)), Decimal(1))
    return max(prediction / truth, truth / prediction)


def percentile95(values):
    ordered = sorted(float(value) for value in values)
    position = 0.95 * (len(ordered) - 1)
    lo, hi = math.floor(position), math.ceil(position)
    return ordered[lo] + (position - lo) * (ordered[hi] - ordered[lo])


def independent_metrics(prediction, reference, truth):
    model_q = [ordinary_q(p, y) for p, y in zip(prediction, truth)]
    reference_q = [ordinary_q(p, y) for p, y in zip(reference, truth)]
    reduction = [math.log(float(r)) - math.log(float(q)) for q, r in zip(model_q, reference_q)]
    return {
        'mean_log_q_reduction': statistics.mean(reduction),
        'median_log_q_reduction': statistics.median(reduction),
        'fraction_queries_improved': sum(q < r for q, r in zip(model_q, reference_q)) / len(truth),
        'fraction_queries_worsened': sum(q > r for q, r in zip(model_q, reference_q)) / len(truth),
        'paired_delta_q95': percentile95(model_q) - percentile95(reference_q),
    }


cases = [
    ('true100_pg10_model100', [100], [10], [100]),
    ('symmetric_under_over_equal', [1000], [10], [100]),
    ('near_truth_equal_over_under', [125], [80], [100]),
    ('near_truth_equal_under_over', [80], [125], [100]),
    ('cross_truth_improved', [200], [10], [100]),
    ('ordinary_q95_interpolation', [1, 100], [1, 1], [1, 1]),
    ('zero_truth_and_prediction_clamps', [math.exp(-5), 1, 10], [1, 10, math.exp(-9)], [0, 0, 0]),
    ('all_rows', [100] * 4, [10, 10, 1000, 10], [100, 100, 100, 10]),
    ('unique_queries', [100] * 3, [10, 1000, 10], [100, 100, 10]),
]
outputs = []
for name, prediction, reference, truth in cases:
    keys = [(f'row-{i}', f'hash-{i}') for i in range(len(truth))]
    expected = independent_metrics(prediction, reference, truth)
    actual = _pairwise_metrics([math.log(p) for p in prediction], [math.log(p) for p in reference],
                               truth, reference_true=truth, query_keys=keys, reference_query_keys=keys)
    for key in expected:
        if not math.isclose(actual[key], expected[key], rel_tol=1e-12, abs_tol=1e-12):
            raise AssertionError((name, key, expected[key], actual[key]))
    outputs.append({'case': name, 'oracle': expected, 'actual': actual, 'passed': True})
source_paths = ['spatial_meta_ce/evaluate.py', 'spatial_meta_ce/metrics.py',
                'tests/test_paired.py', 'tests/test_paired_metrics.py',
                'reports/s8-review-fixes-20261008/a1-recompute.py']
result = {
    'purpose': 'synthetic-unit-evidence-not-research',
    'oracle': 'Decimal cardinality ratios; Python math/statistics; manually interpolated ordinary Q95',
    'oracle_does_not_call': ['q_values', 'summarize', '_pairwise_metrics'],
    'tested_implementation': '_pairwise_metrics',
    'source_sha256': {path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest() for path in source_paths},
    'cases': outputs,
}
output = Path(__file__).with_name('a1-recompute-after-review.json')
output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
print(json.dumps(result, indent=2, allow_nan=False))
