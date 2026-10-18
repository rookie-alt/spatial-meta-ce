"""Contract tests for the independent Gate 1 low-dimensional verifier.

The verifier under test is ``scripts/meta65_lowdim_verify.py``. It must
recompute the frozen Gate 1 screen metrics from raw artifacts with NumPy only,
without calling the production runner's own aggregation functions. Fixtures
mirror the real runner format and use a synthetic 130-row / 100-query
denominator so the checks stay fast and independent of the full corpus.
"""
import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from scripts import meta65_lowdim_verify as verifier

ROOT = Path(__file__).resolve().parents[1]
DATABASES = ('chembl', 'legalacts', 'telstra')
OPERATORS = ('none',
             'last3_gradient_smooth_l1',
             'last3_gradient_mse',
             'last3_ridge_mse')
RUNS = (0, 1, 2, 3, 4)
PAIRINGS = (('last3_gradient_smooth_l1_vs_none', OPERATORS[1], 'none'),
            ('last3_gradient_mse_vs_none', OPERATORS[2], 'none'),
            ('last3_ridge_mse_vs_none', OPERATORS[3], 'none'),
            ('ridge_vs_gradient_mse', OPERATORS[3], OPERATORS[2]),
            ('loss_within_gradient', OPERATORS[2], OPERATORS[1]),
            ('combined_solver_and_loss', OPERATORS[3], OPERATORS[1]))
BUCKETS = ('all', 'four_plus', 'low4')
ROWS_PER_DATABASE = 130
QUERY_COUNT = 30


class VerifierImportTests(unittest.TestCase):
    def test_verifier_module_is_importable(self):
        from scripts import meta65_lowdim_verify  # noqa: F401

    def test_verifier_uses_numpy_only_and_no_runner_import(self):
        source = (ROOT / 'scripts' / 'meta65_lowdim_verify.py').read_text()
        self.assertIn('import numpy as np', source)
        self.assertNotIn('run_meta65_lowdim_screen', source)
        self.assertNotIn('import torch', source)


def support_targets(cards, support):
    """Mirror the runner's float32 widened residual target computation."""
    logs = np.log(np.maximum(cards[support], 1)).astype(np.float32).astype(np.float64)
    return logs[:, 0] - logs[:, 1]


def q_error(log_pred, true):
    return np.exp(np.abs(np.maximum(log_pred, 0.0)
                         - np.log(np.maximum(true, 1.0))))


def metric(q):
    return {'count': len(q), 'status': 'below_min_n' if len(q) < 30 else 'ok',
            'q50': float(np.percentile(q, 50, method='linear')),
            'q95': float(np.percentile(q, 95, method='linear'))}


def pair_metric(left, right, left_log, right_log):
    return {'query_count': len(left), 'status': metric(left)['status'],
            'q50_ratio': metric(left)['q50'] / metric(right)['q50'],
            'q95_ratio': metric(left)['q95'] / metric(right)['q95'],
            'mean_delta_log_q': float(np.mean(left_log - right_log)),
            'median_delta_log_q': float(np.median(left_log - right_log)),
            'mean_delta_q': float(np.mean(left - right)),
            'improved_count': int(np.sum(left < right)),
            'worsened_count': int(np.sum(left > right)),
            'tied_count': int(np.sum(left == right)),
            'improved_fraction': float(np.mean(left < right)),
            'lower_q_error_is_better': True}


def digest(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False,
                      allow_nan=False, separators=(',', ':'))


def sha256_bytes(values):
    import hashlib
    return hashlib.sha256(digest(values).encode()).hexdigest()


def database_rows(database, n=ROWS_PER_DATABASE):
    return [{'id': f'{database}/pretrain/{i + 1}',
             'query_hash': sha256_bytes([i, database, i])} for i in range(n)]


def build_manifest(database, rows):
    return {
        'schema': 1,
        'transform': 'synthetic-fixture',
        'provenance': {
            'source': {'database': database, 'split': 'pretrain',
                       'workload_sha256': sha256_bytes(['workload', database]),
                       'feature_sha256': sha256_bytes(['feature', database])},
            'excluded': [],
        },
        'inspection': {'samples': len(rows)},
        'rows': rows,
        'unique_queries': len(rows),
        'content_hash': sha256_bytes(['content', database]),
    }


def build_offsets(rows, rng):
    counts = rng.integers(1, 4, size=(len(rows), 4))
    counts[:, 2] = np.arange(len(rows)) % 3 + 3
    return np.vstack([np.zeros((1, 4), dtype=np.int64), np.cumsum(counts, axis=0)])


def build_cards(rows, rng):
    # Alternate small/large cardinalities so the low4 bucket is never empty.
    true = np.array([60.0 if index % 2 == 0
                     else float(np.exp(rng.uniform(np.log(500), np.log(5000))))
                     for index in range(len(rows))])
    pg = rng.uniform(1.0, 8.0, len(rows))
    return np.stack([true, pg], axis=1).astype(np.float64)


def build_plan(database, run, rows, rng):
    pool = rng.choice(len(rows), size=100, replace=False).tolist()
    query = [i for i in range(len(rows)) if i not in set(pool)]
    hashes = [row['query_hash'] for row in rows]
    return {
        'database': database,
        'run': run,
        'support_max': pool,
        'supports': {str(k): pool[:k] for k in (5, 10, 20, 50, 100)},
        'query': query,
        'query_digest': sha256_bytes([(i, hashes[i]) for i in query]),
        'support_digests': {str(k): sha256_bytes([(i, hashes[i]) for i in pool[:k]])
                            for k in (5, 10, 20, 50, 100)},
    }


def build_audit(operator, support, rows, cards, target, before):
    after = before if operator == 'none' else target + (before - target) * 0.5
    residual_before = before - target
    residual_after = after - target
    absolute = np.abs(residual_before)
    smooth_before = np.where(absolute < 1, 0.5 * residual_before ** 2, absolute - 0.5)
    absolute_after = np.abs(residual_after)
    smooth_after = np.where(absolute_after < 1, 0.5 * residual_after ** 2,
                            absolute_after - 0.5)
    audit = {
        'operator': operator,
        'method': ('identity' if operator == 'none'
                   else 'proximal_ridge_last' if operator == OPERATORS[3]
                   else 'gradient_last'),
        'training': False,
        'support_count': len(support),
        'feature_dim': 64,
        'changed_parameter_names': [] if operator == 'none' else ['head.4.weight', 'head.4.bias'],
        'update_norm': 0.0 if operator == 'none' else 0.35,
        'condition_number_kind': 'diagnostic_XtX_plus_lambda_I',
        'condition_number': 812.5,
        'design_condition_number': 410.25,
        'support_mse_before': float(np.mean(residual_before ** 2)),
        'support_mse_after': float(np.mean(residual_after ** 2)),
        'support_smooth_l1_before': float(np.mean(smooth_before)),
        'support_smooth_l1_after': float(np.mean(smooth_after)),
        'support_residual_improved': bool(np.mean(residual_after ** 2) < np.mean(residual_before ** 2)),
        'support_residuals': [
            {'id': rows[index]['id'], 'query_hash': rows[index]['query_hash'],
             'target': float(target[position]), 'before': float(before[position]),
             'after': float(after[position]),
             'residual_before': float(residual_before[position]),
             'residual_after': float(residual_after[position])}
            for position, index in enumerate(support)
        ],
    }
    if operator == OPERATORS[3]:
        audit.update({
            'proximal_lambda': 1.0,
            'solver_condition_number': 611.75,
            'solver_support_mse_before': audit['support_mse_before'],
            'solver_support_mse_after': audit['support_mse_after'],
            'solver_update_norm': audit['update_norm'],
        })
    return audit


def write_file(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(payload, str):
        path.write_text(payload)
    elif isinstance(payload, np.ndarray):
        np.save(path, payload)
    else:
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + '\n')
    return path


def build_fixture(root, *, seed=7, sha_overrides=None):
    """Materialize a synthetic Gate 1 raw directory plus its evidence mirror."""
    rng = np.random.default_rng(seed)
    raw_root = root / 'raw'
    run_root = raw_root / 'run'
    input_root = raw_root / 'inputs'
    evidence_root = root / 'evidence'
    source_before = {}
    plans = {}
    cells = []
    pair_entries = []
    csv_rows = []
    for database in DATABASES:
        rows = database_rows(database)
        offsets = build_offsets(rows, rng)
        cards = build_cards(rows, rng)
        tables = np.diff(offsets[:, 2]).astype(np.int16)
        cache = input_root / database
        write_file(cache / 'manifest.json', build_manifest(database, rows))
        write_file(cache / 'cards.npy', cards)
        write_file(cache / 'offsets.npy', offsets)
        source_before[database] = {
            'manifest_content_hash': json.loads((cache / 'manifest.json').read_text())['content_hash'],
            'file_sha256': {name: verifier.sha256(cache / name)
                            for name in ('manifest.json', 'cards.npy', 'offsets.npy')},
        }
        plans[database] = []
        for run in RUNS:
            plan = build_plan(database, run, rows, rng)
            plans[database].append(plan)
            support, query = plan['support_max'][:20], plan['query']
            true, table_counts = cards[query, 0], tables[query]
            values = {
                'query_hash': np.asarray([rows[i]['query_hash'] for i in query], dtype='S64'),
                'query_id': np.asarray([rows[i]['id'] for i in query], dtype='S128'),
                'true': true.astype(np.float64),
                'n_tables': table_counts.astype(np.int16),
                'four_plus': (table_counts >= 4).astype(np.bool_),
                'low4': ((table_counts >= 4) & (true < 100)).astype(np.bool_),
            }
            masks = {'all': np.ones(len(query), dtype=bool),
                     'four_plus': table_counts >= 4,
                     'low4': (table_counts >= 4) & (true < 100)}
            q_values, log_q_values = {}, {}
            log_true = np.log(np.maximum(true, 1.0))
            for index, operator in enumerate(OPERATORS):
                scale = (index + 1) * (1 + run + DATABASES.index(database)) * 0.01
                log_pred = log_true + scale * (1 + np.arange(len(query)) / len(query))
                log_q_values[operator] = np.abs(np.maximum(log_pred, 0.0) - log_true)
                with np.errstate(over='raise', invalid='raise'):
                    q_values[operator] = np.exp(log_q_values[operator])
                values[f'{operator}_log_pred'] = log_pred
                values[f'{operator}_q'] = q_values[operator]
            for name, left, right in PAIRINGS:
                values[f'{name}_delta_log_q'] = log_q_values[left] - log_q_values[right]
                values[f'{name}_q_ratio'] = q_values[left] / q_values[right]
            pair_path = run_root / f'pair-{database}-run{run}.npz'
            pair_path.parent.mkdir(parents=True, exist_ok=True)
            with pair_path.open('wb') as stream:
                np.savez_compressed(stream, **values)
            target = support_targets(cards, support)
            before = target + np.linspace(0.4, 1.2, len(support))
            for operator in OPERATORS:
                audit = build_audit(operator, support, rows, cards, target, before)
                metrics = {bucket: metric(q_values[operator][mask])
                           for bucket, mask in masks.items()}
                cells.append({
                    'operator': operator, 'database': database, 'run': run,
                    'query_digest': plan['query_digest'],
                    'support_digest': plan['support_digests']['20'],
                    'query_count': len(query), 'metrics': metrics,
                    'support_audit': audit, 'state_before': 'a' * 64,
                    'state_after': 'a' * 64, 'checkpoint_state_digest': 'a' * 64,
                    'prediction_file': pair_path.name,
                })
                csv_rows.append({
                    'operator': operator, 'database': database, 'run': run,
                    'query_count': len(query),
                    'all_q50': metrics['all']['q50'], 'all_q95': metrics['all']['q95'],
                    'four_plus_count': metrics['four_plus']['count'],
                    'four_plus_q50': metrics['four_plus']['q50'],
                    'four_plus_q95': metrics['four_plus']['q95'],
                    'low4_count': metrics['low4']['count'],
                    'low4_q50': metrics['low4']['q50'], 'low4_q95': metrics['low4']['q95'],
                    'support_mse_before': audit['support_mse_before'],
                    'support_mse_after': audit['support_mse_after'],
                    'support_smooth_l1_before': audit['support_smooth_l1_before'],
                    'support_smooth_l1_after': audit['support_smooth_l1_after'],
                    'update_norm': audit['update_norm'],
                    'condition_number': audit['condition_number'],
                    'solver_condition_number': audit.get('solver_condition_number', ''),
                })
            pair_entries.append({'database': database, 'run': run,
                                 'identity': values, 'log_pred': {
                                     operator: values[f'{operator}_log_pred']
                                     for operator in OPERATORS}})
    write_file(input_root / 'endpoint_plan.json', plans)
    write_file(input_root / 'checkpoint-metadata.json', {
        'config': {'method': 'fomaml', 'seed': 42, 'steps': 4000,
                   'model': dict(verifier.MODEL)},
        'last_step': 4000,
        'checkpoint_sha256': verifier.CHECKPOINT_SHA256,
        'metadata': {'method': 'fomaml', 'seed': 42,
                     'train_databases': [f'train{index}' for index in range(23)],
                     'validation_databases': list(DATABASES)},
    })
    all_metrics, all_pairs = {}, {}
    for cell in cells:
        all_metrics[(cell['operator'], cell['database'], cell['run'])] = cell['metrics']
    for entry in pair_entries:
        values = entry['identity']
        true = values['true']
        masks = {'all': np.ones(len(true), dtype=bool),
                 'four_plus': values['n_tables'] >= 4,
                 'low4': (values['n_tables'] >= 4) & (true < 100)}
        run_pairs = {}
        for name, left, right in PAIRINGS:
            left_q, right_q = values[f'{left}_q'], values[f'{right}_q']
            left_log = np.abs(np.maximum(values[f'{left}_log_pred'], 0)
                              - np.log(np.maximum(true, 1.0)))
            right_log = np.abs(np.maximum(values[f'{right}_log_pred'], 0)
                               - np.log(np.maximum(true, 1.0)))
            run_pairs[name] = {bucket: pair_metric(left_q[mask], right_q[mask],
                                                   left_log[mask], right_log[mask])
                               for bucket, mask in masks.items()}
        entry['pairs'] = run_pairs
        entry['query_digest'] = sha256_bytes(list(zip(
            values['query_hash'].astype(str).tolist(),
            values['query_id'].astype(str).tolist())))
        all_pairs[(entry['database'], entry['run'])] = run_pairs
    summaries = {}
    for operator in OPERATORS:
        per_database = {}
        for database in DATABASES:
            per_database[database] = {
                f'{bucket}_{field}': sum(
                    all_metrics[(operator, database, run)][bucket][field] for run in RUNS) / 5
                for bucket in BUCKETS for field in ('q50', 'q95')}
        summaries[operator] = {
            'per_database': per_database,
            'macro': {field: sum(per_database[database][field]
                                 for database in DATABASES) / 3
                      for field in per_database[DATABASES[0]]},
            'cell_count': 15}
    pair_macro = {name: {bucket: {f'arithmetic_mean_{field}': sum(
        all_pairs[(database, run)][name][bucket][field]
        for database in DATABASES for run in RUNS) / 15
        for field in ('q50_ratio', 'q95_ratio', 'mean_delta_log_q',
                      'mean_delta_q', 'improved_fraction')}
        for bucket in BUCKETS} for name, _, _ in PAIRINGS}
    paired_summary = {
        'schema': 1, 'experiment': verifier.EXPERIMENT,
        'contrasts': [
            {'name': name, 'left': left, 'right': right, 'attribution': attribution,
             'description': 'fixture'}
            for (name, left, right), attribution in zip(
                PAIRINGS[3:], verifier.ATTRIBUTIONS)],
        'runs': [{'database': entry['database'], 'run': entry['run'],
                  'query_digest': entry['query_digest'], 'pairs': entry['pairs']}
                 for entry in pair_entries],
        'macro': pair_macro, 'note': 'fixture'}
    screen_plan = {
        'schema': 1, 'experiment': verifier.EXPERIMENT, 'k': 20, 'support_max': 100,
        'runs': list(RUNS), 'databases': list(DATABASES), 'plan_sha256': verifier.PLAN_SHA256,
        'runs_binding': {database: [{
            'database': database, 'run': run,
            'support': plans[database][run]['support_max'][:20],
            'query_count': len(plans[database][run]['query']),
            'query_digest': plans[database][run]['query_digest'],
            'support_digest': plans[database][run]['support_digests']['20']}
            for run in RUNS] for database in DATABASES}}
    inputs = {
        'schema': 1,
        'checkpoint': {'path': '/fixture/checkpoint.pt', 'sha256': verifier.CHECKPOINT_SHA256},
        'plan': {'path': '/fixture/endpoint_plan.json', 'sha256': verifier.PLAN_SHA256},
        'lambda_registry': {'path': '/fixture/registry.json',
                            'sha256': verifier.REGISTRY_SHA256,
                            'value': verifier.read_json(
                                ROOT / 'configs' / 'meta65_lowdim_gate1_lambda.json')},
        'cache_root': '/fixture/cache',
        'source_before': source_before,
    }
    inputs['source_after'] = json.loads(json.dumps(source_before))
    names = ('status.json', 'inputs.json', 'cells.json', 'screen_plan.json',
             'macro.json', 'paired_summary.json', 'environment.json')
    status = {'schema': 1, 'experiment': verifier.EXPERIMENT, 'status': 'completed',
              'k': 20, 'operators': list(OPERATORS),
              'lambda_registry_sha256': verifier.REGISTRY_SHA256,
              'selected_lambda': 1.0, 'training': False, 'target_read': False,
              'completed_cells': 60, 'gate': 'screen_completed_not_selected'}
    macro = {'schema': 1, 'experiment': verifier.EXPERIMENT, 'k': 20,
             'metric': 'fixture', 'operators': summaries,
             'selection': 'not_performed_by_runner'}
    cells_document = {'schema': 1, 'experiment': verifier.EXPERIMENT, 'cells': cells}
    environment = {'python': 'fixture', 'numpy': np.__version__, 'torch': None,
                   'device': 'cpu', 'CUDA_VISIBLE_DEVICES': None,
                   'CUBLAS_WORKSPACE_CONFIG': None, 'threads': 1,
                   'deterministic_algorithms': True, 'tf32': False, 'training': False}
    documents = {'status.json': status, 'inputs.json': inputs, 'cells.json': cells_document,
                 'screen_plan.json': screen_plan, 'macro.json': macro,
                 'paired_summary.json': paired_summary, 'environment.json': environment}
    for name in names:
        write_file(run_root / name, documents[name])
    csv_columns = list(csv_rows[0])
    with (run_root / 'metrics.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=csv_columns)
        writer.writeheader()
        writer.writerows(csv_rows)
    for name in (*names[:2], 'screen_plan.json', 'macro.json',
                 'paired_summary.json', 'environment.json', 'metrics.csv'):
        evidence_root.mkdir(parents=True, exist_ok=True)
        (evidence_root / name).write_text((run_root / name).read_text())
    return raw_root, evidence_root


class VerifierFixture:
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name).resolve()
        self.addCleanup(self._tmp.cleanup)
        self.raw_root, self.evidence_root = build_fixture(self.root)
        plan_sha = verifier.sha256(self.raw_root / 'inputs/endpoint_plan.json')
        for name, fields in (('inputs.json', ('plan', 'sha256')),
                             ('screen_plan.json', ('plan_sha256',))):
            path = self.raw_root / 'run' / name
            value = json.loads(path.read_text())
            if len(fields) == 2:
                value[fields[0]][fields[1]] = plan_sha
            else:
                value[fields[0]] = plan_sha
            path.write_text(json.dumps(value))
            (self.evidence_root / name).write_text(path.read_text())
        self.patcher = patch.multiple(
            verifier,
            UNIQUE={'chembl': ROWS_PER_DATABASE, 'legalacts': ROWS_PER_DATABASE,
                    'telstra': ROWS_PER_DATABASE},
            CHECKPOINT_SHA256=verifier.CHECKPOINT_SHA256,
            PLAN_SHA256=plan_sha,
            METADATA_SHA256=verifier.sha256(self.raw_root / 'inputs/checkpoint-metadata.json'),
            REGISTRY_SHA256=verifier.REGISTRY_SHA256)
        self.patcher.start()
        self.addCleanup(self.patcher.stop)

    def result(self, raw_root=None):
        return verifier.verify(raw_root or self.raw_root,
                               evidence_root=self.evidence_root)


class SuccessTests(VerifierFixture, unittest.TestCase):
    def test_synthetic_screen_is_verified(self):
        result = self.result()
        self.assertEqual(result['status'], 'verified')
        self.assertEqual(result['cell_count'], 60)
        self.assertEqual(result['pair_npz_count'], 15)
        self.assertTrue(result['artifact_sha256'])
        self.assertIn('run/macro.json', result['artifact_sha256'])

    def test_recomputed_macro_matches_recorded_macro(self):
        result = self.result()
        recorded = json.loads((self.raw_root / 'run' / 'macro.json').read_text())
        verifier.Checks().equal(result['recomputed_macro'], recorded['operators'], 'test macro')
        recorded_pair = json.loads((self.raw_root / 'run' / 'paired_summary.json').read_text())['macro']
        for name in recorded_pair:
            for bucket in BUCKETS:
                for field, expected in recorded_pair[name][bucket].items():
                    self.assertAlmostEqual(result['pair_macro'][name][bucket][field], expected, places=12)

    def test_returned_limitations_are_explicit(self):
        limitations = self.result()['limitations']
        self.assertGreaterEqual(len(limitations), 3)
        self.assertTrue(any('checkpoint' in item.lower() for item in limitations))
        self.assertTrue(any('not' in item.lower() for item in limitations))


class PlanDenominatorTests(VerifierFixture, unittest.TestCase):
    def test_full100_support_pool_excludes_all_queries(self):
        plan = json.loads((self.raw_root / 'inputs' / 'endpoint_plan.json').read_text())
        rows = json.loads((self.raw_root / 'inputs' / 'chembl' / 'manifest.json').read_text())['rows']
        pool = plan['chembl'][0]['support_max']
        self.assertEqual(len(pool), 100)
        self.assertEqual(len(plan['chembl'][0]['query']), ROWS_PER_DATABASE - 100)
        self.assertFalse(set(pool) & set(plan['chembl'][0]['query']))
        self.assertEqual(len({rows[i]['query_hash'] for i in pool}), 100)

    def test_query_denominator_uses_unique_hashes_not_rows(self):
        rows = database_rows('chembl')
        plan = build_plan('chembl', 0, rows, np.random.default_rng(7))
        rows.append({**rows[0], 'id': 'chembl/pretrain/131'})
        _, query = verifier.check_plan(plan, rows, 'chembl', 0, verifier.Checks())
        self.assertEqual(len(query), 30)
        plan['query'].append(130)
        with self.assertRaisesRegex(ValueError, 'Full query denominator'):
            verifier.check_plan(plan, rows, 'chembl', 0, verifier.Checks())

    def test_k20_only_query_exclusion_is_rejected(self):
        rows = database_rows('chembl')
        plan = build_plan('chembl', 0, rows, np.random.default_rng(7))
        plan['query'] = [i for i in range(len(rows)) if i not in plan['support_max'][:20]]
        with self.assertRaisesRegex(ValueError, 'Full query denominator'):
            verifier.check_plan(plan, rows, 'chembl', 0, verifier.Checks())

    def test_missing_cell_and_duplicate_cell_are_rejected(self):
        cells_path = self.raw_root / 'run' / 'cells.json'
        original = cells_path.read_text()
        for duplicate in (False, True):
            with self.subTest(duplicate=duplicate):
                payload = json.loads(original)
                if duplicate:
                    payload['cells'].append(dict(payload['cells'][0]))
                else:
                    payload['cells'].pop()
                cells_path.write_text(json.dumps(payload))
                with self.assertRaisesRegex(ValueError, 'cells:.*(incomplete|duplicate)'):
                    self.result()

    def test_support_prefix_and_digest_tampering_are_rejected(self):
        plan_path = self.raw_root / 'inputs' / 'endpoint_plan.json'
        plan = json.loads(plan_path.read_text())
        plan['chembl'][0]['supports']['5'] = plan['chembl'][0]['support_max'][1:6]
        plan_path.write_text(json.dumps(plan))
        with patch.object(verifier, 'PLAN_SHA256', verifier.sha256(plan_path)), \
                self.assertRaisesRegex(ValueError, 'Nested support prefixes'):
            verifier.check_plan(plan['chembl'][0], database_rows('chembl'), 'chembl', 0, verifier.Checks())

        plan = json.loads(plan_path.read_text())
        plan['chembl'][0]['supports']['5'] = plan['chembl'][0]['support_max'][:5]
        plan['chembl'][0]['query_digest'] = '0' * 64
        plan_path.write_text(json.dumps(plan))
        with self.assertRaisesRegex(ValueError, 'query digest'):
            verifier.check_plan(plan['chembl'][0], database_rows('chembl'), 'chembl', 0, verifier.Checks())

    def test_frozen_plan_sha_mismatch_is_rejected(self):
        plan_path = self.raw_root / 'inputs' / 'endpoint_plan.json'
        plan = json.loads(plan_path.read_text())
        plan['chembl'][0]['query_digest'] = '1' * 64
        plan_path.write_text(json.dumps(plan))
        inputs_path = self.raw_root / 'run' / 'inputs.json'
        inputs = json.loads(inputs_path.read_text())
        inputs['plan']['sha256'] = verifier.PLAN_SHA256
        inputs_path.write_text(json.dumps(inputs))
        with self.assertRaises(ValueError):
            self.result()

    def test_missing_pair_archive_is_rejected(self):
        (self.raw_root / 'run' / 'pair-chembl-run0.npz').unlink()
        with self.assertRaises(ValueError):
            self.result()


class NpzIntegrityTests(VerifierFixture, unittest.TestCase):
    def load_npz(self, name='pair-chembl-run0.npz'):
        path = self.raw_root / 'run' / name
        with np.load(path, allow_pickle=False) as archive:
            return {key: archive[key] for key in archive.files}

    def rewrite(self, values):
        path = self.raw_root / 'run' / 'pair-chembl-run0.npz'
        with path.open('wb') as stream:
            np.savez_compressed(stream, **values)

    def test_identity_and_mask_fields_match_source(self):
        values = self.load_npz()
        self.assertEqual(values['query_hash'].dtype, np.dtype('S64'))
        self.assertEqual(values['query_id'].dtype, np.dtype('S128'))
        self.assertEqual(values['true'].dtype, np.dtype('float64'))
        self.assertEqual(values['n_tables'].dtype, np.dtype('int16'))
        self.assertEqual(values['four_plus'].dtype, np.dtype('bool'))
        self.assertEqual(values['low4'].dtype, np.dtype('bool'))
        self.assertTrue(np.array_equal(values['four_plus'], values['n_tables'] >= 4))

    def test_identity_tampering_is_rejected(self):
        values = self.load_npz()
        values['query_id'][0] = b'chembl/pretrain/99999'
        self.rewrite(values)
        with self.assertRaises(ValueError):
            self.result()

    def test_mask_tampering_is_rejected(self):
        values = self.load_npz()
        values['low4'][0] = not bool(values['low4'][0])
        self.rewrite(values)
        with self.assertRaises(ValueError):
            self.result()

    def test_dtype_tampering_is_rejected(self):
        values = self.load_npz()
        values['n_tables'] = values['n_tables'].astype(np.int32)
        self.rewrite(values)
        with self.assertRaises(ValueError):
            self.result()

    def test_nonfinite_prediction_is_rejected(self):
        values = self.load_npz()
        values['none_log_pred'][3] = np.nan
        self.rewrite(values)
        with self.assertRaises(ValueError):
            self.result()

    def test_metric_tampering_is_rejected(self):
        cells_path = self.raw_root / 'run' / 'cells.json'
        cells = json.loads(cells_path.read_text())
        for cell in cells['cells']:
            if cell['operator'] == 'none' and cell['database'] == 'chembl':
                cell['metrics']['all']['q95'] += 0.5
        cells_path.write_text(json.dumps(cells))
        with self.assertRaises(ValueError):
            self.result()

    def test_paired_delta_tampering_is_rejected(self):
        values = self.load_npz()
        values['last3_ridge_mse_vs_none_delta_log_q'] = (
            values['last3_ridge_mse_vs_none_delta_log_q'] + 0.25)
        self.rewrite(values)
        with self.assertRaises(ValueError):
            self.result()


class ReportIntegrityTests(VerifierFixture, unittest.TestCase):
    def test_macro_tampering_is_rejected(self):
        path = self.raw_root / 'run' / 'macro.json'
        macro = json.loads(path.read_text())
        macro['operators']['none']['macro']['all_q95'] += 1.0
        path.write_text(json.dumps(macro))
        with self.assertRaises(ValueError):
            self.result()

    def test_csv_tampering_is_rejected(self):
        path = self.raw_root / 'run' / 'metrics.csv'
        text = path.read_text().splitlines()
        parts = text[1].split(',')
        parts[4] = repr(float(parts[4]) + 2.0)
        text[1] = ','.join(parts)
        path.write_text('\n'.join(text) + '\n')
        with self.assertRaises(ValueError):
            self.result()

    def test_paired_summary_tampering_is_rejected(self):
        path = self.raw_root / 'run' / 'paired_summary.json'
        paired = json.loads(path.read_text())
        paired['macro']['last3_ridge_mse_vs_none']['all']['arithmetic_mean_q95_ratio'] += 0.1
        path.write_text(json.dumps(paired))
        with self.assertRaises(ValueError):
            self.result()

    def test_missing_evidence_mirror_is_rejected(self):
        (self.evidence_root / 'macro.json').unlink()
        with self.assertRaises((OSError, ValueError)):
            self.result()

    def test_evidence_mirror_mismatch_is_rejected(self):
        path = self.evidence_root / 'macro.json'
        mirror = json.loads(path.read_text())
        mirror['operators']['none']['macro']['all_q50'] += 0.5
        path.write_text(json.dumps(mirror))
        with self.assertRaises(ValueError):
            self.result()

    def test_metrics_csv_text_equality_is_enforced(self):
        path = self.evidence_root / 'metrics.csv'
        path.write_text(path.read_text().replace('chembl', 'chemblX', 1))
        with self.assertRaises(ValueError):
            self.result()

    def test_status_flags_are_checked(self):
        path = self.raw_root / 'run' / 'status.json'
        status = json.loads(path.read_text())
        status['training'] = True
        path.write_text(json.dumps(status))
        with self.assertRaises(ValueError):
            self.result()

    def test_gate_flag_is_checked(self):
        path = self.raw_root / 'run' / 'status.json'
        status = json.loads(path.read_text())
        status['gate'] = 'selected'
        path.write_text(json.dumps(status))
        with self.assertRaises(ValueError):
            self.result()

    def test_environment_training_flag_is_checked(self):
        path = self.raw_root / 'run' / 'environment.json'
        environment = json.loads(path.read_text())
        environment['training'] = True
        path.write_text(json.dumps(environment))
        with self.assertRaises(ValueError):
            self.result()

    def test_completed_cell_count_is_checked(self):
        path = self.raw_root / 'run' / 'status.json'
        status = json.loads(path.read_text())
        status['completed_cells'] = 59
        path.write_text(json.dumps(status))
        with self.assertRaises(ValueError):
            self.result()


class InputBindingTests(VerifierFixture, unittest.TestCase):
    def test_source_sha_mismatch_is_rejected(self):
        path = self.raw_root / 'inputs' / 'chembl' / 'cards.npy'
        cards = np.load(path)
        cards[0, 0] = cards[0, 0] * 2
        np.save(path, cards)
        inputs_path = self.raw_root / 'run' / 'inputs.json'
        inputs = json.loads(inputs_path.read_text())
        inputs['source_before']['chembl']['file_sha256']['cards.npy'] = verifier.sha256(path)
        inputs['source_after']['chembl']['file_sha256']['cards.npy'] = verifier.sha256(path)
        inputs_path.write_text(json.dumps(inputs))
        with self.assertRaises(ValueError):
            self.result()

    def test_manifest_source_split_must_be_pretrain(self):
        path = self.raw_root / 'inputs' / 'chembl' / 'manifest.json'
        manifest = json.loads(path.read_text())
        manifest['provenance']['source']['split'] = 'test'
        path.write_text(json.dumps(manifest))
        inputs_path = self.raw_root / 'run' / 'inputs.json'
        inputs = json.loads(inputs_path.read_text())
        inputs['source_before']['chembl']['file_sha256']['manifest.json'] = verifier.sha256(path)
        inputs['source_after']['chembl']['file_sha256']['manifest.json'] = verifier.sha256(path)
        inputs_path.write_text(json.dumps(inputs))
        with self.assertRaises(ValueError):
            self.result()

    def test_excluded_rows_must_be_empty(self):
        path = self.raw_root / 'inputs' / 'chembl' / 'manifest.json'
        manifest = json.loads(path.read_text())
        manifest['provenance']['excluded'] = ['chembl/pretrain/1']
        path.write_text(json.dumps(manifest))
        inputs_path = self.raw_root / 'run' / 'inputs.json'
        inputs = json.loads(inputs_path.read_text())
        inputs['source_before']['chembl']['file_sha256']['manifest.json'] = verifier.sha256(path)
        inputs['source_after']['chembl']['file_sha256']['manifest.json'] = verifier.sha256(path)
        inputs_path.write_text(json.dumps(inputs))
        with self.assertRaises(ValueError):
            self.result()

    def test_target_database_is_rejected(self):
        plan_path = self.raw_root / 'inputs' / 'endpoint_plan.json'
        plan = json.loads(plan_path.read_text())
        plan['imdb'] = plan.pop('telstra')
        plan_path.write_text(json.dumps(plan))
        new_sha = verifier.sha256(plan_path)
        inputs_path = self.raw_root / 'run/inputs.json'
        inputs = json.loads(inputs_path.read_text())
        inputs['plan']['sha256'] = new_sha
        inputs_path.write_text(json.dumps(inputs))
        with patch.object(verifier, 'PLAN_SHA256', new_sha), \
                self.assertRaisesRegex(ValueError, 'Target or unknown plan database'):
            self.result()

    def test_checkpoint_metadata_split_and_adaptation(self):
        path = self.raw_root / 'inputs' / 'checkpoint-metadata.json'
        original = path.read_text()
        cases = ((('metadata', 'train_databases'), [f'train{i}' for i in range(22)], 'split'),
                 (('metadata', 'adapted_database'), 'chembl', 'split/adaptation'),
                 (('metadata', 'train_databases'), ['imdb'] + [f'train{i}' for i in range(22)], 'split'),
                 (('config', 'model', 'width'), 32, 'checkpoint model'),
                 (('last_step',), 3999, 'checkpoint step'))
        for keys, value, message in cases:
            with self.subTest(keys=keys, value=value):
                metadata = json.loads(original)
                target = metadata
                for key in keys[:-1]:
                    target = target[key]
                target[keys[-1]] = value
                path.write_text(json.dumps(metadata))
                with patch.object(verifier, 'METADATA_SHA256', verifier.sha256(path)), \
                        self.assertRaisesRegex(ValueError, message):
                    self.result()

    def test_metadata_bytes_are_bound_even_if_checked_fields_unchanged(self):
        path = self.raw_root / 'inputs/checkpoint-metadata.json'
        metadata = json.loads(path.read_text())
        metadata['arm'] = 'Q_other'
        path.write_text(json.dumps(metadata))
        with self.assertRaisesRegex(ValueError, 'Frozen metadata SHA256 mismatch'):
            self.result()

    def test_checkpoint_metadata_sha_is_checked(self):
        path = self.raw_root / 'inputs' / 'checkpoint-metadata.json'
        metadata = json.loads(path.read_text())
        metadata['checkpoint_sha256'] = 'b' * 64
        path.write_text(json.dumps(metadata))
        with self.assertRaises(ValueError):
            self.result()

    def test_input_bindings_must_match_frozen_constants(self):
        path = self.raw_root / 'run' / 'inputs.json'
        inputs = json.loads(path.read_text())
        inputs['lambda_registry']['sha256'] = 'c' * 64
        path.write_text(json.dumps(inputs))
        with self.assertRaises(ValueError):
            self.result()

    def test_source_before_after_must_agree(self):
        path = self.raw_root / 'run' / 'inputs.json'
        inputs = json.loads(path.read_text())
        inputs['source_after']['chembl']['manifest_content_hash'] = 'd' * 64
        path.write_text(json.dumps(inputs))
        with self.assertRaises(ValueError):
            self.result()


class SupportAuditTests(VerifierFixture, unittest.TestCase):
    def test_support_targets_are_recomputed_from_cards(self):
        cards = np.load(self.raw_root / 'inputs' / 'chembl' / 'cards.npy')
        cells = json.loads((self.raw_root / 'run' / 'cells.json').read_text())['cells']
        cell = next(item for item in cells if item['operator'] == 'none'
                    and item['database'] == 'chembl' and item['run'] == 0)
        plan = json.loads((self.raw_root / 'inputs' / 'endpoint_plan.json').read_text())
        support = plan['chembl'][0]['support_max'][:20]
        expected = support_targets(cards, support)
        recorded = [row['target'] for row in cell['support_audit']['support_residuals']]
        self.assertEqual(len(recorded), 20)
        np.testing.assert_allclose(recorded, expected, rtol=0, atol=1e-12)

    def test_target_tampering_is_rejected(self):
        cells_path = self.raw_root / 'run' / 'cells.json'
        cells = json.loads(cells_path.read_text())
        for cell in cells['cells']:
            if cell['operator'] == 'none' and cell['database'] == 'chembl':
                cell['support_audit']['support_residuals'][0]['target'] += 0.75
        cells_path.write_text(json.dumps(cells))
        with self.assertRaises(ValueError):
            self.result()

    def test_method_and_changed_parameters_are_enforced(self):
        cells_path = self.raw_root / 'run' / 'cells.json'
        cells = json.loads(cells_path.read_text())
        for cell in cells['cells']:
            if cell['operator'] == 'none' and cell['database'] == 'legalacts':
                cell['support_audit']['method'] = 'gradient_last'
                cell['support_audit']['changed_parameter_names'] = ['head.3.weight']
        cells_path.write_text(json.dumps(cells))
        with self.assertRaises(ValueError):
            self.result()

    def test_none_operator_must_not_change_parameters(self):
        cells_path = self.raw_root / 'run' / 'cells.json'
        cells = json.loads(cells_path.read_text())
        for cell in cells['cells']:
            if cell['operator'] == 'none' and cell['database'] == 'telstra':
                cell['support_audit']['changed_parameter_names'] = ['head.4.weight']
                cell['support_audit']['update_norm'] = 0.2
                cell['support_audit']['support_residuals'][0]['after'] += 0.1
                cell['support_audit']['support_mse_after'] += 0.1
        cells_path.write_text(json.dumps(cells))
        with self.assertRaises(ValueError):
            self.result()

    def test_residual_and_mse_recomputation(self):
        cells_path = self.raw_root / 'run' / 'cells.json'
        cells = json.loads(cells_path.read_text())
        for cell in cells['cells']:
            audit = cell['support_audit']
            for row in audit['support_residuals']:
                self.assertAlmostEqual(row['residual_before'],
                                       row['before'] - row['target'], places=9)
                self.assertAlmostEqual(row['residual_after'],
                                       row['after'] - row['target'], places=9)
            residual_before = np.array([row['residual_before']
                                        for row in audit['support_residuals']])
            residual_after = np.array([row['residual_after']
                                       for row in audit['support_residuals']])
            self.assertAlmostEqual(audit['support_mse_before'],
                                   float(np.mean(residual_before ** 2)), places=9)
            self.assertAlmostEqual(audit['support_mse_after'],
                                   float(np.mean(residual_after ** 2)), places=9)
            self.assertEqual(audit['support_residual_improved'],
                             audit['support_mse_after'] < audit['support_mse_before'])

    def test_ridge_operator_requires_solver_fields(self):
        cells_path = self.raw_root / 'run' / 'cells.json'
        cells = json.loads(cells_path.read_text())
        for cell in cells['cells']:
            if cell['operator'] == OPERATORS[3] and cell['database'] == 'chembl':
                del cell['support_audit']['solver_condition_number']
        cells_path.write_text(json.dumps(cells))
        with self.assertRaises(ValueError):
            self.result()

    def test_ridge_lambda_must_match_registry(self):
        cells_path = self.raw_root / 'run' / 'cells.json'
        cells = json.loads(cells_path.read_text())
        for cell in cells['cells']:
            if cell['operator'] == OPERATORS[3]:
                cell['support_audit']['proximal_lambda'] = 0.5
        cells_path.write_text(json.dumps(cells))
        with self.assertRaises(ValueError):
            self.result()

    def test_state_digests_are_consistent(self):
        cells_path = self.raw_root / 'run' / 'cells.json'
        cells = json.loads(cells_path.read_text())
        for cell in cells['cells']:
            if cell['database'] == 'legalacts':
                cell['state_before'] = 'e' * 64
        cells_path.write_text(json.dumps(cells))
        with self.assertRaises(ValueError):
            self.result()


class CliTests(VerifierFixture, unittest.TestCase):
    def run_cli(self, *arguments):
        import contextlib
        import io
        from types import SimpleNamespace

        stdout, stderr = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            try:
                code = verifier.main(list(arguments))
            except SystemExit as error:
                code = error.code
        return SimpleNamespace(returncode=code, stdout=stdout.getvalue(), stderr=stderr.getvalue())

    def test_cli_writes_only_new_output_and_returns_zero(self):
        output = self.root / 'out' / 'verification.json'
        output.parent.mkdir()
        result = self.run_cli('--raw-root', str(self.raw_root),
                              '--evidence-root', str(self.evidence_root),
                              '--output', str(output))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(output.is_file())
        payload = json.loads(output.read_text())
        self.assertEqual(payload['status'], 'verified')
        self.assertEqual(payload['cell_count'], 60)

    def test_cli_rejects_existing_output(self):
        output = self.root / 'out' / 'verification.json'
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text('{}\n')
        result = self.run_cli('--raw-root', str(self.raw_root),
                              '--evidence-root', str(self.evidence_root),
                              '--output', str(output))
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(output.read_text(), '{}\n')

    def test_cli_rejects_output_inside_raw_root(self):
        output = self.raw_root / 'run' / 'verification.json'
        result = self.run_cli('--raw-root', str(self.raw_root),
                              '--evidence-root', str(self.evidence_root),
                              '--output', str(output))
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(output.exists())

    def test_cli_failure_returns_nonzero(self):
        (self.raw_root / 'run' / 'pair-telstra-run4.npz').unlink()
        output = self.root / 'out' / 'verification.json'
        result = self.run_cli('--raw-root', str(self.raw_root),
                              '--evidence-root', str(self.evidence_root),
                              '--output', str(output))
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('verification failed', result.stderr)
        self.assertFalse(output.exists())


class EvidenceDefaultTests(unittest.TestCase):
    def test_default_evidence_resolves_to_repository_mirror(self):
        self.assertEqual(verifier.DEFAULT_EVIDENCE,
                         ROOT / 'reports' / 'meta65-lowdim-gate1-20261018')


if __name__ == '__main__':
    unittest.main()
