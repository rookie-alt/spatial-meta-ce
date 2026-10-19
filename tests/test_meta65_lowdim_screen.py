import json
import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from scripts import run_meta65_lowdim_screen as screen
from spatial_meta_ce.data import batch_samples
from spatial_meta_ce.model import SetEstimator

ROOT = Path(__file__).resolve().parents[1]


def feature_values(i, rows, columns, scale, offset=0.0):
    generator = np.random.default_rng(1000 + i * 10 + int(offset * 10))
    return scale * (
        np.abs(generator.standard_normal(
            (rows, columns), dtype=np.float32
        ))
        + 0.1
    )


def sample(i, y=100.0):
    return {
        'groups': [
            feature_values(i, 2, 40, 0.1, 0.0),
            feature_values(i, 1, 40, 0.2, 0.5),
            feature_values(i, 3, 4, 0.3, 1.0),
            feature_values(i, 1, 43, 0.4, 1.5),
        ],
        'true': y,
        'pg': 2.0,
        'id': f'chembl/pretrain/{i + 1}',
    }


def rows(n=130):
    return [
        {'query_hash': f'h{i:03d}', 'id': f'chembl/pretrain/{i + 1}'}
        for i in range(n)
    ]


def plan_for(n=130, database='chembl', run=0):
    source_rows = rows(n)
    support = list(range(100))
    query = list(range(100, n))
    hashes = [row['query_hash'] for row in source_rows]
    return {
        'database': database,
        'run': run,
        'support_max': support,
        'supports': {str(k): support[:k] for k in (5, 10, 20, 50, 100)},
        'query': query,
        'query_digest': screen.digest([(i, hashes[i]) for i in query]),
        'support_digests': {
            str(k): screen.digest([(i, hashes[i]) for i in support[:k]])
            for k in (5, 10, 20, 50, 100)
        },
    }


def metric(count, q50, q95):
    return {'count': count, 'status': 'ok', 'q50': q50, 'q95': q95}


class ContractTests(unittest.TestCase):
    def test_operator_set_and_attribution_boundaries_are_explicit(self):
        self.assertEqual(screen.OPERATORS, (
            'none',
            'last3_gradient_smooth_l1',
            'last3_gradient_mse',
            'last3_ridge_mse',
        ))
        contrasts = {item['name']: item for item in screen.CONTRASTS}
        self.assertEqual(
            (contrasts['ridge_vs_gradient_mse']['left'],
             contrasts['ridge_vs_gradient_mse']['right']),
            ('last3_ridge_mse', 'last3_gradient_mse'))
        self.assertEqual(
            (contrasts['loss_within_gradient']['left'],
             contrasts['loss_within_gradient']['right']),
            ('last3_gradient_mse', 'last3_gradient_smooth_l1'))
        self.assertEqual(
            (contrasts['combined_solver_and_loss']['left'],
             contrasts['combined_solver_and_loss']['right']),
            ('last3_ridge_mse', 'last3_gradient_smooth_l1'))
        self.assertIn('not solver-only', contrasts['combined_solver_and_loss']['description'])
        self.assertEqual(
            contrasts['ridge_vs_gradient_mse']['attribution'],
            'solver_and_proximal_regularization',
        )

    def test_lambda_registry_is_bounded_and_must_be_preregistered(self):
        valid = {
            'schema': 1,
            'experiment': screen.EXPERIMENT,
            'candidate_lambdas': [0.01, 0.1, 1.0],
            'selected_lambda': 0.1,
            'selection_timing': 'before_query_evaluation',
        }
        self.assertEqual(screen.validate_lambda_registry(valid)['selected_lambda'], 0.1)
        invalid = [
            {**valid, 'selected_lambda': 0.2},
            {**valid, 'candidate_lambdas': []},
            {**valid, 'candidate_lambdas': [0.1] * 6},
            {**valid, 'candidate_lambdas': [0.1, 0.1]},
            {**valid, 'selected_lambda': True},
            {**valid, 'selection_timing': 'after_query_evaluation'},
            {**valid, 'unknown': 1},
        ]
        for value in invalid:
            with self.subTest(value=value), self.assertRaises(ValueError):
                screen.validate_lambda_registry(value)

    def test_plan_keeps_full_100_query_exclusion_and_k20_prefix(self):
        source_rows = rows()
        plan = plan_for()
        certified = screen.certify_plan(
            plan, source_rows, 'chembl', 0, expected_unique=130)
        self.assertEqual(certified['support'], list(range(20)))
        self.assertEqual(certified['query'], list(range(100, 130)))
        self.assertFalse(set(certified['support']) & set(certified['query']))
        for mutation in (
                lambda value: value['query'].append(99),
                lambda value: value['supports'].update({'20': list(range(1, 21))}),
                lambda value: value.update(query_digest='bad'),
                lambda value: value.update(run=1)):
            changed = json.loads(json.dumps(plan))
            mutation(changed)
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                screen.certify_plan(
                    changed, source_rows, 'chembl', 0, expected_unique=130)
        with self.assertRaises(ValueError):
            screen.certify_plan(
                {**plan, 'database': 'imdb'}, source_rows, 'imdb', 0,
                expected_unique=130)


class AdaptationTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(42)
        self.model = SetEstimator(arch='sum', width=64)
        with torch.no_grad():
            nn.init.normal_(self.model.head[-1].weight, std=0.01)
            nn.init.normal_(self.model.head[-1].bias, std=0.01)
        self.support = batch_samples([sample(i) for i in range(20)])
        self.query = batch_samples([sample(100 + i, 200.0) for i in range(7)])

    def test_four_operators_share_initial_state_and_audit_support(self):
        before = screen.state_digest(self.model)
        for operator in screen.OPERATORS:
            with self.subTest(operator=operator):
                params, audit = screen.adaptation_params(
                    self.model, self.support, operator, ridge_lambda=0.5)
                self.assertEqual(screen.state_digest(self.model), before)
                changed = {
                    name for name, value in self.model.named_parameters()
                    if not torch.equal(value, params[name])
                }
                if operator == 'none':
                    self.assertEqual(changed, set())
                else:
                    self.assertEqual(
                        changed, {'head.4.weight', 'head.4.bias'})
                self.assertEqual(audit['operator'], operator)
                self.assertEqual(audit['support_count'], 20)
                self.assertEqual(audit['feature_dim'], 64)
                self.assertEqual(len(audit['support_residuals']), 20)
                self.assertTrue(math.isfinite(audit['update_norm']))
                self.assertTrue(math.isfinite(audit['condition_number']))
                self.assertGreater(audit['condition_number'], 0)
                self.assertTrue(math.isfinite(audit['support_mse_before']))
                self.assertTrue(math.isfinite(audit['support_mse_after']))
                if operator == 'none':
                    self.assertFalse(audit['support_residual_improved'])
                elif operator == 'last3_ridge_mse':
                    self.assertTrue(audit['support_residual_improved'])
                if operator == 'last3_ridge_mse':
                    self.assertTrue(math.isfinite(
                        audit['solver_condition_number']))
                    self.assertGreater(audit['solver_condition_number'], 0)

    def test_ridge_stops_before_query_when_support_residual_does_not_improve(self):
        class GuardStore:
            def __init__(self):
                self.rows = rows()
                self.query_loaded = False

            def batch(self, indices, device='cpu'):
                if any(int(index) >= 100 for index in indices):
                    self.query_loaded = True
                return batch_samples([
                    sample(int(index), 50.0 if int(index) >= 100 else 100.0)
                    for index in indices
                ], device)

        store = GuardStore()
        table_counts = np.array([3] * 100 + [4] * 30, dtype=np.int16)
        with self.assertRaisesRegex(ValueError, 'support residual'):
            screen.evaluate_database_run(
                self.model, store, 'chembl', 0,
                support_indices=list(range(20)),
                query_indices=list(range(100, 107)),
                table_counts=table_counts,
                ridge_lambda=1e20,
                device='cpu',
                batch_size=4,
            )
        self.assertFalse(store.query_loaded)

    def test_query_labels_are_loaded_only_after_all_adaptations(self):
        class GuardStore:
            def __init__(self):
                self.rows = rows()
                self.query_loaded = False

            def batch(self, indices, device='cpu'):
                if any(int(index) >= 100 for index in indices):
                    self.query_loaded = True
                return batch_samples([
                    sample(int(index), 50.0 if int(index) >= 100 else 100.0)
                    for index in indices
                ], device)

        store = GuardStore()
        original = screen.adaptation_params

        def guarded(model, support, operator, ridge_lambda):
            self.assertFalse(store.query_loaded)
            self.assertTrue(all(support['ids'][i].endswith(
                f'/{i + 1}') for i in range(20)))
            return original(model, support, operator, ridge_lambda)

        table_counts = np.array([3] * 100 + [4] * 30, dtype=np.int16)
        with patch.object(screen, 'adaptation_params', side_effect=guarded):
            result = screen.evaluate_database_run(
                self.model, store, 'chembl', 0,
                support_indices=list(range(20)),
                query_indices=list(range(100, 107)),
                table_counts=table_counts,
                ridge_lambda=0.5,
                device='cpu',
                batch_size=4,
            )
        self.assertTrue(store.query_loaded)
        self.assertEqual(set(result['operators']), set(screen.OPERATORS))
        self.assertEqual(
            set(result['identity']),
            {'query_hash', 'query_id', 'true', 'n_tables'})
        for operator in screen.OPERATORS:
            self.assertEqual(len(result['operators'][operator]['log_pred']), 7)
            self.assertEqual(
                set(result['operators'][operator]['metrics']),
                {'all', 'four_plus', 'low4'})

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'pair.npz'
            screen.write_pair_archive(path, result)
            with np.load(path, allow_pickle=False) as archive:
                self.assertIn('query_hash', archive.files)
                self.assertIn('query_id', archive.files)
                self.assertIn('true', archive.files)
                self.assertIn('n_tables', archive.files)
                self.assertIn('four_plus', archive.files)
                self.assertIn('low4', archive.files)
                for operator in screen.OPERATORS:
                    self.assertIn(f'{operator}_log_pred', archive.files)
                    self.assertIn(f'{operator}_q', archive.files)
                for pairing in screen.PAIRINGS:
                    self.assertIn(f"{pairing['name']}_delta_log_q", archive.files)
                    self.assertIn(f"{pairing['name']}_q_ratio", archive.files)

    def test_lambda_registry_hash_is_bound_to_the_bytes_that_were_parsed(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'lambda.json'
            value = {
                'schema': 1,
                'experiment': screen.EXPERIMENT,
                'candidate_lambdas': [0.5],
                'selected_lambda': 0.5,
                'selection_timing': 'before_query_evaluation',
            }
            path.write_text(json.dumps(value))
            _, bound_sha = screen.load_lambda_registry(path)
            self.assertEqual(bound_sha, screen.file_hash(path))
            path.write_text(json.dumps({**value, 'selected_lambda': 2.0,
                                        'candidate_lambdas': [2.0]}))
            self.assertNotEqual(bound_sha, screen.file_hash(path))


class AggregationTests(unittest.TestCase):
    def test_five_run_macro_is_unweighted_over_runs_then_databases(self):
        cells = []
        for operator in screen.OPERATORS:
            for db_index, database in enumerate(screen.DATABASES):
                for run in range(5):
                    q95 = 10.0 * (db_index + 1) + run
                    count = (100, 10, 1)[db_index]
                    cells.append({
                        'operator': operator,
                        'database': database,
                        'run': run,
                        'query_digest': f'{database}-{run}',
                        'support_digest': f'{database}-{run}',
                        'metrics': {
                            'all': metric(count, q95, q95),
                            'four_plus': metric(count, q95, q95),
                            'low4': metric(count, q95, q95),
                        },
                        'support_audit': {},
                    })
        summary = screen.summarize_cells(cells)
        self.assertAlmostEqual(
            summary['operators']['none']['macro']['four_plus_q95'], 22.0)
        self.assertAlmostEqual(
            summary['operators']['none']['per_database']['chembl']['four_plus_q95'],
            12.0)
        self.assertAlmostEqual(
            summary['operators']['none']['per_database']['legalacts']['four_plus_q95'],
            22.0)
        self.assertAlmostEqual(
            summary['operators']['none']['per_database']['telstra']['four_plus_q95'],
            32.0)

        broken = json.loads(json.dumps(cells))
        broken[1]['query_digest'] = 'changed'
        with self.assertRaises(ValueError):
            screen.summarize_cells(broken)
        with self.assertRaises(ValueError):
            screen.summarize_cells(cells[:-1])

    def test_pairwise_output_keeps_query_level_identity(self):
        query_hash = np.asarray([f'h{i}' for i in range(6)])
        query_id = np.asarray([f'chembl/pretrain/{i + 1}' for i in range(6)])
        true = np.asarray([10.0, 20.0, 30.0, 40.0, 50.0, 60.0])
        tables = np.asarray([4, 4, 4, 3, 3, 4], dtype=np.int16)
        base = np.log(true)
        log_pred = {
            'none': base + np.asarray([0.3, -0.2, 0.1, 0.4, -0.3, 0.2]),
            'last3_gradient_smooth_l1': base + np.asarray([0.2, -0.1, 0.1, 0.3, -0.2, 0.1]),
            'last3_gradient_mse': base + np.asarray([0.1, -0.1, 0.0, 0.2, -0.1, 0.1]),
            'last3_ridge_mse': base + np.asarray([0.0, 0.0, 0.0, 0.1, 0.0, 0.0]),
        }
        entries = [{
            'database': database,
            'run': run,
            'identity': {
                'query_hash': query_hash,
                'query_id': query_id,
                'true': true,
                'n_tables': tables,
            },
            'log_pred': log_pred,
        } for database in screen.DATABASES for run in screen.RUNS]
        result = screen.summarize_pairings(entries)
        contrasts = {item['name']: item for item in result['contrasts']}
        self.assertEqual(
            contrasts['ridge_vs_gradient_mse']['attribution'],
            'solver_and_proximal_regularization',
        )
        self.assertEqual(
            contrasts['loss_within_gradient']['attribution'], 'loss')
        self.assertEqual(
            contrasts['combined_solver_and_loss']['attribution'],
            'combined_not_solver_only')
        for bucket in ('all', 'four_plus', 'low4'):
            stats = result['runs'][0]['pairs']['ridge_vs_gradient_mse'][bucket]
            self.assertEqual(
                stats['query_count'],
                {'all': 6, 'four_plus': 4, 'low4': 4}[bucket],
            )
            self.assertTrue(math.isfinite(stats['mean_delta_log_q']))
            self.assertTrue(math.isfinite(stats['q95_ratio']))


class SafetyTests(unittest.TestCase):
    def test_metadata_validation_does_not_touch_cards_or_features(self):
        class MetadataOnlyStore:
            def __init__(self):
                self.path = Path('/private/tmp/source/pretrain/chembl')
                self.rows = rows()
                self.offsets = np.zeros((131, 4), dtype=np.int64)
                self.offsets[:, 2] = np.arange(131) * 4
                self.manifest = {'unique_queries': 130}

            def __len__(self):
                return len(self.rows)

            def unique_indices(self):
                return np.arange(len(self.rows), dtype=np.int64)

            @property
            def cards(self):
                raise AssertionError('cards read before adaptation')

            @property
            def groups(self):
                raise AssertionError('features read before adaptation')

        with patch.dict(screen.UNIQUE, {'chembl': 130}):
            unique, tables = screen.validate_source_metadata(
                MetadataOnlyStore(), 'chembl')
        self.assertEqual(len(unique), 130)
        self.assertTrue(np.array_equal(tables, np.full(130, 4)))

    def test_cache_database_and_file_symlinks_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            cache_root = root / 'cache'
            pretrain = cache_root / 'pretrain'
            pretrain.mkdir(parents=True)
            outside = root / 'outside'
            outside.mkdir()
            (pretrain / 'chembl').symlink_to(
                outside, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, 'symlink'):
                screen.validate_source_cache_path(cache_root, 'chembl')

            (pretrain / 'chembl').unlink()
            database_dir = pretrain / 'chembl'
            database_dir.mkdir()
            for name in screen.CACHE_FILES:
                path = database_dir / name
                if name == 'manifest.json':
                    path.write_text('{}')
                else:
                    path.write_bytes(b'x')
            (database_dir / 'group0.npy').unlink()
            (database_dir / 'group0.npy').symlink_to(outside / 'payload.npy')
            with self.assertRaisesRegex(ValueError, 'symlink'):
                screen.validate_source_cache_path(cache_root, 'chembl')

    def test_synthetic_execute_writes_all_cells_and_paired_artifacts(self):
        from spatial_meta_ce.checkpoints import save_checkpoint
        from spatial_meta_ce.features import write_cache

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            cache_root = root / 'cache'
            cache_rows = {}
            for database in screen.DATABASES:
                source_rows = []
                features = []
                for index in range(130):
                    value = 100.0 if index < 100 else 50.0
                    source_rows.append({
                        'id': f'{database}/pretrain/{index + 1}',
                        'query_hash': f'{database}-h{index:03d}',
                        'true': value,
                        'pg': 2.0,
                    })
                    features.append(tuple(
                        torch.from_numpy(feature_values(
                            index, rows, columns, scale, offset
                        ))
                        for rows, columns, scale, offset in (
                            (1, 40, 0.1, 0.0),
                            (1, 40, 0.2, 0.5),
                            (4, 4, 0.3, 1.0),
                            (1, 43, 0.4, 1.5),
                        )
                    ))
                cache_rows[database] = source_rows
                write_cache(
                    cache_root / 'pretrain' / database,
                    source_rows,
                    {
                        'data_features': features,
                        'true_cards': [torch.tensor(row['true']) for row in source_rows],
                        'pg_est_cards': [torch.tensor(row['pg']) for row in source_rows],
                        'n_join_cols': [1] * 130,
                        'n_fanouts': [1] * 130,
                        'n_tables': [4] * 130,
                        'n_filter_cols': [1] * 130,
                    },
                    {'source': 'synthetic'},
                )
            plan = {}
            for database in screen.DATABASES:
                source_rows = cache_rows[database]
                hashes = [row['query_hash'] for row in source_rows]
                support = list(range(100))
                query = list(range(100, 130))
                plan[database] = [{
                    'database': database,
                    'run': run,
                    'support_max': support,
                    'supports': {
                        str(k): support[:k] for k in screen.SHOTS
                    },
                    'query': query,
                    'query_digest': screen.digest(
                        [(i, hashes[i]) for i in query]),
                    'support_digests': {
                        str(k): screen.digest(
                            [(i, hashes[i]) for i in support[:k]])
                        for k in screen.SHOTS
                    },
                } for run in screen.RUNS]
            plan_path = root / 'plan.json'
            plan_path.write_text(json.dumps(plan))
            registry_path = root / 'lambda.json'
            registry_path.write_text(json.dumps({
                'schema': 1,
                'experiment': screen.EXPERIMENT,
                'candidate_lambdas': [0.5],
                'selected_lambda': 0.5,
                'selection_timing': 'before_query_evaluation',
            }))
            checkpoint_path = root / 'last.pt'
            model = SetEstimator(arch='sum', width=64)
            save_checkpoint(
                checkpoint_path,
                model,
                None,
                4000,
                1.0,
                0,
                {
                    'seed': 42,
                    'train_databases': [],
                    'validation_databases': list(screen.DATABASES),
                },
                {},
            )
            with patch.dict(
                screen.UNIQUE,
                {database: 130 for database in screen.DATABASES},
            ):
                result = screen.execute(
                    checkpoint=checkpoint_path,
                    checkpoint_sha256=screen.file_hash(checkpoint_path),
                    plan_path=plan_path,
                    plan_sha256=screen.file_hash(plan_path),
                    cache_root=cache_root,
                    lambda_registry_path=registry_path,
                    output=root / 'runs' / 'screen',
                    device='cpu',
                    root=root,
                )
            self.assertEqual(result['status'], 'completed')
            self.assertEqual(result['cells'], 60)
            output = root / 'runs' / 'screen'
            self.assertTrue((output / 'status.json').exists())
            self.assertTrue((output / 'macro.json').exists())
            self.assertTrue((output / 'paired_summary.json').exists())
            self.assertTrue((output / 'metrics.csv').exists())
            self.assertEqual(
                len(list(output.glob('pair-*.npz'))), 15)

            replacement = {
                'schema': 1,
                'experiment': screen.EXPERIMENT,
                'candidate_lambdas': [2.0],
                'selected_lambda': 2.0,
                'selection_timing': 'before_query_evaluation',
            }
            original_loader = screen.load_lambda_registry

            def load_then_replace(path):
                registry, bound_sha = original_loader(path)
                Path(path).write_text(json.dumps(replacement))
                return registry, bound_sha

            with patch.object(
                screen, 'load_lambda_registry', side_effect=load_then_replace
            ), patch.dict(
                screen.UNIQUE,
                {database: 130 for database in screen.DATABASES},
            ):
                with self.assertRaisesRegex(ValueError, 'Lambda registry changed'):
                    screen.execute(
                        checkpoint=checkpoint_path,
                        checkpoint_sha256=screen.file_hash(checkpoint_path),
                        plan_path=plan_path,
                        plan_sha256=screen.file_hash(plan_path),
                        cache_root=cache_root,
                        lambda_registry_path=registry_path,
                        output=root / 'runs' / 'swapped',
                        device='cpu',
                        root=root,
                    )
            failed = json.loads((root / 'runs' / 'swapped/status.json').read_text())
            self.assertEqual(failed['status'], 'failed')

    def test_output_must_be_new_and_under_runs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            (root / 'runs').mkdir()
            (root / 'used').mkdir()
            with self.assertRaises(ValueError):
                screen.safe_output(root / 'used' / 'screen', root=root)
            with self.assertRaises(ValueError):
                screen.safe_output(root / 'runs' / 'imdb' / 'screen', root=root)
            self.assertEqual(
                screen.safe_output(root / 'runs' / 'screen', root=root),
                (root / 'runs' / 'screen').resolve())

    def test_dry_run_needs_no_dependencies_checkpoint_or_output(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'does-not-exist'
            result = subprocess.run(
                [sys.executable, '-S', '-B',
                 str(ROOT / 'scripts/run_meta65_lowdim_screen.py'),
                 '--out', str(output)],
                capture_output=True,
                text=True,
                check=True,
            )
            report = json.loads(result.stdout)
            self.assertEqual(report['purpose'], 'dry-run-not-execution')
            self.assertEqual(report['operators'], list(screen.OPERATORS))
            self.assertTrue(report['lambda_registry_required'])
            self.assertFalse(output.exists())


if __name__ == '__main__':
    unittest.main()
