import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

AVAILABLE = importlib.util.find_spec('spatial_meta_ce.manifests') is not None
if AVAILABLE:
    from spatial_meta_ce.data import FeatureStore
    from spatial_meta_ce.features import verify_alignment, write_cache
    from spatial_meta_ce.manifests import (
        canonical_sql,
        episode_indices,
        validate_splits,
    )


class DataTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue(AVAILABLE, 'Data preparation has not been implemented')

    def test_database_overlap_rejected(self):
        with self.assertRaises(ValueError):
            validate_splits({'train': ['a'], 'validation': ['a'], 'test': ['b']})

    @unittest.skipUnless(importlib.util.find_spec('sqlglot'), 'SQLGlot checked on server environment')
    def test_canonical_preserves_literals(self):
        self.assertEqual(canonical_sql('SELECT * FROM t WHERE x = 3;'), canonical_sql('select *  from t where x=3'))
        self.assertNotEqual(canonical_sql("select * from t where x='A'"), canonical_sql("select * from t where x='a'"))
        self.assertNotEqual(canonical_sql('select * from t where x=3'), canonical_sql('select * from t where x=4'))

    def test_disjoint_reproducible_episode(self):
        a, b = episode_indices(np.arange(100), 5, 20, np.random.default_rng(4))
        c, d = episode_indices(np.arange(100), 5, 20, np.random.default_rng(4))
        self.assertFalse(set(a) & set(b))
        self.assertTrue(np.array_equal(a, c))
        self.assertTrue(np.array_equal(b, d))
        with self.assertRaises(ValueError):
            episode_indices(np.arange(3), 5, 20, np.random.default_rng(4))

    def test_misaligned_labels_fail(self):
        rows = [{'true': 5.0, 'pg': 2.0}]
        with self.assertRaises(ValueError):
            verify_alignment(rows, {'true_cards': [torch.tensor(6.)], 'pg_est_cards': [torch.tensor(2.)]})

    def test_cache_roundtrip_and_dedup(self):
        with tempfile.TemporaryDirectory() as d:
            rows = [{'id': str(i), 'query_hash': 'a' if i < 2 else 'b', 'true': 10., 'pg': 2.} for i in range(3)]
            data = {'data_features': [(torch.ones(80), torch.ones(80), torch.ones(8), None) for _ in rows],
                    'true_cards': [torch.tensor(10.) for _ in rows], 'pg_est_cards': [torch.tensor(2.) for _ in rows],
                    'n_join_cols': [2]*3, 'n_fanouts': [2]*3, 'n_tables': [2]*3, 'n_filter_cols': [0]*3}
            path = Path(d)/'cache'
            write_cache(path, rows, data, {'source': 'fixture'})
            store = FeatureStore(path)
            self.assertEqual(len(store), 3)
            self.assertEqual(store[0]['groups'][3].shape, (0, 43))
            self.assertEqual(store.unique_indices().tolist(), [0, 2])
            self.assertEqual(store.unique_indices(exclude=['a']).tolist(), [2])
            with self.assertRaises(FileExistsError):
                write_cache(path, rows, data, {'source': 'fixture'})

    def test_cache_manifest_tamper_rejected(self):
        import json

        from test_protocol import fixtures
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            _cfg,dc=fixtures(root)
            path=Path(dc['cache_root'])/'pretrain/a'
            f=path/'manifest.json'
            manifest=json.loads(f.read_text())
            manifest['rows'][0]['query_hash']='tampered'
            f.write_text(json.dumps(manifest))
            with self.assertRaises(ValueError):
                FeatureStore(path)

    def test_nonfinite_cache_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            rows = [{'id': '0','query_hash': 'a','true': 1.,'pg': 1.}]
            data = {'data_features': [(torch.full((40,), float('nan')), torch.ones(40), torch.ones(4), None)],
                    'true_cards': [torch.tensor(1.)], 'pg_est_cards': [torch.tensor(1.)],
                    'n_join_cols': [1], 'n_fanouts': [1], 'n_tables': [1], 'n_filter_cols': [0]}
            with self.assertRaises(ValueError):
                write_cache(Path(d)/'bad', rows, data, {})


if __name__ == '__main__':
    unittest.main()
