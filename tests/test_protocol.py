import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

AVAILABLE = importlib.util.find_spec('spatial_meta_ce.train') is not None
if AVAILABLE:
    from spatial_meta_ce.checkpoints import load_checkpoint
    from spatial_meta_ce.evaluate import evaluate
    from spatial_meta_ce.features import write_cache
    from spatial_meta_ce.train import select_best, train


def fixtures(root):
    cache = root/'cache'
    for split, db in [('pretrain','a'),('pretrain','v'),('finetune','t'),('test','t')]:
        rows = [{'id':f'{db}/{split}/{i}', 'query_hash':f'{split}-{i}', 'true':float(20+i), 'pg':2.} for i in range(20)]
        data = {'data_features':[(torch.ones(40),torch.ones(40),torch.ones(4),None) for _ in rows],
                'true_cards':[torch.tensor(r['true']) for r in rows], 'pg_est_cards':[torch.tensor(2.) for _ in rows],
                'n_join_cols':[1]*20,'n_fanouts':[1]*20,'n_tables':[1]*20,'n_filter_cols':[0]*20}
        write_cache(cache/split/db, rows, data, {'source': 'unit-fixture'})
    dc = {'cache_root':str(cache),'data_root':str(root/'unused'),
          'splits':{'train':['a'],'validation':['v'],'test':['t']}}
    cfg = {'model':{'width':8,'use_pg':True},'method':'fomaml','seed':7,'steps':2,
           'meta_batch':1,'accumulation':1,'outer_lr':0.001,'inner_lr':0.01,'inner_steps':1,
           'adapt_scope':'all','support_sizes':[5],'query_size':4,'loss':'smooth_l1',
           'validate_every':1,'validation_runs':1,'validation_seed':1729,'patience':20,'min_delta':0.0}
    return cfg, dc


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue(AVAILABLE, 'Training/evaluation protocol not implemented')
        torch.set_num_threads(1)

    def test_validation_only_selection(self):
        best, bad, improved = select_best(2.0, 3, 3.0, 0.001)
        self.assertEqual((best,bad,improved),(2.0,4,False))
        best, bad, improved = select_best(2.0, 3, 1.0, 0.001)
        self.assertEqual((best,bad,improved),(1.0,0,True))
        with self.assertRaises(ValueError):
            select_best(2.0, 3, float('nan'), 0.001)

    def test_resume_matches_uninterrupted(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            cfg, dc = fixtures(root)
            train(cfg, dc, root/'full', 'cpu')
            short = dict(cfg, steps=1)
            train(short, dc, root/'split', 'cpu')
            train(cfg, dc, root/'split', 'cpu', resume=root/'split/last.pt')
            a, ac = load_checkpoint(root/'full/last.pt')
            b, bc = load_checkpoint(root/'split/last.pt')
            for k in a.state_dict():
                self.assertTrue(torch.equal(a.state_dict()[k], b.state_dict()[k]), k)
            self.assertEqual(ac['next_step'],bc['next_step'])
            self.assertEqual(ac['best_metric'],bc['best_metric'])

    def test_complete_tiny_evaluation(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            cfg, dc=fixtures(root)
            train(cfg,dc,root/'train','cpu')
            ec={'shots':[0,5],'support_runs':2,'support_seed':2026,'inner_lr':0.01,'inner_steps':1,
                'adapt_scope':'all','loss':'smooth_l1','batch_size':8,'latency_repetitions':2,'model':cfg['model']}
            report=evaluate(ec,dc,root/'eval','cpu',checkpoint=root/'train/best.pt')
            self.assertEqual(len(report['results']),4)
            self.assertTrue((root/'eval/predictions.csv').exists())
            for row in report['results']:
                self.assertEqual(row['metrics']['n'],20)
                self.assertTrue(np.isfinite(row['metrics']['q95']))

    def test_erm_runs(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            cfg,dc=fixtures(root)
            train(dict(cfg,method='erm'),dc,root/'erm','cpu')
            _,state=load_checkpoint(root/'erm/best.pt')
            self.assertEqual(state['metadata']['method'],'erm')


if __name__=='__main__':
    unittest.main()
