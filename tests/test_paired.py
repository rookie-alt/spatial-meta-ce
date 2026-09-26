import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from spatial_meta_ce.data import StorePool
from spatial_meta_ce.evaluate import evaluate
from spatial_meta_ce.features import write_cache
from test_protocol import fixtures


class PairedProtocolTests(unittest.TestCase):
    def test_evaluation_reports_unique_and_paired_metrics(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            cfg, dc = fixtures(root)
            # The test fixture has no checkpoint requirement for PG baseline.
            ec={'shots':[0],'support_runs':1,'support_seed':1,'inner_lr':0.01,
                'inner_steps':1,'adapt_scope':'all','loss':'smooth_l1','batch_size':8,
                'latency_repetitions':1,'model':cfg['model']}
            report=evaluate(ec,dc,root/'eval','cpu',baseline='pg')
            row=report['results'][0]
            self.assertIn('unique_query_metrics',row)
            self.assertIn('paired_vs_reference',row)
            self.assertEqual(report['primary_metric'],'unique_query_q95')

    def test_paired_manifest_is_shared_across_methods(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); cfg,dc=fixtures(root)
            ec={'shots':[0,5],'support_runs':2,'support_seed':2026,'inner_lr':0.01,
                'inner_steps':1,'adapt_scope':'all','loss':'smooth_l1','batch_size':8,
                'latency_repetitions':1,'model':cfg['model']}
            a=evaluate(ec,dc,root/'a','cpu',baseline='pg')
            b=evaluate(ec,dc,root/'b','cpu',baseline='random',training_seed=8)
            am=json.loads((root/'a'/'manifest.json').read_text())
            bm=json.loads((root/'b'/'manifest.json').read_text())
            self.assertEqual(am['episodes'],bm['episodes'])


if __name__=='__main__':
    unittest.main()
