#!/usr/bin/env python3
"""Diagnostic: K=0 unique-query error split by number of joined tables.

Reads existing prediction CSVs only (no training, no strategy selection), so
looking at target databases here is analysis, not tuning.
"""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from spatial_meta_ce.metrics import q_values


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--cache', default='artifacts/cache/test')
    p.add_argument('--pred', nargs='+', required=True, help='name=path/to/predictions.csv')
    p.add_argument('--k', type=int, default=0)
    args = p.parse_args()
    tables = {}
    for db in ('stats', 'genome', 'ergastf1'):
        path = Path(args.cache) / db
        offsets = np.load(path / 'offsets.npy')
        rows = json.loads((path / 'manifest.json').read_text())['rows']
        n_tables = (offsets[1:, 2] - offsets[:-1, 2]).tolist()
        tables.update({r['id']: t for r, t in zip(rows, n_tables)})
    out = {}
    for spec in args.pred:
        name, path = spec.split('=', 1)
        seen = defaultdict(dict)
        with open(path) as f:
            for r in csv.DictReader(f):
                if int(r['k']) != args.k or int(r['run']) != 0:
                    continue
                seen[r['database']].setdefault(r['query_hash'], (r['query_id'], r['log_prediction'], r['true_cardinality']))
        res = {}
        for db, qs in seen.items():
            buckets = defaultdict(lambda: ([], []))
            for qid, lp, t in qs.values():
                buckets[tables[qid]][0].append(float(lp))
                buckets[tables[qid]][1].append(float(t))
            buckets = {n: q_values(lp, t)[0] for n, (lp, t) in buckets.items()}
            res[db] = {n: {'count': len(v), 'q50': float(np.percentile(v, 50)), 'q95': float(np.percentile(v, 95))}
                       for n, v in sorted(buckets.items())}
        out[name] = res
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
