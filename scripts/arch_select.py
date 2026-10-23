#!/usr/bin/env python3
"""Compare model variants on SOURCE VALIDATION databases only.

For every checkpoint, evaluates the default adaptation (config inner_lr /
inner_steps / scope) on a fixed validation episode plan and reports:
  - macro mean log-Q95 (same metric as training early stopping), and
  - zero-shot Q95 split by number of tables in the query, to check that the
    "error grows with table count" pattern also exists on source validation.

Episodes use --plan-seed 2718 (differs from training early-stop seed 1729).
Target databases (imdb/stats/genome/ergastf1) are never opened.
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from spatial_meta_ce.checkpoints import load_checkpoint
from spatial_meta_ce.cli import data_settings, read_json
from spatial_meta_ce.data import StorePool
from spatial_meta_ce.meta import adapt, forward
from spatial_meta_ce.metrics import q_values, summarize
from spatial_meta_ce.train import validation_plan

TARGETS = {'imdb', 'stats', 'genome', 'ergastf1'}


def checkpoint_pool(state: dict, data_config: dict) -> StorePool:
    metadata = state['metadata']
    exposed = set(metadata['train_databases']) | set(metadata['validation_databases'])
    databases = data_config['splits']['validation']
    if metadata.get('adapted_database'):
        raise ValueError('Source selection requires an unadapted initialization')
    if set(databases) != set(metadata['validation_databases']):
        raise ValueError('Checkpoint source validation split changed')
    if (exposed | set(databases)) & TARGETS:
        raise ValueError('Source selection cannot include target databases')
    mapping_config = metadata['training_signature'].get('mapping')
    mapped = state['model_config'].get('arch') == 'sum_map'
    if mapped != bool(mapping_config):
        raise ValueError('Checkpoint mapping configuration disagrees with architecture')
    pool = StorePool(data_config['cache_root'],
                     mapping_root=mapping_config['root'] if mapped else None,
                     mapping_mode=mapping_config['mode'] if mapped else 'mapping')
    provenance = metadata.get('mapping_provenance', {})
    if mapped and set(provenance) != exposed:
        raise ValueError('Checkpoint mapping provenance is incomplete')
    for db in databases:
        store = pool.get('pretrain', db)
        if metadata['manifest_hashes'].get(db) != store.manifest['content_hash']:
            raise ValueError(f'Checkpoint validation cache changed: {db}')
        if mapped and provenance[db] != store.mapping.provenance():
            raise ValueError(f'Checkpoint mapping provenance changed: {db}')
    return pool


def evaluate_checkpoint(model, config, pool, episodes, device):
    model.eval()
    per_ep = []
    zero = defaultdict(lambda: ([], []))
    seen = set()
    for ep in episodes:
        if ep['db'] in TARGETS:
            raise ValueError('Source selection episode contains a target database')
        store = pool.get('pretrain', ep['db'])
        for part in ('support', 'query'):
            expected = [store.rows[int(i)]['query_hash'] for i in ep[part]]
            if ep.get(f'{part}_hashes') != expected:
                raise ValueError('Validation plan query hashes changed')
        support = store.batch(ep['support'], device)
        query = store.batch(ep['query'], device)
        params = adapt(model, support, config['inner_lr'], config['inner_steps'],
                       config['adapt_scope'], config['loss'], training=False)
        with torch.no_grad():
            pred = forward(model, params, query).cpu().numpy()
            base = forward(model, dict(model.named_parameters()), query).cpu().numpy()
        true = query['true'].cpu().numpy()
        per_ep.append({'db': ep['db'], 'k': ep['k'], 'run': ep['run'],
                       'log_q95': summarize(pred, true)['log_q95']})
        n_tables = query['masks'][2].sum(1).cpu().numpy()
        for i, qi in enumerate(ep['query']):
            key = (ep['db'], int(qi))
            if key in seen:
                continue
            seen.add(key)
            bucket = zero[(ep['db'], int(n_tables[i]))]
            bucket[0].append(float(base[i]))
            bucket[1].append(float(true[i]))
    macro = float(np.mean([r['log_q95'] for r in per_ep]))
    by_k = defaultdict(list)
    for r in per_ep:
        by_k[r['k']].append(r['log_q95'])
    buckets = {}
    for (db, n), (lp, t) in sorted(zero.items()):
        q = q_values(np.array(lp), np.array(t))[0]
        buckets.setdefault(db, {})[n] = {'count': len(q), 'q50': float(np.percentile(q, 50)),
                                         'q95': float(np.percentile(q, 95))}
    return {'macro_log_q95': macro, 'by_k': {k: float(np.mean(v)) for k, v in sorted(by_k.items())},
            'zero_shot_by_tables': buckets}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoints', nargs='+', required=True, help='name=path/to/best.pt')
    p.add_argument('--config', default='configs/meta.json', help='source of validation plan + adaptation')
    p.add_argument('--plan-seed', type=int, default=2718)
    p.add_argument('--device', default='cpu')
    p.add_argument('--output', required=True, help='output JSON file path')
    args = p.parse_args()
    config = read_json(args.config)
    data_config = data_settings(config, args.config)
    val = data_config['splits']['validation']
    if set(val) & TARGETS:
        raise ValueError('Validation split contains a target database')
    pool = StorePool(data_config['cache_root'])
    plan_config = dict(config, validation_seed=args.plan_seed)
    episodes = validation_plan(pool, val, plan_config)
    results = {}
    for spec in args.checkpoints:
        name, path = spec.split('=', 1)
        model, state = load_checkpoint(path, args.device)
        exposed = set(state['metadata']['train_databases']) | set(state['metadata']['validation_databases'])
        if exposed & TARGETS:
            raise ValueError(f'{name} was trained on a target database')
        matching_pool = checkpoint_pool(state, data_config)
        r = evaluate_checkpoint(model, config, matching_pool, episodes, args.device)
        r.update(path=path, model_config=model.config,
                 parameters=sum(t.numel() for t in model.parameters()),
                 best_step_metric=state['best_metric'])
        results[name] = r
        print(name, f"{r['macro_log_q95']:.4f}", flush=True)
    out = {'purpose': 'model variant selection on source validation only',
           'validation_databases': val, 'plan_seed': args.plan_seed, 'episodes': len(episodes),
           'adaptation': {k: config[k] for k in ('inner_lr', 'inner_steps', 'adapt_scope', 'loss')},
           'results': results}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(out, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
