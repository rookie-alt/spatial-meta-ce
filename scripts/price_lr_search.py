#!/usr/bin/env python3
"""Select PRICE adaptation inner_lr using SOURCE VALIDATION databases only.

Why: PRICE's 10.7M-parameter model diverges with the small-model default
(inner_lr=1e-3). Before any few-shot comparison against PRICE is valid, its
adaptation step size must be chosen without touching target test data.

Selection rule (frozen before the run):
  - candidates: --lrs (default 1e-6 1e-5 1e-4 1e-3)
  - metric:     macro mean log-Q95 over source-validation episodes
                (equal count per database, K and repeat)
  - tie-break:  smaller lr (fewer parameters displaced, more stable)
  - databases:  chembl / legalacts / telstra only
  - target dbs (stats/genome/ergastf1/imdb): NEVER read by this script

Writes a JSON record of every candidate (not just the winner) so the search
can be audited later.
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from spatial_meta_ce.baselines import PriceAdapter
from spatial_meta_ce.cli import data_settings, read_json
from spatial_meta_ce.data import StorePool
from spatial_meta_ce.train import seed_all, validate, validation_plan


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', default='configs/meta.json')
    p.add_argument('--output', required=True)
    p.add_argument('--device', default='cpu')
    p.add_argument('--lrs', type=float, nargs='+', default=[1e-6, 1e-5, 1e-4, 1e-3])
    p.add_argument('--runs', type=int, default=5, help='support repeats per (db, K)')
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()

    cfg = read_json(args.config)
    data = data_settings(cfg, args.config)
    # Hard guard: this script must never see target test databases.
    for db in data['splits']['test']:
        if db in data['splits']['validation']:
            raise SystemExit('FATAL: target database present in validation split')
    if args.device == 'cpu':
        torch.set_num_threads(1)

    pool = StorePool(data['cache_root'])
    maxima = np.ones(4, dtype=np.int64)
    for db in data['splits']['validation']:
        maxima = np.maximum(maxima,
                            pool.get('pretrain', db).manifest['inspection']['max_counts'])

    seed_all(args.seed)
    weight = str(Path(data['data_root']) / 'results' / 'model_params.pth')
    model = PriceAdapter(weight, maxima, args.device)
    print(f'PRICE params={sum(p.numel() for p in model.parameters())} maxima={maxima.tolist()}',
          flush=True)

    plan_cfg = dict(cfg, validation_runs=args.runs)
    episodes = validation_plan(pool, data['splits']['validation'], plan_cfg)
    print(f'episodes={len(episodes)} dbs={data["splits"]["validation"]} '
          f'K={cfg["support_sizes"]} runs={args.runs}', flush=True)

    trials = []
    for lr in args.lrs:
        candidate = dict(plan_cfg, inner_lr=lr, inner_steps=cfg['inner_steps'],
                         adapt_scope=cfg['adapt_scope'], loss=cfg['loss'])
        started = time.perf_counter()
        score, records = validate(model, pool, episodes, candidate, args.device)
        elapsed = time.perf_counter() - started
        per_db = {}
        for r in records:
            per_db.setdefault(r['db'], []).append(r['metrics']['q95'])
        row = {
            'inner_lr': lr,
            'macro_log_q95': float(score),
            'elapsed_seconds': elapsed,
            'per_db_mean_q95': {k: float(np.mean(v)) for k, v in per_db.items()},
            'per_db_max_q95': {k: float(np.max(v)) for k, v in per_db.items()},
        }
        trials.append(row)
        print(json.dumps(row), flush=True)

    # Frozen rule: lowest macro log-Q95, then smaller lr.
    winner = min(trials, key=lambda t: (t['macro_log_q95'], t['inner_lr']))
    result = {
        'schema': 1,
        'purpose': 'source-validation-only PRICE adaptation lr selection',
        'selection_rule': 'minimum macro mean log-Q95; ties broken by smaller lr',
        'candidate_lrs': args.lrs,
        'validation_databases': data['splits']['validation'],
        'target_databases_NOT_read': data['splits']['test'],
        'support_sizes': cfg['support_sizes'],
        'inner_steps': cfg['inner_steps'],
        'adapt_scope': cfg['adapt_scope'],
        'loss': cfg['loss'],
        'support_runs_per_db_k': args.runs,
        'episodes': len(episodes),
        'seed': args.seed,
        'device': args.device,
        'price_parameters': sum(p.numel() for p in model.parameters()),
        'trials': trials,
        'chosen_inner_lr': winner['inner_lr'],
        'chosen_macro_log_q95': winner['macro_log_q95'],
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, allow_nan=False))
    print(f'CHOSEN inner_lr={winner["inner_lr"]:.0e} '
          f'macro_log_q95={winner["macro_log_q95"]:.4f} -> {out}', flush=True)


if __name__ == '__main__':
    main()
