#!/usr/bin/env python3
"""Select OUR adaptation strategy on SOURCE VALIDATION databases only.

Grid (frozen before the run):
  inner_steps: 0 / 1 / 3 / 5 / 10
  inner_lr:    1e-4 / 3e-4 / 1e-3 / 3e-3 / 1e-2
  scope:       head / all
  fallback:    off / on  (2-fold split of the support set: keep adaptation only if
               the adapted model beats the unadapted one on the held-out half,
               otherwise fall back to zero-shot; uses support labels only)

Selection rule:
  - metric: macro mean log-Q95 over validation episodes, averaged over checkpoints
  - ties (|delta| < 1e-4): fewer steps, then head before all, then smaller lr,
    then fallback off
  - episodes are drawn with --plan-seed (default 2718), which differs from the
    training-time early-stopping seed (1729), so the checkpoints were not
    selected on these exact episodes
  - target databases (imdb/stats/genome/ergastf1) are NEVER opened

Every candidate is written to the output JSON so the search can be audited.
"""
import argparse
import itertools
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from spatial_meta_ce.checkpoints import load_checkpoint
from spatial_meta_ce.cli import data_settings, read_json
from spatial_meta_ce.data import StorePool
from spatial_meta_ce.meta import adapt_trajectory, forward, loss_value
from spatial_meta_ce.metrics import summarize
from spatial_meta_ce.train import validation_plan

STEPS = [0, 1, 3, 5, 10]
LRS = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2]
SCOPES = ['head', 'all']


def subset(batch, idx):
    idx_t = torch.as_tensor(idx, dtype=torch.long, device=batch['label'].device)
    return {
        'groups': [g[idx_t] for g in batch['groups']],
        'masks': [m[idx_t] for m in batch['masks']],
        'counts': batch['counts'][idx_t],
        'label': batch['label'][idx_t],
        'true': batch['true'][idx_t],
        'pg_log': batch['pg_log'][idx_t],
        'ids': [batch['ids'][i] for i in idx],
    }


def holdout_gain(model, support, lr, scope, loss, rng):
    """For each step count, True if 2-fold held-out support loss improves on step 0."""
    n = len(support['ids'])
    order = rng.permutation(n)
    halves = [order[: n // 2], order[n // 2:]]
    totals = {s: 0.0 for s in STEPS}
    for fit, hold in [(halves[0], halves[1]), (halves[1], halves[0])]:
        traj = adapt_trajectory(model, subset(support, fit), lr, STEPS, scope, loss)
        held = subset(support, hold)
        with torch.no_grad():
            for s, params in traj.items():
                totals[s] += float(loss_value(forward(model, params, held), held['label'], loss))
    return {s: totals[s] < totals[0] for s in STEPS}


def score_checkpoint(model, pool, episodes, loss, device, fold_seed):
    """Returns {(steps, lr, scope, fallback): [log_q95 per episode]} plus per-episode info."""
    scores, info = {}, []
    model.eval()
    for e_idx, ep in enumerate(episodes):
        store = pool.get('pretrain', ep['db'])
        support = store.batch(ep['support'], device)
        query = store.batch(ep['query'], device)
        truth = query['true'].cpu().numpy()
        with torch.no_grad():
            zero = forward(model, dict(model.named_parameters()), query).cpu().numpy()
        zero_m = summarize(zero, truth)
        for lr, scope in itertools.product(LRS, SCOPES):
            traj = adapt_trajectory(model, support, lr, STEPS, scope, loss)
            gain = holdout_gain(model, support, lr, scope, loss,
                                np.random.default_rng([fold_seed, e_idx]))
            for s, params in traj.items():
                with torch.no_grad():
                    pred = forward(model, params, query).cpu().numpy()
                m = summarize(pred, truth)
                for fallback in (False, True):
                    use = zero_m if (fallback and not gain[s]) else m
                    key = (s, lr, scope, fallback)
                    scores.setdefault(key, []).append(use['log_q95'])
                    if fallback:
                        scores.setdefault(('accept',) + key, []).append(bool(gain[s]))
        info.append({'db': ep['db'], 'k': ep['k'], 'run': ep['run'],
                     'zero_shot_log_q95': zero_m['log_q95']})
    return scores, info


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', default='configs/meta.json')
    p.add_argument('--checkpoints', nargs='+', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--device', default='cpu')
    p.add_argument('--runs', type=int, default=5)
    p.add_argument('--plan-seed', type=int, default=2718)
    p.add_argument('--fold-seed', type=int, default=31)
    args = p.parse_args()

    cfg = read_json(args.config)
    data = data_settings(cfg, args.config)
    if set(data['splits']['test']) & set(data['splits']['validation']):
        raise SystemExit('FATAL: target database present in validation split')
    if args.plan_seed == cfg['validation_seed']:
        raise SystemExit('FATAL: plan seed equals training early-stopping seed')
    if args.device == 'cpu':
        torch.set_num_threads(1)

    pool = StorePool(data['cache_root'])
    plan_cfg = dict(cfg, validation_runs=args.runs, validation_seed=args.plan_seed)
    episodes = validation_plan(pool, data['splits']['validation'], plan_cfg)
    print(f'episodes={len(episodes)} dbs={data["splits"]["validation"]} '
          f'K={cfg["support_sizes"]} runs={args.runs}', flush=True)

    per_ckpt = {}
    info = None
    for path in args.checkpoints:
        model, state = load_checkpoint(path, args.device)
        if set(state['metadata'].get('train_databases', [])) & set(data['splits']['test']):
            raise SystemExit(f'FATAL: {path} was trained on a target database')
        started = time.perf_counter()
        per_ckpt[path], info = score_checkpoint(model, pool, episodes, cfg['loss'],
                                                args.device, args.fold_seed)
        print(f'{path}: {time.perf_counter() - started:.1f}s', flush=True)

    trials = []
    for s, lr, scope, fallback in itertools.product(STEPS, LRS, SCOPES, (False, True)):
        if s == 0 and (lr != LRS[0] or scope != SCOPES[0] or fallback):
            continue  # all step-0 candidates are identical zero-shot
        key = (s, lr, scope, fallback)
        per_seed = {c: float(np.mean(v[key])) for c, v in per_ckpt.items()}
        by_db, by_k = {}, {}
        for v in per_ckpt.values():
            for x, ep in zip(v[key], info):
                by_db.setdefault(ep['db'], []).append(x)
                by_k.setdefault(ep['k'], []).append(x)
        row = {'inner_steps': s, 'inner_lr': lr if s else None, 'scope': scope if s else None,
               'fallback': fallback,
               'macro_log_q95': float(np.mean(list(per_seed.values()))),
               'per_checkpoint': per_seed,
               'per_db': {d: float(np.mean(x)) for d, x in by_db.items()},
               'per_k': {str(k): float(np.mean(x)) for k, x in sorted(by_k.items())}}
        if fallback:
            row['accept_rate'] = float(np.mean([a for v in per_ckpt.values()
                                                for a in v[('accept',) + key]]))
        trials.append(row)

    zero = next(t for t in trials if t['inner_steps'] == 0)
    best_score = min(t['macro_log_q95'] for t in trials)
    tied = [t for t in trials if t['macro_log_q95'] - best_score < 1e-4]
    winner = min(tied, key=lambda t: (t['inner_steps'], SCOPES.index(t['scope'] or 'head'),
                                      t['inner_lr'] or 0, t['fallback']))
    current = next(t for t in trials if t['inner_steps'] == cfg['inner_steps']
                   and t['inner_lr'] == cfg['inner_lr'] and t['scope'] == cfg['adapt_scope']
                   and not t['fallback'])
    result = {
        'schema': 1,
        'purpose': 'source-validation-only adaptation strategy selection (our model)',
        'selection_rule': 'minimum macro log-Q95 averaged over checkpoints; ties (<1e-4): '
                          'fewer steps, head before all, smaller lr, fallback off',
        'grid': {'inner_steps': STEPS, 'inner_lr': LRS, 'scope': SCOPES, 'fallback': [False, True]},
        'fallback_rule': '2-fold split of support; keep adapted params only if summed held-out '
                         'support loss < unadapted; otherwise zero-shot',
        'validation_databases': data['splits']['validation'],
        'target_databases_NOT_read': data['splits']['test'],
        'checkpoints': args.checkpoints,
        'plan_seed': args.plan_seed, 'fold_seed': args.fold_seed,
        'support_sizes': cfg['support_sizes'], 'runs': args.runs, 'episodes': len(episodes),
        'device': args.device,
        'zero_shot': zero, 'current_default': current,
        'chosen': winner,
        'trials': sorted(trials, key=lambda t: t['macro_log_q95']),
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, allow_nan=False))
    for t in result['trials'][:10]:
        print(json.dumps({k: t[k] for k in ('inner_steps', 'inner_lr', 'scope', 'fallback',
                                            'macro_log_q95', 'per_db')}), flush=True)
    print(f'zero-shot={zero["macro_log_q95"]:.4f} current={current["macro_log_q95"]:.4f} '
          f'CHOSEN={winner["inner_steps"]} steps lr={winner["inner_lr"]} scope={winner["scope"]} '
          f'fallback={winner["fallback"]} score={winner["macro_log_q95"]:.4f} -> {out}', flush=True)


if __name__ == '__main__':
    main()
