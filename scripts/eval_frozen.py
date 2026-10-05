#!/usr/bin/env python3
"""Target evaluation of OUR checkpoint with an adaptation strategy frozen from
scripts/adapt_strategy_search.py (source validation only). Reads the chosen
strategy from the search result JSON so target numbers cannot feed back."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from spatial_meta_ce.cli import data_settings, read_json
from spatial_meta_ce.evaluate import evaluate

TARGETS = ['stats', 'genome', 'ergastf1']


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--search', required=True, help='adapt_strategy_search result.json')
    p.add_argument('--out', required=True)
    p.add_argument('--device', default='cpu')
    p.add_argument('--config', default='configs/evaluation.json')
    p.add_argument('--reference', default='runs/pg_reference_20260926.json')
    p.add_argument('--model-config', help='training config whose "model" must match the checkpoint')
    args = p.parse_args()

    chosen = json.loads(Path(args.search).read_text())['chosen']
    if chosen['fallback']:
        raise SystemExit('Fallback strategy is not implemented in evaluate(); refusing')
    cfg = read_json(args.config)
    cfg.update(inner_steps=chosen['inner_steps'], inner_lr=chosen['inner_lr'] or cfg['inner_lr'],
               adapt_scope=chosen['scope'] or cfg['adapt_scope'])
    if args.model_config:
        cfg['model'] = read_json(args.model_config)['model']
    data = data_settings(cfg, args.config)
    data['splits']['test'] = TARGETS
    report = evaluate(cfg, data, args.out, args.device, checkpoint=args.checkpoint,
                      training_seed=args.seed, reference_manifest=args.reference)
    (Path(args.out) / 'run_config.json').write_text(json.dumps(
        {'checkpoint': args.checkpoint, 'strategy_source': args.search, 'chosen': chosen,
         'device': args.device, 'targets': TARGETS, 'config': cfg}, indent=2))
    for row in report['aggregate']:
        a = row['across_support_runs']
        fi = a.get('paired_unique', {}).get('fraction_queries_improved', float('nan'))
        print(f"{row['database']:<10s} k={row['k']:3d} "
              f"q50={a['all_rows_q50']['mean']:7.3f} q95_all={a['all_rows_q95']['mean']:8.3f} "
              f"q95_uniq={a['unique_query_q95']['mean']:9.3f} improved={fi:.3f}")


if __name__ == '__main__':
    main()
