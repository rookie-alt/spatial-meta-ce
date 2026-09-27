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

from spatial_meta_ce.cli import read_json, data_settings  # noqa: E402
from spatial_meta_ce.evaluate import evaluate  # noqa: E402

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
    args = p.parse_args()

    chosen = json.loads(Path(args.search).read_text())['chosen']
    if chosen['fallback']:
        raise SystemExit('Fallback strategy is not implemented in evaluate(); refusing')
    cfg = read_json(args.config)
    cfg.update(inner_steps=chosen['inner_steps'], inner_lr=chosen['inner_lr'] or cfg['inner_lr'],
               adapt_scope=chosen['scope'] or cfg['adapt_scope'])
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
        print('%-10s k=%3d q50=%7.3f q95_all=%8.3f q95_uniq=%9.3f improved=%.3f' % (
            row['database'], row['k'], a['all_rows_q50']['mean'], a['all_rows_q95']['mean'],
            a['unique_query_q95']['mean'], fi))


if __name__ == '__main__':
    main()
