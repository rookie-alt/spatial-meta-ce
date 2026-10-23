#!/usr/bin/env python3
"""PRICE few-shot evaluation on target databases with an inner_lr chosen on
SOURCE VALIDATION only (see 实验文档/02-PRICE适应学习率搜索.md, chosen 1e-6).

Everything except inner_lr comes from configs/evaluation.json, so the protocol
(shots, support_runs, support_seed, inner_steps, scope) matches our own runs.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from spatial_meta_ce.cli import data_settings, read_json  # noqa: E402
from spatial_meta_ce.evaluate import evaluate  # noqa: E402

TARGETS = ['stats', 'genome', 'ergastf1']


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', required=True)
    p.add_argument('--device', default='cpu')
    p.add_argument('--inner-lr', type=float, default=1e-6)
    p.add_argument('--config', default='configs/evaluation.json')
    p.add_argument('--reference', default='runs/pg_reference_20260926.json')
    args = p.parse_args()

    cfg = read_json(args.config)
    cfg['inner_lr'] = args.inner_lr
    data = data_settings(cfg, args.config)
    data['splits']['test'] = TARGETS
    report = evaluate(cfg, data, args.out, args.device, baseline='price', training_seed=42,
                      reference_manifest=args.reference)
    (Path(args.out) / 'run_config.json').write_text(json.dumps(
        {'inner_lr': args.inner_lr, 'device': args.device, 'targets': TARGETS,
         'lr_source': 'runs/price-lr-search-20260926/result.json', 'config': cfg}, indent=2))

    print('=== SUMMARY ===')
    for row in report['aggregate']:
        a = row['across_support_runs']
        fi = a.get('paired_unique', {}).get('fraction_queries_improved', float('nan'))
        print(f"{row['database']:<10s} k={row['k']:3d} "
              f"q50={a['all_rows_q50']['mean']:7.3f} q95_all={a['all_rows_q95']['mean']:8.3f} "
              f"q95_uniq={a['unique_query_q95']['mean']:9.3f} improved_vs_pg={fi:.3f}")


if __name__ == '__main__':
    main()
