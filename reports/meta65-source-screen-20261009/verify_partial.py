#!/usr/bin/env python3
"""Post-failure read-only artifact check; never nominates an arm or resumes a run."""
import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    root, run = args.root.resolve(), args.run.resolve()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    helper_path = root / 'reports/meta65-source-screen-20261009/verify_endpoint.py'
    spec = importlib.util.spec_from_file_location('meta65_independent', helper_path)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    evidence = helper.Evidence()
    supervisor = evidence.read(run.with_name(run.name + '.supervisor.json'))
    helper.require(supervisor['status'] == 'failed', 'Partial check only after failed formal exit')
    helper.require(not (run / 'decision.json').exists(), 'Failed formal run must not publish decision')
    environment = evidence.read(run / 'environment.json')
    for name, sha in environment['code_sha256'].items():
        evidence.bind(root / name, sha)
    config = evidence.read(root / 'configs/meta65_screen.json')
    helper.require(helper.sha256(root / config['protocol_file']) == config['protocol_sha256'], 'Protocol changed')
    data = evidence.read(root / 'configs/meta65_server_data.json')
    from scripts.precision_screen_evidence import historical_sources, source_evidence
    before = evidence.read(run / 'source_before.json')
    historical = historical_sources(root, list(helper.TRAIN + helper.VAL))
    rehashed = source_evidence(data, historical)
    helper.require(before == rehashed, 'Input cache changed after failed run')
    stores = helper.load_validation(Path(data['cache_root']), before, evidence)
    plans = evidence.read(run / 'endpoint_plan.json')
    helper.verify_plans(plans, stores)
    completed, partial_counts = {}, {}
    for arm in helper.ARMS:
        folder = run / arm
        partial_counts[arm] = len(list((folder / 'predictions').glob('*.npz')))
        endpoint = folder / 'endpoint.csv'
        if not endpoint.exists():
            continue
        rows = helper.load_csv(endpoint, evidence)
        helper.require(len(rows) == 75, 'Completed arm must have 75 cells')
        identities = set()
        for row in rows:
            helper.require(row['arm'] == arm and row['database'] in helper.VAL, 'Unexpected arm/source')
            key = (row['database'], row['k'], row['run'])
            helper.require(key not in identities, 'Duplicate cell')
            identities.add(key)
            plan = plans[row['database']][row['run']]
            helper.require(row['query_digest'] == plan['query_digest'] and
                           row['support_digest'] == plan['support_digests'][str(row['k'])], 'Plan mismatch')
            path = helper.path_checked(run / row['prediction_file'], run)
            evidence.bind(path, row['prediction_sha256'])
            with np.load(path, allow_pickle=False) as archive:
                arrays = {k: archive[k] for k in archive.files}
            helper.check_arrays(arrays, stores[row['database']], plan['query'])
            mask = arrays['n_tables'] >= 4
            helper.require(row['query_count'] == len(plan['query']) and
                           row['four_plus_count'] == int(mask.sum()), 'Counts mismatch')
            for prefix, selection in (('all', np.ones(len(mask), dtype=bool)), ('four_plus', mask)):
                metrics = helper.q_metrics(arrays['log_pred'][selection], arrays['true'][selection])
                for name, value in metrics.items():
                    helper.equal_tree(row[f'{prefix}_{name}'], value, f'{arm}/{key}/{name}')
        expected = {(db, k, repetition) for db in helper.VAL for k in helper.SHOTS for repetition in range(5)}
        helper.require(identities == expected, 'Completed endpoint coverage changed')
        metadata = evidence.read(folder / 'metadata.json')
        helper.require(metadata['last_step'] == metadata['counts']['optimizer_steps'] == 4000,
                       'Completed arm training budget changed')
        evidence.bind(folder / 'last.pt', metadata['checkpoint_sha256'])
        completed[arm] = {'verified_cells': 75, 'macro_four_plus_q95': math.fsum(r['four_plus_q95'] for r in rows) / 75,
                          'macro_all_q95': math.fsum(r['all_q95'] for r in rows) / 75,
                          'metadata': metadata,
                          'per_database_four_plus_q95': {db: math.fsum(r['four_plus_q95'] for r in rows if r['database'] == db) / 25
                                                        for db in helper.VAL}}
    evidence.recheck()
    result = {'status': 'partial_artifacts_verified_not_formal_completion',
              'formal_status': 'failed', 'formal_failure_reason': supervisor.get('failure_reason'),
              'candidate_selection': 'not_permitted_incomplete_matrix', 'completed_arms': completed,
              'raw_prediction_file_counts': partial_counts, 'source_rehashed_after_failure': rehashed,
              'limitations': ['No complete225-cell matrix; no nomination or target-generalization claim.',
                              'Post-failure CPU verification does not retroactively pass the timed-out formal run.']}
    helper.path_checked(args.out)
    with args.out.open('x') as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')
    print(json.dumps({k: v for k, v in result.items() if k in ('status', 'formal_failure_reason', 'raw_prediction_file_counts')}))


if __name__ == '__main__':
    main()
