"""Archived source summary arithmetic, NOT prediction/label re-evaluation."""

import hashlib
import json
import math
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE = 'b3390b7463d0dd0fc002903f98214b68c696a11b'
STUDIES = ('s4-pg-output-anchor-20261006-pidfix', 's5-complex-weighting-20261007',
           's6-mean-tail-20261007')
VALIDATION = ('chembl', 'legalacts', 'telstra')


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    studies = []
    for study in STUDIES:
        relative = f'实验文档/results/{study}/summary.json'
        raw = (ROOT / relative).read_bytes()
        baseline = subprocess.check_output(['git', 'show', f'{BASE}:{relative}'], cwd=ROOT)
        archived = json.loads(raw)
        rows = []
        for run in archived['runs']:
            endpoint = run['endpoint']
            per = endpoint['per_database']
            if set(per) != set(VALIDATION):
                raise ValueError('source-only summary has unexpected databases')
            counts = {db: {bucket: per[db][bucket]['queries']
                          for bucket in ('all_tables', 'four_plus_tables')} for db in VALIDATION}
            if sum(counts[db]['all_tables'] for db in VALIDATION) != endpoint['unique_queries']:
                raise ValueError('stored complete unique denominator is inconsistent')
            calculated = {}
            for bucket, field in (('all_tables', 'macro_all_table_q95'),
                                  ('four_plus_tables', 'macro_four_plus_q95')):
                values = [per[db][bucket]['q95'] for db in VALIDATION]
                if any(type(v) not in (int, float) or not math.isfinite(v) or v < 1 for v in values):
                    raise ValueError('invalid stored source Q95')
                macro = math.fsum(values) / 3
                if not math.isclose(macro, endpoint[field], rel_tol=1e-14, abs_tol=0):
                    raise ValueError('archived macro does not match per-database arithmetic')
                calculated[field] = {'archived': endpoint[field], 'reaggregated': macro}
            rows.append({'arm': run['arm'], 'archived_endpoint_rows_sha256': run['endpoint_rows_sha256'],
                         'identity_digest': endpoint['identity_digest'],
                         'pg_identity_digest': endpoint['pg_identity_digest'],
                         'unique_queries': endpoint['unique_queries'], 'counts': counts,
                         'per_database_q95': {db: {bucket: per[db][bucket]['q95'] for bucket in
                                                  ('all_tables', 'four_plus_tables')} for db in VALIDATION},
                         'aggregate': calculated})
        studies.append({'study': study, 'summary_sha256_current': sha(raw),
                        'summary_sha256_baseline': sha(baseline), 'byte_unchanged': raw == baseline,
                        'historical_decision_preserved_not_reselected': archived['decision'], 'runs': rows})
    report = {'scope': 'archived_source_summary_preservation_and_macro_arithmetic_only',
              'baseline_commit': BASE, 'source_database_whitelist': list(VALIDATION),
              'passed': all(study['byte_unchanged'] for study in studies), 'studies': studies,
              'actual_per_query_predictions_recomputed': False,
              'per_query_recomputation_status': 'blocked_missing_local_frozen_prediction_label_pg_identity_inputs',
              'candidate_reselected': False, 'target_read': False,
              'note': 'Archive equality and arithmetic do not independently authenticate raw Q95 values.'}
    output = Path(__file__).parent / 'source-main-preservation.json'
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    print(json.dumps({'passed': report['passed'], 'studies': len(studies),
                      'per_query_recomputed': False}))


if __name__ == '__main__':
    main()
