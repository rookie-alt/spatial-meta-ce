"""Record arithmetic only, no producer imports and no matrix/array access."""

import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OLD = ROOT / 'reports/s8-g1-readonly-p-retained-kappa-20261008/real-proof'
OUTPUT = Path(__file__).parent / 'retained-archive-recheck.json'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    worker = json.loads((OLD / 'worker-proof.json').read_text())
    top = json.loads((OLD / 'proof.json').read_text())
    rows = []
    for feature, profiles in worker['features'].items():
        for stage, record in profiles.items():
            path = OLD / f'{feature}-{stage}.json'
            cert = record['certificate']
            check = cert['retained_subspace_check']
            rank = cert['observed_ranks']['svd']
            side_rows = []
            for side in ('reported', 'reference'):
                bound = check[f'{side}_bound']
                residual = check[f'{side}_residual']
                eps = 2**-52
                kappa = bound['sigma1'] / bound['sigma_r'] if rank else 0
                scale = max(bound['projection_norm'], bound['target_scale'], eps)
                floor = 1e-12 * (1 + bound['projection_norm'])
                reachable = 8 * eps * kappa * scale
                tol = max(floor, reachable)
                def equal(x, y):
                    return math.isclose(x, y, rel_tol=1e-12, abs_tol=0)
                passed = (
                    check['scope'] == 'real_retained_condition_number_fp64_reachable_bound'
                    and bound['eps'] == eps and bound['accumulation_constant'] == 8
                    and all(type(v) in (int, float) and math.isfinite(v) and v >= 0
                            for k, v in bound.items() if k != 'scope')
                    and 0 < bound['sigma_cutoff'] < bound['sigma_r'] <= bound['sigma1']
                    and equal(bound['sigma_cutoff'], 1e-12 * bound['sigma1'])
                    and equal(bound['kappa_retained'], kappa)
                    and equal(bound['engineering_floor'], floor)
                    and equal(bound['reachable_bound'], reachable)
                    and equal(bound['tol'], tol)
                    and equal(bound['amplification_scale'], scale)
                    and type(residual) in (int, float) and math.isfinite(residual)
                    and 0 <= residual <= tol
                )
                side_rows.append({'side': side, 'record_formula_passed': passed,
                                  'recomputed_tolerance': tol, 'stored_tolerance': bound['tol'],
                                  'stored_residual': residual})
            same = json.loads(path.read_text()) == record == top['features'][feature][stage]
            rows.append({'feature': feature, 'stage': stage, 'source': str(path.relative_to(ROOT)),
                         'source_sha256': sha(path), 'aggregate_single_consistent': same,
                         'sides': side_rows, 'actual_residual_independently_certified': False})
    report = {
        'scope': 'archived_retained_record_arithmetic_only_not_real_P_rerun',
        'original_top_status': top['status'], 'original_G1_status': top['original_G1_status'],
        'formal_ready': top['formal_ready'], 'strict_certificate': 'unknown',
        'can_retroactively_pass_original_G1': False,
        'passed': all(row['aggregate_single_consistent'] and all(side['record_formula_passed']
                      for side in row['sides']) for row in rows), 'rows': rows,
        'bindings': {name: sha(OLD / name) for name in ('proof.json', 'worker-proof.json', 'supervisor.json')},
        'uncertified': ['actual_singular_extrema', 'actual_projection_and_target_scales',
                        'actual_retained_residuals', 'true_rank', 'strict_linear_optimality'],
        'opened_numeric_arrays': [], 'real_P_rerun': False,
    }
    OUTPUT.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    print(json.dumps({'passed': report['passed'], 'stages': len(rows),
                      'original_top_status': report['original_top_status']}))


if __name__ == '__main__':
    main()
