#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=/gly/weilin/spatial-meta-ce
PY=/gly/weilin/PRICE-master/venv/bin/python
RUN="$ROOT/runs/p1-input-audit-20261002"
cd "$ROOT"
export PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
[[ ! -e "$RUN" ]] || { printf 'Output already exists\n'; exit 1; }
mkdir "$RUN"
trap 'rc=$?; if (( rc != 0 )); then printf "failed exit=%s time=%s\n" "$rc" "$(date -Is)" > "$RUN/status.txt"; fi' EXIT
printf 'running time=%s\n' "$(date -Is)" > "$RUN/status.txt"
DATABASE_GROUPS=(
  'chembl legalacts telstra'
  'accidents airline baseball basketball carcinogenesis'
  'ccs consumer credit employee financial'
  'fnhk grants hepatitis hockey movielens'
  'sakila sap seznam ssb'
  'talkingdata tournament tpc_h tubepricing'
)
pids=()
for index in "${!DATABASE_GROUPS[@]}"; do
    group="${DATABASE_GROUPS[$index]}"
    read -r -a databases <<< "$group"
    "$PY" -B -u scripts/audit_input_mapping.py --databases "${databases[@]}" --recreate --output "$RUN/group-$index" > "$RUN/group-$index.log" 2>&1 &
    pids+=("$!")
    printf '%s\n' "$!" > "$RUN/group-$index.pid"
done
failed=0
for pid in "${pids[@]}"; do
    if ! wait "$pid"; then failed=1; fi
done
"$PY" -B - "$RUN" <<'PY'
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
run = Path(sys.argv[1])
summaries = [json.loads(p.read_text()) for p in sorted(run.glob('group-*/summary.json'))]
expected = json.loads(Path('configs/data.json').read_text())['splits']
databases = expected['train'] + expected['validation']
reports = [r for s in summaries for r in s['reports']]
seen = [r['database'] for r in reports]
flags = Counter()
for r in reports:
    flags.update(r.get('sql_flags', {}))
summary = {'completed_at': datetime.now().astimezone().isoformat(),
           'groups_completed': len(summaries), 'expected_databases': databases,
           'reported_databases': seen, 'missing_databases': sorted(set(databases) - set(seen)),
           'duplicate_databases': sorted(d for d in set(seen) if seen.count(d) > 1),
           'checked_rows': sum(r.get('totals', {}).get('checked_rows', 0) for r in reports),
           'mapping_recoverable': sum(r.get('totals', {}).get('mapping_recoverable', 0) for r in reports),
           'sql_mapping_recoverable': sum(r.get('totals', {}).get('sql_mapping_recoverable', 0) for r in reports),
           'rows_with_risk_flags': sum(r.get('totals', {}).get('rows_with_risk_flags', 0) for r in reports),
           'bucket_denominators': 'Table-count buckets are exclusive; predicate and risk buckets overlap',
           'failed_databases': [r['database'] for r in reports if not r['alignment_passed']],
           'sql_flags': dict(flags), 'reports': reports,
           'scope': 'Source pretrain only; feature recreation is in memory; no training or cache writes',
           'interpretation': 'mapping_recoverable is structural recovery, not certification of predicate semantics'}
(run / 'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False))
assert len(summaries) == 6 and len(seen) == len(databases) and set(seen) == set(databases), 'Incomplete audit'
assert not summary['failed_databases'], 'Alignment failures are recorded; do not rebuild caches or train'
PY
(( failed == 0 ))
printf 'completed time=%s\n' "$(date -Is)" > "$RUN/status.txt"
