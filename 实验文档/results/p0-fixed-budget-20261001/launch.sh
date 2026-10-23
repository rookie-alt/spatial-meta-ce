#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=/gly/weilin/spatial-meta-ce
PY=/gly/weilin/PRICE-master/venv/bin/python
RUN="$ROOT/runs/p0-fixed-budget-seed42-20261001"
RECORD="$ROOT/实验文档/results/p0-fixed-budget-20261001"
CONFIG=configs/erm_sum_fixed_budget.json
cd "$ROOT"
export PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=6 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
[[ ! -e "$RUN" ]] || { printf 'Run directory already exists\n'; exit 1; }
mkdir "$RUN"
trap 'rc=$?; if (( rc != 0 )); then printf "failed exit=%s time=%s\n" "$rc" "$(date -Is)" > "$RUN/status.txt"; fi' EXIT
printf 'running short-budget time=%s\n' "$(date -Is)" > "$RUN/status.txt"
"$PY" -B -c 'import json,hashlib,subprocess,os; from pathlib import Path; root=Path.cwd(); record=Path("实验文档/results/p0-fixed-budget-20261001"); snap=json.loads((record/"preflight.json").read_text()); assert all(hashlib.sha256((root/p).read_bytes()).hexdigest()==h for p,h in snap["input_sha256"].items()), "Preflight input changed"; cfg=json.loads(Path("configs/erm_sum_fixed_budget.json").read_text()); assert cfg["patience"]>cfg["steps"]//cfg["validate_every"]; gpu=subprocess.check_output(["nvidia-smi","--query-gpu=memory.used,utilization.gpu","--format=csv,noheader,nounits","-i","6"],text=True).strip().split(","); assert int(gpu[0])<=100 and int(gpu[1])==0, "GPU 6 is occupied"; assert os.statvfs(str(root)).f_bavail*os.statvfs(str(root)).f_frsize>5*1024**3, "Insufficient disk space"'
"$PY" -B -u -m spatial_meta_ce.cli train --config "$CONFIG" --steps 4000 --device cuda --output "$RUN/training" > "$RUN/train-4000.log" 2>&1
"$PY" -B - "$RUN" 4000 <<'PY'
import hashlib
import json
import shutil
import sys
from pathlib import Path
from spatial_meta_ce.checkpoints import load_checkpoint
run, budget = Path(sys.argv[1]), int(sys.argv[2])
_, state = load_checkpoint(run / 'training/last.pt')
assert state['next_step'] == budget, 'Budget not completed'
assert state['extra']['counts']['optimizer_steps'] == budget
archive = run / f'budget-{budget}'
archive.mkdir()
for name in ('last.pt', 'best.pt', 'history.json', 'config.json', 'validation_manifest.json'):
    shutil.copy2(run / 'training' / name, archive / name)
summary = {'budget': budget, 'last_step': state['next_step'], 'counts': state['extra']['counts'], 'total_seconds': state['extra']['total_seconds'], 'best_metric': state['best_metric'], 'best_step': load_checkpoint(archive / 'best.pt')[1]['next_step'], 'sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in archive.iterdir()}}
(archive / 'summary.json').write_text(json.dumps(summary, indent=2))
PY
printf 'running long-budget time=%s\n' "$(date -Is)" > "$RUN/status.txt"
"$PY" -B -u -m spatial_meta_ce.cli train --config "$CONFIG" --device cuda --output "$RUN/training" --resume "$RUN/training/last.pt" > "$RUN/train-20000.log" 2>&1
"$PY" -B - "$RUN" <<'PY'
import hashlib
import json
import shutil
import sys
from pathlib import Path
from spatial_meta_ce.checkpoints import load_checkpoint
run = Path(sys.argv[1])
_, state = load_checkpoint(run / 'training/last.pt')
assert state['next_step'] == 20000, 'Long budget not completed'
assert state['extra']['counts']['optimizer_steps'] == 20000
short = json.loads((run / 'budget-4000/history.json').read_text())
long = json.loads((run / 'training/history.json').read_text())
assert short == long[:len(short)], 'Training history prefix changed'
assert (run / 'budget-4000/validation_manifest.json').read_bytes() == (run / 'training/validation_manifest.json').read_bytes(), 'Validation queries changed'
assert state['extra']['counts']['query_presentations'] > short[-1]['counts']['query_presentations']
archive = run / 'budget-20000'
archive.mkdir()
for name in ('last.pt', 'best.pt', 'history.json', 'config.json', 'validation_manifest.json'):
    shutil.copy2(run / 'training' / name, archive / name)
summary = {'budget': 20000, 'last_step': state['next_step'], 'counts': state['extra']['counts'], 'total_seconds': state['extra']['total_seconds'], 'best_metric': state['best_metric'], 'best_step': load_checkpoint(archive / 'best.pt')[1]['next_step'], 'history_prefix_verified': True, 'validation_manifest_verified': True, 'sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in archive.iterdir()}}
(archive / 'summary.json').write_text(json.dumps(summary, indent=2))
root = Path.cwd()
snap = json.loads(Path('实验文档/results/p0-fixed-budget-20261001/preflight.json').read_text())
assert all(hashlib.sha256((root / p).read_bytes()).hexdigest() == h for p, h in snap['input_sha256'].items()), 'Inputs changed during training'
PY
printf 'running source-comparison time=%s\n' "$(date -Is)" > "$RUN/status.txt"
"$PY" -B -u scripts/arch_select.py --config configs/meta.json --plan-seed 2718 --device cuda --checkpoints "short_last=$RUN/budget-4000/last.pt" "long_last=$RUN/budget-20000/last.pt" "short_best=$RUN/budget-4000/best.pt" "long_best=$RUN/budget-20000/best.pt" baseline=runs/erm-sum-seed42-20260927/best.pt --output "$RUN/source-comparison.json" > "$RUN/source-comparison.log" 2>&1
"$PY" -B - "$RUN" <<'PY'
import json
import math
import sys
from pathlib import Path
run = Path(sys.argv[1])
result = json.loads((run / 'source-comparison.json').read_text())
assert result['validation_databases'] == ['chembl', 'legalacts', 'telstra']
short, long = (result['results'][key] for key in ('short_last', 'long_last'))
buckets = [{'database': db, 'tables': int(n), 'count': s['count'], 'short_q95': s['q95'], 'long_q95': long['zero_shot_by_tables'][db][n]['q95'], 'ratio': long['zero_shot_by_tables'][db][n]['q95'] / s['q95']} for db, group in short['zero_shot_by_tables'].items() for n, s in group.items() if int(n) >= 4]
assert all(b['count'] == long['zero_shot_by_tables'][b['database']][str(b['tables'])]['count'] for b in buckets), 'Query counts changed'
db_ratios = {db: math.exp(sum(math.log(b['ratio']) for b in buckets if b['database'] == db) / sum(b['database'] == db for b in buckets)) for db in result['validation_databases']}
ratio = math.exp(sum(math.log(r) for r in db_ratios.values()) / len(db_ratios))
decision = 'strong_positive' if ratio <= 0.2 and all(r <= 1.0 for r in db_ratios.values()) else 'partial_improvement' if ratio < 1.0 else 'no_overall_improvement'
summary = {'primary_macro_ratio': ratio, 'database_ratios': db_ratios, 'buckets': buckets, 'decision': decision, 'adapted_macro_log_q95': {key: value['macro_log_q95'] for key, value in result['results'].items()}, 'budget_summaries': {str(b): json.loads((run / f'budget-{b}/summary.json').read_text()) for b in (4000, 20000)}, 'target_evaluation_performed': False, 'limitations': ['Single seed; presentation counts include repeated queries; current budget is not PRICE-scale; no causal exclusion from a negative outcome.']}
(run / 'decision.json').write_text(json.dumps(summary, indent=2, allow_nan=False))
PY
printf 'completed time=%s\n' "$(date -Is)" > "$RUN/status.txt"
printf 'FIXED_BUDGET_COMPLETE %s\n' "$RUN"
