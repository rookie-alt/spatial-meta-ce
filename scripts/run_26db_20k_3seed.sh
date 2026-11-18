#!/usr/bin/env bash
# Three-seed 26-database pre-registered training (doc 122).
set -Eeuo pipefail
ROOT=/gly/weilin/spatial-meta-ce-26db-20261118-code
RUN=/gly/weilin/spatial-meta-ce-26db-20261118-runs
PY=/gly/weilin/PRICE-master/venv/bin/python
CONFIG=configs/erm_sum_26db_20k.json
cd "$ROOT"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
[[ ! -e "$RUN" ]] || { printf 'Run directory already exists\n'; exit 1; }
mkdir -p "$RUN"
printf 'running time=%s\n' "$(date -Is)" > "$RUN/status.txt"

# One GPU per seed, all verified idle before launch.
declare -A GPU=( [42]=3 [43]=5 [44]=6 )
for seed in 42 43 44; do
  gpu=${GPU[$seed]}
  read -r mem util < <(nvidia-smi --query-gpu=memory.used,utilization.gpu \
    --format=csv,noheader,nounits -i "$gpu" | tr -d ',')
  [[ "$mem" -le 200 && "$util" -eq 0 ]] || {
    echo "GPU $gpu occupied before launch: mem=$mem util=$util"; exit 1; }
done

for seed in 42 43 44; do
  gpu=${GPU[$seed]}
  out="$RUN/seed$seed"
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" -B -u -m spatial_meta_ce.cli train \
    --config "$CONFIG" --device cuda --seed "$seed" --output "$out" \
    > "$RUN/train-seed$seed.log" 2>&1 &
  printf 'launched seed=%s gpu=%s pid=%s\n' "$seed" "$gpu" "$!"
done
wait

"$PY" -B - "$RUN" <<'PY'
import json
import sys
from pathlib import Path

from spatial_meta_ce.checkpoints import load_checkpoint

run = Path(sys.argv[1])
summary = {}
for seed in (42, 43, 44):
    out = run / f"seed{seed}"
    last = out / "last.pt"
    best = out / "best.pt"
    if not last.is_file():
        raise SystemExit(f"seed{seed} produced no last.pt")
    _, last_state = load_checkpoint(last)
    _, best_state = load_checkpoint(best)
    if last_state["next_step"] != 20000:
        raise SystemExit(f"seed{seed} stopped at {last_state['next_step']}")
    if len(last_state["metadata"]["train_databases"]) != 26:
        raise SystemExit(f"seed{seed} did not train on 26 databases")
    summary[str(seed)] = {
        "last_step": last_state["next_step"],
        "best_step": best_state["next_step"],
        "best_metric": last_state["best_metric"],
        "counts": last_state["extra"]["counts"],
        "total_seconds": last_state["extra"]["total_seconds"],
        "holdout_digest": last_state["metadata"]["holdout_digest"],
        "holdout_queries": sum(
            last_state["metadata"]["holdout_queries"].values()
        ),
    }
(run / "training_summary.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
PY
printf 'completed time=%s\n' "$(date -Is)" > "$RUN/status.txt"
