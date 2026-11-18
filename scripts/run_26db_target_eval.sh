#!/usr/bin/env bash
# Four-target K=0 population evaluation for the 26-database arm (doc 122).
# Read-only inference: no training, no adaptation, no checkpoint selection.
set -Eeuo pipefail
ROOT=/gly/weilin/spatial-meta-ce-26db-20261118-code
RUN=/gly/weilin/spatial-meta-ce-26db-20261118-runs
PY=/gly/weilin/PRICE-master/venv/bin/python
PRICE=/gly/weilin/PRICE-master/results/model_params.pth
CACHE=/gly/weilin/spatial-meta-ce/artifacts/cache
OUT=$RUN/target-k0
[[ ! -e "$OUT" ]] || { printf 'Output exists\n'; exit 1; }
mkdir -p "$OUT"
cd "$ROOT"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

for seed in 42 43 44; do
  CUDA_VISIBLE_DEVICES="" "$PY" -B scripts/target_population_eval.py \
    --cache-root "$CACHE" \
    --split test \
    --databases imdb stats genome ergastf1 \
    --allowed-targets imdb stats genome ergastf1 \
    --checkpoint "$RUN/seed$seed/best.pt" \
    --price-checkpoint "$PRICE" \
    --maxima "12,14,8,10" \
    --maxima-rule "doc 102 section 2.2: elementwise max of test+finetune inspection.max_counts over stats/genome/ergastf1; imdb excluded" \
    --device cpu --batch-size 128 \
    --output "$OUT/seed$seed.json" \
    --protocol "$OUT/seed$seed-protocol.json" \
    > "$OUT/seed$seed.log" 2>&1 &
done
wait
printf 'completed time=%s\n' "$(date -Is)" > "$OUT/status.txt"
ls -l "$OUT"
