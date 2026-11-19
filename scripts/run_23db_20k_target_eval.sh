#!/usr/bin/env bash
# Matched-budget baseline: 23-database arm at the SAME 20,000 steps as the
# 26-database arm, four-target K=0 population evaluation.
set -Eeuo pipefail
ROOT=/gly/weilin/spatial-meta-ce-26db-20261118-code
PY=/gly/weilin/PRICE-master/venv/bin/python
PRICE=/gly/weilin/PRICE-master/results/model_params.pth
CACHE=/gly/weilin/spatial-meta-ce/artifacts/cache
CKPT=/gly/weilin/spatial-meta-ce-learning-curve-20261118-runs/training/step-20000.pt
OUT=/gly/weilin/spatial-meta-ce-26db-20261118-runs/target-k0-23db20k
[[ ! -e "$OUT" ]] || { printf 'Output exists\n'; exit 1; }
mkdir -p "$OUT"
cd "$ROOT"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
CUDA_VISIBLE_DEVICES="" "$PY" -B scripts/target_population_eval.py \
  --cache-root "$CACHE" --split test \
  --databases imdb stats genome ergastf1 \
  --allowed-targets imdb stats genome ergastf1 \
  --checkpoint "$CKPT" \
  --price-checkpoint "$PRICE" \
  --maxima "12,14,8,10" \
  --maxima-rule "doc 102 section 2.2: elementwise max of test+finetune inspection.max_counts over stats/genome/ergastf1; imdb excluded" \
  --device cpu --batch-size 128 \
  --output "$OUT/seed42.json" --protocol "$OUT/seed42-protocol.json" \
  > "$OUT/seed42.log" 2>&1
printf 'completed time=%s\n' "$(date -Is)" > "$OUT/status.txt"
