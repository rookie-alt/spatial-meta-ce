#!/bin/sh
# Doc 110 L3 extension: imdb and genome, same four arms and sampling seed as
# stats/ergastf1.  Statement cap raised to 300 s because a single imdb execution
# runs tens of seconds (probe: median 4.4 s, max 82 s) -- PRICE's own e2e used
# sql_execution_timeout = 300000 ms.  Sequential on purpose: overlapping runs
# would contend for the same CPU and corrupt every timing here.
set -e
cd /gly/weilin/spatial-meta-ce-latency-20261025-runs
PY=/gly/weilin/PRICE-master/venv/bin/python
DATA=/gly/weilin/PRICE-master
for db in genome imdb; do
  echo "=== E2E $db ==="
  CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 "$PY" -B scripts/run_pg_e2e.py \
    --database "$db" --allow-database stats ergastf1 imdb genome \
    --workload-all "$DATA/datas/workloads/test/$db/workloads_all.sql" \
    --cache-root /gly/weilin/spatial-meta-ce/artifacts/cache \
    --checkpoint /gly/weilin/spatial-meta-ce/runs/formal-fomaml-20260925-seed42/best.pt \
    --price-checkpoint "$DATA/results/model_params.pth" \
    --sample 20 --runs 3 --statement-timeout-ms 300000 --timeout 600 \
    --output "out/e2e-$db.json"
done
echo "=== EXTRA E2E DONE ==="
