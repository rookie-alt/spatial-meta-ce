#!/bin/sh
# Doc 110 full measurement, sequential: overlapping runs would contend for the
# same CPU threads and corrupt every timing in this file.
set -e
cd /gly/weilin/spatial-meta-ce-latency-20261025-runs
PY=/gly/weilin/PRICE-master/venv/bin/python
CACHE=/gly/weilin/spatial-meta-ce/artifacts/cache
DATA=/gly/weilin/PRICE-master
PRICE=$DATA/results/model_params.pth
FOMAML=/gly/weilin/spatial-meta-ce/runs/formal-fomaml-20260925-seed42/best.pt
ERM=/gly/weilin/spatial-meta-ce/runs/formal-erm-seed42-20260926/best.pt
DBS="imdb stats genome ergastf1"
mkdir -p out

l1() {
  label=$1; ckpt=$2; dev=$3; vis=$4; price=$5
  echo "=== L1 $label ==="
  CUDA_VISIBLE_DEVICES="$vis" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
    "$PY" -B scripts/measure_inference_latency.py --cache-root "$CACHE" --checkpoint "$ckpt" \
    ${price:+--price-checkpoint "$price"} --databases $DBS --allowed-targets $DBS \
    --device "$dev" --threads 1 --repeat-times 100 --output "out/latency-$label.json"
}

l2() {
  label=$1; ckpt=$2; dev=$3; vis=$4; price=$5
  echo "=== L2 $label ==="
  CUDA_VISIBLE_DEVICES="$vis" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
    "$PY" -B scripts/measure_estimation_latency.py --cache-root "$CACHE" --data-root "$DATA" \
    --checkpoint "$ckpt" ${price:+--price-checkpoint "$price"} --databases $DBS \
    --allowed-targets $DBS --device "$dev" --threads 1 --verification-sample 20 --warmup 5 \
    --output "out/estimation-$label.json"
}

l1 cpu-fomaml "$FOMAML" cpu "" "$PRICE"
l1 cuda-fomaml "$FOMAML" cuda:0 0 "$PRICE"
l1 cpu-erm "$ERM" cpu "" ""
l1 cuda-erm "$ERM" cuda:0 0 ""
l2 cpu-fomaml "$FOMAML" cpu "" "$PRICE"
l2 cuda-fomaml "$FOMAML" cuda:0 0 "$PRICE"
l2 cpu-erm "$ERM" cpu "" ""
l2 cuda-erm "$ERM" cuda:0 0 ""

for db in stats ergastf1; do
  echo "=== E2E $db ==="
  CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 "$PY" -B scripts/run_pg_e2e.py \
    --database "$db" --allow-database stats ergastf1 \
    --workload-all "$DATA/datas/workloads/test/$db/workloads_all.sql" \
    --cache-root "$CACHE" --checkpoint "$FOMAML" --price-checkpoint "$PRICE" \
    --sample 20 --runs 3 --output "out/e2e-$db.json"
done

echo "=== ALL DONE ==="
ls -l out
