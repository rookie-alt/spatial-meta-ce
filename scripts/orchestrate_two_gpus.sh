#!/usr/bin/env bash
# Prepare gate + two independent FOMAML seeds on GPUs 0 and 7.
# This script does not modify the original PRICE-master; launch only after
# both physical GPUs have been verified free, and recheck before execution.
set -Eeuo pipefail

ROOT=/gly/weilin/spatial-meta-ce
PY=/gly/weilin/PRICE-master/venv/bin/python
PREP_PID=${1:?Pass the PID of the already-running source cache preparation}
RUN_TAG=${2:?Pass a unique run tag}
cd "$ROOT"
export PYTHONDONTWRITEBYTECODE=1

DATASETS=(accidents airline baseball basketball carcinogenesis ccs chembl consumer credit employee financial fnhk grants hepatitis hockey legalacts movielens sakila sap seznam ssb talkingdata telstra tournament tpc_h tubepricing)
COUNT=${#DATASETS[@]}
SEED42="runs/${RUN_TAG}-seed42"
SEED43="runs/${RUN_TAG}-seed43"
if [[ -e "$SEED42" || -e "$SEED43" || -e "${SEED42}.log" || -e "${SEED43}.log" ]]; then
    printf 'OUTPUT_PATH_EXISTS: refusing overwrite\n'
    exit 70
fi

while true; do
    ready=0
    for db in "${DATASETS[@]}"; do
        [[ -f "artifacts/cache/pretrain/${db}/manifest.json" ]] && ((++ready))
    done
    printf '%s SOURCE_CACHE_GATE %s/%s\n' "$(date -Is)" "$ready" "$COUNT"
    [[ "$ready" -eq "$COUNT" ]] && break
    if ! kill -0 "$PREP_PID" 2>/dev/null; then
        printf 'SOURCE_CACHE_FAILED: preparing process %s stopped before 26/26; no GPU training started\n' "$PREP_PID"
        exit 71
    fi
    sleep 45
done

for db in "${DATASETS[@]}"; do
    "$PY" -B -c 'import sys; from spatial_meta_ce.data import FeatureStore; db=sys.argv[1]; store=FeatureStore("artifacts/cache/pretrain/"+db); assert len(store)>100' "$db"
done
printf 'ALL_SOURCE_MANIFESTS_VALIDATED count=%s\n' "$COUNT"

# Require five GiB free for logs/checkpoints even though shared storage is
# displayed as 100% used when its quota is near the limit.
free_bytes=$("$PY" -B -c 'import os; s=os.statvfs("."); print(s.f_bavail*s.f_frsize)')
if (( free_bytes < 5 * 1024 * 1024 * 1024 )); then
    printf 'DISK_GATE_FAILED free_bytes=%s\n' "$free_bytes"
    exit 72
fi

for _ in 1 2; do
    for gpu in 0 7; do
        memory=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$gpu" | tr -d '[:space:]')
        if [[ ! "$memory" =~ ^[0-9]+$ ]] || (( memory > 100 )); then
            printf 'GPU_GATE_FAILED gpu=%s memory_mib=%s; no training started\n' "$gpu" "$memory"
            exit 73
        fi
        printf '%s GPU_AVAILABLE gpu=%s memory_mib=%s\n' "$(date -Is)" "$gpu" "$memory"
    done
    sleep 5
done

# Each process sees exactly one physical GPU. It is NOT a cross-GPU gradient
# synchronization run; the independent seeds are necessary to estimate variance.
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 nohup "$PY" -B -u -m spatial_meta_ce.cli train \
    --config configs/meta.json --seed 42 --device cuda --output "$SEED42" \
    > "${SEED42}.log" 2>&1 < /dev/null &
PID42=$!
CUDA_VISIBLE_DEVICES=7 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 nohup "$PY" -B -u -m spatial_meta_ce.cli train \
    --config configs/meta.json --seed 43 --device cuda --output "$SEED43" \
    > "${SEED43}.log" 2>&1 < /dev/null &
PID43=$!
printf '%s TRAINING_STARTED gpu0_seed42_pid=%s gpu7_seed43_pid=%s\n' "$(date -Is)" "$PID42" "$PID43"
printf '%s\n' "$PID42" > "${SEED42}.pid"
printf '%s\n' "$PID43" > "${SEED43}.pid"

set +e
wait "$PID42"
S42=$?
wait "$PID43"
S43=$?
set -e
printf '%s TRAINING_FINISHED seed42_status=%s seed43_status=%s\n' "$(date -Is)" "$S42" "$S43"
if (( S42 != 0 || S43 != 0 )); then exit 74; fi
