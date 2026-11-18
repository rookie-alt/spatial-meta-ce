#!/bin/bash
# Operational queue only. Every run uses the frozen, supervised module runner.
set -euo pipefail

module_base=/gly/weilin/module26-prep-20261118
module_launch="$module_base/training-launch-20261119"
module_code="$module_base/code-training"
module_python=/gly/weilin/PRICE-master/venv/bin/python

cd "$module_launch"
sha256sum --check queue.tsv.sha256
exec 9>queue.lock
flock --nonblock 9
test ! -e queue.status
printf '%s\n' running > queue.status

finish_queue() {
    queue_rc=$?
    trap - EXIT
    if [ "$queue_rc" -eq 0 ]; then
        printf '%s\n' completed > "$module_launch/queue.status"
    else
        printf 'stopped exit=%s\n' "$queue_rc" > "$module_launch/queue.status"
    fi
    printf 'queue_end time=%s exit=%s\n' "$(date -Is)" "$queue_rc"
    exit "$queue_rc"
}
trap finish_queue EXIT

export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
printf 'queue_start time=%s pid=%s gpu=1\n' "$(date -Is)" "$$"

while IFS=$'\t' read -r module_key module_spec module_spec_sha module_auth module_auth_sha; do
    printf '%s\n' "$module_key" > current-run.txt
    printf 'run_start time=%s key=%s\n' "$(date -Is)" "$module_key"
    cd "$module_code"
    "$module_python" -I -B scripts/run_official_module_experiment.py run \
        --spec "$module_spec" --spec-sha256 "$module_spec_sha" \
        --authorization "$module_auth" --authorization-sha256 "$module_auth_sha" \
        --execute
    cd "$module_launch"
    printf 'run_complete time=%s key=%s\n' "$(date -Is)" "$module_key" >> completed-runs.log
done < "$module_launch/queue.tsv"
