#!/bin/bash
# 23 个源训练库 mapping sidecar 构建：纯 CPU、PRICE 树只读、不触目标库
# 启动: bash launch.sh（在服务器 /gly/weilin/spatial-meta-ce 下执行）
# 查看: status 用 python3 -c 读 $OUT/status.json；日志 $OUT.log；不轮询
set -euo pipefail
P=/gly/weilin/spatial-meta-ce
cd "$P"
OUT=runs/p1-mapping-sidecar-train23-20261004
AUDIT=$P/runs/p1-input-audit-20261002
SHA=1228d153514f8e08ed27cf298119863b11e61e15347bc7ca2392dd9aa0f1c98f
DBS="accidents airline baseball basketball carcinogenesis ccs consumer credit employee financial fnhk grants hepatitis hockey movielens sakila sap seznam ssb talkingdata tournament tpc_h tubepricing"
nohup env PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES="" \
  /gly/weilin/PRICE-master/venv/bin/python -B scripts/build_mapping_sidecar.py \
  --config configs/data.json \
  --audit "$AUDIT" --audit-summary-sha256 "$SHA" \
  --databases $DBS \
  --output "$OUT" \
  > "${OUT}.log" 2>&1 &
echo "PID $!"
