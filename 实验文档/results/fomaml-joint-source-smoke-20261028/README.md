# 112 smoke / 计时预检记录（2026-10-28）

- 代码分支：`feat/fomaml-joint-objective-20261028`；服务器代码目录 `...-fomaml-joint-source-20261028-code`。
- 输出目录：`/gly/weilin/spatial-meta-ce-fomaml-joint-source-20261028-smoke-20261028`
- 运行命令：

```bash
cd /gly/weilin/spatial-meta-ce-fomaml-joint-source-20261028-code
CUDA_VISIBLE_DEVICES=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 \
MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
/gly/weilin/PRICE-master/venv/bin/python -B scripts/run_fomaml_joint_smoke.py \
  --data-config configs/data.json \
  --source-plan /gly/weilin/spatial-meta-ce-fomaml-joint-source-20261028-plans/source_eval_plan.json \
  --out /gly/weilin/spatial-meta-ce-fomaml-joint-source-20261028-smoke-20261028 \
  --max-seconds 1800
```

全部产物标记 `purpose="smoke-not-research"`，不是正式训练或精度结果。`smoke_protocol.json` 文件 SHA256：`cedf65eb8c9ba85a8254aa5d6ba332791e5662d81d9b51d5cce6edf0aaa621ee`。

## 实测

| 阶段 | 结果 |
|---|---|
| 重载 smoke（2 updates） | 47.10 s；`reload_next_step=2`；`best.pt` 不存在；`gradient_audit.json` 存在；无 validation 读取 |
| A_erm 200 updates | 68.96 s；`clipped_count=20/200` |
| B_fomaml 200 updates | 111.23 s；`clipped_count=94/200` |
| C_joint 200 updates | 125.85 s；`clipped_count=93/200` |
| 峰值 RSS | 2,584,128 KiB（约 2.46 GiB） |
| CUDA allocator 峰值 | 29,101,056 B（约 27.8 MiB）；GPU 进程观测约 0.94 GiB |
| 输出体积 | 17 MB |
| source eval | chembl K0 12.18 s / K20 predict 10.90 s；legalacts 11.45 / 10.88 s；telstra 11.62 / 10.67 s；K20 adapt 0.07–0.11 s |

三臂 200-step 计时只说明 smoke 资源量级：C 比 B 慢约 13%，比 A 慢约 83%；不作配方选择、精度比较或正式预算外推。

## 边界

- 只读取 23 个训练库；source eval 阶段读取 `chembl/legalacts/telstra`，在 protocol 的 `validation_monitor_reads` 单独记录；目标库 `imdb/stats/genome/ergastf1` 未读取。
- 无正式训练、无精度表、无目标评估；smoke checkpoint 不接续正式训练。
- 未使用 `best.pt`；每臂只有 `last.pt`，200 步 checkpoint 仅用于本次 smoke。
