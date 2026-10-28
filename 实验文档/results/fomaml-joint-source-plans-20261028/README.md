# FOMAML 联合目标源侧计划生成记录（2026-10-28）

- 代码分支：`feat/fomaml-joint-objective-20261028`，提交 `58c3a40`（本地工作树部署到服务器）。
- 服务器代码目录：`/gly/weilin/spatial-meta-ce-fomaml-joint-source-20261028-code`
- 服务器计划目录：`/gly/weilin/spatial-meta-ce-fomaml-joint-source-20261028-plans`
- 生成命令：

```bash
cd /gly/weilin/spatial-meta-ce-fomaml-joint-source-20261028-code
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' \
  /gly/weilin/PRICE-master/venv/bin/python -B scripts/build_fomaml_joint_plans.py \
  --config configs/fomaml_joint_sum_4k.json \
  --out /gly/weilin/spatial-meta-ce-fomaml-joint-source-20261028-plans \
  --seeds 42 43 44
```

生成物只含计划 JSON，不读目标库、不训练、不推理、不评估。训练计划每 seed 16,000 episodes，覆盖 23 个训练库；source-eval 计划每库 100 条 support（5×20），Q_d 为 chembl 49,900 / legalacts 49,404 / telstra 49,900。

## 核对

服务器上用 `load_training_plan` / `load_source_eval_plan` 重新校验 schema、digest、support/query 隔离与 store 内容 hash；`verify_plan_against_stores` 和 `verify_source_eval_plan_against_stores` 全部通过。另用相同命令生成到 `-plans-recheck`，四个计划文件的 SHA256 与原目录逐一相同。

文件 SHA256、plan SHA256、文件大小与每库计数见 [verification.json](verification.json)；服务器原始 `plan_binding.json` 已复制到本目录。

## 边界

本次只做只读计划生成与 hash 核对；未启动训练、smoke 或任何模型评估，未读取 `imdb/stats/genome/ergastf1`。计划文件体积约 463 MB，仅保留在服务器；仓库只归档 hash 与小体积 binding/verification。
