# Spatial Meta-CE

面向空间漂移的轻量级、监督式跨数据库基数估计实验。使用现有 PRICE 查询特征和真实基数，不生成随机元组替代真实数据。

## 当前状态（2026-09-25）

- 已实现：99,777 参数集合模型、FOMAML、同结构 ERM、无状态少样本适应、严格 checkpoint、数据库级划分、重复 SQL 排除、Q-error、运行 manifest、PG/PRICE/随机初始化基线和正式多种子运行器。
- 已在服务器实际 Python 3.11 / PyTorch 2.1 / SQLGlot 18.17 环境通过核心与协议测试；最新测试数量/输出见 `reports/server-tests.txt`。
- 全量审计了 33 个现有特征文件、1,457,239 条样本。IMDB finetune 特征缺失，不计为通过。
- 使用 fnhk 训练、chembl 验证，在 STATS / Genome / ErgastF1 上完成真实数据 CPU smoke。四目标库完整验收**尚未完成**：IMDB finetune 统计缺少必要列，无法完整重建。
- 只准备了 smoke 所需的两个源库以及目标库现有可用分区缓存；正式 23/3 训练前须执行全部源库 prepare。尚未执行正式 GPU 训练、3-seed × 5-support-repeat 实验，不能声称精度优于基线。
- 原项目 46 个审查源码文件 hash 与开始时相同。新项目位于 `/gly/weilin/spatial-meta-ce`，原数据只读。

## 模型与数据

四种变长集合：连接直方图、fanout、表、过滤条件。每组 token 经过共享 MLP 和 masked mean/max 池化，回归头预测 PG 对数估计的残差。支持不同数量的表/列，不依赖训练集最大维度。

- 参数：99,777；纯 FP32 参数 399,108 字节（约 390 KiB）。
- 监督：真实 `SQL || cardinality || PG estimate`；预测目标为 log-cardinality。
- 元训练：23 个源库；验证：chembl/legalacts/telstra；目标：imdb/stats/genome/ergastf1。
- Support 来自目标 finetune，test 使用独立 test workload。SQL AST 规范化保留字面量，去除相同 query hash。
- 训练不裁剪输出，不伪造缺失标签，不以训练 Q95 选 checkpoint。

详细设计见 [实验协议](docs/protocol.md)，旧代码问题见 [审查报告](docs/audit.md)。

## 使用环境

以下命令在服务器 `/gly/weilin/spatial-meta-ce` 内执行。复用已有环境，不修改 PRICE-master：

```bash
cd /gly/weilin/spatial-meta-ce
```

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES="" /gly/weilin/PRICE-master/venv/bin/python -B -m unittest discover -s tests -v
```

`torch/numpy/sqlglot` 版本见 `requirements.txt`。仅加载用户可信的 pickle/.pth；这两类格式不能安全用于任意不可信下载。

## 审计与缓存

只读审计全部现有特征：

```bash
CUDA_VISIBLE_DEVICES="" /gly/weilin/PRICE-master/venv/bin/python -B -m spatial_meta_ce.cli audit --config configs/data.json
```

生成指定数据库的紧凑、带来源缓存：

```bash
CUDA_VISIBLE_DEVICES="" /gly/weilin/PRICE-master/venv/bin/python -B -m spatial_meta_ce.cli prepare --config configs/data.json --databases fnhk chembl stats genome ergastf1
```

完整准备（IMDB 当前会按设计报错，需要先解决其统计缺失）：

```bash
CUDA_VISIBLE_DEVICES="" /gly/weilin/PRICE-master/venv/bin/python -B -m spatial_meta_ce.cli prepare --config configs/data.json
```

缓存写入 `artifacts/cache/`，绝不覆盖原始 features。缓存已有且源 hash 一致时复用；不同 hash 时要求新的 cache_root。`--allow-exclusions` 可记录并排除无法提取的真实查询，但**未经用户确认不要用于 IMDB**：300 条随机抽查仅 30 条可完整生成，筛选会改变支持分布。

## CPU smoke

完整四库 smoke 要求四库 support cache 都可用，目前因 IMDB 阻塞。已完成的三库诊断报告在 `reports/smoke_summary.json`，不是正式精度实验。

```bash
CUDA_VISIBLE_DEVICES="" /gly/weilin/PRICE-master/venv/bin/python -B -m spatial_meta_ce.cli smoke --config configs/meta.json --device cpu
```

## 单次训练与评估

确认 GPU 授权和空闲卡后，使用 CUDA_VISIBLE_DEVICES 显式指定 GPU；以下仅给 CPU 命令，不会自动启动长期任务。

```bash
/gly/weilin/PRICE-master/venv/bin/python -B -m spatial_meta_ce.cli train --config configs/meta.json --device cpu --output runs/meta-seed42
```

```bash
/gly/weilin/PRICE-master/venv/bin/python -B -m spatial_meta_ce.cli train --config configs/erm.json --device cpu --output runs/erm-seed42
```

```bash
/gly/weilin/PRICE-master/venv/bin/python -B -m spatial_meta_ce.cli train --config configs/meta.json --device cpu --output runs/meta-seed42 --resume runs/meta-seed42/last.pt --steps 3000
```

评估初始化：

```bash
/gly/weilin/PRICE-master/venv/bin/python -B -m spatial_meta_ce.cli evaluate --config configs/evaluation.json --checkpoint runs/meta-seed42/best.pt --output runs/meta-seed42-eval
```

生成单库适应模型：

```bash
/gly/weilin/PRICE-master/venv/bin/python -B -m spatial_meta_ce.cli adapt --config configs/evaluation.json --checkpoint runs/meta-seed42/best.pt --database stats --k 50 --output runs/adapted-stats
```

已适应 checkpoint 用于部署，不允许回到 evaluate 当作独立元初始化反复适应，避免跨 episode 污染。

可用选项：`--seed`、`--inner-lr`、`--inner-steps`、`--scope head`、`--no-pg`。训练/评估模型配置和预处理必须一致；不用 PG 的 checkpoint 在评估时也要指定 `--no-pg`。

PG / 随机初始化 / PRICE 基线可用 `evaluate --baseline pg|random|price`。PRICE 保留其 log1p(card)+1 标度和原前向；微调超参数要在源验证集确定，不能直接照搬轻量模型学习率。历史 smoke 统一 LR 的 PRICE 适应发生明显恶化，保留为诊断，不作为比较证据。

## 正式实验运行器

默认只打印计算任务，不执行：

```bash
/gly/weilin/PRICE-master/venv/bin/python -B scripts/run_experiments.py --output runs/formal
```

确认数据和计算预算后，再添加 `--execute --device cuda` 并设置可用卡。运行器先完成所有源验证选择并写 `frozen_selection.json`，随后才访问目标测试进行评估：

- 主模型：ERM / FOMAML，3 个种子，源验证选择内循环 LR。
- PRICE 和随机初始化适应 LR 也只在源验证库选择；PRICE 使用 1e-6/1e-5/1e-4 候选，而不是按目标测试反馈挑选。
- 目标 K=0/5/10/20/50/100，每 K 五次支持抽样。
- `--ablations` 增加无 PG、仅适应 head 的变体。
- 指标在每库内报告，支持随机性和训练种子随机性分开聚合。

## 产物

每次训练：`config.json`、固定 `validation_manifest.json`、`history.json`、`last.pt`、`best.pt`。

每次评估：`manifest.json`（支持 ID、源 hash、完整设置）、`predictions.csv`（逐查询）、`summary.json`（标准 Q-error、独立平滑指标、unique-query、耗时）。正式多种子运行器还输出 `across_training_seeds.json`。

推理耗时不包括 SQL 特征生成、真实标签获取和数据库执行，不能据此宣称数据库端到端加速。集合编码器不显式编码完整连接图；模型是否有精度优势必须由完整实验决定。

## 部署与许可

`scripts/deploy.py` 只更新自身 manifest 管理的源码，拒绝覆盖服务器手工改动，不删除 artifacts/runs；通过 SSH key/agent 或外部 wrapper 认证，不保存密码。

`vendor/PROVENANCE.json` 记录三份 PRICE 代码来源和修改。原 checkout 没有 LICENSE/NOTICE，本项目仅供当前私有研究；公开代码前先核对上游许可。
