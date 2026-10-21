# IMDB 与源库广度对照 — 新窗口提示词

把下面「提示词正文」整段复制到新窗口即可。

---

## 提示词正文

我在 `spatial-meta-ce` 项目里继续做空间漂移（跨数据库少样本元学习）的评估工作。请先读以下文件，再动手：

**必读（按顺序）**

1. `CLAUDE.md`（项目红线与验证命令）
2. `实验文档/101-IMDB与源库广度对照执行计划.md`（**本任务的完整计划：口径、成本、验收，照它做**）
3. `实验文档/97-P2-2后续补证任务交接计划.md` §11（任务 C 的 PRICE 同口径源侧对照结果与口径边界）
4. `实验文档/98-P2-2三seed复核.md`（方向信号三 seed 复核结果与两条限定）
5. `实验文档/14-离线实验统一对照与论文结论.md` §1–§3（**论文主表的指标口径与不确定性规则**）
6. `scripts/price_population_eval.py`（任务 C 的脚本，本任务要扩展它）

**当前分支**：`experiment/p2-2-three-seed-20261020`（已推送 Gitee，与远端同步）
**你要新建的分支**：`experiment/source-broad-imdb-20261021`（从上面那个切出）
**服务器代码目录**（新建，别复用别的任务目录）：`/gly/weilin/spatial-meta-ce-p2-2-source-broad-20261021-code`

**已确认的事实（不要重新验证，也不要重新推导）**

- **IMDB 的 test 缓存完整可用**：`/gly/weilin/spatial-meta-ce/artifacts/cache/test/imdb`，766 行 / **521 条唯一查询**，
  transform `price40-group-logfanout-v1`、schema 1，真实基数与 PG 估计 100% 有效，表数全 ≥1，与 stats 同构。
  `finetune/imdb` **不存在** → **IMDB 只能做 K=0**（K=0 不需要 support）。
- **PRICE 和我们都没有跑过 IMDB**：`price-zero-summary.json` 只有 stats/genome/ergastf1。
- **PRICE 用全部 26 个源库预训练**（`price-zero-summary.json` 的 `metadata.source_exposure`）；我们只训练了 23 个源库。
  → 4 个目标库是**唯一对等**的漂移比较点；源库上的比较测的是拟合，不是漂移。
- 26 个源库 cache 在 `/gly/weilin/spatial-meta-ce/artifacts/cache/pretrain/`，合计 **1,300,000 行**；
  其中 **`sap` 只有 29,714 条唯一查询**（其余 50,000 / legalacts 49,504）。
- **PRICE 的 `maxima` 是布局参数**（PRICE 原项目 `meta_evaluate.py:156` 按被评估集合取；
  `vendor/price_baseline/module.py` 里 maxima 决定特征在打包输入向量中的读取偏移）。
  任务 C 用的是三源验证库的最大值 `[11,12,7,10]`；**26 源库表必须用 `[12,12,7,10]`**。
- PRICE 的实测速率：**13.9 ms/条**（CPU 单线程、batch 128；任务 C 149,504 条 / 2,078 s）。
  130 万条单进程约 5 小时，**12 进程并行约 25–40 分钟**。
- **我们相对 PG 与 PRICE 的现状**（不要重新跑，直接用）：
  - 目标库 Q95：stats 我们最好 77.32 / PRICE 39.52 / PG 75.65；genome 220.30 / 31.34 / 1907.32；
    ergastf1 23.70 / 15.29 / 27.61 → 三个目标库 Q95 **全输 PRICE**，stats 还输 PG。
  - 源库 K=0 population：chembl 我们 68.23 / PRICE 33.43 / PG 146.06；legalacts **9.71 / 11.82 / 26.00（我们两个都赢）**；
    telstra 7.99 / 5.65 / 6.97。
  - 我们 K=0 的**中位数在三个源库都不如 PG**（2.589/1.935/1.580 vs 2.276/1.512/1.390）。
- 本轮检查时**服务器 8 张 GPU 全被占**（每张 ~10.5 GB、util 60–99%），导出要准备 CPU 兜底；CPU load 约 11/96 核。

**这次任务的目标**

把「面对空间漂移有效果吗」这个问题的**边界测清楚**，分两件：

1. **任务一**：把 **IMDB 补成第四个目标库**（K=0 population 对照 ours / PRICE / PG）。
   这是唯一能新增的、对等的漂移比较点。
2. **任务二**：把 **26 个源库全跑一遍** K=0 population，出**完整 26 行表 + 分布统计**。
   定位是**拟合诊断**（PRICE 见过全部 26 库），不是漂移证据。

**具体要求**

1. **只评估，不训练、不适应、不选 checkpoint、不做新的模型选择。** 用已冻结的 seed42 baseline
   `runs/p1-screen-20261020/baseline/last.pt`（sha256 `563ec159…7d8a3`）与 PRICE 权重
   `/gly/weilin/PRICE-master/results/model_params.pth`（sha256 `4f889c4d…53f92`）。
2. **任务一** 新增 `scripts/target_population_eval.py`（不要改 `export_source_predictions.py` 的 `assert_source()`，
   它拒绝目标库是对的；`evaluate.py` 会去取不存在的 `finetune/imdb`，也不能用）。
   要求：只做 K=0；三个估计量共用同一批 521 条唯一查询；标准 unique-query Q95（不用 all-rows、不用 PRICE shifted）；
   `protocol.json` 必须写 `target_databases_read: ["imdb"]`、**显式传入的 `maxima`** 及其来源、checkpoint 与 PRICE 权重的 sha256、
   cache content_hash、查询数。maxima 必须**与 14 号目标表三行完全一致**（按 `evaluate.py` 同一逻辑复算：
   三个目标库 × {test, finetune} 的 `inspection.max_counts` 逐元素最大值），不因加入 imdb 而改动其他三行。
3. **任务二** 的我们侧：新增一份只导 K=0 的配置（沿用 `configs/eval_p0_3.json`，只改 `shots: [0]` 与 `support_runs: 1`），
   用 `scripts/export_source_predictions.py` 对 26 个库导出（`--databases` 传 26 个非目标库）。
   约 130 万行。
4. **任务二** 的 PRICE 侧：扩展 `scripts/price_population_eval.py`，**只加可选参数**
   （`--maxima` 显式覆盖、`--jobs N` 按库进程池并行），默认行为必须逐位不变。
   **改完必须做回归检查**：用任务 C 的原始命令重跑，比对
   `实验文档/results/p2-2-price-population-20261020/summary.json` 是否逐字节相同；不一致就改成纯增量。
5. **必做布局敏感性自检**：用 `[12,12,7,10]` 重跑 chembl，与任务 C 的 chembl 行（`[11,12,7,10]`）对比，
   把差异写进报告——差异小则两批可互为佐证，差异大则 26 行表只在自己内部可比，任务 C 的三行不可与之并列。
6. **任务二必须报告全部 26 行**，不得只展示某一档；另给三个分布统计：我们优于 PG 的库数、我们优于 PRICE 的库数、
   比值（PRICE/我们、PG/我们）的 median/min/max，并按「优于 PG 且优于 PRICE / 优于 PG 但劣于 PRICE / 劣于 PG」
   三档分类计数。
7. 结论写入新文档 `实验文档/102-IMDB与源库广度对照结果.md`；
   证据 JSON 存 `实验文档/results/p2-2-target-imdb-20261021/` 与
   `实验文档/results/p2-2-price-population-broad-20261021/`。
8. 更新 `实验文档/99-进度与下一步.md`。

**必须遵守的红线**

- 目标库 `imdb/stats/genome/ergastf1` **不进入训练、超参选择、checkpoint 选择**。
  任务一只做冻结 checkpoint 的最终评估，且必须把 `target_databases_read: ["imdb"]` **显式写进产物**（不是 `[]`）。
- **不裁剪输出**：IMDB 的 521 条唯一查询全部参与，样本量小就如实标注。
- 数字只写实跑输出；单 seed（42）结论必须标 `needs_evidence`。
- 任何两次 PRICE 数字并列前先确认 `maxima` 相同。
- 源库行必须在报告里明确写成 **in-sample 拟合诊断**，不是漂移证据（见 101 号 §1 的曝光矩阵）。
- 不得按结果挑库、挑子集；26 行全报。
- `/gly/weilin/PRICE-master` 只读，不修改。
- 改动前开新分支（`experiment/` 前缀），完成后 push Gitee。
- 服务器验证用 `ssh myserver "..."`；GPU 任务显式指定 `CUDA_VISIBLE_DEVICES` 并先 `nvidia-smi` 确认卡空闲；
  并行前看 `uptime`，机器是共享的，别把别人的作业挤掉。
- 本地改完必须跑通 `PYTHONDONTWRITEBYTECODE=1 python3 -B -m unittest discover -s tests`
  和 `ruff check scripts tests spatial_meta_ce`。
  注意：全量套件有**既有**偶发失败（`tests/test_meta65_joint_supervisor.py` 的进程监管时序项，
  父提交上 6 跑 3 败、该文件单跑 40 项全过），与本任务无关，别去追。

**服务器部署要点（本轮实测踩过的）**

- 新的代码目录要自己建，并把 cache 指过去：`ln -s /gly/weilin/spatial-meta-ce/artifacts/cache <code_dir>/artifacts/cache`。
- `configs/eval_p0_3.json` **不在 git 里**（只在服务器 `.../p1-loss-gate-20261020-code/configs/`），需要从那里复制。
- macOS 自带 rsync 不支持 `--info=progress2`；服务器 rsync 也不一定支持，用朴素 `rsync -az`。
- 拉大文件（364 MB 级）用 `rsync -az`，别加花哨参数。

**预期交付**

- `实验文档/102-IMDB与源库广度对照结果.md`（含 IMDB 行、26 行表、分布统计、布局敏感性自检、判定）
- 两份证据 JSON（各带 `protocol.json`，含 `maxima`、sha256、`target_databases_read`）
- `scripts/target_population_eval.py` + `tests/test_target_population_eval.py`（新）
- `scripts/price_population_eval.py` 的可选参数扩展 + 对应单测
- 更新 `实验文档/99-进度与下一步.md`
- 分支 `experiment/source-broad-imdb-20261021` 已 push Gitee

**如果卡住**

服务器连不上、数据缺失、或发现口径有问题（比如 `maxima` 复算不出来、PRICE 加载失败），
直接停下来说明具体阻塞点，不要循环重试（同一命令最多 3 次）。
**也请不要为了让结果好看去改口径**——本任务的成功标准是把漂移问题的边界测清楚，
不是找到我们占优的库；该是负结果就如实写负结果。
