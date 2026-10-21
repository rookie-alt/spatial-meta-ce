# 101-IMDB 与源库广度对照执行计划

日期：2026-10-21
用途：**新窗口执行用**。本文只写计划、口径与验收，不含结果。
前置阅读：[97号](97-P2-2后续补证任务交接计划.md)、[98号](98-P2-2三seed复核.md)、[100号](100-论文正文整合执行计划.md)、任务 C 的 [scripts/price_population_eval.py](../scripts/price_population_eval.py)

> **命名消歧**：本文的「任务一 / 任务二」与 [97号 §5](97-P2-2后续补证任务交接计划.md) 的
> **任务 B（23 库跨库迁移 LODO）**是三件不同的事，不要混。本计划不含 97 号任务 B。

---

## 0. 目标

用户诉求：再找几个库，找我们**与 PRICE 接近、比 PG 好**的案例。拆成两件可执行的事：

- **任务一**：把 **IMDB 补成第四个目标库**（K=0 population 对照）。这是**唯一对等**的漂移比较。
- **任务二**：把 **26 个源库全跑一遍** K=0 population，出**完整 26 行表**（含分布统计，不挑子集）。

**任务一必须做，任务二必须做。** 理由见 §1：任务二单独做出来是拟合诊断而非漂移证据，
但它和任务一合起来才能把「面对空间漂移有效果吗」这个问题说完整。

---

## 1. 曝光矩阵（这是两份计划的立论点，必须先讲清楚）

项目自己的记录（`实验文档/results/price-zero-summary.json` 的 `metadata.source_exposure`）写着：
**「Provided PRICE weights trained on all 26 source databases; supplemental reference, not matched 23/3 initialization」**。

| 库 | 我们训练过 | PRICE 训练过 | 这个比较测的是什么 |
|---|---|---|---|
| 23 个源训练库 | 是 | **是** | 拟合能力，**不是空间漂移** |
| 3 个源验证库 | 否（但用于选方案） | **是** | 我们 out-of-sample vs PRICE in-sample |
| hepatitis / sakila（81号留出） | 否 | **是** | 只有我们是真漂移，PRICE 不是 |
| 4 个目标库 | 否 | 否 | **唯一对等的漂移比较** |

**因此**：
- 在源库上找到「我们比 PRICE 好」的案例是可能的，但那个证据测的是**拟合**；
  PRICE 见过全部 26 个库。对漂移问题的说服力弱于目标库。
- **IMDB 是唯一能新增的、对等的漂移比较点**——它同时不在 PRICE 的 26 源库、也不在我们的 23 训练库。
- 任务二的价值是**诊断**：把「模型弱」和「迁移失败」分开（seen 基准），并给出「在 N/26 个库上我们优于 PG」的分布结论。这比挑 3 个库写进论文强得多。

**报告纪律（红线级）**：任务二必须**承诺报告全部 26 行**。先跑一遍再挑「好看的几个」= 按目标结果挑，
这不但不能写进论文，还会让现有 3 个目标库的结论一起被怀疑。

---

## 2. 红线（与 CLAUDE.md 一致）

- 目标库 `imdb/stats/genome/ergastf1` **不进入训练、超参选择、checkpoint 选择**。
  任务一只做**冻结 checkpoint 的最终评估**，且必须把 `target_databases_read: ["imdb"]` **显式写进 protocol.json**。
- **不训练、不适应、不选 checkpoint、不做新的模型选择**。任务一用已冻结的 seed42 baseline `last.pt`。
- **不裁剪输出**：IMDB 的 521 条唯一查询全部参与，样本量小就如实标小。
- 数字只写实跑输出；单 seed 结论标 `needs_evidence`。
- `/gly/weilin/PRICE-master` 只读。
- 同一命令最多重试 3 次，仍失败就停下汇报。
- 改动前开新分支 `experiment/source-broad-imdb-20261021`，完成后 push Gitee。

---

## 3. 口径的关键决定：PRICE 的 `maxima` 是**布局参数**，不是固定值

这一点必须在新窗口开工前就钉住，否则两批数字不可比。

- 读 PRICE 原项目 `meta_evaluate.py:149-175`：它**按被评估集合取 maxima**
  （`max_n_join_col = max(test_n_join_cols)` …），再把同一组 maxima 传给模型与输入构造。
- 读 `vendor/price_baseline/module.py:150-210`：`n_join_col / n_fanout / n_table / n_filter_col`
  只作为**切分输入向量的循环计数与偏移**（`bias = n_join_col*hist_dim + …`），
  不改变 `nn.Linear` 的形状，所以**换 maxima 不会导致权重加载失败**——
  但会**改变特征被读到的位置**，结果随之改变。
- 结论：`maxima` 是**必须写进 protocol.json 的一等口径参数**，两次运行的 maxima 不同则数字不可并列。

### 3.1 两份任务各自该用哪个 maxima

| 任务 | maxima | 理由 |
|---|---|---|
| 任务一（IMDB） | **与 14 号目标表完全相同**的那一组 | 目标表三行的布局不能因加入 imdb 而变。复算方式：`evaluate.py` 的同一逻辑——`splits["test"]` 的各目标库 × {`test`,`finetune`} split 的 `inspection.max_counts` 逐元素最大值（历史那次 test 列表是 stats/genome/ergastf1，finetune 只有这三个）。把复算结果与「为何不取 4 目标库或 26 源库的全局值」一起写进 protocol.json |
| 任务二（26 源库） | **26 个源库 pretrain cache 的逐元素最大值 = [12, 12, 7, 10]** | 这 26 个库就是 PRICE 的训练集，取它们的最大值最接近 PRICE 的训练布局。注意：**任务 C 用的是三源验证库的最大值 [11, 12, 7, 10]**，只差 `n_join_col` 一位（11→12） |

### 3.2 必须做的布局敏感性自检

用任务二的 maxima `[12,12,7,10]` **重跑 chembl**，与任务 C 的 chembl 行（`[11,12,7,10]`）对比，
量化布局差异有多大。这个对比要写进 protocol.json / 报告：
- 若差异很小 → 说明结论对布局不敏感，任务 C 与任务二可以互为佐证；
- 若差异明显 → 任务二的 26 行表**只在自己内部可比**，任务 C 的三行数字**不可与之并列**，必须分别标注。

---

## 4. 任务一：IMDB 作为第四个目标库（K=0 population）

### 4.1 已核实的前置条件（服务器 `/gly/weilin/spatial-meta-ce/artifacts/cache/`）

| 项 | 值 |
|---|---|
| `test/imdb` 存在 | ✅ 766 行 / **521 条唯一查询** |
| transform / schema | `price40-group-logfanout-v1` / `1`（与 stats 一致） |
| `cards[:,0]` 真实基数 | 100% > 0 |
| `cards[:,1]` PG 估计 | 100% > 0，无 0 值 |
| 表数 | 全部 ≥ 1 |
| `finetune/imdb` | ❌ 不存在 → **few-shot 不可做，只能 K=0** |
| 历史是否跑过 | ❌ PRICE（`price-zero-summary.json` 只有 stats/genome/ergastf1）与我们都没有 IMDB 的任何结果 |

### 4.2 实现：新增 `scripts/target_population_eval.py`

不能复用现有脚本，原因有二：
- `scripts/export_source_predictions.py` 的 `assert_source()` **明确拒绝目标库**（这是对的，不要改它）；
- `spatial_meta_ce/evaluate.py` 会去取 `pool.get("finetune", db)`，imdb 没有这个 split，必然崩。

新脚本职责（**只评估，不选择**）：

```
scripts/target_population_eval.py
  --cache-root      artifacts/cache
  --split           test
  --databases       imdb            # 目标库列表；写进 target_databases_read
  --checkpoint      runs/p1-screen-20261020/baseline/last.pt   # 冻结，不选
  --price-checkpoint /gly/weilin/PRICE-master/results/model_params.pth
  --maxima          "9,12,7,10"      # 显式传入，见 §3.1
  --output / --protocol
```

行为要求：
1. **只做 K=0**：不采 support、不适应、不比较 K。
2. 三个估计量共用同一批唯一查询：`ours_k0`（冻结 checkpoint）、`price_k0`（解码到 log 空间）、
   `pg`（`cards[:,1]`）。
3. 指标沿用 14 号主表口径：标准 **unique-query Q95**，不用 all-rows、不用 PRICE shifted。
4. `protocol.json` 必须含：`target_databases_read: ["imdb"]`、`maxima` 及其来源、
   checkpoint 的 sha256、PRICE 权重的 sha256、cache 的 content_hash、查询数 521。
5. **守卫**：脚本只允许对**已冻结**的 checkpoint 出报告；若传入的目标库不在显式列表里就报错退出。

### 4.3 单测

`tests/test_target_population_eval.py`，至少覆盖：
- 目标库**必须**被显式记录进 `target_databases_read`（不是 `[]`）；
- 未在列表里的库被拒绝；
- `maxima` 显式传入且被写进 protocol；
- 三个估计量在合成 cache 上对齐（复用 `tests/test_price_population_eval.py` 的 FakeStore 手法）。

---

## 5. 任务二：26 个源库全表（K=0 population）

### 5.1 已核实的源库清单（服务器 `artifacts/cache/pretrain/`，26 个，合计 1,300,000 行）

23 训练库（`accidents … tubepricing`）+ 3 源验证库（`chembl` / `legalacts` / `telstra`）。
注意 **`sap` 只有 29,714 条唯一查询**（50,000 行里有重复），其余库都是 50,000 / 49,504。
`legalacts` 49,504（与源验证那次一致）。

### 5.2 我们的 K=0 预测

新增一份只导 K=0 的配置（沿用 `configs/eval_p0_3.json`，只改两处）：

```json
{ "shots": [0], "support_runs": 1, ...其余字段与 configs/eval_p0_3.json 完全相同... }
```

- `support_runs=1` 合法：K=0 不采 support，K=0 行在各 run 间本来就完全相同（任务 C 的审计已验证过）。
- 导出用 `scripts/export_source_predictions.py`（它允许任意非目标库，`--databases` 传 26 个）。
- 产物约 130 万行 / 250 MB CSV。

### 5.3 PRICE 的 K=0

扩展 `scripts/price_population_eval.py`（**只加可选参数，默认行为必须逐位不变**）：

| 新参数 | 作用 |
|---|---|
| `--maxima "12,12,7,10"` | 显式 maxima 覆盖（默认仍按库列表计算，保持任务 C 行为） |
| `--jobs N` | 按库并行：进程池，每个 worker 独立建 `PriceAdapter`（同一组 maxima）、`torch.set_num_threads(1)`，主进程汇总成一张表 |

**若不想改既有脚本**，替代方案：按 DB 分片跑多次（每次 `--databases <若干>` + `--maxima` 显式），
再写一个 20 行的合并脚本。**但必须先做回归检查**（§6.3）。

### 5.4 成本估算（服务器实测速率换算）

任务 C 实测：PRICE 149,504 条 / 2,078 s = **13.9 ms/条**（CPU 单线程，batch 128）。

| 项 | 单进程 | 12 进程并行 |
|---|---:|---:|
| PRICE K=0 × 130 万条 | ≈ 5.0 h | **≈ 25–40 min** |
| 我们的 K=0 导出（GPU） | — | 10–20 min（GPU 忙则 CPU 40–60 min） |

注意：**GPU 当前 8 张全被占**（每张 ~10.5 GB、util 60–99%），先 `nvidia-smi` 确认；
CPU 侧 load 约 11/96 核，并行空间充足。

### 5.5 报告形态（**全部 26 行，不挑子集**）

每行一个库，列为：`ours_k0` / `price_k0` / `pg` × {Q50, Q95, log_Q95} + 4+ 桶 Q95 + 表数分桶。
外加三个分布统计：

1. 我们优于 PG 的库数 / 总数；
2. 我们优于 PRICE 的库数 / 总数；
3. 比值（PRICE/我们、PG/我们）的分布（median / min / max）。

再按「优于 PG 且优于 PRICE / 优于 PG 但劣于 PRICE / 劣于 PG」三档分类计数。
**逐库分类必须全部列出**，不得只展示某一档。

---

## 6. 执行顺序与并行

### 6.1 顺序

```
第 1 步  任务一：IMDB K=0 population 对照（~15 min）
         └─ 产出：实验文档/results/p2-2-target-imdb-20261021/
第 2 步  任务二：26 源库 our K=0 导出（10–20 min GPU / 40–60 min CPU）
第 3 步  任务二：PRICE K=0，12 进程并行（25–40 min）
         └─ 含 §3.2 的 chembl 布局敏感性自检
第 4 步  汇总、文档、提交
```

第 1 步与第 2 步可并行（不同资源）。

### 6.2 分支

```bash
git checkout -b experiment/source-broad-imdb-20261021 experiment/p2-2-three-seed-20261020
```

### 6.3 **回归检查（改脚本前必做）**

`scripts/price_population_eval.py` 若被修改，必须验证**默认路径逐位不变**：

```bash
# 用任务 C 的原始命令重跑，比对 summary.json 是否逐字节相同
sha256sum 实验文档/results/p2-2-price-population-20261020/summary.json
```

不一致就说明改动污染了既有行为，必须改成纯增量（新参数默认不生效）。

---

## 7. 产物

```
实验文档/results/p2-2-target-imdb-20261021/{summary.json, protocol.json}
实验文档/results/p2-2-price-population-broad-20261021/{summary.json, protocol.json}
scripts/target_population_eval.py                # 新增
scripts/price_population_eval.py                 # 追加可选参数（--maxima / --jobs）
tests/test_target_population_eval.py             # 新增
tests/test_price_population_eval.py              # 追加新参数的单测
实验文档/102-IMDB与源库广度对照结果.md          # 新增
```

`summary.json` 必须含 `target_databases_read`（任务一为 `["imdb"]`，任务二为 `[]`）、
`maxima`、`n`（每库参与的唯一查询数）、以及 §5.5 的分布统计。

---

## 8. 验收清单

- [ ] IMDB 521 条唯一查询**全部**参与，无裁剪；`n` 与 `target_databases_read: ["imdb"]` 都写进 protocol
- [ ] 任务一的 maxima 与 14 号目标表一致（复算并记录），不因加入 imdb 改动其他三行
- [ ] 任务二 26 行**全部**出现在 summary.json 里，带分类计数
- [ ] 两份 protocol.json 都记录 maxima 及其来源、各输入的 sha256 / content_hash
- [ ] §3.2 的 chembl 布局敏感性自检已做并写进报告
- [ ] 任务二的 `ours` 侧是**单 seed（42）**，标 `needs_evidence`
- [ ] 报告明确写出：源库行是 **in-sample 拟合诊断**，不是漂移证据（§1 的曝光矩阵）
- [ ] IMDB 行明确写出：只有 K=0（无 finetune split），521 条的 Q95 噪声大于现有三库
- [ ] 未读任何目标库参与训练/选择；`/gly/weilin/PRICE-master` 只读
- [ ] `PYTHONDONTWRITEBYTECODE=1 python3 -B -m unittest discover -s tests` 通过
- [ ] `ruff check scripts tests spatial_meta_ce` 无问题
- [ ] §6.3 回归检查通过（若改了 `price_population_eval.py`）
- [ ] 新分支已 push Gitee；`实验文档/99-进度与下一步.md` 已更新

---

## 9. 已知坑

1. **`maxima` 会改变结果**（§3）。任何两次 PRICE 数字并列前先确认 maxima 相同。
2. **`price_population_eval.py` 的默认行为不能变**（任务 C 的产物已提交，是既有证据）。
3. **`sap` 只有 29,714 条唯一查询**，26 行表里它是小样本，Q95 噪声大，报告要标 `n`。
4. **IMDB 只有 K=0**（无 finetune split），不能与 14 号目标表的 K=0/5/10/20/50/100 混列成一张表；
   它是目标表的一个 **K=0 补充行**，不是第四个完整目标库验收。
5. **GPU 可能被占满**（本轮检查时 8 张全忙），导出要准备 CPU 兜底。
6. **全量测试有既有偶发失败**：`tests/test_meta65_joint_supervisor.py` 的进程监管时序项
   （父提交上 6 跑 3 败、单跑 40 项全过），与本任务无关。
7. 26 源库并行会同时开 12 个 CPU 进程；机器是共享的，**先看 load**，别把别人的作业挤掉
   （本轮开始时 load 曾升到 101）。

---

## 10. 提前写下的预期（避免"没找到我们赢的库"变成意外）

诚实地说，这两个任务**大概率不会**产生「我们在很多库上赢 PRICE」的结论：

- **任务一（IMDB）**：现有 3 个目标库上我们 Q95 全输 PRICE、stats 还输 PG；且源库上我们的
  **中位数在三个库都不如 PG**。IMDB 大概率延续这个格局。**但它是第一个真正对等的漂移比较点**，
  无论正负都值得写。
- **任务二（26 源库）**：PRICE 见过全部 26 库，我们只见过 23；源库的 PG 估计很差（chembl PG Q95 146），
  所以我们相对 PG 很可能占优，但相对 PRICE 很可能仍落后，且差距应比目标库小。
- **真正已有的是 legalacts**（源验证库，K=0 population：我们 9.71 < PRICE 11.82 < PG 26.00，
  4+ 桶同样），它是唯一一个我们同时赢 PRICE 和 PG 的库；而 98 号显示 legalacts 恰好也是
  LODO 唯一超过 0.5、方向信号库内 AUC 最高的库，PG 方向与其他两库反号。
  这三条互相印证，指向的是**「方法在该库有效，是因为它的方向偏置与源池一致」**——
  比多测几个库去凑更有分量，值得作为 case study 写进论文。

**因此本计划的成功标准不是「找到我们赢的库」，而是「把漂移问题的边界测清楚」**：
目标库侧给对等的四行（含 IMDB），源库侧给完整的 26 行分布与分类计数。

---

## 11. 未决

1. **论文定位**（100 号 §9 第 1 问，尚未回答）：任务一/二的结果出来后，§4.7 与 §7 的措辞要相应调整。
2. **是否把这些新证据并入论文正文**（100 号）：本计划产出后，100 号的步骤 3/4 需扩到新证据。
3. **97 号任务 B（23 库跨库迁移 LODO）** 不在本计划内，仍待决定；
   注意它与任务二的区别：任务二比的是**精度指标**（拟合/对等漂移），
   任务 B 比的是**方向信号能否跨库迁移**（LODO AUC）。
