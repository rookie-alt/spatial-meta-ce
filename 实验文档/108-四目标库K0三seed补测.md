# 108-四目标库 K=0 三 seed 补测

日期：2026-10-23
分支：`experiment/target-3seed-20261023-local`
性质：**只读评估**。不训练、不适应、不选 checkpoint、不做新的模型选择。
前置：[102号](102-IMDB与源库广度对照结果.md)（四目标库与 26 源库对照）、[103号](103-误差结构诊断.md)（30 库误差分解）、[94号](94-P1精度杠杆实现与源侧筛查.md)（seed 噪声量级）、[101号](101-IMDB与源库广度对照执行计划.md)（评估协议）

---

## 0. 一句话结论

**102/103 号的"四目标库全输 PRICE"此前建立在 n=1 上；补成三 seed 后结论不变且更强。**

12 个（seed × 库）组合**全部**输 PRICE 的 Q95（0/12 我们赢），9/12 赢 PG。
PRICE/我们 的中位比值从单 seed 的 0.561 变为 **0.570**，方向与量级都不变。

**但 genome 的 seed 离散度大到必须逐 seed 报告**：Q95 三 seed 为
170.83 / 323.93 / 81.28（均值 192.02，样本 SD 122.70）——SD 是均值的 64%。
这条不能只写均值。

---

## 1. 为什么要补

104 号（导师汇报）与 102/103 号（实验结论）都引用"四目标库 Q95 全输 PRICE"。
但那些数字来自**单一 checkpoint**（seed42 ERM-sum baseline），而项目自己已实测出：

- **seed 间标准差 6.28pp**（94 号 §6.1b，源侧 4+ Q95 macro），远大于任何已筛杠杆的效应量；
- 方向信号的库内强度与跨库迁移都随 seed 变（98 号三 seed 复核）。

也就是说：**单 seed 的差距方向可信，但数值不可引用**。审稿人只要问一句"为什么只有一颗种子"，
整个目标侧结论就站不住。本轮补测就是为了消除这一条。

---

## 2. 设置

### 2.1 三个 checkpoint

| seed | 路径 | 训练配置 |
|---:|---|---|
| 42 | `p1-screen-20261020/baseline/last.pt` | ERM / sum / 4000 步 |
| 43 | `p1-screen-20261020/seed43/baseline/last.pt` | 同上 |
| 44 | `p1-screen-20261020/seed44/baseline/last.pt` | 同上 |

**三 seed 的训练配置除 `seed` 字段外逐字相同**，已在服务器上逐项核对三份
`config.json`：method=erm、steps=4000、meta_batch=4、query_size=64、outer_lr=3e-4、
inner_steps=0、adapt_scope=all、validate_every=100、validation_runs=5、validation_seed=1729、
patience=41、min_delta=1e-4、loss=smooth_l1、model{width:64, layers:2, heads:4, arch:sum, use_pg:true}、
data_config 同路径、23 训练库 / 3 源验证库。

> seed43/44 的 baseline 是 94 号为**源侧筛选**跑的三 seed 复核的一部分（同一协议、同一预算），
> 本轮只把它们当作**已存在的冻结 checkpoint**用于只读评估，不重训、不重选。

### 2.2 评估协议

复用 102 号 §3 已验证的只读脚本 `scripts/target_population_eval.py`：

| 项 | 值 |
|---|---|
| split | `test` |
| 目标库 | stats / genome / ergastf1 / imdb（**全部四个**，非只有 imdb） |
| K | 0（零样本，单次全量，无 support、无适应） |
| 口径 | 标准 unique-query Q95：`exp(\|max(log_pred,0) − log(max(true,1))\|)` 的 95 分位 |
| PRICE maxima | `12,14,8,10`（与 102 号 §2.2 同布局，未变） |
| device | CPU，单线程 |
| 曝光守卫 | 脚本强制：checkpoint 的 train/validation 曝光不得与四个目标库相交；否则拒绝运行 |

**未新增任何评估代码路径**：`target_population_eval.py` 与 `price_population_eval.py`
在本地与服务器的 SHA256 逐字节一致（`593da187…` / `00d684aa…`），即 102 号 IMDB 那次用的同一份代码。

### 2.3 服务器运行

代码目录 `/gly/weilin/spatial-meta-ce-p2-2-source-broad-20261021-code`（复用，未新建），
运行目录 `runs/p2-2-target-3seed-20261023/`，三次独立运行（seed42/43/44 各一次），
每次约 5 分钟，全部 `CUDA_VISIBLE_DEVICES` 置空、CPU 单线程。

---

## 3. 主结果：整库 Q95（三 seed）

| 库 | n | 我们 Q95（均值 ± SD） | 我们逐 seed | PRICE Q95 | PG Q95 | 我们赢 PRICE |
|---|---:|---:|---|---:|---:|---:|
| stats | 2,603 | **81.065 ± 2.444** | 83.723 / 80.559 / 78.914 | 39.521 | 75.650 | **0/3** |
| genome | 1,699 | **192.015 ± 122.702** | 170.833 / **323.929** / 81.283 | 31.336 | 1907.321 | **0/3** |
| ergastf1 | 1,258 | **23.444 ± 1.540** | 22.281 / 22.860 / 25.190 | 15.293 | 27.609 | **0/3** |
| imdb | 521 | **33.130 ± 1.792** | 32.367 / 35.177 / 31.845 | 21.025 | 87.110 | **0/3** |

**12 个（seed × 库）组合全部输 PRICE。** 9/12 赢 PG（输 PG 的是 stats 的三个 seed）。

### 3.1 与单 seed 数字的对照

| 库 | 102 号单 seed（我们） | 本轮三 seed 均值 | 本轮三 seed 中位 |
|---|---:|---:|---:|
| stats | 83.72 | 81.07 | 80.56 |
| genome | 170.82 | 192.01 | 170.83 |
| ergastf1 | 22.28 | 23.44 | 22.86 |
| imdb | 32.367 | 33.13 | 32.37 |

单 seed 值都落在三 seed 的取值范围内，**方向未变、量级未变**。
PRICE/我们 的中位比值：单 seed 0.561 → 三 seed **0.570**（区间 0.183 ~ 0.669）。
PG/我们 的中位比值：单 seed 1.965 → 三 seed **1.950**（区间 0.939 ~ 11.165）。

**即：102/103 号的结论在扩到三 seed 后完全成立，唯一变化是数字更可信。**

---

## 4. Q50（中位数）

| 库 | 我们（均值 ± SD） | PRICE | PG | 判读 |
|---|---:|---:|---:|---|
| stats | 1.9357 ± 0.0213 | 1.9538 | 1.9839 | **我们优于两者** |
| genome | 1.2830 ± 0.1400 | 1.8938 | 1.1944 | 输 PRICE，赢 PG |
| ergastf1 | 1.6051 ± 0.0473 | 1.4658 | 1.5399 | 输两者 |
| imdb | 2.0125 ± 0.0294 | 1.8626 | 2.2778 | 输 PRICE，赢 PG |

与 102 号 §3.5 一致：IMDB 的 Q50 首次优于 PG **在三 seed 下依然成立**
（三个 seed 分别为 2.037 / 2.021 / 1.980，全部优于 PG 的 2.278），但仍劣于 PRICE 的 1.863。
stats 上三 seed 的中位数（1.9211 / 1.9259 / 1.9601）与 PRICE 的 1.9538 基本持平，
3 个 seed 中 2 个优于 PRICE、1 个略差，与 PG 的 1.9839 相比则是 3/3 全部优于。

**这条只能写成「stats 上中位数与 PG 相当或略优、与 PRICE 接近」，不能写成显著优势**——
差距 1~2%，而 seed SD 0.021。

---

## 5. 4+ 表桶 Q95（尾部所在）

| 库 | n(4+) | 我们（均值 ± SD） | PRICE | PG |
|---|---:|---:|---:|---:|
| stats | 906 | 271.262 ± 35.830 | 141.756 | 230.057 |
| genome | 544 | 1278.206 ± 1189.342 | 64.031 | 9986.278 |
| ergastf1 | 258 | 93.789 ± 14.913 | 58.224 | 107.297 |
| imdb | 118 | 440.493 ± 82.682 | 88.910 | 1174.911 |

4+ 桶上我们 4/4 输 PRICE，且**三个 seed 全部输**（PRICE 在 4+ 桶上的优势比整库更大，
因为我们的尾部正是集中在多表查询）。IMDB 的 4+ 桶只有 118 条查询、Q95 由最差约 6 条决定，
引用具体倍数时须带 n。

---

## 6. 这次补测暴露的两件新事

### 6.1 genome 的 seed 离散度大到不能只写均值

Q95 三个 seed：170.83 / 323.93 / 81.28，样本 SD 122.70，是均值的 64%。
4+ 桶更极端：956.57 / 2595.29 / 282.76，SD 1189.34。

**含义**：genome 上「我们输 PRICE 多少倍」不是一个稳定量。用单 seed 的 170.82 说"差 5.4 倍"
或用均值说"差 6.1 倍"，都不如说「三个 seed 全部输，倍数在 2.6× ~ 10.3× 之间」诚实。
项目此前在 05 号已记录 genome K=0 的 seed 方差（248.69 ± 67.29，三 seed），
本轮是同一现象在 **K=0 population 口径**下的独立再现。

### 6.2 stats 上 Q50 我们与 PRICE 持平、与 PG 略优

单 seed 时 1.921 vs 1.954 只差 1.7%，单看会被当噪声；三个 seed 分别
1.9211 / 1.9259 / 1.9601，全部低于或接近 PRICE 的 1.9538（仅 seed44 略高），
与 PG 的 1.9839 相比则是 3/3 全部优于。

**但差距只有 1~2%，小于 seed SD 0.021**，所以只能写"持平/略优"，不能写优势。

---

## 7. 口径澄清：与 103 号那 1e-5 级差异的来源

102 号 §3 的 IMDB 行（32.367）与 103 号的三库行（83.72 / 170.82 / 22.28）来自**两个不同脚本**，
本轮复算时发现两者在 stats/genome/ergastf1 上有 1e-5 ~ 1e-4 的相对差（IMDB 逐位相同）。已定位成因：

| 脚本 | 分位数空间 |
|---|---|
| `price_population_eval.py` / `target_population_eval.py`（`spatial_meta_ce/metrics.py: summarize`） | **倍数空间**取分位：`percentile(exp(\|Δlog\|), 95)` |
| `error_structure_diagnosis.py`（`summarize`） | **log 空间**取分位再 exp：`exp(percentile(\|Δlog\|, 95))` |

`percentile(exp(x), p) ≠ exp(percentile(x, p))`，两者只在同一插值规则下近似相等。
本轮验证了差异的分布规律：IMDB 的 n=521 使 `0.95×(n−1)=494` 恰为整数（无插值），
故两种口径**逐位相同**；其余三库索引为 2471.9 / 1613.1 / 1194.15（需插值），
故有 1e-5 ~ 1e-4 的相对差。

**这不是 bug，是两个口径的差别**。两套数字各自成立、不可互换。
本轮（108 号）全部使用 `target_population_eval.py` 的**倍数空间**口径，
因此 108 号的 seed42 行与 103 号的 seed42 行在 stats/genome/ergastf1 上有上述微小差异，
在 IMDB 上逐位相同。引用时须写明用的是哪个口径。

---

## 8. 产物与验证

| 类型 | 路径 |
|---|---|
| 三 seed 原始证据 | `实验文档/results/p2-2-target-3seed-20261023/seed{42,43,44}-{summary,protocol}.json` |
| 聚合结果 | `实验文档/results/p2-2-target-3seed-20261023/{summary,protocol}.json` |
| 聚合脚本 | `scripts/target_three_seed_summary.py`（新增，只读） |
| 单测 | `tests/test_target_three_seed_summary.py`（11 项） |
| 服务器运行目录 | `/gly/weilin/spatial-meta-ce-p2-2-source-broad-20261021-code/runs/p2-2-target-3seed-20261023/` |

### 8.1 验证

- 本次新增测试 **11 项全过**（含三个守卫：PRICE 跨 seed 必须逐位相同、PG 跨 seed 必须逐位相同、
  checkpoint seed 必须与文件名匹配，否则拒绝聚合）。
- 全量 `python3 -B -m unittest discover -s tests`：**1488 项，1 error（16 skip）**。
  该 error 是 `test_meta65_joint_supervisor.py::test_each_external_phase_timeout_without_worker_deadline`，
  属 97 号 §9.7 已记录的**既有偶发**（进程监管时序项，父提交上 6 跑 3 败、单跑 40 项全过）；
  本轮单独复跑该文件 **40 项全过**。本次改动只新增一个脚本、一个测试、一份证据目录，
  与该测试无代码交集。
- `ruff format` + `ruff check`：改动文件无问题。
- 格式化后重跑聚合，输出与格式化前**逐字节相同**。

### 8.2 聚合脚本的两条守卫

`scripts/target_three_seed_summary.py` 在发布前强制：

1. **PRICE 与 PG 跨 seed 必须逐位相同**（PRICE 只有一个 checkpoint，PG 是固定估计）。
   若三次运行给出不同的 PRICE/PG Q95，说明三次运行不可比，脚本**拒绝输出**。
2. **只报均值 ± 样本 SD（ddof=1），不报 best-of-seeds**，并逐 seed 记录数值与
   「赢 PRICE / 赢 PG 的 seed 数」，让读者看到方向是否每个 seed 都成立。

---

## 9. 边界

1. **三 seed 是项目现有覆盖度**。它界定了 seed 离散度，不使任何单库 Q95 成为稳定点估计。
2. **genome 的离散度本身就大**（SD 122.70 / 均值 192.02），任何引用 genome 倍数的句子
   必须带上三 seed 区间或 SD。
3. **PRICE 只有一颗 checkpoint**，其列没有 seed 离散度，不是训练条件匹配的对照。
4. **IMDB 仍是 K=0 补充行**，不是第四个完整 few-shot 目标库验收（服务器无 `finetune/imdb`）。
5. **Q50 的 stats 差距只有 1~2%**，小于 seed SD（0.021），不能写成显著优势。
6. 本轮是**只读评估**：没有训练、没有适应、没有选 checkpoint、没有读目标库参与任何选择
   （目标库的读取仅限最终评估，且脚本守卫强制 checkpoint 曝光与四目标库不相交）。
7. `/gly/weilin/PRICE-master` 只读，未修改。

---

## 10. 对论文表述的影响

**结论不变，只是从 n=1 变成 n=3：**

| 可以写 | 不能写 |
|---|---|
| 四个目标库、三个 seed、12 个组合**全部**输 PRICE 的 Q95 | 「我们在任何配置下都不如 PRICE」（PG 上我们 9/12 赢） |
| 中位数上 stats 与 imdb 优于 PG，stats 与 PRICE 持平 | 「中位数已全面超越 PRICE」 |
| 我们的相对优势集中在尾部修复（PG/我们 中位 1.95） | 「我们整体优于传统估计器」 |
| genome 的倍数在 2.6× ~ 10.3× 之间，方向 3/3 一致 | 用单 seed 说「差 5.4 倍」 |
| 轻量模型参数为 PRICE 的 1/80.80 | 把参数量比直接当端到端加速 |

**与 102/103 号的关系**：102 号的四目标库表、103 号的 dCorr 与 Q95 翻转结论，
在把 seed42 换成三 seed 之后**结论方向不变、量级不变**。
103 号「相关性优势精确跟随训练曝光」的机制解释不受影响——它基于 30 库的分组对比，
而本轮补的是四个目标库内部的 seed 离散度。
