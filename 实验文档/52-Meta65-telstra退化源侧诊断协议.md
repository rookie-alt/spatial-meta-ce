# Meta65 telstra 局部退化的源侧描述性诊断协议

日期：2026-10-10（Asia/Shanghai）。用户要求围绕“收益集中在部分源库、telstra 却退化”开展下一步受控操作。本协议只做**既有预测的描述性诊断**，不训练、不扩 seed、不搜超参、不做目标评估、不改模型与适应机制。

## 1. 已冻结的事实

上一轮补齐阶段（提交 f08bd49）在源验证库上得到 4+表 unique-query Q95 等权算术 macro：

| 臂 | 元训练内循环 → 部署适应 | 主 4+ Q95 | 相对 A |
|---|---|---:|---:|
| A_all5_head3 | all/5 → head/3 | 44.497979 | 参照 |
| B_head3_head3 | head/3 → head/3 | 39.000403 | −12.3547% |
| C_last3_last3 | last/3 → last/3 | 39.010383 | −12.3322% |

跨库（4+ Q95，跨 K 与 run 均值）：

| 库 | A | B | C | C 相对 A |
|---|---:|---:|---:|---:|
| chembl | 109.440223 | 92.070262 | 93.745172 | −14.34% |
| legalacts | 17.363760 | 17.489044 | 14.658522 | −15.58% |
| telstra | 6.689954 | 7.441904 | 8.627455 | **+28.96%** |

C 在 telstra 的 K=5/10/20/50/100 上，4+ Q95 相对 A 分别退化 25.2980% / 22.6481% / 28.7385% / 33.3468% / 35.7019%。B 在 telstra 也有 4 项超过 10% 退化上限。

## 2. 已知混淆（不得忽略）

- A→B 同时改变训练 scope（all→head）与 inner steps（5→3）。
- B→C 只改变 scope（head→last），inner steps 均为 3；但 B 与 C 是**两次独立元训练**，共享同一 seed42 初始化，外循环轨迹不同，末态参数不同。
- 因此任何臂间差异都是“训练后的元初始化 + 适应算子”的联合结果。本协议**不产出**“65 参数容量不足”或“元学习无效”这类单因素因果结论。
- 三个源验证库（chembl/legalacts/telstra）在历史上被反复用于开发与选择，不是盲测集；单 seed 不提供显著性。

## 3. 只读输入

- A/B 预测：`/gly/weilin/spatial-meta-ce-meta65-20261009-frozen/runs/meta65-source-screen-20261009/{A_all5_head3,B_head3_head3}/predictions/*.npz`（各 75 份）。
- C 预测：`/gly/weilin/spatial-meta-ce-meta65-completion-20261009-code/runs/meta65-endpoint-completion-20261009/C_last3_last3/predictions/*.npz`（75 份）。
- 支持/query 计划：新补齐目录 `endpoint_plan.json`（与旧运行逐字节同源，SHA `ebeb178d…`）。
- 每份 NPZ 字段：`log_pred`、`true`、`n_tables`、`query_hash`、`query_id`，全部 float64/int64，未被裁剪。
- PG 锚：**本轮 225 份 NPZ 不含 pg 字段**（验证器把字段冻结为五个：log_pred/true/n_tables/query_hash/query_id）。PG 只能从只读源缓存 `cards.npy`（列 0=true，列 1=PG）取得。
- 5.0 更正：同一 (db,k,run) 内三臂 `support_digest` **相同**（三臂共享同一支持集），不同 run 的 support 不同；同一 (db,run) 内 query 集合**跨 K 完全相同**，K 只改变 support 大小。

PG 锚执行前置（必须逐项通过，否则 D3 的 PG 部分标为 `pg_anchor_status=failed/not_attempted`，不得静默跳过）：

1. 绑定 `cards.npy`、`manifest.json`、`offsets.npy` 的 SHA256 到本轮输入；
2. `query_id` 为 `db/pretrain/<row>` 字符串，需解析为 manifest 行号，再按 `query_hash` 第一次出现去重后与 NPZ 的 `query_hash` 顺序对齐；不得把 `query_id` 当数组索引；
3. 逐项校验 `cards[rows,0] == npz["true"]` 与 `offsets 第 2 列差 == npz["n_tables"]`；
4. 输出 `pg_anchor_status ∈ {verified, failed, not_attempted}`。

全部输入在诊断前后复核 SHA；诊断不写回上述任何路径，输出到新目录。

## 4. 诊断问题与分层

对每个 (db, k, run) 单元，用该单元格的查询集合与该库可见的 NPZ 逐查询计算 standard Q-error。telstra 为主分析对象，chembl/legalacts 作对照。

- **D1 表数分层**：`n_tables ∈ {2,3,4,5+}`。报告各桶的查询占比、Q50/Q95，以及 A/B/C 在同一桶内的变化。
- **D2 真实基数分层**：`log10(true)` 分桶。报告同结构。
- **D3 修正方向分层**：以 A 的 `log_pred - log(max(true,1))` 判定高估/低估；若提供 PG，则同时报告 PG 高估/低估方向。方向分层只用于描述误差结构，不作为因果解释。
- **D4 配对查询级差异**：固定 K，对 5 个 run 的 query_hash 取交集，逐 query 求各臂跨 run 平均 `log_q`；报告 `Δ(C−A)`、`Δ(B−A)` 的分位数，以及差异最大的若干 query（只报 hash/id，不报 SQL 文本）。
- **D5 支持抽样稳定性**：逐 run 计算 A/B/C 的 4+ Q95 与 `C/A`、`B/A` 比值，检查退化是否由个别 run 驱动。
- **D6 尾部集中度**：每臂 Q95 由前若干百分位查询贡献的比例；比较 A/B/C 的尾部是否更厚。

## 5. 预登记的判读规则（只用于决定下一步方向，不用于选模型）

### 5.0 主代理预检已观察到的关键结构（待 D1-D6 复核，不是结论）

以下都是只读统计，尚未写成结论；D1–D6 必须独立复现或推翻它们：

- 75 个 cell 上，同一 (db,k) 的 `query_digest` 全部相同、`support_digest` 全部不同、`query_count/four_plus_count` 三臂一致 → 查询集在臂间共享，**支持集在 run 间不同**，D4/D5 有效。
- 更正与细分：同一 (db,k,run) 内三臂 support 相同、query 相同；**同一 (db,run) 内 query 集合跨 K 完全相同**，K 只改变 support 大小；不同 run 的 support 不同（`four_plus_count` 在 run 间不同）。
- C 的 `log_pred` 标准差在三个库都约 2.7–3.2，B 和 A 也在同一量级 → **C 没有塌缩成常数预测**。
- telstra 4+ 表（K=20）：A 平均偏置 −0.099（59.5% 低估），B +0.123，C +0.419（67.1% 高估）；5 个 run 的 `mean(C−A)` 均为正（+0.39~+0.64）→ 指向**系统性高估**，而非随机尾部。
- 但 chembl 的 `mean(C−A)` 更大（+0.65~+1.03）→ 高估漂移**不是 telstra 独有**。telstra 只是绝对 Q95 小（6–9），因此同样的绝对偏移在百分比上被放大。
- telstra 的 C/A 逐 run 比值在 K≥10 稳定在 1.25–1.41，**不是个别 run 主导**。
- 按表数分层（K=20、5 run 平均）出现**符号翻转**，比“整体高估”更具体：同一“4+ 表”类查询，C/A 在 chembl 为 **0.832（改善）**、在 telstra 为 **1.287（恶化）**；2/3 表上则相反（telstra 2 表 C/A=0.741、chembl 2 表 C/A=1.087）。因此“C 整体高估”不成立，退化与**库×表数交互**相关。D1 必须按 `(库, 表数桶)` 分别报告，不能只给跨库汇总。

据此，判读规则补充三条强制要求：

1. 任何“telstra 退化”陈述必须同时给出**绝对差**（Q95 差、log 尺度差）与相对百分比；不得只报百分比。
2. 必须报告 A/B/C 在**三个库**上的同一偏差指标；若 chembl/legalacts 也有同向漂移，则不得把结论写成 telstra 特有问题。
3. 不得由这些统计推断“65 参数容量不足”或“元学习无效”；只允许表述为“该元初始化下的系统性高估 + 小绝对尺度放大”。
4. 跨库比较**禁止**用原始百分比排序严重性；必须给出四元组（A 绝对值、C−A 绝对差、C/A 比值、log10(C/A)），跨库排序用绝对差或 log 比值。
5. D5 必须同时报三类量：主口径 5-run 均值之比、逐 run 比值、C−A 绝对差与“去掉单个 run 后符号/K 排序是否翻转”；逐 run 比值只作稳定性描述，主口径仍以均值之比为准。
6. D4 的“取交集”在当前计划下是恒等操作：同一 (db,run) 各 K 的 query 集合相同。因此 D4 应表述为“验证 5 run 的 query 集合，然后对每个固定 query 跨 run 平均 log_q”；K 间 Q95 差异来自 support 规模，不来自 query 组成变化。
7. D1–D6 只描述误差结构，**不能**直接推出“表征/特征不足”或“适应机制缺陷”。至少区分三类混淆：训练 scope（all/head/last）、inner steps（5/3）、以及 B/C 的独立元训练末态。若要给下一步方向，必须明说这只是候选方向。
8. 桶内样本过少的处理：D2 的 log10 分桶对样本数低于最小阈值的桶只报 `n` 与 `status=below_min_n`，不报 Q50/Q95；D6 的尾部集中度固定为“Top 5% q-error 的 sum(log q) 占全部 sum(log q) 的比例”，这是 log 尺度、可加总的定义，不得与“超过 Q95 的查询数占比”混用。

- 若退化**稳定**（D5 无单 run 主导）且**集中在特定桶**（D1/D2/D3）→ 下一步优先研究表征/特征信息。
- 若退化**稳定**且**弥散**（各桶普遍变差，D4 差异分布近似整体平移）→ 下一步优先研究适应机制（scope、步数、正则）。
- 若退化由**个别 support run** 主导（D5）→ 下一步优先研究支持抽样方差与支持集构造。
- 上述规则是启发式方向选择，**不构成任何候选或门槛**。不得据诊断结果回选 B/C，也不得据此调整既有冻结门槛。

## 6. 交付与验证

输出精简 `telstra_diagnosis.json` 与分层 CSV，写 `reports/meta65-telstra-diagnosis-20261010/` 与 `实验文档/results/meta65-telstra-diagnosis-20261010/`。大体积产物不新增，原始 NPZ 不入 Git。

先 TDD：分层/配对/分位数逻辑要有可失败的单元测试；实现后跑完整 CPU unittest、改动文件 Ruff、密钥扫描与 diff 审查（`/ecc:verification-loop` 六阶段）。诊断执行只用 CPU，不占 GPU，不连接数据库。

结论必须是描述性的，明确写清混淆与单 seed 限制。协议与实现的 SHA 在运行前后复核；输入不变才收尾。诊断本身不授权任何训练、搜索或目标评估。
