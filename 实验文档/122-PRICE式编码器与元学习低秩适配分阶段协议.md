# 122 — PRICE 式编码器与元学习低秩适配分阶段协议

> 状态：`protocol_reviewed / implementation_not_started / training_not_started / target_read_not_authorized`
>
> 日期：2026-11-11。本文只冻结研究问题、最小实现边界与停止条件，不授权训练、目标评估或修改 PRICE。
> 本方案依据导师提出的“原模型可能过于简单、可参考 PRICE、使用 PG、采用 LoRA 微调”意见，并应用户要求保留元学习。它是待证伪候选，不是已证明的改进路线。

## 0. 审查结论与本轮范围

**允许进入实现前的协议审查，不允许直接跑四组件全组合。** 原“PRICE + PG + FOMAML + LoRA”在数学上可组合，但存在三个混杂：容量、训练目标与适配算子同时改变；PRICE 原输出层重复使用 PG；标签空间不一致。本文将其拆为架构、适配、元学习三道门。

- 保留元学习是研究目标，不是“FOMAML 已在各架构有效”的事实。[09 号](09-ERM-sum对照.md) §2 记录了随架构/K 翻转的结果，[104 号 §7.13](104-导师汇报报告与文献对标.md) 记录了 FOMAML-sum 在三个已评估目标库的 K=0 population 不优于 ERM-sum，且 imdb 反向。
- [107 号](107-架构对照重做协议.md) 与 `reports/arch-compare-20261024/summary.json` 的历史六 seed 架构比较均为 `inconclusive`；这不证明现有结构足够，也不证明加深有效。
- 历史“PG+残差不可用”针对当时输出路线；本协议依据当前研究意见重新启用 **PG 残差锚**，不声称延续 PG-free 或结构锚点主线，也不撤销 [120 号](120-结构锚点ERM训练与目标库评估结果.md) 的负结果。
- 本候选不增加原始统计信息。PRICE 式编码器只改变相同四组 PRICE 特征的交互方式，不能宣称解决连接配对、条件相关性或真实选择率缺口。
- 本轮只允许写协议与只读审查。实现、服务器部署、GPU/长 CPU 任务、正式训练、目标库评估须分别授权。

## 1. 研究问题与可证伪假设

**总问题**：在相同 PRICE 四组特征和严格跨库划分下，两阶段编码器能否改善源验证库泛化；低秩内循环能否减少少样本负迁移；FOMAML 能否在此基础上提供额外适配收益？

| 编号 | 假设 | 否证情形 |
|---|---|---|
| H-A | 相同 PG-anchor-only 接口下，PRICE 式 scaling→filtering 编码优于 `sum_anchor_only`；容量影响另行估计 | 主配对差异不可判定，或小容量方向对照不能复现方向 |
| H-B | 同一带 LoRA 的 checkpoint 上，LoRA-only 评估适配比 full/head-only 更稳定地利用 support | support 后相对自身 K=0 仍恶化，或不优于 head-only/full |
| H-C | 对同一 backbone、同一内循环算子，FOMAML 优于 ERM | K=0 或适配后的配对差异不可判定/反向 |

H-A、H-B、H-C 必须分别判定。任何一臂胜出都不能自动证明另外两个假设。LoRA 是**待检验的优化约束**，不是论文创新点，也不保证 zero-shot 改善；K=0 时没有内循环更新。

## 2. 固定数据边界

1. 训练库为 `configs/data.json` 的 23 个 train；源验证库仅 `chembl/legalacts/telstra`；目标库 `imdb/stats/genome/ergastf1` 不进训练、超参、rank、内循环步数/LR、容量、checkpoint 或进入下一阶段的决策。
2. A/B/C 阶段只读 23 个训练库及三个源验证库的 `pretrain` 缓存。官方 PRICE 权重使用全部 26 个源库，含三个源验证库，只能作为补充参考，不能初始化本候选或作为严格 23/3 主对照。
3. 源验证库已被多轮开发使用，结果是开发集筛查而非新盲测。若后续连续多轮候选均使用这三库，应报告选择偏差，不能将配对规则写成独立显著性证明。
4. 训练 episode 按 seed 配对，使用 hash 绑定的训练计划：4000 updates、meta_batch=4、query_size=64、K 从 `{5,10,20,50,100}` 均匀抽样，support/query 按规范化 query hash 不重复。优先复用 [112 号](112-FOMAML适应前后联合目标源侧小实验方案.md) 的 sealed plan；复用前核对数据 hash 和完整 episode，不重新随机抽样。
5. 主分析只用独立 population plan：每库五组互斥 support block（每组 20 条，`support_seed=2026`），从全部 unique query 排除这 100 条后作为评估 population。block 内顺序由该 seed 固定；每个 block 是一次 support run，K=20 取全部 20 条，K=5 取该顺序前五条，不另设 `shot_seed`，完整 ID/hash 写入 manifest。历史 `plan_seed=2718` 的 75-episode 抽样只用于诊断，不与主 population 混合，也不用于本协议选型。两种计划不得混称或拼接；所有臂共享相同 ID/hash 与排除分母。
6. 支持查询不得与评估查询规范化 hash 重叠；相同 SQL 重复行不作为独立证据。输出每库 unique 分母、排除数、支持 ID/hash 和缓存 SHA256。
7. IMDB 没有完整 support 特征。即使获准进入 D，IMDB 仅可报告 K=0 补充行；K>0 主评估限 stats/genome/ergastf1。未经用户另行批准不得使用 `--allow-exclusions` 或伪造/补齐特征。

## 3. 模型契约

### 3.1 候选编码器 `price_lite`

复用 vendored PRICE 的 `ScaleEmbedding → scale Encoder → FilterEmbedding → filter Encoder → CLS/length 融合` 思路，但替换最终预测头：

```text
PRICE 四组特征
  → scaling tokens / scaling encoder
  → 加 table/filter tokens / filtering encoder
  → query representation + log1p(四类数量) + log Cartesian
  → 零初始化 residual head
ŷ = log(max(PG, 1)) + residual
```

首版固定 `d_model=64, layers=2 per stage, heads=4, dropout=0`，输出头与当前模型一致的 `128→64→1` 形式。它称为 **PRICE-style**，不是 PRICE 原版。

- PG **只**作为残差锚，不送进 PRICE trunk 或最终特征。删除候选中的 `final_linear1/final_linear2`，避免 PG 双重计入。
- 训练和评估标签统一为 `log(max(true_card,1))`，预测、support loss、query loss 和 Q-error 都在此空间。禁止对候选使用 `price_support` 或 `price_log_predictions`；这两个函数只保留给官方 PRICE 基线。
- 零初始化最后一层，使训练前、K=0/0-step 输出逐位等于 `pg_log`。不做输出 clamp 或以 exponent 溢出换指标。
- fanout 在缓存中为原值，经 `batch_samples` 后为 `log1p`；进入 PRICE embedding 前必须按原适配器对 batch 中的该组 `expm1` 还原，不得重复变换缓存。保持 PRICE padding `-1000`、table padding 首维为 0、mask 及 `log1p(counts)` 的约定。
- padding 容量只从 23 个训练库的全量缓存 `max_counts` 计算并随模型配置冻结；禁止从源验证或目标库求 maxima。超过容量的源验证查询属于该模型 `unsupported`，报告数据库、query hash、组名、实际值/容量；**不得截断、静默排除或更换容量**。若源验证存在 unsupported，H-A 的全 population 判定停止，仅可做明确标注的覆盖诊断，不能用于候选选型。

### 3.2 容量与资源边界

按当前构造器静态计算，`price_lite` 使用 `128→64→1` 残差头时为 **293,777** 参数（不含原 PRICE `final_linear1/2`）；当前 `sum` 为 **132,545**，`sum_anchor_only` 去掉 head 的 PG 标量输入后为 **132,417**，官方 PRICE 为 **10,709,330**。最终数值必须由实现后的实际模块重算，不写死为实验结果。`price_lite` 约为 anchor-only 基线的 2.22 倍参数，须报告实测 FLOPs/时延。

首版不跑官方 256 维/6 层全容量重训。A 阶段另含 `price_tiny`：`d_model=32, layers=2 per stage, heads=4`，其余不变，作为**预先冻结的小容量对照**（静态估计 86,641 参数），不是看到 lite 结果后才选择的配置，也不宣称与 sum 等参数。H-A 同时要求 lite/tiny 对 anchor-only 的方向一致；若只有 lite 改善，只能写“容量/结构未分离”，不得归因于两阶段结构。

### 3.3 LoRA 与元学习定义

- LoRA 仅注入两个 encoder 的 attention **Query/Value**，`r=4, alpha=4`，即 `ΔW=(alpha/r)BA`，B 零初始化、A 按冻结 seed 初始化。四组 embedding、output head 不叫 LoRA。
- `scope=lora`：support 内循环只更新注册的 A/B 参数；base 权重不变。`scope=head` 与 `scope=all` 保持现有语义，不用 `requires_grad=False` 冻结外循环基础参数。
- 计算图必须由 `functional_call` 替换参数，不允许闭包缓存 delta、`.data` 原地更新或跨 episode 保存 fast weights。
- FOMAML 内层保持现有 SGD 与 `create_graph=False`；query loss 对**所有基础参数、head、LoRA 初始值**反传，不是只训练 LoRA。首版不引入 Adam 内循环、二阶 MAML、rank 网格或 per-layer LR。
- vendor 每个 attention head 的 Q/V 为 `Linear(64→16)`；每个线性层的 A+B 为 `4×(64+16)=320`，两 encoder、各 2 block、每 block 4 heads × Q/V 共 **10,240** 个内循环参数。filtering-only 为 5,120，但首版不启用。若改为拼接投影，必须另列参数口径，不能与 per-head 混算。

## 4. 阶段与最小矩阵

### P0：实现与协议预检（无研究训练）

必须先完成 §7 全部测试、合成 episode 的 FOMAML 外循环反传、模型 checkpoint 往返、23 个训练库容量生成及三个源验证库 manifest 的 max_counts 扫描。若源验证出现 unsupported，A 在训练前即停止，不做全 population 选型。CPU/GPU smoke 须另行授权；单次最多 2 updates/5 分钟，CPU 单线程、RSS 4 GiB 或 GPU 4 GiB，且不占未确认空闲的卡。smoke 标 `smoke-not-research`，不用于 H-A/B/C 判定或目标调参。P0 不通过则停止。

### A：架构门（仅源验证）

| 臂 | backbone | PG 接口 | 训练 | 评估适配 |
|---|---|---|---|---|
| A0-ref | 当前 `sum` | head 输入 + anchor | ERM | full，LR=1e-3，5 步 |
| A0 | `sum_anchor_only` | 仅 anchor | ERM | full，LR=1e-3，5 步 |
| A1 | `price_lite` | 仅 anchor | ERM | full，LR=1e-3，5 步 |
| A2 | `price_tiny`（32 维容量臂） | 仅 anchor | ERM | full，LR=1e-3，5 步 |

- 42/43/44 三个训练 seed，4000 updates、`last.pt`、同一训练计划、同一 ERM 样本呈现和 Adam 外循环配置。验证不早停，不取 `best.pt`。A 最多 12 个训练 run；A0-ref 仅在完全匹配 sealed plan、selection=none、训练未读源验证库且仅使用 `last.pt` 时复用，否则重训。
- 主终点 K=0；K=5/20 仅作固定评估适配诊断，不据此选架构。ERM 训练无内循环；C 的 FOMAML 训练内循环与评估适配保持同一 scope/LR/步数。
- **H-A 主比较为 A1−A0**，A2−A0 是小容量方向诊断，A1−A0-ref 只描述相对历史基线的完整变化。A1/A2 均须满足 §5.2 的改善与逐库保护，才可进入 B；只赢 A0-ref 不算结构证据。A2 不是等参数对照，容量影响仍未单独识别。
- **B 固定 A1 `price_lite`**；A2 只作方向诊断，不进入 B/C，避免根据 A 结果事后选择 backbone。
- A 失败则停止 B/C/D，不改 rank、容量或训练量继续找好结果。结论限定为当前结构与预算下未检出收益，不推断 PRICE 或所有更大模型无效。

### B：适配门（仅当 A 通过）

固定 A 选出的 PRICE-style backbone，以 ERM 训练带注册 LoRA 参数的 checkpoint（3 seeds）；ERM 外循环学习全部 base/head/LoRA 参数。**同一 checkpoint 分别执行**：

| 臂 | support 更新参数 | LR |
|---|---|---|
| B-full | 全部参数 | 1e-3 |
| B-head | `head.*` | 1e-3 |
| B-lora | 两个 encoder 的 Q/V LoRA A/B | 1e-3 |

- 三种 scope 共享初始化、训练权重、K=0 预测和 support manifest；K=0 三者必须逐位一致。B 只增加 3 个 ERM 训练 run，不能为每个 scope 分别训练后再宣称纯适配算子对照。
- K=5/20，五次 support run，评估适配 5 步；每个 run 从同一 checkpoint 重新开始，detached fast weights 不共享。
- B 按 §5.2 的判定表选择 LoRA 或固定 full/head 对照；LoRA 未通过时该 LoRA 臂不进入 C，其他 scope 也无可接受负迁移时停止。该门只回答“同一初始化上 LoRA 是否有用”，不宣称普遍防止过拟合。

### C：元学习门（仅当 A、B 通过）

固定 B 选出的 backbone 和评估适配 scope/LR/5 步，只改变 ERM→FOMAML。训练仍使用相同 episode、seed 与 4000 外循环更新、`last.pt`。C 两臂始终保留 B checkpoint 的相同参数化（含 LoRA 注册）；ERM 对照使用 B 的同一个带 LoRA checkpoint，并重新执行 B 选定的评估 scope；不是另选 B-full/B-head/B-lora 的不同训练权重。

- 主指标为 K=5/20 的适配后 macro log-Q95；K=0 为必报共同初始化指标，若 macro K=0 的 FOMAML−ERM > 0.1 或触发同一逐库保护则 C 不通过，不能以 K>0 改善掩盖。
- 同时报告样本呈现、support 重复呈现、内/外 backward 次数、wall time、峰值内存。FOMAML 与 ERM 不宣称等计算成本，只保证同 episode 与更新预算。
- FOMAML 相对 ERM 的配对改善须超过配对 SD；否则不将“元学习增益”带入主结论。

### D：冻结后目标评估（不自动执行）

须用户另行授权，且 A/B/C 的配置、checkpoint、计划、代码 hash 和结果摘要全部冻结。只带通过门槛的候选及必要基线，禁止按目标结果回头改任何参数。

- K=0：四目标库全量 unique query；K=5/20：仅三个具有完整 support 的目标库，5 runs。
- 参照：PG、官方 PRICE zero-shot/既有合规 LR 的少样本、当前 sum-ERM、随机初始化+support。官方 PRICE 的 26 源库曝光单列，不能写成严格同训练分布。
- 若增加官方 PRICE+LoRA，只用于检验大模型少样本稳定性；它保留 PRICE 原标签/前向，使用独立源验证选择，不与本候选的 PG residual 规则混用，也不计入本方法贡献。

## 5. 指标、配对与预算

### 5.1 指标

1. **架构主指标**：每库 K=0、全部可用 unique query 的 `log_q95=percentile(abs(max(log_pred,0)-log(max(true,1))),95)`，再三库等权算术平均。预测 floor 完全沿用 `metrics.py`，不是新增输出裁剪；它是 log 空间 Q95，不等于倍数空间 `q95`。
2. **适配主指标**：相同 population 下每库/K/run 的 macro log-Q95；同一 seed/support 的 K>0 与 K=0 配对。逐库报 Q50/Q90/Q95/Q99、log-Q95、恶化比例及分母。
3. **次要诊断**：每库把全部 ≥4 表查询合并为一个 4+ 桶，在倍数空间计算 Q95，再对非空库等权；空桶记 null，不按查询数加权。不能把 episode 64-query 分位数与全 population 分位数混列。
4. Q-error 定义沿用 `metrics.py`：`max(max(pred,1)/max(true,1), max(true,1)/max(pred,1))`。`price_shifted_q*` 仅供 PRICE 对照，不与主指标混称。

### 5.2 判定

对 A/B/C 指定比较，先固定 seed；B/C 再在每个 seed 内对同一 support run 的 macro log-Q95 求差，并对五个 run 等权平均，最后对三个训练 seed 的差值做配对汇总：

- 负值为改善；报告 `mean(Δ)`、`sd(Δ,ddof=1)`、逐 seed 和逐库值。support run 不是训练 seed，不并入三 seed 的 SD。
- `abs(mean Δ) < sd Δ` → `inconclusive`，不得称“更优/等价”；A 中 `mean Δ ≥ 0` 为未通过，`mean Δ < 0` 且达到上述幅度才是正向。
- n=3 时该规则是预先冻结的保守筛查，不是显著性检验。若 `sd=0`，只在全部差值严格同号且非零时判方向；全为 0 为不可区分。
- 逐库保护：先对每个 seed 的五次 run 等权平均，再对三 seed 等权平均；若某库 `candidate_log_q95-reference_log_q95 > 0.1`，该阶段不通过。逐 seed/run 全量披露，但不以单 run 波动直接否决；A 的 K=0 无 support 平均。不能用 macro 改善掩盖。
- B/C 的负迁移定义为同臂/seed/run/K 的 `log_q95(K)-log_q95(0)>0`，每个评估 scope 保留 3 seeds×3 库×2 K×5 runs=90 个原始单元；六格指三库×两 K，每格先平均五 run 再三 seed。该格均值大于 0 为负迁移格。
- **B-lora 通过须同时满足**：对 full、head 两个对照，K=5 与 K=20 都满足上述配对改善规则；自身负迁移格数分别不高于 full/head；每库/K 的五 run `log_q95(K)-log_q95(0)` 配对差值 SD 的三 seed 均值分别不高于 full/head。任何一项不满足即 `lora_no_candidate`。
- 若 LoRA 未通过，仅当 full/head 中至少一个的负迁移格数不超过 2（六格）时，才可按 K=5/20 macro 均值较优者作为 C 的固定适配参照；不宣称该 scope 优于另一个。两者均超过 2 则停止。
- **C 通过须同时满足**：K=5/20 分别满足 FOMAML−ERM 的配对改善规则；K=0 macro Δ≤0.1 且不触发逐库保护；K>0 的 FOMAML 负迁移格数不超过 ERM 对照。才可主张该 backbone/适配方式下的元学习增益。

### 5.3 预算与停止

- 固定 4000 updates、meta_batch=4、query_size=64、accumulation=1、outer Adam LR=3e-4、SmoothL1、评估适配 5 步、seed 42/43/44；C 的 FOMAML 训练内循环同样 5 步。不得按结果延长某臂或选择 best checkpoint。
- 实现后先 smoke 测单步时间、内存、磁盘与 4000 步外推，外推只标 `estimate`，不得写成实测。正式运行前另行提交总 GPU/CPU 小时与资源上限，经用户批准。
- 任一步出现非有限损失/梯度、错误标签空间、目标库读取、来源 hash 变化、容量超限静默处理、跨 episode fast weights 污染，立即失败留痕，不自动重试。
- 同一命令第三次失败即停止汇报。失败不通过改 seed、删查询、过滤困难库或放宽判据解决。

## 6. 实现边界与最小文件职责

当前训练固定构造 `SetEstimator`，checkpoint 只识别 `grouped_set_residual`，`adapt_scope` 只有 `all/head/last`；所以本方案不是配置开关。最小实现范围为：

1. 新 PRICE-style residual wrapper，明确截断原 PG 融合头、输入/输出标度与容量配置。
2. 模型工厂与 checkpoint `model_type` 分派，保持旧 checkpoint 严格加载不变。
3. `scope=lora` 参数选择与内外循环接口，训练/评估配置一致性检查；将 fixed-budget 的 `FIXED_MODEL`/`FIXED_ADAPT_SCOPE` 绑定到阶段配置，而不是删除守卫。其余预算字段保持冻结，新增错误 model/scope 的拒绝测试。
4. 只读源计划、预算/资源监督、hash 绑定与结果归档。不得修改 `/gly/weilin/PRICE-master`、历史缓存或现有未提交文件。

实现遵循测试先行，另开实现分支；本协议通过审查不等于实现授权。

## 7. 必过验证

| 类别 | 拒绝条件/断言 |
|---|---|
| PG 恒等 | 零初始化 forward == `pg_log`；候选 trunk/head 无第二路 PG；0-step 不变 |
| 标签 | 候选 support/query loss 与预测均为 `log(max(y,1))`；调用 PRICE 标签转换即拒绝 |
| 输入 | fanout 还原、padding 与 mask 正确；单样本与混合 batch 对同 query 一致 |
| 容量 | 容量恰等 max count 通过，超过即显式 unsupported；容量只由训练库生成并写入 checkpoint |
| LoRA | support 只改变 A/B，base 不变；query loss 对 base 有非零梯度；K=0/0-step 无变化 |
| FOMAML | 一阶梯度符合定义，fast weights 不跨 episode；非有限内外梯度失败 |
| checkpoint | 新旧模型各自往返预测一致；模型类型/config/容量/缓存 hash 不匹配拒绝 |
| 确定性 | eval 模式两次 support 适应与预测一致；dropout=0，训练/eval 模式不混用 |
| 数据 | 仅允许的源库分片；support/query 无 hash 交集；目标库守卫生效 |
| 统计 | 独立 NumPy 复算 log-Q95、macro、配对 SD、空桶和负迁移；不能仅调用生产汇总函数自证 |

## 8. 归档与允许结论

每阶段保存协议及代码 hash、配置、sealed plan、源缓存 hash、模型容量、适配参数量、完整训练/评估 manifest、逐 query 预测、每 seed/库/K/run 汇总、资源计数和失败原因。checkpoint/大数组留在授权运行目录，不提交 Git。

允许写：

- “在指定源验证库、lite 容量和固定预算下，观察到/未观察到指定配对差异。”
- “LoRA-only 内循环在该 backbone 的指定 K 下改善/未改善适配。”
- “FOMAML 相对同结构 ERM 的增益成立/不可判定。”

禁止写：

- “证明旧模型太简单”“PRICE 架构解决跨库漂移”“LoRA 必然防止过拟合”。
- 把官方 PRICE 权重当严格 23/3 初始化，把源验证筛查当独立盲测，把历史目标库数字用于本轮选型。
- 把参数减少等同于训练/推理成本降低，把离线 Q-error 改善等同于 PostgreSQL 端到端加速。

## 9. 与既有路线的关系

- 不重启此前独立 worktree 中的 121 号联合分布采样草案，不把其二库/三库联合信息假设偷换进本协议。该文件不在当前 `main` 基线，本协议不引用其实验结论或视为已交付路线。
- 不推翻结构恒等式负结果，不重新将 `struct_log` 引入本候选；PG 与 struct 不叠加。
- 不将本候选自动替换主表。若 A/B/C 未通过，保留历史 ERM-sum 结论及本候选负结果；若通过并经 D 验证，再讨论是否作为“PRICE-style 编码器 + 低秩元适配”的新实验章节。
- 本协议不宣称轻量主线已放弃：lite 首先检验结构，官方大容量 PRICE 不是首版候选。

## 10. 当前审查与授权状态

- 2026-11-11 已对原组合进行两路独立只读审查：训练数学可行，但发现 PG 重复、标签空间、LoRA scope、checkpoint、容量与对照混杂问题；本稿据此修订。
- 本稿经未参与撰写的独立只读审查及两轮复审：初审指出 PG 接口/容量混杂、per-head LoRA 计数、B 训练轨迹混杂、判定分母与 fixed-budget 等缺口；复审确认 W2–W8、S1–S9 及 PG 接口问题已关闭，未发现新的执行阻塞。
- 最终补充预注册：B 固定 lite、A2 只作小容量方向诊断；C 两臂保留 B 的相同 LoRA 参数化和 ERM checkpoint；稳定性使用相对 K=0 的配对差值 SD；A0-ref 复用须满足无早停、未读源验证、只取 last。**容量影响仅部分识别，A2 不是等参数对照。**
- 文档已执行相对链接、`git diff --check`、关键约束文本及静态参数构造复算；未运行模型单元测试/ruff（本轮无代码改动），未验证服务器 sealed plan、目标库或历史 checkpoint。独立审查不构成实现、训练或目标评估授权。
- **未实现、未训练、未读目标库、未部署、未授权正式运行。**
