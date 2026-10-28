# 112 FOMAML 适应前后联合目标：源侧小实验方案

## Material Passport

- Origin Skill / Mode：academic-research-suite / experiment-agent / plan。
- Origin Date：2026-10-27（Asia/Shanghai）。
- Verification Status：DESIGN_ONLY / UNVERIFIED（研究假设与执行设计，非实测结果）。
- Version Label：fomaml_joint_source_plan_v2（独立初审后修订）。
- 用户范围：**只写方案；不改训练代码，不部署，不启动 smoke、训练、推理或目标评估。**
- 本方案独立于 [111 号](111-FOMAML-sum主表评估计划.md) 的已冻结目标评估，不替换其结果或报告表格。

## 1. 研究问题与已有事实

**问题：在其他条件固定时，给 FOMAML 加入适应前的 query 监督，能否在不明显损害零样本精度的同时，改善少样本适应后的精度？**

当前 `meta_loss` 只优化适应后 query loss（[meta.py](../spatial_meta_ce/meta.py)）；K=0 不做内循环，但使用的仍是元训练初始化，因此 K=0 可以评价其零样本泛化，不能预先断言它一定最差或与元学习无关。

当前训练以 SmoothL1 均值为目标，checkpoint 选择已使用源验证 episode 的 macro log-Q95（[train.py](../spatial_meta_ce/train.py)）。目标错位可能影响结果，但**尚未证明它是效果不佳的原因**。

现有 [evaluate.py](../spatial_meta_ce/evaluate.py) 在每次 support 适应后评估全量唯一 test 查询；“support 重复的 Q95 均值”不等于只测小 query batch。旧 few-shot 使用 `best.pt`、111 号补测使用 `last.pt`，两批数字不同不能全部归因于 episode/population。

本轮只检验联合目标，不同时改尾部 loss、架构或适应算子。[S6](23-当前任务清单与源侧改进总结.md) 与 [P1](94-P1精度杠杆实现与源侧筛查.md) 已试过上尾混合、log-L1、pinball，不能把这些当成全新且保证有效的方案。**本方案不声称直接优化总体 Q95，也不预设 FOMAML 胜出或把联合目标本身认定为原创算法。**

## 2. 唯一干预：适应前后联合目标

每个源训练库为一个任务，support 为 S、query 为 Q，二者 query_hash 不相交。标签保持 `log(max(true_cardinality, 1))`，逐样本损失保持原 SmoothL1（beta=1），不新增输出裁剪或标签变换。

```text
θ' = U(θ, S)                         # 原 FOMAML 内循环，5 步
L_pre  = mean SmoothL1(f_θ(Q),  y_Q)  # 未适应初始化上的 query loss
L_post = mean SmoothL1(f_θ'(Q), y_Q)  # 适应后的 query loss

原 FOMAML：L_B = L_post
联合目标：L_C = 0.5 × L_pre + 0.5 × L_post
```

- **仍为 FOMAML**：内循环 `create_graph=False`，保留初始化的一阶梯度通路；`L_pre` 直接回传到 θ，`L_post` 经 fast weights 回传到 θ。不能 detach 整个 `L_post`，也不能把适应后的权重写回公共模型再累计到下一个任务。
- 两项使用**同一组 Q**，没有额外标签、额外 query 或验证库反馈进入训练。每个任务完成后重新从公共初始化适应。
- 固定权重 0.5/0.5，只设这一候选；不是已验证最优，不扫 λ，不增加多步 loss、tail loss、学习率搜索或 warm-start。
- C 对 B 同时增加 pre 项、将 post 项系数减半。因此估计的是**联合配方的整体效果**，不能分离“pre 监督”与“post 梯度变小”的各自贡献；不通过额外调大学习率补偿。

## 3. 三臂、三 seed、同一训练流

| 臂 | 架构 | 训练目标 | 角色 |
|---|---|---|---|
| A_erm | sum64 + PG residual | 原 ERM：对 S∪Q 的 SmoothL1 均值 | 无元训练对照 |
| B_fomaml | 同上 | L_post | 原 FOMAML 对照 |
| C_joint | 同上 | 0.5 L_pre + 0.5 L_post | 唯一候选 |

固定训练 seed **42/43/44**，共 **9 次从头训练**；不按 seed42 的好坏决定是否补另两个 seed。训练失败导致缺格时整矩阵标 invalid，不用两 seed 冒充三 seed。

| 固定项 | 设计值 |
|---|---|
| 源训练/验证 | 23 库 / chembl、legalacts、telstra，沿用当前 data split |
| 模型 | `width=64, use_pg=true, arch=sum, layers=2, heads=4`，无额外门控 |
| 初始化 | 同一 seed 三臂的初始 state_dict 完全相同并绑定 SHA256；不同 seed 独立 |
| 更新预算 | 每臂 4000 次 optimizer update；meta_batch=4、accumulation=1 |
| 优化器 | Adam、outer_lr=3e-4，固定 LR；保留现有梯度范数裁剪 1.0 |
| episode | 数据库等概率；K 在 {5,10,20,50,100} 等概率，query_size=64 |
| 内循环/评估适应 | SmoothL1，lr=1e-3，5 步，scope=all |
| checkpoint | 仅第 4000 次更新后的 `last.pt`，不使用 `best.pt` |
| 训练中验证 | 正式训练关闭验证库读取与早停；9 个 last 的 SHA 全部封存后再做源侧端点评估 |

以 [meta_sum_4k.json](../configs/meta_sum_4k.json) 为配置参考；固定预算下必须**关闭训练期验证读取与早停**，不直接照搬其中 `validate_every=100` / `patience=10`，也不能仅将 patience 设为 4000 后仍周期性读取验证库。未来训练入口只用 23 个训练库；三个源验证库留到所有 9 个 last checkpoint 的 SHA 和配置均封存后再访问。固定 4000 更新，不以验证指标挑 checkpoint；仅保存/评估第 4000 次后的 `last.pt`，生成的 `best.pt` 一律拒绝。实现测试须证明 4000 步训练期间未调用 `validation_plan/validate`、未读取 validation store，且训练库闭包不变。这是必要的验证模式控制，不是本轮代码改动。

每 seed 事前生成并封存 16,000 个 episode 的数据库、K、S/Q query_hash 及顺序；三臂读同一份计划，不依赖训练中消耗 RNG 的偶然一致性。采样和模型初始化 RNG 分离。全配置 diff 白名单只允许臂的 method/联合目标开关及输出路径；跨 seed 只允许 seed 与对应计划/初始化 hash 不同。

**成本不假装匹配**：A 对 S∪Q 普通回归，B/C 另有内循环，C 重用 Q 增加一次前向及梯度路径。固定的是初始化、采样和外更新数，不是相同 FLOPs、样本呈现次数或秒数。分别记录 unique 标签身份、support/query 呈现次数、forward/backward 调用、更新数、wall time、峰值内存；B/C 一次合并 outer backward 的调用次数相同也不代表算力相同。

**历史 checkpoint 不复用**：不拿旧 ERM 或 FOMAML 的 best/last 与新 C 做主比较，不从旧 best 续训，不根据刚看过的目标库成绩选模型。历史产物尚未认证符合本轮封存的初始化 SHA、episode 计划与固定 4000 更新规则，不能仅凭文件同名就替代重新训练的 A/B 对照。PG 可从同一批源 query 的原始缓存给出静态参考；本小实验不重跑 PRICE。

## 4. 源侧完整查询评估：support 与 query 永久隔离

本轮只读源 `pretrain` 缓存，不读目标 `finetune/test`。三验证库均不进入任何臂的训练。评估 K 固定 **{0,5,20}**，**K20 是唯一主适应终点**，K5 为小标签预算诊断，不能在 K20 失败后改用 K5 宣称成功。

现有训练器的 `validation_plan/validate` 会读取三验证库全池中的 episode；本方案的正式固定预算模式必须禁用这条通路，而不是保留日志后宣称“从未读取”。`binding.json` 记录 `validation_monitor_reads: []` 并由实际 store 访问审计验证。全部 9 个 last 封存后，才按下面的事前规则生成 source_eval_plan 和计算端点；方案允许的未来工程 smoke 另行授权、记录其读取，不能被称为独立盲测。

每库建立一份全臂、全 seed 共用的固定计划：

1. 用 query_hash 去重；若同 hash 的真实标签或特征不一致，停止，不任意选一个。规范代表按 `(query_hash, id, source_line)` 稳定排序并写入计划；`source_line` 是现有 manifest 字段，不发明 `source_row`。
2. 用独立 `support_seed=2026`、固定库顺序 `(chembl, legalacts, telstra)` 及库索引生成一次无放回排列，不按真实基数、PG 误差或模型误差分层筛选。
3. 前 **100** 个 unique query 作为保留 support 区，按排列顺序切为 **5 个互不重叠的 20 条 block**。第 r 次 support run 使用对应 block：K5 取前 5 条，K20 取全部 20 条。
4. **Q_d 为该库其余全部 unique query**。整个 100 条 support 区在所有 run、所有 K（包括 K0）中均不进入 Q_d，因此 Q_d 在三臂、三 seed、K0/K5/K20、五次 support 抽样间完全相同。
5. 固定计划保存库级 S/Q 列表、cache content hash、标签/特征身份摘要及 plan SHA256。没有重抽 support、丢弃坏 run 或根据结果补样本的路径。

这是**固定的全量留出 query population**，不是含 support 的全库所有查询。按 [81 号](81-源库留出协议.md) 已有 unique 规模，计划分母为 chembl 49,900 / legalacts 49,404 / telstra 49,900（各扣 100）；这是由历史规模推导的**计划值，不是本轮实测**。执行前重新认证，任何不一致先阻塞并解释，不更改分母硬凑。每 run 只用 5/20 条 support，不因预留 100 条就声称用了 100 个标签。

- K0 每个模型/库只推理一次；五个 run 共用该值，不复制成五个独立样本。
- K>0 每次均从该臂同一 `last.pt` 的公共初始化重新适应，然后完整遍历 Q_d。每次只消费本 run 的 support；不累计跨 run、跨 K 或跨库适应。
- 矩阵为 `3 臂 × 3 seed × 3 库 × (1 个 K0 + 2 个 K × 5 run) = 297` 个评估单元，其中 K0 为 27 个，K>0 为 270 个。每单元分批推理，但 Q95 从完整 Q_d 合并计算，**不平均 batch Q95**。
- 现有 `evaluate.py` 的全量推理/去重思路可以复用，但不能把验证库强塞进 `splits.test` 绕过 split guard。现有 [export_source_predictions.py](../scripts/export_source_predictions.py) 会让 source support 落入全量 query，**原样运行不符合本方案**；需未来实现独立的 source-only、显式不交叠路径，不改旧审计定义。

## 5. 指标与汇总：先库内全量，再 run，再库/seed

唯一主指标是**全表 standard unique-query Q95**，不是 4+ 表 macro、PRICE shifted Q95、pooled 跨库 Q95 或训练 loss。本轮固定 query 子集与 94 号源侧 4+ macro（44.1443 基线）、14 号目标库 per-K 表、111 号目标库全量均不同源，不能将绝对数值当成同条件对照并排排名、换算或归因。

```text
log_q_i = abs(max(raw_log_pred_i, 0) - log(max(true_cardinality_i, 1)))
q_i = exp(log_q_i)
Q95 = numpy.percentile(q_i, 95, method="linear")
```

沿用 [metrics.py](../spatial_meta_ce/metrics.py) 的标准指标 floor，不对预测做任何新增裁剪。`exp(percentile(log_q,95))` 单列为 log 空间诊断，不冒充上式；这两个分位数插值结果不保证完全相同。非有限输出或指数溢出即 invalid，不删除/截断后出表。

令 `q[a,s,d,k,r]` 为一个全量 Q_d 的 Q95：

- `qbar[a,s,d,0]` 为唯一 K0 值；K>0 的 `qbar` 是该 seed/库/K 下 5 个 support run 的 Q95 算术均值。
- `M[a,s,k] = mean_d qbar[a,s,d,k]`，三库等权算术平均，不按查询数加权、不合并预测后取分位。
- 主差异 `g_s(C,B,k)=1-M[C,s,k]/M[B,s,k]`；正值为改善。先按 seed 配对，再报三个 `g_s` 的均值、样本 SD、最小/最大值、正向 seed 数。
- 同样完整报告 C 对 A、B 对 A，以及各臂自己的 `g_self_s(a,k)=1-M[a,s,k]/M[a,s,0]`。各臂自己的适应增益不能用 PG K0 替代。
- 逐库/逐 K 报三 seed 的 `qbar` 均值±样本 SD；另报每 seed 的 5-run support SD，并在正文完整列出 per-seed/per-db 的 C/B、C/A 和自身 K20/K0 比值，不让跨 seed 均值隐藏单格退化。三个 seed 是训练重复，不能把 15 个值或 297 个单元当成独立训练重复。
- Q50/Q90/Q99、逐 query 配对 log-error 差、恶化查询比例、适应时间作为诊断全部保留；4+ 桶不参与本轮判定。

**不做显著性承诺**：n=3 仅为开发性筛查；不用“效应小于某组 SD”“SD 区间不重叠”当显著性检验，也不把历史 ERM 的 6.28pp 噪声当成本轮 FOMAML 的已知噪声。新增 seed 不能解决验证库被反复开发使用的问题。

## 6. 事前决策规则：联合目标、适应收益与元学习收益分开

下面的 **5% 改善/保护、10% 逐库保护是本小实验的操作阈值**：5% 表示本方案拟采用的最小相对收益/容许折衷，数值来自设计选择，不来自数据估计或已知噪声；10% 是逐库防退化线。它们不是功效计算、置信区间或对历史方案阈值的事后放宽，不控制假阳性率。看结果后不修改；过门至多支持讨论下一份确认协议，不自动解锁新库读取或目标评估。

### 6.1 联合配方是否有开发价值（C 对 B）

必须同时满足：

1. K20 的 `mean_s g_s(C,B,20) >= 0.05`，且至少 **2/3** 配对 seed 为正。
2. K0 保护：每个 seed 均满足 `M[C,s,0]/M[B,s,0] <= 1.05`。
3. 每库分别在 K0、K20 上：三 seed 平均 Q95 的 C/B 比值均 `<=1.10`；同口径 Q50 的 C/B 比值也均 `<=1.10`。不得用另一个库的改善掩盖单库明显退化。

通过只叫 `joint_objective_source_signal`。若仅 K0 改善、K20 不过门，记录该权衡，不能改用零样本作为成功主终点。

### 6.2 是否真正产生了有用适应（C 的 K20 对自己的 K0）

还须 `mean_s g_self_s(C,20) >=0.05`、至少 2/3 seed 正向，且每库三 seed 平均的 `Q95_C,K20/Q95_C,K0 <=1.10`。K5 的好结果不能替代 K20。解释按实际条件分开：

| 条件 | 允许的描述 |
|---|---|
| 6.2 全过 | 本开发矩阵内，C 的 K20 适应达到预设的自身 K0 改善线 |
| 自身平均增益 >0，但未过 5%/seed 一致性/逐库保护之一 | 有聚合正向观察，但有用适应门槛未通过，逐项报告失败 |
| 自身平均增益 <=0 | 适应未改善聚合精度；负值为聚合负迁移。仅当 B、C 自身平均增益都为负、且 C 更接近 0 时，才描述“聚合负迁移较 B 轻”，同时展示每库/seed |

只有 6.1 通过时，才能另外说“C 比 B 的适应后绝对精度达到改善线”；这不是“C 相对自身 K0 变好”。不得把失败自动改称“减轻负迁移”。

### 6.3 固定适应协议下，相对同结构 ERM 的配方收益（C 对 A）

K20 的 `mean_s g_s(C,A,20)>=0.05` 且至少 2/3 seed 正向，并且每库三 seed 平均 K20 Q95 的 C/A 比值 `<=1.10`；同步报告 B 对 A，不能隐藏原 FOMAML 对照。

A 的适应超参数未单独优化、计算成本也未匹配，因此此比较只支持“固定相同适应协议下 C 优于 A”，不单独识别纯元学习机制或其普遍优势。若 A 的 K20 比自身 K0 更差，必须披露，另报 `M[C,s,20]/M[A,s,0]`，防止把对照被适应拖坏当成新方法的绝对优势；这个诊断不参与选模或改变上述判定。

只有 **6.1–6.3 全过**才标为 `promising_source_screen`，仍不是跨目标库有效或算法创新已成立。部分通过标 `partial_source_signal` 并逐项列失败原因；6.1 未过为 `no_candidate`；缺格、泄漏或认证失败为 `invalid`，不与科学负结果混淆。

不按最好的 seed、K、库、checkpoint 重命名主线。若失败，停止本方案，不自动扫 λ、改 inner lr/steps、加 tail loss、补 seed 或开启目标评估。

## 7. 未来实现和验收要求（本轮不实现）

最小实现范围：`meta.py` 的联合目标路径；`train.py` 的显式目标配置/签名与 loss 日志；source-only 固定计划和全量端点评估；必要的配置、测试与汇总。复用现有 adapt/functional_call/StorePool/metrics，不另造通用实验框架，不修改历史 runner 或默认训练行为。

- 新目标及权重进入 training signature、checkpoint metadata 和配置 diff 白名单；不同目标不得 resume 同一旧 optimizer/checkpoint。
- 当前训练器启动时的 `sources=train+validation`、`manifest_hashes` 和固定 `metadata.selection` 也须在新固定预算路径中一起对齐：只打开 23 个训练库，训练 checkpoint 的 `manifest_hashes` 只含训练库，`selection="none (fixed 4000 updates, last.pt only)"`、`selection_databases=[]`、`validation_databases=[]`；预定源评估库名单放独立 binding，不冒充已经验证/选择过的曝光记录。正式训练的“未读取验证库”包括不打开其 rows、特征/标签数组及 manifest，不用 mmap/unique_indices 的间接访问绕过审计。旧训练模式保持原行为，新模式测试须在构造 StorePool/metadata 时就断言白名单。
- B 与 C 都记录 `L_pre/L_post/L_total`（B 的 pre 只作无梯度日志），防止“记录的是联合 loss、反向仍是 post-only”。保存少量确定性梯度审计样本供独立复算。
- 三臂均记录每次 outer update 的裁剪前梯度范数，以及 per-seed 的 count/mean/max/clipped_count；裁剪阈值仍为 1.0，不因 C 的梯度不同而调阈值。若更新受裁剪的比例不同，作为优化解释保留，不据此事后修改配方；负结果只能否定当前联合配方，不能证明所有联合目标无效。
- TDD 覆盖：在固定合成任务上，C 的 outer gradient 等于两条一阶路径梯度各半之和；pre/post 两项均确实影响 θ；只适应 fast weights 不改公共参数；任务/run/K 重置；旧目标路径保持兼容。
- source/target 白名单、正式训练读取验证库、S/Q 不交叠、nested support、完整 Q_d 分母、不同 batch size 的合并指标、唯一 K0、三 seed 完整矩阵、非有限、SHA 错配、未满 4000 步、非 last checkpoint、错误臂/seed/config 全部有拒绝测试。
- 汇总独立用 NumPy 从逐 query 预测重新计算 Q95/Q50/判定；不只调用生产汇总函数然后自称独立验证。
- 实现后先本地单测/全量 unittest 与 ruff，再独立只读审查。未通过不得提交执行包或启动研究运行。

## 8. 证据与未来产物

所有路径均为**拟定**，现在不创建实验输出：

| 产物 | 最少内容 |
|---|---|
| `binding.json` / `protocol.json` | 方案/代码/配置 SHA、环境版本、白名单、预算、`training/selection/evaluation_databases`、正式训练 `validation_monitor_reads: []`、`target_databases_read: []` |
| `initializations.json` / `training_plan.json` | 每 seed 共享初始权重 SHA、16,000 episode 顺序与 S/Q 身份、23 库闭包、独立 RNG |
| `source_eval_plan.json` | 3 库固定 100-support 区、5 个 block、Q_d 全列表、内容 hash 与分母 |
| `runs/<arm>/<seed>/last.pt` | 4000 更新、目标/权重/数据/采样签名、optimizer/RNG；只留服务器 |
| `loss_audit.jsonl` / `runs.json` | 实际 pre/post/total、梯度抽检/裁剪统计、实际更新与成本、各格状态，不写臆测执行声明 |
| `predictions.csv` | `arm,seed,database,k,run,query_id,query_hash,true_cardinality,log_prediction`；全量只留服务器 |
| `summary.json` / `decision.json` | per-run、per-seed、per-db 指标，配对差/比例，6.1–6.3 逐项布尔与失败原因 |

时间戳用带偏移的 ISO-8601（例如 `2026-10-27T22:00:00+08:00` 为格式示例，不是执行时间）。所有 summary 绑定 checkpoint SHA256 和 query-plan SHA256，不能只记录文件名。训练/评估标签与缓存只读，不 prepare、不重建、不补伪标签、不连接数据库执行 SQL。

拟定服务器目录为 `/gly/weilin/spatial-meta-ce-fomaml-joint-source-20261027-code` 与对应 `-runs`；本地精简归档 `实验文档/results/fomaml-joint-source-<实际执行日期>/`。日期以真实执行日为准，不倒填。checkpoint、逐 query 大文件、`artifacts/`、`runs/` 不提交。

## 9. 成本、阶段与停止

这是控制变量数量的小实验，不是已知分钟级的便宜实验：正式最多 9 次训练、36,000 次外更新、297 个全量评估单元；C 有额外计算，当前无这条配方的计时结果，**不承诺耗时或精度**。

后续分阶段、分别获得执行授权：

1. **实现阶段**：只改代码、合成测试与只读文件认证，固定新配置/源码 SHA；不因方案批准就启动训练。
2. **工程 smoke / 计时预检**：未来另行批准后，先做极少更新与重载验证，再对 A/B/C 各做固定 200 更新计时、对每个源验证库各做一个 K0 和一个 K20 全量 query 单元计时，记录首次加载与稳定吞吐；均标 `smoke-not-research`，不按精度挑配方或外推收益。smoke 权重不接续正式训练，验证库访问单列，不混进正式训练的 `validation_monitor_reads: []`。计时只归档时间、数量、资源及身份 hash，不生成精度表或保留逐 query 预测供挑方案；200 步计时只提供带余量的成本外推，不保证后续每步等时。预检是独立授权、独立预算，不计入正式父任务的 43,200 秒，但须单独报告并计入方法总开发成本；预检自身资源/超时和源码 hash 同样需先绑定并另行授权。
3. **正式阶段**：展示最终 SHA、9-run/297-cell 矩阵、实际计时与资源预算后再获授权。固定 seed 顺序 42/43/44、每 seed A/B/C；从头训练全矩阵，不按中途成绩停掉较差臂。全部 last 固定后才做本方案端点判定。
4. **收尾**：独立数值复算与只读审查，保留负结果；不自动扩展新库、训练或目标评估。

拟定保护上限：父任务总 wall time 43,200 秒、单次训练 3,600 秒、单评估单元 600 秒、RSS 12 GiB、VRAM 8 GiB、总输出 4 GiB。父总时限是唯一累计预算，单训练/单评估上限是硬 timeout 子界，不能把 9×3600 与 297×600 的最坏上限当成可同时满足父时限，也不能据此假设矩阵可完成。**当前没有 C 配方或 297 单元矩阵的实测耗时，整个小实验是否能塞进 43,200 秒未知。** 未来 smoke/预检必须实测 A/B/C 代表训练以及单个 source-only 全量评估单元，并按 9 训练、297 单元、调度/归档余量保守外推；仅当预计总成本留在 43,200 秒内且每单项低于子 timeout，才另行展示正式预算等待批准，否则 blocked、重新设计更小矩阵并另行冻结，不自动放宽总预算。单个评估单元到 600 秒即停止该运行，整矩阵不完整标 invalid；不能把“子界通过”误作“总预算充足”。输出 4 GiB 也须由 smoke 按逐 query CSV 实测估算总量后再确认。选择现场空闲的单 GPU，显式 `CUDA_VISIBLE_DEVICES`、确定性配置及单线程；忙则 blocked，不抢占。

NaN/Inf、梯度非有限、OOM、超时、输出超限、hash/config/身份错配、目标库路径访问、正式训练期读取验证库或 support/query 重叠立即停止，保留失败证据，只清理本次自有进程组；不自动 retry/resume/换卡/补格。正式训练禁用性能早停，资源/完整性失败停止与之分开。

## 10. 解释边界与后续门槛

- `chembl/legalacts/telstra` 已被反复用于开发，这次仍是**开发性源侧筛查**，不是独立盲测。3 seed 和新 support 计划都不能把旧库变成新库。
- 若通过，再另立验证方案；可参考 [81 号源库留出](81-源库留出协议.md) 的防泄漏做法，但 hepatitis/sakila 也已有历史结果，不能再称全新盲测，不自动复用其初始化或改动 81 号冻结协议。
- C/B 通过只支持联合训练配方的局部效果，不证明 loss 错位是唯一根因，不证明 Q95 最优，不证明胜过 PRICE，不产生端到端加速结论。
- 若不过门，不能因为“需要元学习创新点”而保留一条无效主张；仍可报告形式化、受控比较及负迁移边界，但方法的原创性与实用收益需分别论证。

## 11. 本轮状态与独立审查

- 已完成：文档设计与现有代码/协议核对；独立只读初审（无阻塞，指出预算解释、验证读取、收益归因、失败措辞、裁剪统计与口径边界等改进项）。
- 初审处理：补成本可行性门；正式训练关闭验证读取；限定 C/A 为固定适应协议下的配方比较；负迁移结论按条件分支；补梯度裁剪统计、历史口径禁混和 per-seed/per-db 表。未增加 λ 网格、对照臂或盲测主张。
- 未采纳为事实：父总时限与单项 timeout 必须相加的说法、未验证的 5% 假阳性概率；改为明确层级预算与操作阈值的不确定性。不新增单 seed 1.25 门槛或早停通用开关，避免不必要扩展。
- 本轮文档链接、代码块、矩阵算术与空白检查通过，Ruff 通过。全量 unittest 实跑 1572 项，16 skip、1 error：`test_meta65_joint_supervisor.SupervisorTests.test_each_external_phase_timeout_without_worker_deadline` 在 `killpg` 收到 `PermissionError`；未修改该代码或重试求通过，因此不推进 main。这不是新方法的实验结果。
- 未完成且未授权：联合目标实现、配置/plan 文件生成、smoke、训练、全量端点评估、目标评估。
- 实现状态（2026-10-28，仅代码与合成测试）：`feat/fomaml-joint-objective-20261028` 上的 `1bed3c7`/`8b75c39` 已实现联合目标、fixed-budget 训练、训练/source-eval 计划、K0/K5/K20 全量 Q_d 推理、独立 NumPy 汇总、梯度审计与 §7 拒绝测试。全量 1629 项 OK（16 skip）、ruff 通过；两轮独立只读复审 APPROVE。
- 计划生成（2026-10-28，仅只读源侧计划）：已在服务器生成 seeds 42/43/44 各 16,000 episodes 与 source-eval 计划；`plan_binding.json` SHA256 `bb1bf8…`，四份计划文件重新生成后 SHA 逐一相同，`load_*`/`verify_*_against_stores` 全部通过；证据 `results/fomaml-joint-source-plans-20261028/`。
- smoke/计时预检（2026-10-28，仅 `smoke-not-research`）：2-update 重载通过；A/B/C 200 updates 为 68.96/111.23/125.85 s；峰值 RSS 2.46 GiB、CUDA allocator 27.8 MiB、输出 17 MB；三库 K0 11.45–12.18 s、K20 predict 10.67–10.90 s、K20 adapt 0.07–0.11 s；`smoke_protocol.json` SHA256 `cedf65eb…`，证据 `results/fomaml-joint-source-smoke-20261028/`。只作资源量级，不作配方/精度/正式预算结论。未执行正式训练、端点评估或目标评估，未读目标库；已知残余是逐 query 配对 log-error 与恶化查询比例未汇总（`predictions.csv` 保留全行可补算）。实现代码与计划存在不等于执行认证或效果结论。
- 修订版独立复审：**通过（无未解决阻塞，仅设计文档）**。同一独立只读审查代理复核 v2 及最终 metadata/计时两段；未代跑测试或实验，测试、SHA 与链接校验由主代理实际执行并区分记录。§8 使用字段组简写，精确定义以 §7 为准。审查通过不等于实现完成、效果认证或执行授权。
