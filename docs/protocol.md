# 第一阶段实验协议

## 问题与范围

研究目标是空间漂移：来自不同数据库的数据分布、schema 与 workload 变化。目标域允许取得少量带真实基数的查询反馈。时间漂移、动态增删改、在线上下文学习、PostgreSQL 插件和端到端优化器收益不在本次实验范围。

## 划分与防泄漏

- 源数据库：26 个。`chembl / legalacts / telstra` 为验证库，其余 23 个为训练库。
- 目标数据库：`imdb / stats / genome / ergastf1`，不进入训练或超参数选择。
- 目标 support 来自 finetune，test 来自单独 test workload。支持池移除与 test 规范化 query hash 相同的查询；相同查询身份不能出现在 support 与 query 两侧。
- SQLGlot 18.17.0 对 SQL 做保守 AST 规范化：忽略注释/格式与非引号标识符大小写，保留字面量。不保证检测一切语义等价 SQL，例如连接顺序重排。
- 相同 SQL 的重复结果不能直接当独立证据。保留完整测试 workload 口径，同时报告 unique-query 指标。
- 新缓存保留源行号、SQL hash、源文件 hash、原始 double 精度标签。历史 pickle 的 float32 标签用于顺序匹配；匹配后使用 workload 标签。
- 标准缓存采纳之前检查全部字段/维度/有限值、完整标签序列，并对首/中/尾查询重新生成特征。抽查生成一致不是全体特征内容的独立再验证，报告中明确该限制。
- finetune 的统计信息是部署可取得的无标签数据库信息；统计文件生成过程是否受 workload 选择影响仍需披露。不把统计预处理称为零成本。

## 模型

四组共享 token MLP：连接列直方图 40 维、fanout 40 维、表 4 维、过滤 43 维，各使用 64→64 隐层。每组 masked mean/max 池化后拼接四类数量、PG 对数估计和表 log-size 之和，经 128→64→1 head 输出对 PG 对数估计的残差。

- 实际参数数目 99,777；纯 FP32 参数为 399,108 字节，不等于包含优化器和元数据的 checkpoint 文件大小。
- 训练标签 `log(max(card,1))`；fanout 做 log1p；表规模特征已经取 log，不再重复处理。
- 不拟合额外数据集统计量，不使用 Dropout 或 BatchNorm；所有变换固定、版本化。
- 对组内顺序不敏感，适应未见长度的特征集合，但不显式建模完整 join graph，不能提前保证其精度超过 PRICE。
- 默认 residual 最后一层为零，因此未训练模型从 PG 输出开始；“随机初始化”对照也是这个起点，而非随机基数，必须说明。

## 训练与模型选择

- FOMAML：每个数据库为一个任务，K 从 5/10/20/50/100 均匀选择，query=64；默认内循环 5 步、meta batch=4、外循环 Adam lr=3e-4、SmoothL1 loss。
- ERM：同结构同源库，普通监督学习每个 episode 的 support+query 样本；目标适应预算相同。两个算法反向次数不同，记录样本曝光、backward 次数、更新数和 wall time，不能按同 epoch 宣称等成本。
- 内循环 use `functional_call`；一阶梯度截断而初始化通路保留。所有 episode 从共享初始化重新开始；部署适应使用 detached fast weights。
- 使用固定验证 episode：每个验证库、每个 K、每个 repeat 的 query/support IDs 保存到 manifest。主选择指标为验证适应后的 macro mean log-Q95。
- 默认 constant LR，没有调度器；checkpoint 保存 optimizer/RNG/episode RNG/best metric/early-stop/config/source manifests。改变训练目标、数据 hash 或超参数时禁止直接 resume。
- 内循环学习率候选 1e-3/1e-2 由验证集决定；没有在测试集挑选“最佳适应次数”。候选搜索成本计入方法成本。
- 提供的 PRICE 权重使用了全部 26 个源库，含本研究验证库，故只能作为补充参考基线，不能作为严格 23/3 主实验初始化。

## 对照与消融

主对照：PG、PRICE 原前向+提供权重的零样本/少样本微调、轻量 ERM 零样本/微调、轻量 FOMAML 零样本/适应、轻量随机初始化+支持集训练。

所有目标测试遵循相同支持 manifest；每 run 的 K 使用嵌套支持集合。主实验 3 个训练种子 × 5 次支持抽样，分开报告两类不确定性。消融包括不用 PG、仅 head 适应、0/1/5/10 适应步数。运行器的默认步数是起始预算，不是已验证的最优预算。

## 指标和计时

- 标准 Q-error：`max(max(pred,1)/max(true,1), max(true,1)/max(pred,1))`；报告 Q50/Q90/Q95/Q99 与均值。
- PRICE 兼容值独立命名 `price_shifted_q*`，使用 `(pred+1)/(true+1)` 口径；禁止与标准 Q-error 混列比较。
- 每数据库、每 K 保存全部 run，分别汇总支持抽样与训练种子波动；不把 workload 中重复子查询当统计独立重复。
- 逐查询 CSV 保存 query ID/hash、double 精度真实基数、log prediction；不把模型输出 exponent 溢出悄悄裁掉制造好结果。
- 推理计时包含 warm-up，GPU 同步；报告设备、线程、batch=1 p50/p95。参数规模与文件字节数分别报告。
- 支持集适应计时不含已有标签采集和特征提取；离线结果不能证明 PostgreSQL 端到端加速。

## Smoke 与正式实验

`smoke` 使用极少数源库 episode、每目标少量真实 test 查询，只验证数据→训练→验证→重载→适应→预测闭环。所有产物标为 `smoke-not-research`；不能据此写“精度提高 X%”。正式实验启动前必须完成全部源库缓存、IMDB 数据障碍处理以及 GPU/时间预算确认。

## IMDB 当前障碍

2026-09-25 核查：finetune SQL 共 50,000 条，但对应直方图只覆盖部分列；第一条查询要求 `imdb_cc.movie_id` 等未包含的统计。test 统计同样没有这些列；现有三个 PRICE zip 也无 IMDB finetune 特征或原始表。随机抽查 300 条支持 SQL，30 条可生成，270 条因缺失统计失败。

不得：生成随机特征、用 test 查询做 support、从 test 标签拟合模型、静默排除约 90% 不可生成查询。可选后续处理是获得完整统计/原始 IMDB 表后重建，或经用户确认仅在可完整提取特征的真实 finetune 子集上做适应并披露覆盖偏差。当前实现默认报错；`--allow-exclusions` 仅供明确批准过滤方案后使用，并把每条失败 ID/原因写入 manifest。
