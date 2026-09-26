# PRICE 现有代码评估与处理决定

日期：2026-09-25。对象：服务器 `/gly/weilin/PRICE-master`，Git HEAD `0a9358eafb332201678e411ac35dc78141f751bd` 及其工作区改动。源码快照和诊断保存在本地 `price-audit-20260925/`。此报告是工程/实验协议审查，不是文献新颖性评审。

## 结论

现有数据适合第一阶段跨数据库少样本研究，但旧适应/评估链路不适合直接产生论文结论。保留 PRICE 原项目和数据，只在独立目录实现新流程。最新 V2 训练器的监督式 FOMAML 基本梯度通路有效，不应将旧适应分支的问题泛化为“所有元学习代码都无效”。

## 已验证问题

| 问题 | 来源位置（原项目） | 证据与后果 | 新流程措施 |
|---|---|---|---|
| 适应不改变基数预测器 | `quick_adapt_v2.py:39–49`、`model/meta_encoder_v2.py:85–121` | CPU 小模型适应 3 步：42 个元组参数 tensor 改变，基数参数改变 0 个，预测差 0；重建 loss 从 0.7617 降至 0.6391 仍无基数收益 | support 必须是查询特征+真实基数；同一个回归模型做内外循环 |
| 缺失数据用随机元组替代 | `utils/model/tuple_dataset.py:27–41,88–97` | 预期 `datas/statistics/imdb/table_stats.pkl` 不存在，实际统计有分区且文件名不同；触发高斯噪声 fallback | 缺失数据报错，禁止静默合成支持集 |
| checkpoint 被忽略 | `meta_evaluate_v2.py:48–49`、`quick_adapt_v2.py:171–175` | trainer 保存包裹字典，loader 当 state_dict 加 `strict=False`；诊断 166 missing keys、0 个参数被加载 | 版本化 checkpoint，严格 key/shape/预处理校验 |
| 评估接口不兼容 | `meta_evaluate_v2.py:70–98`、`utils/model/dataset.py:52,58,81` | test loader 返回 8 项却解包 7 项；缺失 test_list_lens；不存在 batch_size 参数。诊断均复现 | 统一 FeatureStore 与分组 batch 接口，不吞异常 |
| 最佳模型按训练 Q95 选 | `meta_train_v2.py:851–908` | 默认 q95 参数来自 train_epoch；传入更差验证值仍选择训练 Q95 | 只按固定源验证任务适应后指标选模型 |
| 旧 full-MAML 断图 | `model/meta_encoder_v2.py:283–333` | `.data` 加载与 fast_weights 的 autograd 不相连；CPU 2 步 fast 参数变化 0 | functional_call，无 .data 改写 |
| 输出硬裁剪造成死梯度 | `meta_train_v2.py:375,601,605` | 负预测被裁到 0，非零误差下 inner/outer 梯度都为 0 | 训练预测不裁剪；指标解码使用基数下限 |
| 官方基线前向被改变 | `model/encoder.py:62` | 特征生成已 `log(table_size)`，新增 log1p 再次改变权重输入语义 | 独立 vendor 保留 Git HEAD 变更前 forward；严格加载已有权重 |
| 旧 LOO 不是逐库重训 | `meta_evaluate_v2.py:143–183` | 每个库加载同一 checkpoint，没有验证该库排除于训练；且 usage=finetune 对许多源库无文件 | 固定 23/3/4 数据库划分；本阶段不声称完成 LOO |
| few-shot 指标不是基数误差 | `quick_adapt_v2.py:225–304` | 输出最终重建损失，并未在独立查询集评估 Q-error | 独立 test 查询 + 标准 Q-error + 单独命名平滑指标 |
| 数据存在重复 | 全部 workload 扫描 | Genome finetune/test 至少 2 条 exact SQL 交叉；SAP 50,000 行仅 29,714 条 exact SQL | 基于查询身份去重，目标 support 排除 test hash，报告 unique-query 口径 |
| IMDB 适应特征缺失 | `setup/features/finetune/imdb/` | SQL/统计存在，目标 finetune pickle 不存在 | 新项目独立生成带追溯 ID 缓存，不借 test 支持样本 |

## 正向证据

- 最新 `meta_train_v2.py:369–406` 的 functional_call 一阶更新有正确初始化梯度。诊断 w=2、lr=0.1：fast w=1.8、outer grad≈1.6、原 w 保持 2。
- 四个目标测试集以及 fnhk-pretrain、ergastf1-finetune 六份 pickle 的字段长度、分组尺寸、有限值检查通过。后续全库审计结果另存，不能把六份检查说成全部验证。
- workload 含 SQL、真实基数、PG 估计，可以开展不依赖原始数据库实例的离线监督式跨库实验。

## 边界

诊断使用小型人工输入仅验证软件行为，不能当作研究结果。现有权重本身不能据此判定“全部无效”，但其生成代码/配置/评估流程需要可追溯。数据文件有效也不等于真实标签已独立重新执行 SQL 验证；本阶段信任用户提供的 PRICE 数据源。原项目无 LICENSE/NOTICE，vendor 仅用于当前私有研究，不应未经许可公开分发。
