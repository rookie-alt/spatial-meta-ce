# 剩余输入门槛与本轮盘点

日期：2026-10-07；[只读目录/环境盘点](input-inventory.json)实际时间14:45:43（Asia/Shanghai）。盘点只读取已授权项目/源路径的目录元数据和工具可用性，不读取目标标签、测试内容或数据库凭据，不执行SQL、训练或推理。

## IMDB完整few-shot

- 项目 `/gly/weilin/spatial-meta-ce/artifacts/cache/finetune/imdb` 不存在。
- 原PRICE的 `/gly/weilin/PRICE-master/setup` 在三层目录扫描中发现 `setup/features/test/imdb/features40.pkl` 的文件名，但没有对应finetune目录。文件内容未读取，test数据不能补support。
- `/gly/weilin/PRICE-master/data`、`datasets` 和 `/gly/weilin/spatial-meta-ce/data` 不存在。该有限盘点不证明其他位置没有数据。
- 历史障碍仍按 [通用协议](../../docs/protocol.md#imdb-当前障碍) 及 [14号汇总](../../实验文档/14-离线实验统一对照与论文结论.md) 记录：支持SQL所需统计缺列，不能随机补特征或默认为所有不可提取行可删。

**状态：blocked_missing_authenticated_support_input**。解锁需要完整对应的真实IMDB统计/原表或可认证的finetune特征来源及版本链，再核对完整分母、失败记录、目标support/test身份隔离。若只做覆盖受限子集，需要用户明确批准`--allow-exclusions`及报告偏差，不自动降级。收到路径前不下载或猜测外部数据来源，不使用目标test作为支持。

## P2真实源表/SQL语义/真实选择率

- 本次可执行路径中未发现psql、pg_ctl；默认本地PostgreSQL socket未发现，PGHOST/PGDATABASE/DATABASE_URL仅检查存在性，均未配置。没有导出任何环境值。
- 与 [历史G1 inventory](../../实验文档/results/p2-source-semantics-20261005-03/g1_inventory.json) 一致，尚未认证同版本源表/统计历史来源链。工具和socket有限检查不证明全服务器没有数据库。
- [13号冻结协议](../../实验文档/13-P2源侧语义与真实选择率审计协议.md)仍要求G1通过及B独立/真实上下文认证后才能冻结C计数。当前没有安全的完整C输入，也没有SQL计数授权下的可执行计划。

**状态：blocked_snapshot_unverified / deferred_by_prior_decision**。解锁需要用户恢复P2、提供或授权定位对应chembl/legalacts/telstra源表的只读连接/快照，以及历史导入与统计来源链；连接信息不能入Git。不能用PG估计、join标签或合成小表计数替代真实过滤计数。

## 本轮完成的门槛工作

重新核对了可用输入并记录可观察解锁条件，未将输入门槛通过当作结果完成。离线主线、论文章节与工程验证继续推进，不因这两项阻塞停工。没有安装PostgreSQL、连接未知数据库、重建目标缓存或写入PRICE原项目。
