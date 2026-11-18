# 126独立审查与验证记录

日期2026-11-18，worktree `price-meta126-targets-20261118`，base `947addb`。审查者`/root/review_target126`未参与实现；主代理负责runtime/监督接口，两个实现代理分别负责计划和聚合。审查全程只读，不直接SSH/GPU或读取服务器目标文件。

## 方案闭环

审查确认四目标授权、125封存、分母、支持池及矩阵。PG同population现场复算、IMDB缺支持证据和STATS扩展兼容实证均已落实。batch动态宽度等于最终padding的疑虑，经PriceAdapter固定self.counts padding/超长拒绝核实不成立；7权重×2603test最大差3.8147e-6、state一致、长support完整。3h总限与120秒单格限同时为硬限，非每格耗时预测；超任何限invalid停止，未扩大预算。

## 实际代码审查

覆盖runtime/plans/results、三个tests、CLI、监督器126小diff、协议及输入证据。审查者实跑43项126专项，4.811秒OK；复算38项closure为0missing/0mismatch，plan/binding digest一致，协议SHA `50953e60708faa76b893391b0198988a09c2e45ca5f7957f0dab22d507c75bbf`一致。结论：**通过，无未解决阻塞，可执行**。

核对fast weight每格重置、functional参数不写回、state_sha前后比对；125代码/binding/seal/六last/训练状态SHA；原生标签与独立解码；double PG同查询身份；IMDB实际缺支持且禁止静默增fewshot；exact242格、先block平均再三库log均值及三流配对；四库K0另列；SIGALRM120秒/监督3h、UUID整卡限额、白名单与软链拒绝。

主代理最终冻结本地全量1929项、344.257秒、OK(22skip)、exit0；首轮1928项OK(22skip)日志也保留。Linux172项OK无skip、Ruff/format/diff通过。代码审查不替代真实GPU或科学成绩。

## 实际效果证据审查

完成242格后，审查者直接从units手算，不调用生产aggregate/metrics：M20/E20−12.427438%（2/3流）、M20/P20−1.702541%（1/3）、M20/M0+7.360386%（1/3）、M20/P0+32.469146%（0/3）、M20/E0−10.690318%；四库M0/P0+43.933173%，负迁移21/45，全部一致。

各格population为target_test_unique，分母521/2603/1258/1699；Σn=437848=48648(K0)+194600(K5)+194600(K20)。242预测文件及七权重SHA齐备，partial_target_signal、四门false独立推导一致，无all-rows/pooled/三库四库混用。监督completed/exit0、560.078秒、VRAM773849088B、RSS1168842752B、输出28847597B、最大单格约3.02秒，均合规。结论：**效果证据通过，无口径夸大**。

解释须限定：M对E收益集中Genome（逐库均值比−30.44%），另两库近持平；对P20仅1.70%且仅1/3流改善，不称稳定超过官方；适应自身总体更差，四库M0均值均劣于P0。P只有一个公共预训练模型，不把三流或五支持组称独立预训练复现。

审查者未SSH读242份CSV，逐行验证由主代理独立脚本在服务器运行并保存independent-verification.json；审查者核对其逻辑、文件级SHA/格级指标及行数加总。限制不扩写为独立重跑GPU或形式证明无缺陷。
