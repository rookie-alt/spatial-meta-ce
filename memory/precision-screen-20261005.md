# 精度提升首轮接手记忆（2026-10-05）

用户目标：提高精度，已明确使用离线数据；要求规划/记忆先落盘，然后新建任务分支操作。协议权威入口：[15-离线精度提升首轮实验计划](../实验文档/15-离线精度提升首轮实验计划.md)，进度仍以 [99](../实验文档/99-进度与下一步.md) 和真实 JSON/logs 为准。

- 第一阶段是四臂 ERM-sum：width64/128 × log SmoothL1/MSE，seed42，每臂4000更新，A本轮重跑；不先改变模型拓扑/采样/数据，也不启动FOMAML。
- 所有臂训练监控 inner_steps=0，端点 last.pt；保留全部监控/结果，不挑best。
- 23源训练库+chembl/legalacts/telstra验证；只读pretrain源缓存，不读目标数据。原PRICE-master及旧cache/checkpoint只读。
- 主指标4+表unique Q95三库等权算术macro；10%主指标改善且每库all/4+退化不超过10%才是单seed promising，非显著性/超PRICE结论。不改变历史P0/P1/P2协议或指标。
- 同宽度损失臂共享初始化；四臂共享训练采样、实际计数、验证计划与源哈希。真实推理前后计划和真值逐项核验。
- 12小时累计上限，顺序单空闲卡，检查CUDA_VISIBLE_DEVICES与UUID。2026-10-05规划时8卡均忙；必须新鲜查卡，不能凭旧记录抢占。若无空卡仍可完成代码和CPU测试，但正式GPU smoke/训练写blocked。
- 只授权本轮4臂seed42；seed43/44、少样本策略搜索、目标评估和数据库部署不自动执行。非有限数、OOM、来源/配对不符立即停，不改lr/batch/阈值重试。
- 先落盘协议/本记忆，再创建feat/precision-screen-20261005；本轮仅提交本记忆新增文件，不提交另一窗口未跟踪的AGENTS.md或其他memory文件。
- 模型支持mse和width配置，无需改损失/模型本体；新入口复用train/checkpoint/endpoint。先RED后GREEN，服务器只读PRICE环境，隔离现有graphify依赖；验证状态和启动状态分开。
