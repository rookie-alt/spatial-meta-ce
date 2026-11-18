# 官方 26/4 训练准备交付

2026-11-19：工程阻塞已关闭，最终独立复审 passed，真实输入、训练文件和 dry-run 均完成。未启动研究训练或四库模型预测。分支 `feat/official-module-training-prep-20261118`，共同基础 `2d1f766`。

## 固定矩阵

官方26库训练，`validation=[]`；`imdb/stats/genome/ergastf1` 为固定K0终点评估集，不用于选择。五臂 A1-PG、GroupMean、GroupAttention、RelationTrue、RelationShuffled，三个seed42/43/44，共15次。从头初始化，每次24,000 updates，同seed共用sealed plan，只取last.pt，两个模块不叠加。

## 验证与审查

- [最终全量](conversion-final-full-unittest.json)：**1928 tests，OK（16 skipped），rc=0**，284份源码before/after SHA一致。输入专项25项；NaN/±Inf、timedelta64、longdouble超范围、float32溢出、complex/string反例覆盖准备、历史报告复验和直接加载。
- Ruff0.11.2全仓通过；0.17.5改动F/E（忽略E501）通过，全规则668条告警，未声称通过。中间失败/源码变化记录保留历史含义。
- 源码/文档暂存空白检查通过；原始日志中5行尾随空格保留采集字节，故包含日志的完整diff-check报告这些空白，不修改审查已绑定的日志SHA。
- [真实输入](input-certification-v3-report.json)、[监督](input-certification-v3-supervisor.json)：26训练库 **1,300,000行**，四库 **7,239行**，unsupported/dedup conflict/失败库均0；test全行重建。CPU4进程，墙钟2237.5秒、峰值RSS10.64GB、VRAM0。只认证结构和冻结特征编码，不外推真实SQL语义或精度。
- [三份真实计划](real-plan-validation.json) 各96,000 episodes，source cache/label/features、容量、ID/hash、support/query互斥及canonical digest通过；[监督正常](real-plan-supervisor.json)。
- [最终四库输入复验](conversion-test-input-verification.json) 通过，原报告SHA不变；真实 [30库可选数组](real-struct-values-audit.json) 均float64且有限。生产目录冻结，训练代码独立部署。
- [最终训练准备](training-ready-v2/readiness.json)、[监督](spec-preparation-v2-supervisor.json)：公共训练输入和三份计划完整验证，15个spec＋15个未签模板，15次unsigned execute全部拒绝、执行输出未创建，首份完整dry-run通过。墙钟637.2秒、峰值RSS1.35GB、VRAM0。全部spec的37文件闭包与本地最终代码、测试版本及[服务器部署](final-v2-deployment-check.log)一致；本地30份文件SHA与readiness逐项相同。
- [Linux合成监督](linux-supervision-result.json)：真实CPU子进程、合成CUDA counters，正常预检和缺失遥测停止均通过，全局GPU查询0次；非真实CUDA成本测量。服务器模块[33项通过](linux-module-tests-v2.log)，转换反例[通过](linux-struct-conversion.log)。

原四项审查问题（实际设备/自身supervisor、只读树、全局GPU查询、可选文件清单）均已独立关闭。新增struct非有限值及消费者转换缺口已复现修复：guard只收实数整数/浮点dtype，float64/float32转换后均须有限。最终[窄范围复审](t3-thread://v2/64920c42-55dd-48e3-bc10-fb310e06e3e4/thread:delegated-task:command%3Amcp%3Ade9c5909-c833-419e-aed3-8e3fca649d55%3Adelegate-task%3A70c33b91-73e4-44ff-92a9-584583870f5e) **passed，无未解决阻塞**；[清单](review-conversion-input.json)24文件10证据开始/结束SHA相同。独立核验8类非法值拒绝，12种合法dtype、3个有限边界和None通过；producer AST、快照及37文件闭包通过。真实数据/spec证据由父任务另验。

原生产报告和sidecar不改写；两个固定producer快照与当前verifier分别绑定，除verifier及新增guard定义/唯一调用外，生产AST严格一致。原extractor/builder路径字节与当前绑定同时核对。历史inputs-v1/v2、首版training-ready和初始部署清单保留；当前使用 **training-ready-v2**、**final-conversion-code-closure.json**。

## 文件、预算与启动条件

- 代码：`/gly/weilin/module26-prep-20261118/code-training`。
- 输入：`/gly/weilin/module26-prep-20261118/inputs-parallel-v3`。
- 最终spec/未签模板：`/gly/weilin/module26-prep-20261118/training-ready-v2/<arm>-seed<seed>/`。
- 预定输出：`/gly/weilin/module26-prep-20261118/outputs/<arm>-seed<seed>`，本轮未创建。

[每run上限](training-budget-draft.json)：墙钟30,000秒、单run18,000秒、CPU30,000 core-seconds、GPU24,000 allocated-seconds、RSS24GiB、VRAM2GiB、输出2GiB。15次GPU累计上限100小时、墙钟上限加总125小时，均非预计耗时。当前spec绑定GPU1按run串行；改多卡须重新准备设备/输出/授权绑定，启动前刷新设备状态。

只读核验示例：

```bash
cd /gly/weilin/module26-prep-20261118/code-training
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' \
  /gly/weilin/PRICE-master/venv/bin/python -B \
  scripts/run_official_module_experiment.py verify \
  --spec /gly/weilin/module26-prep-20261118/training-ready-v2/A1-PG-seed42/training_spec.json \
  --spec-sha256 2505a409f72d3223cfa3b74074ade8d928e696c51cd8299e6e885430af85e221
```

本轮授权为准备和推送。模板authorized=false、批准人/时间为空；真实启动仍需用户确认具体预算后签发，旧授权不可复用，不代签P0。四库模型评估有独立授权及全部15份checkpoint/父进程预算认证门禁。不快进已分叉且由其他任务占用的main。
