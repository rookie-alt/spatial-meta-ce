# 官方 26/4 模块训练准备记录

日期：2026-11-19。**全部工程阻塞已关闭，最终独立复审passed，真实输入和训练文件准备完成。** 未启动研究训练或四库模型预测。最终状态以[交付记录](DELIVERY.md)为准，范围见[132号](../../实验文档/132-模块候选官方26训练4评估准备.md)；下文阶段性“进行中”为历史记录。

## 已推送

三个交付分支均已推送 Gitee 和 GitHub，并逐项核对远端 SHA：共同交付 `2d1f76637c5b97ca07c42344084ef5b11a9f259e`、分组 `2f9c55c570a652d9a47112b2de1fa8284ff63003`、关系 `2a921f9024bce9a87dce1ba30cf3eb89ed6187ed`。本轮准备分支从共同交付版本派生，原候选分支保留。

## 新入口与本地验证

- `module_experiment.py` 固定 26/4、五臂、三个 seed、24k、last.pt、无验证选择；启动另需 spec/auth/output/budget/hash 绑定。
- `run_official_module_experiment.py` 提供 prepare/verify/run/worker/evaluate-ready。评估准备须先封存 15 个 checkpoint 与父进程预算完成证据。
- 官方 mapping 使用独立 scope，pretrain 仅 26 库、test 仅四库；旧 source-only 接口未放宽。
- 最终全量 **1928 项，OK（16 skipped），rc=0**，before/after SHA 相同：[日志](conversion-final-full-unittest.log)、[记录](conversion-final-full-unittest.json)。先前各轮记录均为对应历史版本，不代替最终版本。struct-final 中间运行在源码变动期间出现一项旧 supervisor 信号时序失败，单项复跑通过；保留原记录，不记为最终验证。
- 输入专项 25 项、并行准备 3 项通过；并行真实 spawn 与串行 sidecar 字节一致。服务器 Linux 模块回归 **33 项通过**，含五臂各两步合成 train→checkpoint，见 [日志](linux-module-tests-v2.log)。首次服务器测试缺 configs 文件，补齐部署后通过，源码未改。新增 NaN/±Inf 失败路径在服务器也通过，见 [日志](linux-struct-values.log)。
- Ruff 0.11.2 全仓、0.17.5 改动 F/E（忽略 E501）通过；0.17.5 全规则 668 条（基础 624 条及新增风格建议），未声称全规则通过，见 [记录](lint-final.json)。
- 四项修复包括 train 公共入口实际设备/own supervisor、全部准备和执行路径只读树保护、新 runner owned RSS + mandatory CUDA 遥测、optional sidecar 缺失状态绑定。Linux 真实短时子进程验证正常合成遥测和缺失停止通过，[记录](linux-supervision-result.json)；未使用真实 CUDA。

## 输入认证与来源

隔离目录：旧生产 `/gly/weilin/module26-prep-20261118/code`、当前 `/gly/weilin/module26-prep-20261118/code-ready`；原 PRICE 和缓存只读。认证禁 CUDA、RSS 16 GiB、墙钟 5400 秒、输出 3 GiB，使用现有 supervisor，超限停止不自动扩预算。

首次 inputs 因旧清单缺后来追加的可选 struct_log.npy/struct_manifest.json 而拒绝，已停止自己的监督进程并保留记录。新 inputs-v2 对当前全部文件绑定，仅允许旧清单缺这两个可选后加文件；六个基础数组、manifest、workload、特征和统计要求不变。测试覆盖可选 sidecar 之后漂移仍拒绝。

串行 inputs-v2 实测 3473 秒只完成 11 个源库，预估超出固定预算，核对自身 parent/PID 后停止并保留 [监督记录](input-certification-v2-supervisor.json)，未认证通过。新 `inputs-parallel-v3` 复用逐库算法，在相同墙钟/RSS/输出预算下使用最多 4 个单线程 CPU 进程，已完整完成。

真实 [输入报告](input-certification-v3-report.json) 与 [监督记录](input-certification-v3-supervisor.json)：26 训练库共 **1,300,000 行**、四库共 **7,239 行**，unsupported=0、dedup_conflict=0、失败库为空，完整分母认证；test 四库逐行 frozen extractor 重建。监督 `preflight_completed`，墙钟 **2237.5 秒**、峰值 RSS **10.64 GB**。本过程仅认证结构/特征，不执行模型，不证明真实 SQL 选择率语义。

后续struct非有限值及消费者转换缺口已修复：限定实数dtype并检查float64/float32转换后有限；准备/复验共同执行。真实[30库数组](real-struct-values-audit.json)均float64且有限。原code-ready冻结，最终code-training独立部署；[最终test4复验](conversion-test-input-verification.json)通过，原报告SHA不变；pretrain26和[第二版spec准备](training-ready-v2/readiness.json)全部完成，15次未签执行拒绝、首份完整dry-run通过。

旧构建器 SHA `304601dba325a65743f1b5d3f49099602db1877ed982331236efc730daee8dc4` 来自 Git fa9770e，已保存 [旧源码](historical_mapping_builder.py.txt)。当前格式清理后的字节不同，运行时绑定旧/当前 SHA、AST 等价，且逐条重算关系和布局对比；语义改变拒绝，不执行旧源码或修改旧 sidecar。

官方 plans-24k 的三个训练计划各 96,000 episodes、约 895 MB，五臂按 seed 共用。运行器流式认证 episode，缓存复用前仍完整核对文件 SHA，不以 mtime 替代内容。

三份真实计划已逐条认证通过：[结果](real-plan-validation.json)、[监督记录](real-plan-supervisor.json)。seed42/43/44 的 episode_count 均 96,000；source cache/label/features、capacity、sample IDs/hash、support/query 互斥及 canonical plan digest 均通过，code before/after 一致，监督状态 `preflight_completed`，墙钟 274.5 秒、峰值 RSS 1.11 GB。没有构造模型或访问目标库。

生产字节与当前verifier分别绑定，两个固定快照及原extractor/builder路径字节均核对，原报告不改写；除verifier及新增guard定义/唯一调用外，生产AST严格一致。spec的37文件闭包同时绑定快照与当前代码。[最终复审清单](review-conversion-input.json)24文件10证据SHA一致，结论**passed，无未解决阻塞**，详见[交付记录](DELIVERY.md)。

## 预算草案和启动条件

[training-budget-draft.json](training-budget-draft.json) 是单 arm/seed 的**上限建议，尚未批准**：墙钟 30,000 秒、RSS 24 GiB、VRAM 2 GiB、输出 2 GiB、GPU 分配秒 24,000、单 run 18,000 秒。若 15 个 run 均用此上限，总 GPU 上限 100 小时、墙钟上限合计 450,000 秒；这是最坏上限加总，不是预计耗时。

历史 A1-PG 24k 实测墙钟约 6,267–6,699 秒、GPU 秒 6,206–6,642、RSS 约 8.1 GB，只是历史参考；新模块成本未测。完整输入认证与独立复审通过后，才冻结最终可执行 spec。GPU0 Unknown Error 的全局查询依赖已从新入口移除，旧 supervisor 保留；[GPU1–7 盘点](gpu-inventory-preparation.txt) 均可查询且当时空闲，但启动前须刷新状态。

当前授权为推送与准备。训练/评估需另行明确预算与执行授权，模板 authorized=false、批准人/时间为空；不代签 P0。四库只做固定终点评估、selection_use=none，unique-query 主口径和 all-rows 辅助口径分别报告。
