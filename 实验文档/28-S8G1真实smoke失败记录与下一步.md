# 28 S8 G1：真实 CPU smoke 失败记录与下一步

日期2026-10-07；执行协议[27号](27-S8G1运行器与真实smoke协议.md)，基础04561e8，分支feat/s8-g1-smoke-runner-20261007。本轮按用户要求实现并执行下一工程阶段，**整体结果为failed，不是completed_smoke，也不是research no_candidate**。保留失败产物，不自动重跑、不覆盖或追认通过。

## 1. 实际执行到哪里

服务器新隔离代码：`/gly/weilin/spatial-meta-ce/runs/s8-g1-execution-code-20261007`；新输出：`/gly/weilin/spatial-meta-ce/runs/s8-g1-smoke-20261007`。14文件执行包SHA与本地被测版本一致，前后未变；CPU最多4线程，CUDA_VISIBLE_DEVICES为空，1800秒/12GiB与外层OS timeout限制。旧PRICE环境/原项目源码及历史run未改。

开始前检测到冻结venv缺threadpoolctl，controller在建目录/训练前安全退出；这一失败不是F0执行失败。主代理不安装包，先只读确认加载OpenBLAS64/libgomp直接线程计数接口均4，再加入有界直接计数/来源检查、未知backend拒绝测试；最后595项完整测试通过后启动一次真正smoke。

真实任务外层56.483秒退出1，状态内部54.096秒failed；非超时/OOM/目录覆盖错误。F0完成后小矩阵probe数值证书失败，日志明确`Small-matrix smoke probe numerical verification failed`。本轮未生成summary、nomination、完整端点或metrics/verification；缺失文件不伪填。

[原始status](results/s8-g1-smoke-20261007/status.json)、[events](results/s8-g1-smoke-20261007/events.jsonl)、[执行日志](../reports/s8-g1-smoke-20261007/execution.log)、[controller与归档SHA](../reports/s8-g1-smoke-20261007/server-controller.json)。checkpoint/P矩阵留服务器，未下载/提交大产物。

## 2. 已完成阶段的有限独立核验

失败后主代理只读复核已有文件，没有训练F0/重启原任务。独立采样复算器从真实T source身份重算固定P和seed43抽样，183历史input及14源码before/afterSHA不变。

| 本轮真实F0工程事项 | 认证结果与边界 |
|---|---|
| 更新及episode | 原runner及checkpoint重载路径记录2次optimizer更新；独立重放8个episode/backward记录、全部sample来自P（不宣称partial verifier重验optimizer状态） |
| 两采样角色呈现 | support260、query512，共772；ERM合并监督，不是inner-loop |
| 实际曝光unique | 772个database+query身份；不等于19456P全量已训练 |
| 新checkpoint | SHA `596c615e7cc5b48bf38b2494964cc3c21ac53950310aaf16cc70700d49bccd25` 与reload事件一致 |
| checkpoint重载桥接 | 原runner实际固定T raw delta=0；后置独立验证核记录与字节，不冒称第二次模型推理 |
| H阶段 | events无nomination_sealed或H_semantic_load；未进入H语义阶段 |
| 整体smoke | **failed / formal_ready=false**，局部阶段验证不改此状态 |

[partial-stage-verification.json](../reports/s8-g1-smoke-20261007/partial-stage-verification.json)只证明已有P-only身份/来源/记录与checkpoint字节，不认证全部G1或完整模型真实性。optimizer_steps=2是固定契约/原runner路径记录，partial verifier未独立重载optimizer状态，不能冒称第三方认证；详见 [partial-stage证据注记](../reports/s8-g1-smoke-20261007/partial-stage-annotation.json)。P完整表示已提取并存矩阵；这不是probe正式19456训练、更不是Q/V/H研究评估。

## 3. 数值拒绝的具体来源

用已保存P-h/P-x/FP32label/pg_log矩阵只读重建同一1216拟合面板，独立NumPy数学式计算固定2步Adam，不调用torch optimizer/backward；同cutoff LS及证书逐字段复查，原matrix/last/status/contract字节SHA不变。**该诊断不是继续训练、第二次smoke或改原status。**

[solver-diagnosis.json](../reports/s8-g1-smoke-20261007/solver-diagnosis.json)显示：

| 字段 | h设计矩阵 | x设计矩阵（后置只读诊断，不是原run已执行） |
|---|---:|---:|
| 参数列数 | 65 | 775 |
| cutoff后rank | 46 | 586 |
| P零std特征数 | 18 | 177 |
| 奇异值metadata不匹配项 | 19 | 189 |
| normal residual检查 | 通过，约1.67e-16 | 通过，约5.92e-11 |
| 相对Adam目标检查 | 通过 | 通过 |
| 同cutoff参考J/预测检查 | 通过 | 通过 |
| singular spectrum metadata/reference检查 | **失败** | **失败** |

原证书比较`lstsq`和另一次`svd(compute_uv=False)`的全部奇异值，使用rtol1e-12、atol0；被丢弃的近零奇异值在不同分解路径有FP误差，约1e-14的绝对差却要满足约1e-27的相对限。例如h第46项：1.6430e-14 vs7.4813e-15；两者都远低于固定cutoff约1.6116e-10。**拒绝点是证书metadata条件，不是已证明LS拟合或源标签错误。**

真实h矩阵rank-limited不能推为表征信息一定不足，x后置诊断也不能推为新方法更好。诊断J是训练面板smoke数值，不进入正式Q95表、不作候选或空间漂移收益结论。所有full_linear_optimality标签仍False。

## 4. 为什么到此停止，以及下一可执行任务

失败分支按27号执行：保留原status/run/code，不把code修正后未来结果写回本次run，不临时改rcond、LR、训练步数或预测裁剪，不丢失败行，不自动重启训练。formal S8仍not_started，S6no_candidate不变。

下一工程任务应在新版本/新分支先完成：

1. **独立审查并事前冻结奇异谱数值认证规则**：区分保留方向与cutoff丢弃方向的数值尺度、rank一致性与边界。不能为迁就本次结果无依据放宽所有singular/normal/objective/预测检查；原rcond/科学gate不改。
2. 先用跨backend可复现的零/冗余/tiny/临界rank/伪造谱负例做TDD。明确截断方向的数值比较能认证什么，保留full-linear-optimality=False。
3. 如需验证新规则，可先只读认证本次已保存矩阵，另写新证明，不覆盖原failed。这不等于smoke整体成功，因后续fullP资源、seal/H和20352完整端点尚未执行。
4. 规则经独立审查且新版本工程测试通过后，再另获真实smoke执行指示，采用新目录单次重试；不复用failed状态的partial checkpoint当正式初始化，不对研究做配置搜索。
5. smoke全链通过后，才审定完整G2–G4/F0 4000/probe1000/Q/V/H runner和预算；不因2步smoke局部过就正式开训。

本轮已交付runner/state/checker代码、真实有限F0和失败诊断，**没有完成全部G1验收**，不报告GPU或正式精度。IMDB/P2缺认证输入保持原阻塞。最终提交/推送以实际Git核验记录；新代码、失败证据和最新待办一同可追溯。
