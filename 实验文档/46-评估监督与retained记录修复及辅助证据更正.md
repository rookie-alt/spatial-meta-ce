# 评估、进程监督与 retained 记录验收修复及辅助证据更正

日期：2026-10-08（Asia/Shanghai）。本任务 worktree：`/private/tmp/spatial-meta-ce-s8-review-fixes-20261008`；分支 `audit/s8-review-fixes-20261008`；基线 `b3390b7463d0dd0fc002903f98214b68c696a11b`。

## 1. 结论与授权边界

**三项本地修复已实施；阶段 A 的真实源侧辅助统计更正仍 blocked，不能宣称阶段 A 整体通过或恢复正式实验。** 最终测试/独立审查/提交与同步状态见第5、8节及本任务证据；正文不以旧轮次测试数字替代当前结果。

当前建议：**暂停 S8 开跑，优先论文收尾**。本轮恢复的是工程与辅助指标的正确计算/验收，不是模型精度提高。[47号同h两臂草案](47-S8同h两臂窄实验协议待确认草案.md)仅为未批准、未执行的条件备选：它有可证伪的窄问题，但当前输入、真实Linux和数值前置验收未闭合，不值得自动投入下一轮训练。

没有连接服务器、GPU、真实smoke、真实P重跑、新F0/研究Adam训练、重新推理或目标评估。完整unittest包含合成计算及既有归档摘要一致性检查（包括历史论文摘要fixture），不是重新读取真实目标workload/缓存或开展目标评估；其结果不参与本轮选型。原 S8 worktree与主目录另一窗口未提交修改不动，不合并main。

历史状态永久保留：收紧后单次 **A failed**、原 **G1 failed**、`formal_ready=false`、`strict_certificate=unknown`、`full_linear_optimality_verified=false`、`can_retroactively_pass_original_G1=false`。真实环境与服务器数据未在本轮重新核验。

## 2. A1：配对指标从预测值大小改为真实误差

变更：[evaluate.py](../spatial_meta_ce/evaluate.py)、[paired协议测试](../tests/test_paired.py)、[配对数值/身份测试](../tests/test_paired_metrics.py)、[eval_frozen消费者](../scripts/eval_frozen.py)、[PRICE消费者](../scripts/price_fewshot.py)。

旧 `abs(reference_log_prediction)-abs(model_log_prediction)` 没有使用真值，量不是Q-error改善。复现包含真实数值断言失败，不能仅将新接口TypeError当RED；详见[a1原RED](../reports/s8-review-fixes-20261008/a1-red.log)及[a1分项记录](../reports/s8-review-fixes-20261008/a1-report.md)。

新口径与既有 `q_values` 完全相同：

```text
log_true = log(max(true,1))
log_q_model = abs(max(model_log_prediction,0) - log_true)
log_q_reference = abs(max(reference_log_prediction,0) - log_true)
log_reduction = log_q_reference - log_q_model
paired_delta_q95 = percentile95(exp(log_q_model)) - percentile95(exp(log_q_reference))
```

- mean/median reduction 单位为natural-log Q-error，正数为改善。
- fraction按两侧真实误差比较。仅机器精度tie：`abs(log_reduction)<=4*eps*max(abs(clamped_model_log),abs(clamped_reference_log),abs(log_true))`，处理对数相减舍入；不是统计显著性门槛，不裁剪raw reduction、不改Q95。
- `paired_delta_q95` 保留字段名，但现在确为**普通Q-error单位的 model Q95 − reference Q95**，负数为改善；既不是log-Q95差，也不是逐query误差差值的Q95。不能与旧字段数值直接比较。
- all-rows与unique-query都传入同一原始真值、有序 `(query_id,query_hash)`；原始标签比较发生在clamp之前。长度、维度、顺序、身份、标签、非有限、Q溢出及冲突重复均拒绝。
- CSV reference loader按id+hash绑定，允许完全相同K0重复，拒绝冲突；要求显式 `metadata.method='pg'`，并输出 `reference_comparison='pg_baseline_not_same_model_k0'`。这不是同模型K0适应收益。两个消费者只把标签改成 `improved_vs_pg=`，未改默认历史路径、模型或训练。
- `metrics.py`、主 `summarize`、unique去重/主聚合公式不改。无效reference以前可能产出错误辅助值，现在fail-closed；不能将“有效相同输入下主指标不变”误写成“所有旧无效输入仍会执行成功”。

独立合成复算使用Decimal基数比、Python math/statistics和手写普通Q95线性插值，**不调用生产指标函数生成期望**。true100/PG10/model100应为Q10→1、log reduction=ln10、Q95差=-9；true100/model125/PG80与反向都为Q1.25且fraction中性。独立审查发现后者未被初始4eps相对log-Q tie覆盖，先保存[有效RED](../reports/s8-review-fixes-20261008/a1-near-equality-red.txt)再修；误拼测试class的[命令错误](../reports/s8-review-fixes-20261008/a1-near-equality-command-error.txt)不是有效RED。既有1e-10真实改善测试仍保留。

独立审查还发现生成报告仍声明旧tie规则；已补[a1-definition-red](../reports/s8-review-fixes-20261008/a1-definition-red.txt)并同步fraction定义，不仅改注释。最终[专项GREEN](../reports/s8-review-fixes-20261008/a1-definition-green-final.txt)30项通过；[独立复算JSON](../reports/s8-review-fixes-20261008/a1-recompute-after-review.json)、[同次日志](../reports/s8-review-fixes-20261008/a1-recompute-after-review.log)、[绑定](../reports/s8-review-fixes-20261008/a1-recompute-after-review-binding.json)9/9通过、JSON/日志逐字节相同、源码SHA当前。早期 `a1-recompute.log/json` 仅属中间版本，不能与最终结果拼成同次证据。

## 3. A2：有界退出确认与有界失败清理

变更：[readonly-P监督入口](../scripts/verify_s8_readonly_p.py)、[退出/清理测试](../tests/test_s8_supervisor_exit.py)。不重新执行[45号真实P](45-S8G1收紧后单次真实P执行失败记录与暂停建议.md)。

- `/proc`缺VmRSS/VmSize、消失窗口或其他采样异常：只在 `min(0.15s,剩余wall预算)` 内 `wait(timeout=...)` 确认并回收退出。退出码非零、signal或缺worker proof仍失败；存活不可观测不忽略，确认超时fail-closed。
- 保留RSS stop、AS budget、wall deadline及异常退出检查。完成退出确认后若实际总wall已越界，仍记wall_timeout，不能用grace延期。
- 总是向自有session/process group发SIGKILL，即使leader先退出，避免留下后代。ProcessLookupError只表示group已不存在，不把其他清理异常视为成功。
- 独立审查发现初版finally仍有旧无界wait；已补[禁止无timeout等待及永久回收超时RED](../reports/s8-review-fixes-20261008/a2-cleanup-red.txt)，清理回收现最多1秒，不扩worker预算。kill/wait异常、未回收状态仍写失败证据并恢复信号处理器，returncode可为null，`child_reaped=false`不冒充正常。
- [最终专项](../reports/s8-review-fixes-20261008/a2-cleanup-green.txt)12项，10通过、2 skip。覆盖退出窗口、持续不可观测、非零/缺proof、deadline、消失group、永久reap超时、kill异常/证据/处理器恢复。既有RSS/AS/timeout/SIGTERM回归也在完整suite中。

**尚未验证：** 本机macOS，真实Linux `/proc`快速退出、leader已退出而后代仍存活的清理集成均skip；加上13个旧Linux握手测试，共15个Linux相关skip。mock通过不证明真实Linux全过程；需要新的独立Linux验证授权，不能据此追认历史执行或重跑真实P。

## 4. A3：retained 数值记录不再只信布尔值

变更：[独立smoke验收器](../scripts/s8_independent_verify.py)、[readonly-P验收器](../scripts/verify_s8_readonly_p.py)、[手写v2 fixture](../tests/test_s8_independent_v2.py)、[记录负例](../tests/test_s8_retained_record_acceptance.py)。未改solver、RCOND、rank定义或任何原数值门槛。

两验收入口分别使用字面冻结常数，核对check/bound的存在与scope；拒绝bool/string/null、NaN/Inf/溢出、负残差/容差及非法范围；校验eps=2^-52、accumulation=8、sigma1/sigma_r/cutoff、kappa、scale、floor、reachable与tol关系，以及既有SVD cutoff interval。rank0要求零谱/零projection分支；两侧共享spectrum/projection/target必须一致。

残差接受使用**重新算出的冻结tol**，不使用可损坏存储tol；即使存储tol略增仍在元数据closeness内，也不能放宽残差界。aggregate/单文件canonical一致性仍核对，但两份同时损坏并相等也会因内部公式/范围关系拒绝。

[A3原RED](../reports/s8-review-fixes-20261008/a3-red.txt)确实包含同步损坏未拒绝的反例；最初测试导入意外带入旧fixture，且日志混有OMP环境失败，**不能把226个failure统称226个独立新增缺陷**。后改为模块fixture调用。[最终专项](../reports/s8-review-fixes-20261008/a3-green-final.txt)9项通过；[v2回归](../reports/s8-review-fixes-20261008/a3-v2-regression.txt)57项通过。

**认证限度：** 元数据算术一致≠实际矩阵/向量认证。实际sigma extrema、projection/target scale、retained residual、true rank和strict optimality保持未认证；独立返回明确 `uncertified_quantities`。完全同步且算术自洽的伪记录不能由摘要恢复原矩阵真伪。未扩大读取范围去补这些量。

[独立归档算术脚本](../reports/s8-review-fixes-20261008/recheck_retained_archive.py)只打开45号已归档JSON，不导入solver或生产validator；[结果](../reports/s8-review-fixes-20261008/retained-archive-recheck.json)四面板公式/aggregate/单文件一致，所有原始文件SHA绑定；实际量仍未认证，原顶层status仍failed。这不是P重跑，也不是新的G1证书。

## 5. 本轮真实测试与静态检查

实际环境：[local-runtime.json](../reports/s8-review-fixes-20261008/local-runtime.json)：`/opt/anaconda3/bin/python`、Python3.11.7、macOS、NumPy1.26.4、Torch2.3.0、SQLGlot18.17.0。与旧真实Linux环境不同，不能代替Linux认证。

- 最初readonly-P回归69项有3个失败，包含沙箱OpenMP SHM错误与受影响子进程；按权限规则在沙箱外重跑相同本地fixture，69项通过。两份日志都保留。
- 初次完整810项，1失败/14skip，真实原因是旧合成SIGALRM测试在触发alarm前被默认BLAS/OpenMP线程超4拒绝。没有放宽gate，显式把OMP/OPENBLAS/MKL/VECLIB/NUMEXPR线程设4，实际runtime两threadpool均4。
- 初轮4线程完整817项通过/14skip；A2清理P1修后完整820项通过/15skip。两者是中间版本，不能替代A1后续等误差修复的最终回归。
- **最终全量：822项/88.019s，OK，15项Linux相关skip，exit0**，见[原始日志](../reports/s8-review-fixes-20261008/unittest-final-definition.txt)及[源码/环境绑定](../reports/s8-review-fixes-20261008/unittest-final-definition-binding.json)。12份源码/测试运行前后SHA一致，且与当前文件一致；这是本地工程回归，不是Linux或正式实验通过。
- Ruff0.16.10全仓库仍有**110项既有告警**，与b3390b7按file/code/message计数完全相同，新增0，未批量清理；本任务修改/新增代码范围Ruff通过，见[差异JSON](../reports/s8-review-fixes-20261008/ruff-comparison.json)、[作用域输出](../reports/s8-review-fixes-20261008/ruff-final-scope.txt)。提交前再次核对最终范围。原始unittest/Ruff日志含工具输出行尾空格，暂存全量diff检查会报告；按证据原字节保留，不修改原日志。代码/测试/JSON/Markdown排除原始日志后暂存空白检查通过，见[登记](../reports/s8-review-fixes-20261008/staged-whitespace-check.json)。

## 6. 源侧辅助更正：输入缺失，不制造更正数字

只有已有冻结逐query预测、同一真实标签、PG与查询身份、原SHA/完整分母才能做有效复算。源输入盘点结论为**原始冻结预测复算blocked**；不会连接服务器、补跑、重新推理或读取目标数据取得缺输入。详见[输入盘点与更正阻塞](../reports/s8-review-fixes-20261008/auxiliary-correction-status.json)。当前及旧S8 worktree的六份S4/S5/S6 `endpoint_rows.jsonl`不在本地，三个检查根的`artifacts/cache/pretrain`实体也缺失；26份P1 source manifest的bundle SHA为`3d708226437e105c88562970fdd0b2ccf711a833105b4ec8c66d899aa61f79a6`，只认证记录绑定。当前远端文件存在性未检查、保持unknown；不能用summary或历史passed代替原始行。

需要撤回/更正的口径：

| 历史量/结论 | 本轮处理 |
|---|---|
| 由旧 `_pairwise_metrics` 输出的mean/median log-Q reduction | 不能解释为误差改善，撤回该含义；真实新数值待原始输入 |
| 该函数的fraction improved/worsened | 基于预测绝对log大小，不能支持“改善了多少查询”或适应有效；不靠旧汇总补出新fraction |
| 旧 `paired_delta_q95` | 不是真实Q-error Q95差；旧数值与新普通Q单位断开，不改个名字后继续沿用结论 |
| 相对PG的改善写成适应收益 | 撤回；适应收益必须另做同模型K0配对，本轮未做 |
| 其他source runner独立生成的诊断fraction/raw prediction shift | 不因字段相似就一律归于该缺陷；须逐消费者追溯。没有原始输入，不宣称新的逐query再认证 |

### 主 Q95 是否不变：三层证据要区分

1. **代码/合成：** `metrics.py`与主公式未改；all-rows/unique主计算及独立分位数测试保持，配对修复没有改变预测或标签。这不是新模型收益。
2. **已归档源结果/聚合：** [独立保留脚本](../reports/s8-review-fixes-20261008/preserve_source_summaries.py)只读S4/S5/S6源摘要，与基线git blob SHA逐字节比对，并用stdlib `math.fsum`重聚合三库既存Q95；[证据](../reports/s8-review-fixes-20261008/source-main-preservation.json)三份摘要未变，逐库all/4+Q95、每库分母、完整unique149504、identity/PG digest/原endpoint rows SHA均保留，算术macro一致。没有重新排名/选择候选。
3. **真实逐query再计算：blocked。** 摘要字节不变和算术一致不能重新认证原始预测/标签Q95。因此可说“原冻结主结论与归档数值未被本修复改变”，不能说“本轮以原始预测重新算证全部主Q95不变”。

原归档S6 A/B 4+ macro仍为44.14432045246672 / 39.98662687913146，原no_candidate及逐库保护失败不改。这些是历史源摘要，不是本轮新运行结果；辅助更正缺输入不授权重选checkpoint或候选。

## 7. 下一步建议与未知

暂停开跑、论文收尾的依据：S1–S6没有通过原门槛的源候选，S7仅描述性且优化/表征原因unknown；本轮修复不能提供新的拟合差距或留出迁移证据。相反，真实Linux、h-full rank stability、原矩阵数值独立认证和冻结预测复算仍有前置成本/输入缺口。F0与全量h提取仍占主要成本，删x不是免费实验，也不能凭“数值参考MSE更低”期待主精度改善。

条件草案[47号](47-S8同h两臂窄实验协议待确认草案.md)给出同h、同样本/标签/MSE两臂，T-only新F0与P/Q/V/H隔离、固定solver/cutoff/步数、相对10%加绝对1e-4拟合差距、10%Q95改善加5%逐库保护；H历史已见、V反复开发、seed不增加数据库数明确披露。统一计算预算G0 30min+G1-A 15min+唯一G1-B tiny15min+formal4h，总上限5h，人工审查另记，均为上限建议非ETA。

这些规则需用户在任何新数据揭示前确认。无实质差距即停；有差距却V/H不过则完整归档有效负结果停；来源/数值/监督失败只能failed/blocked/unknown；不因接近阈值扩LR/solver/层级/步数/seed。历史formal_ready=false不改名绕过。只有合格候选才另提独立seed/数据库隔离确认和adaptation-vs-K0、FOMAML-vs匹配ERM；本轮均未执行。

## 8. 提交、同步与接手

本报告、47号草案及[进度第53节](99-进度与下一步.md#53-独立任务审查缺陷修复与下一步草案2026-10-08)只写本任务worktree，不覆盖其他窗口。

独立审查已闭合本轮代码发现：Plato的A1批准与数值/报告定义/证据复审见[审查归档](../reports/s8-review-fixes-20261008/independent-review-plato.md)；James的A2原P1关闭、A3记录式验收和source输入盘点见[审查归档](../reports/s8-review-fixes-20261008/independent-review-james.md)。这是限定本地代码审查，不是阶段A整体/真实Linux/正式实验批准。

本轮最终源码/日志/文档绑定、实际commit/push状态记录在 `reports/s8-review-fixes-20261008/` 的交付状态；提交与推送是独立状态，不能把本地生成或测试通过当作已提交/已推送。不合并main，且不因交付草案自动开始实验。
