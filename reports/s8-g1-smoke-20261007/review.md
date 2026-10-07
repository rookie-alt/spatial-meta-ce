# S8 G1 runner、真实 CPU smoke 与资源验收

日期2026-10-07；基准04561e8；分支feat/s8-g1-smoke-runner-20261007；独立worktree `/private/tmp/spatial-meta-ce-s7-diagnosis-20261007`。用户授权按计划执行下一阶段，本轮冻结[27号协议](../../实验文档/27-S8G1运行器与真实smoke协议.md)：仅engineering smoke，不进行正式4000F0/1000Adam或完整Q/V/H研究评估。

## 分工、修改与边界

主代理仅新增run_s8_smoke/其测试、执行harness、27号、总进度和报告；写入隔离子代理新增标准库s8_state/测试及独立s8_independent_verify/测试。state作者完成后做单次只读端到端复审，未修改主runner。原model/train/data/checkpoints与S8contract/probe/preflight及24/25/26旧协议/结果逐字节核对04561e8，历史输入清单独立保存。原共享检出的AGENTS/MEMORY/委派草稿不触碰、不提交。

完整运行入口只CPU，固定roots、预算、smoke mode不允许formal4000；目标读拒绝。P-only候选hash顺序与F0原raw顺序抽样分别定义；实际batch入口监督数据只P。新checkpoint签名/step/optimizerstates/FP32/tensor/byteSHA，hash认证后trusted本次pickleload，无旧权重或resume。封存期望SHA从生成canonicalpayload字节独立锁定，H前及每Hget先require；seal工程不选择候选。

## TDD与本轮实际验证

- runner-red.log先缺模块；runner-green.log初始7项；新增实际SIGALRM及thread budget负例、最新runner逻辑均入最终suite。checkpoint-red.log验证缺optimizer和非FP32被放过后修复，包含momentshape/step/budget签名与overwrite拒绝。
- state-red至state-green.log：31项标准库测试过；不可回写seal、targets/source/mode/资源固定、双参考gate、ties/threshold边界、symlink/race/tamper、smoke忽略metrics。单测formalgate不是正式运行权限。
- independent-green.txt：23项、29.355秒；完整20352端点/52桶、独立P固定/8episode采样、FP64labels/PG/table identity/metrics及事件，稀疏有限性/溢出也拒绝。不导生产统计求解器，也不证明actualmodelfit或checkpoint，需要分立门槛。
- 首次targeted S8 111项31.746秒通过；全suite沙箱592项有两项既有helper进程启动失败，日志local-tests-sandbox.log保留。最终沙箱外595项80.529秒通过（local-tests-final.log），最后stage includes newly threadbackend guard。没有沿用旧531项或删除失败记录。
- runner超时缺陷独立子进程复现后修复：SIGALRM明确TimeoutError进failed，finally取消。新增子进程测试沙箱OpenMP SHM失败保留，沙箱外runner9项3.221秒通过，最终suite含10项runner。原生BLAS handler可能延迟，OS outer timeout/returncode始终另保留。
- 端到端只读审查认为timeout/线程门槛修复后未发现阻止受限CPU smoke的缺陷，核心P-only/checkpoint/seal-H/独立分母/resource系数隔离通过。详见[independent-review.md](independent-review.md)。

## 真实环境启动前阻塞及兼容处理

首次controller在写执行包/输出和训练之前发现冻结venv无threadpoolctl而安全停止。原server-controller.stderr/stdout.log保留为启动前失败，不算真实smoke失败、也不自动安装依赖。只读SSH实际测得NumPy1.26.1 OpenBLAS64 get_num_threads64_与Torch2.1.0+cu121 libgomp omp_get_max_threads均4，执行包/输出不存在；证据thread-backend-precheck.json。这里服务器后端事实为主代理实测，不冒称由review代理检查。

runtime_threadpools在缺inspect包时只接受冻结Linuxvenv已加载OpenBLAS64/libgomp直接C计数，保存库来源/接口和actualthreads；未知backend/origin/计数失败停止。对应合成TDD后通过，未改PRICEvenv或装包。环境三thread vars4在Pythonimport前设置，Torch两线程配置4，summary实际counter必须不超过4。这不是仅凭env推断。

## 真实 smoke 与失败归档

一次真实CPU工程任务已执行并failed：外层56.483秒/exit1，status内部54.096秒；14执行源码SHA unchanged，F0两更新/八episode实际完成、checkpoint reload事件maxdelta0。首个h small-matrix probe在metadata/参考singular比较被拒绝，未进入fullP资源LS、nomination/H或完整端点/metrics。原status失败和可用六文件smallarchive按server字节SHA认证保存，largecheckpoint/P矩阵留server，未下载大数据。

失败后两次有界只读核验有不同具体目标：一项定位numericalcertificate字段，用已存P矩阵与独立2步Adam数学式，无F0训练/H读取；另一项重放实际P-only8episode并认证已有checkpoint字节和183input/14codebeforeafterSHA。不重新执行smoke、不覆盖status或阈值。完整证据见solver-diagnosis.json与partial-stage-verification.json。

数值诊断：h rank46/65且18zero_std，x rank586/775且177zero_std；normal/objective/referenceJ/预测通过，singular metadata检查失败。rtol1e-12/atol0强制被截断近零值逐值一致，lstsq/SVD对约1e-14噪声给不同估计，要求约1e-27相对限。不是由此证明源标签或模型拟合错误，也不声称训练面板J带来正式精度提升。原run只执行h失败；x是后来只读数值诊断，不能将其写成原runnerx已完成。

局部认证：P-only曝光772unique，support采样260/query512，2optimizer/8backward；183input/14codeSHA一致；无H/nomination事件。checkpointdelta0为原runner实际记录，独立后验核字节/记录，不二次推理。整体仍failed/formal_readyfalse；下一步另审cutoff-aware数值规则并TDD，不在本轮自动修阈值或重跑新训练。

本轮真实CPU线程接口实际检查通过才到source/training，但失败前尚未写summary.runtime，所以不能拿缺失runtimefile证明本次峰值线程所有瞬间；外层导入前env4、代码当时的直接C计数failclosed与先前SSH实测四线程是不同证据。原源环境未装依赖/修改。

## 已知剩余门槛

G1通过也不能写S8研究完成或选型成功：训练仅2步、每探针2步/1216fit；V每256/H每32工程桥接与完整P矩阵资源profile都不是完整Q/V/H。profilefullP coefficient不用于predict或selection。事件顺序不等实际I/O的独立观察，actual入口gate及sourcebefore/after是另门槛。full-linear-optimality始终False；数值证书不是科学最优证明。

下一阶段需要审查正式固定4000F0/1000Adam、完整Q/V/H单次报告与提名接受/否决、独立full端点认证，以及明确资源budget/启动授权，不自动扩seed/改loss/rcond。IMDB/P2来源阻塞不变，S6no_candidate不变。本轮不graphify、不数据库count、不目标评估，不以smoke推论尾部改善。


## 本轮最终状态与自检

已实现签名/seal/H入口/独立endpoints模块，595项完整回归通过、改动文件Ruff通过；真实smoke整体失败，数值规则独立冻结/TDD及未来新目录真实smoke仍待执行。27协议原字节和失败run/code保留，新增28号失败定位与nexttask；更新00/23/99，明确局部证据与整体状态，不graphify。原model/train/旧core/24/25/26与历史results严格核原04561e8byte。

自检均4/5：准确性（失败完整披露、SHA/partial认证，不推精度/因果）；完整性（实现并执行用户下一G1，但未全链通过，阻断数值认证不是隐藏）；清晰性（28/36入口局部vs整体）；可操作性（有新runner与证据、下一规则需事前冻结，不能直接formal）；简洁性（摘要独立，日志保留）。平均4.0。后续改善重点是独立认证近零奇异值元数据规则，不能拿自评分或单次smoke failure替代方法结论。提交/推送在实际成功后补记。


## 失败只读复核与证据范围补记

独立诊断复核接受actualmetadata拒绝定位、六原文件与14源码绑定；不接受将partial verifier的optimizer_steps硬编码2外推为独立重验optimizerstate。已以partial-stage-annotation.json明确这一范围，原proof/stdout/failedstatus保持不改。原runner确实完成两更新及其checkpoint检查，但后置只读验证仅重放8个P-onlyepisode与现存checkpointbytes/记录，不第二次推理。H未semanticload不等全部H字节未读（metadata身份/count/SHA允许）。

下一版本数值规则需独立冻结、考虑FP64epsilon/维度/尺度和cutoff间隔，nearbound仍unknown、不修改科学门槛；本轮不修证书容差、不重启真实run。整体实验工程失败与本地code/tests通过同时记录，不以595tests掩盖真实failed。
