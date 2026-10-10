# Meta65适应算子runner实施与CPU预检

日期：2026-10-10。基准 `55c777e382fda863a5d42078ff49d9edf476fc3b`；独立worktree `/private/tmp/spatial-meta-ce-meta65-joint-runner-20261010`，分支 `feat/meta65-joint-control-runner-20261010`。主目录和上一轮worktree未改。正式协议 [56号](56-Meta65同初始化适应算子最小受控协议.md) 不重写，原 `no_candidate` 不改。

## 范围与预检合同

本轮用户授权evaluation-only runner、测试、CPU监督预检、launch-binding和独立审查，**不授权正式135cells、GPU适应、重新元训练、扩seed、调参、目标库或S8**。实现分为 [runner](../scripts/meta65_joint_runner.py)、独立 [verifier](../scripts/meta65_joint_verify.py)、外部 [supervisor](../scripts/meta65_joint_supervisor.py)；不调用旧screen的train_arm/decide/prepare，不导入train模块。

CPU监督预检固定 `CUDA_VISIBLE_DEVICES=""`、OMP/MKL/OPENBLAS=1：

- 认证完整三源库缓存、三个checkpoint SHA/大小、完整endpoint plan（非重新生成）、45个K20旧anchor身份/FP64真值/表数。
- 固定9个micro：A-head3/B-head3/C-last3 × chembl/legalacts/telstra × run0，每micro仅对原排序query前32条新计算并核对原始log_pred容限1e-5/1e-6和该micro standard Q95相对容限1e-4；不拿32条Q95冒充正式完整query Q95认证。
- 每micro另做同checkpoint none的32条工程路径核验，none不调用adapt，均记录元初始化前后digest不变。合计576次query呈现、9次CPU适应×3步=27inner更新，仅工程预检，无元训练。
- 所有15个共享身份侧表可写隔离输出；CPU结果只写microcells/verification与`preflight_completed`，formal cells=0，不写正式matrix summary或decision。输出不能进入PRICE/原run/缓存，原输入只读。
- CPU硬预算总900秒、preparation300/evaluation120/certification300；阶段串行且总预算优先，4GiB RSS、VRAM0、1GiB输出。预算是本轮预检专用，不扩正式4800/300/3600/900预算。没有第三次机械重试，错误必须带新诊断/修正后才可再运行。

正式未来执行仍按56号：135cells、270inner更新、0元训练；4800秒总、4GiB RSS、8GiB VRAM、1GiB输出。CPU预算/小样本micro通过不等于正式完整矩阵回归已通过；正式运行须完成45个完整diagonal回归后才接受135cells，部分产物均无结论资格。

## 输出与认证

共享身份侧表用S64 query_hash/S128 query_id/float64 true/int16 n_tables；编码超长失败，不截断。每cell只保存float64 raw log_pred及侧表SHA，完整身份/真值/表数由绑定侧表恢复。所有operator/checkpoint读取同一固定query侧表，n/count必须完整；输出记录selected与actually_changed参数名称/数量、steps、耗时、state digest、worker lifetime峰值RSS与cell CUDA allocated/reserved峰值，父监督另采owned进程树RSS及GPU进程显存。RSS lifetime HWM不是可重置的独立cell增量峰值，不混称。

独立verifier重算standard Q50/Q95与30联合桶分母，不复用生产分位数函数；配对log-Q变化、五run等权均值/均值比与四桶交互**尚未实现**，由后续独立审计脚本完成。空桶null、小桶标注、缺run不得删除。checkpoint是固定区组，禁止pooled跨初始化。CPU模式仅认证工程micro，不产生正式统计结论。

launch-binding绑定完整code/config/56协议/来源文档/input/checkpoint/plan/environment与命令SHA，记录预期schema和预算。GPU UUID与一次性用户approval只能在未来获授权、现场选空闲卡后补齐，不由预检伪造。`launch_ready`工程状态与`execution_authorized`分开；正式执行入口未提供匹配approval时必须拒绝。

## 验证状态

本文件为实施期记录。实际测试、远端CPU预检、独立审查、Git提交/推送和最终launch-ready判定写入 [验证报告](../reports/meta65-joint-runner-20261010/verification.md) 及 [launch-binding](../reports/meta65-joint-runner-20261010/launch-binding.json)。尚未完成的检查不得写通过；完整矩阵仍未执行。

### 2026-10-10 实施结果

- 本地聚焦 39 项通过；完整单线程 CPU 回归 942 项通过（15 skip），112.595 秒。
- 服务器 CPU 监督预检 `preflight_completed`：9 micro、退出码 0、20.146 秒、峰值 RSS 约 1.05 GiB、VRAM 0、输出约 144 MiB、`state_before==state_after` 全部成立；A/B=107521 参数/3 步，C=65 参数/3 步。
- `launch_ready` 仅在重跑 CPU 预检、绑定全部 9 个 code SHA 并通过独立复核后才可为 true；当前为 false。`execution_authorized=false`、`gpu_formal_executed=false`。正式 135-cell 仍需用户一次性 GPU 授权、空闲 GPU UUID 绑定和 approval 文件。
