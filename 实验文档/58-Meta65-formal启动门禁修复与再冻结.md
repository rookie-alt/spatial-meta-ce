# Meta65 formal 启动门禁修复与再冻结

日期：2026-10-10（Asia/Shanghai）。任务 worktree：`/private/tmp/spatial-meta-ce-meta65-formal-safety-20261010`，独立分支 `audit/meta65-formal-safety-20261010`，从 `2d0c0606c218bbb3a4669cb203e4782033e9e557` 建立。原 [56 号冻结实验协议](56-Meta65同初始化适应算子最小受控协议.md) 不变；旧 `no_candidate` 不变。

## 授权边界

用户曾授权依据 `2d0c060` 与旧 [launch-binding](../reports/meta65-joint-runner-20261010/launch-binding.json) 运行一次 135-cell GPU formal。启动前只读独立审查发现监督和一次性门禁缺口，因此**未创建 approval、未执行正式实验**；原一次性执行没有被消耗。用户随后只同意另开分支**修复门禁、补测试并重新冻结**，不构成对修改后 code SHA 的 GPU 执行授权。本轮不能启动 GPU、训练、目标库评估、调参、扩 K/seed 或自动 S8。

旧 [launch-binding](../reports/meta65-joint-runner-20261010/launch-binding.json) 仅对旧 SHA 有效；改动后必须重建绑定，旧的 `launch_ready=true` 不再作为本分支可启动证据。结果/回归/执行状态以本次 [验证记录](../reports/meta65-formal-safety-20261010/verification.md) 和最终再冻结 binding 为准，不能沿用 943 项/旧 CPU preflight 的历史数字作为本轮实测。

## 启动保护增量

1. 正式 parent 每次重新校验 committed binding 的 9 个代码 SHA、配置/协议/来源文件、CPU 预检证据、冻结预算；formal `approval.json` 必须绑定此 binding 文件 SHA、完整 code/config/protocol SHA、唯一输出路径和现场 GPU UUID。approval 的来源为**用户在交互中直接授权**，JSON 只记录操作员按该授权生成的收据，不能单靠字段证明消息来源。
2. `--execute` formal 由 parent 在 GPU 空闲复核（指定 UUID、显存≤100 MiB、利用率0、无计算进程；`nvidia-smi` 调用限时）后，以 `O_EXCL` 创建 `approval.json.consumed.json`。子进程核对收据 SHA、输出和 parent PID。收据在成功/失败后均保留，不自动删除或重跑；dry-run 不消费。**本轮不生成可用 approval**。
3. 输出不能位于整个 `/gly/weilin/spatial-meta-ce-meta65-20261009-frozen` 根及 PRICE/缓存；旧数据、checkpoint 与权重仍只读。
4. 外部 watchdog 的 `nvidia-smi` 调用有 5 秒子超时；`/proc` 根不可枚举或存活 worker 无法出现在进程表时 fail-closed。Linux 通过 `waitid(WNOWAIT)` 在后代确认/清理前保留组长 PID，所有退出路径（含非零退出、父监督异常）均核对同组后代，拒绝向身份不匹配或已回收且可能复用的 PGID 发信号；旧阶段超时不能被 phase 切换抹去。状态/sidecar 写回后再核对总时限与最终输出字节上限，超限不得标记 completed。
5. 56 号冻结的 `4800s` 总时限、`300/3600/900s` 阶段预算、4 GiB RSS、8 GiB VRAM、1 GiB 输出及 135 cells、45 个 diagonal 锚均不改变。每秒采样是该监督协议的离散观测，不能声称能在采样间隙观测任意瞬时峰值；worker 记录的 per-cell 显存高水位也应作为失败检查。没有完整正式矩阵，不能声称正式资源预算已验证。

## 再冻结与停止条件

- 先对每项缺口写可失败测试，再在本地跑聚焦测试、完整 CPU 回归和改动文件 Ruff；随后以 `CUDA_VISIBLE_DEVICES=""` 运行有界**服务器 CPU 预检**，只核对新代码下的监督/回归路径，不生成正式指标。
- binding 绑定新的 code SHA 和真实 CPU 预检证据 SHA，保留 config、56 号协议、输入/三个 checkpoint 与完整 plan 的原 SHA；同新代码发布后再次独立只读复审。文档、binding、预检状态必须明确区分 `launch_ready`、`execution_authorized` 和 `gpu_formal_executed`。
- 如原输入、资源预算、测试或预检无法认证，报告 `blocked`，不重试拿到好看结果；任何新的 GPU 执行仍须用户**再次明确确认**该新 commit 和 binding，再现场选空闲 GPU 并生成**本次专用** approval。

## 本次已核验状态（2026-10-10）

- 新测试先失败后修复：包括调用超时、leader 先退出后代忽略 TERM、非零退出/父监督异常清理、采样不可见、跨 deadline 返回、phase 跃迁、最终输出容量、approval 消费与新代码预检来源绑定。完整 macOS CPU 回归 964 项通过（16 项平台跳过），合成 supervisor+runner+verifier 聚焦 61 项通过（macOS 含 1 项 Linux-only skip）。服务器 Linux/CUDA 隐藏聚焦 61 项通过，四项 Linux 专项额外测试通过。
- 服务器新代码 SHA 的隔离 CPU 监督预检 `preflight_completed`、worker 验证 `preflight_verified`、formal cells=0、9 个 micro 与 9 个 none_checks，wall `29.194701` 秒、峰值 RSS `1,055,506,432` bytes、峰值 VRAM 0、最终输出 `150,803,170` bytes，watchdog 未触发。完整证据见 [CPU 预检](../reports/meta65-formal-safety-20261010/cpu-preflight/) 和 [验证记录](../reports/meta65-formal-safety-20261010/verification.md)。此前 `cpu-preflight-20261010` 是较早候选 SHA 的诊断，**不可**用于新绑定。
- 这些只是 CPU 微型工程验证，不是新的正式统计结果；源侧 `no_candidate` 原决策不变。新的 [launch-binding](../reports/meta65-joint-runner-20261010/launch-binding.json) 生成且复核后才可称 `launch_ready=true`；`execution_authorized=false`、`gpu_formal_executed=false` 始终保持。本轮不创建可运行的 approval，也不占用 GPU 执行额度。
