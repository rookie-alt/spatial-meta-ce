# 收紧后单次真实 P 执行：独立只读复审（APPROVED-WITH-NITS）

审查代理 Noether（01a11b52-d6a9-7e43-8088-7bda597fa23b），只读、未写文件、未联网/连服务器、未跑训练、未递归。
基线 HEAD `e06d849`。审查对象为本轮真实 P 执行证据（`real-proof/` 17 文件 + `execution-result.json` + 部署/预检/清单）。

## 结论：APPROVED-WITH-NITS（8 条清单全部成立）

| # | 检查项 | 结论 | 关键证据 |
|---|---|---|---|
| 1 | 失败定位真实性 | 通过 | `supervisor.failure_reason=resource_observation_failed`、`error="'VmRSS'"`；`sample_process` 727–735；`supervise` 752–758 是唯一可产生该 reason 的分支；同签名先例见 handshake-v2-real |
| 2 | 不是数值判据失败 | 通过 | 四面板 `engineering_verified=true`、`retained/reference_retained/reference_pass` 全 true；`scope=real_retained_condition_number_fp64_reachable_bound`；独立复算 `proof_complete=True` |
| 3 | 新 κ 判据实际生效 | 通过 | 逐面板复算 `κ=σ₁/σ_r`、`reachable=8·eps·κ·scale`、`tol=max(floor,reachable)`；h 面板 floor 主导、x 面板 κ 项主导；与 `panel-bound-comparison.json` 八项精确相等；`kappa_cutoff=1e12` 字段消失 |
| 4 | 不是资源超限 | 通过 | worker RSS 1.231GiB、采样峰值 AS 3.995GiB、wall 7.52/7.92s vs 12GiB/900s；RLIMIT_AS 与 RSS 分开报告；无阈值 reason |
| 5 | 来源/访问边界 | 通过 | access-log 去重恰为八输入；`H/target/checkpoint/GPU/smoke` 全 false；八输入 before==after==contract |
| 6 | 单一尝试 | 通过 | `attempts=1`、`no_rerun=true`、`retry_allowed=false`；9 成员 SHA 三方一致；独立解包复算与冻结包逐字节相同 |
| 7 | 本地与服务器一致 | 通过 | 17/17 文件 SHA256 与字节数 = 服务器 inventory |
| 8 | 状态边界 | 通过 | 顶层 `failed`、`formal_ready=false`、`original_G1_status=failed`、`can_retroactively_pass_original_G1=false`；`A_engineering_passed_this_attempt=false` 而 `A_physical_checks_passed_this_attempt=true` |

## NIT 与处置

1. **run 结果未落盘/未提交（重要）** → 已处置：新增本目录 `execution-result.json`、`real-proof/`（17）、`independent-execution-review.md`，并新增 45 号记录；随后单独提交。
2. 行号 751–757 应为 752–758（可选）→ 45 号已采用 752–758。
3. access-log 原始 40 条 vs proof 去重 8 条无标注（可选）→ 45 号 §2 已说明"去重后恰为八输入"。
4. `/proc` 竞态为复发故障模式（可选·设计观察）→ 45 号列为未来工作，需新授权协议修复。
5. h/x full `certificate.pass=false` 来自 `rank_stability=unknown`（可选）→ 45 号 §2 已标注。
6. execution.stderr 仅含计时（可选）→ 保持与既有记录风格一致。

## 未执行/未复算（登记）

未联网/未连服务器（仅本地副本 + 服务器 SHA 记录交叉）；未重跑 774/158/18（仅核对提交内日志）；未独立复算 SVD 的 σ 值本身；未读取原始 .npy/.pt 内容。剩余未知：远端引用是否等于 e06d849（已在推送前由主代理核实）；服务器侧残留进程/proof 目录现状。
