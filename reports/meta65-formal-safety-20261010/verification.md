# Meta65 formal 启动门禁修复验证记录

日期：2026-10-10（Asia/Shanghai）。独立 worktree `/private/tmp/spatial-meta-ce-meta65-formal-safety-20261010`，任务分支 `audit/meta65-formal-safety-20261010`，从已推送的 `2d0c0606c218bbb3a4669cb203e4782033e9e557` 开始。本轮原始冻结 [56 号协议](../../实验文档/56-Meta65同初始化适应算子最小受控协议.md) 与 config 未改变；本记录只对应门禁修复，**未创建 approval、未运行 GPU formal、未产生 135-cell 统计结论**。

## 现场阻塞与 TDD

用户曾明确授权基于旧代码的一次 formal；启动前确认工作树/远端九项代码与旧 binding SHA 一致，GPU3 一度满足 1 MiB/0%，但只读复审发现：`nvidia-smi` 采样无超时、组长先退时后代可能存活、最终状态写回后未再查输出上限、approval 无消费/防重放。这些不能满足“超限立即失败”，故**没有生成 approval，也没有启动**，原一次性执行未消耗。用户随后授权另开分支修复、测试和再冻结；该授权不扩展为新 SHA 的 GPU 执行许可。

先让新负例失败，后修改实现到通过：完整冻结根的兄弟输出路径、旧/缺失 code 预检 evidence、重复 approval、过忙 GPU、per-cell RSS/VRAM 高水位、`nvidia-smi` 不可响应、leader 提前退出后忽略 TERM 的后代、最终 sidecar 超容量、存活 worker 对进程表不可见、成功采样越过总/阶段 deadline、非零退出和监督器异常留下后代。工作中一次 macOS 沙箱 `os.killpg` 返回 `EPERM`，沙箱外复验通过；合成测试未改变实验配置或阈值。

## 真实验证

- 本地：`PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 CUDA_VISIBLE_DEVICES="" /opt/anaconda3/bin/python -B -m unittest discover -s tests -q`，最终 **964 项 OK、16 项平台跳过、110.597 秒、Python exit 0**。此前一次 `-v` 同为 964 OK/16 skip，但 zsh 收尾脚本误将只读变量 `status` 赋值，导致**外层脚本** exit 1；最终重跑消除了歧义。聚焦 Meta65 61 项通过（macOS 1 项 Linux-only skip）；改动脚本/测试 Ruff 均通过。全仓 Ruff 110 项与 `2d0c060` 基准逐文件/规则/消息一致，新增 0 项，未改邻近代码。
- Linux：服务器隔离代码根 `/gly/weilin/spatial-meta-ce-meta65-formal-safety-20261010-code`，`CUDA_VISIBLE_DEVICES=""`、单线程、外部 `timeout 180s`：Meta65 聚焦 **61 项通过**、0 跳过，包含 `waitid(WNOWAIT)` 的真实 Linux 测试；四项针对组清理与旧阶段截止时间的专项测试也通过。
- 最终候选隔离 CPU 预检输出 `/gly/weilin/spatial-meta-ce-meta65-formal-safety-20261010-runs/cpu-preflight-20261010-v2`：`preflight_completed`、returncode 0、phase=certification、`failure=null`、`29.194700518855825` 秒，峰值 RSS `1,055,506,432` bytes、VRAM 0、输出 `150,803,170` bytes。`verification.json` 为 `preflight_verified`、formal_result=false、9 micro；`microcells.json` 有 9 个 none_checks。六份精简证据 SHA 收录于本次 [cpu-preflight/](cpu-preflight/) 与新 [launch-binding](../meta65-joint-runner-20261010/launch-binding.json)；原始 15 个身份侧表与原NPZ仅留服务器，不入Git。

## 识别边界与待授权事项

- CPU micro 的进程树 RSS 是一秒采样峰值，不能推断 formal 的 4 GiB/8 GiB/1 GiB 预算已在真实 GPU 路径上通过；外部监督器会在正式运行时独立认证/失败终止。出于资源安全，worker 还对每 cell 生命周期高水位作同限检查；其余采样间隙风险不能冒充连续计量证明。
- 正式结果所需 `135 cells`、`45 diagonal` 回归锚、same-checkpoint none/head3/last3 对照**均未执行**。目标库没有读取，PRICE 原项目及历史权重保持只读，旧正式 `decision=no_candidate` 不变。
- 本任务最后只能交付重新冻结的**待用户确认** binding，标注 `launch_ready`（代码/输入/CPU门禁就绪）、`execution_authorized=false`、`gpu_formal_executed=false`。任何一次新的 GPU formal 需用户针对**本分支实际提交 SHA、binding SHA、现场空闲 GPU UUID 和唯一输出路径**再次明确确认，再由操作员单独生成与之绑定的 approval 文件；JSON 本身不是用户授权来源证明，消费记录只防止本工具自动重放。
- 独立只读复审曾指出采样失明、阶段跃迁和非零退出后代遗漏；逐项修复并补失败测试，窄复审未再发现确定的非零/异常路径误杀或漏杀问题。最终只读复审独立核对新 binding SHA `f7a2332d4e61518a340f3faf3c1ad08924660d92ec7739cb9914450cc42a3952`、9 项代码、7 项输入、6 项 CPU 证据及 `input-binding.json` 的九项新代码 SHA，未发现来源或状态不一致；没有重复跑测试或SSH。新 Markdown 链接、`git diff --check`、新增行凭据模式扫描通过。核心代码/证据提交 `260eb07eaefd92d78abd363b1bfea08978ec7f61` 已成功推送 Gitee 的 `audit/meta65-formal-safety-20261010`；收尾的进度记录另行提交并以最终 Git 状态确认。
