# Meta65 GPU6 正式运行失败记录（2026-10-11）

用户本轮明确授权：按原冻结的 135-cell 同初始化 `none/head3/last3` 方案，检查通过后在 GPU6 启动**一次**。执行代码仍为冻结提交 `ac893b65bcf078dff940fa30ebcff72c60bdc4c6` 对应的九项代码 SHA；本任务 worktree 为 `audit/meta65-formal-safety-20261010`，启动时 HEAD `76c785c0748aee977600f6b30e2cab5d0b5a007c`，仅追加历史记录。launch-binding SHA256 为 `f7a2332d4e61518a340f3faf3c1ad08924660d92ec7739cb9914450cc42a3952`，原 [56号协议](56-Meta65同初始化适应算子最小受控协议.md)、config 与资源上限未改。

## 启动前和一次性授权

- 本地、服务器隔离代码根 `/gly/weilin/spatial-meta-ce-meta65-formal-safety-20261010-code` 均通过 `verify_launch_binding()`：9 项代码、7 项配置／协议／来源、6 项 CPU 预检证据与冻结 binding 一致。
- 服务器 `input_binding()` 只读认证 83 个文件、三个源验证库的完整 plan 和 45 个旧回归锚；三份 checkpoint、完整 endpoint plan 和源缓存 SHA 均吻合。启动前未发现同方案的运行进程或 approval。
- GPU6 UUID `GPU-0050e82c-8c24-d85d-00d3-b73f12638df0` 在创建 approval 前现场核对为显存 1 MiB、利用率 0%、无计算进程；冻结入口在消费 approval 前再次执行空闲门禁。本次没有换用当时也显示空闲的 GPU2。
- 新目录 `/gly/weilin/spatial-meta-ce-meta65-formal-safety-20261010-runs/formal-gpu6-20261011T101117Z/` 下的 `approval.json` 绑定唯一输出 `matrix/`、GPU UUID、代码／配置／协议与 launch-binding SHA。approval SHA256 `6cf772e94176289fdfdbd7e4ca007f69c9d50731dd67f7231d36c711f9e46431`，一次性收据 `approval.json.consumed.json` 已产生，supervisor PID 42234；**不能复用或重试**。

## 实际运行与失败

正式 supervisor 启动 worker PID 42238，在 `evaluation` 阶段失败，`returncode=1`、`failure=child_exit`、`elapsed=44.81220413418487` 秒。worker 日志定位到首个 GPU 前向 `functional_call → F.linear`：已启用 `torch.use_deterministic_algorithms(True)`，但 CUDA≥10.2 的 CuBLAS 操作在**未设置** `CUBLAS_WORKSPACE_CONFIG` 时被 PyTorch 拒绝。错误明确建议在运行 PyTorch 前设置 `CUBLAS_WORKSPACE_CONFIG=:4096:8` 或 `:16:8`；该环境改动没有包含在本次冻结执行中，**本轮未修改环境后重试**。CPU 预检不经过 CuBLAS，因而其通过不能证明正式 GPU 前向可执行。

监督器记录峰值 RSS `1,317,855,232` bytes、采样 GPU VRAM `0` bytes、输出 `150,734,020` bytes。已有 15 个共享身份 NPZ，**预测文件 0、完成 cells 0/135、diagonal 回归锚 0/45**；`verification.json` 不存在，独立正式结果认证未执行，`status.json` 为 `failed`/`formal_result=false`。这些资源数字是失败前的观测，不能用来宣称完整正式预算验收。没有受控统计结论，不改旧 `no_candidate` 或参考基线。

退出后 `ps` 不再列出 supervisor/worker PID，GPU6 复查为 1 MiB、0%、无计算进程；服务器再次只读通过相同的 9/7 项 binding 与 83 文件、三库、45 锚检查。PRICE、旧权重／plan／源缓存未修改；不终止别人的进程、不扩大 seed/K/库，也不启动 S8。

## 原始证据与后续边界

- 服务器原始输出在上述唯一目录，保留失败状态、身份侧表和一次性收据，不删除或追认成功。
- 本地精简归档：[supervisor.json](results/meta65-joint-control-formal-20261010/gpu6-20261011-supervisor.json) SHA256 `fc54373061f6053ae64c19d3dd3a39bf30b1b9141f0c9192a11191c4c2aa85ce`；[status.json](results/meta65-joint-control-formal-20261010/gpu6-20261011-status.json) SHA256 `eb346474f1ea78a712908c2e72551961161ad6c05a1f6e2c89a388609846a2ad`；[worker.log](results/meta65-joint-control-formal-20261010/gpu6-20261011-worker.log) SHA256 `20e538ff45b80b48b7c14ad7fdee6aff3b873d52db8b979bd7e38e907554d48c`。三份本地 SHA 与服务器原文件一致。
- 若要继续，需先为确定性 CuBLAS 环境建立新的可验证执行绑定与 GPU 窄预检，审查环境变化不会破坏原协议与锚容限；随后另获一次明确授权、重新核验输入与空闲卡、创建**新的**唯一输出与 approval。当前授权已消费，本轮不自动修复或再跑 135 cells。
