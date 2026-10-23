# Meta65 CuBLAS 环境修复与 GPU6 预检阻塞（2026-10-11）

接续 [60 号首次 GPU6 正式运行失败记录](60-Meta65-GPU6正式启动失败记录.md)。用户本轮要求“修复 CuBLAS 环境问题，预检通过后再运行一次”，并确认在新分支提交推送修复／证据后才进行正式运行。新分支 `experiment/meta65-cublas-preflight-20261011`，独立服务器代码根 `/gly/weilin/spatial-meta-ce-meta65-cublas-20261011-code`、输出根 `/gly/weilin/spatial-meta-ce-meta65-cublas-20261011-runs`；原冻结代码、PRICE、旧checkpoint/plan/cache及上次失败目录保持只读。

## 修复边界

- 冻结 `CUBLAS_WORKSPACE_CONFIG=:4096:8`；正式和 GPU 窄预检的 parent 在审批消费／启动前检查精确值与三个单线程变量，worker 在创建输出前复核。PyTorch 仍启用确定性算法与禁用 TF32，不放宽旧的代码／输入 SHA、模型、适应算子或正式 `4800s`／4GiB RSS／8GiB VRAM／1GiB 输出上限。
- 新增 `gpu_preflight`：GPU空闲门禁通过后，只跑三个已有checkpoint × 三个源验证库、run0的9个对角 micro cell（每cell最多32条query），另做9次`none`检查；没有正式approval、135-cell矩阵、目标库评估或新元训练。监督器对本任务进程树采样GPU显存；900秒总界、准备／评估／认证各300秒，4GiB RSS、8GiB VRAM、1GiB输出上限。
- 正式启动门禁要求新 CPU/GPU 预检证据、GPU6 UUID和完整绑定；审批消费前逐一对齐GPU原始目录与归档JSON/sidecar SHA、15身份NPZ/9预测NPZ清单，独立复算GPU微型预测，并比较运行时 Python/NumPy/Torch/CUDA 版本。**当前 GPU 预检未执行，新 launch-binding 尚不存在，`launch_ready=false`。**

## 已运行验证（不等于 GPU 通过）

- 新负例先失败再修复；本地单线程完整回归 **972项 OK、16项平台跳过**，最终运行约108.824秒。Meta65三模块聚焦本地69项（macOS一项条件跳过）；服务器目标环境 `Python 3.11.4/PyTorch 2.1.0+cu121` 聚焦 **69项 OK**。改动文件 Ruff 0.11.2通过、`git diff --check`通过。全仓Ruff在本地0.11.2版仍报告51项未改文件的历史问题；改动文件0项。本轮未获得覆盖率百分比（当前本地解释器无 `coverage` 包），不宣称达到80%。
- 服务器 CPU 监督预检返回 `preflight_completed`／exit0／`failure=null`，wall **71.39693674631417秒**、峰值RSS **1,060,610,048 bytes**、VRAM 0、输出 **150,803,895 bytes**。9个micro cell与9次none检查、`formal_result=false`；独立 `verify()` 与原verification一致，现场输入绑定复核86个文件、3库plan、45回归锚。六份精简证据归档于 [CPU预检目录](../reports/meta65-cublas-preflight-20261011/cpu-preflight/status.json)，本地与服务器逐一SHA一致。CPU路径不会调用CuBLAS，不能据此替代GPU检查。
- 服务器**全套**测试尝试未通过：观察到隔离部署未包含旧CSV测试夹具、该venv缺 `jsonschema`、Linux缺 `/private/tmp` 等其他模块测试错误。因此只报告服务器聚焦69项通过，不把全仓服务器回归说成通过，也不为跑通无关测试修改原项目／部署规则。

## GPU6 现场门禁与停止

单次只读 `nvidia-smi` 显示指定 UUID `GPU-0050e82c-8c24-d85d-00d3-b73f12638df0` 显存占用 **1522 MiB**、利用率 **35%**，计算进程PID **79773**（单独查询其显存为1532 MiB；两次采样非原子）。不符合显存≤100MiB、利用率0且无计算进程的冻结条件。因此本轮停在 GPU 预检前：**GPU micro 0/9，正式cells 0/135，diagonal 0/45**；没有创建/消费新approval、没有监督器或worker在GPU6运行，也没有正式结果或受控统计结论。没有等待、自动重试、换用其他卡或触碰他人进程。精简状态见[结构化记录](../reports/meta65-cublas-preflight-20261011/status.json)。

## 后续边界

本轮可提交推送已通过的修复、CPU证据与阻塞记录，但不得发布 `launch_ready=true` 或伪造GPU证据。用户下次明确继续后，重新核验GPU6与全部冻结输入；若空闲才运行一次有界GPU微型预检。其原始结果通过并核验后，另建与新代码/证据 SHA 绑定的正式launch-binding，提交推送并向用户展示具体提交和binding SHA，再由用户明确确认一次性正式GPU运行。原失败approval已消费，不能复用；不扩seed/K/库、不调参、不自动推进S8。
