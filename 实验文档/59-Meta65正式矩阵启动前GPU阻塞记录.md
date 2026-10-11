# Meta65 正式矩阵启动前 GPU 阻塞记录

日期：2026-10-10（Asia/Shanghai）。本轮用户明确授权冻结提交 `ac893b65bcf078dff940fa30ebcff72c60bdc4c6` 与 [launch-binding](../reports/meta65-joint-runner-20261010/launch-binding.json) SHA `f7a2332d4e61518a340f3faf3c1ad08924660d92ec7739cb9914450cc42a3952` 下的一次 135-cell GPU 适应与推理矩阵；不重新元训练、不新增或修改实验代码、不开放调参。

## 实际状态

**`blocked_no_idle_gpu`，formal 未启动**。指定 worktree `/private/tmp/spatial-meta-ce-meta65-formal-safety-20261010`、分支 `audit/meta65-formal-safety-20261010` 与接手 HEAD 已核对，接手时工作树干净；本地和远端隔离代码 `verify_launch_binding()` 均通过，9 项代码、7 项来源/配置/协议、6 项 CPU 证据与指定 binding 一致。

现场 `ssh myserver` 查询的 8 卡全部有计算进程，显存 6678-10566 MiB、利用率 7-99%；没有一张符合显存≤100 MiB、利用率0、无计算进程。完整 UUID/PID 与命令见 [快照](../reports/meta65-joint-control-formal-20261010/gpu-snapshot.txt)，记录时间在查询之后为 2026-10-10 18:32:34 +08:00，不冒充 GPU 自带时间戳或持续空闲状态。

因此按用户停止条件，不创建 approval/消费收据，不选择 GPU，不创建 formal 输出目录、不启动监督器/worker、不等待或自动重试，不影响任何其他进程。完整 plan、checkpoint、源缓存本轮未重新现场认证；历史 CPU 证据的输入绑定不能代替这些现场前置检查。

## 工程验收与统计结论

135 cells=0/135，diagonal=0/45，独立 CPU 正式结果认证未执行；完整分母、query 身份、FP64 真值、表数、更新范围与初始化隔离的正式验收未达成。没有新预测或部分矩阵，**不产生受控统计结论**，也不根据启动阻塞判断初始化差异、算子影响或交互。

正式 GPU 适应更新=0，元训练更新=0，query 呈现=0。没有正式进程的 wall/RSS/VRAM/output 遥测，这些量记为未测，而不把他人显存或旧 CPU 峰值写成本次正式资源消耗。冻结 300/3600/900 秒阶段、4800 秒总监督、CPU线程1、RSS≤4 GiB、VRAM≤8 GiB、输出≤1 GiB 均未改变，未执行不等于预算验收通过。

旧 `decision=no_candidate`、基线、冻结协议/输入及历史权重不变，PRICE 保持只读。验证与同步细节见 [verification.md](../reports/meta65-joint-control-formal-20261010/verification.md)，精简证据见 [startup-status.json](results/meta65-joint-control-formal-20261010/startup-status.json)。本次仅归档启动阻塞，不复写旧 binding 的历史授权字段，不重建生成索引。

## 一条下一步建议

等待用户明确继续后，重新现场检查空闲卡并完整重验冻结输入，再为该次 UUID/唯一输出路径生成一次性 approval，运行原 135-cell 矩阵。证据只支持“当前没有可启动 GPU”，不支持增加模型容量、重新元训练、扩 seed/K/库、搜索超参数、读取目标库或自动推进 S8。

## 2026-10-11 用户恢复后的单次现场复查

用户明确要求重新查看 GPU 并继续实验。指定 worktree 的接手 HEAD 为 `d9a1798884cda02c2895263b6f31af9fe301f0a8`（在冻结执行提交 `ac893b65bcf078dff940fa30ebcff72c60bdc4c6` 上仅追加上一次阻塞归档），接手时工作树干净；本地 launch-binding SHA 仍为 `f7a2332d4e61518a340f3faf3c1ad08924660d92ec7739cb9914450cc42a3952`。本次未重新认证远端代码、完整 plan、checkpoint 或源缓存；不能将上一次核验写成本次完成。

新的 [GPU 快照](../reports/meta65-joint-control-formal-20261010/gpu-snapshot-20261011.txt)显示8卡均有计算进程，显存6324–21533 MiB、利用率52–96%，再次不满足显存≤100 MiB、利用率0且无计算进程的门槛。按冻结停止条件即刻停止：`blocked_no_idle_gpu`，未创建或消费approval、未选择UUID、未启动监督器/worker或正式输出；本次0/135 cells、0/45 diagonal，正式资源峰值未测，无受控统计结论。保留先前快照，不把两次查询视为持续监测；不自动轮询、重试或占用他人资源。旧 `no_candidate`、预算及PRICE/冻结输入不变。下一次仅在用户明确恢复后重新现场核验GPU及全部输入。
