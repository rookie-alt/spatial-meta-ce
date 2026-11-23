# server68 四卡实验准备快照

> 截止时间：2026-11-23 21:55:47 +08:00。整理自主执行任务及其本机操作记录；这是历史准备状态，不是本次记忆维护重新执行的训练、测试或实时资源检查。

## 实验范围与路径

- 本次仅准备 `GroupMean`、`GroupAttention`、`RelationTrue`、`RelationShuffled` 四组，各 seed42/43/44、24,000 updates，共 12 次。GPU0–3 各绑定一组，GPU4–7 不在本次范围。
- 远端启动目录：`/home/liushuaixi/weilin/module26-prep-20261118/launch-parallel-20261123-2136`。
- 远端输出目录：`/home/liushuaixi/weilin/module26-prep-20261118/outputs-parallel-20261123-2136`。
- 执行环境：`/home/liushuaixi/weilin/module26-runtime/venv/bin/python`；冻结的 37 份训练源码不变。
- 最终启动清单 SHA256：`eb80957daf62a680b7d0d80fe8ab86b91975ad7dba4d5927785a7be4fa190171`。
- 启动脚本 SHA256：`05c8b2907ba996858ec3a39346ae52c6fe6d7274918684b5744df0d92a2b8c0e`。

## 截至该时点的记录

- GPU0–3 逐卡 CUDA 初始化、128×128 矩阵运算及反向传播与 CPU/解析梯度交叉核对通过，进程随后退出；不是持续空闲或训练成功的保证。
- 12 份 spec 只调整设备与输出绑定，研究配方、步数及输入保持冻结；本轮授权单独绑定。
- 四组的 seed42 完整 dry-run 通过，`signed_and_bound`、`executed=false`，未创建研究训练输出。没有声称 12 份都完整 dry-run。
- 首轮独立审查发现监督父进程异常退出及日志写入失败时可能遗留自有训练进程；已复现修复，Linux 9 项队列合成测试通过，修复后的独立复审待返回。合成测试不是正式训练或精度结果。
- 状态仍为**待启动**；未产生本轮训练 checkpoint 或精度结果。复审、再次检查 GPU 空闲和正式启动由主执行任务继续处理，本记忆维护不执行这些步骤。
- 用户选择本次不补跑 A1-PG，不执行四库评估。旧权重缺失且等价性未认证，旧数字只能作历史参考；不得绕过 15 份权重的正式评估门禁。

## 来源与核验方式

本机来源目录为 `E:\weilin\operations-20261123`。22:02:14 +08:00 读取的本机记录 SHA256（后续追加会改变哈希）：

- `progress.md`：`86e55c643b534ebb3322b3e9986a73dc823f87b64281be521984098fcb53fa0a`。
- `launch-record.md`：`9833c9611b34e62176f081b5fd63206ac11c605afc40ed81ddad2150c4e42492`。
- `parallel-training-plan.md`：`d941f10d8342504edf3d2f9b614b51ca7312240bebd193755eb8973d868174d9`。

这些 SHA 绑定整理时的本机记录，不替代远端原始执行证据。主任务后续追加时这些 SHA 将与当前文件不同，须按时点解释；本快照不跟踪后续实时状态。接手时核对远端启动目录中的 CUDA 验证、dry-run、队列与完成记录，读取最新状态；不要重跑历史训练来补记忆，也不要把本快照视为新执行授权。
