# Windows/server68 环境与模块实验接手

> 记录日期：2026-11-23。来源为本轮用户决策、项目部署记录及主执行任务的启动记录；本文件保存接手背景与证据入口，不是自动更新的运行状态。

## 环境与位置

- 本机工作区为 `E:\weilin`，代码副本为 `E:\weilin\code-training`；2026-11-23 核对时均无 `.git`，本机 `git` 命令也不可用。文件修改、已提交、已推送须分别核验。
- SSH 使用已有别名 `server68`。项目根为 `/home/liushuaixi/weilin`，训练代码位于 `module26-prep-20261118/code-training`，Python 为 `module26-runtime/venv/bin/python`，均相对该根目录。
- 部署环境记录为 Python 3.11.18、PyTorch 2.1.0+cu121、NumPy 1.26.1、SQLGlot 18.17.0。旧 Mac 路径和 `/gly/weilin` 仅作历史来源，不能直接套用到 server68。
- GitHub 仓库为 `rookie-alt/spatial-meta-ce`，本轮准备来源分支为 `feat/official-module-training-prep-20261118`，核对提交为 `12c189b8378a1bf8223e84c56fc613028acff2c1`。Gitee 历史仓库为 `william-forest/spatial-meta-ce`；两端同步须分别报告。
- 已提交的 [server68 部署记录](https://github.com/rookie-alt/spatial-meta-ce/blob/12c189b8378a1bf8223e84c56fc613028acff2c1/reports/official-module-migration-20261123/server68-deployment-20261123.md) 属于较早的“部署完成、未做 CUDA 运算”阶段；后续四卡运算验证见下方准备快照。

## 本轮实验决策

- 沿用 [132 号官方 26/4 准备协议](../实验文档/132-模块候选官方26训练4评估准备.md) 的冻结配方与 37 份训练源码；本轮是 26 库训练、`validation=[]`，四个目标库不参与训练或选择。不得把历史 23/3 配方当成本轮已执行方案。
- 用户要求使用空闲 GPU 0–3 并行运行四个新增组，并选择本次不补跑 A1-PG 基线。分配为 GPU0 `GroupMean`、GPU1 `GroupAttention`、GPU2 `RelationTrue`、GPU3 `RelationShuffled`；每组依次跑 seed42/43/44，共 12 次。
- 每次 24,000 updates，同 seed 共用 sealed episode 计划，保留 `last.pt`。预算、监督运行器、失败停止和授权绑定沿用本轮正式启动包，不因本记忆另行授权或扩大范围。
- 四组均保留 A1-PG 的 PG 残差输出；分组聚合改变已有 token 的读出方式，关系感知加入 mapping 结构提示。工程验证不能证明精度收益。
- 截至下方 21:55:47 快照，旧 24k A1-PG 结果只作历史参考，旧权重未随迁移带来，也未认证与当前实现完全等价。当前正式评估门禁要求五臂 × 三 seed 共 15 份协议相符权重；本次 12 次训练不能代替完整对照，四库评估暂缓。后续权重补齐和认证情况须读取新的来源证据。

## 接手时核对

先读 [2026-11-23 四卡准备快照](../reports/server68-parallel-preparation-20261123.md)，再核对主执行任务最新记录、远端队列日志、进程身份与 checkpoint。快照截至 21:55:47 +08:00 为“待启动、清理修复复审待返回”，不能据此宣称训练运行、完成或有改进效果。

本机原始操作记录在 `E:\weilin\operations-20261123`，由主任务维护；侧对话不覆盖该目录、共享进度文档或冻结训练源码。后续状态优先核对新的真实证据，不把历史 GPU 空闲状态、测试数量或 PID 当作实时状态。
