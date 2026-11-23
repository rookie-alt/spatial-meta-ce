# 项目背景、范围与边界

> 整理日期：2026-10-05。来源：Claude 项目记忆 `spatial-meta-ce-project.md`。
> 本文件是整理后的项目内副本，不会自动与 Claude 原记忆同步；原文件保持不变。

## 背景与范围

毕业论文主题为面向数据漂移的自适应基数估计。历史设想涉及元学习、时间漂移及数据库集成，但用户在 2026-09-26 明确：当前只做空间漂移，即跨数据库少样本元学习；时间漂移、上下文学习、PostgreSQL 插件暂不做。

## 项目位置

> 2026-11-23 更新：本轮已迁移到 Windows/server68；旧路径保留为历史来源。

- Windows 工作区：`E:\weilin`；代码副本：`E:\weilin\code-training`。核对时两者均无 `.git`，不能把文件落盘视作 Git 提交。
- server68 项目根：`/home/liushuaixi/weilin`；训练代码：`module26-prep-20261118/code-training`；环境：`module26-runtime/venv`，均相对该根目录。
- 当前使用的 GitHub 仓库：`rookie-alt/spatial-meta-ce`；训练准备来源分支：`feat/official-module-training-prep-20261118`。Gitee 历史远程为 `william-forest/spatial-meta-ce`；同步状态按远端核对，不能互相替代。
- 历史 Mac 仓库：`/Users/williamlin/Downloads/13/spatial-meta-ce`；旧服务器项目与只读数据源分别为 `/gly/weilin/spatial-meta-ce`、`/gly/weilin/PRICE-master`。这些不是 server68 的活跃部署路径。
- 环境、实验范围与证据入口详见 [server68 接手记忆](server68-handoff-20261123.md)；项目指令入口为 [AGENTS.md](../AGENTS.md) 与 [CLAUDE.md](../CLAUDE.md)。

## 长期实验约束

- 目标库 `imdb/stats/genome/ergastf1` 不用于训练和模型/参数选择。
- 源验证库为 `chembl/legalacts/telstra`；适应策略只能在源侧选择。
- 主口径为 unique-query Q95；具体冻结终点与聚合规则以每次正式协议为准。
- 原 PRICE 数据及权重只读，不能用伪造数据补缺口。
- 历史记录中 IMDB finetune 因统计缺列阻塞完整四目标库验收；是否解除须查当前证据，不可默认为已解决。
- PG 对照口径校验是跨论文比较的前置条件，不足以认证所有模型、分区和指标都可直接跨论文比较；优先采用同评估器实测对照。

开始新实验先读 [实验总览](../实验文档/00-实验总览.md)、[工作流](spatial-meta-ce-workflow.md) 与 [证据状态入口](current-state.md)。
