# Spatial Meta-CE 项目记忆索引

整理日期：2026-10-05。本目录将 Claude 对应该工作区的相关长期记忆整理到项目内，供 Codex 与其他 agent 明确读取。

## 读取顺序

1. [项目指令](../AGENTS.md)。
2. [项目背景与范围](spatial-meta-ce-project.md)。
3. [进度与下一步](../实验文档/99-进度与下一步.md) 和 [证据状态入口](current-state.md)。
4. 在 Windows/server68 环境接手时，补读 [server68 环境与模块实验接手](server68-handoff-20261123.md)。
5. 按具体任务读取下表；不要把所有历史记忆当作当前运行状态。

| 项目内记忆 | 对应 Claude 原文件 |
|---|---|
| [项目背景、范围和数据边界](spatial-meta-ce-project.md) | `spatial-meta-ce-project.md` |
| [Windows/server68 环境、四组实验决策与证据入口](server68-handoff-20261123.md) | 2026-11-23 项目接手补充，非 Claude 原记忆副本 |
| [分支开发、交付后快进 main、协作与同步](spatial-meta-ce-workflow.md) | `spatial-meta-ce-workflow.md` |
| [收尾记录与新会话接手](spatial-meta-ce-progress-sync.md) | `spatial-meta-ce-progress-sync.md` |
| [证据图谱更新与历史查询经验](spatial-meta-ce-graphify.md) | `spatial-meta-ce-graphify.md` |
| [重试和轮询限制](stop-after-three-repeats.md) | `stop-after-three-repeats.md` |
| [精度首轮计划与用户暂缓训练](precision-screen-20261005.md) | 本项目新增，非 Claude 原记忆副本 |
| [自动子代理委派、方案审查与验证](spatial-meta-ce-agent-delegation.md) | 2026-10-07 用户确认的项目执行约定，非 Claude 原记忆副本 |

[证据与状态入口](current-state.md) 为本次新增，不是 Claude 原记忆的逐字副本；它仅记录恢复证据的方法与明确日期的已提交历史基线。

## 来源位置与整理方式

- Claude 原记忆目录：`/Users/williamlin/.claude/projects/-Users-williamlin-Downloads-13/memory`。
- Claude 原索引：[MEMORY.md](/Users/williamlin/.claude/projects/-Users-williamlin-Downloads-13/memory/MEMORY.md)。
- 指令参考：[仓库 CLAUDE.md](../CLAUDE.md) 与 [工作区 CLAUDE.md](../../CLAUDE.md)。
- 内容按主题整理，不逐字搬运；移除会话 ID、无关全局记忆、易过期的索引规模和未经本轮复验的强结论。
- 来源文件中的长期约束保留；历史环境、测试、指标与状态明确标注日期并指向正式证据。
- 原 Claude 文件没有修改。本目录与原目录不自动双向同步。
- `feedback_cc_switch_plugins.md` 属于全局插件配置反馈，与本项目实验无关，因此未迁入。

### 整理时读取的原文件 SHA256

- `spatial-meta-ce-project.md`：`bce610c50ec7df8c918db0d81311404d1e8d9d8ef73e858bc391a893ad595fb3`
- `spatial-meta-ce-workflow.md`：`5c98c6baada2f2e085a045c0e0117b66139e78fec7016836bac4d66d5fd8963e`
- `spatial-meta-ce-progress-sync.md`：`18d3e1f54575057c0c400471abcb623b6578b39e91d1654878908650e44ad3ba`
- `spatial-meta-ce-graphify.md`：`250254c87a72b13fa63de62a2aa2acfbcabc0385fd99e1fb0cabb10ed7b0bad5`
- `stop-after-three-repeats.md`：`60b8eab4982f736345f30aad3babfaab403d624dc7265d9ea8cb0ccb60fac086`

## 维护规则

- 只记稳定约束、可复用经验、背景决策与证据入口，避免复刻不断变化的实时进度。
- 新增前检查已有内容；有更新优先修改相应主题，并标注日期、来源、验证状态。
- 密码、私钥、访问 token、数据库凭据和含凭据的命令不得进入本目录。
- 验证日志和实验数字仍存 `reports/`、`实验文档/results/`，不要伪装成记忆里的新证据。
- 不假设目录名 `memory/` 会自动加载：通过 [AGENTS.md](../AGENTS.md) 的读取说明进入。
