# AGENTS.md

面向空间漂移的轻量级、监督式跨数据库基数估计实验（Spatial Meta-CE）。
本文件参照 [CLAUDE.md](CLAUDE.md) 和 Claude 对应该工作区的项目记忆整理（2026-10-05），是 Codex 的项目指令入口。

## 接手与项目记忆

每次接手项目，先阅读：

1. [项目记忆索引](memory/MEMORY.md) 与 [项目背景及范围](memory/spatial-meta-ce-project.md)。
2. [进度与下一步](实验文档/99-进度与下一步.md)，再核对当前 `git status`、最新提交和对应结果 JSON。
3. 执行、分析或审查任务前，必读 [自动子代理委派、方案审查与验证](memory/spatial-meta-ce-agent-delegation.md)；按该记忆中的用户授权与边界判断是否自动委派。
4. 根据任务阅读 [实验总览](实验文档/00-实验总览.md)、[实验协议](docs/protocol.md) 和相关实验正式协议。

按需补读：

| 场景 | 记忆文件 |
|---|---|
| 改代码、提交、同步结果 | [工作流](memory/spatial-meta-ce-workflow.md) |
| 收尾或新会话接手 | [进度同步](memory/spatial-meta-ce-progress-sync.md) |
| 重复失败、状态未变化 | [重试限制](memory/stop-after-three-repeats.md) |
| 区分历史进度与当前证据 | [证据与状态入口](memory/current-state.md) |

项目记忆文件需明确读取，不假设它们会因位于 `memory/` 下而自动注入。
它们是可追溯的背景摘要，不替代原始证据，不授予服务器、数据库、训练或部署权限。
日期较早的测试结果、进程、GPU 空闲情况、分支和索引规模不能自动沿用为当前状态。

## 实验红线

- 本轮仅研究空间漂移（跨数据库少样本元学习）；时间漂移、上下文学习和 PostgreSQL 插件不在当前范围。
- 目标库 `imdb/stats/genome/ergastf1` 不进入训练、超参数选择、适应策略选择或 checkpoint 选择。
- 适应策略只能在源验证库 `chembl/legalacts/telstra` 选择。不得看目标测试结果再选择方案。
- 不伪造缺失标签或随机特征，不借用 test 数据补 support，不静默过滤失败查询，不裁剪输出美化结果。
- 正式实验数字只记录真实运行输出；估计、推断、合成测试和 smoke 结果必须明确标注，不能冒充正式结果。
- 主口径为 unique-query Q95；每次实验遵循其冻结协议，不把 all-rows、pooled、macro 或不同 seed 聚合方式混用。
- 冻结统计到特征编码一致，不等于真实 SQL 语义或真实选择率正确；审计相关性不等于预测误差的因果证据。
- 推理耗时不等于数据库端到端加速；没有完整对照不能声称优于 PRICE。
- `/gly/weilin/PRICE-master` 原项目、数据和历史权重只读，禁止覆盖或修改。
- `*.pt/*.pth/*.npy/*.pkl`、`artifacts/`、`runs/` 和生成索引不提交 Git。

## 目录地图

| 路径 | 内容 |
|---|---|
| `spatial_meta_ce/` | 库本体：特征与缓存、模型、训练、评估、CLI |
| `configs/` | 数据预处理、训练与评估配置 |
| `tests/` | 单元测试与实验协议约束测试 |
| `scripts/` | 正式运行器、部署、审计和分析 |
| `实验文档/` | 实验协议、记录、结论与 `results/` 精简证据 |
| `docs/` | 通用协议与历史代码审查 |
| `reports/` | 验证日志、状态与审计产物 |
| `memory/` | 项目长期记忆与来源索引；不存凭据或大型产物 |
| `vendor/PROVENANCE.json` | PRICE 引用代码的来源及修改记录 |

## 工作流与共享工作区

- 每个独立实现任务使用 `feat/`、`docs/`、`experiment/` 或 `audit/` 前缀的任务分支，不直接修改或推送 `main`。
- 先检查已有修改及其归属。共享工作区或侧对话不得擅自切分支、重置、清理、覆盖或提交其他窗口的工作。
- worktree 是独立检出；在 worktree 内仅使用该检出的绝对路径，不回主仓库操作。
- 修改应最小化，与既有风格一致；不借审计或记忆维护批量重构邻近代码。
- 提交信息用 `feat:` / `fix:` / `docs:` / `test:` / `chore:`，正文说明实际验证方式。
- 在主执行任务的授权范围内及时提交并同步 Gitee；只纳入本任务的修改。仅完成本地文件编写，不等于已提交或已推送。
- 会话收尾更新进度与下一步；若该文件正被另一窗口编辑，先协调，不覆盖其未完成内容。
- 不把密钥、密码、数据库凭据、访问 token 或含凭据的命令复制到记忆、文档、日志或提交中。

## 验证命令

以下命令是验证说明，不是自动运行授权；执行前确认环境和任务范围。

本地代码改动后：

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -B -m unittest discover -s tests -v
ruff check scripts tests spatial_meta_ce
```

使用实际具备依赖的 Python；当前 shell 的 `python3` 未必等于训练或测试环境。
只改文档时至少检查本次文件的 Markdown 链接、空白和来源；未经运行不要声称测试通过。

服务器真实环境：

```bash
ssh myserver 'cd /gly/weilin/spatial-meta-ce && PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES="" /gly/weilin/PRICE-master/venv/bin/python -B -m unittest discover -s tests -v'
```

`myserver` 使用用户 SSH 配置及密钥，不把认证信息写入项目。GPU 任务须获授权、先检查空闲卡，并显式指定 `CUDA_VISIBLE_DEVICES`；不要自动启动长期任务。

## 代码索引

- 代码结构与调用关系统一使用 GitNexus；日常问答、修改、实验收尾和提交不强制先查询索引，仍以协议、代码、结果 JSON/CSV 和 Git 为准。
- 使用 GitNexus 时先核对索引对应提交和是否陈旧；陈旧索引只能作为导航，不能替代原始实验证据。
- 实验数字、来源绑定、统计口径和结论仍直接核对原始协议与结果文件；代码索引不认证实验执行、数据正确性或因果结论。

## 常见坑

- IMDB finetune 统计缺列时，完整 `prepare` 会按设计报错；`--allow-exclusions` 会改变支持分布，未经用户确认不用于 IMDB。
- 训练与评估的模型配置、预处理必须一致；不用 PG 的 checkpoint 评估时也使用 `--no-pg`。
- 已适应 checkpoint 不得反复作为独立元初始化适应，避免跨 episode 污染。
- `run_experiments.py` 默认只打印；`--execute --device cuda` 才会真正执行，不得把计划打印当作运行完成。
- P2-C 必须先认证与冻结统计同版本的源表、历史来源链和只读一致性快照；PG 估计、join 后基数或标签不能替代真实过滤计数。
- 同一实质命令重复而无新增信息时停止；不发第三次机械重复，报告已有证据和阻塞原因。

<!-- gitnexus:start -->
# GitNexus — Code Intelligence

This project is indexed by GitNexus as **spatial-meta-ce** (12159 symbols, 19872 relationships, 276 execution flows). Use the GitNexus MCP tools to understand code, assess impact, and navigate safely.

> If any GitNexus tool warns the index is stale, run `npx gitnexus analyze` in terminal first.

## Always Do

- **MUST run impact analysis before editing any symbol.** Before modifying a function, class, or method, run `gitnexus_impact({target: "symbolName", direction: "upstream"})` and report the blast radius (direct callers, affected processes, risk level) to the user.
- **MUST run `gitnexus_detect_changes()` before committing** to verify your changes only affect expected symbols and execution flows.
- **MUST warn the user** if impact analysis returns HIGH or CRITICAL risk before proceeding with edits.
- When exploring unfamiliar code, use `gitnexus_query({query: "concept"})` to find execution flows instead of grepping. It returns process-grouped results ranked by relevance.
- When you need full context on a specific symbol — callers, callees, which execution flows it participates in — use `gitnexus_context({name: "symbolName"})`.

## Never Do

- NEVER edit a function, class, or method without first running `gitnexus_impact` on it.
- NEVER ignore HIGH or CRITICAL risk warnings from impact analysis.
- NEVER rename symbols with find-and-replace — use `gitnexus_rename` which understands the call graph.
- NEVER commit changes without running `gitnexus_detect_changes()` to check affected scope.

## Resources

| Resource | Use for |
|----------|---------|
| `gitnexus://repo/spatial-meta-ce/context` | Codebase overview, check index freshness |
| `gitnexus://repo/spatial-meta-ce/clusters` | All functional areas |
| `gitnexus://repo/spatial-meta-ce/processes` | All execution flows |
| `gitnexus://repo/spatial-meta-ce/process/{name}` | Step-by-step execution trace |

## CLI

| Task | Read this skill file |
|------|---------------------|
| Understand architecture / "How does X work?" | `.claude/skills/gitnexus/gitnexus-exploring/SKILL.md` |
| Blast radius / "What breaks if I change X?" | `.claude/skills/gitnexus/gitnexus-impact-analysis/SKILL.md` |
| Trace bugs / "Why is X failing?" | `.claude/skills/gitnexus/gitnexus-debugging/SKILL.md` |
| Rename / extract / split / refactor | `.claude/skills/gitnexus/gitnexus-refactoring/SKILL.md` |
| Tools, resources, schema reference | `.claude/skills/gitnexus/gitnexus-guide/SKILL.md` |
| Index, status, clean, wiki CLI commands | `.claude/skills/gitnexus/gitnexus-cli/SKILL.md` |

<!-- gitnexus:end -->
