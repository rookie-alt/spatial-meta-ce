# CLAUDE.md

面向空间漂移的轻量级、监督式跨数据库基数估计实验（Spatial Meta-CE）。研究协议与现状看 [README.md](README.md)、[docs/protocol.md](docs/protocol.md)、[实验文档/00-实验总览.md](实验文档/00-实验总览.md)。

## 红线（不可违反）

这些是实验诚信约束，任何"更快出结果"的取巧都不要做：

- 目标库 `imdb/stats/genome/ergastf1` **不进入训练、超参选择、checkpoint 选择**。
- 不伪造缺失标签、不裁剪输出、不按目标测试挑 checkpoint。
- 数字只写服务器实际运行输出，估计值/推断值不得当结果写入实验文档。
- 推理耗时不等于数据库端到端加速；没有完整实验就不能声称优于基线。
- `/gly/weilin/PRICE-master` 原项目只读，不修改。
- checkpoint 产物（`*.pt/*.pth/*.npy/*.pkl`）与 `artifacts/`、`runs/` 永不提交。

## 目录地图

| 路径 | 内容 |
|---|---|
| `spatial_meta_ce/` | 库本体：`data.py`（特征/缓存）、`model.py`（集合模型）、`train.py`、`cli.py` |
| `configs/` | `data.json` 预处理、`meta.json`/`erm*.json` 训练、`evaluation.json` 评估 |
| `tests/` | 单元与协议测试；`test_protocol.py` 覆盖训练/评估协议约束 |
| `scripts/` | `run_experiments.py` 正式运行器、`deploy.py` 部署及审计、分析脚本 |
| `实验文档/` | 实验记录、结论与 `results/` 精简结果 |
| `docs/` | `protocol.md` 实验协议、`audit.md` 旧代码审查 |
| `reports/` | 服务器测试输出、状态与审计产物 |
| `vendor/PROVENANCE.json` | 三份 PRICE 代码来源与修改记录 |

## 工作流

- **每个任务开一个分支**，命名 `feat/`、`docs/`、`experiment/`、`audit/` 前缀 + 简短主题；完成后及时 push 到 Gitee（`origin` = `gitee.com/william-forest/spatial-meta-ce`），不要长期留在本地。
- Claude Code 建的 worktree 在 `.claude/worktrees/<name>`，是独立检出；在 worktree 内**只用该目录的绝对路径**操作，不要 `cd` 回主仓库。
- 提交信息用 `feat:` / `fix:` / `docs:` / `test:` / `chore:` 前缀，正文写清验证方式。
- 提交前自查：`git status` 确认只含本次改动；`.gitnexus/`、`.claude/`、`__pycache__` 不入库。
- 会话结束前更新实验进度记录，供下次接手。

## 验证命令

本地（改动后必须跑通再声称完成）：

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -B -m unittest discover -s tests -v
```

```bash
ruff check scripts tests spatial_meta_ce
```

服务器（真实环境，`torch/numpy/sqlglot` 版本以 `requirements.txt` 为准）：

```bash
ssh myserver "cd /gly/weilin/spatial-meta-ce && PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=\"\" /gly/weilin/PRICE-master/venv/bin/python -B -m unittest discover -s tests -v"
```

`myserver` 是 `~/.ssh/config` 里的别名（`root@172.16.33.129:30678`，密钥免密）。GPU 任务必须显式指定 `CUDA_VISIBLE_DEVICES`，并先确认卡空闲——本机不会自动起长任务。

## 代码索引

- 代码结构与调用关系统一使用 GitNexus；索引输入是代码库，不是实验证据数据库。
- 使用前核对索引对应提交和是否陈旧；陈旧索引只用于导航，不能替代协议、结果 JSON/CSV、日志或 Git 的直接核验。
- GitNexus 不能认证实验是否执行、数据来源是否正确、统计口径是否一致或结论是否有因果意义。

## 常见坑

- IMDB finetune 特征缺失，`prepare` 全量会**按设计报错**；`--allow-exclusions` 会改变支持分布，未经用户确认不要用于 IMDB。
- 训练/评估的模型配置与预处理必须一致；不用 PG 训练的 checkpoint 评估时也要 `--no-pg`。
- 已适应 checkpoint 不能回到 `evaluate` 当独立元初始化反复适应（跨 episode 污染）。
- `run_experiments.py` 默认只打印不执行，加 `--execute --device cuda` 才真跑。
- 同一命令重试到第 3 次仍失败就停下汇报，不要循环重试。

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
