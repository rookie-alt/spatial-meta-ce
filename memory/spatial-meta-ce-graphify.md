# graphify 证据索引与查询经验

> 整理日期：2026-10-05。来源：Claude 项目记忆 `spatial-meta-ce-graphify.md`。
> 本文件是整理后的项目内副本，不会自动与 Claude 原记忆同步；原文件保持不变。

## 两种图与同名输出

- 实验证据索引：`graphify-out/experiment_graph.json`、`manifest.json`、`semantic.json`，由 `scripts/build_experiment_graph.py` 消费或生成，记录来源 SHA256。
- graphify 实体/关系图：通过 graphify 工具构建，包含提取/推断关系及社区分析。
- 两者会写同名的 `graph.json`、`graph.html`、`GRAPH_REPORT.md`、`analysis.json`；Claude 的 2026-10-04 记忆还记录 `manifest.json` 共享路径。后运行者可能覆盖，必须先辨认产物来源。
- 生成图不入 Git；索引本身不证明实验执行或独立复现。

## 更新与查询

1. 重要代码、文档或结果变化后整体重建证据索引，不用 `graphify --update`。增量只增不删可能残留旧证据。
2. 修改 Markdown 后先重新提取受影响语义片段及 SHA256 绑定；仅改哈希、不更新语义会把旧结论伪装成新证据。
3. 增加文档或记忆前先检查现有覆盖，避免重复。
4. 回答实验/架构问题前优先查已有索引，但先确认它是否覆盖当前文件并保持 hash 一致。索引缺失或过期时回到原始来源，不能沿用旧查询结论。
5. `graphify query`、`path`、`explain` 的实际可用参数以安装版本的 help 为准；历史命令不保证适用于当前版本。
6. 新增记忆文件或共享工作区有其他写入者时，不擅自重建主任务的生成索引；交由主执行任务确认刷新时机。

```bash
python3 scripts/build_experiment_graph.py --root . --out graphify-out --semantic graphify-out/semantic.json
```

## 历史经验：2026-10-04，不当作当前实测

- 当时证据语义文件没有仓库内自动生成器，受影响 Markdown 需要人工重新提取语义。
- 当时查询有向图叶子可能返回节点但没有边；这不等于图损坏，可能是遍历方向或检索词问题。
- 可先换过程词，如 `sidecar`、`source26`、`映射`、`验证`；需要入边时使用适合的双向只读遍历，并说明方法。
- 不照搬旧节点数、文件数、成本或构建耗时为当前数字；重新读取实际产物再报告。
- `model_input_tokens = null` 表示未测量，不等于 0；UTF-8/4 只是估算。

详细原始记录位置与来源哈希见 [记忆索引](MEMORY.md)。相关：[项目指令](../AGENTS.md)、[进度同步](spatial-meta-ce-progress-sync.md)。
