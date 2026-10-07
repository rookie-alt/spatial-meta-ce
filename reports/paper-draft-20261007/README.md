# 论文实验章初稿交接与验证

> 主代理整合更新（2026-10-07）：本交接记录下方14项/代码hash属于writer交付快照，保留不改。此后独立数字/图表复核通过，并修复导出祖先symlink保护；当前15项paper专项通过，新的完整回归及最终源码绑定见 [续做验收](../s7-completion-20261007/review.md)。原19份派生输出/章节科学数字不变。

日期：2026-10-07。仓库 `/private/tmp/spatial-meta-ce-s7-diagnosis-20261007`，基准 `8fe6de4`，任务分支 `feat/s7-completion-and-paper-20261007`。本任务未切分支、提交、推送、访问服务器/网络，未训练、推理、读原始目标数据或重建 graphify，也未委派。

## 交付（ready，待主代理独立审查及总 suite）

- `/private/tmp/spatial-meta-ce-s7-diagnosis-20261007/实验文档/25-论文实验章节与图表初稿.md`：完整中文实验章初稿，包含设置、规模、三目标全部 K 主表、消融/负结果、诊断、公平性、局限与结论，不是完整毕业论文或虚构模板。
- `/private/tmp/spatial-meta-ce-s7-diagnosis-20261007/scripts/build_paper_results.py`：仅标准库的确定性导出器，不调用历史导出器，避免覆盖 14 号。
- `/private/tmp/spatial-meta-ce-s7-diagnosis-20261007/tests/test_paper_results.py`：14 项专项测试。
- `/private/tmp/spatial-meta-ce-s7-diagnosis-20261007/实验文档/results/paper-draft-20261007/`：最终 **14 CSV + 4 SVG + 1 精简 hash JSON = 19 派生产物**，共约 167 KB；没有大原始数据、权重或逐查询预测。早期口头“15 CSV”计数有误，以此实际数量为准。
- `/private/tmp/spatial-meta-ce-s7-diagnosis-20261007/reports/paper-draft-20261007/export-verification.json`：导出、前后 hash、确定性重建与图表表值核验记录；包含交付代码的 hash。

科学证据固定绑定 14 号机器 summary 及其原 **36 个来源 hash**、原始冻结协议、22 号和 S4/S5/S6/S7 历史结果，共 70 份不同输入。**AGENTS、00/23/99 实时清单不读取、不认证、不固定 digest**；专项测试在缺失 AGENTS、23 内容任意的临时证据副本中确认仍可正常读取。章节不把 23 当科学来源。未改变 26-source 或目标冻结。

## TDD 与实际测试

1. `tdd-red.log`：先写测试，导出器尚不存在，ImportError / FileNotFoundError，退出失败；不是通过记录。
2. `tdd-first-green-attempt.log`：首次实现遇到 S4 `metrics.csv` 的 initial/final 同 arm/db/bucket，两阶段不是非法重复。修复以 stage 作为完整键，校验 initial，主端点使用 final；不静默丢弃非法重复、不改历史 CSV。
3. `tdd-second-green-attempt.log`：11 项通过；随后补充实时清单解耦、缺指标/错聚合、完整 SVG 点位与 SD 标签测试。
4. `tdd-green.log`：最终 **14 项通过，0.772 秒**。实际命令：

```bash
cd /private/tmp/spatial-meta-ce-s7-diagnosis-20261007
PYTHONDONTWRITEBYTECODE=1 /opt/anaconda3/bin/python -B -m unittest discover -s tests -p test_paper_results.py -v
```

测试包含：完整 K/seed、训练 seed SD 与 PRICE support SD 分离、单 seed 三库等权 macro、目标 episode 重复/缺行、非法数值/缺指标、support 身份、JSON 重复 key、CSV 重复/缺行、源臂重复/缺失/不完整/非有限、SHA 缺失/篡改失败且不写 success、确定性导出和 SVG XML/点数。

`ruff.log`：仅本任务两份代码 **All checks passed!**。初始 ISC004 隐式字符串拼接通过加括号修复，记录保留于 `lint-initial-fix.log`。

```bash
/Users/williamlin/.cache/uv/archive-v0/YGscLdzyjxxZtKYJn0d_H/ruff-0.16.10.data/scripts/ruff check /private/tmp/spatial-meta-ce-s7-diagnosis-20261007/scripts/build_paper_results.py /private/tmp/spatial-meta-ce-s7-diagnosis-20261007/tests/test_paper_results.py
```

**未运行全套，未宣称项目总回归通过**；主代理负责独立检查与总 suite。

## 导出验证与重建

```bash
cd /private/tmp/spatial-meta-ce-s7-diagnosis-20261007
PYTHONDONTWRITEBYTECODE=1 /opt/anaconda3/bin/python -B /private/tmp/spatial-meta-ce-s7-diagnosis-20261007/scripts/build_paper_results.py
```

`export.log` 与 `export-verification.log/json` 记录实际运行。独立检查程序验证：

- 14 原 36 来源及全部 70 输入 hash；读取后及导出后仍一致。
- 14 份目标摘要共 1260 episode；Q50/Q90/Q95/Q99/mean 的支持重复聚合及离线 CSV，252 个初始化层单元、108 个方法×库×K 单元完整。
- 表 4–6 的全部 108 个显示均值和对应 SD 按 CSV 独立格式化逐格核对。
- S4–S6 的 36 个 final 源端点及三库 macro 与历史 JSON/CSV 一致；不是 pooled Q95，没有伪造单 seed SD。
- 两次实际 CLI 导出，章节和所有 19 产物逐字节 hash 相同；全部 CSV/SVG 与章节 hash 与 `source-bindings.json` 一致。
- 10 个 Markdown 本地链接存在；本任务稿件及派生产物无行末空白。
- 4 个 SVG XML 合法，每个目标图 4×6=24 点，源图 3×4=12 点，全部 K 保留。

正式图是 **SVG**。`*.svg.png` 是 macOS 本地 QuickLook 的非权威验证预览，不能把旧 thumb 的裁剪视为正式 SVG 不可用。当前 SVG viewBox 宽 768，重新生成的三目标预览已逐一查看：曲线、全 K、右图例及两类 SD 说明完整；源图包含失败库与 macro。QuickLook 的方形画布/空白不属于论文图的 intrinsic SVG 页面。无浏览器或网络请求。预览日志仅记录本地 OS 渲染。

## 科学边界与共享工作区

目标表先五 support 均值，再三训练 seed 均值与 ddof=1 SD；PRICE 单 checkpoint support SD、PG 无 SD。源侧 seed42 的 equal-database macro 独立呈现。原始逐查询源预测未重新求分位数，本轮复核的是已认证 JSON/CSV 的聚合及 hash；没有重新认证 raw target 标签、SQL 或独立数据库语义。

不声称四目标齐全、优于 PRICE、显著/因果或端到端加速；IMDB 完整 support 仍 blocked，P2 真实 SQL/选择率暂缓。S4/S5/S6 no_candidate 不变，S7 仅描述性诊断，机制 unknown。

本任务仅新增授权文件与结果/验证目录。共享工作区中主代理对 00/23/99、S7 工程及 S8 草案的修改属于主代理；本任务没有改写、纳入提交或清理它们。不据共享工作区的非空 git status 声称本任务失败，也不把其他窗口修改算成本任务变更。

## 自检（不替代独立审查）

- 准确性 4/5：1260 episode、原 36 hash、CSV 和表值实核；限于归档内部证据，未认证原始数据。
- 完整性 4/5：章节和 19 派生输出齐备；学校模板/完整论文与总 suite 按要求留外。
- 清晰性 4/5：训练 seed SD、support SD、源 macro 分开，图表顺序与限制显式；正式排版可再缩放 SVG。
- 可操作性 4/5：给出重建、测试、Ruff 命令与 hash；仍需主代理独立审查及整合。
- 简洁性 4/5：只一个导出器与一个测试文件；章节模板较长是完整实验章所需。

无评分低于 4 的已知交付缺口；下一步是主代理复核而非继续扩功能。
