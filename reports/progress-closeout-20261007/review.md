# 2026-10-07 当前任务收尾与独立审查

## 范围和归属

- 基础提交28077f3；独立检出 `/private/tmp/spatial-meta-ce-s7-diagnosis-20261007`，任务分支 `audit/s7-certification-closeout-20261007`。
- 用户授权查清未完成任务并继续；主代理仅进行接手、只读服务器确认、本地复核、必要修复和进度收尾。两个只读子代理分别核对结果证据、审查S7代码和测试，没有递归委派，也不写同一文件。
- 原主检出 `a52b11f` 的其他窗口修改不纳入本轮，不合并main，不运行新训练/适应/推理/目标评估/数据库计数，不更新graphify，不部署代码。

## 当前验证（不是历史通过数）

- 基线：`PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /opt/anaconda3/bin/python -B -m unittest discover -s tests -v`，457项通过、35.166秒。见 [baseline-tests.log](baseline-tests.log)。CPU单线程环境用于本地测试，不是服务器正式实验。
- 全项目 `ruff check scripts tests spatial_meta_ce`（0.16.10）失败，110项既有lint问题，见 [ruff.log](ruff.log)。本轮不把它写成全项目lint通过，也不为此重构历史S2–S6代码。
- S7五文件专项Ruff通过，见 [s7-ruff.log](s7-ruff.log)。这不能替代全项目lint。
- 历史输入前哈希见 [historical-inputs-before.json](historical-inputs-before.json)：333个文件，包含全部已归档结果文件、14号冻结汇总、21号S7协议、22号S7结果文档；初次收尾检查前后无变化。
- `git fetch origin`后S7本地和远端均为28077f3、ahead/behind=0/0；只读SSH查见S7 completed、不训练/不读目标；没有匹配run_s2–s7/train_erm/train_meta的Python进程。该进程筛查有模式覆盖限制，不表示全服务器空闲。

## 独立审查与后续验证

- [两个只读子代理初审摘要](independent-review.md)：归档聚合一致，但S7认证器存在可复现覆盖缺口；通过负例先复现后修复，没有把审查当作当前数字已错误的证据。
- [TDD red](tdd-red.log)：24方法，31 failure/8 error；[green](tdd-green.log)：24项通过、7.917秒。新增6方法包含多组篡改负例，fixture覆盖完整23源训练库。
- [最终完整回归](final-tests.log)：463项通过、43.786秒。[改动文件及核验脚本Ruff](final-ruff.log)通过；全项目110项既有问题不归为本轮通过。
- [归档确定性重建](archive-rebuild.log)：仅在临时副本运行，9个结果文件逐字节不变，overview/CSV与原归档相同，原工作区结果没有被重写。
- [服务器强化再认证](server-recheck.json)：CPU单线程40.744秒，300秒/8GiB上限；94208样本、46采样cohort、1504统计单元、9178 legalacts尾分母通过。210份历史输入/执行源码/输出在前后重新SHA核验。公共采样重放也重新认证，未使用checkpoint最终RNG。
- 再认证通过SSH标准输入以内存加载新复算器，未写服务器文件、未覆盖旧源码或正式verification；源cache/完整预测/旧checkpoint均只读。没有新模型加载推理、训练、适应、目标读取、数据库计数或GPU占用。本地只接收新证明及日志，不下载大预测或权重。
- 新验证器SHA与server-recheck.json绑定。原protocol中旧源码指纹表示原运行版本，不改写为本轮新源码SHA；新认证证明与历史运行严格区分。

本地不持有完整原预测，但本轮服务器已实际读取既有完整行做强化独立复算；这是统计再认证，不是第二次推理。


## 最终独立复审及限制

只读代码审查代理复审接受原六类主要修复，核对本地新验证器SHA与server-recheck.json一致；未发现需要阻断当前既有产物收尾、否定现有结果或重推理的实际缺陷。它没有重复执行主代理的完整回归或服务器核验。

仍保留两个非阻断工程P2覆盖缺口，不能扩大“验证通过”的范围：

- 少于200条的训练cohort在pair_check中只返回计数，非法A/B预测未额外做无条件数值检查。当前46个正式采样cohort均2048条，全体都已进入all数值复算，该路径不影响本次实际归档；后续复用到稀疏群体应补负例和检查。
- seen/unseen、sample_presentations与当前诊断使用support+query联合曝光。新验证器核对两角色联合总数，不独立重建每角色的完整计数；不可宣称“每个角色已独立认证”。当前公共采样digest、源输入及实际联合曝光再认证通过。

最终补测归档正常路径、symlink/hardlink拒绝、写后原始证据被改时不得生成passed证明；测试闭包lint修正后专项24项通过8.487秒，见final-s7-tests.log。完整463项最新运行43.786秒，生产复算器/归档器版本与该运行及服务器核验相同。

本地coverage模块不可用（coverage-unavailable.log），没有安装依赖或改训练环境；本轮没有重新测覆盖率，不沿用S7旧94.47%作为新代码覆盖率。

## 自检（不是外部认证）

| 维度 | 分数 | 证据及缺口 |
|---|---:|---|
| 准确性 | 4/5 | 完整既有行再认证及210前后hash通过；不能扩大为因果结论或所有稀疏输入安全 |
| 完整性 | 4/5 | 六类主要认证缺陷修复且回归通过；两个非阻断P2、IMDB输入及下一实验仍未完成 |
| 清晰性 | 4/5 | 23号区分正式结果、旧分支状态及真实待办；历史阶段文档仍较长，入口已明确 |
| 可操作性 | 4/5 | 有只读recheck_server.py、测试及独立报告；缺少完整输入的任务不能直接执行 |
| 简洁性 | 4/5 | 用户总结与详细证据分开；审查/测试日志较多，保留用于追溯 |

平均4.0/5。后续优先在新的独立任务补稀疏数值/逐角色认证负例，而非据当前结果继续扩loss网格；用户是否接受以当前任务产物及已明示边界为准。
