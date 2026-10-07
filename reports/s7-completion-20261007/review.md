# S7剩余工程补齐与文档工作

基准8fe6de4；分支feat/s7-completion-and-paper-20261007；日期2026-10-07。主代理只修改S7独立复算器、其测试和总进度。两个子代理分别负责S8协议草案与论文实验章节/图表导出，其写入范围独立，不编辑00/23/99或S7代码。原主检出其他窗口AGENTS/MEMORY/委派记忆未触碰。

## 缺陷复现与修复

- 原版本合成valid fixture交换support/query计数后仍verify passed，证实先前仅联合曝光认证；稀疏pair在计数返回前未无条件检验非有限或溢出输出。
- TDD red：26方法、11 failures（tdd-red.log）。fixture先改为真实小规模NumPy采样生成23训练库角色计数与公共hash链，保持单元测试为合成证据，不能冒充正式实验。
- 修复numbers：一维配对、finite、非负真值、Q exponent和raw square overflow明确拒绝。pair即便n<200也先检查当前非空桶数值，再仅保留计数，不增加小桶分位数或披露。
- 独立角色重放器不调用生产replay/advance/sampler；用原初始化、源顺序、抽样调用顺序逐episode生成支持/查询身份，独立累积两个role计数并核对完整digest/episode/总呈现，再逐库逐query角色数组比较。仍使用相同NumPy RNG API，不称为跨库/跨随机数引擎的独立复现。
- TDD green：26项通过、9.797秒（tdd-green.log）；改动文件Ruff通过（ruff.log）。

## 服务器实际再认证

[server-recheck.json](server-recheck.json)为新证据，不覆盖原S7/S6结果：CPU单线程45.641秒、8GiB/300秒上限；94208样本、46采样cohort、1504统计单元、9178 legalacts尾分母通过，210历史文件前后SHA一致。逐query角色曝光全部匹配；support采样角色597915、query采样角色1024000，合计1621915。**这些是episode采样角色**，ERM实际合并support+query监督，不能把support计数解释为inner-loop监督量或与历史ERM counts中的support_presentations=0直接比较。

SSH标准输入以内存载入新验证器，读历史缓存/预测并重放抽样，不加载模型推理，不训练、适应、访问目标或数据库；不部署文件、不覆盖原源码或旧verification。新verifier SHA绑定新证明。再次统计认证不改变S6 no_candidate，不推断因果。

## 输入缺口

本轮现场盘点及后续解锁条件详见[input-blockers.md](input-blockers.md)。IMDB完整support与P2同版本来源仍blocked，不用test补支持、`--allow-exclusions`或PG标签绕过。

## 最终验证与文档整合

- 最终全项目unittest：481项通过、44.860秒（final-tests.log），包含27项S7及15项论文导出测试。新增两个角色/稀疏方法及unsigned顺序/boolean accumulation负例通过；边界TDD失败/成功日志保留。最终改动文件Ruff通过，全项目仍110项原有问题（final-all-ruff.log），不借此批量修改其他历史代码。
- 独立只读S7复审确认两核心修复有效、原proof SHA匹配；追加发现的unsigned indices排序和accumulation类型边界按TDD修复。新源码再次对服务器原完整结果核验，最新SHA与本地一致、210文件前后不变；只读重验有新增代码原因，不是无信息重复运行。
- 两个写入隔离子代理交付24号S8完整draft和25号实验章/确定性导出。S8身份holdout公式、模型x774/h64尺寸和草案状态已实测核对，s8-plan-verification.json另存；仍not_started，未把新资源预算标approved。
- 论文导出器14项专项、19个派生CSV/SVG/JSON及章节逐字节重建通过；主代理再次实际CLI/临时副本验证70科学来源/原36来源、333历史文件未变、20输出字节一致（paper-rebuild-verification.json）。主代理视觉检查Genome及源取舍预览，当前SVG图例、全部K和两类SD标识可读；PNG仅工程预览，正式图为SVG。
- 24号draft的详细方法有限：层x/h参数类不同、只看线性probe，不认证纯压缩因果或原输入信息上限；19库面板限定、新F0额外训练成本与曾见H均披露。之后是否实施先冻结预算，不因草案完成而claim源侧已改善。
- 实验章是完整实验章草稿，不是整篇毕业论文或学校格式终稿。IMDB与P2仍由现场输入盘点及解锁条件约束，真实阻塞不改为completed。

所有上述测试为本轮实际执行，不沿用旧数。独立论文复审、最后来源/链接/Git核验随收尾记录；没有graphify更新。


## 文档与收尾状态

00号总览、23号待办、99号进度均已更新最新状态；24号协议与25号实验章各自区分草案/正式历史证据。原14/21/22协议/结果文档与旧结果保留不改，不在历史source binding上偷偷换哈希。代码、文档、图表与新验证报告只提交本轮分支，不切原共享检出，不混入其他窗口AGENTS/MEMORY草稿。

## 自检（不是科学或外部认证）

| 维度 | 分数 | 证据和边界 |
|---|---:|---|
| 准确性 | 4/5 | 481实际测试、完整源行/逐角色再认证及图表重建通过；无因果/显著性扩大解释 |
| 完整性 | 4/5 | 两工程缺口、受控协议、实验章图表均交付；IMDB/P2缺认证输入、S8未执行与整篇论文终稿仍未完成 |
| 清晰性 | 4/5 | 23/34最新入口和24/25分开；source单seed与target三seed指标明确，但历史长文仍需入口导航 |
| 可操作性 | 4/5 | 可用导出/只读核验脚本、测试、source hash；下一新GPU实验预算需冻结，不能自动通过 |
| 简洁性 | 4/5 | 主总结与数据表/原始日志分开；保留失败与调试记录便于追溯 |

平均4.0/5。不是“剩余全部研究已完成”；目前不能补齐的输入、真正未执行实验和终稿缺口逐项公开，用户可据此选择下一执行阶段。


## 论文独立复审及最后路径修复

[独立复审](independent-review.md)确认所有目标/源表聚合、SD、SHA和四图坐标正确；发现祖先symlink条件可覆盖任意指定output路径下同名历史文件，当前实际路径未受影响。主代理新增paper-path-red/green负例，修复mkdir前祖先拒绝和章节预检，15项paper专项通过（0.771秒），改动文件Ruff通过。原writer14项报告及旧代码hash保留为交付快照，不伪称最终版本；最终数值/输出确定性和源码hash另重新认证。

对拒绝全部symlink祖先的策略，临时测试在安全基目录显式resolve；CLI默认worktree ROOT已resolve。用户自定义路径若包含别名须提供真实目录路径，不静默follow链接。此检查不提供并发恶意文件系统race安全保证，正常本地受控工作区禁止别名覆盖的契约成立。


独立路径补丁复审通过；最终完整481项/44.860秒覆盖27项S7与15项paper，最新代码与测试SHA由final-verification.json绑定。旧writer export-verification.json保留为修复前快照，当前来源/输出仍相同，不能将旧code SHA当最终源码。行末规范化的工程展示日志保留同名gzip原始字节与SHA；失败、lint问题和原实验数字均不删除。
