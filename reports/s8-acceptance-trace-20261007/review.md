# S8 验收追溯与待批准建议摘要

日期2026-10-07；基准a216a4b，同指定worktree。只读原文/源码与待审文档，不实验、不数学计算、不联网，不改生效协议。

## 推荐

仅G1完整P resource-only工程/数值稳定性分离，小面板旧pass及所有数据/求解/metadata/backend/参考objective/预测/normal/投影/资源硬gate原样保留。新engineering_verified不含rank_stability合取，但三谱观测rank必须一致；旧pass/solver_verified/unknown不改，strict数学证据仍unknown及scope。该变更会改变27/30现行runner停止条件，必须用户确认。

下一轮若确认，直接按33§5–6进入本地生产TDD（实际diff复核而非泛泛方法重审），不继续Fraction/高精度研究。只改指定数值返回字段、resource gate、新smoke版本签名/状态与独立记录核验；无正式GPU/训练入口。

## 关键追溯

- 24§6.3规定固定rcond/objective/normal；26§1.1已明确fixed-cutoff truncated LS不是原完整线性最优，有限可达J改进仍可观察。
- 27§1第11–13行把small fit和full-P resource-only分开，后者系数不参与预测/选择；27§5/§6依然要求数值失败停止，不能自行豁免。
- 30§1第4/7项与§3明确rank interval unknown阻止runner，故资源分离是待批准修订而非追溯解释。
- 30gamma/eta和v1.1 Gram门槛均为工程包络，不是严格真秩/子空间/最优性证明。严格unknown不能泛化成忽略工程失败。

## 独立只读复核

允许建议分层，不构成协议/执行批准。复核明确否决：resource-only外豁免、资源theta预测/选择、unknown改pass、三rank不一致/其它gate失效被容忍、漏成本/失败证据、追认旧smoke成功、越过V/H与完整分母。原24强问题不由工程通过回答，正式full-P四臂接受契约仍须另确认。

本轮独立代理未执行测试/数学复算，未读真实P/服务器，仅核原协议并写指定日志；是独立上下文AI，不是外部科学认证。主代理整合后33正文仅是待批准规范，AI同意不是用户同意。

## 当前状态与下一步

原G1 failed、formal_ready=false；full-P真实未认证，不把结构性unknown称为本轮真实failed；各unknown/failed/not_run见33§3。a216a4b未推送，当前远端只知道本地跟踪快照，不联网查。

本轮交付完整建议与明确批准问题：是否同意限定G1 v2仅full-P资源gate分离，正式强机制问题不自动解锁，然后下一轮直接本地生产TDD。不同意则全部现行gate保持，G1/正式阻断不变。真实P、smoke、正式训练分别待授权。

自检4/5：准确性（区分未运行full-P与包络阻断）；完整性（追溯/字段/函数/TDD/权限交付，仍待用户协议决策）；清晰性（工程与N/S证据分层）；可操作性（确认后直接具体TDD清单）；简洁性（正文详细，主摘要短）。平均4.0，不替代测试或用户批准。
