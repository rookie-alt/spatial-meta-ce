# S8 审查与下一阶段工程记录

日期2026-10-07，基准19470ee，分支feat/s8-review-and-preflight-20261007，独立worktree `/private/tmp/spatial-meta-ce-s7-diagnosis-20261007`。用户授权先审查，无问题再下一步。本轮发现实质方法漏洞，修订后推进有限G1工程；未把原草案写成无条件通过或直接启动长期GPU。

## 分工与归属

主代理实现源身份/面板契约、表示捕获、metadata预检、文档和整合；一个写入隔离子代理只实现s8_probe与test_s8_probe及probe-*；另一个只读子代理方法及source/hook复核。没有递归子代理、同文件并发写入或更改原主检出其他窗口AGENTS/MEMORY。子代理是AI独立上下文，不是外部科学认证。

## 审查发现与修订

- 原24号fixed-cutoff LS不保证原线性类最优。标准库两行反例Z=[[-1e-14,1],[1e-14,1]],r=[-1,1]可使截断J1、合法系数J0、正规方程检查仍过；对比有限Adam不足以认证最优。
- 主代理追加fullrank病态伪造反例（系数1e-11不被截断），仅normal/objective对Adam仍可被骗；模块追加同cutoff重解的objective/预测核对。rank限定与fullrank数值状态分开，full_linear_optimality*永远False，保守不认证独立科学最优。
- 明确population std/actualscale/zero/tiny masks；P-only归一化，不应用诊断masks修改留出数据。
- 提名前H只身份、offsets表数/count/SHA；H不走read_cache（否则会取标签）、不进行token值检查或预测。全部源数据字节SHA的流式计算仅认证版本，不作基于H误差的选择；H内容验证在提名锁定后仍待实施。
- preflight入口强制S6 reference原字节SHA，拒绝“自洽但来源未冻结”的用户提供JSON在路径/cache读取前绕过授权；26源集和7文件白名单完整唯一。
- 正式训练batch/backend/checkpoint/F0-P-only监督、不可回写提名、监督器及实际资源probe尚未完成。合成/metadata验收不能转formal_ready。

完整审查修订见[26号](../../实验文档/26-S8方案审查与G1工程预检.md)；24号旧草案字节不变。

## 本轮真实验证（不沿用旧481项）

- contract-red.log先复现缺模块，contract-green.log五项通过（0.085秒）；测试验证固定身份/禁止H/V/目标训练面板、P/Q身份置换不变、eval原forward/hook exact及异常清理。开发时合成sample遗漏id、返回inference tensor被测试暴露，随后修正fixture并在inference context外clone；不是改历史模型。
- preflight-red/token-red/reference-red.log先失败；preflight-green.log七项通过（0.062秒），含重复token冲突、非有限/负fanout、H无cards/groups时仍只做身份count，以及未冻结reference拒绝。
- probe-fullrank-green.log三十二项通过（0.867秒）；solver TDD及数值反例详见[probe-review.md](probe-review.md)，包含input/certification tamper/overflow/rank/normalizer/Adammoment等负例。两次lstsq加一次SVD的新路径成本须正式profile，不把tiny测试耗时外推。
- local-tests-sandbox.log：525项、43.848秒，两项旧进程监督测试因子进程未启动失败；日志保留，未把失败包装成通过。沙箱外相同测试的实际最终结果另行补记录。
- 初次真实只读source metadata预检source-preflight.json/log：21.181秒、峰值RSS463668KiB，182源缓存+reference183输入前后SHA一致；T/V全groupfinite/shape/非负fanout、20782重复token对通过，H仅identity/count/SHA。P19456/Q910258，T929714/H200000/V149504；不是新模型推理或SQL计数。
- 对source入口reference认证等最后代码硬化，最终server proof须绑定最终源码，再认证执行结果另记录，不能沿用初次代码SHA冒充最终版本。

## 明确范围

只读server preflight经必要的沙箱外网络审批、300秒/8GiB限制、CUDA_VISIBLE_DEVICES为空；代码经stdin在内存执行，不部署文件、不覆盖PRICE或项目历史源码/结果。本地Adam调用均tiny合成测试，不是源模型训练；没有实际F0、probe正式面板、适应、目标读取、GPU任务或数据库计数。没有graphify更新。

G1模块/metadata已推进，**G1真实完整smoke及G2–G4 not_started/formal_ready=false**。下一步应落实完整P-only训练runner/checkpoint、提名/H访问状态机、独立端点证明及真实smoke/资源后再谈正式GPU，不凭此报告直接启动4000步或改变既有门槛。


## 最后硬化与新版实际证明

首轮source/hook复审发现H inspection统计解析、目录替换、重复表数冲突及hook注册清理问题，负例先失败后修复。新identity projection仅解码允许字段，不解码H inspection/provenance值或重算其semantic digest；缓存所有语义读取经逐目录FD与O_NOFOLLOW（文件/目录）保护。新source metadata全部输入绑定固定reference，H标签/token后验检查仍待提名阶段。目录FD保持已认证路径对象，版本稳定性仍靠文件前后SHA；不宣称任意恶意文件系统的全隔离或任意JSON完整语法认证。

旧source-preflight-initial.json/log保留：初次21.181秒及RSS只属旧实现，旧“未读H标签数组”不能扩大为从未解析标签范围。新版source-preflight.json/log正式重验23.650秒、RSS525856KiB，183输入前后SHA一致、同P/Q/T/H/V分母，H_inspection_values_decoded=False，source/hook最终代码SHA绑定。没有新训练或推理、不写服务器。新的预检是源码/语义硬化后的新核验，不是无新增信息机械重复。

独立审查原四发现均闭合，18限定测试通过；主代理校正一个转义JSON测试夹具后50项S8专项通过（0.908秒），生产代码不再变动。最优性/实验状态等标签保持保守。完整回归、最后Ruff及Git同步后补实际结果。


## 最终本轮验收

- 最后完整unittest 531项通过、45.866秒（local-tests-final.log），含S8新50项（contract6/preflight12/probe32）。沙箱内旧进程监督2失败、沙箱外525及后续531的阶段日志均保留；未删失败或借旧481数称新验收。
- 新增/改动Python文件Ruff通过（final-ruff.log）；全项目仍110项既有问题（full-ruff.log），未进行相邻代码批量格式化。
- 24草案、25实验章、历史results及model/train/data/checkpoints以19470ee原字节逐一核对，绑定历史输入清单；source proof与source/hook最终SHA、probe报告与最终数值源码SHA一致。
- 本轮新增26号审查修订，并更新00/23/99的新阶段入口与阻断，不把G1模块/metadata、tiny solver测试写成S8正式训练或精度结果。没有GPU任务、目标评估、数据库count或graphify更新。
- 默认CLI真实执行输出planned/default-plan.json，不称默认打印为执行成功；execute仍只读取源metadata。

## 自检（不替代独立认证）

准确性4/5：531实际测试、新source实测及两轮只读复审闭合已知缺陷；最优性标签保守False、真实SQL语义未知。完整性4/5：本轮审查与下一G1推进完成；freshF0/提名状态机/真实smoke及正式预算未完成，不能扩大为正式实验完成。清晰性4/5：26和35节区分initial与最终proof、数值/机制/GPU状态，旧文档仍较长。可操作性4/5：可执行preflight和模块/测试可复用，完整正式runner和资源proof仍需下一工程阶段。简洁性4/5：结果摘要与日志分开，安全输入和负例必要导致模块有一定篇幅。平均4.0/5。下一优先实现P-only runner/签名/提名与独立端点证明，而非继续扩seed或loss网格。

用户可核对这些边界；本自检不是科学质量评分或收益预测。
