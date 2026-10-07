# G1 engineering v2 本地生产TDD报告

日期2026-10-07；基准603611e；分支feat/s8-g1-engineering-v2-20261007。用户明确批准33§8，仅完整P resource-only工程gate分离，非正式研究认证或训练授权。

## 已实际完成

- 先新增34已批准新版本协议，同时保留旧27/30原字节。bindings schema2/policy/旧protocol SHA/新acceptance SHA签名贯穿checkpoint、seal、H入口；schema1旧语义与正式旧gate保留，不能混搭/把v2用于formal。
- REDcheckpoint5f3b829：主线probe6错误、runner真实2新行为错误（另保留临时路径别名fixture失败）、checkpoint新schema拒绝；隔离state45含旧31通过、独立verifier42含旧23通过。GREEN前生产未改；原始日志保留。
- probe新增engineering合取、三观测rank和严格unknown范围，旧pass/solver_verified及全部门槛保持。小面板还是旧pass；唯一gate改变在full-P resource-only。
- runner每阶段存完整证书（失败也保留）、profile分标准化/solver认证时间、stage/completed_stages、schema2新状态completed_engineering_smoke。资源theta丢弃，预测仍仅用small-panel系数；formal_readyFalse/smoke_not_selection/strictunknown。
- state.signature仅schema/policy/SHA最小分支。checkpoint外层序列化schema1仍是原格式，bindings schema2区分新验收契约；hash先校及optimizer/模型原检查不变。新checkpoint测试只是手工合成optimizer端点状态，不是F0更新证据。
- 独立verifier不调用生产probe/solver；新增完整数值记录、四证书文件一致性、hard布尔合取、三rank与旧pass/stability一致、profile尺寸/bytes/时间/计数/strict范围。这是record consistency，不是重新求解或严格数学证书。原身份/8episode/20352端点/52桶/H顺序/来源范围说明保留。
- 主代理复核发现uppercase SHA、spectrum/reference-pass/分解计数字段一致性缺口，补3个RED负例后最小修复；原始RED/GREEN均保存。未引入新数值阈值或放宽check。

## 本地真实验证

- probe旧32+新6=38通过，旧numeric20通过；state45通过；checkpoint1通过；runner v2 mock5通过，旧拒绝2通过。新runner完全mock训练/source/checkpoint/solver，却实际生成20352合成endpoint校验资源theta不泄漏，不能当真实smoke。
- 完整unittest **669项通过、88.700秒、exit0**，空CUDA、单线程、授权沙箱外（既有OpenMP/信号子进程受限）。测试含既有合成优化/模型回归，不当成真实P或F0实验；未运行run_s8_smoke --execute。
- 四生产文件+相关测试改动Ruff通过；全库Ruff仍110项历史问题，未扩大修复。RED日志原始subTest行尾空格不修饰；代码/Markdown diff check单独通过。
- 首次独立实际diff复核发现完整证书数值/shape证据可缺失或与True旗标冲突，拒绝整包。经5项RED（94失败子例）→5项GREEN的记录层最小修补，原49专项通过；复审已关闭P2，最终日志批准待归档。本轮不是泛泛方法或数学研究，不继续Fraction，不部署/联网/真实P/目标/GPU/smoke。

## 尚未完成／下一轮最小授权

原G1 failed、formal_readyFalse，真实P新版核验与新G1未执行；工程E是否在实际full-P通过未证，严格秩/子空间/最优仍unknown。正式G2–G4验收/runner/预算/科学解释和GPU授权未完成。

1. **真实P只读请求**：允许myserver读取旧P-h/x/label/pg_log四数组及status/contract/source provenance必要元数据，核solver-diagnosis原SHA，固定来源/身份重建P-only面板与normalizer；新执行包/新proof目录，旧文件/code/status前后不变。CPU最多4线程、空CUDA、wall15分钟、RLIMIT_AS12GiB、父监督RSS/exitcode/硬截止。仅LS及证书重算，zero-reference用于resource-only工程核验，0真实F0/Adam更新、不load旧checkpoint、不读H/目标。若另核小面板零参考，只能称诊断，不冒充原2步Adam probe。任一来源/数值/资源/超时拒绝即停，一次运行不fallback/改cutoff/重试；报告E/stability/strict各状态。
2. **独立新G1请求**：若只读P来源可认证且用户另授权，部署冻结新执行包（不可覆盖旧failed/code）并新目录单次CPUsmoke；wall1800秒/AS12GiB/最多4线程、batch128、emptyCUDA，freshF0 seed42两更新/8episode，每small Adam2更新，small旧gate/full-P工程gate，全seal/H入口与20352/52桶独立闭环、源码/输入前后SHA。父监督超时/OOM/非finite/工程hardgate/来源变化立即结束自有进程组、failed留痕；SIGKILL/未写status以exit证据记失败，不auto resume。即使completed_engineering_smoke也strictunknown/formal_readyFalse，不能正式训练或提研究候选。

这两项都还需要明确执行授权；本轮不发生服务器访问、部署或真实实验。预算是硬上限而非完成时间保证。AS不是RSS，RSS持续记录但不声称AS直接实现硬RSScap；不得把matrix_bytes当峰值。正式GPU/完整训练另议，不能复用旧partial F0或resource theta。

## 本轮Git

RED已本地提交5f3b829；GREEN与最终报告准备在验证/复核完成后只提交本任务文件。用户禁止联网，未推送/ls-remote，不能把本地跟踪快照冒充远端实时状态。


## 恢复收尾后的最终回归（本轮实际）

恢复HEAD5f3b829及全部未提交工作，先核669snapshot四源码SHA全部匹配；669/88.700日志保留历史，verifier修补后不沿用其结果。新增完整46字段/shape/dtype/scope/finite/旧数值门槛记录一致性，仅工程记录，不重跑求解证明真实谱。原3其它生产模块和34协议SHA未变。

完整新回归 **674项通过、90.850秒、exit0**，空CUDA/单线程/授权沙箱外；改动Ruff通过，全库仍110旧问题。source与testSHA重新绑定final-verification.json，最终只读独立日志审批后再本地提交。S2审查与P1独立事项均未进入G1修改/提交，不将S2审查当G1批准。

尚未执行真实P/新G1/正式训练，旧failed/formal_readyFalse/strictunknown原样。下一只读P脚本尚未部署或运行，须另授权其编写/冻结/部署/有界执行；不能用run_s8_smoke替代只读核验，因为该入口会fresh训练。新G1仍单独待授权。


## 最终G1独立复审批准

同一G1只读审查代理（非S2审查）核实际diff、新674回归90.850秒、改动Ruff及全库旧110诊断字节，整包本地工程批准、剩余代码阻断0；[最终批准日志](independent-final-approval.log)绑定最新源码/测试/三原始日志SHA。上轮P2和中途审查保留。日志中的final-verification旧669/SHA提醒是读取时间差，主代理已核最新JSON为674及修补后SHA，最终一致性再次确认。

代码已测试、已独立复审；GREEN准备仅本地提交，未推送、未部署、真实P/新G1仍待授权。自检准确性/完整性/清晰性/可操作性/简洁性各4/5，具体未闭合项为真实源矩阵及实际全链/资源与正式科学认证，不用自评替代这些门槛。


## 实际本地GREEN提交

`c8d1d4e` 已成功提交本任务GREEN与完整证据，提交后worktree干净。RED checkpoint为5f3b829；本段和JSON提交状态另做文档收尾。没有push/联网/远端实时核对，没有merge main或server部署，S2 P1未纳入。所有生产/测试SHA与最新674回归及G1独立批准快照一致，无需因文档收尾重跑数值测试。
