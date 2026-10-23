# 2026-10-08 真实 A 失败记录独立审查

审查代理Dalton（01a116eb-3ae6-7bd1-9a74-58f419e4979e），本轮只读本地实际proof/log/source并独立计算SHA；未联网/server、未运行checker/LS/test/训练、未写文件、没有独立数值复算。主代理将原审查结论归档于此，不将自身检查冒充代理动作。

## 结论

**本次失败记录可信且内部一致，可以作为“一次已执行、在启动握手阶段失败的A尝试”归档；不能认证A工程核验通过。** 当前有1个启动路径绑定缺陷阻断后续成功执行；原G1failed/formal_readyfalse/strictunknown保持。

proof/supervisor/日志一致：attempt1，无retry，child2/signalnull，parent/outer/SSH1；监督childwall0.1512896628秒，不是部署/预检/SSH/恢复端到端时间。五个新proof文件，无worker-proof、panel证书、runtime/access-log/worker-resource，四panel均not_run/engineeringnull，缺失不是通过。必要字段和阶段状态完整，内嵌supervisor吻合独立文件，没有伪填数值或成功。

## 根因与最小边界

36号与execution-command传父contract相对路径；main给child改绝对路径；verify_worker_handshake却逐字比较parent argv与child绝对路径，SHA/output匹配也准确触发`Supervisor arguments do not bind this worker`。失败在worker()前，非LS/rcond/标签/矩阵。此前本地批准未认证Linux实际握手，本次暴露集成缺口。

最小后续修订：统一父子contract路径规范化规则（依据parent cwd），仍严校固定package/contract/output/SHA/pipe；不能删除或放宽握手。补Linux相对/绝对同文件正例、错误cwd/错文件负例，更新规范命令、另冻包和新隔离目录。rcond/数值阈值/LS/Fraction/训练/访问范围不变。本轮一次授权已消耗，即使未读P也不视为尚未尝试；再次A须另获授权，当前不能用A作为B前置通过依据。

## 来源与读取证据强度

代理独立核：runtime预检与post inventory的8SHA清单相同，吻合本地/manifest；五个proof本地副本SHA及大小吻合server inventory；contract与授权/manifest/proof同绑定。runtime通过只认证其记录范围部署环境，未认证P或数值阶段。

冻结源码顺序、握手错误、空输入记录共同支持未进入八旧run输入读取路径。但预检/inventory空列表为脚本声明，worker audit guard尚未安装，无独立OS I/O审计，**不能称系统级零读取证明**。input before/after空、opened_input_paths空表示未建核验记录；original_inputs_unchangedfalse是未认证，不是发现变化，也不是已现场rehash证明不变。

## 审查实核SHA

- entry：0f781f3500ea438eebe5c1d4962da0c454be3826a81c85df75412d71f97fcf35。
- contract：5d1ea6f4c9cd0f061f3db9ad5d9d7a88485485a88a169c213ddd7a25ed7e8650。
- proof：99d088474e0d35a3f5fb3510bc0f09c6cf1d27ae08dd56545a4baf212925a6da。
- supervisor：be5e36c039e4a8aa640be4e11b52e947f72c8fee018a038361a0972162006b2d。
- child stderr：0d999801aacfad578367a3abfb9645984b693e985ce6d04b2ec39c859623f968。

审查时实际HEAD3047494f3a301bf78d56f71bf0873fe7e289f96b。此独立审查接受失败记录，不授予执行或修改权限，不替代新的数值proof。
