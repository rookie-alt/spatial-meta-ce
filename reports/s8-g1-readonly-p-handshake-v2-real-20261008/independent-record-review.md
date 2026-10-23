# 新版A真实P失败记录独立只读审查

日期2026-10-08；审查代理Euclid（01a11a06-5495-7a93-b21d-540372bb4a0f）。代理未写文件、未联网/server、未重跑checker/LS/tests/训练、未递归。结论来自原始proof/log/源码与SHA交叉核验，不以主代理execution-result.json汇总为依据。

## 总体结论

**新版A记为failed，不能进入B。** 失败不是资源超限；直接失败点是x small 1216×775工程证书中的`retained_subspace_pass=False`与`reference_retained_subspace_pass=False`。h small与h full engineering通过；x full未执行。原G1 `failed/formal_ready=false/strict=unknown`不变。

## 执行次数与绑定

审查开始HEAD=1f411db7b91115d8b9aa1a91fc05446e0b1daa1f，与授权基线一致。transport记录：目录检查、上传/解包、runtime预检、真实checker、回收、post inventory均各1次；checker入口实际仅1次，无retry/resume。preflight与inventory是独立部署/证据清点步骤，不是checker重跑。

八输入before/after均一致且等于contract冻结SHA。设计上每个engineering阶段通过后重读八输入更新after；h small/h full通过后已有完整8项after，x small失败后立即停止，没有失败后的又一轮全量重哈希。因此“before/after一致”成立，时间点是最后一次成功阶段后的认证。

access-log去重后旧run路径恰好八文件；source_paths仅新冻结包成员；runtime_open_count=1243是Python runtime/venv相关open计数，不是额外数据路径。preflight和post inventory均未打开旧P。该口径是Python audit hook，不是OS级I/O trace，不能等同OS sandbox证明。

preflight/post/本地worktree的9个部署成员SHA完全一致；archive、contract、entry关键SHA均为冻结值。15个新proof文件本地字节与服务器inventory逐项一致。proof与worker-proof除父进程新增supervisor和contract SHA字段外语义一致，无篡改迹象。

## 阶段结果

- h small：shape1216×65，rank46，engineering true，rank stability stable，strict unknown；只是zero-reference诊断。
- h full：shape19456×65，rank50，engineering true，rank stability unknown，strict unknown；resource-only通过，系数不得用于预测/选择。legacy pass false只因旧pass要求stable，不是资源失败。
- x small：shape1216×775，rank586，engineering false。metadata/normal/objective/reference objective/reference prediction/full spectrum/SVD backend/observed rank agreement均通过；失败字段是retained_subspace与reference_retained_subspace，进而reference_pass/engineering/pass false。
- x full：not_run，engineering null，rank not_run。

x small直接根因是固定`rtol=1e-12, atol=1e-12`的retained-subspace projection检查中，reported LS与reference LS预测均未匹配SVD retained projection。775列/数值rank586的弱子空间问题可能导致该不一致，但证据未记录projection residual范数且系数已丢弃，不能量化误差，也不能断言LAPACK分支差异、conditioning、截断边界或容差本身是具体原因；不得把该推断写成事实。

## 资源与VmRSS

supervisor：returncode1、signalnull、wall4.25075秒；采样峰值RSS501362688 bytes≈478MiB，AS3816312832 bytes≈3.56GiB，均低于12GiB。worker ru_maxrss501633024 bytes≈478MiB。`resource_observation_failed: 'VmRSS'`发生在child poll后读取`/proc/<pid>/status`期间，child可能已退出导致字段缺失，是fail-closed观测竞态，不是RSS/AS超限，也不能改写为成功。RSS采样非硬上限，AS≠RSS，matrix_bytes≠峰值。

## 状态与最小下一步

proof明确formal_ready=false、original_G1_status=failed、can_retroactively_pass_original_G1=false、real_smoke/F0/Adam/H/target/checkpoint/GPU均false。原状态不变，B阻断。

最小下一步：保持本次A v2 failed，不重试/resume/自动申请B；完成证据归档和本地提交。不改RCOND、数值阈值、LS、Fraction或科学gate。若用户未来明确要求解释x small retained-subspace失败幅度，只能另行设计diagnostic-only、单次授权、只新增记录字段的方案，例如记录projection residual范数与相关诊断；不得通过调整阈值让失败消失。该诊断不是进入B的必要条件。

## 审查实核关键SHA

- archive：128d1b3c164812152e43bf13e712149db897a9e347903d7ff0cd5c462b456fb2
- contract：35a7ad5fa21374b33a14f13745af8b297d7b857d7cc741eb548a05c900a67b96
- entry：5320ffe926cf7b56ddf2970b7888f79391e18fb62cf50298127e4eb2af8e5471
- proof：1719d7472ea0cfcceb9a10e463846f07228f52885dc9c8705ec23b40e81cb5a3
- worker-proof：9d733b2263e02fbb77b3f77f459c809c261a6227a6b7009c57073a6d1ae18c8d
- supervisor：a4de9deda42d23b459e5fdd001b071b2383d33f0ac953eaff1889b452366a16f
- runtime：4b24a6f7ddb505cafa14475742ea44ca8138e56a4945a9970929792dc8f02050
- access-log：316d348041f07157c1fdc6f12af5ca1053cbd577809e816a06743251bd9d2463
- worker-resource：9d64b7baf7cd4535d2d12f3c482b07cc01c1074b77c656c3855df95f132def88
- x-small stage：db2efe4d43d301467089693d96e94f70e3a28b5bb43453d8b63e8abb3c5cf49f

审查结论：**failed归档可信；B阻断；最小下一步是停止并归档，不改数值gate。**
