# S8 独立 state / nomination / metadata signature 交接

日期：2026-10-07。工作区：`/private/tmp/spatial-meta-ce-s7-diagnosis-20261007`。
基准：`04561e8d320d98998cd05b96fc4c4ffb435e77ab`；分支：`feat/s8-g1-smoke-runner-20261007`。

## 范围与状态

只新增 `spatial_meta_ce/s8_state.py`、`tests/test_s8_state.py` 和本目录 `state-*` 报告。
未修改现有 core / contract / probe；未训练、未执行真实 smoke、未访问 V 预测/H 语义/目标输入，未网络/server/GPU/graphify/递归委派、未提交、未推送、未切分支。
`formal_ready=false`；formal API 的合成单测不构成正式实验运行授权或冻结完成。
共享工作区中出现的 `scripts/s8_independent_verify.py`、`tests/test_s8_independent.py` 及 `independent-*` 报告不是本任务写入；未修改或纳入本任务验证声明。

读取 AGENTS、用户指定的委派记忆、项目记忆入口及 24 号 §8.2 / 26 号 H 顺序修订；最终字段按用户本轮 runner 冻结补充对齐。

## 主 runner 必须提供的 bindings

所有输入为 JSON-native dict/list/scalar，字段名与最新 runner 约定一致，不要求传 `T19/H4/V3` 键。

| 字段 | 必需值/边界 |
|---|---|
| `schema` | 整数 1，拒绝 bool/float |
| `mode` | `smoke` 或未来 `formal` |
| `purpose` | smoke=`s8-engineering-smoke-not-research`；formal=`s8-research` |
| `protocol_sha256` | 小写 64 位 SHA256；主 runner 传 27 号冻结文档的原字节 hash |
| `source_binding_sha256` | 小写 64 位 SHA256；主 runner 计算 preflight.input_sha256 的约定 digest |
| `panel_identity_digest` | 小写 64 位 SHA256；主 runner 计算每 db 原始完整 P 哈希列表 digest，不使用 fit 子集替代 |
| `train_databases` | JSON list，精确等于导出 `T19` 的顺序和内容 |
| `holdout_databases` | JSON list，精确等于 `H4`：accidents/grants/employee/hockey（hash 顺序） |
| `validation_databases` | JSON list，精确等于 `V3`：chembl/legalacts/telstra |
| `model_config` | 必需 arch=sum,width=64,use_pg=true,pg_output_scale=1；可附 layers=2,heads=4；拒绝其余模型键 |
| `seed` | 整数 42 |
| `train_steps` | smoke 整数 2；formal 整数 4000 |
| `meta_batch` | 整数 4 |
| `query_size` | 整数 64 |
| `support_sizes` | `[5,10,20,50,100]`（精确顺序和整数类型） |
| `resourcefixedfields` | smoke 必需，精确六项：batch_size=128,threads=4,probe_fit_per_database=64,probe_adam_steps=2,timeout_seconds=1800,address_space_bytes=12884901888；均为整数 |

其他安全 JSON 辅助字段允许，全部参与 signature；非有限值、非 JSON 类型、循环引用、未知辅助数据库标识，以及 target 数据库标识（包括路径中的独立 target 名）拒绝。辅助字段被绑定不等于其语义/科学正确性获得认证。

本模块不读取上述 SHA 所指向的文件、不重算实际 panel 身份、不运行资源测量。
`resourcefixedfields` 是 smoke 配置一致性检查，不是实际 CPU/内存/预算遵守证明；Pfull=19456 的 fixed-cutoff LS 数值路径计时和峰值、F0 trusted pickle SHA-before-load、optimizer step=2、拒绝恢复/旧权重均由主 runner 负责。

## API

### `signature(bindings) -> str`

校验边界后返回整个 bindings canonical JSON 的 SHA256；JSON 使用 `sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False`，UTF-8，无尾换行。

### `select_candidate(metrics) -> dict`

仅 V 开发 gate。输入结构：

```python
metrics[arm][database][cohort] = {'n': int, 'q50': number, 'q95': number}
# arm: A_h_adam, B_h_ls, C_x_adam, D_x_ls, F0，完整五项
# database: chembl, legalacts, telstra，完整三项
# cohort: all, four_plus，完整两项
```

每项 n>=200；每 db/cohort 五臂计数一致；four_plus<=all；Q 有限且 >=1，q50<=q95。
完整性/数值工程错误抛 `ValueError`，不伪装 `no_development_candidate`。
逐库等权 4+ Q95 macro 对 A/F0 同时 <=0.9，逐库 all/4+ Q50/Q95 对 A/F0 同时 <=1.1。
符合者按 macro、B/C/D 顺序排序。返回 `decision/candidate/macro_four_plus_q95/eligible_candidates`；无候选时 decision=`no_development_candidate`、candidate=None。

汇总数值/相同计数不证明实际 query identity 配对或 LS 数值认证；主 runner 须事先认证这些，不能只拿本 API 当完整端点核验器。

### `seal_nomination(path, metrics, bindings) -> dict`

smoke 完全不检查/序列化 metrics、不调用 selection：decision=`smoke_not_selection`、candidate=None、metrics=None、selection=None；绝不称作 no-candidate。
formal 才调用 V selection。
返回并保存字段：`schema/mode/purpose/decision/candidate/bindings_sha256/bindings/metrics/selection/body_digest`。
`body_digest` 绑定除自己以外的整个 canonical body，包含全部 bindings 和 formal metrics/selection。

父目录必须预先存在；逐级目录 FD + `O_NOFOLLOW`，拒绝任意层 symlink 和 `..`；最终文件 `O_CREAT|O_EXCL|O_NOFOLLOW`、权限 0600，文件与父目录 fsync；拒绝覆盖。
`O_EXCL` 原子占用文件名，并不保证读者永远看不到写入中的部分字节；部分文件无法通过完整 digest 认证。I/O 失败 fail-closed，可能留下不可重写的无效文件；不自动重跑/覆盖。
macOS 的 `/tmp`、`/var` 别名本身是 symlink，调用者应先建立并传入无 symlink 的真实绝对路径（例如 `/private/tmp/...`）；本模块不自动 resolve 来绕过拒绝。

### `require_nomination(path, expected_file_sha, bindings) -> dict`

依次认证：完整文件 SHA、canonical JSON、无重复键、body_digest、完整 bindings/mode/purpose 和重新构造的全部 payload；formal 重算 V gate、smoke 再确认不是选择状态。缺 seal、source/protocol/panel SHA 不符、异 mode、任何篡改都拒绝。
必须先通过此函数，主 runner 才能读取 H labels/tokens 语义。`expected_file_sha` 必须在 seal 时锁定并另存，不能在 H 加载时从潜在已被修改的文件重新取 hash。body_digest 是冗余一致性证据，不是密钥签名；若攻击者同时控制 seal、bindings 和外部 pinned SHA，本 API 不能提供真实性证明。

本轮只实现 V 锁定/读前认证，不实现 H gate，不负责 H 接受/否决后的 complete 报告状态转移；runner 须维持其固定报告，不允许根据 H 换候选。

## 真实验证与 TDD 记录

- `state-red.log`：首次 red，专项退出 1，因为新增模块尚不存在（ModuleNotFoundError）。
- `state-green-initial.log`：首次实现后 26 项中 4 个 subtest 报错，来自测试夹具 Q95<Q50；修正合法夹具，未放宽生产验证。
- `state-runner-contract-red.log`：主 runner 更换字段及资源冻结契约后，先改测试，31 项退出 1（14 errors）；实现尚要求旧字段，另有新增循环/溢出边界测试。
- `state-resource-initial.log`：对齐实现后 31 项退出 1（10 errors）；辅助数据库键检查误把 `probe_fit_per_database` 数量当数据库身份；修正为数量键不是数据库标识。
- `state-green.log`：最终 31 项通过，退出 0，0.114 秒；包含门槛等号、macro 非 pooled、双参考保护、tie/no-candidate、缺臂、稀疏、NaN、配对计数、source/mode 变更、overwrite/symlink/父目录替换竞态、篡改-before-H 认证。
- `state-ruff.log`：只检查两个新增 Python 文件，退出 0，`All checks passed!`；使用 `--no-cache` 避免新写缓存。单处 `noqa: TRY004` 保留公开 ValueError 契约，不改异常类型。
- `state-stdlib.log`：Python `-S` 隔离验证通过，无 torch/numpy 导入；AST 只读比对现有 contract 的 literal T/H/V 一致。
- 仅专项，未 full suite；未执行实际主 runner 集成/真实 smoke/LS 资源测量。

最终源码/测试 SHA、命令/退出码见 `state-verification.json`。日志与本报告是工程合成证据，不是研究精度/训练/资源结果。

## 自检与剩余边界

自评：准确性4/完整性4/清晰性4/可操作性4/简洁性4，均值4.0。依据：真实专项、lint 和隔离检查通过；但尚无主 runner 集成/真实烟测，身份/LS/Hgate/checkpoint/resource 实测明确不属于此模块交付；上述来源 digest 生成细节和完整报告由 runner 接续。改进优先级：主 runner 用实际冻结 bindings 集成调用，再做获得授权的真实 G1 smoke；不以单测通过冒充 formal_ready。
