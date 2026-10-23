# S8 G1 smoke 独立复算器交付

日期：2026-10-07。基准：`04561e8d320d98998cd05b96fc4c4ffb435e77ab`；接手分支：`feat/s8-g1-smoke-runner-20261007`。

本交付只新增 [复算器](../../scripts/s8_independent_verify.py)、[专项测试](../../tests/test_s8_independent.py) 和本目录 `independent-*` 报告。没有修改 runner/state/进度文件；没有切分支、提交、推送、网络/服务器/GPU、graphify 或委派。共享区其他未提交文件由其负责人处理。

## 1. API 与输入约定

```python
from scripts.s8_independent_verify import verify_smoke
result = verify_smoke(run_dir, bindings, panels, caches)
assert result['passed']
```

复算器只读，不输出文件。缺失文件抛 `FileNotFoundError`；非法/不一致证据抛 `ValueError`，不静默过滤失败项，不返回部分通过。

- `bindings` 使用主 runner/state 的 schema1 字段：`mode='smoke'`、`purpose='s8-engineering-smoke-not-research'`、`seed=42`、`train_steps=2`、`meta_batch=4`、`query_size=64`、`support_sizes=[5,10,20,50,100]`。
- 名单字段为 `train_databases`、`holdout_databases`、`validation_databases`，分别严格等于协议 T19/H4/V3 的顺序；不使用前期草拟的短字段名。
- `resourcefixedfields` 至少含 `batch_size=128`、`threads=4`、`probe_fit_per_database=64`、`probe_adam_steps=2`，均须是整数。额外资源/来源/model 配置可由主 runner/state 认证；本复算器不宣称运行时资源已经测量。
- `panels` 是 19 个 T 库到 unique-position 整数列表的映射。独立重算 `SHA256('s8-train-panel-v1|db|hash')`，以 `(sha, hash)` 排序取前1024，并比较完整列表及其顺序。每个 T cache 至少1224 unique；测试未缩小这项安全门槛。
- `panel_identity_digest` 为 `{db: [hash for position in panel_positions]}` 经 `json.dumps(..., sort_keys=True, ensure_ascii=False)` **默认 separators** 编码后的 SHA256。面板保持 SHA 顺序，不按 raw 行排序。
- `caches` 必须恰含26个源库，各库至少有 `indices`（原始首次行号的整数 ndarray）、`query_hash`（唯一非空字符串 list）、`true/pg/n_tables`（一维 ndarray，长度相同）。拒绝负值、非有限值、重复身份/raw行号及无效表数。文件字节/重复 token 来源认证仍由主 runner/preflight 负责。
- `predictions` 中五臂值为 **raw natural-log cardinality**，不是 residual 或原始 cardinality；没有输出裁剪。

## 2. 必需 artifacts

### `episodes.jsonl`

每行恰含 `step, episode, database, support_positions, query_positions, support_hashes, query_hashes`。

独立 `np.random.default_rng(43)`：按固定 T19 顺序均匀 `choice(database)`，按冻结 K 列表 `choice(K)`，再从

```python
pool = np.sort(caches[db]['indices'][panel_positions])
raw = rng.choice(pool, size=K + 64, replace=False)
```

抽样；通过 raw→unique-position 字典还原位置及 hash。完整8行逐项比较：`step=[1,1,1,1,2,2,2,2]`、`episode=1..8`。拒绝使用 P 的 SHA 排列直接随机抽样、使用 raw行号冒充 position、Q/H/V/target 进入训练以及任何身份错配。

### `endpoints.jsonl`

每行恰含 `database, cohort, query_hash, true, pg, n_tables, predictions`；`predictions` 恰含 `A_h_adam/B_h_ls/C_x_adam/D_x_ls/F0`。

身份选择只读 query_hash：

| cohort | 选择 | 总数 |
|---|---|---:|
| P | 全部19库的1024 SHA面板 | 19456 |
| V | 各库 `s8-bridge-v1|db|hash` SHA前256 | 768 |
| H | 各库 `s8-smoke-h-v1|db|hash` SHA前32 | 128 |
| 合计 | 完整 smoke endpoint 分母 | 20352 |

拒绝 extra/duplicate/missing/cohort 错配和 source true/PG/表数不一致；大整数值比较不先舍入成 float。全部行/全部臂检查 raw exp、Q exp、误差及平方是否有限，即使该桶 n<200 也不豁免。

### `metrics.json`

```json
{"rows":[{"database":"airline","cohort":"P","bucket":"all","n":1024,"status":"descriptive","arms":{"A_h_adam":{"q50":1.0,"q95":1.0,"raw_log_mae":0.0,"raw_log_mse":0.0}}}]}
```

上面仅为字段布局示意，**不是有效证据或实测值**；顶层可以带 `purpose='smoke-not-research'`，若出现其他 purpose 则拒绝。真实文档必须覆盖全部26库×`all/four_plus`的52桶，descriptive 桶包含全部五臂。禁止缺桶、重复、额外 bucket 或混用 pooled/macro。

- 标准 Q：`exp(abs(max(raw_log_pred,0) - log(max(true,1))))`；先校验未裁剪 raw exp，不因 Q 的标准 max(pred,1) 而掩盖溢出。
- Q50/Q95：独立排序、`(n-1)*p` 的线性插值，不调用生产 metrics，也不调用 NumPy percentile/quantile。
- raw MAE/MSE：相对 `log(max(true,1))` 的未裁剪误差，使用自然对数。
- n>=200：`status='descriptive'`，`arms` 中每臂恰含 q50/q95/raw_log_mae/raw_log_mse；数值比较 `rel_tol=1e-12, abs_tol=1e-12`。
- n<200：只允许 database/cohort/bucket/n/status 五个字段，`status='below_min_cell_n'`；包括 n=0，不报告详细指标或补 Q=1。

### `nomination.json` 与 `events.jsonl`

nomination 文件由 state 模块负责；本 checker 不排名，不替代 state 的 body/signature/development gate。读取原字节 SHA，要求恰有一次

```json
{"event":"nomination_sealed","file_sha256":"<nomination.json SHA256>"}
```

及至少一次后续 `H_semantic_load`，其 `database` 必须属于 H4，且含非空字符串 `hash` metadata，或者主 runner 实际使用的 `nomination_sha256`。后一字段必须等于已封存 nomination 文件 SHA；两个字段同时出现时都检查。拒绝先读 H、缺 seal/H 事件、重复 seal、file SHA 不一致、H 事件绑定 SHA 错配及缺 database/hash。验证结束再检查 nomination 文件没有变化。

这里只认证 **记录中的顺序**，不证明实际进程从未提前读取 H，不把 H event 的 hash metadata 称为独立源文件 SHA 认证；返回 `actual_H_access_order_verified=False`。完整四库 H 行仍由 endpoint 分母独立检查。

## 3. 返回值与边界

返回 `passed=True`、`purpose='smoke-not-research'`、`verified_counts`、`query_count=20352`、指标 units、panel/nomination SHA 和记录序校验状态。

始终保守返回：

- `formal_ready=False`
- `full_linear_optimality_verified=False`
- `source_file_provenance_verified=False`
- `actual_H_access_order_verified=False`

每 T P SHA前64用于小矩阵 fit，合计1216；full-P resource LS只记成本。复算器核验这项资源 binding 和 Pfull endpoint，却**没有 fit 权重、矩阵或 checkpoint 独立重载的认证输入**，因此不声称认证了实际 fit、F0 两次 optimizer 更新、population normalizer/oracle 或完整 probe 数值最优；这些由其他模块/主 runner验证。

本 API 不默认执行 formal4000 或正式 Q/H/V 全分母；formal bindings 明确拒绝。不会生成任何研究候选或提名文件。

## 4. 本轮实测验证

环境：`/opt/anaconda3/bin/python`；标准库+NumPy。以下均为**合成测试**，本轮未取得或认证真实 smoke artifacts。

1. [首次RED](independent-red.txt)：复算器尚不存在，ModuleNotFoundError，退出1。
2. [采样定义修订RED](independent-red-sampling-focused.txt)：冻结顺序/计数的合法 fixture 被旧实现拒绝，`episode exact RNG replay mismatch: row 0`，退出1。
补充 [runner接口RED](independent-red-runner-interface.txt)：新增实际 H nomination SHA / metrics purpose 的合法合成证据在兼容前被拒绝，退出1；兼容后完整复跑通过。

3. [最终GREEN](independent-green.txt)：23个测试方法全部通过，29.355秒，退出0；包含身份/采样/分母/全部五臂、200/199阈值、空桶、低n非有限/溢出、事件顺序/元数据/SHA、文件缺失/重复JSON字段、大整数舍入、独立插值已知值、无生产模块导入及标签不影响SHA选择。
4. [定向Ruff](independent-ruff.txt)：`All checks passed!`，退出0。
5. [合成API返回示例](independent-synthetic-api.json)：明确标注 synthetic unit fixture，不是实测 smoke、正式结果或研究候选；附复算器/测试文件 SHA。

实际最终命令：

```bash
PYTHONDONTWRITEBYTECODE=1 /opt/anaconda3/bin/python -B -m unittest discover -s tests -p test_s8_independent.py -v
/Users/williamlin/.cache/uv/archive-v0/Y8ucSZ-tmHCUuyMswFwm0/bin/ruff check --no-cache scripts/s8_independent_verify.py tests/test_s8_independent.py
```

全仓 suite、runner/state 集成、真实 CPU smoke、真实来源认证、GPU/服务器未执行；由主 runner 负责人按授权范围整合。未提交、未推送。

## 5. 自检

- 准确性4/5：专项测试和lint实测通过，但没有真实smoke证据可独立认证。
- 完整性4/5：本API约定内身份/指标/事件齐备；checkpoint/fit/来源/实际访问顺序明确留给其他模块，不能扩称全链路认证。
- 清晰度4/5：字段/采样顺序/计数/单位已明确；主runner尚需核对其输出与本契约完全一致。
- 可执行性4/5：函数和验证命令可直接调用；真实端到端集成由主负责人完成。
- 简洁性4/5：严格负例带来较多测试代码；未新增CLI、状态机、训练或研究gate实现。
