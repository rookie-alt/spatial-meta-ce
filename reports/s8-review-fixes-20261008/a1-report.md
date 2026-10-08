# A1 配对评估最小修复交接证据（2026-10-08）

## 范围与状态

- 指定共享独立 worktree：`/private/tmp/spatial-meta-ce-s8-review-fixes-20261008`。
- 初始现场核对：分支 `audit/s8-review-fixes-20261008`，HEAD `b3390b7463d0dd0fc002903f98214b68c696a11b`，初始工作区干净。
- 本轮仅编辑下列三份代码/测试及本目录 `a1-*` 证据；未创建 worktree/子代理，未提交/推送。
- 未访问服务器、GPU、真实目标文件，未执行训练、真实 smoke/P 或模型推理；`test_paired.py` 的 StorePool、预测、模型、adapt、latency 均 mock，forward 如被调用会直接报错。
- 共享工作区后续出现 A2/A3 改动；未编辑、覆盖或纳入本证据的 A1 source hashes。
- 本 worktree 缺 `memory/spatial-meta-ce-agent-delegation.md`；按用户本轮明确禁止委派执行。
- 完整 unittest 由主代理最终复跑，本报告不将中间全量运行写为 A1 全量验证。

## Changed paths 与最终 SHA256

| 文件 | 改动 | 最终 SHA256 |
|---|---|---|
| [evaluate.py](/private/tmp/spatial-meta-ce-s8-review-fixes-20261008/spatial_meta_ce/evaluate.py) | loader、配对指标、all-rows/unique 调用与参照定义 | `1b637fcf3a45b5fe2fedc2857d6a2017e1c77a26b183cf684155f10767f2296f` |
| [test_paired.py](/private/tmp/spatial-meta-ce-s8-review-fixes-20261008/tests/test_paired.py) | 原两项协议测试改为 mock；新增数值、标签及两个 scope 的顺序绑定测试 | `22855b754aa432720bfc7d1d067c70c3be18fd4c8a7bff30380476b1bde2672c` |
| [test_paired_metrics.py](/private/tmp/spatial-meta-ce-s8-review-fixes-20261008/tests/test_paired_metrics.py) | 合成指标/loader 回归（新增） | `50eb34ee8491bc0b583e67b669e96d6dfffcf4a99be76e1777c97167a2d0a4b9` |

原有 `spatial_meta_ce/metrics.py` 未修改，unique 查询选择及 `summarize` 普通 Q95 未改变。最终 SHA256 同时绑定在 [a1-recompute.json](/private/tmp/spatial-meta-ce-s8-review-fixes-20261008/reports/s8-review-fixes-20261008/a1-recompute.json) 与 [a1-static-checks.log](/private/tmp/spatial-meta-ce-s8-review-fixes-20261008/reports/s8-review-fixes-20261008/a1-static-checks.log)。

## 修复的定义及身份契约

`evaluate.py:77` 的 `_reference_predictions`：

1. 有 `pg_prediction_path` 时必须显式 `metadata.method == 'pg'`，不以文件名猜测；缺失、null、空方法或非 PG 都拒绝。未提供 reference/path 仍不做配对，保留原可选接口。
2. CSV 必含 database/k/query_id/query_hash/true_cardinality/log_prediction，仅取所请求数据库的 K=0 行。
3. 身份是 `(query_id, query_hash)`，不只 `query_id`。相同 ID 不同 hash 各自保留。相同复合身份的多次 K=0 记录仅在 raw label 与 prediction 完全相同的情况下允许；冲突直接失败，不用 first-wins。
4. 空/空白身份、非有限预测/标签、负标签、缺列和缺失覆盖均拒绝。reference 标签与 `FeatureStore.cards[indices,0]` 原始 float64 标签精确相等后才返回。
5. 返回 prediction、raw labels、有序复合身份；按实际 store 顺序重排。all-rows 只加载一次，unique reference 从已绑定数组按原有 unique positions 切片，避免两个 loader 各自去重错位。

`evaluate.py:123` 的 `_pairwise_metrics`：

- 要求四个非空同形 1-D 数组、两个同序同长度的合法 `(id, hash)` 序列，raw labels 相等（在 clamp 前校验，0 和 1 不能当作同标签）。
- 两侧均调用现有 `q_values`，即 `log_prediction = max(log_prediction, 0)`、`true = max(true,1)`，`logQ=abs(log_prediction-log(true))`。
- `mean_log_q_reduction`/`median_log_q_reduction` = 两侧逐查询 `logQ_reference - logQ_model` 的 mean/median；正数表示相对参照误差降低。
- fraction 按真实 log-Q 误差优劣计数；只将 `isclose(logQ_model, logQ_reference, rtol=4*float64.eps, atol=0)` 的舍入级差异视为平局，两个 fraction 均不计平局。此规则明确写入报告元数据；仅用于浮点 log 运算的 tiny tie handling，不代表统计显著性、置信区间或实际效果阈值；没有归零或裁剪 mean/median/Q95。
- **`paired_delta_q95 = percentile(Q_model,95) - percentile(Q_reference,95)`，普通 Q 单位**；负数表示模型 Q95 更小。不是 log-Q95 差，也不是逐查询差值的 Q95。
- 有限 log 输入导致普通 Q 溢出也报错，不生成 inf-inf/NaN，不裁剪输出。
- all-rows 与 unique 调用分别从实际 prediction records 提取身份，与 reference 校验；不是只依据请求 indices 自行声称对齐。

报告新增 `reference_comparison='pg_baseline_not_same_model_k0'` 和 `paired_metric_definitions`；原 numeric paired 字段及其跨 support runs 算术平均保留。PG K=0 不等于同 checkpoint/model 的 K=0，**这些配对字段不是适应收益**。显式 PG 方法声明仅为此输入契约，不构成独立 PostgreSQL provenance 认证。

## RED → GREEN 实际命令和输出

所有命令在上述 worktree 运行。指定解释器现场验证：`/opt/anaconda3/bin/python`、NumPy 1.26.4、Torch 2.3.0、SQLGlot 18.17.0。

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' /opt/anaconda3/bin/python -B -m unittest discover -s tests -p 'test_paired*.py' -v
```

| 阶段/日志 | 实际输出 |
|---|---|
| [原实现 RED](/private/tmp/spatial-meta-ce-s8-review-fixes-20261008/reports/s8-review-fixes-20261008/a1-red.log) | `Ran 19 tests in 0.050s`；`FAILED (failures=15, errors=29)`；exit 1（计数包括子用例） |
| [首轮实现尝试，仍失败](/private/tmp/spatial-meta-ce-s8-review-fixes-20261008/reports/s8-review-fixes-20261008/a1-green-attempt-1.log) | `Ran 19 tests in 0.074s`；`FAILED (failures=2)`；exit 1 |
| [身份顺序/溢出边界 RED](/private/tmp/spatial-meta-ce-s8-review-fixes-20261008/reports/s8-review-fixes-20261008/a1-red-boundary.log) | `Ran 22 tests in 0.090s`；`FAILED (failures=4)`；exit 1 |
| [显式 PG/非法身份契约 RED](/private/tmp/spatial-meta-ce-s8-review-fixes-20261008/reports/s8-review-fixes-20261008/a1-red-contract.log) | `Ran 25 tests in 0.074s`；`FAILED (failures=9, errors=1)`；exit 1 |
| [最终 GREEN](/private/tmp/spatial-meta-ce-s8-review-fixes-20261008/reports/s8-review-fixes-20261008/a1-green.log) | `Ran 26 tests in 0.192s`；`OK`；exit 0 |

原 RED 在生产代码未改时直接复现错误数值：实际 `mean_log_q_reduction=-4.440892098500626e-16`，独立预期 `1.151292546497023`；同时证实标签漂移、ID/hash/重复冲突未 fail-closed。部分新 API 测试因原接口尚无 truth/identity 参数产生 TypeError，未把这些当作唯一 RED 证据。

首轮有一处测试手算错误，保留失败日志后按实际 Q 数组更正：unique PG Q=[10,10,1] 的 Q95=10，model Q=[1,1,10] 的 Q95=9.1，差为 -0.9（不是 0）；不是按函数输出调整期望。

初次 sandbox 执行遭 `OMP Error #179: Can't open SHM`，其 exit 134 不是有效 RED；按 sandbox 规则申请升级后取得表中有效测试失败输出。首次日志包装还使用了 zsh 只读变量 `status`，已改用 `rc`；这些环境/包装失败没有冒充测试证据。

最终邻近专项回归：

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' PYTHONPATH=tests /opt/anaconda3/bin/python -B -m unittest test_paired test_paired_metrics test_core.CoreTests.test_metrics_standard_and_shifted -v
```

[a1-regression.log](/private/tmp/spatial-meta-ce-s8-review-fixes-20261008/reports/s8-review-fixes-20261008/a1-regression.log)：`Ran 27 tests in 0.136s`、`OK`、exit 0。只选取 `test_core` 的既有纯指标测试，不运行其模型/训练测试。未运行读取真实历史目标结果的 `test_paper_results`。

## 独立复算与验证边界

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' /opt/anaconda3/bin/python -B reports/s8-review-fixes-20261008/a1-recompute.py
```

[复算脚本](/private/tmp/spatial-meta-ce-s8-review-fixes-20261008/reports/s8-review-fixes-20261008/a1-recompute.py) 的 oracle 用 Decimal 基数比、Python math/statistics、手写线性插值 Q95，不调用 q_values/summarize/被测函数计算期望；再单独调用被测函数对照。7/7 合成案例通过，exit 0，见 [复算日志](/private/tmp/spatial-meta-ce-s8-review-fixes-20261008/reports/s8-review-fixes-20261008/a1-recompute.log) 和 JSON。这不是正式实验数字。

关键独立预期：true100/PG10/model100：mean=median=ln10、improved=1、worsened=0、Q95差=-9；true100/预测1000和10：两侧基数比 Q=10，两个 fraction 都为0；Q=[1,100] 与参考 Q=[1,1] 的 Q95差=94.05（不是 exp(log-Q95)）；all-rows Q95差=-1.35，unique Q95差=-0.9。

新增合成真实 FeatureStore API 链路测试（无真实目标文件/模型）：`write_cache` 的 raw label 16777217（>2**24，float32 不能精确表示）和100.125，经过 float64 cards.npy → FeatureStore.cards → `batch_samples['true']` torch.float64 → `_predict_rows` 的 PG 数值收集 → NumPy float64/records，标签逐位保持不变。该测试不调用 model forward（调用即抛错），PG log 的 float32 与评估 true 的 float64 是不同字段；训练用 log label float32 不改变评估 raw label 口径。

覆盖欠估、过估、跨真值、同误差/同预测、零真值、负 log clamp、空数组、长度/维度/身份/标签不匹配、非有限值、Q溢出、不同 hash 共 ID、相同/冲突 K0 重复、标签绑定、CSV 缺列、PG 显式声明、all-rows/unique、保持 summarize unique Q95。

AST parse 四份 Python 文件通过，限定 paths 的 `git diff --check` exit 0。初始 Ruff 不在 PATH，指定 Python 无 Ruff module；主代理随后提供既有 executable `/Users/williamlin/.cache/uv/archive-v0/Y8ucSZ-tmHCUuyMswFwm0/bin/ruff`（0.16.10）。未安装依赖。最终实际执行：

```bash
/Users/williamlin/.cache/uv/archive-v0/Y8ucSZ-tmHCUuyMswFwm0/bin/ruff check --no-cache spatial_meta_ce/evaluate.py tests/test_paired.py tests/test_paired_metrics.py
```

[修复前 Ruff](/private/tmp/spatial-meta-ce-s8-review-fixes-20261008/reports/s8-review-fixes-20261008/a1-ruff-before.log)：11 errors，exit 1；仅清理本轮新增/触及的 I001、SIM117、C408，以及 B023（显式绑定 changed_order 的 drift_length）。[最终 Ruff](/private/tmp/spatial-meta-ce-s8-review-fixes-20261008/reports/s8-review-fixes-20261008/a1-ruff.log)：`All checks passed!`，exit 0。生产指标逻辑未为 lint 重构。独立复算 JSON/log 在最终 lint/patch 后已重新生成，source hashes 与当前文件一致。

## Reference 消费者及辅助输入可能位置（仅代码检索）

- [evaluate.py](/private/tmp/spatial-meta-ce-s8-review-fixes-20261008/spatial_meta_ce/evaluate.py)：loader=77、metrics=123、load=220、all-rows call=255、unique call=258、comparison=271、definitions=303；aggregate 保留各 support-run paired unique 指标的均值，不改为 pooled Q95。
- [eval_frozen.py](/private/tmp/spatial-meta-ce-s8-review-fixes-20261008/scripts/eval_frozen.py)：27 默认 `--reference`，41–42 传入 evaluate，48 读取 aggregate `fraction_queries_improved`，51 打印通用 `improved=`。
- [price_fewshot.py](/private/tmp/spatial-meta-ce-s8-review-fixes-20261008/scripts/price_fewshot.py)：28 默认 `--reference`，35–36 传入 evaluate，44 读取该 fraction，47 打印通用 `improved=`。
- [build_paper_results.py](/private/tmp/spatial-meta-ce-s8-review-fixes-20261008/scripts/build_paper_results.py)：725 明确历史 `paired_unique_vs_reference` 的参照是 PG；本轮未编辑/运行。
- [cli.py](/private/tmp/spatial-meta-ce-s8-review-fixes-20261008/spatial_meta_ce/cli.py)：109/111/181 调用 evaluate，但不传 reference；[run_experiments.py](/private/tmp/spatial-meta-ce-s8-review-fixes-20261008/scripts/run_experiments.py)：90/96 同样不传 reference。不存在需要迁移到新私有配对函数签名的其他 production caller。
- 代码/配置/通用文档检索未发现其他 numeric paired 字段直接消费者；本结论限已检索源码，不代表已读取或认证所有历史结果文件。
- 两个脚本默认辅助 JSON 字符串均为 `runs/pg_reference_20260926.json`。从本 worktree 启动时可能解析到 `/private/tmp/spatial-meta-ce-s8-review-fixes-20261008/runs/pg_reference_20260926.json`；从部署目录启动时随 CWD 改变。**未检查该文件存在性或内容**。
- JSON 的顶层 `pg_prediction_path` 指向 PG CSV（必有上述身份/标签列）；现实现按 CWD 解析其相对路径，而非相对于辅助 JSON 的父目录。evaluate 输出的 `predictions.csv` 本身含所需列。真实路径及历史 JSON 是否含显式 PG method 未读、未认证；不能猜测或补写。
- 本轮不改范围外消费者。建议主代理后续把两个脚本打印标签改为 `improved_vs_pg=`；历史 paired 数字及标签由主代理更正脚本处理。本轮不提供同模型 K0 适应收益接口，也不修改历史结果。

## 自查（非独立审查）

准确性5：27专项+7独立复算及源码绑定；完整性5：要求的指标/loader边界覆盖；清晰度5：方向、普通Q尺度、PG与适应收益区别明确；可操作性5：命令/日志/hash及消费者交接完整；简洁性4：为禁止真实推理，原协议测试的 mock fixture 比原测试更长。总体4.8/5；可在后续多处复用需求明确时再考虑缩减 fixture，本轮不做跨文件抽象。自查不替代主代理最终全量回归与代码审查。
