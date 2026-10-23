# 44 S8 G1 retained 判据收紧：κ 改用真实保留条件数（本轮唯一一次后续修订）

日期 2026-10-08（Asia/Shanghai）。承接 [41](41-S8G1retained判据限域修订说明.md)/[42](42-S8G1A单次真实P执行通过记录与判据强度限制.md)/[43](43-新窗口交接-S8G1三阶段完成与待授权后续.md)。
本轮只做**一次**收紧：把 retained 投影一致性判据的 κ 从 `1/RCOND` 常量改为**实测保留条件数** `κ_retained = σ₁/σ_r`。**不改** LS、`RCOND=1e-12`、rank 定义、小面板其他 gate、h 判定、数据与 H 隔离、`strict_certificate=unknown`、`formal_ready=false`、旧 A failed 与旧 G1 failed。

## 1. 公式（新旧对比）

```
floor           = 1e-12 + 1e-12·|proj|∞                     # 硬下界，永不收紧
旧 reachable    = 8·eps·(σ₁/σ_cutoff)·scale = 8·eps·(1/RCOND)·scale      （κ=1/RCOND 恒为 1e12）
新 reachable    = 8·eps·κ_retained·scale,  κ_retained = σ₁/σ_r          （σ_r = 最后一个保留奇异值）
tol             = max(floor, reachable)
scale           = max(|target|∞, |proj|∞, eps),  eps = 2.220446049250313e-16,  系数 = 8
```
依据：同固定 cutoff 下两独立数值解在 FP64 下可差 `O(eps·κ)`；σ_r 已得，直接用真实保留块条件数即可把界对齐真实可达精度。**a-priori、不读已观测残差**；因 rank 定义保证 `σ_r > RCOND·σ₁`，故 `κ_retained ≤ 1/RCOND`，**新界绝不宽于旧界**。`σ_r` 非有限或 `≤ σ_cutoff` 时 fail-closed（raise）。

## 2. 四面板新旧界数值对比（用冻结 real-proof 的真实 σ₁/σ_r 复算）

| 面板 | rank | κ_retained=σ₁/σ_r | floor | 新 reachable | 新 tol | 旧 tol | 旧/新宽度比 | 残差 | 新判 | 主导项 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|:--:|---|
| h small (1216×65) | 46 | 1.2299e2 | 5.8752e-12 | 1.8208e-12 | **5.8752e-12** | 1.4805e-2 | **2.52e9** | 2.576e-14 | pass | floor |
| x small (1216×775) | 586 | 2.7768e9 | 6.8947e-12 | 4.1110e-05 | **4.1110e-05** | 1.4805e-2 | **360.12** | 7.493e-08 | pass | κ 项 |
| h full (19456×65) | 50 | 1.1261e2 | 5.6451e-12 | 2.4643e-12 | **5.6451e-12** | 2.1884e-2 | **3.88e9** | 1.257e-13 | pass | floor |
| x full (19456×775) | 596 | 1.6994e9 | 6.6174e-12 | 3.7190e-05 | **3.7190e-05** | 2.1884e-2 | **588.44** | 3.773e-08 | pass | κ 项 |

要点：
- 四面板在**收紧后的界**下仍全部 pass；h 小面板不再被削弱（界回到 1e-12 量级、与真实可达精度同量级），x 面板由 κ 项主导、收窄约 360×（x small）/588×（x full）。
- `reported_residual == reference_residual`（两组独立解逐位一致）保持不变；x 小面板 7.493e-08 的逐位一致仍是判 pass 的核验对象。
- 这证明"检查强度已恢复到与真实可达精度同量级"：旧界锚定 1e12 是常数、与面板良态无关；新界随真实 κ 变化。

## 3. 保留不变与结论边界

保留：LS/`lstsq(rcond=RCOND)`、`RCOND=1e-12`、rank 定义 `σ > RCOND·σ₁`、小拟合面板旧 pass 其它项、h 判定、数据隔离与 H 访问边界、`strict_certificate=unknown`、`full_linear_optimality_verified=false`、`formal_ready=false`、旧 A failed、旧 G1 failed（`original_G1_status=failed`）。
**能支持**：同一固定 cutoff 下两个独立 LS/投影解在 FP64 下数值一致，且判据强度与真实可达精度同量级。
**不能支持**：科学线性最优、表征信息缺失、strict 由 unknown 转 verified、追认旧 A 或旧 G1；也不改变 39/40 号记录的 x 失败根因叙述（该失败源于当时 1e-12 绝对式阈值，非新界）。

## 4. TDD 与验证

- RED `tests/test_s8_retained_tolerance.py`：18 项，对 HEAD 旧实现 **3 失败 + 7 错误**（7 错误为 `KeyError:'kappa_retained'`，失败为旧 scope/旧 fail-closed 行为/条件数异常不误过）。
- GREEN：18/18 通过。
- 专项 **158/158**（13 Linux skip）；完整回归 **774/774 / 97.445s / exit 0**（13 Linux skip）；改动文件 Ruff（0.16.10）`All checks passed`；仓库全库 Ruff 110 项均为既有告警、无改动文件。
- 覆盖：良态面板用紧界、强秩亏面板用宽界、残差略超新界仍失败、条件数异常（σ_r 极小或非有限）返回 unknown/fail-closed 而非误过、不读已观测残差（结构化签名断言 + 未保留谱不敏感性断言 + 容忍-变异断言）。
- 独立只读审查代理 Turing：**APPROVED-WITH-NITS**（8 条清单全部成立；nits 已处置：强化不读残差断言、补 `.gitattributes`、完成新包冻结）。见 [审查记录](../reports/s8-g1-readonly-p-retained-kappa-20261008/independent-review.md)。

## 5. 冻结（新执行契约与包 SHA）

- 新隔离包 archive `40ca43c6d8e73e7e73e1466864fb92dc11b2b4778343af779b7c870cf48bb658`（9 成员）、contract `54f9c24c47062b1a73e5f2be4b651fb143760a1ce689ae75bd27e63567ddd99f`。
- 成员：`scripts/verify_s8_readonly_p.py` bba63497…（仅 3 个路径常量重绑定到新隔离目录，无任何数值逻辑改动）、`spatial_meta_ce/s8_probe.py` 7c7847ce…、`spatial_meta_ce/__init__.py`…、`tests/test_s8_readonly_p.py`…、`tests/test_s8_retained_tolerance.py` eff6c5ff…、`实验文档/27、34` 与 `proof.schema.json`（与旧包同 SHA）。
- 与旧包关系：`prior_packages` = handshake_v2 `128d1b3c…`、diagnostic `ea075b85…`、tolerance `e9fe330f…` 三旧包 SHA 全部保留、未覆盖；旧**八输入** SHA 原样保留在 `files`/`input_files_sha_preserved`。
- 旧 41/42/43 记录与旧 failed/unknown 证据未改写。

## 6. 状态边界（不得改写）

原 G1 仍 `failed`；`formal_ready=false`；`strict_certificate=unknown`；`full_linear_optimality_verified=false`；`can_retroactively_pass_original_G1=false`；`identity_scope=historical_sha_bound_panel_attestation_not_fresh_query_semantics`。
本轮**未**启动 B/F0/Adam/训练/GPU；未读 H/目标库/checkpoint/训练状态；未改 PRICE；未合并 main。
