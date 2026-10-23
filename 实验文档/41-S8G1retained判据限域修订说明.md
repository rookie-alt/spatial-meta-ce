# 41 S8 G1 retained 投影判据：限域工程修订说明（本轮唯一一次修订）

日期2026-10-08（Asia/Shanghai）。本页只记录一次限域工程判据修订，用于对齐 x 1216 小面板 retained 一致性检查与 FP64 可达精度。**本次修订只对齐工程一致性与 FP64 可达精度。**

## 1. 修订依据（先给依据，再套用）

既有检查比较两个**独立、同一固定 cutoff 截断**的数值解：
- reported LS：`design @ lstsq_theta`；
- SVD 保留投影：`U_r (U_rᵀ target)`。

精确算术下二者相等；FP64 下二者可差 `O(eps · κ)` 乘问题尺度，其中 κ 是保留块的条件数。由于所有保留方向满足 `σ > σ_cutoff = RCOND·σ₁`，`κ_cutoff = σ₁/σ_cutoff = 1/RCOND` **上界**了真实保留块条件数 `σ₁/σ_r`，且**不依赖未知的最小保留奇异值**。因此取

```
reachable_bound = A · eps · (σ₁/σ_cutoff) · scale
   A = 8（维度相关的保守积累常数）
   eps = FP64 机器精度 2.220446049250313e-16
   σ_cutoff = RCOND · σ₁（与既有固定 cutoff 一致）
   scale = max(|target|_inf, |proj|_inf, eps)
tol = max(工程下限 1e-12 + 1e-12·|proj|_inf, reachable_bound)
```

**依据来自浮点可达精度推导，不从未通过残差反推。** 该界只由冻结 cutoff、谱和问题尺度决定，函数签名/实现对都不读取已观测残差（有测试断言 `tol` 输入中无 `observed_residual/calibrated/residual`）。

## 2. 适用与不适用范围

**适用**：判定“同一固定 cutoff 下，两个独立 LS/投影解是否数值一致”，即 `retained_subspace_pass` / `reference_retained_subspace_pass`。

**不适用/必须保持**：
- 不改 LS 求解方式、不改 `RCOND=1e-12`、不改 rank 定义（仍是 `σ > RCOND·σ₁`）；
- 不改小拟合面板旧 `pass` gate 的其它合取项，不改 h 面板既有判定；
- 不改数据隔离与 H 访问边界；
- `strict_certificate`（true_rank / rigorous_retained_subspace / strict_linear_optimality）**一律保持 unknown**；
- `formal_ready` 保持 false；
- 旧 A failed 与旧 G1 failed 证据**不得改写或追认通过**。

**unknown / 失败条件**：rank=0 用工程下限；非有限/空/秩越界输入 raise；rank 区间不稳定仍 `rank_stability=unknown`；`reachable_bound` 超界或 cutoff 无效时 fail-closed。

## 3. 能支持与不能支持的结论

- **能支持**：该面板两个独立 LS 解在同一固定 cutoff 下数值一致（工程一致性）。
- **不能支持**：科学线性最优、表征信息缺失、strict 由 unknown 转 verified、追认旧 A 或旧 G1 通过。

## 4. TDD 与本地验证

- RED：`tests/test_s8_retained_tolerance.py` 首轮 11 项（7 错误 + 1 失败），因缺函数/字段/新行为。
- GREEN：最小实现后 12 项通过（含“残差略超新界仍失败”“条件数异常仍 strict unknown”“rank=0 用工程下限”“well-conditioned 不收紧”“strongly rank-deficient 通过”）。
- 一次真实回归修正：新界最初**收紧**了 well-conditioned 情形，触发既有 `test_truncated_solver_certification_is_not_full_linear_optimality` 回归失败；改为 `tol = max(工程下限, reachable_bound)`（永不收紧）后通过。
- 完整回归 **768 项 / 94.044 秒 / exit 0**（13 Linux skip）；改动文件 Ruff 通过。

## 5. 冻结

修订后冻结新的 diagnostic/工程执行契约与包成员 SHA，与旧包、旧八输入 SHA 绑定关系一并记录；旧 36/38/39/40 包与失败证据保持原样，不覆盖。
