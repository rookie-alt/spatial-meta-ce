# retained κ 收紧修订：独立只读审查（APPROVED-WITH-NITS）

审查代理 Turing（01a11b40-fa7c-7fb0-9e45-02791801d1f4），只读、未写文件、未联网/连服务器、未跑训练、未递归。
基线 HEAD `ac93f26`，审查对象为未提交改动：`spatial_meta_ce/s8_probe.py` + `tests/test_s8_retained_tolerance.py`（+ 本报告目录）。

## 结论：APPROVED-WITH-NITS（8 条清单全部成立）

| # | 检查项 | 结论 | 关键证据 |
|---|---|---|---|
| 1 | 限域（只改 κ 项） | 通过 | `git diff HEAD --name-only` 仅 2 文件；`scripts/` 无 diff；runner SHA=85e0ccda…；RCOND(第13行)/rank 定义(第144行)/strict(第546行块) 未改 |
| 2 | a-priori、不读残差 | 通过 | 先算 residual 再调 bound；bound 入参仅 singular/rank/projection/target_scale；返回键无 residual/observed_residual/calibrated；对探测残差 tol 恒等 |
| 3 | 永不收紧 + 数学一致 | 通过 | `tol=max(floor,reachable)`；σ_r>RCOND·σ₁ ⟹ κ_retained<1/RCOND；rank=0 走 floor；字段自洽 |
| 4 | fail-closed | 通过 | σ_r 非有限或 ≤σ_cutoff 抛 ValueError（旧实现静默放宽）；ValueError 穿透 `_reject_numeric_failure` → runner 置 failed；κ≈1e11 时错误解仍 False |
| 5 | RED 真实性 | 通过 | `red-final.log` Ran 16 / failures=3+errors=6；用例名与 green 逐一对应；3 失败为旧行为/旧 scope，6 错误为 `KeyError:'kappa_retained'` |
| 6 | 回归/Ruff 一致 | 通过 | full 772 OK(13 skip)；targeted 143 OK；changed-ruff 0.16.10 All checks passed；repository-ruff 110 项无改动文件 |
| 7 | 数值对比 | 通过 | 四面板 σ₁/σ_r/rank/残差与 real-proof 逐位一致；新界全 pass；x small 收窄 360.12×、x full 588.44×；h small/h full 由 floor 主导 |
| 8 | 结论边界 | 通过 | 未表述为科学结论/G1 通过；strict=unknown、full_linear_optimality_verified=false、formal_ready=false、original_G1=failed 保留；runner 硬校验 |

## NIT 与处置

1. **`test_bound_does_not_use_observed_residual` 断言强度不足（重要）** → 已在本轮补齐：
   新增 `test_bound_signature_and_outputs_exclude_residual_inputs`（`inspect.signature` 参数集与返回键均排除 residual 语义）
   与 `test_bound_is_insensitive_to_unretained_spectrum`（扰动未保留段 σ[rank:] 不改变 tol；移动 σ_r 使 κ 与 tol 成比例变化）。
2. **`red.log` 与 `red-final.log` 并存且无 staging 说明（可选）** → 保留两者；`red-final.log` 为最终 18 项名，`red.log` 为首版 15 项，二者均为真实 RED 记录，不删除、不覆盖。
3. **新增 report 目录缺 `.gitattributes`（可选）** → 已补 `*.log -text`。
4. **新包冻结未落盘（重要，执行真实 P 前置）** → 本轮审查后随包冻结完成（见 `package-manifest.json` / `execution-contract.json`）。

## 未执行/未复算（登记）

- 未直连远端（沙箱 DNS 不可达），远端=本地仅凭本地 `origin/*` 跟踪引用；
- 未执行本轮单次真实 P 核验，也无新 proof/新隔离目录；
- 未在报告记录的原始解释器上复跑（用 pyenv 3.13.2 独立复跑专项 16/16 OK 作替代证据）；
- 未复算 real-proof 的 σ 值本身（只交叉核对与既有 real-proof 一致）。
