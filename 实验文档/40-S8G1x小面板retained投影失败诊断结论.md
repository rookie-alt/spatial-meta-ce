# 40 S8 G1 x 1216 小面板 retained-subspace 失败：诊断结论

日期2026-10-08（Asia/Shanghai）。本轮仅 diagnostic-only：不改变任何生产gate、rcond、LS、阈值或34号协议，不把 A 改判通过。一次真实P诊断执行，只读原授权八输入，预算4CPU/空CUDA/900秒/12GiB AS。

## 1. 诊断方式与绑定

新增`retained_projection_diagnostic`，仅在既有`retained_subspace_pass`或`reference_retained_subspace_pass`为False时记录reported/reference的投影残差范数、逐元素阈值、比值、涉及U方向及shape/dtype/rank/cutoff/backend；不落盘theta，不改变任何gate。独立只读审查APPROVED（含4项非阻塞观察，其中linalg_calls记录精度已修正为5）。

- RED `bfb32082`；诊断实现与证据本地提交 `40fbf81`、`9b83286`（entry SHA `7dc783034d4ce5e48469605296adb253c737b6e866dbdeca34c04591e042bae7`）。
- 诊断包archive `ea075b85475481ae16ecac9290503620e00685b741c1422ff0339dcc34f84949`，contract `5b350dc533697adbe1bad226542c56b2cdd4c8a8737fcc0c2456f6d7a6bfc1c7`（8成员，旧handshake包`128d1b3c…`未变）。
- 本地69专项/5.422秒、756完整/98.967秒（13 Linux skip）、改动Ruff通过。
- 单次真实执行：8部署成员SHA一致，八输入before/after一致，access-log仅八输入+冻结源码/runtime；child/parent/outer/SSH=1、signal null，wall 3.4735秒，worker ru_maxrss 538783744 bytes，采样峰值RSS同值、AS 3792367616 bytes，均远低于12GiB。

## 2. 关键数值（x 1216×775，rank 586）

| 量 | 值 |
|---|---|
| reported LS 残差 L2 | 2.1955430818421994e-07 |
| reported LS 残差 L∞ / max abs | 7.49297310864705e-08（max row 750） |
| reference LS 残差 L2 | 2.1955430818421994e-07（与reported逐位相同） |
| reference LS 残差 L∞ / max abs | 7.49297310864705e-08（相同） |
| max row 处阈值 | atol+rtol·|projection| = 1.7812471482339885e-12 |
| 比值（残差/阈值） | 42065.88129039776 |
| 主方向 U[2] | σ=248.855，残差坐标-5.131e-08，投影坐标-0.81049，阈值1.81049e-12，比值28342 |
| 次方向 U[4] | σ=201.176，残差坐标-4.097e-08，投影坐标-0.31959，比值31050 |
| U[11] | σ=124.059，残差坐标-3.731e-08，投影坐标1.02300，比值18442 |
| U[3] | σ=236.229，残差坐标-3.177e-08，投影坐标1.81086，比值11303 |
| 最大奇异值 σ₁ | 340.68582209356845 |
| cutoff | 3.4068582209347655e-10（rank cutoff interval一致） |
| 条件数 σ₁/cutoff | ≈1.000e+12（chunk后精确写出） |
| eps·(σ₁/cutoff) | ≈2.220e-04 |

诊断入口 `gate_effect=diagnostic_only_no_gate_change`，`projection_kind=left_retained_U_projection`，`formula=U_r @ (U_r.T @ target)`，rtol=atol=1e-12，NumPy 1.26.1。

## 3. 结论：不是代码错误、不是面板异常、不是资源问题；失败根因是 retained 投影检查在强秩亏面板上的数值可达精度不足（机制标签未证成）

1. **不是代码错误。** reported LS 与 reference LS 的残差逐位相同，说明两次独立lstsq解在FP64下完全一致；同一检查在h small/h full均通过。因此握手、来源、shape/dtype/finite、rank、谱、objective、reference prediction 都正常。
2. **不是真实面板异常。** 八输入SHA一致、finite、rank三谱一致为586、rank_stability stable；数据本身没有出现NaN/Inf、错配或来源变化。
3. **失败发生在既有 retained 投影检查上，该检查在此强秩亏面板达到 FP64 数值可达精度的边界之外。** 事实：reported 与 reference 两个 ls​t​sq 解的投影残差**逐位相同**（L2=2.1955430818421994e-07，L∞=maxabs=7.49297310864705e-08，max row 750），说明两次独立求解结果一致、没有分支错误；同一检查在 h small / h full 却通过，说明逻辑本身可用。该面板 775 列、数值 rank 586、σ₁=340.686、固定 cutoff=3.40686e-10，条件数 σ₁/cutoff ≈ 1.0e12，`eps·(σ₁/cutoff)` ≈ 2.2e-04。检查要求
   `|design@theta - U_r(U_rᵀtarget)| <= 1e-12 + 1e-12·|proj|`，而实际逐元素残差约 7.5e-08，比值约 **4.2e4**。
   
   **边界声明（必须随记录保留）：** 本证据证明的是「该检查在 FP64 下于本面板无法满足 1e-12 绝对式容差」，**不能**单独判定其机制究竟属于「(a) 门槛/检查对象不匹配（不同算法口径）」还是「(b) 同一检查固有的条件数放大」；区分二者需要更高精度（如扩展精度）独立重算或对检查语义的独立论证，本轮**没有**做（`independent_numeric_recompute=false`）。因此报告只把它写成**数值可达精度不足**，不指定 (a)/(b)。
4. **失败字段定位：** 只有 `retained_subspace_pass`、`reference_retained_subspace_pass`（及其派生的`reference_pass`、`engineering_verified`、`pass`）为False；`metadata_pass`、`normal_residual_pass`、`objective_pass`、`reference_objective_pass`、`reference_prediction_pass`、`spectrum_pass`、`reference_spectrum_pass`、`svd_backend_pass`、`observed_rank_agreement` 均为True。
5. h full 19456 resource-only工程通过（rank_stability unknown）；x full 19456 因 x small fail-fast 而 not_run，未评估。

## 4. 若要协议变更：具体变更与不能支持的结论

**本轮不改任何gate，以下是“另需批准”的建议，不自动执行：**

- 具体变更（仅一项）：把 retained-subspace 一致性检查的判据从绝对式 `atol=1e-12 + rtol=1e-12·|proj|` 改为**与条件数/backend可达精度挂钩**的相对判据，例如以 `eps·(σ₁/σ_cutoff)·scale` 作为工程容差上界，或对 `x` 这类强秩亏面板单列工程阈值。检查仍比较同一对象，不改LS、不改rcond、不改rank定义。
- **不能支持的结论：** 该变更**不能**证明x或h在科学意义上线性最优，也**不能**把 strict_certificate 从unknown改为verified，**不能**追认旧A或旧G1通过。它只能让“同算法数值一致性”的工程门槛与FP64可达精度一致。
- **不能**用“放宽阈值让失败消失”的方式处理，必须保留原始失败证据并另立新协议、新目录、独立审查。

## 5. 若判定为代码错误的处理

不适用：本轮证据显示 reported 与 reference LS 完全一致，且同检查在 h 面板通过，没有发现代码分支错误。唯一发现的记录问题是 `linalg_calls` 在失败分支低估分解次数（原记3，实际诊断加了1次lstsq+1次SVD），已修正为5并重跑本地回归，不涉及gate。

## 6. 路线可行性建议

- **A 仍为 failed，B 依旧阻断；本轮不启动 B。** 诊断已把失败原因量化清楚：这是强秩亏面板（775列/rank586、条件数~1e12）在 FP64 下 `lstsq` 解与 SVD 保留投影无法在1e-12量级一致，而检查用的是1e-12绝对式阈值。
- 是否值得再投入严格认证：若目标是“A工程通过→B”，只需按§4建议另审一个限域工程判据修订，成本低且可复现；若目标是“strict线性最优/表征信息缺失”的强机制结论，则**当前路线在合理预算内不可行**，A/B都无法提供该证明。
- 建议：**不要继续无上限扩展Fraction或严格谱认证**。可（a）另审§4的限域工程判据修订以完成A的工程闭环，或（b）若强机制结论不可得，暂停S8，转入已证实实验（h/x工程一致性、空间漂移Q95实测对照）的论文收尾，并明确标注strict unknown与numerical-edge限制。

## 6b. 本证据不能支持的结论（随记录保留）

- **数值未独立重算**：只核对了字段算术自洽（阈值律、恒等式、排序、rank/shape）、逐位相同与 SHA 绑定，未独立复算 residual；保持 `independent_numeric_recompute=false`。
- **机制未证成**：不能区分「门槛/检查对象不匹配」与「同一检查的条件数放大」，不得表述为已证成机制。
- **退出码口径**：原始文件只有 `ssh_and_outer_returncode` 合并值与 `supervisor.returncode`；parent 单独 rc 未序列化，主代理 `execution-result.json` 中的分列是汇总，不作为独立依据。
- 不得推出：A/G1 可改判通过、失败对新数据无害、residual 幅值说明数据有问题、面板新鲜性或真实 SQL/选择率正确（`identity_scope` 明示仅为历史 SHA 绑定取证）。

## 7. 原状态保留

原G1failed、formal_readyfalse、strict_certificateunknown不变；本次诊断不能追认旧G1成功，也没有独立数值复算（诊断只是记录并复现那条失败比较）。身份仍为历史SHA绑定attestation，未重验逐行query_hash/历史normalizer/真实SQL/选择率。
