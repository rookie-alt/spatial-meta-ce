# S8 G1 local numerical probes — engineering / synthetic only

日期：2026-10-07。检出：`/private/tmp/spatial-meta-ce-s7-diagnosis-20261007`；基准 `19470ee`；分支 `feat/s8-review-and-preflight-20261007`。

## 状态及范围

仅完成本地数值模块和合成测试。**不是 S8 正式执行、机制结果或冻结协议完成**。
只新增/修改 [module](../../spatial_meta_ce/s8_probe.py)、[tests](../../tests/test_s8_probe.py) 和本目录 `probe-*` 日志/报告。
未运行全 suite、真实面板训练/适应、目标读取、服务器/GPU/network、cache/图谱、提交/推送/切分支或子代理。
主代理新增的 source contract/model/preflight 文件未修改；其验证不属于本报告。

## 最小 API 与方法审查后的显式变更

六个函数签名不变；`solve_ls` 固定 `np.linalg.lstsq(rcond=1e-12)`，不 ridge/改 cutoff/fallback。
`residual_targets` 严格要求两个输入为一维、非空、finite float32 ndarray，分别升 FP64 后相减。
`solve_adam` CPU FP64、全零参数、MSE、固定 lr=3e-4/betas=(.9,.999)/eps=1e-8/weight_decay=0；无剪裁/早停。
零初始化不使用随机操作，不修改全局 RNG；合成测试核对了 Torch/NumPy RNG 和输入不变。
非法 dtype/shape/rank、非有限、运算溢出/solver 失败显式 ValueError；有限但认证不通过返回 False。

- normalizer 现在返回 `mean`、原始 population `std` (ddof=0)、实际分母 `scale`、bool `zero_mask`/`tiny_mask`。
  `scale=1` 当 std=0，否则 max(std,1e-12)；tiny_mask 只包含 0<std<1e-12，zero_mask 包含 std=0。
  std 使用代数等价的缩放 population variance，避免非零小方差平方下溢；非恒定列 std 仍下溢则明确拒绝。
  `standardize_apply` 只用 P 的 mean/scale，不重新 fit、不使用 P masks 清零 Q、不修改 normalizer。
  **旧 mean/std-only normalizer 现在拒绝，不静默沿用；旧 std 字段曾是 scale，已纠正语义。**
- `certify_pair` 的 `pass == solver_verified` 只表示固定截断数值对照通过。
  重新计算 objective、normal residual、SVD/rank，并核对报告字段；metadata 以 rtol/atol=1e-12 数值核对，不是 hash 来源证明。
  **fullrank 修订追加同一 rcond=1e-12 的 reference lstsq，核对 J 和逐行预测（沿用 metadata tolerance），失败不能认证。**
  正规方程门槛仍为 `<=1e-8`；objective 门槛仍为 `J_ls <= J_adam + 1e-8*max(1,J_adam)`。
  `rank_limited == numerical_rank_limited == (rank < n_columns)`，含欠定和精确冗余，不根据误差选择规则。
  `fullrank_numeric_solver == (solver_verified and not rank_limited)`；它只是相对同一固定求解器参考的数值认证。
  **旧 `full_linear_optimality_verified` / `full_linear_optimality` 兼容保留，但现在始终 False，不自动发最优性标签。**
  秩不足时完整类最优性仍为未认证/unknown，**不是证明不最优，也不是表示缺信息证据**。
  满列秩也不构成独立科学最优证明；调用方不能把本模块自重算称为独立验证。
  新返回 `reference_objective/reference_objective_pass/reference_prediction_pass/reference_pass` 和 `certification_linalg_calls=2`。

## 阻断反例与上一轮结果差距

新增用户指出的 `Z=[[-1e-14,1],[1e-14,1]], r=[-1,1]` 负例。
固定截断 LS 和该 2×2 synthetic 的 Adam1000 均 J≈1；可行 theta=[1e14,0] 的 J=0。
小 normal residual 可过 gate，但现在只允许 solver_verified/pass=True；numerical_rank_limited=True，
full_linear_optimality_verified/full_linear_optimality=False。精确 zero 冗余也执行同一保守规则。
**审查前 25 项通过只证明旧工程契约，未覆盖此阻断，不能作为原完整线性类最优性的证据。**
进一步加入 fullrank 病态伪造负例：`Z=[[-1e-11,1],[1e-11,1]], r=[-1,1]`。
旧门槛对 rank2/零 theta/J1/normal≈1e-11 会误通过；新参考解 J≈0、theta≈[1e11,0]，
使伪造证书 reference objective/prediction 均失败，solver_verified/pass/fullrank_numeric_solver 均 False。
真实解通过数值认证，但 full_linear_optimality 两字段仍 False。见 [现场合成复算](probe-fullrank-counterexample.log)。
原有门槛边界测试现在单独认证 gate 的等号/边界外，不再把人工构造的非 LS 系数当完整 certificate 应通过。
没有新增 rcond 搜索、投影阈值、完整运行器、新 F0 或正式实验。

## 本轮真实 TDD/验证

- [首次 red](probe-red.log)：缺模块的 ImportError，exit 1；不是逐项运行后的 17 个 failures。
- [首次 green](probe-green.log)：17 项通过，0.864 秒。
- [方差边界 red](probe-boundary-red.log)：22 项中 1 failure；[修复 green](probe-boundary-green.log)：22 项通过，0.748 秒。
- [审查前验证](probe-final-tests.log)：25 项通过，0.686 秒（旧 API，不是最终契约）。
- [最优性/API red](probe-optimality-red.log)：29 项中 6 errors，exit 1。
- [截断修订 green](probe-optimality-green.log)：29 项通过，0.840 秒，exit 0；该轮尚未覆盖 fullrank 伪造。
- [fullrank red](probe-fullrank-red.log)：32 项中 2 failures、1 error，exit 1，复现旧实现错误通过伪造证书。
- **[当前最终 green](probe-fullrank-green.log)：32 项通过，0.867 秒，exit 0。**
- **[当前最终 lint/format](probe-fullrank-lint.log)：限定两个新增 Python 文件，check 和 format --check 均 exit 0。**
- 调用次数测试包裹实际 NumPy 函数：标准配对精确观察到 2 次 lstsq（均 rcond=1e-12）+ 1 次 SVD；reference 失败只调用一次后 reject。
- 实际命令：`PYTHONDONTWRITEBYTECODE=1 /opt/anaconda3/bin/python -B -m unittest discover -s tests -p test_s8_probe.py -v`。
  NumPy 1.26.4 / Torch 2.3.0；Ruff 0.16.10，使用指定 cache executable，`--no-cache` 不生成项目 lint cache。
- oracle 含精确线性和已知非零 MSE；覆盖 rank cutoff、冗余/欠定、置换、FP32 转换、P-only apply、dtype/shape/overflow、
  Adam 固定参数参考/确定性/零 PG 起点/moments 溢出、tamper/solver 失败，以及门槛等号与边界外。

## 风险与主代理接入边界

本模块不能认证输入来自 P/source-only，调用方负责来源/身份/归一化配对和完整分母。
主代理接入时须改用 `scale`，使用 solver_verified/fullrank_numeric_solver，不依赖旧 full_linear_optimality 字段变 True；不得把 pass 当最优性。
本轮最大数据矩阵只有 4 行；1000 更新仅用于上述 2×2 反例，其余 Adam 测试不超过 7 更新。
未测正式 19456×775 面板的耗时/RSS或完整流水线。
**当前标准配对累计三次线性代数调用：solve_ls 的 lstsq + certify_pair 的 SVD + reference lstsq。**
这是两次最小二乘求解和一次 SVD，不是“三次 LS”；certify_pair 自身为两次，返回字段记录该计数。
额外完整矩阵求解/分解、预测和副本具有实际 CPU/RSS 成本；future 正式预算必须现场测，不用这次 tiny 时长外推。
同一 NumPy solver 的 reference 仅闭合系数/预测伪造，不提供另一 solver 或独立科学最优性证据。
cutoff 邻近秩判定或不同 BLAS 误差可能保守拒绝，不自动重试/换 cutoff。
未改全局线程设置；部署线程/资源预算由正式执行层冻结，不据此启动实验。

## 当前源码绑定

- module SHA256：`93abad1c9d4a2ae5ebec83a4b05b0b476d020d415b4df1cb7aa4776d731df154`
- tests SHA256：`e10d534786083358f0d7658d57697a0f8ec74b29a534946b254854c748491bed`

## 简短自评（非独立审查）

准确性 4/5：截断与 fullrank 伪造已纳入，但 reference 同 solver，非独立科学证明；完整性 4/5：32 项覆盖请求，正式集成按授权未做；
清晰度 4/5：别名及 solver/full-optimality 区分已写明，主代理仍需迁移 scale；可操作性 4/5：专项与 lint 有真实日志，
接入仍由主代理完成；简洁性 4/5：仅两个 Python 文件，验证/拒绝分支仍有一定篇幅。平均 4.0/5。
优先改进：主代理迁移 scale/认证字段；授权后另测真实尺寸资源，不用 synthetic 时长估计预算。
用户应能核对这些界限；本自评不能替代独立方法审查。
