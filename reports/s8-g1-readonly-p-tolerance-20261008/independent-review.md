# 阶段1 retained 判据限域修订：独立只读审查 APPROVED

审查代理 Planck（01a11b0d-2786-71d0-9d97-1bed02fa2fff）。实际读文件与日志；未写文件、未联网/server、未重跑 checker/测试/训练、未递归。基线 ad50aa2，审查 HEAD f47bda9。

## 结论：APPROVED（1 条记录瑕疵，非阻塞）

1. **限域正确 — 通过**：`git diff ad50aa2..f47bda9` 只含 `s8_probe.py` 的 4 个 hunk（全在 retained 段）、新测试、报告日志、41 号说明与进度。删除行只有原两处 `np.allclose` 与一行旧 docstring。`RCOND=1e-12`（:13）、`_rank_and_singular`（:129-147）及 rank 定义（:144）、`solve_ls`/`lstsq(rcond=RCOND)`（:150-155）、`metadata_ok/normal_ok/objective_ok/svd_backend_pass/observed_rank_agreement/reference_prediction_ok`（:489-493）、`strict_certificate.status=unknown`、`full_linear_optimality(_verified)=False`、`formal_ready` 均未改；h 面板无独立判定代码，仅经 `retained_ok` 合取项间接受影响。
2. **a-priori 且永不收紧 — 通过**：bound 仅消费谱/rank/投影范数/`target_scale`，先算 residual 才调 bound，residual 不回灌；`floor=1e-12+1e-12·|proj|∞`，`tol=max(floor, reachable)` 必 `>=floor`；rank=0 走工程下限、非有限/越界/cutoff 无效均 raise 或 fail-closed。次级观察：`reachable` 未显式校验有限性，理论上可 inf，但 `kappa_cutoff=1e12` 恒定、需 `scale≈1e304` 才溢出，工程不可达（非阻塞）。
3. **RED 实质成立，但存在记录瑕疵**：`red.log` 首轮 `Ran 11 tests, failures=1, errors=7`，7 个 ERROR 为缺少 `retained_projection_bound/check`、1 个 FAIL 为 `test_ill_conditioned_rank_deficient_pair_passes_with_new_bound` 在旧固定 tol 下真实失败——确为缺函数/旧行为。**但**该日志的 11 个用例名与最终提交的 12 个用例名全部不同（重命名前版本），故不能逐条绑定最终测试集。主任务已补 `red-final.log`（11 项按最终名 error）与 `red-behavioral.log`（最终名 1 项在旧 tol 下真实失败）修正此瑕疵。
4. **回归与 Ruff 一致 — 通过**：`full-tests.log` `Ran 768 in 94.044s`、`OK (skipped=13)`，12 项最终名全 ok；`changed-ruff.log` `All checks passed!`。
5. **无越界/为通过而放宽 — 通过**：唯一"放宽"为 `max(floor, reachable)`，有 a-priori 推导注释；LS/rank/科学 gate 未在 diff 出现；41 号 §4 记录了"新界曾收紧 well-conditioned"的真实回归修正。
6. **结论边界一致 — 通过**：仅支持"同固定 cutoff 下两独立解数值一致"；不支持科学最优/strict 转 verified/追认旧 A 或旧 G1。
7. **无新读取/泄漏 — 通过**：`s8_probe.py` 全文无 `open/os/subprocess/requests/socket/urllib/environ/np.load/torch.load/Path(`；新增函数只消费入参。

审查不要求重跑；批准仅限该限域修订，不授予真实 P、B 或训练权限。
