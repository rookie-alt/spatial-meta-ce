# Diagnostic-only 独立只读审查：APPROVED

审查代理 Bacon（01a11a40-f6d7-7ad0-838b-799ea5677bd9）。只读文件/哈希/git 元数据；未写文件、未联网/server、未重跑测试或 checker、未递归。

## 结论：APPROVED，无阻塞

1. **不改任何 gate**：`retained_projection_diagnostic`（scripts/verify_s8_readonly_p.py:432-479）只返回新 dict，唯一写入 `result[name]`；失败分支只写 `row['retained_subspace_diagnostic']`，随后 `row.update` 的 `status/engineering_verified/rank_stability/strict_certificate/certificate` 全取自 certificate。`certificate_complete/stage_record_complete/proof_complete` 零改动；`s8_probe.py` 相对基线无 diff，RCOND/NORMAL_TOL/OBJECTIVE_TOL 原样；34 协议未动。
2. **读取范围仍为八输入、theta 不落盘**：允许集仍 `ARRAYS+METADATA`；诊断仅用内存对象；`reference_theta` 在 :562 `del`，返回 dict 不含 theta，测试断言 stage/result 无 theta。
3. **residual/threshold/ratio 与 np.allclose(rtol=atol=1e-12) 语义一致**：`a=design@theta, b=projection`，`residual=a-b`，`threshold=atol+rtol*|projection|`；rank/投影口径与认证侧同构（singular>RCOND*singular[0]，U_r 投影）。
4. **只在 retained 失败分支执行**：守卫为两 pass 字段任一 False；真实 x-small 两字段均 False，会触发。
5. **冻结包只指向新 diagnostic 目录、旧包保留**：常量与新 contract 同步；archive/contract SHA 实测吻合；旧 handshake-v2 archive `128d1b3c…` 未变。
6. **可用于解释 x small 1216 失败，且不能把 A 改判通过**：诊断恰量化该失败；诊断行在 `certificate_complete` 失败前写入，失败仍 fail-closed；proof 要求 status=verified 且 original_G1_status=failed、can_retroactively_pass_original_G1=false。

## 非阻塞观察项及处理

- O1：失败分支 `linalg_calls` 低估分解次数 → 已最小修正为 5（3 基准 + 诊断 lstsq + SVD），对应测试断言更新。
- O2：freeze 表达式冗余 → 已简化。
- O3：diagnostic 包 8 成员，较 v2 少 `tests/test_s8_readonly_handshake_linux.py`；旧包保留，属有意缩减（diagnostic 不重跑 Linux 握手）。
- O4：失败条件重复求值 → 已合并为 `retained_failed`。

O1/O2/O4 为记录/可读性修正，不触碰 gate；修正后 69/5.422s 与 756/98.967s 本地回归及 Ruff 重新验证通过。此审查仅覆盖 diagnostic-only 代码与冻结，不授予真实 P、B 或训练权限。
