# 122 P0 收口交付的独立只读审查记录

完成日期：2026-11-12；审查对象为 2026-11-11 的 P0 收口证据。基准提交 `89623047d644842dca134867b6a6ec5f38c0e756`，分支 `audit/122-p0-closure-20261111`。本文件记录**审计交付物**的复审，不是 P0 认证、训练授权或 A/B/C/D 实验结果。

## 审查范围与版本

未参与撰写的独立子代理阅读了[122 协议](../实验文档/122-PRICE式编码器与元学习低秩适配分阶段协议.md)、[验收与预算报告](122-p0-closure-and-a-readiness-20261111.md)、[证据包](122-p0-closure-evidence-20261111/README.md)、相关生产代码与历史测试日志，并核对进度入口新增段落。最终增量复审的冻结字节 SHA256：

| 路径 | SHA256 |
|---|---|
| `reports/122-p0-closure-and-a-readiness-20261111.md` | `e027de57931173bde4376f18d464263d9c8a8d7990f9ec379b1368e47f9da6b6` |
| `reports/122-p0-closure-evidence-20261111/commands.json` | `8b046b62e5ecc5f28a6acd5a4855017f1394e3000ef185da3b05123fcdcd9ae5` |
| `reports/122-p0-closure-evidence-20261111/manifest.json` | `e40109def4744d4773d70aed58199e69d19be220f109980c6877cbfa29188d3d` |
| `reports/122-p0-closure-evidence-20261111/README.md` | `4825bb3751a1334fdebe8256267f761585959046c2a751a04141fada132a5830` |
| `实验文档/99-进度与下一步.md` | `937706e84d554527c509ff0a5caba8257c10c8c75011cae48e67bfd39335e4b2` |

审查过程对脚本与报告的实际字段进行静态对照；审查者**没有**重新访问服务器、复算 SHA、执行 NumPy/模型、跑 unittest 或验证 A/B/C/D。初次增量复审曾使用只读 Bash（超出当时限定的 Read/Grep/Glob 工具集），审查者自行披露；最终针对 M1/M2 的复审仅使用 Read/Grep/Glob，无写入。主代理另行执行并记录的 SHA、链接、JSON、Ruff 和 `git diff --check` 结果不冒充审查者重跑。

## 发现与处理

| 编号 | 严重度 | 发现及定位 | 处理与复审结果 |
|---|---|---|---|
| B1 | 阻塞 | 把 NaN label 的 `FloatingPointError` 错记为非有限**内梯度**守卫已验证；实际上 `meta.py:66–67` 的非有限 loss 检查先抛出，未到 `meta.py:134–138`。 | [验收报告 §3](122-p0-closure-and-a-readiness-20261111.md) 改为“非有限 loss 已验证；有限 loss 下的非有限内梯度**仍缺证据**”，并把内、外两条受控梯度失败路径都列 P1 阻塞；进度入口同步。首轮增量复审确认修订准确。 |
| S1 | 建议 | 说“不读 tensor storage”不准确：对整个 `.pt` 求 SHA 会读取其原始字节。 | 报告、证据 README 和命令台账改为“全文件字节仅用于 SHA；未 `torch.load`、未 unpickle、未按 storage 成员解码或使用 tensor 值”。保留原始 stdout，解释其 `tensor_storage_read=false` 字段语义；首轮增量复审确认。 |
| M1 | 重要 | `test_price_model.py:303–339` 未断言 dropout=0，报告却把 dropout 列在历史测试已验证范围。 | [验收报告 §3](122-p0-closure-and-a-readiness-20261111.md) 区分适配确定性测试与 `price_model.py:56–57` 构造守卫/12 份配置的静态证据，明确“无 dropout 专项拒绝测试”。最终增量复审：**fixed**。 |
| M2 | 重要 | 16 项远端代码/配置 SHA 中，`metrics.py` 的**本地**侧原未单列到 manifest，C10 的“expected/actual 见 manifest”不完整。 | [manifest.json](122-p0-closure-evidence-20261111/manifest.json) 新增 `deployment_extra_code_sha256.spatial_meta_ce/metrics.py`，本地/远端 SHA 均为 `db99399e61df1cf090ebfef0535b3386764737a468713fe2331ee9710ad8c4b3`，主代理实际复算相等；C10 措辞及 `commands.json` 新 SHA/bytes 同步。最终增量复审：**fixed**。 |

**最终交付审查：APPROVE／无未解决交付阻塞。** 此结论只限上述文件、版本及审查范围，不声称没有潜在代码缺陷。审查者未取得本轮模型负例或真实工作负载成本证据，不替代 P0 必过验证，也不授权训练。

## 保留的 P0 与 A 阻塞

- P1：有限 loss 下的非有限内梯度、非有限外梯度两条受控失败路径仍缺证据；不以 NaN loss 测试补位。
- P2：前置委派的子代理执行了不在只读委派范围内的 CPU 模型构造及两次全量 unittest，无本轮要求的 smoke supervisor 记录；如实标记执行合规失败，不追认其结果为 P0 认证。
- A1：12 份模板的相对 `cache_root` 在已核验服务器代码目录不存在；上轮仅内存 override。
- A2/A3：122 四臂真实工作负载计时、GPU/CPU 小时和父总上限未确定；专用监督、独立授权入口及 12 run/396 单元封存尚未认证。具名历史 A0-ref 三 seed 不可复用。

**总门保持 `p0_gate_passed=false`、`formal_training_authorized=false`；预算未定／待授权；不启动 A/B/C/D，不修改 122 协议执行状态，也不合并 main。**
