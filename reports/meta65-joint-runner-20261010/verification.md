# Meta65适应算子runner验证报告

日期：2026-10-10。分支 `feat/meta65-joint-control-runner-20261010`，基准 `55c777e382fda863a5d42078ff49d9edf476fc3b`。主目录与旧 worktree 未改动。

## 范围

本轮只实现 evaluation-only runner、外部监督器、独立 verifier、测试、服务器 CPU 监督预检和 launch-binding。**未执行正式 135-cell 矩阵，未执行 GPU 适应，未重新元训练，未扩 seed，未调参，未做目标库评估，未推进 S8。**

## 本地验证

- 聚焦测试：`tests.test_meta65_joint_runner` + `tests.test_meta65_joint_supervisor` + `tests.test_meta65_joint_verify` 共 **39 项通过**。
- 完整 CPU 回归：单线程沙箱外 `unittest discover` → **942 tests OK (skipped=15)**，112.595 秒，退出码 0。使用单线程 BLAS/OpenMP 环境变量，沙箱内首轮曾出现 OpenMP 共享内存与 `ps` SID 兼容问题，均为测试环境问题，已在最终命令中消除。
- 改动文件 Ruff：3 个生产脚本 + 3 个测试文件全部 `All checks passed`。
- `git diff --check`、AST 解析通过；未发现目标库字符串；敏感信息扫描未发现凭据。

## 服务器 CPU 监督预检

命令（隔离目录、单线程、CUDA 显式隐藏、外部 960s 超时）：

```text
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
/gly/weilin/PRICE-master/venv/bin/python -B scripts/meta65_joint_runner.py \
  --execute --mode cpu_preflight \
  --config $ROOT/configs/meta65_joint_control.json \
  --out $ROOT/runs/cpu-preflight-20261010e
```

结果（修复前首轮）：`status=preflight_completed`，`phase=certification`，退出码 0，20.146 秒，峰值 RSS 1,054,859,264 bytes（约 1.05 GiB），峰值 VRAM 0，输出 150,795,082 bytes（约 144 MiB）。外部 supervisor 记录 `failure=null`。

产物核对：9 个 micro cell；`verification.json` 为 `preflight_verified`、`formal_result=false`、`cells_verified=9`、`all_metrics_verified=true`、`summary=null`；全部 cell `state_before==state_after`；A/B 使用 head3（选中参数 107,521、3 步），C 使用 last3（选中参数 65、3 步）；9 个 cell 的回归锚均 `regression_anchor_checked=true`。

失败留痕（未删除）：e00-e04 中 e00 因 supervisor cwd 相对 config 失败，e01/e02 因隔离目录缺少 provenance JSON 失败，e03 因准备阶段在完整 query 上误比旧锚 `log_pred` 失败；e04 为锚点修复后的成功预检。上述均为实现/部署缺陷，已在 e04 前修复并复验。

## 边界

- 预检只验证 9 个 micro cell 的工程路径、FP64 真值链路、参数隔离、身份/真值/表数/计划/回归锚路径；**不构成正式 135-cell 结果，也不发布任何正式统计结论**。
- 正式执行仍需：现场选空闲 GPU 并绑定 `CUDA_VISIBLE_DEVICES/UUID`、用户一次性授权、补齐 launch-binding 中 GPU UUID 与 approval 文件，然后运行 `--mode formal`。
- 首轮 `launch_ready=true` 经独立审查判定证据链不完整（binding 仅覆盖 3/9 code 文件、verifier 漏检同 checkpoint 初始化一致性、none 预检未落盘），已修码；该轮 `launch_ready` 作废。重跑后的最终判定另记。`execution_authorized=false`、`gpu_formal_executed=false`。CPU 预检 RSS/输出为 micro 路径采样，不构成正式 4 GiB / 1 GiB 预算安全证明；正式仍由 watchdog 实测并在超限时失败。

## 独立审查

监督器实现由独立只读复核（Carson）给出根因与最小修正，已采纳：macOS `ps` 无 SID 列、采样竞态、失败写回与 symlink 处理、watchdog 时限判定。runner 与 verifier 交叉实现后由主代理整合并复跑全套测试；本轮 39 项聚焦与 942 项完整回归以最终落盘代码为准。
