# Meta65 scope 双预检与重新冻结记录

日期：2026-10-17（Asia/Shanghai）。本轮只完成工程预检与重新冻结，不授权或执行正式矩阵。

## 工作区与范围

- 本地：`/private/tmp/spatial-meta-ce-meta65-scope-safety-20261017`。
- 分支：`experiment/meta65-scope-safety-20261017`。
- 执行代码基准：`b7b68a22ea3f507d9b4d506bce0a4642c3d7d51f`。
- 服务器代码：`/gly/weilin/spatial-meta-ce-meta65-scope-safety-20261017-code`。
- 服务器产物：`/gly/weilin/spatial-meta-ce-meta65-scope-safety-20261017-runs`。
- 本轮未修改代码、配置、冻结协议、预算或缓存；未合入独立 cache schema 2 分支。
- 主目录三份既有未提交指令文件未触碰；本轮所有归档只在上述隔离 worktree。

## 现场资源与运行

17:03 首次 `nvidia-smi` 显示 GPU1/2/3 各 1 MiB、0%；GPU6 有他人进程，不使用旧 GPU6 binding。
17:04:53 再查 GPU1 仍为 1 MiB、0%。固定 GPU1 UUID `GPU-722d57aa-e2e6-8b0a-6c6f-94e99f3031ed`，没有抢占或自动换卡。
`df -h /gly` 显示可用 13T。启动前未发现本项目 scope runner、supervisor 或 probe 残留进程。

本轮运行：

```text
CUDA_VISIBLE_DEVICES=GPU-722d57aa-e2e6-8b0a-6c6f-94e99f3031ed
CUBLAS_WORKSPACE_CONFIG=:4096:8
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
PYTHONDONTWRITEBYTECODE=1
/gly/weilin/PRICE-master/venv/bin/python -B \
  /gly/weilin/spatial-meta-ce-meta65-scope-safety-20261017-code/scripts/meta65_scope_runner.py \
  --config /gly/weilin/spatial-meta-ce-meta65-scope-safety-20261017-code/configs/meta65_scope.json \
  --data-config /gly/weilin/spatial-meta-ce-meta65-scope-safety-20261017-code/configs/meta65_server_data.json \
  --out /gly/weilin/spatial-meta-ce-meta65-scope-safety-20261017-runs/gpu-preflight-20261017T1705 \
  --mode gpu_preflight --execute
```

启动父进程 PID 68401，监督 worker PID/PGID 68407。每臂仅两步训练，每库一个 run、32 条 query，共 12 个微单元；不是正式 4000 步两臂训练。

## 已验证证据

| 项目 | CPU 今日已有运行，本轮重验 | GPU 本轮新运行 |
| --- | --- | --- |
| 输出目录名 | `cpu-preflight-20261017` | `gpu-preflight-20261017T1705` |
| worker / supervisor | `preflight_completed` | `preflight_completed` |
| 独立原始产物认证 | `preflight_verified`，12/12 | `preflight_verified`，12/12 |
| returncode / failure | 0 / null | 0 / null |
| cleanup_verified | true | true |
| 运行耗时（秒） | 543.7221979559399 | 239.44163151970133 |
| 父监督采样峰值 RSS（bytes） | 1957085184 | 2538340352 |
| 父监督外部采样峰值 VRAM（bytes） | 0 | 0 |
| cell 内 PyTorch 峰值 VRAM（bytes） | 0 | 83886080 |
| 输出体积（bytes） | 159607431 | 159612999 |

CPU 预检不是本轮重跑：本轮读取完整当前输入、代码闭包、运行环境、checkpoint 和原始 NPZ，通过 `launch._preflight(...)` 重新认证。部署的 40 份代码/协议/来源/配置文件 SHA 与基准 worktree 全部一致。
GPU 外部周期采样的 0 不代表未使用 CUDA；cell 内 PyTorch 记录峰值 80 MiB。两种测量不同，不能互换。
四份 `*.supervisor.json` / `*.verification.json` 从服务器逐字节复制，本地 SHA 与服务器一致；checkpoint/NPZ 保留服务器，不提交。

本地本轮专项验证：

```text
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' \
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=tests /opt/anaconda3/bin/python -B -m unittest \
  test_meta65_scope_launch test_meta65_scope_runner test_meta65_scope_verify \
  test_meta65_scope_decision -v
Ran 98 tests in 11.133s; OK; exit 0
```

本轮未重跑全量测试或全仓 Ruff，不能把历史全量通过写成本轮验证。

## 构建与授权边界

首次 binding 构建在写文件时因归档父目录不存在抛出 `FileNotFoundError`，退出 1；未留下有效 binding。创建全新归档目录后重跑构建，不重跑预检，不修改输入、代码或预算。

第二次 `scripts/meta65_scope_build_binding.py --root <服务器代码根> --config <冻结配置> --cpu-preflight <今日 CPU 输出> --gpu-preflight <本轮 GPU 输出> --gpu-uuid <固定 GPU1 UUID> --out <服务器代码根>/reports/meta65-scope-refreeze-20261017/launch-binding.json --verify` 退出 0，输出 `verify_launch_binding=passed`，完整重验 264 份输入、32 份代码、每种预检 53 份原始产物及其独立认证结果。
本地 [launch-binding.json](launch-binding.json) 为服务器原件逐字节复制，SHA256 `d09c2ad74215ddc8e84002233e35ca381e0a3b32e000ac27d5c0fbdfe22c008d`。
本地缺少服务器绝对路径对应的缓存及原始产物，不声称在 macOS 本地通过完整 `verify_launch_binding()`；本地只核对代码/配置/协议闭包及复制原件 SHA。

正式预算保持 14400 秒 / 12 GiB RSS / 8 GiB VRAM / 1 GiB 输出。RSS/VRAM 是父监督周期采样口径，不是内核资源隔离。
`launch_ready=true` 仅表示工程就绪；必须保持 `execution_authorized=false`、`formal_executed=false`。
新 runs 根目录未发现 formal、approval 或 consumed 路径；新正式矩阵仍为 0/60，旧 B 正式复现锚仍为 0/15。
下一步必须先向用户展示新 binding SHA 和归档提交 SHA，取得一次性执行确认；随后现场重查固定 GPU1、重验全部输入并使用全新输出，R 先过 15 个旧 B 复现锚才启动 T。
失败即停，不自动重试、换卡、扩 K/seed、恢复 S8 或读取目标库；不从微预检数字推断精度提升，旧 `no_candidate` 不变。

## 独立审查与操作偏差

首名独立代理核对实际 diff、32 份代码与配置/协议 SHA、四份复制原件、目标路径隔离、0/60 与无 approval 边界；服务器单次重跑 `verify_launch_binding()` 通过，本地独立复跑同四模块 98 项 / 9.957 秒 OK。其技术核验未发现阻塞或重要问题，但不能写成严格只读审查已完成：代理在取 SHA 时误写辅助 `reviewer-code-shas.json` 至服务器 code 树，随后自行移至 runs 根的 `reviewer-code-shas-20261017.json`。该文件未修改任何冻结代码、输入、预检原件或 binding，且不在绑定清单中，但仍违反本轮明确的只读职责；保留披露，不删除文件以掩盖操作。

因此另请未参与本轮实现或上述审查操作的代理对最终归档独立只读复核，禁止任何文件输出、迁移或远端写入。最终是否达到只读审查门槛，以第二名代理的实查结果为准。
外部 VRAM 采样 0 的根因本轮未单独复现实验，不把容器 PID 可见性解释当作已证明的因果结论；只报告外部采样与内部 CUDA 高水位两种原始事实。

第二名代理严格只读复核最终 diff、所有本地冻结代码与配置/协议 SHA、四份复制原件及其服务器原件 SHA、binding 本地/服务器 SHA、服务器输入抽样、目标路径隔离、无 formal/approval/consumed 及辅助 SHA 文件不在绑定清单中，结论：**通过，无归档阻塞发现**。未重跑全量输入扫描、测试或 GPU；未写入或迁移任何本地/远端文件。
其最初提出的解释器不可用 P2 已撤回：审查命令误拼为 `/gly/weinlin/PRICE-master/venv/bin/python`，文档和实际运行均为正确的 `/gly/weilin/PRICE-master/venv/bin/python`。主代理与审查代理随后现场只读确认正确路径可执行，Python 3.11.4、NumPy 1.26.1、Torch 2.1.0+cu121、CUDA 12.1 与预检一致；这不是环境缺陷，也不修改既有证据。第二名代理未完成完整服务器 verifier 复现，完整认证依据仍为主代理的构建后认证及首名代理单次独立重验，不能冒充第二次完整认证。
