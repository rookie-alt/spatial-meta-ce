# Meta65 GPU6 窄预检通过；正式矩阵待确认（2026-10-11）

接续[61号 CuBLAS 修复与首次预检阻塞](61-Meta65-CuBLAS修复与GPU6预检阻塞.md)。用户本轮只要求“再检查 GPU 6，空闲就继续预检”，**未授权本轮直接启动 135-cell 正式矩阵**。执行基于已推 Gitee 的 `experiment/meta65-cublas-preflight-20261011` 提交 `f9a6e3c6d64b25e7434927e04121a628a9973a1b`；原冻结输入、PRICE 和第一次失败目录均保持只读。

## 现场门禁与实际执行

- GPU6 UUID `GPU-0050e82c-8c24-d85d-00d3-b73f12638df0` 现场显存 1 MiB、利用率0、无计算进程。服务器隔离代码根 `/gly/weilin/spatial-meta-ce-meta65-cublas-20261011-code` 的23项代码／测试／配置／协议／CPU证据文件与部署清单哈希一致；`input_binding()` 实时复核86文件、三源验证库完整plan、45个旧回归锚。新输出路径此前不存在。启动器在实际启动前再查指定GPU空闲；未换卡。
- 在 `CUBLAS_WORKSPACE_CONFIG=:4096:8`、单线程和显式GPU UUID下，仅启动一次 `--mode gpu_preflight --execute`，输出到 `/gly/weilin/spatial-meta-ce-meta65-cublas-20261011-runs/gpu-preflight-20261011`。外部supervisor记录 `preflight_completed`、returncode0、failure=null、phase=certification、wall **34.273990720976144秒**；峰值RSS **1,486,675,968 bytes**，输出 **150,804,005 bytes**。一秒周期的外部VRAM采样峰为0，而9个cell各自记录的PyTorch显存高水位均为 **73,400,320 bytes**；前者错过短暂CUDA使用，不能把0写成“没有用GPU”，每cell仍已执行显存上限检查。
- 9个micro cell使用三个已有checkpoint × chembl/legalacts/telstra、run0、每cell 32个固定query及原K20支持；A/B选择107,521个head参数、C选择65个last参数，各3步。另有9次 `none` 检查，均0参数/0步且初始化状态不变。9个对角回归锚路径均执行，所有cell的`state_before==state_after`。`verification.json` 报 `preflight_verified`、9 cells、`all_metrics_verified=true`、`formal_result=false`、`summary=null`；服务器从原始NPZ独立重算与之相同，并在运行后重验86个输入文件与45锚。

## 证据认证与范围

本地[GPU预检结构化证据](../reports/meta65-cublas-preflight-20261011/gpu-preflight/status.json)的status、verification、environment、input-binding、microcells、supervisor六份JSON与服务器原件逐一SHA相同。[原始产物清单](../reports/meta65-cublas-preflight-20261011/gpu-preflight/raw-manifest.json)绑定服务器保留的15个共享身份NPZ、9个预测NPZ和6份运行JSON共30个文件及supervisor sidecar SHA。服务器隔离代码根已同步这些精简证据；用本次内存中的候选绑定数据验证了六份归档的状态／环境／代码／输入SHA及30份原始产物SHA，并再次执行`verify_gpu_preflight_output()`，全部通过。原始NPZ仅留服务器，不提交Git。

退出后worker PID 92462不在进程表；一次只读复查GPU6为1MiB/0%且无计算进程（仅该时刻状态）。**正式矩阵0/135、正式diagonal 0/45**；未生成正式approval、未运行正式supervisor、未读取目标库或开展新元训练，也没有受控统计或模型候选结论。旧 `no_candidate` 不变。新正式launch-binding文件尚未生成，`launch_ready=false` 仍为上轮状态；9-cell预检通过不能代替135-cell正式结果。

## 下一步

若用户随后明确要求正式运行，应从已归档的GPU证据构建新launch-binding（包含本次GPU UUID、代码/配置/协议/CPU/GPU证据及原始产物清单），独立校验后提交推送并展示**新commit和binding SHA**，再由用户针对该精确执行包确认一次性授权。正式启动前还需重查GPU6空闲、全体冻结输入和输出路径，创建新的唯一approval；不复用已消费的上次approval，也不自动扩seed/K/库、换卡或推进S8。
