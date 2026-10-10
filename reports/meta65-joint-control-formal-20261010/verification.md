# Meta65 formal 本次启动阻塞证据

日期：2026-10-10，Asia/Shanghai。只操作指定 worktree `/private/tmp/spatial-meta-ce-meta65-formal-safety-20261010`；冻结 HEAD `ac893b65bcf078dff940fa30ebcff72c60bdc4c6`，分支 `audit/meta65-formal-safety-20261010`，接手时工作树干净。

## 实际命令与结果

- `git status --short --branch`：分支正确、没有未提交文件；`git log -3 --oneline`：HEAD 为 `ac893b6`，前序为 `260eb07`、`2d0c060`。完整 HEAD 用 `git rev-parse HEAD` 在断言中核对。
- 本地 `python3 -B -c` 调用 `verify_launch_binding(binding, config_path=..., root=...)`：exit 0，`local_launch_binding_verified`，9 项代码、7 项来源/配置/协议、6 项 CPU 预检文件通过；同时断言 HEAD 和 binding SHA 等于用户指定值。首次只读调用误用位置参数，exit 1/`TypeError`，随后按现有函数的 keyword-only 签名正确核对；未改函数或实验代码。
- 通过 `ssh myserver` 执行两项 `timeout 10s nvidia-smi` 查询：SSH exit 0，完整 stdout 见 [GPU 快照](gpu-snapshot.txt)。8 卡全部有计算进程，显存 6678-10566 MiB、利用率 7-99%，没有符合三项空闲条件的卡。沙箱内 SSH 曾报 `Operation not permitted`；审批后只读查询成功，这不是正式实验失败或重试。
- 阻塞后仅补记远端代码来源：隔离根 `/gly/weilin/spatial-meta-ce-meta65-formal-safety-20261010-code` 内，`CUDA_VISIBLE_DEVICES="" /gly/weilin/PRICE-master/venv/bin/python -B -c` 调用同一 `verify_launch_binding`，exit 0/`remote_launch_binding_verified`，9/7/6 项文件通过；断言 binding SHA 为 `f7a2332d4e61518a340f3faf3c1ad08924660d92ec7739cb9914450cc42a3952`。未导入 Torch 或反序列化 checkpoint。此检查不认证远端 Git revision，也不重新认证完整 plan/checkpoint/cache 的现场内容。
- `git remote -v`：origin 为 Gitee `william-forest/spatial-meta-ce`；主目录未切分支、修改或提交。

指定 worktree 缺少 `memory/spatial-meta-ce-agent-delegation.md`，已只读补读主目录相应文件；委派不扩展权限。独立代理仅复核本次阻塞证据与结论边界，无 GPU/SSH/改码/提交权限。

## 未执行与资源边界

用户本轮已直接授权固定 135-cell 矩阵；本次按“无空闲卡立即停止”终止于 approval 之前。没有创建或消费 approval/收据，没有 formal invocation、GPU UUID 选择、输出运行目录、监督器或 worker；没有自动重试、等待 GPU、终止他人进程或放宽预算。

完整分母/身份/FP64真值/表数/参数更新范围/初始化隔离、135 cells、45 diagonal 与独立 CPU 结果认证均未执行，不能用旧 CPU preflight 代替。没有新预测或部分矩阵，因此不发布受控统计结论，不能判断初始化差异、算子影响或交互。

正式适应步数、元训练步数、query 呈现均为 0。正式 wall/RSS/VRAM/output 没有采样，记录为 null 而非虚构峰值；快照中的显存属于其他已有进程。300/3600/900/4800 秒、单线程、4 GiB RSS/8 GiB VRAM/1 GiB output 冻结上限保持不变，但本轮不认证其 formal 实测通过。

精简结构化状态见 [startup-status.json](../../实验文档/results/meta65-joint-control-formal-20261010/startup-status.json)；结果入口见 [59号记录](../../实验文档/59-Meta65正式矩阵启动前GPU阻塞记录.md)。本次只新增记录与更新进度，不改原 launch-binding、56/58 号冻结记录、代码/config/PRICE/缓存/权重、旧 `no_candidate` 或基线。不重建共享证据索引，直接使用本次文件与原 binding。

## 归档验证与同步

本轮 `python3 -B -` 只读归档校验 exit 0：`json_and_snapshot_consistency=passed`，8 GPU 行/8 compute 行逐项匹配，10 个新增文档链接目标存在，冻结 tracked 文件未改，`formal_result=false`。校验还断言所有 GPU 均超100MiB/利用率>0/有PID，0cells/0diagonal/0formal invocations、没有 approval 或重试、正式资源四项为 null，以及当前 binding SHA。首次归档校验因 Git 默认将中文路径显示为转义形式而断言失败；改用 `git diff --name-only -z` 的结构化路径输出后通过，没有改实验或放宽断言。

`git diff --check` exit 0，无空白错误；新增文件凭据模式扫描 `rg -n -i 'password|token|secret|id_ed25519|sshpass|api[_-]?key|BEGIN.*PRIVATE'` exit 1/无匹配。实际 diff 仅追加进度第60节与本次4个新记录；没有大型预测/权重，也没有实验代码/config/冻结文件变更。

独立只读代理实际阅读全部5文件，执行 `nl -ba`、第60节 `rg`、`shasum -a 256`、Git 只读查询和标准库 `python3 -B -c` 严格 JSON/CSV/状态/hash/链接断言：exit 0/`PASS`，未发现实质一致性问题。代理只独立核对本地 binding hash及归档一致性，没有SSH、GPU、重跑门禁或修改文件；远端通过来自主代理保存的命令结果，不宣称代理独立认证了远端。复核结束后代理已关闭。

归档只在本任务分支提交/同步；具体归档提交 SHA 和推送结果以 Git 与本轮收尾消息为准，执行冻结提交仍是 `ac893b65bcf078dff940fa30ebcff72c60bdc4c6`，不将文档归档提交偷换为新的实验执行冻结点。没有重跑历史代码测试或 CPU 预检，不沿用历史通过数作为本轮验证。
