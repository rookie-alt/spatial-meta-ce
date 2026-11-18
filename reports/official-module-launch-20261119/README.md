# 官方 26/4 五臂训练启动

用户于 2026-11-19 明确授权：“开始训练吧，先检查一下 GPU，按这个方案跑。”本次按已交付的 132 号方案和 `training-ready-v2` 执行，仅签发训练授权；原未签模板和全部冻结 spec 保留。

代码基准为 `21e6be5451ab778c97c895ff80220c0fe8821288`。五臂为 A1-PG、GroupMean、GroupAttention、RelationTrue、RelationShuffled，各 seed42/43/44，共 15 次；每次 24,000 ERM updates，从头初始化，只取 `last.pt`。训练使用官方 26 库，validation=[]，四库保留为独立固定终点评估集，不参与训练或选择。

资源沿用已准备的逐 run 预算：GPU1 单卡串行，墙钟 30,000 秒、单 run 18,000 秒、CPU 30,000 core-seconds、GPU 24,000 allocated-seconds、RSS 24 GiB、VRAM 2 GiB、输出 2 GiB。累计 GPU 上限 100 小时为预算加总，不是预计耗时。

`run-training-queue.sh` 只依次调用原冻结 runner，不更改训练、采样、模型、预算或监督。队列清单和逐 run spec/auth 均绑定 SHA；每次执行重新认证实际训练输入。任一 runner 非零退出，队列立即停止，不重试、不续跑、不跳过失败项。四库评估不在本队列中。

服务器启动目录为 `/gly/weilin/module26-prep-20261118/training-launch-20261119`；训练输出仍为原 spec 的 `/gly/weilin/module26-prep-20261118/outputs/<arm>-seed<seed>`。实际授权、启动 PID、GPU 检查和启动后状态以本目录后续生成的 JSON/日志为准。

启动前 [服务器复核](prelaunch-check.json) 确认 GPU1 为 RTX 3090、显存 1 MiB、利用率 0%；37 份代码、15 份训练授权和全部输出不存在检查通过。[队列合成验证](queue-synthetic-validation-final.json) 覆盖正常完成、第二 run 非零即停、重复启动和清单篡改拒绝；退出状态路径问题先复现后修复。[独立窄范围审查](independent-review.json) passed，无未解决阻塞。

2026-11-19 01:21:41 +08:00 已启动：[进程记录](launch-process.json) 队列 PID **8503**，首个 runner PID **8518**，当前为 `A1-PG-seed42` 的真实输入复验阶段。启动队列已建立；GPU 训练和 checkpoint 更新将在实际观察后另记，不能把此时的输入复验称为 optimizer 已开始。

队列在服务器独立 session 中运行，SSH 断开不影响执行。查看启动目录的 `queue.status`、`queue.log`、`current-run.txt` 和 `completed-runs.log`；逐 run 的监督记录为输出目录的 `worker.supervisor.json`，完成认证为 `completion.json`。checkpoint 只留服务器，不提交 Git。
