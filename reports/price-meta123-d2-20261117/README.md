# 123 D2单卡成本探针：执行完成，成本门未通过

日期：2026-11-17。用户授权：“那就开始 D2 单卡成本探针吧，找一张空闲卡执行。”

**执行终态为completed；下一阶段仍为NO-GO。** 冻结成本公式给出总17.3374小时和M单run约4.3691小时，分别超过16小时总上限与2小时单run上限。另外外部VRAM计量为0、worker allocator峰值非0，不能认证12GiB显存硬限。没有接续D3校准、D4正式训练或目标评估。

## 运行绑定与资源

- 代码提交：`7dc0dc0aad6665020d448c89448a858e90311ace`，本轮不改生产代码。
- 结果分支：`experiment/price-meta123-d2-20261117`。
- 本地worktree：`/Users/williamlin/Downloads/13/spatial-meta-ce/.claude/worktrees/price-meta123-d2-20261117`。
- 远端独占根：`/gly/weilin/price-meta-123-impl-20261116`。
- binding：`c990d72a98229fcac5c4f98fc9c2f54036c099aa9a6ae8201dc95618abaf5ed1`，启动及结束均复核有效，33个代码依赖与本地一致。
- 初始checkpoint：方案规定的23库v2，SHA `dee83c462944db8c5901ac32da9c844c61df6175dcb04bc50923f96babff070f`。
- GPU：物理6，`GPU-0050e82c-8c24-d85d-00d3-b73f12638df0`，RTX3090。启动前确认1MiB、0%利用率、无该卡compute进程；结束后仍1MiB、0%。
- D2总限额：1800秒、24GiB RSS、12GiB VRAM、512MiB输出；只访问23个训练库。

原PRICE、数据缓存和其他实验只读。本轮授权只含prepare-training和probe，每阶段一次，不自动重试或放宽限额。

## 实际执行

先用CPU生成并封存三条后续训练任务计划，每条8000任务；这是D2前置身份准备，不是研究训练。prepare阶段耗时99.9511秒，峰值RSS约2.1560GiB，VRAM=0，输出270438492字节。三个计划文件SHA由远端再次独立核验一致，见[remote-diagnostics.json](remote-diagnostics.json)。

GPU探针预算扣减准备开销，设为1700秒和266432420字节输出，内存上限不变。两臂从同一P重新开始，各2次闭环更新+50次计时更新，inner LR固定1e-6；完成后测量训练库支持适应和批量推理/压缩CSV成本。探针权重不得进入正式训练。

| 计量 | 真实结果 |
|---|---:|
| GPU探针监督墙钟 | 395.5631秒 |
| 准备+探针监督墙钟 | 495.5142秒，约8分16秒 |
| 探针峰值RSS（外部监督） | 2545094656字节，约2.3703GiB |
| 峰值CUDA allocator（worker） | 938945536字节，约0.87446GiB |
| 外部监督VRAM峰值 | 0，计量不可信，不能作通过证据 |
| 准备+探针输出 | 532077542字节，507.4287MiB |
| 512MiB输出余量 | 4.5713MiB |
| E/M optimizer step | 均52，重载核验一致 |
| checkpoint所有权重 | 均finite |
| 训练库计时CSV | 2000行及49900行，无目标库 |

主要原件：[prepare监督](prepare-training.supervisor.json)、[probe监督](probe.supervisor.json)、[probe完成](probe-completion.json)、[远端重载/行数复核](remote-validation.json)。完成副本与磁盘原件等值、timing SHA匹配、监督returncode=0且failure=null。成功退出时cleanup.attempted=false/verified=false，不把它解释成失败；终态检查未发现本任务残留，GPU已释放。

## 耗时与预算投影

下面E/M更新耗时是各50次计时更新的**最大值**，不是均值、吞吐率或带置信度的耗时预测。

| 输入量 | 真实输出 |
|---|---:|
| E单次outer更新最大耗时 | 0.683793680秒 |
| M单次outer更新最大耗时 | 6.049468774秒 |
| 2000-query计时单元 | 2.364040080秒 |
| 完整query计时外推单元 | 24.402046512秒 |
| setup/io保守量 | 362.831975365秒 |
| verification设计余量 | 60秒 |
| 预测字节外推输入 | 118.340828516字节/行 |

setup/io实现包含短训练探针本身，属于保守估计；verification_seconds为实现中的60秒下限，并非完整独立认证的实测。CSV计时为训练库probe四列文件，正式报告十列文件及独立复算路径不同，这些是预算局限；不以其替代正式阶段计时。

独立按冻结公式重算：

```text
1.3 * [362.831975365
       +6000*0.683793680+6000*6.049468774
       +192*2.364040080+276*24.402046512+60]
=62414.647402848秒=17.337402056小时
```

| 门槛 | 重算结果 | 判定 |
|---|---:|---|
| 全矩阵≤16小时 | 17.3374小时 | 不通过 |
| E每run≤2小时 | 29.6311分钟 | 通过 |
| M每run≤2小时 | 4.3691小时 | 不通过 |
| 单评估单元≤120秒 | 最大外推31.7227秒，含30%余量 | 当前公式下通过 |
| RSS≤24GiB | 约2.3703GiB | 当前探针通过 |
| 输出≤6GiB | 外推约1.9944GiB | 当前公式下通过 |
| VRAM≤12GiB | worker自报0.87446GiB，外部0 | 外部认证未完成 |

原始[budget_projection.json](budget_projection.json)的feasible=false与独立复算一致。该文件中的vram=true仅比较worker allocator数值，不代表外部硬限生效，不能原样宣称所有资源门通过。

## 显存监督的证据缺口

supervisor按其自有`/proc` PID集合匹配`nvidia-smi --query-compute-apps=pid,used_memory`，本次367次采样最终VRAM仍为0；同一探针的worker记录了非零CUDA分配。

结束后的只读诊断显示，当时其他活跃GPU进程的三个NVML PID在本会话`/proc`均不可见（[remote-diagnostics.json](remote-diagnostics.json)）。这支持PID可见性/namespace不匹配的解释，但不能证明它是本次不一致的唯一原因。**零匹配不应被当成零显存；本次12GiB外部限制没有取得可信认证。** 本轮不修改或静默更换冻结监督器，不重跑GPU，保留后续修复事项。

## 结论与下一步

1. D2实际探针已完成，有限更新、权重重载、query输出和GPU释放证据齐全；这是工程成本测量，不是精度结果。
2. 当前冻结预算不可进入下一阶段：完整矩阵与M单run时限均未通过。不能用平均值替换最大值、减少训练流/更新/分母或放宽上限后直接接续。
3. 后续先修复并独立验证外部显存计量，再决定另行修订资源预算或研究矩阵。任何重新计时、改预算或D3执行需明确范围；本轮不自动执行。

本轮只新增阶段授权/真实结果/复核记录，更新进度；未变更生产代码或冻结123协议。[独立审查](review.md)通过，无未解决阻塞或重要问题，确认成本门失败与显存认证缺口的结论。验证命令为已有runner两阶段执行、只读SHA/JSON/矩阵算术、checkpoint重载与CSV行数检查；不冒称本轮重跑全量单测。
