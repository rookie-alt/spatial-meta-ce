# PRICE 123 显存修复、D2复测与源侧启动

日期：2026-11-17。用户授权：“那就先修复显存计量，预算可以适当放宽，继续推进吧”。

**工程显存缺口已闭合，修复后D2与P校准已通过；源训练正在运行，尚无完整源报告或元学习精度结论。** 本轮只推进已冻结的源侧实验，目标库`imdb/stats/genome/ergastf1`未进入执行。

## 隔离与来源

- 本地分支：`fix/price-meta123-vram-budget-20261117`，基准`f6106365b70eb79a39bb4abf0d8d21d3e98ba915`。
- 首次服务器根：`/gly/weilin/price-meta-123-vramfix-20261117`，binding `48204dfc5f5911b611ae04a8d2c777baef0ec85d540d7d492fcfe7bee63123bb`。GPU6被其他任务占用，启动预检正确拒绝；worker未启动，0更新、0B输出、0.203秒，记录保留。
- 实际运行根：`/gly/weilin/price-meta-123-vramfix-reselect-20261117`，binding `5d81667b8ad958c92f15fab4b28dbd5c421e7fe8fe02796846298f200d8934f3`。34个绑定文件与已审本地版本逐项一致；三条训练计划内容SHA与首次准备一致。
- 实际GPU：2，UUID `GPU-cf1ca2b2-761c-66cb-7b36-221d730920c7`；启动前显存1MiB、利用率0、无该卡compute-app。
- 原[123协议](../../实验文档/123-原版PRICE元学习增量实验方案.md)字节与SHA不变；资源按新增[124修订](../../实验文档/124-原版PRICE显存监督与预算修订.md)另行绑定。已有PRICE、旧D2、其他实验不覆盖。

## 实际修改

`price_meta123_supervisor.py`运行中按授权UUID读取整卡显存，记录`device_total`、baseline、UUID/index。该路径不依赖容器PID与NVML PID求交集；监测缺失或异常停止，超限只清理本任务可验证进程。启动前校验index/UUID映射及空闲状态，子进程固定完整UUID。

`price_meta123_runtime.py`绑定124文件并实施累计24h、E每run2h、M每run6h上限；显存12GiB、RSS24GiB、输出6GiB、评估单元120秒不变。进入后续阶段须有同binding成功探针，外部峰值>16MiB且>baseline；旧外部0证据不能通过。模型、2000更新数、任务与指标未改。

`run_source_continuation.py`是本次任务的固定串行入口，单独按SHA写入逐阶段授权。六个训练run后封存，再进行E校准和R报告；任一步失败即停，不自动重试或放宽预算。

## 修复后D2实测

| 项目 | 真实结果 |
|---|---:|
| CPU准备 | 70.8437秒 |
| GPU探针 | 433.9407秒；E/M各52更新 |
| D2合计 | 504.7844秒；532,078,126B（507.4292MiB） |
| 外部整卡显存峰值 | 1,409,286,144B（1.3125GiB） |
| worker allocator峰值 | 938,945,536B（0.87446GiB） |
| 空闲baseline | 1,048,576B（1MiB） |
| 外部RSS峰值 | 2,556,133,376B（2.3806GiB） |
| 最大E/M更新秒数 | 0.734018 / 7.589808 |
| 完整矩阵成本投影 | 20.9865小时，含冻结公式30%余量 |
| M单run投影 | 5.4815小时，含30%余量 |

全部新预算检查通过。两份探针checkpoint优化器step均52、模型全部finite；训练库计时CSV分别2,000和49,900行且无重复hash，timing SHA/完成副本一致；探针结束GPU2回到1MiB/0%。只打开23训练库，探针权重禁止研究使用。外部显存大于allocator是两个计量口径的正常差异。

成本投影由独立审查者用另写公式复算一致。M单run投影距6h约剩8.6%，后续机器负载改变仍可能触发硬限；本轮不自动加预算。详见[实际验证](reselected-gpu2/d2-postrun-validation.json)、[timing](reselected-gpu2/timing.json)、[投影](reselected-gpu2/budget_projection.json)和[审查](review.md)。

## D3与运行状态

S/C/R已固定并逐库验证hash两两不重叠：各库S100、C2000；R为chembl47,900、legalacts47,404、telstra47,900，共143,204。构建计划为身份/一致性校验读取源缓存，报告区在此阶段没有模型前向或指标计算。

P校准完成48单元，监督耗时121.2464秒。独立从96,000行预测重算48格Q50/Q95和J，全部一致，选择α=`1e-6`（非边界，与探针一致，无需`confirm-alpha`）。三候选均finite，但`1e-5`在chembl出现极大误差，按冻结规则保留完整结果，没有裁剪或扩大网格。校准结果不作为源报告或元学习收益。见[校准](reselected-gpu2/p-calibration.json)、[独立复算](reselected-gpu2/calibrate-p-postrun-validation.json)。

后台串行任务于**2026-11-17 15:18:26北京时间**启动，PID **7151**，`/proc` starttime **120556879**。首个状态快照为`train:E:42`运行中；15:27:13的[真实进度快照](reselected-gpu2/continuation-progress-observation.json)已记录E42 **701/2000**更新，训练循环463.845秒，六run尚未完成。运行顺序固定为E42→M42→E43→M43→E44→M44→seal-training→calibrate-e→report。每阶段使用新Adam及共同P，不接探针权重；六个last封存前不打开R做前向。各阶段还会再次做空闲预检，遇忙卡即停，不切卡重启。

已启动仅代表有界实验在执行，**不代表六训练、144格E校准、最多276格报告或科学门已通过**。本地`continuation-status.json`是取回时的快照，之后应核对服务器原件：

```bash
ssh -o BatchMode=yes -o ConnectTimeout=12 -o ConnectionAttempts=1 \
  -o ControlMaster=no -o ControlPath=none myserver \
  'cat /gly/weilin/price-meta-123-vramfix-reselect-20261117/evidence/continuation-status.json'
```

若失败，保留`continuation-failure.json`、对应stage的supervisor/failure和既有输出；不自动恢复、重跑或移除证据。全部完成后核对六个2000步last、seal、E校准及完整R分母/独立指标和科学判定，再记录实际精度。目标评估仍需独立授权。

## 验证与交付范围

- 本地真实全量：1857项，`OK (skipped=22)`、无失败，211.315秒；见[日志](local-full-suite.log)。
- 服务器Python3.11/Torch2.1/Linux：101项123专项，`OK`、无跳过，28.356秒；见[日志](linux-tests-initial.log)。
- Ruff全项目、四个修改Python的格式、Git空白与Markdown本地链接检查通过。
- 固定串行脚本另做两项模拟验证：九阶段成功顺序；第二阶段失败后停止，无后续调用；Ruff/format通过。这些是模拟，不计研究实验。
- 未参与实现的独立只读代理已审生产diff/124、实际D2证据和固定串行脚本，均无未解决阻塞。限制和原件/副本边界见[审查记录](review.md)。

交付包含代码、资源修订、精简JSON与日志；数据、计划大文件、checkpoint留在服务器，不提交。Git提交/推送状态以实际Git及交付回复为准，正在运行的科学阶段另行如实交接。
