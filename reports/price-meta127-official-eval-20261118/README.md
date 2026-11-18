# 127 官方设置重评：四库K0完成，完整微调运行中

用户要求“那就按照官方的设置和统计口径重新评估吧”。[127协议](../../实验文档/127-官方设置与全行口径重新评估.md)事前固定配方；分支`experiment/price-meta127-official-eval-20261118`基于`254efd946179a7532f15a97d661f979fc96782fc`。本目录为启动期证据快照，接手时必须读服务器最新状态。

## 已完成的零样本结果

完整原始test共7,239行，七份模型与PG共32格、57,912条逐行预测，均已实际推理并独立复算。所有格无溢出/退化，不删行、不去重。官方Q95越低越好；E/M为三个封存训练流的逐格Q95算术均值，不pool预测。

| 数据库 | 官方P K0 | E K0三流均值 | M K0三流均值 | PG |
|---|---:|---:|---:|---:|
| IMDB | 15.4516 | 20.7541 | 22.6555 | 49.4370 |
| STATS | 35.5513 | 41.3817 | 41.1723 | 73.3495 |
| ErgastF1 | 14.3196 | 16.6252 | 17.9581 | 27.4723 |
| Genome | 15.6697 | 47.3289 | 31.0004 | 373.2727 |

P的IMDB/STATS/Genome对齐论文截图15.45/35.55/15.67；ErgastF1对齐发布README14.3196，与截图14.57不同。PG Genome发布公式373.2727，截图383.09。保留来源差异，禁止逐库换公式对齐。当前M0的各库三流均值仍都差于P0；四库×三流等权log-Q95配对的M0/P0几何比1.32405，仅为零样本描述，完整微调效果尚未产生。

证据：[当前绑定](reselected/binding.json)、[逐格指标](reselected/zero-units.json)、[完成](reselected/zero-completion.json)、[独立复算](reselected/zero-independent-verification.json)、[监督](reselected/zero.supervisor.json)。GPU6重选后的32格未舍入Q95与初次GPU0结果逐格完全相同。

## 官方设置与比较边界

锁定[StCarmen/PRICE 487f8d8669b5592b537ce294b6902378a500c256](https://github.com/StCarmen/PRICE/tree/487f8d8669b5592b537ce294b6902378a500c256)，[七文件SHA](upstream/manifest.json)与原文件归档。服务器历史args的batch已改64、qerror加clip，不作为官方依据。

- 原始test为IMDB766/STATS2749/ErgastF1 1422/Genome2302行；126的unique分母521/2603/1258/1699另存。
- 发布Q-error：float32 `exp(z-1)-1`，转float64加1，双方比值max，`nan_to_num(nan=inf)`，numpy线性percentile及numpy标量round4，无clip。保留未舍入值、逐行编码/基数及标准max(card,1)诊断，不混主表。
- 每库50,000行、100epoch、有效batch15,000、Adam、README示例最大LR1e-5/weight decay5e-5、原生MSE。每epoch三个15,000批和一个5,000尾批，400更新/500万呈现；OneCycle参数同发布脚本，每epoch末调度一次；不挑best/提前停止。
- 8卡DataParallel改为单卡固定micro512、按实际样本数加权累积；有效batch/更新语义一致，Dropout RNG、FP32累加与推理分批有数值差异，不称逐位复现。原生fanout不额外取log，padding依上游；训练取本库最大长度，测试统一(9,12,7,10)。
- P及E/M三流来自125封存链，每格从原初始化独立开始，优化器/RNG重置；不按目标成绩选择LR/epoch/初始化。

finetune/test重叠为STATS41个不同hash/41原始行，ErgastF1 294/294，Genome164/166。保留官方全部50,000行，因此是**有已知训练/测试重叠的发布工作负载复现**，不能作clean holdout证据。126排重少样本结果不变。

IMDB虽有50,000条SQL和CSV归档，但缺完整finetune特征，发布统计缺`imdb_cc.movie_id`等列，历史生成报错，三份PRICE压缩包也缺此文件。见[缺口证据](data-availability.json)。IMDB本轮仅K0，微调标`blocked_missing_official_features`，不造特征、不借test、不静默筛SQL。

## 执行、预算与交接

首次根`/gly/weilin/price-meta-127-official-eval-20261118`、binding`903c1f2f…ab8cf4`，GPU0 K0完成62.3117秒，整卡峰值4,952,424,448B。probe预检发现外部进程后拒绝启动，pid/pgid/returncode为null，未执行训练更新；保留[预检失败](probe-preflight.supervisor.json)，不覆盖。

主代理重新检查空闲GPU6，在新根`/gly/weilin/price-meta-127-official-reselect-20261118`重绑`c7f123f9a8e9a1696c7c38ec1512ee1ef70d0376f84791c93beee38c3090fc0c`。49份执行依赖和配方相同，新根重新执行K0，不导入旧completion。见[重选](reselected/reselection.json)、[授权](reselected/authorization.json)。这是主代理明确重选，各阶段仅一次，无自动重试循环。

新K0 55.2083秒、整卡765,460,480B；32格独立复算通过。STATS/P完整一epoch探针50,000行/4更新，macro耗时5.80056/5.15403/5.05750/1.72036秒。总wall55.6487秒，整卡5,802,819,584B（5.4043GiB）、RSS2,266,316,800B、输出3993B，均在30min/12GiB/24GiB/64MiB内。探针MSE7.74163仅作工程诊断，权重不用于研究。见[epoch](reselected/probe-epochs.jsonl)、[监督](reselected/probe.supervisor.json)、[成本](reselected/cost.json)、[独立成本](reselected/probe-independent-budget.json)。

冻结公式取最慢macro×400及加载/测试余量，再加30%：单格3212.53秒、21格及固定阶段71663.16秒（19.91h），满足每格8h/全序列72h。此为预测，不是完成时间。另设timeout259080秒，2GiB总输出，失败即停，不自动改batch/扩限/续跑。

序列`2026-11-18T08:51:50Z`后台启动，timeout父PID26147/starttime129757301、序列PID26150、GPU6 UUID`GPU-0050e82c-8c24-d85d-00d3-b73f12638df0`。归档快照第一格`finetune-P-0-stats`为7/100epoch、28/400更新，完成格数0/21。见[启动](reselected/sequence-launch.json)、[状态快照](reselected/sequence-status.json)、[第一格快照](reselected/first-run-progress.json)。接手只读服务器当前`evidence/sequence-status.json`和对应`outputs/<stage>/progress.json`，禁止重复启动。

## 验证

本地全量1984项OK(22skip)/273.290秒；增加独立复算器测试后1986项OK(22skip)/250.701秒，见[初轮](local-full-suite.log)、[后续全量](local-final-full-suite.log)。之后仅复算器汇总修复与两项回归测试变动，最终127专项59项全部通过/5.255秒（本轮主代理终端输出；另有完整Linux232项日志包含最终127测试），Ruff/格式/diff通过。

完整staged whitespace检查仅报告七份上游原始归档中的15处历史尾随空格；为保持官方字节与SHA不改这些副本。排除该归档目录后staged diff检查通过，七份上游文件均与manifest SHA一致；此例外不适用于本轮自写代码或文档。

Linux初轮部署fixture缺失和float32测试参照错误修复后，核心230项OK(2skip)，两skip是当时未上传的脚本测试，上传后26项全过；新根28项metrics测试全过。最终完整Linux专项**232项全部通过、无skip，92.159秒**，见[最终Linux日志](reselected/linux-final-all123-127.log)。独立只读审查无未解决阻塞，处理见[审查记录](review.md)。[独立脚本](verify_metrics.py)不导入生产统计，逐CSV复核原始身份/基数/PG/编码标签与全部分位数；最终53格另核report，期间`matrix_complete=false`。

微调尚在运行，无完整效果结论。21格完成后执行`verify_metrics.py --root <新根> --stage all`，仅53格及report全一致才认证完整矩阵。本轮无E2E/p-error，不声称复现论文整张表或数据库端到端加速。
