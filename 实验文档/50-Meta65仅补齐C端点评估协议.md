# Meta65：仅补齐 C 端点评估与完整矩阵认证

日期：2026-10-09（Asia/Shanghai）。用户已要求“接着完成剩下的内容”，授权本阶段源侧评估补齐与记录，不授权重新训练、调参、第二 seed 或目标评估。

## 固定输入和执行范围

沿用分支 experiment/meta65-source-screen-20261009。旧正式运行仍为 failed_timeout，保留原目录及所有文件，不覆盖、删除、resume 或改写其状态。本阶段新建独立代码部署和输出目录，stage=endpoint_completion_only。

只使用 C_last3_last3 的原始 last.pt（4000 更新，未适应元初始化），SHA256=58f2f4db6f54cd4578fd6127d5a4aa350dbdc2268a36a057b56f7161cb6097a4。保留原完整 endpoint_plan.json，文件SHA256=ebeb178de15b134722bd9fbc24ff4b4752f795a311580c1a17cb23c550bfd0f0；所有支持/query 身份和顺序不变。26 源 pretrain 缓存与 P1 来源绑定、旧运行来源记录及执行后再认证必须一致。

为消除被中断文件的不完整性，C 的75个端点均在新目录完整计算，不沿用旧30份部分预测。旧 A/B 的各75个完整端点只读复用，并重新认证预测文件SHA、标签、身份和指标。不复制或提交大体积NPZ/checkpoint。

## 不变的评估和门槛

仅 chembl/legalacts/telstra，K=5/10/20/50/100，每K5次支持抽样；各run最大100条支持，K取前缀，query排除完整100条。query分母49900/49404/49900。每个episode从同一个原C初始化重新适应last/3步/lr1e-3/SmoothL1，不跨episode累积；外模型权重前后不变。batch512、原固定FP32前向与FP64标签；原始log prediction不裁剪，standard Q-error按固定max(pred,1)/max(true,1)定义。

三臂共享原模型结构、数据、初始化及训练抽样记录；本阶段不做optimizer.step/backward训练，不重新选择checkpoint。主指标仍为每臂75-cell的4+表unique-query Q95等权算术macro，不pooled、不改K、不改10%主改善和逐db/K最多10%退化门槛。完整225-cell独立认证通过才发布本阶段 decision.json（单seed工程筛选，不能证明显著性、目标泛化或优于PRICE）。旧运行仍failed，不追认为原两小时协议成功。

## 预算、输入认证及独立验证

单空闲RTX3090、显式GPU UUID、OMP/MKL单线程。整阶段（输入哈希核验、75端点、执行后认证、225-cell CPU独立复算）硬上限3000秒，最多5秒TERM/KILL cleanup，只终止自有组；失败留痕、不重试、不延长或缩减评估。

启动前绑定旧监督/状态、原config/data config、checkpoint/历史训练metadata、完整支持计划、A/B CSV/metadata与各原始预测文件SHA；旧文件/目录前后不变。新代码、config及本文SHA记录并在末尾复核。训练元数据必须4000更新/16000episode/query呈现1024000；C backward64000、support呈现1793745，归一化support597915，与A/B sample-plan一致。

计算后输出新的 C 75-cell CSV/原始NPZ及225-cell组合CSV；组合每行明确source_run=retained或completion。暂存 decision.provisional.json，CPU独立验证从225份预测复算Q50/Q95/log-Q95，核对FP64标签、query身份、表数、分母、支持/query隔离、所有输入SHA与原冻结候选门槛。验证成功、末尾输入认证及deadline通过后才发布 decision.json/阶段completed；父监督exit0还须最终确认。失败不发布正常决策路径。

## 记录和验证

新增入口先做RED/GREEN：篡改checkpoint/计划/配置拒绝，组合来源与完整矩阵保护，失败不发布decision；旧all/head/last行为不改。只修改新的补齐入口、独立验证、配置和记录；完成改动文件Ruff、必要完整CPU回归、服务器CPU集成和独立审查后提交冻结，再实际执行。

精简CSV/JSON、真实验证/执行日志、逐库/K指标和预算记录写实验文档/results/meta65-endpoint-completion-20261009及reports。旧49号记录保留，另写本阶段结果并更新99号。Git仅提交本阶段文件及精简证据，不提交runs、cache、checkpoint、NPZ；最终报告区分新阶段成功/失败与原运行failed。
