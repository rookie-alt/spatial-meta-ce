# 122 号真实源数据侧 P0 预检（不训练）

日期：2026-11-11。实现基准 `e6b60f04c963dd760d1d6348e86407e61d3f0325`；本次在独立 worktree 分支 `experiment/price-meta-lora-p0-real-20261111` 归档源侧证据，不修改主工作树、协议正文或 PRICE 原目录。

## 结论与精确状态

**真实源数据预检 `data_preflight_passed`，但 122 P0 总门仍未通过、未授权训练。** `p0_gate_passed=false`、`formal_training_authorized=false` 是预检工具写入的状态，不能解释为可启动 A。未运行 4000-step A、B/C/D 或任何研究训练/目标评估；未使用 GPU；仅读 23 个训练库及三个源验证库的 `pretrain` 缓存，未读取 imdb/stats/genome/ergastf1 的任何目标分片。

- 实际训练容量：四组 `[join_hist,fanout,table,filter]=[12,12,7,10]`，绑定 SHA256 `1b7326c0ce8a562c3988ac793f817f8a6834d0925c6068616f8009eeda95f1f8`。容量仅从 23 训练库的 manifest `max_counts` 生成；三源库 offsets 与 manifest maxima 逐行交叉校验，超容量记录 **0**。无截断、自动扩容或排除。
- 原 112 sealed 计划 seed42/43/44 各 16,000 episode：SHA256 与历史 `plan_binding.json` 一致，完整逐条校验训练数据库、support/query 的 query hash、缓存 label/feature SHA256。三源库计划原文件 SHA `7c212763ff3d3ae818f152e297e33afcb607e4f800dad5b1444d7d2c1591b972`；122 增补 ID/hash 的封存计划 SHA `b760ec2383d4fe0048b3a4e4e6aa2e10f815eca9cd23e4e6c7b98ad08a991764`。未重新抽样、未使用历史 `plan_seed=2718`。
- 源评估人口：chembl 50,000 unique→排100→49,900；legalacts 49,504→排100→49,404；telstra 50,000→排100→49,900。每库 5×20 个互斥 support ID/hash，K5 是对应 block 前五，K20 是全部，support/query hash 交集 0。
- 只做一次真实 chembl 源查询的 CPU 零头前向，`torch.equal(prediction, pg_log)` 为 true；这是 PG 恒等验证，不是性能/精度实验。用真实容量在本地 CPU 构造模型重算参数：A0-ref 132,545、A0 132,417、A1 293,905、A2 86,769，A1 LoRA 内循环 10,240。未测真实 workload FLOPs/时延。

## 执行与验收证据

| 命令/步骤 | 真实退出码与结果 |
|---|---|
| `ssh myserver` + 单线程 Python **只读、探索性**扫描 23+3 pretrain `manifest.json/offsets.npy` | exit **0**；初探容量 `[12,12,7,10]`、三源 `unsupported=0`；最终结论以下一行已审 `prepare` 的原始 JSON 为准，未触目标库 |
| `ssh myserver` + 已审 `scripts.price_meta_p0.prepare(data, sealed_plans, output)` | exit **0**；`status=data_preflight_passed`，14 个 JSON 输出（含12份**未执行**的 A 配置模板）；输出 `/gly/weilin/spatial-meta-ce-price-meta-lora-p0-20261111-code/runs/p0-real-source-20261111/` |
| 独立进程重跑**同一生产函数** `build_capacity/scan_source_capacity`、`verify_plan_against_stores` ×3、`verify_source_eval_plan_against_stores` | exit **0**；48,000 episode 的来源与 hash、三源 population 的 ID/hash 均匹配；不冒充独立算法复算，未训练 |
| 单条 chembl 源查询 CPU 预测 `torch.equal(pred, pg_log)` | exit **0**；仅 PG 恒等，未做指标计算 |
| 本地 JSON/hash 校验和真实容量的模型静态构造 | exit **0**；四臂参数和 LoRA 10,240 与实现重算相符 |

源侧结果的可审计摘要在 [preflight_summary.json](../实验文档/results/price-meta-lora-p0-20261111/preflight_summary.json)；另已只读复制服务器工具原始输出 [remote_preflight.json](../实验文档/results/price-meta-lora-p0-20261111/remote_preflight.json)，逐字节 SHA256 为 `fcd8758f9a4db87538c519da37cda128be7462ed4d186a483062f03369b538a9`。输入配置使用已审提交的 `configs/data.json`（SHA256 `a9057d65c7cb220bd21e7cd67dacbc15c0c20dcc87f61d9b3e6470c0ea094d90`），**仅在内存中**把 `cache_root` 指向只读 `/gly/weilin/spatial-meta-ce/artifacts/cache`；按 `json.dumps(sort_keys=True, ensure_ascii=False)` 得运行时配置 SHA256 `8766c14d1c2e434407de445a8bacc29ff5af3d541d028421c217357feea7b522`，未修改磁盘上的数据配置。摘要明确以 `derived_from` 区分工具直出字段与人工派生统计；两次复核中第二次是**独立进程重跑同一组生产校验函数**，不是协议 §7 所称的独立 NumPy 指标复算。原始服务器 stdout/退出码由操作进程观察，除 `preflight.json` 外未逐字归档所有临时命令输出；单条 chembl PG 恒等和本地静态构造的退出码在摘要 `checks` 登记，但原始逐行 stdout 未单独留档。摘要保存容量的 23 个逐库 maxima、cache content hash、manifest 文件 SHA256、三源完整统计、三份 training plan 与源计划 SHA、输出14份 JSON 的文件 SHA；这些远端缓存/大计划 SHA 尚未在本地逐文件复算，原始 `preflight.json` 已可本地复算。大计划（约 462 MB）、缓存和生成的完整 source eval plan 仅保留服务器隔离目录，不提交；精简摘要不含真实标签或目标数据。原训练库缓存从 `/gly/weilin/spatial-meta-ce/artifacts/cache` **只读**，封存计划从 `/gly/weilin/spatial-meta-ce-fomaml-joint-source-20261028-plans` **只读**；代码仅在新的隔离目录 `/gly/weilin/spatial-meta-ce-price-meta-lora-p0-20261111-code`，不修改 `/gly/weilin/PRICE-master`。

## 停止与后续

真实数据侧覆盖已完成，但 P0 §7 的整体认证、真实成本/资源预算与正式运行授权不在本次执行范围；12份配置模板均未使用，`p0_gate_passed=false`。A0-ref 历史 checkpoint 复用资格亦未核验，不能自行复用。**不要把本轮 P0 源数据预检解释为 A/B/C/D 研究执行通过或架构、LoRA、元学习有效。** 如后续要启动正式 A，先按协议另行冻结剩余 P0 门槛与资源上限并取得明确授权。
