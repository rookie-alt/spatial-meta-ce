# A1+PG 头输入三 seed 源侧对照（探索性）

> 日期：2026-11-17。范围：23 个源训练库；chembl、legalacts、telstra 三个源验证库；仅 K=0 为主比较。没有读取 imdb/stats/genome/ergastf1，也没有改变协议 122 已冻结的四臂 A 判定。本报告是协议外、见过历史目标诊断后的探索性对照，不是独立盲测或正式 A 判定。

## 设计与执行

- A1：`price_lite`，`log(PG)` 仅作为输出残差锚；A1-PG：同一编码器、同一输出锚，额外把 `pg_log` 输入到残差头。新列零初始化，旧权重初值与同 seed A1 对齐，参数数为 293,905 → 294,033。
- 种子 42/43/44 各使用与 A1 同一份 sealed 训练计划，4000 updates、16,000 backward 调用、`last.pt`、同一 23 个源训练库；逐 seed query presentations 分别 1,619,640 / 1,615,165 / 1,616,880（同 seed 两臂一致）。
- 服务器隔离根：`/gly/weilin/122-a1-pg-contrast-20261117/`；新代码 `pg-code/`，三份新配置 `pg-configs/`。旧 rebound/preflight/certificate 未改；新证书 `pg-certificate.json` 经 19 个代码文件、15 份配置、旧源计划及缓存哈希核验后签发，spec/authorization 为三个 seed 分别绑定。证书的 `signed` 是流程字段与 SHA256 完整性检查，不是密码学签名。
- A1 seed42 的既有源评估复用已封存结果；seed43/44 在各自新训练封存后使用同一源评估计划单独受监督评估；A1-PG 的三次训练和 33 单元/seed 的源评估均由同一 supervisor 监控并绑定 checkpoint、spec、authorization 和预测 CSV SHA。
- 本地全量 unittest 在最终模型/门禁版本上 1806 项通过（16 skip），定向与关键 Ruff F/E 检查通过；独立代码终审未发现阻塞。

## 真实 K=0 结果

主指标是对每个源库 `percentile(abs(max(log_pred,0)-log(max(true,1))),95)`，再按三库等权平均 `macro log-Q95`。越低越好；差值 = A1-PG − A1。

| seed | A1 macro log-Q95 | A1-PG macro log-Q95 | 配对差值 |
|---|---:|---:|---:|
| 42 | 3.057926 | 2.882172 | −0.175754 |
| 43 | 3.017746 | 3.056876 | +0.039130 |
| 44 | 3.255031 | 2.869428 | −0.385604 |
| **三 seed** | **3.110235** | **2.936159** | **均值 −0.174076；样本 SD 0.212372** |

| 源验证库 | A1 Q95 三 seed 均值 | A1-PG Q95 三 seed 均值 | log-Q95 配对差值三 seed 均值 |
|---|---:|---:|---:|
| chembl | 83.383 | 84.925 | +0.000161 |
| legalacts | 14.235 | 10.999 | −0.233892 |
| telstra | 9.856 | 7.668 | −0.288497 |

**判定：方向看似改善，但幅度不足以判定。** 冻结的源侧筛查规则要求 `|mean(Δ)| ≥ sd(Δ,ddof=1)` 且逐库 `mean(Δ) ≤ +0.1`；此处 0.174076 < 0.212372，尽管逐库保护均通过，仍是 `inconclusive`。seed43 的 macro 差值为正，不能写成三 seed 一致改进，更不能声称解决目标库问题。

## 核验与资源限制

- 每个 seed 的 A1/A1-PG 源侧预测各 1,641,244 行；逐行核对 `(database,k,run,query_id,query_hash,true_cardinality)`，三 seed 均 **0 处错配**。每份源评估 33/33 单元、三库 K0 unique-query 分母分别 49,900/49,404/49,900。每份预测与 protocol 的 SHA 校验通过，新臂的每个单元经独立 CSV log-Q95 复算。五次新增运行均有 `completed`、`failure=null`、`target_databases_read=[]`。
- **资源认证缺口**：三次 A1-PG 的 supervisor 样本计数为 2,838/2,755/2,787，均报告 `peak_vram_bytes=0` 与 `gpu_allocated_seconds=0`；实际运行时 GPU0/5/6 的 `nvidia-smi` 出现非零显存与利用率。服务器可见 `nvidia-smi` 报告的 GPU PID 不能在该进程的 `/proc/<PID>` 查到（属其他 PID 命名空间），按 owned PID 匹配的 VRAM 采样因此恒为 0。**0 不是 GPU 零使用的实测结论；本报告不声称 VRAM/GPU 秒预算已有效执行，也不追认这三轮。**
- **2026-11-17 计量修复（仅代码，未重跑）**：新增 worker 侧 CUDA 计量——worker 用 `torch.cuda.memory_allocated/reserved` 与 `max_memory_allocated/reserved` 周期性写 `memory.json`，supervisor 采样时读取该文件（与 status 字段取最大）参与 `peak_vram_bytes` 与 `gpu_allocated_seconds`，不再依赖 PID 归属。CUDA 运行若全程未观测到非零 worker 显存，`BudgetTracker.finalize()` 拒绝出结论，不再把 0 当测量。服务器实测（临时目录、GPU4、256 MiB 合成分配后立即释放）：`memory.json` 记录 `max_allocated_bytes=268435456`，与 `nvidia-smi` PID 不可见的条件无关。该修复**尚未用于任何真实训练**，上述三 seed 结果仍按未认证资源对待；下一轮须在新冻结 spec/authorization 下重新计量。独立复审后补充：`memory.json` 改为每个采样间隔重写，supervisor 对超过 30 秒未更新的遥测判 stale 并失败，避免线程静默死亡后继续用陈旧值计量。
- 代码树、计划与训练分母等证据保留服务器原件；因 GPU 监控缺口，本轮只报告观察到的源侧精度与预算限制，不将其视作合规的正式 122 A 阶段通过。下一步若考虑增加训练量，须先修复 GPU PID/采样认证并独立复审，再单独冻结两臂等预算的新训练计划；不按已经见过的目标结果调参。

## 证据位置

- A1 seed42：`/gly/weilin/122-source-eval-A1-seed42-20261115/output/worker/evaluation/`；A1 seed43/44：`/gly/weilin/122-a1-pg-contrast-20261117/A1-seed{43,44}/source-evaluation/`。
- A1-PG seed42/43/44：`/gly/weilin/122-a1-pg-contrast-20261117/A1-PG-seed{42,43,44}/output/` 中的 `completion.json`、`worker/single_run_seal.json`、`worker/exploratory_source_seal.json`、`worker/eval/A1-PG-seed*/eval_protocol.json` 和 `predictions.csv`。
