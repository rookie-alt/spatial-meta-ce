# server68 四组模块实验结果（2026-11-24）

12/12 新增训练及 15 份权重的四库 K=0 评估已完成。**本轮未观察到稳定整体改善；RelationTrue 未优于同轮 RelationShuffled。** 全部三 seed 和负结果均保留。本目录归档已完成的 132 号执行与历史比较；133 号实验不在此次发布范围。

## 实验内容与协议

GroupMean、GroupAttention、RelationTrue、RelationShuffled 各训练 seed42/43/44，每 run 完成 24,000 optimizer updates / 96,000 episodes，固定取 last.pt。四条 lane 分别使用 GPU0–3，同组 seed 顺序执行。官方 26 库训练、validation=[]，四目标库 imdb/stats/genome/ergastf1 不进入训练或选参。训练完成记录 target_databases_read=[]，37 份冻结源码无漂移。

训练于 2026-11-23 22:36:57 +08:00 启动，2026-11-24 10:49:21 完成。自动评估于 10:49:36–10:50:06 执行，worker exit0、墙钟 24.085 秒、GPU allocated 19.831 秒。

比较协议为 `module26-historical-baseline-comparison-v1`，`strict_new132_matrix=false`、`selection_use=none`。12 份新增终点与 3 份历史 122-A1-PG24k 权重使用同一评估器及四库 query population。旧完整源码、初始化字节和数值环境的完全一致未证明，因此相对 A1-PG 的差值属于历史观察比较，未认证严格 132 的 15 份新训练矩阵完成。详见[可比性与执行记录](execution-and-provenance.md)。

冻结准备源码：[4421e567](https://github.com/rookie-alt/spatial-meta-ce/tree/4421e5671595fbe9fdf93f465362586079efaa90)。132 协议保存在该提交的 `实验文档/132-模块候选官方26训练4评估准备.md`；执行时 37 份源码的逐文件 SHA256 以完整归档为准。本次只新增结果目录及进度入口，不把整个准备分支合入 main。

## 主结果

主指标：每 seed、每库先对逐查询 log-Q-error 计算 95% 分位数（线性插值），再四库等权平均。越低越好。它与先对 Q-error 求 Q95 再取 log 存在插值差异，采用原评估器的 log_q95 字段。± 为三个训练 seed 的样本标准差，非置信区间；Δ 为同 seed 新组减历史 A1-PG。log 差值不能直接当作 Q-error 改善百分比。

| 组 | macro log-Q95 均值 ± SD | 配对 Δ 均值 ± SD | 改善 seed |
|---|---:|---:|---:|
| 历史 A1-PG | 3.854852 ± 0.363834 | — | — |
| GroupMean | 4.121969 ± 0.045304 | +0.267117 ± 0.360439 | 1/3 |
| GroupAttention | 3.815751 ± 0.213457 | −0.039101 ± 0.270677 | 2/3 |
| RelationTrue | 4.058758 ± 0.044700 | +0.203906 ± 0.401209 | 1/3 |
| RelationShuffled | 3.747495 ± 0.728688 | −0.107358 ± 0.833520 | 2/3 |

| 组 | seed42 macro | seed43 macro | seed44 macro |
|---|---:|---:|---:|
| 历史 A1-PG | 3.709123 | 4.268961 | 3.586473 |
| GroupMean | 4.168088 | 4.120291 | 4.077527 |
| GroupAttention | 3.928933 | 3.948782 | 3.569539 |
| RelationTrue | 4.101268 | 4.012151 | 4.062855 |
| RelationShuffled | 4.552564 | 3.556816 | 3.133103 |

同轮 True−Shuffled 的三 seed 差为 −0.451296 / +0.455335 / +0.929752，均值 **+0.311264 ± 0.701706**，True 仅 1/3 seed 更好。genome 的 True 三 seed 均比 Shuffled 差，是主要退化来源。n=3 只作描述性判断，不作显著性或模块因果结论。

## 分库与成本

下表为每 seed 先算 unique-query Q95，再对三个 seed 取算术均值；不是 pooled Q95，也不等于对 macro log-Q95 取 exp。

| 组 | imdb Q95 | stats Q95 | genome Q95 | ergastf1 Q95 | 累计 GPU allocated 小时 |
|---|---:|---:|---:|---:|---:|
| 历史 A1-PG | 21.51 | 188.80 | 83.22 | 24.87 | 不作跨环境耗时比较 |
| GroupMean | 28.60 | 175.08 | 111.32 | 30.36 | 3.2641 |
| GroupAttention | 34.96 | 236.28 | 25.87 | 24.13 | 3.4610 |
| RelationTrue | 22.84 | 134.90 | 167.06 | 28.03 | 11.8500 |
| RelationShuffled | 26.57 | 182.18 | 69.08 | 30.11 | 11.7070 |

GroupAttention 的局部收益集中于 genome，但 imdb/stats 退化。新增训练累计 30.2822 GPU allocated 小时，非并行墙钟或 utilization 积分。all-rows 为独立辅助口径，其 macro log-Q95 三 seed 均值依次为 3.695074 / 3.835611 / 3.556882 / 3.768260 / 3.520781，不能与 unique-query 混合。

## 与 PRICE 的历史比较

历史 PRICE unique-query K0 macro log-Q95 为 **3.223644**；本轮四个新增组的三 seed 均值均更差。RelationShuffled 的事后最佳 seed44 为 3.133103，比该历史 PRICE 低 0.090541，但该组均值为 3.747495（比 PRICE 高 0.523851），seed42 为 4.552564。不能用事后最佳 seed 声称稳定胜过 PRICE。

| 模型 | imdb Q95 | stats Q95 | genome Q95 | ergastf1 Q95 |
|---|---:|---:|---:|---:|
| 历史 PRICE K0 | 21.02518 | 39.52059 | 31.33616 | 15.29288 |
| Shuffled seed44（事后最佳） | 17.10078 | 57.78307 | 10.60467 | 26.45268 |

PRICE 未在本轮重新推理，来源是旧 `official-split-20261117` 的标准 Q-error / unique-query K0 汇总，不能混用 127 号官方全行 float32 解码后 +1 Q-error。原始 PRICE 汇总副本与哈希见 [price-reference.json](price-reference.json)。本比较不用于挑选 seed、checkpoint 或追加目标驱动调参。

## 归档、复核与复现

- [完整归档索引](artifact-index.json)：116,877,756 bytes / 483 文件，含新旧权重、15 份预测、完整训练及审计日志、首轮失败、两轮 launch、评估工具、封存输入、37 份冻结训练源码。权重及大产物不进入 Git 历史。
- [结果 Release](https://github.com/rookie-alt/spatial-meta-ce/releases/tag/module26-results-20261124)：附件是否可下载以 Release 页面为准；提交本目录本身不代表附件已上传。
- [已有迁移与历史基线 Release](https://github.com/rookie-alt/spatial-meta-ce/releases/tag/module26-migration-20261123)：认证数据迁移包及历史基线来源，保留不覆盖。
- [逐 seed 配对比较](comparison.json)、[独立 CSV 统计复核](independent-statistics-verification.json)、[发布验证](publication-verification.json)、[独立发布审查](publication-review.json)。完整 120 单元 Q50/Q90/Q95/Q99 和预测 CSV 在附件内。

独立统计复算未调用生产统计函数：15 CSV / 120 单元 / 199,800 行（unique 91,215 / all 108,585），unique 分母 imdb521 / stats2603 / genome1699 / ergastf1 1258。ID/hash/truth/顺序、有限性及完整分母通过；700 浮点项最大绝对差 3.638e-12，macro、配对差、均值及样本 SD 差为 0。此为历史核验记录；本次发布另检查归档逐文件 SHA、文本来源和链接。

恢复时先按 artifact-index 校验下载包 SHA256，再解包到新目录，逐项检查 `MODULE26-ARTIFACT-MANIFEST.json` 的 bytes/SHA256。`frozen-training-source/` 是实际冻结源码闭包，非完整 Python 环境；还需要已有迁移包中的认证缓存及匹配环境。归档保留原绝对路径和内容绑定，搬迁后须按正式工具重新绑定与验证，不直接重放旧授权命令。当前结果不能作为目标选参依据。
