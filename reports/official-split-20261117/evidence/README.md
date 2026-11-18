# 官方 26/4 划分结果的精简原始证据

2026-11-18 从只读服务器目录 `/gly/weilin/official-split-20261117/` 同步。23 份 JSON 保留原始字节，合计 94,317 字节；[manifest.json](manifest.json) 记录本地 `path`、服务器 `source_path`、大小与 SHA256，均与 SSH 读取的源文件摘要逐项一致。本地 `training-4k/` 对应服务器 `runs/`，仅更改目录名以避开生成运行目录的 Git 忽略规则。

- [4k 六次训练完成记录](training-4k/completion.json) 与 [checkpoint/计划绑定](training-4k/worker/training_records.json)。
- 24k 三次训练完成记录：[42](runs-24k-seed42/completion.json)、[43](runs-24k-seed43/completion.json)、[44](runs-24k-seed44/completion.json)，各目录保留对应 `worker/training_records.json`。
- unique-query K=0：[4k 汇总](target-eval/worker/summary.json)、[24k 汇总](target-eval-24k/worker/summary.json)、[PG/PRICE 汇总](target-baselines/worker/summary.json)，各自的 `completion.json` 保留执行状态与预算记录。
- all-rows K=0：[42](allrows-24k/seed42/summary.json)、[43](allrows-24k/seed43/summary.json)、[44](allrows-24k/seed44/summary.json)，同目录 `protocol.json` 保留 checkpoint 与输入绑定。

all-rows 原始摘要中的 `evidence_level` / `needs_evidence_reason` 是旧脚本硬编码的单 seed 描述。此次实际有三个独立目录；原文保留，不能把静态文字当作本次运行数量。all-rows 与 unique-query 是不同总体，二者不能混合聚合。

复算结果：A1-PG 的四库 macro log-Q95 从 4k 的 `4.36325680173097` 变为 24k 的 `3.8548526365434395`；三个 all-rows STATS Q95 分别为 `238.83452356993018`、`158.41620379914926`、`105.06717682649786`，算术均值 `167.43930139852577`。PRICE 为 `35.551276813587485`，PG 为 `73.34986293961228`。完整解释见 [127 号结果](../../../实验文档/127-24k剂量实验结果.md)。

这里只归档既有结果，不启动训练或推理、不修改服务器原件、不补签任何授权，也不把探索性结果提升为正式盲测结论。checkpoint、缓存与逐查询 CSV 保留在服务器。训练 spec 与授权原件位于本目录上一级；其冻结 SHA 与历史代码版本绑定，不代表当前代码可直接复用历史授权执行。

本次先提交结果档案；关联实现仍在任务 worktree，尚未交付。独立审查已确认结果数字与绑定链，另登记两个后续代码问题：长步数声明需限定探索范围，spec 的模型模块哈希覆盖需补齐。该限制不改变已归档数值，也不应把本提交单独视作可执行复现代码。
