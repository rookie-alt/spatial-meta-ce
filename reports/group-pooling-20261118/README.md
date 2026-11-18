# A1-PG grouped pooling 实现与合成验证

日期：2026-11-18。用途：`smoke-not-research`。只实现 grouped pooling；没有真实训练、GPU、SSH、真实缓存或目标库分析，没有精度收益结论。首次哈希修复曾通过独立复审并保存本地提交 `e432454`；随后追加容器身份修复，最终独立复审同样通过，无未解决阻塞。未推送或合并。

工作目录：`/Users/williamlin/.t3/worktrees/spatial-meta-ce/t3-301bc0d8-051470e3a3f3b8df`。分支：`feat/a1-pg-group-pooling-20261118`。起始与复审基准 HEAD 均为 `11069d8060e99df4e96a276d3462a692ef378a95`，审查对象为相对此基准的候选 diff。未修改父任务的 130/99 号文档、vendor、训练门禁或冻结 spec。

代码位于 [price_model.py](../../spatial_meta_ce/price_model.py)、[model.py](../../spatial_meta_ce/model.py)、[checkpoints.py](../../spatial_meta_ce/checkpoints.py)；专项测试为 [test_price_group_pooling.py](../../tests/test_price_group_pooling.py)。

默认 `group_pooling="none"`，canonical baseline config 不增加此键。仅 `price_lite` 且 `pg_head_input=True` 时可显式使用 `"mean"` 或 `"attention"`。拒绝非字符串 pooling、拼写变化和非 boolean PG flag。配置示例：

```python
from spatial_meta_ce.model import create_model

# capacity_binding 必须是完整且通过校验的源训练容量绑定。
model = create_model({
    "arch": "price_lite",
    "capacity": capacity_binding,
    "pg_head_input": True,
    "group_pooling": "attention",  # 均值对照为 "mean"
})
```

保留最终 filter encoder 的 CLS 和原 `[CLS(64), len_net(log1p(counts))(16), log_cartesian(1), PG_log(1)]` 前缀，按 `capacity.max_counts` 固定偏移，在最终 encoder 的 join_hist、fanout、table、filter 四组分别汇总有效 token，再依次追加四个 64 维 summary；head 输入从 82 变为 338。输出保持 `PG_log + residual`，最终残差层继续零初始化。先构造 baseline，再扩展 head 第一层，保留共享参数、原 head 权重列与 bias 的初始化；新增列使用 Linear 初始化。

mean 除以有效 token 数。attention 使用四个独立 `Linear(64,16) → Tanh → Linear(16,1)`，各组内 masked softmax。打分和值运算前清零 padding，避免 `inf×0`；空行避开全 `-inf` softmax，并将无效权重置为精确零。零容量组返回零 summary，同时给未使用 scorer 保留零导数，strict functional SGD 无需 `allow_unused` 或假 token。

候选身份为 `price_lite_pg_group_mean_residual`、`price_lite_pg_group_attention_residual`，baseline 仍为 `price_lite_pg_residual`。checkpoint 沿用 schema=1，加载时校验 type/config/canonical/hash、训练 cache hashes 和 strict state_dict；保存前绑定初始候选配置，拒绝 runtime 身份、模式或配置被改写。修复后对两种新增类型额外要求重建模型配置摘要等于保存摘要，拒绝 `d_model=64.0`、`dropout=0` 等 Python 等值但哈希不同的别名；旧类型兼容路径保持。没有新磁盘 schema。

以下参数由 `sum(p.numel() for p in model.parameters())` 实测，LoRA 关闭，容量来自现有 synthetic helper；token 容量不影响这些参数量。证据：[parameter-counts.json](parameter-counts.json)。

| 模式 | 总参数 | 相对 A1-PG 增量 | scorer 参数 | head 输入维度 |
|---|---:|---:|---:|---:|
| baseline | 294,033 | 0 | 0 | 82 |
| mean | 326,801 | 32,768 | 0 | 338 |
| attention | 331,029 | 36,996 | 4,228 | 338 |

环境：macOS、`/opt/anaconda3/bin/python3`、Torch 2.3.0。所有测试设 `PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1`。下表命令在本 worktree 执行；同名 JSON 保存 argv 与真实 returncode。

| 验证 | 命令 | returncode 与结果 |
|---|---|---|
| 首次 red | `/opt/anaconda3/bin/python3 -B -m unittest discover -s tests -p 'test_price_group_pooling.py' -v` | `1`；15 项，缺少实现，[red.log](red.log) |
| 首次实现专项 | 同上 | `1`；14 项通过，1 项 mock 将 StorePool 错写为 FeatureStore，已修正，[focused-first.log](focused-first.log) |
| 初次 PRICE 回归 | `/opt/anaconda3/bin/python3 -B -m unittest discover -s tests -p 'test_price*.py' -v` | `0`；102 项通过，[focused.log](focused.log) |
| 首次全量 | `/opt/anaconda3/bin/python3 -B -m unittest discover -s tests -v` | `0`；1842 项、skip 16，[full-unittest.log](full-unittest.log) |
| 保存 flag 反例 red | 专项命令 | `1`；两个 bool/int 篡改反例证明字典等值比较不足，[save-flags-red.log](save-flags-red.log) |
| 最终 PRICE 回归 | PRICE 回归命令 | `0`；102 项通过，[focused-final.log](focused-final.log) |
| 最终全量 | 全量命令 | `0`；1842 项、skip 16、191.027s，[full-unittest-final.log](full-unittest-final.log)、[执行记录](full-unittest-final.json) |
| Ruff 全仓 | `ruff check scripts tests spatial_meta_ce` | `0`；[ruff-full-final.log](ruff-full-final.log) |
| 改动文件 F/E | `ruff check --select F,E --ignore E501 spatial_meta_ce/price_model.py spatial_meta_ce/model.py spatial_meta_ce/checkpoints.py tests/test_price_group_pooling.py` | `0`；[ruff-changed-fe-final.log](ruff-changed-fe-final.log) |
| 空白检查 | `git diff --check` | `0`；[diff-check-final.json](diff-check-final.json) |

baseline 修改前后，5 个 lite/tiny、PG、LoRA 合成配置的 config、model_type、参数总数、完整初始 state SHA、RNG SHA、head features SHA 和非零残差输出 SHA 全部匹配；见 [baseline-before.json](baseline-before.json) 与 [baseline-after.json](baseline-after.json)。容量绑定来自临时 synthetic manifests，临时缓存和 checkpoint 不纳入交付。

专项 15 项覆盖实际分组边界、counts/mask、最终 encoder 接线、任意 padding/inf/NaN、非零 encoder padding、batch/额外 padding 不变性、空行精确零与有限反传、全局零容量组、live head 下四组 scorer 非零权重梯度、strict functional_call、all/head adaptation、meta_loss、checkpoint 往返及 type/config/hash/cache/state 篡改拒绝。保持原容差，未删除或跳过测试；全量 16 项 skip 是既有 Linux `/proc`、waitid 和进程清理集成测试。

候选在旧 `122-A1`/`122-A1-PG` 配置下被 `_validate_fixed_budget` 和 train 精确模型比较拒绝，mock `StorePool` 验证拒绝发生在缓存读取前。旧 formal122、PG exploratory spec/证书不会自动授权新候选。共同基础的长步数白名单范围、部分旧 spec 模型语义 hash 覆盖遗留仍保留，未修改或认证旧 spec。

后续先由父任务独立只读审查，再另行冻结严格 23 库训练、chembl/legalacts/telstra 源验证的等预算比较，绑定完整模型语义 hash、来源和容量，取得真实执行授权。初始化/教师不得见过源验证库；mean/attention 研究比较还需控制新增 head 容量。目标四库继续禁止参与模块、参数、checkpoint 或策略选择。合成验证不能替代真实源数据认证、精度、性能或训练门禁验收。

## 首次审查发现与修复验证

首次独立只读审查结论为 failed：两种模式加载配置 `d_model=64.0` 或 `dropout=0` 并重算文件摘要后，Python 字典比较会接受，但返回 canonical config 的摘要不同。主代理新增测试先复现四个反例失败，见 [red 日志](canonical-load-red.log)；随后仅对两种新 grouped 类型增加 canonical digest 比较，适应两步后的正常保存/重载继续通过。

新增 1 项专项测试，专项现为 16 项；修复后 PRICE 回归 **103 项通过**，见 [定向日志](canonical-load-green.log) 与 [命令记录](canonical-load-green.json)。完整 unittest **1843 项，OK（16 skipped），rc=0**，见 [全量日志](review-fixed-unittest.log) 和 [命令记录](review-fixed-unittest.json)。没有放宽容差或增加 skip。

首次审查还登记四条 Ruff 0.17.5 新测试建议，已局部修正 import 顺序、循环闭包绑定、字典 items 和 with 结构。明确路径 `/Users/williamlin/.pyenv/versions/3.9.9/bin/ruff`（0.11.2）全仓 rc=0；`/opt/anaconda3/bin/ruff`（0.17.5）全仓仍为共同基础的 **624** 条既有告警，rc=1，新增四条已消除。改动文件 F/E 排除 E501、diff-check 均 rc=0。版本与命令见 [lint 记录](review-fixed-lint-versions.json)；曾用 shell 名称运行的 [检查记录](review-fixed-checks.json) 真实显示当时 shell 已指向 0.17.5，不能把该记录称为旧版通过。

首次 [review-input.json](review-input.json) 保留为修复前证据，已被新修复版本取代；修复后待审版本见 [review-fix-input.json](review-fix-input.json)。该清单保留复审时 README 的摘要；本段追加结论后 README 字节改变，四个源码/测试摘要保持一致，不改写历史清单。

## 首次修复版本的独立复审结论

[修复后独立复审](t3-thread://v2/64920c42-55dd-48e3-bc10-fb310e06e3e4/thread:delegated-task:command%3Amcp%3A29ca037a-4fc9-4fdb-a239-38f6b3e01754%3Adelegate-task%3A09ce4d3f-3604-4a13-b8ae-79f636539c46) 于 2026-11-18 返回 **passed（无未解决阻塞）**。复审自行核对 9 份文件/证据摘要、实际 diff 和调用方，独立重跑 PRICE 103 项通过；另有 28/28 组适应→保存→重载、6/6 旧模型数值别名兼容、5/5 原始模型直接比较通过。首次哈希阻塞与四条 lint 关注点均关闭，无新增重要缺陷。

复审清单 SHA256 为 `d7c5ed29900b2dc456d9b562741263d120049679a6842ba24cc96db7aee30e1d`。源码摘要：`price_model.py=d4eec4b77451b1bd6e0b4b0b82cb5fb236fb083224e815f84d87f717a39fa1b6`；`model.py=d60077ed891d61cada60932cfb5d3aa07b84c419457bc0e1eeb08ab2db29cb12`；`checkpoints.py=8cb14d2ccfcd45a06b2c0aa89db831362ec2f20e9f5bb3154bae364ab81cf788`；`test_price_group_pooling.py=e87fe163c6589a4cc091c3a1f06c9b9d3a317ff00ffefcda2b49b80f3c767a39`。

本结论为工程验收；精度、真实源数据认证及运行器执行授权仍待后续，不能用此记录启动真实实验。

## 容器身份补充修复

关系分支复审发现 list→tuple 虽改变配置结构，JSON digest 却相同。主代理对本分支同步复现该问题，见 [容器反例 red](container-red.log)。现保存前同时要求配置与独立构造配置结构相等、digest 相等；对 `model.capacity` 也作相同检查。结构比较捕获容器别名，摘要比较继续区分数值/boolean 别名，两者缺一不可。

新增测试覆盖 mean/attention × 普通/LoRA × 配置/运行时容量 × 三类 list→tuple 修改，验证保存拒绝、既有文件字节不变、无临时文件及新父目录未创建。PRICE 回归现 **104 项通过，rc=0**，见 [green 日志](container-green.log)。本修复相对 `e432454` 为追加差异，旧 passed 仅对应旧源码；最终版本的独立结论见下段。

本次全量 **1844 项，OK（16 skipped），rc=0**，见 [全量日志](container-full-unittest.log) 和 [命令记录](container-full-unittest.json)。记录分别保存 `source_sha256_before`、`source_sha256_after`，并记录 `source_unchanged=true`，不以单组摘要替代 before/after 留证。[最终静态检查](container-checks.json) 为 Ruff 0.11.2 全仓 rc=0，0.17.5 全仓仍 624 条既有告警、rc=1，改动 F/E 和 diff/protected rc=0。最终待审清单为 `container-review-input.json`，旧清单保留各自历史版本。

## 最终独立复审通过

[双分支容器身份修复复审](t3-thread://v2/64920c42-55dd-48e3-bc10-fb310e06e3e4/thread:delegated-task:command%3Amcp%3A29ca037a-4fc9-4fdb-a239-38f6b3e01754%3Adelegate-task%3Acd6cd0eb-3c5d-4068-9d79-484315f90f96) 返回 **passed（无未解决阻塞）**，关闭本分支同步发现的容器身份问题。审查核对实际 diff、全部清单绑定和 before/after 记录，独立重跑 PRICE **104 项**，另有两步适应后不同 dtype 保存/重载 **28 组**、无效保存保护 **96 次**、旧路径别名 **6 组**、原实现直接比较 **5 组**，全部通过。最终全量日志 **1844 项、16 skipped、rc=0** 已核验，复审未重复全量。

审查绑定清单 [container-review-input.json](container-review-input.json) 的 SHA256 为 `1ccc0fc66a693b0a699f51c00c68a78fe15fcdc34ca06e4dd94ab215cd124f4c`。本次只追加复审结论，源码和测试与该清单保持一致；README 的历史摘要不改写。工程通过不提供精度证据，真实源侧训练比较仍待授权与绑定。
