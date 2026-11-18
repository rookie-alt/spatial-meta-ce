# A1-PG relation attention 实现记录

日期：2026-11-18。状态：模型与本地合成验证完成；数值别名、可变保存身份和容器类型问题已修复，最终独立复审通过，无未解决阻塞。源码按通过版本保存本地，未推送或合并。所有前向、反向、适应和 checkpoint 测试均为 `synthetic-smoke-not-research`，不提供精度或迁移收益证据。

起始与独立复审基准 HEAD 均为 `11069d8060e99df4e96a276d3462a692ef378a95`，分支 `feat/a1-pg-relation-attention-20261118`，worktree 为 `/Users/williamlin/.t3/worktrees/spatial-meta-ce/t3-3d588399-9e6111f5ce74ddc9`。起始工作区干净，只在此 prepared worktree 中修改。审查对象为相对此基准的候选 diff。

启用方式：向已有规范 A1-PG 配置增加 `"relation_attention": true`。要求 `arch="price_lite"`、`pg_head_input=true`；其余维度、层数、head、dropout 与原路径一致。规范模型身份为 `price_lite_pg_relation_residual`。未启用时不增加 config 字段或 state 参数，原始 defaults、checkpoint 类型和 `p+r` 输出保持。实际合成规范配置见 [synthetic-model-config.json](synthetic-model-config.json)，其中 capacity、manifest/cache hash 全为合成 fixture，不能用作真实运行绑定。

实际修改入口为 [price_model.py](../../spatial_meta_ce/price_model.py)、[model.py](../../spatial_meta_ce/model.py)、[checkpoints.py](../../spatial_meta_ce/checkpoints.py)，新增 [price_relations.py](../../spatial_meta_ce/price_relations.py) 和 [专项测试](../../tests/test_price_relation_attention.py)。关系模块在全部基线与 LoRA 权重初始化后包装现有 vendor modules，保留共同权重名称，不消耗额外随机初始化；vendor 文件和 provenance 未改动。checkpoint 沿用 config SHA256、规范配置比较、capacity/cache 绑定和 strict state 加载，拒绝基线/其他候选类型及配置、hash、参数缺失等篡改。

固定关系定义如下。矩阵第 `i` 行是 attention query，第 `j` 列是 attention key；所有索引均为查询内局部 token 索引。

| ID | 类别 | 条件与方向 |
|---|---|---|
| 0 | none | 不符合其他类别；CLS 与其他 token 的边；padding 行/列 |
| 1 | self | 有效 token 的自身，包括 CLS |
| 2 | column_to_table | join 或 filter 列 → 其所属 table |
| 3 | table_to_column | table → 其所属 join 或 filter 列 |
| 4 | fanout_to_join | fanout → `fanout_join_columns` 指定的起始 join 列 |
| 5 | join_to_fanout | 上述边的反方向 |
| 6 | paired_fanout | 邻接的 `(2k, 2k+1)` 两个有向 fanout token，两个方向同类 |
| 7 | same_table | 共享局部 table 归属的两个 token；fanout 继承其起始 join 列归属 |

优先级由高到低为 `self > paired_fanout > fanout/join > column/table > same_table > none`。三个显式边家族 token 类型互异，同一级内没有冲突。same-table 先填，再由显式边和 self 覆盖。独立方向打乱可能使一对 fanout 指向同一 join 列；只要两端完整且合法便接受，paired-fanout 覆盖 same-table，不按结果筛选或修复打乱关系。

固定容量为 `(J,F,T,C)` 时，位置为 `CLS=0`、`join_start=1`、`fanout_start=1+J`、`table_start=1+J+F`、`filter_start=1+J+F+T`。scale 阶段使用完整矩阵的左上 `1+J+F` 子矩阵；filter 阶段使用完整矩阵。四个 block 的每个 head 独立持有 8 项 bias，执行 `QK/sqrt(d) + bias[relation]`，随后沿用 vendor 的 key mask、softmax 和 dropout。

实际测量：A1-PG 为 **294,033** 参数，relation 为 **294,161**，新增 **128**（`2 stages × 2 blocks × 4 heads × 8 classes`）；relation + 既有 Q/V LoRA 为 **304,401**，其中 LoRA 为 10,240。true/shuffled 两种 mapping mode 使用同一配置、类型、权重及参数量，差别只在传入的 indices。测量命令与源码 SHA256 见 [evidence.json](evidence.json)。

关系开启时必须提供单一合法 `mapping_mode` 与恰好三种 mapping LongTensor。模型校验 batch/shape/device、counts 的有限非负整数性及容量、至少一个 table、各组 prefix mask/count 一致、有效 mapping 的非负及真实 token 范围、padding 恰为 `-1`、fanout 有效数量为偶数且每个邻接有向对完整。关闭时仍拒绝 mapping。direct batch API 要求显式 `databases`，拒绝 `imdb/stats/genome/ergastf1`（含大小写/空白变体）。数据库名称只用于准入校验，不用于关系类别。关系构造不读取 PG、标签、误差或特征数值。

实际验证环境为 `/opt/anaconda3/bin/python3`，Python 3.11.7、Torch 2.3.0、CPU，`PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`。以下命令均由 [verify.py](verify.py) 执行并保存 returncode：

| 检查 | 实际命令 | 结果与日志 |
|---|---|---|
| TDD red | `python3 -B -m unittest discover -s tests -p test_price_relation_attention.py -v` | 20 项，25 个 error/subtest，rc=1；缺失开关/关系模块，[red-focused.log](red-focused.log) |
| 专项 green | 同上 | 20 项通过，rc=0，[first-green-focused.log](first-green-focused.log) |
| 基线回归 | `python3 -B -m unittest discover -s tests -p test_price_model.py -v` | 13 项通过，rc=0，[first-green-regression.log](first-green-regression.log) |
| 全量 | `python3 -B -m unittest discover -s tests -v` | 1,847 项，199.809 s，`OK (skipped=16)`，rc=0，[validation-full.log](validation-full.log) |
| 仓库 Ruff | `ruff check scripts tests spatial_meta_ce` | Ruff 0.11.2，rc=0，[validation-ruff.log](validation-ruff.log) |
| 修改文件 F/E | `ruff check --select F,E --ignore E501` 加本任务六个 Python 文件 | rc=0，[final-ruff_changed.log](final-ruff_changed.log) |
| 空白检查 | `git diff --check` | rc=0，[final-diff.log](final-diff.log) |
| 保护边界 | `git diff --exit-code HEAD -- vendor spatial_meta_ce/mappings.py spatial_meta_ce/train.py spatial_meta_ce/price_protocol.py 实验文档/130-双分支模块实现计划.md 实验文档/99-进度与下一步.md` | 无差异，rc=0，[final-protected.log](final-protected.log) |

完整命令参数/环境/returncode 可查 [red-checks.json](red-checks.json)、[first-green-checks.json](first-green-checks.json)、[validation-checks.json](validation-checks.json) 和 [final-checks.json](final-checks.json)。全量的 16 个 skip 均为既有 macOS 缺失 Linux `/proc` / non-reaping waitid 等能力的测试；没有新增 skip、删除测试或放宽 tolerance。没有历史 Ruff 告警需要本轮清理。新建的 Python 文件用 `ruff format` 格式化，既有源文件未作无关格式调整。

专项验证包含手工构造的精确关系矩阵、容量偏移和方向；同表/显式边优先级；padding 的 NaN/Inf 不影响输出；singleton、空组、零容量及真正零宽输入；live residual head 下同初始化共同权重和零 bias 输出逐位等价；非零 bias 改变注意力/模型输出并获得有限梯度；true/shuffled 结构差异和相同参数；mapping 与目标库负例；strict functional replacement；`all/head/last/LoRA` 适应与外循环梯度；LoRA/普通 checkpoint roundtrip、model_type/config/hash/state/cache tamper 拒绝。

源侧 sidecar 认证、真实 token 顺序与 query_hash/cache 的完整绑定仍 **pending**；模型校验不能认证调用方声称的数据库身份或 pretrain 来源。`MappingStore` 原 source/pretrain-only、target 拒绝及绑定逻辑未放宽，也未生成任何 target sidecar。当前 `train.py` 仍只为 `sum_map` 加载 mapping，`fixed_model_scope` 不允许此候选；因此本交付 **不具备生产训练运行器接线或新的执行授权**，不能借用 122/A1-PG 已签发 spec。后续真实运行需先完成源侧认证、独立规范/运行器绑定与明确预算授权。未访问真实 cache、SSH 或 GPU，未开展目标库分析或真实目标评估，未安装依赖，未启动真实训练；全量单测沿用既有合成 fixture 和仓库内历史证据断言。关系矩阵构造开销尚未测量。

父任务下一步为独立只读审查此 worktree 的实际 diff、新文件和本轮验证日志；本子任务按要求不递归委派。130/进度99 由父任务更新；本实现保留本地待审，未执行提交/推送/合并。

## 首次审查发现与修复验证

首次独立审查 failed，登记两项阻塞，均为新增候选复用的 checkpoint helper 缺口：加载时 Python 数值等值比较接受配置摘要别名；保存时配置与自身比较，允许写出无法按声明身份重载的文件。关系矩阵、attention 注入、梯度和适应未发现确认缺陷。

主代理先新增 2 项测试复现 red，见 [失败日志](checkpoint-review-red.log)，再作候选专属校验：加载 relation 类型时要求重建 canonical config 摘要等于保存摘要；构造完成后独立保存 relation 配置与 state 名称/形状，保存前核对 model_type、runtime 开关、容量、counts、config digest 与 state 结构。合法学到的权重数值不受此结构校验限制；旧模型兼容路径不变。失败在文件写入前发生，反例同时核对既有 checkpoint 字节未变且无临时文件。

普通及 LoRA 路径均覆盖 `d_model=64.0`、`dropout=0/False` 加载拒绝、配置/类型/开关/容量/state 形状与 key 改写的保存拒绝，以及两步合成适应后正常保存/重载。该修复版本关系专项共 **22 项**，PRICE 回归共 **109 项通过、rc=0**，见 [green 日志](checkpoint-review-green.log)。完整 unittest **1849 项，OK（16 skipped）、rc=0**，见 [该版日志](review-fixed-unittest.log) 和 [命令记录](review-fixed-unittest.json)；记录含一组源码 SHA、明确 Python 路径/版本、线程设置及日志摘要，不能由此独立核验历史开跑前后的两组源码摘要。该版执行脚本内曾断言前后相同，但未将两组摘要分别落盘；现已修正原先过强表述。

最终检查使用明确可执行路径：Ruff **0.11.2** 全仓 rc=0；Ruff **0.17.5** 全仓为共同基础相同的 **624** 条既有告警、rc=1；0.17.5 改动文件 F/E 排除 E501、diff-check、受保护路径检查均 rc=0。详见 [最终检查记录](review-fixed-checks.json)，不把旧版本通过推广到所有版本。

原 [evidence.json](evidence.json) 保留为修复前版本的源码/日志快照；旧 verify.py 的事后摘要不能单独证明历史命令运行时源码。本次修复后的源码与日志以 `review-fix-input.json` 和上述开跑绑定记录为准。工程复审仍待独立结论；真实 source mapping、训练入口、预算授权和精度结论的 pending 状态不变。

## 容器身份补充修复

首次修复后的独立复审仍 failed：将配置 capacity 的 `max_counts`、`databases` 或 `database_max_counts[db]` 从 list 改为同值 tuple，digest 不变，先前校验仍会保存不可重载文件。主代理补充 [red 反例](container-red.log) 确认后，在 digest 之外增加配置与独立构造配置的结构比较，运行时 `model.capacity` 同样检查。两种比较共同覆盖容器和数值别名，保存校验仍在目录创建与写入之前。

新增 1 项测试，关系专项现 **23 项**，覆盖普通/LoRA × 配置/运行时容量 × 三类容器修改，确认既有文件字节保留、无临时文件、新父目录不创建；PRICE 回归 **110 项通过，rc=0**，见 [green 日志](container-green.log)。本次全量记录 `container-full-unittest.json` 实际分别保存 `source_sha256_before`、`source_sha256_after` 和 `source_unchanged`，供独立核对，不再使用仅一组摘要的记录作过强陈述。

本次完整 unittest **1850 项，OK（16 skipped），rc=0**，见 [全量日志](container-full-unittest.log) 和 [before/after 命令记录](container-full-unittest.json)；两组源码 SHA 相同且 `source_unchanged=true`。最终 Ruff 0.11.2 全仓 rc=0，0.17.5 全仓仍 624 条既有告警、rc=1，改动 F/E、diff/protected rc=0，见 [静态检查](container-checks.json)。最终审查清单为 `container-review-input.json`，修复前清单保留各自历史版本；最终工程结论见下一节。真实 sidecar、运行器、预算授权及精度验证门槛保持 pending。

## 最终独立复审通过

[双分支容器身份修复复审](t3-thread://v2/64920c42-55dd-48e3-bc10-fb310e06e3e4/thread:delegated-task:command%3Amcp%3A29ca037a-4fc9-4fdb-a239-38f6b3e01754%3Adelegate-task%3Acd6cd0eb-3c5d-4068-9d79-484315f90f96) 返回 **passed（无未解决阻塞）**。relation 的加载别名、保存身份和文档证据问题全部关闭。审查核对实际 diff、全部文件/日志绑定与 before/after 摘要，独立重跑 PRICE **110 项**；不同 dtype 两步适应后保存/重载 **14 组**、无效保存保护 **48 次**、旧路径别名 **6 组**、原实现直接比较 **5 组**，全部通过。完整 **1850 项、16 skipped、rc=0** 的日志及源码绑定已核验，复审未重复全量。

审查绑定清单 [container-review-input.json](container-review-input.json) SHA256 为 `d3a94884bd5707eef8dcab67d68da0b594a9c82c8a6c6d18a7e9dac5ec1a271e`。此后仅追加结论并更新文档当前状态，源码和测试保持清单字节；历史 README 摘要不改写。通过范围为候选工程与合成验证，真实 source mapping、训练运行器及精度比较仍 pending。
