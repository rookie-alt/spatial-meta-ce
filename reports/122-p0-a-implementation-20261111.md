# 122 号 P0+A 基础设施实现与本地预检

日期：2026-11-11。协议基准：`docs/price-meta-lora-protocol-20261111` / `1e1766def13e42f3704e767dfb81408da8c0ea80`。

## 状态与授权边界

- 交付对象是**可实现、可验证的 P0+A 基础设施**，不是 A/B/C/D 的研究执行结果。
- 在 `.claude/worktrees/price-meta-lora-impl-20261111`、`feat/price-meta-lora-impl-20261111` 中实现；不切换、不修改主工作树的既有改动。
- 编码均由主 agent 完成；前置规划与测试建议只读委派。初审两路已完成并发现阻塞，已按合成负例修复；最终差异只读复审状态见末节。
- CPU 合成单元／集成测试、受限 2-update smoke 已完成，统一为 `smoke-not-research`。未连接服务器、未使用 GPU、未读取真实目标库、未修改 `/gly/weilin/PRICE-master`、未新增依赖。
- **P0 真实数据门未通过／未完成**：本 worktree 无 `artifacts/cache` 和服务器 sealed plan。实际预检退出 2，`unsupported=null`（未知，不是 0）；不得启动正式 A。
- 不修改 [122 号协议](../实验文档/122-PRICE式编码器与元学习低秩适配分阶段协议.md) 正文或将其标为已执行。不执行 B/C/D，不合并 main，不 force push。

## 实现清单

| 契约 | 实现与验收 |
|---|---|
| PRICE 两阶段候选 | `spatial_meta_ce/price_model.py` 直接导入现有 vendor 的 `ScaleEmbedding/FilterEmbedding/Encoder`；不实例化 `RegressionModel`，不存在 `final_linear1/final_linear2` |
| 单路 PG | `CLS + len_net(log1p(counts)) + log Cartesian → 128→64→1`；最后线性层 weight/bias 全零；PG 仅在输出处加一次；lite64/tiny32 |
| anchor-only sum | `SetEstimator(arch="sum_anchor_only")` 少一个 PG head 输入，PG residual anchor 保留；旧 sum 不变 |
| 标签 | 候选沿用 `batch_samples` 的 `log(max(true,1))`，不调用官方 PRICE 的 `price_support/price_log_predictions`；不对原始输出 clamp |
| 输入 | fanout 在 batch 中 `log1p` 后，仅在 PRICE 输入适配处 `expm1`；padding=-1000、table padding 首维=0；容量和 mask/counts 先校验，不能截断真实 token |
| LoRA | 仅两个 encoder 每 block 每 head 的 Q/V；r=4、alpha=4，A 独立 `torch.Generator(seed=42)` 初始化，B=0；A/B 是 `nn.Parameter`，不 merge/delta-cache、不 `.data` 更新 |
| 适配 | all/full/head/last/lora 参数选择（`full` 仅是 `all` 别名）；SGD、`create_graph=False`、`functional_call(strict=True)`；内循环与整条 meta objective mode 一致、恢复调用方所有子模块 mode；fast weights 不写回模型 |
| checkpoint | 新类型 `price_lite_residual/price_tiny_residual/sum_anchor_only_residual`；配置 SHA、容量/hash 绑定；旧 `grouped_set_residual` schema=1、原 config shape、strict state_dict 加载保留 |
| 容量 | `price_capacity.py` 只从 frozen 23 train 的完整 pretrain manifest `inspection.max_counts` 求 maxima，保存逐库 cache hash、manifest 文件 SHA256、逐库 maxima 与 capacity SHA；源验证容量不扩大训练容量 |
| unsupported | 超容量抛 `UnsupportedCapacity`，记录 database/query_hash/group/actual/capacity；无条件核对三源 manifest `max_counts` 与 offsets，低报直接拒绝；不改容量、不排除、不截断 |
| fixed-budget | 旧 112 的 `FIXED_MODEL/FIXED_ADAPT_SCOPE` 默认守卫保留；`fixed_stage=122-A0-ref/A0/A1/A2` 绑定冻结模型与 all scope，仅 ERM；4000 updates、4 meta_batch、64 query、accumulation1、LR/loss/seed/last-only 仍逐项校验；B/C stage 拒绝 |
| A 前门 | 训练必须消费 hash 绑定的 source preflight、完整容量/coverage、sealed plan hash，且 `p0_gate_passed=true`；本轮生成报告保持 false，不能用模板直接启动 A；训练不打开验证分片 |
| population 计划 | 复用现有 112 五组互斥 20-query block，不重采样；122 plan 增加完整 support/query ID、hash、excluded_unique=100、population_primary；K5 是各 block 前5，K20 全20；2718/shot_seed 不可作为主计划 |
| 源评估 | `source_eval.py` / 既有 source CLI 支持 A 四臂；逐 query CSV、33 单元/seed/臂、Q50/90/95/99/log-Q95、空4+桶=null、相对自身K0恶化比例；A gate 只用K0，三个seed的宏差/样本SD、逐库+0.1保护、lite/tiny同方向 |
| 范围守卫 | 通用目标 `evaluate/adapt_one` 对122候选拒绝；不新增正式 B/C/D runner |
| P0 工具 | `scripts/price_meta_p0.py prepare` 仅读源缓存和外部 sealed plan、生成预检及12配置；`smoke` 复用既有 stdlib supervisor，最多2 updates/300秒/4GiB RSS/0 VRAM/128MiB输出、单线程 |

## 参数量与数学限定

本地实际构造（不是研究训练结果）：

| 模型 | 参数量 |
|---|---:|
| sum / A0-ref | 132,545 |
| sum_anchor_only / A0 | 132,417 |
| price_lite / A1 | 293,905 |
| price_tiny / A2 | 86,769 |
| lite Q/V LoRA 额外参数 | 10,240 |

协议 §3.2 的 lite/tiny **静态估计**是 293,777 / 86,641；其规定最终须按实现重算。保留协议明确要求的 `log Cartesian` 作为 head 输入，增加 `1×128=128` 个 weight，因此实际分别多128。没有为了凑静态数删除该特征或改变模型设计；本轮未改协议数字。

零 residual head 时 trunk/LoRA 初始梯度为零，B=0 时 A 的首步梯度为零，均为链式法则的必然结果。测试分别验证零头恒等、B=0 的 A 零梯度/B 非零梯度；使用合成非零输出头检验多步 LoRA 更新 A/B、base 不变、query loss 对 base/head/A/B 初始值的组级非零梯度。不能据此宣称 fresh-zero-head 的 LoRA-only 内循环已经改变预测。

训练容量某组为0时，vendor 不调用该 embedding。wrapper 为该组保留显式零导数，不造 dummy token、不扩容，使 strict inner gradients/gradient audit 仍可处理。`adaptation_mode` 在 `meta_loss/meta_loss_terms` 中显式同步 support/query 的训练 mode；这属于既有 112 调用方的行为扩展，独立只读审查在现有无 dropout/BN 的 sum/set 上实测 train/eval fast weights 逐位相同，未来新增 mode 敏感层时须重验。容量的 `cache_hashes` 与训练 metadata 的 `manifest_hashes` 均是 manifest `content_hash`，不同于另存的 `manifest_sha256` 文件哈希。

## 本地验证与真实退出状态

工作目录均为本实现 worktree；执行环境 Python3.11 / torch2.3.0 / numpy1.26.4。CPU 命令显式 `CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`。

| 命令 | 结果 |
|---|---|
| `PYTHONDONTWRITEBYTECODE=1 python3 -B -m unittest discover -s tests -v`（另加CPU单线程环境变量） | 最终 **exit 0，1784 tests / 175.439s，16 skipped**；完整日志 `122-unittest-final-reviewed.log`；前一轮 `122-unittest-postreview.log` 的1,783项只对应 checkpoint 守卫加入之前的版本 |
| `ruff check scripts tests spatial_meta_ce` | **exit 0** |
| `git diff --check` | **exit 0** |
| `python3 -B -m unittest discover -s tests -p test_price_model.py -v` | 最终 **exit 0，12项**，增加畸形 checkpoint metadata 负例（RED→GREEN） |
| `python3 -B -m unittest discover -s tests -p test_price_protocol.py -v` | 审查修复后 **exit 0，17项**；含 manifest 低报、损坏 checkpoint、null 阶段、失败留痕、目标拒绝不建目录及两步合成端到端 |
| `python3 -B -m unittest discover -s tests -p test_fixed_budget_train.py -v` | **exit 0，13项** |
| `python3 -B -m unittest discover -s tests -p test_source_eval.py -v` | **exit 0，4项** |
| `python3 -B -m unittest discover -s tests -p test_meta65_scope_gradients.py -v` | **exit 0，9项**；spy迁移到工厂，保留全部行为断言 |
| `python3 -B scripts/price_meta_p0.py smoke --output <worktree>/runs/price-meta-p0-synthetic-20261111 --updates 2` | **smoke子命令/worker exit 0**，supervisor=`preflight_completed`；同一 shell 随后的 `python3 -B -m coverage --version` 因无 coverage 包 exit 1，不能混称整条 shell 为0 |
| `python3 -B scripts/price_meta_p0.py prepare --plans <worktree>/runs/sealed112-unavailable --output <worktree>/runs/price-meta-p0-source-preflight-20261111` | **exit 2（预期的缺证据）**；缺 `artifacts/cache/pretrain/accidents/manifest.json`，未扫描源库，unsupported未知 |

TDD：新增测试分别先因缺 `create_model`、`price_protocol`、P0入口失败后实现。集成测试还先复现了零容量 unused parameter、签名不含steps和mode混用问题，再修复。全量第一轮 exit1（1775 tests、1 error、16skip）：旧测试patch `train.SetEstimator`；改为观察工厂后最终全量通过，未删除断言或跳过失败。

**覆盖率口径**：未安装 coverage，也不安装新依赖；使用 stdlib `trace.Trace(count=True)` 执行两组新增测试及两步合成 smoke，修复后输出在 `runs/122-stdlib-coverage-postreview/`。新生产文件可执行行覆盖：

- `price_capacity.py`：111/(111+23)=**82.84%**。
- `price_model.py`：151/(151+10)=**93.79%**。
- `price_protocol.py`：145/(145+28)=**83.82%**。
- `scripts/price_meta_p0.py`：266/(266+40)=**86.93%**。

这是 stdlib 可执行行覆盖，不是 branch coverage、不覆盖全仓、不等价于 coverage.py 的统计。已有模块低覆盖是本轮定向 trace 的分母，不能声称全项目≥80%。

## CPU 合成 smoke（smoke-not-research）

证据在 `runs/price-meta-p0-synthetic-20261111/` 与对应 supervisor JSON；精简归档见 `122-p0-a-validation.json`。

- 2 updates、CPU单线程；监管墙钟3.762827917秒，峰值RSS280,477,696 bytes（约267.5MiB）、VRAM0、输出3,943,037 bytes；全部在上限内。
- 模型内两步更新计时0.232433958秒；checkpoint往返逐位一致；LoRA内循环参数10,240。
- 合成 batch2 的10次前向均值：sum0.581ms、anchor-only0.454ms、lite3.822ms、tiny3.137ms。仅此合成形状、未作真实 workload 计时，不用于结构/速度结论。
- `estimate_4000_synthetic_updates_seconds_not_research_budget≈464.868s` 仅为两步合成外推，**不是实测4000步，不是正式 A 的预算**。正式预算需真实源侧计时/内存与用户另行批准。
- 未取得完整 FLOPs；不能用参数量代替 FLOPs 或端到端加速。

本轮报告另核验最终冻结代码 smoke `runs/price-meta-p0-synthetic-final-20261111/`：**exit 0**，同为2 updates/CPU单线程，supervisor墙钟 **3.267915959秒**、峰值RSS **291,483,648 bytes**、VRAM0，逐位 checkpoint 往返仍成立。`runs/122-synthetic-profile.json` 只含 PyTorch profiler 可计数算子（batch=2 合成容量 `[4,4,4,4]`）：A0-ref/A0/A1/A2 分别 **823,562 / 823,050 / 14,231,554 / 3,729,410** counted FLOPs；不含未支持算子，**不是完整模型 FLOPs 或真实工作负载性能结论**。

## 未完成项与停止点

1. 23个真实训练库容量生成及其完整cache/array来源验证。
2. 三源验证库 manifest max_counts / unsupported 实测扫描；本轮真实 unsupported 为未知。
3. 服务器112 sealed训练/source plan hash与episode逐条核验；本地测试只验证合成同构流程，不冒充真实复用。
4. P0完整真实证据认证及 `p0_gate_passed=true`；正式资源预算/上限和启动授权。
5. A0-ref历史checkpoint复用资格未核验，默认不能复用；12配置按全部从头训练预算生成。
6. 正式A训练/评估、B/C/D、真实FLOPs/时延全部未执行；不声称架构、LoRA或元学习有效。

## 独立只读审查

两路初审只读代理均自行核对了 base `1e1766d`、实际 diff、新增文件、冻结代码 SHA、相关单元测试、全量测试日志 SHA 和真实缺证据状态；初审发现的事项：

- **阻塞 B1（已修复，待最终复审）**：`price_capacity.py` 在源 manifest `max_counts` 低报时跳过 offsets 校验，合成复现得到假 `full_population_supported=true`；现无条件校验三源 offsets/maxima，低报会拒绝，新增负例先失败后通过。
- **阻塞 B2（已修复，待最终复审）**：畸形 A checkpoint 缺 `training_signature` 时冒 `KeyError/TypeError`；现先显式拒绝为 `ValueError`，新增五个畸形元数据负例先失败后通过。
- **重要 I1（已修复）**：`fixed_stage:null` / 非字符串触发 `AttributeError`；现以明晰 `ValueError` 拒绝候选，旧 legacy 缺字段维持旧默认，已加负例。
- **重要 I2（已修复）**：损坏 manifest 的 prepare 抛错却不写失败记录；现 `preflight.json` 写 `status=failed`、异常类别及未授权状态后重抛，已加负例；缺缓存仍按 `incomplete_missing_evidence`/exit2 报告。
- **建议 S1（已修复）**：122 checkpoint 拒绝会留下空目标输出目录；移至拒绝通过后才创建目录，新增负例先失败后通过。
- 其余意见：`full` 仅是 `all` 别名；hash 字段名不同但值的比较口径自洽，已补注释；`adaptation_mode` 是显式 mode 保证，已限定现有 sum/set 无 mode 敏感层的事实。计划 ID 顺序由 `verify_source_eval_plan_against_stores` 在真实缓存核对；不在 plan-only 层伪造身份保证。

修复后 17 项阶段测试、最终全量 **1,784 项**（16 条平台跳过）、Ruff 与 diff-check 全部 exit0。第一次最终复审对 `b2aeacb7…` 版本给出无 BLOCK 的 APPROVE，同时指出一个畸形 PRICE checkpoint `metadata=None/[]/字符串` 冒 `AttributeError` 的非阻塞 MEDIUM；该问题已按最小守卫修复，新增负例先 RED 后 GREEN，模型专项 **12项**、全量 **1784项**再通过。最终增量独立只读复审已核对修复版快照中 27 个路径、测试日志 SHA 与新增 metadata 类型守卫；结果 **APPROVE / 无未解决 BLOCK**。审查者未重跑全量或 smoke（由主 agent 实际执行），未查看真实 23+3 缓存；通过只表示本轮代码及合成验证范围内无已知阻塞，**不是 P0 真实数据门或研究有效性认证**。
