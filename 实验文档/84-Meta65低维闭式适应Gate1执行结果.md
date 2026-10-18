# Meta65 低维闭式适应 Gate 1 执行结果

> 日期：2026-10-18（Asia/Shanghai）。分支：
> `feat/meta65-lowdim-adaptation-20261018`。执行前 HEAD：
> `90148a0c2a2a4b6a69acc079dcbf1a41c1a99367`。worktree：
> `/private/tmp/spatial-meta-ce-meta65-lowdim-adaptation-20261018`。
>
> 服务器运行状态：`completed`。Gate 状态：
> `screen_completed_not_selected`。研究决定：
> **ridge 无真实收益，停止该路线；不进入 Gate 2。**
>
> 本轮没有训练、没有启动 GPU、没有读取
> `imdb/stats/genome/ergastf1`。单 seed、单 K 的源验证屏幕不构成目标库泛化结论。

## 1. 路径偏离与授权

原执行清单冻结的 code 路径
`/gly/weilin/spatial-meta-ce-meta65-lowdim-adaptation-20261018-code`
在只读预检时已存在，因此按原停止条件立即中止；没有删除、覆盖或检查其内容。

用户随后明确授权新目录，并允许写文档记录。本轮实际使用：

| 用途 | 路径 |
|---|---|
| 部署代码 | `/gly/weilin/spatial-meta-ce-meta65-lowdim-adaptation-20261018-code-rerun1` |
| 运行目录 | `/gly/weilin/spatial-meta-ce-meta65-lowdim-adaptation-20261018-code-rerun1/runs/gate1-20261018` |
| λ registry | `/gly/weilin/spatial-meta-ce-meta65-lowdim-adaptation-20261018-code-rerun1/configs/meta65_lowdim_gate1_lambda.json` |
| 本地原始产物 | `/private/tmp/meta65-lowdim-gate1-raw-20261018` |

原 `...-code` 目录未修改。部署仅包含 runner、实际导入闭包中的 8 个
`spatial_meta_ce/` 文件和 λ registry，共 10 个文件。部署后逐文件 SHA 校验全部
通过；macOS `tar` 产生的 10 个 `._*` 元数据文件已在本轮内删除，最终目录恰为
上述 10 个文件。运行目录在启动前不存在，由 runner 新建。

## 2. 输入冻结与前后一致性

| 输入 | SHA256 | 结果 |
|---|---|---|
| R 臂 checkpoint（`R_head3/last.pt`） | `586bc0d0a8883c1e4cf0645c26a389d95cb5f80751df2b0f0f85c02c64add63f` | 前后一致 |
| checkpoint metadata | `186d69d63d34c9f63f2d1d622c78816ebd9fc5e188c844b1e541dc74a3656dcc` | 执行前读取一致 |
| 冻结 endpoint plan | `ebeb178de15b134722bd9fbc24ff4b4752f795a311580c1a17cb23c550bfd0f0` | 前后一致 |
| λ registry | `651f70197fd33a387b5a5c363ee203de57aff6d07cb8390487ee099ee0870dd3` | 前后一致 |
| screen runner | `4b4d20af4901a6d3dc44a0fbd770b9d492c9dc320847e1f7c9caa607923f1729` | 本地与服务器一致 |

checkpoint 大小为 `2,574,664` bytes，plan 为 `11,104,915` bytes。checkpoint
验证结果：`model_config={"arch":"sum","use_pg":true,"width":64,"layers":2,"heads":4}`、
`next_step=4000`、metadata `seed=42`、
`validation_databases=["chembl","legalacts","telstra"]`、`adapted_database=null`，
训练/验证库与目标库交集为空。

三源 cache 的 `manifest.json`、`cards.npy`、`offsets.npy` 和 `group0.npy`～
`group3.npy` 共 21 个文件已逐文件计算执行前后 SHA256，全部不变。三库 manifest
SHA 分别为：

| 来源库 | `manifest.json` SHA256 |
|---|---|
| chembl | `7a936bbe92c1f919bb4a9d8e0276f1cba418f93e1bb0a5595b51cf0c8d837952` |
| legalacts | `a87b4ceb61d687f0f0525ea763422b4b8f29cad4bb56b664ee6d305f40e7d225` |
| telstra | `3df79cbd9b0913d1cb8d6f7b0b58eca8bf9ef7094f2048657ade2fc49c8ec144` |

## 3. 唯一一次执行

执行使用 PRICE Python `3.11.4`、NumPy `1.26.1`、PyTorch `2.1.0+cu121`；
PyTorch wheel 带有 CUDA 支持，但本命令固定 `--device cpu`、单线程并移除
`CUDA_VISIBLE_DEVICES`，不使用 CUDA。

```bash
env -u CUDA_VISIBLE_DEVICES \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  PYTHONDONTWRITEBYTECODE=1 \
  /gly/weilin/PRICE-master/venv/bin/python -B \
  /gly/weilin/spatial-meta-ce-meta65-lowdim-adaptation-20261018-code-rerun1/scripts/run_meta65_lowdim_screen.py \
  --checkpoint /gly/weilin/spatial-meta-ce-meta65-scope-safety-20261017-runs/formal-20261017T1740/R_head3/last.pt \
  --checkpoint-sha256 586bc0d0a8883c1e4cf0645c26a389d95cb5f80751df2b0f0f85c02c64add63f \
  --plan /gly/weilin/spatial-meta-ce-meta65-20261009-frozen/runs/meta65-source-screen-20261009/endpoint_plan.json \
  --plan-sha256 ebeb178de15b134722bd9fbc24ff4b4752f795a311580c1a17cb23c550bfd0f0 \
  --cache-root /gly/weilin/spatial-meta-ce/artifacts/cache \
  --lambda-registry /gly/weilin/spatial-meta-ce-meta65-lowdim-adaptation-20261018-code-rerun1/configs/meta65_lowdim_gate1_lambda.json \
  --out /gly/weilin/spatial-meta-ce-meta65-lowdim-adaptation-20261018-code-rerun1/runs/gate1-20261018 \
  --device cpu --execute
```

执行窗口为 `2026-10-18T02:42:48Z` 至 `2026-10-18T02:47:51Z`，外层墙钟约
`303.0s`；`status.json` 内部耗时 `299.3298913659528s`。只执行一次，没有自动
重试、覆盖、λ 搜索、训练或 resume。

最终状态：

- `status=completed`
- `completed_cells=60`
- `k=20`
- `selected_lambda=1.0`
- `training=false`
- `target_read=false`
- `gate=screen_completed_not_selected`
- `operators=none / last3_gradient_smooth_l1 / last3_gradient_mse / last3_ridge_mse`

## 4. 四算子结果

主口径为 standard unique-query Q-error：每库五个 run 先做算术平均，再对三库做
等权 macro。下表为三库 macro：

| 算子 | 4+ Q50 | 4+ Q95 | 4+ Q95 相对 none | low4 Q50 | low4 Q95 |
|---|---:|---:|---:|---:|---:|
| none | 2.501896 | 41.440836 | 0.00% | 4.747515 | 142.380057 |
| SmoothL1 last3 | 2.502805 | 41.035807 | -0.98% | 4.828171 | 146.970635 |
| MSE last3 | 2.515828 | 39.323504 | -5.11% | 5.327786 | 171.027300 |
| ridge MSE | 2.763848 | 99.230620 | +139.45% | 41.606744 | 3124.871537 |

逐库结果：

| 算子 | 库 | 4+ Q50 | 4+ Q95 | low4 Q50 | low4 Q95 |
|---|---|---:|---:|---:|---:|
| none | chembl | 3.179243 | 99.638277 | 9.785906 | 405.663024 |
| none | legalacts | 2.608538 | 17.661586 | 2.466847 | 11.278152 |
| none | telstra | 1.717907 | 7.022647 | 1.989792 | 10.198994 |
| SmoothL1 last3 | chembl | 3.184089 | 98.652620 | 9.998225 | 419.213775 |
| SmoothL1 last3 | legalacts | 2.606802 | 17.422608 | 2.496943 | 11.468191 |
| SmoothL1 last3 | telstra | 1.717525 | 7.032192 | 1.989346 | 10.229938 |
| MSE last3 | chembl | 3.235872 | 93.901372 | 11.349916 | 490.531114 |
| MSE last3 | legalacts | 2.595886 | 16.985608 | 2.622120 | 12.181627 |
| MSE last3 | telstra | 1.715727 | 7.083532 | 2.011321 | 10.369157 |
| ridge MSE | chembl | 3.604227 | 265.671689 | 117.437254 | 9328.429381 |
| ridge MSE | legalacts | 2.747223 | 22.754530 | 4.838685 | 32.101154 |
| ridge MSE | telstra | 1.940094 | 9.265641 | 2.544293 | 14.084076 |

## 5. 逐 query 配对

配对使用同一 `database/run` 内四个算子的同一 query 身份。每库五个 run 的 query
presentation 数分别为 chembl `169,405`、legalacts `45,796`、telstra `101,350`。
下表 macro 是十五个 `database/run` 配对结果的算术平均；Q-ratio 小于 1 表示左算子
更差项更少，improved fraction 表示左算子 Q-error 更小的 query 比例。

| 配对（左 vs 右） | 4+ Q95 ratio | 4+ improved | low4 Q95 ratio | low4 improved | 4+ mean ΔlogQ |
|---|---:|---:|---:|---:|---:|
| SmoothL1 last3 vs none | 0.992649 | 51.6755% | 1.017760 | 30.7124% | -0.001130 |
| MSE last3 vs none | 0.970937 | 50.3896% | 1.101986 | 27.0517% | -0.001413 |
| ridge MSE vs none | 1.757331 | 44.4935% | 9.069783 | 31.2418% | +0.156103 |
| ridge MSE vs MSE last3 | 1.823188 | 44.2532% | 7.471563 | 32.4163% | +0.157516 |
| MSE last3 vs SmoothL1 last3 | 0.977927 | 50.3483% | 1.081420 | 25.8703% | -0.000283 |
| ridge MSE vs SmoothL1 last3 | 1.772778 | 44.3499% | 8.749531 | 31.3240% | +0.157233 |

逐库 4+ Q95 ratio：

| 配对 | chembl | legalacts | telstra |
|---|---:|---:|---:|
| SmoothL1 last3 vs none | 0.990119 | 0.986468 | 1.001361 |
| MSE last3 vs none | 0.942412 | 0.961726 | 1.008674 |
| ridge MSE vs none | 2.664401 | 1.288389 | 1.319204 |
| ridge MSE vs MSE last3 | 2.821060 | 1.342835 | 1.305668 |
| ridge MSE vs SmoothL1 last3 | 2.694342 | 1.307282 | 1.316712 |

归因边界：

- `ridge MSE vs MSE last3` 同时改变求解器并加入 proximal regularization，只能解释为
  **solver + proximal regularization 组合对照**，不能归因于 solver 单独作用。
- `ridge MSE vs SmoothL1 last3` 同时改变损失、求解器和正则，是组合干预；不得将差异
  归因于 solver。
- `MSE last3 vs SmoothL1 last3` 在相同三步梯度路径上改变损失，才可作为损失口径对照。

## 6. 独立验证与数值审计

运行结束后把 114 MB 原始产物复制到本地 `/private/tmp`，没有把 NPZ 放入仓库。
独立验证器不导入 runner，直接从冻结 plan、三源 cache 的 `manifest/cards/offsets`
和 15 份 NPZ 复算，共通过 **10,354 项断言**：

- `cells.json` 恰有 60 个唯一 cell；每库每 run 恰有四个算子，query/support digest
  完全配对。
- 恰有 15 份 pair NPZ；query hash、query id、真值、表数、`four_plus` 和 `low4`
  掩码逐条匹配冻结 plan 与 cache。
- 每份 NPZ 内部的 Q-error、pair delta-log-Q 和 Q-ratio 均由独立 NumPy 实现重算。
- 4+/low4 的 Q50/Q95、五 run 算术平均和三库 macro 与 `macro.json` 一致。
- 六组逐 query 配对的 count、improved/worsened/tied、Q ratio 和 delta-log-Q 与
  `paired_summary.json` 一致。
- 60 个 cell 的 `state_before` 与 `state_after` 相同，说明评估未原地修改共享
  checkpoint。

跨机器 NumPy 的 `exp/log` 存在末位差异，因此浮点比较使用
`rtol=1e-12, atol=1e-12`；最大绝对差 `7.28e-12`，最大相对差 `1.11e-14`。
身份、真值、bucket 掩码、哈希和计数均使用精确比较。

数值审计：

| 项目 | 实测范围 |
|---|---:|
| parameter update norm | 0.0–4.159097 |
| diagnostic `XᵀX + λI` condition number | 460.907021–1093.400574 |
| design condition number | 268.948494–29424.662625 |
| ridge solver condition number | 460.907021–1093.400574 |
| ridge solver update norm | 0.828560–4.159097 |
| support MSE before | 0.331696–5.945526 |
| support MSE after | 0.196125–5.945526 |
| ridge support MSE improvement | 25.6972%–74.0600%（15/15 改善） |

所有 support 残差、更新范数和条件数均有限。没有超过 `1e12` 的病态警告；没有
梯度或 ridge 算子的 support MSE 恶化项；15/15 个 ridge cell 均改善 support 残差。

目标隔离扫描：

- 所有运行文本产物中 `imdb/stats/genome/ergastf1` 命中数为 0。
- 所有 NPZ query id 只属于 `chembl/legalacts/telstra/pretrain`，目标 query id 数为 0。
- 部署代码中的目标库名称只出现在 denylist 常量与路径/checkpoint 校验，不构成数据
  读取路径；runner 的数据循环只枚举三个源验证库。
- `status.json` 明确 `training=false`、`target_read=false`。

本轮没有挂载操作系统级 I/O tracer。因此“目标库读取为 0”的证据是 runner 的受限
数据路径、冻结输入绑定、状态字段、产物文本扫描和 query 身份扫描的综合结果，不是
对系统调用的逐次追踪。

## 7. 反例与决策

**主要反例**：ridge 在三个库的 4+ Q95 全部恶化；十五个 `database/run` 的
ridge-vs-none 配对中，每个 run 的 4+ Q95 都比 none 更差。三库 macro 从
`41.440836` 升至 `99.230620`（+139.45%），low4 Q95 从 `142.380057` 升至
`3124.871537`。

ridge 的 support MSE 在 15/15 个 cell 中明显下降，却没有转化为 query 泛化；
说明“闭式解更好拟合 20 条 support”不足以支持该适应器。该证据同时表明不能把
support 残差改善当作 query 收益的代理指标。

MSE last3 的 4+ Q95 operator-macro 为 `39.323504`，相对 none 的
`41.440836` 改善 `5.11%`；但 low4 Q95 operator-macro 为 `171.027300`，
相对 none 的 `142.380057` 恶化 `20.12%`。逐 run 配对的 low4 Q95 ratio
算术平均为 `1.101986`（即配对口径 `+10.20%`），且 telstra 的 4+ Q95 变差。
它只是本屏幕中的一个描述性信号，不是经预先定义保护条件验证的候选，也不改变本轮
停止决定。

最终决定：

- **停止低维 proximal ridge 路线，不进入 Gate 2。**
- 不提出元训练协议草案，不训练，不读取目标库。
- 该结果不证明所有闭式适应方法无效，也不证明 ridge 在其它 λ、损失或参数化下必然
  无效；本轮不扩网格、不追加 seed/K，也不在看到结果后重新选 λ。

## 8. 精简证据

- [运行状态](../reports/meta65-lowdim-gate1-20261018/status.json)
- [执行环境](../reports/meta65-lowdim-gate1-20261018/environment.json)
- [输入绑定](../reports/meta65-lowdim-gate1-20261018/inputs.json)
- [screen plan 摘要](../reports/meta65-lowdim-gate1-20261018/screen_plan.json)
- [四算子 macro](../reports/meta65-lowdim-gate1-20261018/macro.json)
- [逐 query 配对汇总](../reports/meta65-lowdim-gate1-20261018/paired_summary.json)
- [60-cell 指标表](../reports/meta65-lowdim-gate1-20261018/metrics.csv)
- [独立验证摘要](../reports/meta65-lowdim-gate1-20261018/verification_summary.json)

完整 `cells.json`、15 份 NPZ 和独立验证的 10,354 项断言明细保留在服务器运行目录
及本地 `/private/tmp/meta65-lowdim-gate1-raw-20261018`，不提交 Git。

## 9. 独立验证器重建与复跑（2026-10-18）

结果归档提交 `fa77310` 时，生成历史 `verification_summary.json` 的临时验证器没有
入库。本次新增 [独立验证器](../scripts/meta65_lowdim_verify.py) 与
[回归测试](../tests/test_meta65_lowdim_verify.py)，不改运行器、冻结协议或研究决策。
这是依据原始产物格式重建的实现，不是找回历史脚本；历史 10,354 项断言及其报告
保留，不将新的检查计数冒充历史断言或独立实验重复数。

验证器只依赖 Python 标准库与 NumPy，不导入 runner、Torch 或模型代码。读取本地
原始副本的 `run/` 与 `inputs/`，重新验证完整 100-query support pool 的 K20 前缀、
first-unique query 分母、三源身份与标签、表数及 bucket 掩码；重算所有 Q-error、
六组逐 query 配对量、60-cell 指标、五 run 均值和三库等权 macro；检查 support
残差、MSE/SmoothL1、状态摘要与 CSV。support 的 residual-head target 从 cards 的
true/PG 经 float32 log 后转 float64 作差复算，与实际输入精度一致。

默认比对仓库已提交的 Gate 1 精简证据，可用 `--evidence-root` 显式指定可信镜像。
JSON 镜像按字节 SHA256 比较；CSV 允许 Git 的 CRLF/LF 换行转换，字段仍逐项核验。
输入的远端绝对路径仅作为历史元数据，不会访问；输出必须位于输入与镜像目录外，
已存在输出、符号链接、缺失或不一致的证据均拒绝，不覆盖原始产物或历史验证报告。

在包含原始副本的机器上，从仓库根目录复跑（输出文件名必须尚不存在）：

```bash
PYTHONDONTWRITEBYTECODE=1 /opt/anaconda3/bin/python -B \
  scripts/meta65_lowdim_verify.py \
  --raw-root /private/tmp/meta65-lowdim-gate1-raw-20261018 \
  --evidence-root reports/meta65-lowdim-gate1-20261018 \
  --output /private/tmp/meta65-gate1-verification-new.json
```

`/opt/anaconda3/bin/python` 是本机解释器示例，其他机器使用具有 NumPy 的 Python。
完整原始副本未入 Git；仅 clone 仓库不能凭精简证据重新生成 NPZ，也不会伪造缺失
输入。原始目录仍需另外持久备份，本次未迁移、删除或修改它。

认证边界：本地副本不含 checkpoint 与 `group0-3.npy`，它们只有历史 hash 记录，
本轮没有重新校验相应字节；没有重跑模型前向、梯度或 ridge 求解，因此 update norm
和 condition number 只检查记录的有限性与约束，不是独立复算。共享 state digest
相等是产物层证据，不替代内存跟踪；目标隔离也不等于系统调用级 I/O tracing。

本轮 [重建验证报告](../reports/meta65-gate1-verifier-rebuild-20261018.json) 已从真实
60 cells/15 NPZ 生成，4+ Q95 macro与原归档一致；专项51项、全套1194项（16项跳过）
通过。新文件Ruff专项通过；全项目176项既有问题与同版本基线逐项一致，并非全项目
lint通过。独立只读审查发现并修复metadata冻结SHA遗漏后复审通过。完整验证记录
与实验边界见 [进度第83节](99-进度与下一步.md#83-gate-1-独立验证器重建与可复跑证据2026-10-18)。
