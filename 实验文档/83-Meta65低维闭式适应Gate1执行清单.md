# Meta65 低维闭式适应 Gate 1 执行清单

> 日期：2026-10-18（Asia/Shanghai）。分支：
> `feat/meta65-lowdim-adaptation-20261018`。基准提交：
> `21272ab60b54e573d275ffad0261234552d04abc`。
>
> 状态：`inputs_frozen_execution_not_started`。服务器输入已通过只读认证；本文仍不
> 授权 GPU、训练或目标库读取，也不把屏幕结果预先声明为正式候选。

## 1. 目的与边界

本清单冻结 Meta65 Gate 1 离线只读屏幕的输入和运行边界。运行时只允许：

- `--device cpu`、`--execute`、单线程；
- 不设置 `CUDA_VISIBLE_DEVICES`，不将模型或张量放到 CUDA；
- 不调用训练入口，不更新任何模型参数以外的持久状态；
- 只读取 `chembl`、`legalacts`、`telstra` 三个源验证库；
- 不读取 `imdb`、`stats`、`genome`、`ergastf1`，不把它们作为 support、query、
  缓存、标签或监督信号；
- 不修改 `/gly/weilin/spatial-meta-ce` 主项目、冻结运行、`PRICE-master` 或历史结果。

本屏幕的 `status=completed` 只表示屏幕完成，不是正式候选、训练结果或跨库泛化结论。

## 2. 本地输入冻结

| 输入 | 路径 | SHA256 |
|---|---|---|
| λ registry（本地源） | `/private/tmp/spatial-meta-ce-meta65-lowdim-adaptation-20261018/configs/meta65_lowdim_gate1_lambda.json` | `651f70197fd33a387b5a5c363ee203de57aff6d07cb8390487ee099ee0870dd3` |
| 实现计划 | `/private/tmp/spatial-meta-ce-meta65-lowdim-adaptation-20261018/实验文档/82-Meta65低维闭式适应实现计划.md` | `dd6de12ab5678f0795758a4f17bf46d60e0aa61120628338cf84727109c5e649` |
| screen runner | `/private/tmp/spatial-meta-ce-meta65-lowdim-adaptation-20261018/scripts/run_meta65_lowdim_screen.py` | `4b4d20af4901a6d3dc44a0fbd770b9d492c9dc320847e1f7c9caa607923f1729` |
| screen 测试 | `/private/tmp/spatial-meta-ce-meta65-lowdim-adaptation-20261018/tests/test_meta65_lowdim_screen.py` | `5175352ce4946e0b1cb51d8a16206f10ef97029e3d7dcd83c6f7cedd50f413bb` |
| 低维适应实现 | `/private/tmp/spatial-meta-ce-meta65-lowdim-adaptation-20261018/spatial_meta_ce/lowdim_adaptation.py` | `8ae34f91425f08ee824f96df7b18d659a66387844c15e53b913f068444ae365a` |

λ registry 内容固定为：

```json
{
  "schema": 1,
  "experiment": "meta65-lowdim-gate1-20261018",
  "candidate_lambdas": [1.0],
  "selected_lambda": 1.0,
  "selection_timing": "before_query_evaluation"
}
```

该值在首次 query 评估前固定；看到任何 query 指标后不得改动。部署到服务器后的
registry 必须与上述 SHA256 完全相同。

## 3. 服务器只读输入

### 3.1 R 臂 checkpoint

| 项目 | 值 |
|---|---|
| 候选根目录 | `/gly/weilin/spatial-meta-ce-meta65-scope-safety-20261017-runs/formal-20261017T1740/` |
| 文件 | `/gly/weilin/spatial-meta-ce-meta65-scope-safety-20261017-runs/formal-20261017T1740/R_head3/last.pt` |
| 字节数 | `2574664` |
| SHA256 | `586bc0d0a8883c1e4cf0645c26a389d95cb5f80751df2b0f0f85c02c64add63f` |
| metadata | `/gly/weilin/spatial-meta-ce-meta65-scope-safety-20261017-runs/formal-20261017T1740/R_head3/metadata.json` |
| metadata SHA256 | `186d69d63d34c9f63f2d1d622c78816ebd9fc5e188c844b1e541dc74a3656dcc` |

执行前必须只读核对以下全部条件；任一项不满足即停止：

- 文件确为 80 号正式运行的 R 臂 `last.pt`，不是 `best.pt`、旧 C checkpoint、
  preflight checkpoint 或 adapted checkpoint；
- `_load_fixed_checkpoint()` 接受的模型配置为
  `{"arch":"sum","use_pg":true,"width":64,"layers":2,"heads":4}`；
- metadata 的 `seed=42`、`next_step=4000`；
- `validation_databases=["chembl","legalacts","telstra"]`；
- 不存在非空 `adapted_database` 标记；
- `train_databases` 与 `validation_databases` 均不包含目标库；
- 计算并回填服务器原件的 SHA256；执行前后必须保持不变。

2026-10-18 只读认证结果：checkpoint state 的 `model_config` 为
`{"arch":"sum","heads":4,"layers":2,"use_pg":true,"width":64}`，
`next_step=4000`，metadata `seed=42`，
`validation_databases=["chembl","legalacts","telstra"]`，
`adapted_database=null`；train/validation 与目标库交集为空。

### 3.2 冻结 endpoint plan

| 项目 | 值 |
|---|---|
| 路径 | `/gly/weilin/spatial-meta-ce-meta65-20261009-frozen/runs/meta65-source-screen-20261009/endpoint_plan.json` |
| 字节数 | `11104915` |
| 预期 SHA256 | `ebeb178de15b134722bd9fbc24ff4b4752f795a311580c1a17cb23c550bfd0f0` |

必须使用同一份完整 plan 的原始 K20 前缀，不重建 query pool，不修改 100-query
support 排除集合、support digests 或 query digests。

### 3.3 源缓存

| 项目 | 值 |
|---|---|
| cache root | `/gly/weilin/spatial-meta-ce/artifacts/cache` |
| 允许读取的库 | `chembl`、`legalacts`、`telstra` |

每个库只读取冻结的 `manifest.json`、`cards.npy`、`offsets.npy` 和
`group0.npy`～`group3.npy`。任何符号链接、逃逸 cache root、目标库路径或 SHA
前后变化均停止。

## 4. 固定运行矩阵

- 固定算子：`none`、`last3_gradient_smooth_l1`、`last3_gradient_mse`、
  `last3_ridge_mse`。
- `K=20`，support 使用完整 100-query pool 的嵌套前 20 项。
- 每库 5 runs（run 0～4），三源验证库 `chembl/legalacts/telstra`。
- 预期 60 cells（4 operators × 3 databases × 5 runs）。
- 预期 15 份 pair NPZ（3 databases × 5 runs），每份含四算子、query 身份、真值和
  bucket 掩码。
- 主口径：standard unique-query Q-error；库内五 run 算术平均，再做三库等权
  macro。不得混合 pooled、macro-log、all-rows 或不同 seed 聚合。

## 5. 服务器写入范围

等待用户一次性授权后，只允许写入以下独立路径：

| 用途 | 精确绝对路径 |
|---|---|
| 部署代码目录 | `/gly/weilin/spatial-meta-ce-meta65-lowdim-adaptation-20261018-code` |
| 独立运行目录 | `/gly/weilin/spatial-meta-ce-meta65-lowdim-adaptation-20261018-code/runs/gate1-20261018` |
| 部署后的 λ registry | `/gly/weilin/spatial-meta-ce-meta65-lowdim-adaptation-20261018-code/configs/meta65_lowdim_gate1_lambda.json` |

部署前必须确认代码目录和运行目录均不存在；运行目录必须由 runner 新建，禁止预先
创建或覆盖。不得把输出写到 `/gly/weilin/spatial-meta-ce`、冻结运行目录、
`PRICE-master`、目标库目录或历史结果目录。

## 6. 执行门

所有条件满足后才允许执行以下 CPU-only 命令：

```bash
env -u CUDA_VISIBLE_DEVICES \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  PYTHONDONTWRITEBYTECODE=1 \
  /gly/weilin/PRICE-master/venv/bin/python -B \
  /gly/weilin/spatial-meta-ce-meta65-lowdim-adaptation-20261018-code/scripts/run_meta65_lowdim_screen.py \
  --checkpoint /gly/weilin/spatial-meta-ce-meta65-scope-safety-20261017-runs/formal-20261017T1740/R_head3/last.pt \
  --checkpoint-sha256 586bc0d0a8883c1e4cf0645c26a389d95cb5f80751df2b0f0f85c02c64add63f \
  --plan /gly/weilin/spatial-meta-ce-meta65-20261009-frozen/runs/meta65-source-screen-20261009/endpoint_plan.json \
  --plan-sha256 ebeb178de15b134722bd9fbc24ff4b4752f795a311580c1a17cb23c550bfd0f0 \
  --cache-root /gly/weilin/spatial-meta-ce/artifacts/cache \
  --lambda-registry /gly/weilin/spatial-meta-ce-meta65-lowdim-adaptation-20261018-code/configs/meta65_lowdim_gate1_lambda.json \
  --out /gly/weilin/spatial-meta-ce-meta65-lowdim-adaptation-20261018-code/runs/gate1-20261018 \
  --device cpu --execute
```

不得增加训练参数、CUDA 环境变量、重试包装、λ 搜索或替代 checkpoint。

## 7. 独立只读核验清单

执行结束后必须重新从原始文件独立核验，不只调用被验 runner 的汇总结论：

- checkpoint、plan、λ registry 和三库源缓存的前后 SHA256 未变；
- `cells.json` 恰为 60 cells，每库每 run 都有四算子且 query/support digest 完全配对；
- 15 份 pair NPZ 的 query hash、query ID、真值和 `n_tables` 一致；
- 从 pair NPZ 独立复算 4+ 与 low4 的 Q50/Q95、五 run 算术 macro；
- 四算子的 support 残差、更新范数、Gram/solver 条件数全部有限；
- `status.json` 明确 `training=false`、`target_read=false`，并保持
  `screen_completed_not_selected` 语义；
- 扫描运行输入和产物，确认目标库读取数为 0；
- 明确区分 ridge 对 MSE 梯度是“solver + proximal regularization”组合对照；
  ridge 对 SmoothL1 三阶梯度是求解器 + 损失 + 正则的组合，不得归因于 solver；
- 记录全部反例和失败项，再判断是否存在进入 Gate 2 草案的信号。

## 8. 停止条件

- checkpoint 不满足 `_load_fixed_checkpoint()`、metadata 或 split 约束；
- plan SHA 与预期不符，K20 前缀或完整 query 分母不匹配；
- 任一 cache 文件、符号链接、标签、特征或 query 身份不合法；
- ridge 未改善 support 残差，或出现非有限值、病态数值、参数爆炸；
- 任一目标库被读取，或试图修改只读输入；
- 出现把完成屏幕解释为正式候选、训练结果或跨库泛化结论的表述。

本轮无论结果好坏均不启动训练。若 ridge 无真实收益，明确停止；若存在收益，只形成
Gate 2 元训练协议草案，等待用户单独授权。
