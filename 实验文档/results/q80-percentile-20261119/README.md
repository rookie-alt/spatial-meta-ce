# q50 / q80 / q90 三档分位结果（2026-11-19）

## 为什么重跑

`spatial_meta_ce/metrics.py` 原来只算 50/90/95/99，`q80` 在项目里完全不存在。
Q50/Q90 本来就有（历史 JSON 里一直在写），缺失的只有 **q80**；PRICE 论文 Table 2
用 50/80/90/95/99 作对照列，所以补 q80 是为了能和论文逐列对齐。

已有结果 JSON 不含逐查询预测，无法事后补算，只能**只读重跑评估**。
重跑用的是同一批冻结 checkpoint，不训练、不适应、不选模型。

## 改了什么

| 文件 | 改动 |
|---|---|
| `spatial_meta_ce/metrics.py` | `(50, 90, 95, 99)` → 常量 `PERCENTILES = (50, 80, 90, 95, 99)` |
| `scripts/allrows_target_eval.py` | `PERCENTILES` 同步加 80 |
| `scripts/price_population_eval.py` | `metric_definitions` 增加 q80 说明键；`headline()` 未改 |

`summarize()` 是**纯增量**：只新增 `q80` / `price_shifted_q80`，不删除任何旧键，
因此 `evaluate.py`、`train.py`、`source_eval.py` 的既有消费方不受影响。
`headline()` 刻意不加 q80/q90，因为测试用固定键的合成 report 调用它。

## 复现验证

| 面板 | 比对项 | 差异 |
|---|---|---|
| 26db K=0（三个 seed） | 180 个 q50/q90/q95/q99/log_q95 值 | 0，逐位一致 |
| 24k all-rows（三个 seed） | 144 个值 | 70 个有差异，**最大相对差 3.48e-06** |
| 24k K=0（三个 seed） | 48 个 ours_k0 值 | 30 个有差异，**最大相对差 2.70e-06** |

小差异不是口径变化，而是这几个数原本由另一份代码树算的（24k 面板出自
`official-split-20261117/code`），浮点累加顺序不同。量级 1e-6，不影响任何判读。

复现命令（服务器，CPU 只读）：

```bash
# sum（26db）
/gly/weilin/spatial-meta-ce-26db-20261118-runs/run_q80_panels.sh
# 24k A1-PG（price_lite），独立代码副本，不动封存树
/gly/weilin/spatial-meta-ce-26db-20261118-runs/run_q80_24k.sh
```

产物位置：

- 26db sum：`/gly/weilin/spatial-meta-ce-26db-20261118-runs/target-k0-q80/`
- 24k K=0：`/gly/weilin/spatial-meta-ce-26db-20261118-runs/target-k0-24k-q80/`
- 24k all-rows：`/gly/weilin/official-split-20261117/allrows-24k-q80/`

## 结果（三个 seed 的均值）

### 26db sum 检查点，unique-query K=0

| 库 | 估计器 | q50 | q80 | q90 | q95 |
|---|---|---|---|---|---|
| imdb | ours_k0 | 1.84 | 5.64 | 10.27 | 25.86 |
| imdb | price_k0 | 1.86 | 5.71 | 11.21 | 21.03 |
| imdb | pg | 2.28 | 10.40 | 27.75 | 87.11 |
| stats | ours_k0 | 2.52 | 12.01 | 51.71 | 229.90 |
| stats | price_k0 | 1.95 | 5.83 | 13.36 | 39.52 |
| stats | pg | 1.98 | 7.66 | 21.04 | 75.65 |
| genome | ours_k0 | 1.74 | 8.71 | 18.91 | 157.09 |
| genome | price_k0 | 1.89 | 3.63 | 5.23 | 31.34 |
| genome | pg | 1.19 | 8.73 | 24.98 | 1907.32 |
| ergastf1 | ours_k0 | 1.92 | 5.48 | 12.21 | 30.97 |
| ergastf1 | price_k0 | 1.47 | 3.17 | 6.66 | 15.29 |
| ergastf1 | pg | 1.54 | 4.23 | 11.34 | 27.61 |

### 24k A1-PG 检查点，unique-query K=0

| 库 | 估计器 | q50 | q80 | q90 | q95 |
|---|---|---|---|---|---|
| imdb | ours_k0 | 1.79 | 5.77 | 11.27 | 21.51 |
| imdb | price_k0 | 1.86 | 5.71 | 11.21 | 21.03 |
| imdb | pg | 2.28 | 10.40 | 27.75 | 87.11 |
| stats | ours_k0 | 2.16 | 8.95 | 34.23 | 188.80 |
| stats | price_k0 | 1.95 | 5.83 | 13.36 | 39.52 |
| stats | pg | 1.98 | 7.66 | 21.04 | 75.65 |
| genome | ours_k0 | 3.24 | 8.30 | 19.28 | 83.22 |
| genome | price_k0 | 1.89 | 3.63 | 5.23 | 31.34 |
| genome | pg | 1.19 | 8.73 | 24.98 | 1907.32 |
| ergastf1 | ours_k0 | 1.64 | 4.62 | 11.67 | 24.87 |
| ergastf1 | price_k0 | 1.47 | 3.17 | 6.66 | 15.29 |
| ergastf1 | pg | 1.54 | 4.23 | 11.34 | 27.61 |

### 24k A1-PG 检查点，all-rows（PRICE Table 2 口径）

| 库 | 估计器 | q50 | q80 | q90 | q95 |
|---|---|---|---|---|---|
| imdb | ours_k0 | 1.70 | 4.26 | 8.46 | 16.48 |
| imdb | price_k0 | 1.78 | 4.08 | 8.40 | 16.15 |
| imdb | pg | 1.96 | 6.04 | 19.04 | 49.44 |
| stats | ours_k0 | 2.06 | 8.11 | 30.27 | 167.44 |
| stats | price_k0 | 1.87 | 5.50 | 12.46 | 35.55 |
| stats | pg | 1.87 | 7.11 | 20.27 | 73.35 |
| genome | ours_k0 | 3.04 | 8.17 | 14.03 | 45.95 |
| genome | price_k0 | 1.65 | 3.60 | 5.17 | 15.67 |
| genome | pg | 1.17 | 4.34 | 15.12 | 383.09 |
| ergastf1 | ours_k0 | 1.70 | 4.98 | 11.75 | 24.55 |
| ergastf1 | price_k0 | 1.43 | 3.12 | 6.45 | 14.58 |
| ergastf1 | pg | 1.60 | 4.26 | 11.30 | 27.48 |

## 这些数字说明什么

- **q80 层的差距比 q95 小，但方向一致**：除 genome 的 q50 外，A1-PG 在 26db 与 24k
  上都不占优；24k 在 imdb 上追平了 PRICE（q50 1.79 vs 1.86，q95 21.51 vs 21.03），
  但 stats / genome / ergastf1 仍然落后。
- **genome 是最难的一列**：A1-PG 的 q50 随训练量从 1.74 升到 3.24，说明 24k 在
  genome 上不只是尾部退化，中位数也被推高了。这是"修尾牺牲典型"的又一个实例。
- **q80 补上之后，误差曲线形状更清楚**：A1-PG 在 stats 上 q50→q80 从 2.16 跳到 8.95
  （4.1 倍），PRICE 只从 1.95 到 5.83（3.0 倍）。误差不是在尾部才拉开的，
  70–80 分位就已经开始分叉。

## 边界

这是**只读重评估**，不构成新的精度结论：没训练、没选 checkpoint、没动目标库上的任何
选择。三个 seed 的均值也不能替代预注册的判定流程；doc 122 的 `worse` 判定不变。