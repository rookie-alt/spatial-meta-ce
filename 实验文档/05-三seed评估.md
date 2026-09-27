# 05 第 3 个 seed（s44）与三 seed 冻结评估（2026-09-27）

## 1. 做了什么

1. 按 s42/s43 完全相同配置训练 seed 44（只差 seed，已 diff config 校验）：
   - FOMAML：`configs/meta.json --seed 44 --device cuda` → `runs/formal-fomaml-20260927-seed44/`，2000 步，best val macro log-Q95 **2.671**（s42/s43 为 2.71 左右）
   - ERM：`configs/erm.json --seed 44 --device cuda` → `runs/erm-seed44-20260927/`，1700 步早停，best **2.718**
2. 用三个 FOMAML checkpoint 重跑适应策略搜索（只读 chembl/legalacts/telstra）→ `runs/adapt-search-3seed-20260927/result.json`
3. 用新冻结策略跑三个 seed 的目标库评估 → `runs/fomaml-s4{2,3,4}-frozen3-20260927/`
4. ERM s44 用与 s42/s43 相同的默认适应配置评估 → `runs/erm-s44-paired-20260927/`

## 2. 三 seed 策略搜索结果

| 排名 | 步数 | lr | scope | macro log-Q95 |
|---:|---:|---:|---|---:|
| 1（选定） | 5 | 1e-3 | all | **2.6554** |
| 2 | 5 | 1e-3 | head | 2.6557 |
| 3 | 1 | 3e-3 | head | 2.6570 |
| 5 | 3 | 1e-3 | all | 2.6597 |
| 6 | 3 | 1e-3 | head（2 seed 时的选择） | 2.6598 |
| — | 0（零样本） | — | — | 2.7034 |

- **选中的就是原默认配置**（5 步 / 1e-3 / all）。2 seed 时选的 3 步 / head 降到第 6。
- 前 8 名相差 < 0.006，选择本质上由噪声决定：加一个 seed 就换了赢家。结论是**适应超参在源验证上不可分辨**，搜索的价值在于证明"默认配置不是在目标库上挑出来的"。
- 按协议，正式结果改用 3 seed 搜索的冻结策略；`04` 文档中的 2 seed 冻结结果作为历史记录保留。
- 复现校验：s42/s43 在新冻结策略（= 默认配置）下的结果与 2026-09-26 paired 评估一致（unique Q95 最大差 0 / 8.5e-5）。

## 3. 目标库结果（unique-query Q95，三 seed 均值 ± 标准差）

| 库 | K | FOMAML s42 / s43 / s44 | FOMAML | ERM | PRICE(1e-6) |
|---|---:|---|---:|---:|---:|
| stats | 0 | 74.5 / 79.3 / 78.1 | 77.3 ± 2.5 | 79.4 ± 5.9 | 39.5 |
| stats | 5 | 86.0 / 106.0 / 96.8 | 96.3 ± 10.0 | 116.1 ± 18.6 | 43.6 |
| stats | 20 | 79.7 / 86.6 / 86.3 | 84.2 ± 3.9 | 88.7 ± 8.3 | 41.8 |
| stats | 100 | 80.7 / 90.7 / 89.0 | 86.8 ± 5.4 | 94.8 ± 12.4 | 42.4 |
| genome | 0 | 325.6 / 200.8 / 219.7 | 248.7 ± 67.3 | 374.1 ± 27.4 | 31.3 |
| genome | 5 | 209.3 / 157.2 / 149.3 | 172.0 ± 32.6 | 220.3 ± 23.8 | 42.4 |
| genome | 20 | 142.9 / 115.4 / 110.6 | 123.0 ± 17.4 | 130.0 ± 14.4 | 40.8 |
| genome | 100 | 163.1 / 127.0 / 122.9 | 137.7 ± 22.1 | 137.3 ± 14.5 | 39.1 |
| ergastf1 | 0 | 24.9 / 23.5 / 25.2 | 24.5 ± 0.9 | 26.1 ± 2.0 | 15.3 |
| ergastf1 | 5 | 24.3 / 22.0 / 24.8 | 23.7 ± 1.5 | 24.2 ± 1.2 | 14.8 |
| ergastf1 | 20 | 24.4 / 22.9 / 25.2 | 24.2 ± 1.2 | 24.0 ± 2.1 | 14.5 |
| ergastf1 | 100 | 24.2 / 21.9 / 25.1 | 23.8 ± 1.6 | 23.5 ± 1.6 | 14.5 |

K=10/50 及中位数见 `results/frozen3-20260927/`。中位数（unique Q50）：genome 上 FOMAML 所有 K 都优于 PRICE（1.23~1.30 vs 1.60~1.89），stats / ergastf1 略差。

## 4. 结论

1. **seed 44 与 s42/s43 同向**：STATS 负迁移三个 seed 都复现（所有 K 都劣于零样本，配对改善比例 11~26%），不是 seed 偶然。
2. **元学习 vs ERM**：零样本和小 K（≤10）时 FOMAML 三库均值都优于 ERM，genome K=0 差距最大（249 vs 374）；K≥20 后两者差距在一个标准差内，genome/ergastf1 K=50/100 ERM 略好。即元学习的收益集中在"少样本/零样本"端，这与论文动机一致，但不能声称大 K 时更好。
3. **seed 方差大**：genome K=0 标准差 67（s42 326 vs s43 201），单 seed 结论不可靠；ergastf1 上 FOMAML 与 ERM 的差距（< 1 个标准差）不可断言显著。
4. **与 PRICE 的尾部差距仍然存在**且远大于 seed 方差（stats 约 2 倍，genome 3~8 倍，ergastf1 约 1.6 倍）。超参调节已穷尽，下一步只剩 P2（表示层 / join 结构编码）。

## 5. 文件

| 内容 | 服务器 | 本地 |
|---|---|---|
| s44 训练 | `runs/formal-fomaml-20260927-seed44/`、`runs/erm-seed44-20260927/` | — |
| 3 seed 搜索 | `runs/adapt-search-3seed-20260927/result.json` | `results/adapt-search-3seed-20260927/` |
| 3 seed 冻结评估 | `runs/fomaml-s4{2,3,4}-frozen3-20260927/` | `results/frozen3-20260927/` |
| ERM s44 评估 | `runs/erm-s44-paired-20260927/` | `results/erm-s44-summary.json` |
