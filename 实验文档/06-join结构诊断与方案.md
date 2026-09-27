# 06 join 结构：诊断与改进方案（2026-09-27，方案待确认）

## 1. 诊断：误差是否随 join 数增长

脚本 `scripts/error_by_joins.py`，只读已有预测（K=0、run 0、unique query），不训练、不选参。结果 `results/error-by-joins-20260927.json`。

unique Q95（PRICE / FOMAML s42 / s43 / s44）：

| 库 | 表数 | 查询数 | PRICE | s42 | s43 | s44 |
|---|---:|---:|---:|---:|---:|---:|
| stats | 2 | 828 | 6.3 | 10.1 | 9.6 | 9.1 |
| | 3 | 869 | 41.3 | 98.4 | 111.6 | 99.3 |
| | 4 | 578 | 150.7 | 366.0 | 421.4 | 445.1 |
| | 5 | 244 | 106.8 | 171.8 | 213.0 | 175.3 |
| genome | 2 | 553 | 3.7 | 11.5 | 14.7 | 12.2 |
| | 3 | 602 | 20.1 | 82.2 | 79.5 | 67.9 |
| | 4 | 385 | 65.5 | 1864.3 | 1713.3 | 1139.6 |
| | 5 | 134 | 63.9 | 2473.4 | 2157.7 | 1559.8 |
| | 6 | 25 | 59.3 | 2864.2 | 1736.9 | 1603.9 |
| ergastf1 | 2 | 578 | 4.6 | 5.4 | 4.7 | 5.0 |
| | 3 | 422 | 14.2 | 23.4 | 21.6 | 24.7 |
| | 4 | 187 | 52.0 | 90.3 | 65.4 | 89.4 |

结论：
- 2 表查询我们和 PRICE 接近；差距随表数增长，genome 4+ 表差 20~45 倍，三 seed 一致。
- genome 4 表中位数我们反而更好（1.2~1.3 vs 3.5），说明问题在尾部的多 join 组合，不是整体偏差。

## 2. 对原假设的修正

原假设是"池化丢了 join 配对关系，PRICE 用 self-attention 显式建模"。读 PRICE 源码（`model/module.py`、`model/encoder.py`）后：
- PRICE 的 token 没有位置编码，也没有 join 配对 ID，**它同样不显式知道哪两个 fanout 属于同一个 join**。
- PRICE 与我们的真正区别是：(a) token 之间有 attention 交互，(b) 两阶段——先 join 列/fanout（scaling），再拼上表和过滤列（filtering），(c) 容量大 107 倍。

另一个更便宜的可疑点：log 基数近似是各表规模、选择率、fanout 的**和**。我们只有 mean/max 池化，head 要从 mean×count 还原"和"，MLP 做乘法很难，而且表越多越难。这和误差随表数增长的形态吻合。

## 3. 方案（按成本递增，逐级做，前一级够用就停）

| 变体 | 改动 | 参数增量 | 检验的假设 |
|---|---|---:|---|
| A. sum 池化 | 每组在 mean/max 之外加 masked sum | ~+33k（head 输入 8w→12w） | 缺"可加"聚合 |
| B. token attention | 各组编码后加类型 embedding，拼成一个序列过 2 层小 Transformer（width 64, 4 头），再池化 | ~+100k | 缺 token 交互（PRICE 同款） |
| C. B + join 配对 ID | fanout token 加 join 序号 embedding（第 2j、2j+1 个 fanout 属于第 j 个 join） | ~+2k | 显式配对，PRICE 没有 |

C 只对 fanout 可行：缓存里 fanout 的顺序对应 join 顺序，但 join 列→表的映射没有存。完整的 join 图需要改特征抽取并重建缓存，暂不做。

## 4. 协议（防泄漏）

1. **只用源验证库选变体**：每个变体 3 seed，按源验证 macro log-Q95（与训练 early stop 同指标）比较；先在源验证上也做一遍按表数分桶，确认同一形态存在，否则选择依据不成立。
2. 选定后每个变体**只跑一次**三目标库（K=0~100，默认适应策略 5 步/1e-3/all），不根据目标结果回头改。
3. 变体 A 为主对照；若 A 已把 genome 4+ 表拉到 PRICE 同量级，就不做 B/C。
4. ERM 同步改架构跑一套，保证"元学习有效"的结论在新架构下仍成立。
5. 模型体积继续报告：C 约 20 万参数，仍比 PRICE 小 50 倍以上。

## 5. 成本

每轮 3 seed × 1 变体 FOMAML + 3 seed ERM = 6 个训练，占卡 1~6 并行，约 45 分钟；目标评估每个 checkpoint GPU 约 5 分钟。A→B→C 全做约 3~4 小时。

代码改动：`model.py` 加 `pooling` / `encoder` 配置项（默认保持现结构，旧 checkpoint 可加载），新增单元测试（mask 下 sum 池化、attention 对 padding 不变），新分支 `feat/join-structure`。

## 6. 实现与启动（2026-09-27）

- 代码：`spatial_meta_ce/model.py` 新增 `arch` 参数（`set` / `sum` / `attn` / `attn_pair`）。`arch='set'` 的 config 与旧版一致，旧 checkpoint 可原样加载。
- 参数量：set 99,777 / sum 132,545 / attn 199,745 / attn_pair 200,769（仍比 PRICE 小约 53 倍）。
- attention 用 pre-norm、dropout=0，padding 用 `src_key_padding_mask` 屏蔽，池化前再次清零 padding。
- 测试：新增 `tests/test_arch.py`（初值=PG、padding 屏蔽、batch 组合不变性、置换不变性、配对 embedding 对错位敏感、元梯度覆盖全部参数、checkpoint 往返），服务器 44 测试全部通过。
- 配置：`configs/meta_{sum,attn,attn_pair}.json`（及 `erm_*`），除 `model` 外与 `meta.json` 完全相同。
- 冒烟（100 步）：三变体均正常，速度 105~129 秒/100 步，2000 步约 35~45 分钟。
- **执行顺序调整**：服务器 8 卡全空，三变体 × 3 seed 共 9 个训练同时启动（原计划是逐级做）。选择规则不变，仍只看源验证，只是节省时间。
- 输出：`runs/arch-{sum,attn,attn_pair}-seed{42,43,44}-20260927/`

## 7. 源验证分桶检查（协议第 1 步，已通过）

`scripts/arch_select.py`，原模型（set）三 seed，零样本、unique 查询、按表数分桶的 Q95（s42 / s43 / s44）：

| 库 | 2 表 | 3 表 | 4 表 | 5 表 | 6 表 | 7 表 |
|---|---:|---:|---:|---:|---:|---:|
| chembl | 4.0 / 4.5 / 4.0 | 11.3 / 10.2 / 10.0 | 40.9 / 34.3 / 35.0 | 262 / 223 / 234 | 210 / 208 / 187 | 194 / 217 / 195 |
| legalacts | 9.8 / 9.2 / 10.0 | 8.0 / 7.9 / 7.9 | 19.1 / 21.1 / 18.7 | – | – | – |
| telstra | 2.4 / 2.5 / 2.5 | 3.9 / 3.9 / 3.8 | 8.1 / 8.3 / 8.1 | 5.9 / 6.5 / 5.4 | – | – |

源验证上同样存在"表越多误差越大"的形态（chembl 最明显，2 表→5 表放大约 60 倍），所以用源验证选变体的依据成立。基线源验证 macro log-Q95：set s42/s43/s44 = 2.6625 / 2.6620 / 2.6418（plan-seed 2718，默认适应 5 步/1e-3/all）。

结果文件：`实验文档/results/arch-20260927/select-set.json`
