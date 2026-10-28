# 111 FOMAML-sum 主表评估计划

> 状态：**已执行（2026-10-27）**，结果与验收见 [104 号 §7.13](104-导师汇报报告与文献对标.md) 与 `results/p2-2-fomaml-sum-target-3seed-20261027/`。
> 本文保留原计划与验收标准；执行中的口径差异与结论见 §9。

## 1. 为什么需要这一步

报告 02–07 节的主表（「我们」列）是 **ERM-sum seed42 baseline**，不是元学习模型。
这条 population 评估线（102/103/108 号）当初定义为「只评估、不训练、不选 checkpoint」，
沿用了 P1 筛选时冻结的 ERM-sum checkpoint；**FOMAML-sum 从未跑过同口径的 population 评估**。

后果：报告讲的是元学习方法，主表却是它自己的对照臂。这不是数字错误，但会让「我们」这一列
在答辩时无法自证。104 号 §7.9 已把这条写成显式口径声明；本计划的目的是把声明变成实测。

## 2. 已有资产（2026-10-27 核对）

三个 checkpoint 都在服务器，`method=fomaml`、`arch=sum`、4000 步、三 seed 不同：

| seed | 目录 | last.pt sha256 |
|---|---|---|
| 42 | `runs/arch4k-sum-seed42-20260927/last.pt` | `ebdaf1afe37e2dbe8606b7bef20c65e8c3a67d5f9cffd69900022c1b2473c994` |
| 43 | `runs/arch4k-sum-seed43-20260927/last.pt` | `3b30acb4d9f4c01b9e7d22da24a33b1233f559d37e1dbf24b36b8272808a42bc` |
| 44 | `runs/arch4k-sum-seed44-20260927/last.pt` | `e364ebd552fa66bfad96e0b5c1d66cc7607847b5161f0e2c812ee790b3d5ce31` |

- `config.json` 已核对：`{"width":64,"use_pg":true,"arch":"sum","layers":2,"heads":4}`，`method: fomaml`，`steps: 4000`。
- 这三个 checkpoint 就是 07 号选型那轮（`results/arch-20260927/select4k-sum.json`）的产物，
  选型只用了源验证库 chembl/legalacts/telstra。
- **口径选择**：用 `last.pt` 而不是 `best.pt`，因为 ERM-sum baseline 的 population 评估用的就是
  `last.pt`（`runs/p1-screen-20261020/baseline/last.pt`）。两侧必须同一条规则，否则比的是两次选择。
- 已有同模型但**不同口径**的数（`results/sum-target-20260927/`，episode 均值）：
  stats 78.0 / ergastf1 24.0 / genome 220.3（三 seed 均值）。**不能直接搬进 02 表**——口径不同（见 §5）。

## 3. 要跑什么

对 **3 个 seed × 3 个目标库（stats / ergastf1 / imdb）× K=0** 跑 population 评估：

```bash
# 每个 seed、每个库各一次；示例为 seed42 / stats
python3 scripts/target_population_eval.py \
  --cache-root artifacts/cache \
  --split test \
  --databases stats \
  --checkpoint runs/arch4k-sum-seed42-20260927/last.pt \
  --price-checkpoint /gly/weilin/PRICE-master/results/model_params.pth \
  --maxima "12,14,8,10" \
  --output /gly/weilin/spatial-meta-ce-fomaml-sum-pop-202610xx-runs/s42-stats.json \
  --protocol /gly/weilin/spatial-meta-ce-fomaml-sum-pop-202610xx-runs/s42-stats.protocol.json
```

要点：

1. **K=0、全库唯一查询、单次全量、无 support**，与 108 号完全一致。
2. `--maxima "12,14,8,10"` 与现有目标库表一致（不是源库的 `[12,12,7,10]`）。
3. 目标库只读一次、不训练、不选 checkpoint；这是评估，不是选择。
4. 三个 seed 的结果**不许挑**：全部进表，报均值 ± SD。

## 4. 验收标准

- 每份 `protocol.json` 的 `target_databases_read` 只含被评估的那一个库；`checkpoints` 记录 sha256 与 §2 一致。
- 每份输出的 `n` 与现有 ERM-sum 表逐库一致：stats 2,603 / ergastf1 1,258 / imdb 521。
  不一致就停下查口径，不许硬凑。
- 三 seed 汇总后，02 表能并排放 **ERM-sum** 与 **FOMAML-sum** 两行（或多一列），PRICE / PG 列不变。
- 若 FOMAML-sum 在三个 seed 上方向不一致（有的赢有的输），照实写，不取最好的 seed。
- 报告 01 节的口径声明随之改写：主表换成 FOMAML 后，「主表是 ERM 臂」这句话必须删掉或改写。

## 5. 为什么不能拿现有的 episode 数凑（重要）

`results/sum-target-20260927/` 里确有 FOMAML-sum 的 K=0 数（78.0 / 24.0 / 220.3），但它是
**episode 均值**口径，与 02 表的 **population** 口径不同：

- 同一个 ERM-sum seed42，episode 口径 stats 84.86、population 口径 83.72；
  genome 差距更大：episode 226.69 vs population 170.82（约 33%）。
- 两个口径的差别来自唯一查询去重、maxima 布局与百分位空间，不是模型差异。
- 108 号已记录两套算法在部分库上逐位相同、在另一些库差 1e-5~1e-4；
  但这里 33% 的量级说明**不是同一套数据在算**，不能混排。

所以 02 表现在**不能**改成 FOMAML-sum——不是不想改，是这条口径的数还不存在。
任何把 episode 数填进 population 表的做法都等于伪造。

## 6. 成本与风险

- **零训练**：只是 3 seed × 3 库的前向推理，按 102 号的经验单库分钟级；总耗时远小于重训。
- **风险 1：checkpoint 与缓存不匹配**。若 `artifacts/cache` 已变，`prepare` 的特征版本
  （`price40-group-logfanout-v1`）必须一致；不一致会让 `target_population_eval.py` 直接报错，不会静默。
- **风险 2：FOMAML-sum 可能更差**。若结果如此，就照实写「元学习臂在主表口径上不优于 ERM 对照」，
  并把 ERM 留在主表或两行并列；不允许换 seed、换库、换 maxima 去找好看的数。
- **风险 3：三个 seed 的离散度**。genome 在 108 号已见 SD/均值 64% 的先例；若 FOMAML-sum 同样散，
  表中必须带 SD 与三 seed 区间，不能只写均值。

## 7. 完成后要改的地方

1. `reports/advisor-report-20261027-talk.html` 的 02 节：加 FOMAML-sum 列/行，注明三 seed。
2. 01 节口径声明：从「主表是 ERM-sum 臂」改为「主表两臂并列」或「主表为 FOMAML-sum」。
3. 06 节：episode 均值表与新的 population 表并存时，必须各自标明口径。
4. `实验文档/104-导师汇报报告与文献对标.md` 新增一节记录结果与决策。
5. `实验文档/99-进度与下一步.md` 入口更新。

## 8. 不做的事

- 不重训、不调参、不因为结果不好而换 checkpoint。
- 不把 population 与 episode 两种口径混进同一张表。
- 不在目标库上做任何选择（这是红线），只做评估。

## 9. 执行记录（2026-10-27）

- 已执行：3 seed × 3 库（stats / ergastf1 / imdb）K=0 population，结果归档 `results/p2-2-fomaml-sum-target-3seed-20261027/`，逐 seed 数字、对照与结论见 [104 号 §7.13](104-导师汇报报告与文献对标.md)。
- 服务器：`/gly/weilin/spatial-meta-ce-fomaml-sum-pop-20261027-runs`；代码目录 `/gly/weilin/spatial-meta-ce-p2-2-source-broad-20261021-code`（`scripts/target_population_eval.py` sha256 `593da187…7d71`）。
- 与 §4 的口径差异：实际为每个 seed 单次调用合并评估三库，因此每份 `protocol.json` 的 `target_databases_read` 记录三库，不是 §4 字面要求的单库一份；未读取 genome，评估范围未超出三库。
- 结果：FOMAML-sum Q95 均值 stats 78.19±4.91、ergastf1 22.31±1.21、imdb 45.96±16.78；ERM-sum 对照 81.07±2.44、23.44±1.54、33.13±1.79。主表口径下元学习臂不占优，02 表维持 ERM 主表。
