# 98-P2-2 三 seed 复核

日期：2026-10-20
分支：`experiment/p2-2-three-seed-20261020`（从 `experiment/p2-2-price-population-20261020` 切出）
性质：只读复核，不训练、不适应、不选 checkpoint，`target_databases_read: []`
前置阅读：[97号 §4 任务 A](97-P2-2后续补证任务交接计划.md)、[96号](96-P2-2方向信号可预测性审计.md)

---

## 0. 一句话结论

96 号的方向信号结论在 seed42/43/44 上**全部复现**，97 §4.3 的三条冻结门槛**全部通过**（HGB 主模型）：
库内 AUC 0.910–0.985、LODO AUC 0.382–0.613、telstra LODO 三 seed 均 < 0.5。
**96 号的「库内极强、跨库不迁移」可以进论文正文。**

但必须同时带上一条 96 号已披露、本轮量化的限定：**跨库门槛是在 HGB 主模型上判定的。**
同一门槛套到 logistic 辅助视图会**不通过**——telstra 的 logistic LODO 为 0.6522 / 0.7487（seed43/44），
最大值 0.7487 > 0.65。也就是说「telstra 系统性反向」是 HGB 特有的，logistic 下 telstra 是**弱迁移**。
论文里任何「跨库不迁移」的表述都必须写明是哪个模型类。

---

## 1. 目的与冻结门槛

96 号的 AUC 0.92–0.98 / LODO 0.38–0.60 全部来自 **seed42 单个 checkpoint**。
本轮用同协议的 seed43 / seed44 复核，判定门槛在 [97号 §4.3](97-P2-2后续补证任务交接计划.md) 于开跑前冻结：

| 判据 | 门槛 | 模型 |
|---|---|---|
| 库内 AUC 三 seed 最小值 | 必须 > 0.85 | HGB（96 号主模型） |
| 库内 AUC 三 seed 标准差 | 报告，不设门槛（描述性） | HGB |
| LODO AUC 三 seed 最大值 | 必须 < 0.65 | HGB（96 号主模型） |
| LODO 方向一致性 | 三 seed 中至少 2 个 seed 的 telstra LODO < 0.5 | HGB |

判定脚本 `scripts/three_seed_audit_summary.py` 里的阈值是写死的常量，注释标明「在 seed43/44 开跑前冻结」；
汇总一律用 **mean ± 样本 SD（ddof=1）**，不报最好种子。任一门槛不过 → 96 号降级 `inconclusive`，只能进附录。

---

## 2. 输入与一致性核对

| 项 | seed42 | seed43 | seed44 |
|---|---|---|---|
| checkpoint | `p1-screen-20261020/baseline/last.pt` | `.../seed43/baseline/last.pt` | `.../seed44/baseline/last.pt` |
| checkpoint sha256 | `563ec159…7d8a3` | `5617f7dd…7ad15` | `2546821e…bec093` |
| arm / 步数 / 参数量 | baseline / 4000 / 132,545 | baseline / 4000 / 132,545 | baseline / 4000 / 132,545 |
| predictions.csv sha256 | `325a2160…62c88a` | `d647d8a1…de06c5e` | `55c29c4a…73f0d` |
| 导出行数 | 2,990,080 | 2,990,080 | 2,990,080 |
| cache content_hash | chembl `ab9ce1fc…` / legalacts `cb7401d4…` / telstra `9258ff2f…` | 同左 | 同左 |

逐项核对（97 §4.5）：

- **行数一致**：三个 CSV 各 2,990,080 行数据行（+1 表头）。逐库逐 K 计数相同：
  chembl 250,000 × 4、legalacts **247,520** × 4、telstra 250,000 × 4。
- **导出协议一致**：三个 seed 用同一份 `configs/eval_p0_3.json`
  （`shots=[0,5,20,100]`、`support_runs=5`、`support_seed=2026`），同一份 `scripts/export_source_predictions.py`
  （三处 sha256 均为 `064acd9793e2a411df80bd74c5d4c069ff5c9b4e3018485b098c51f6c834db42`，
  与 seed42 那次导出的脚本逐字节相同）。
- **训练配置只差 seed**：`tail_aligned_screen.json` / `_seed43.json` / `_seed44.json` 三者 diff 只显示
  `"seed": 42 → 43 → 44`，其余差异为 JSON 重新格式化（`purpose` 的 § 转义、数组展开、`guard` 0.10/1.10、
  `support_sizes`、`model` arch/width、五个 arm 的 loss 完全一致）。
- **cache 同一**：三库 content_hash 三个 seed 完全相同，且与 96 号审计记录的一致。
- **审计协议一致**：三个 audit 用同一份 `scripts/direction_signal_audit.py`、同一组参数
  （`--databases chembl legalacts telstra --k 20 100`），protocol.json 逐字段相同。

---

## 3. 逐 seed 结果（HGB 主模型）

### 3.1 库内 pooled OOF AUC

| 库 | seed42 | seed43 | seed44 | 均值 ± SD |
|---|---:|---:|---:|---:|
| chembl | 0.9167 | 0.9115 | 0.9102 | **0.9128 ± 0.0034** |
| legalacts | 0.9553 | 0.9633 | 0.9650 | **0.9612 ± 0.0052** |
| telstra | 0.9771 | 0.9768 | 0.9847 | **0.9795 ± 0.0045** |

95% CI 下界（bootstrap 1000 次）逐 seed：

| 库 | seed42 | seed43 | seed44 |
|---|---|---|---|
| chembl | [0.9143, 0.9191] | [0.9092, 0.9138] | [0.9078, 0.9126] |
| legalacts | [0.9538, 0.9568] | [0.9621, 0.9647] | [0.9637, 0.9664] |
| telstra | [0.9761, 0.9780] | [0.9758, 0.9778] | [0.9840, 0.9855] |

### 3.2 近重复结构签名分组后（敏感性检验）

| 库 | seed42 | seed43 | seed44 | 均值 ± SD |
|---|---:|---:|---:|---:|
| chembl | 0.9055 | 0.9011 | 0.8985 | 0.9017 ± 0.0035 |
| legalacts | 0.9447 | 0.9549 | 0.9583 | 0.9527 ± 0.0071 |
| telstra | 0.9284 | 0.8926 | 0.9452 | 0.9221 ± 0.0268 |

### 3.3 留一库外推（LODO）

| 留出库 | seed42 | seed43 | seed44 | 均值 ± SD |
|---|---:|---:|---:|---:|
| chembl | 0.5412 | 0.5081 | 0.5365 | 0.5286 ± 0.0179 |
| legalacts | 0.5962 | 0.6130 | 0.5755 | 0.5949 ± 0.0188 |
| telstra | **0.3822** | **0.4374** | **0.4721** | 0.4306 ± 0.0453 |

**telstra 三个 seed 全部 < 0.5**（0.3822 / 0.4374 / 0.4721），96 号的「telstra 系统性反向」在主模型上复现。

### 3.4 辅助模型类（Logistic）

库内 pooled OOF AUC：

| 库 | seed42 | seed43 | seed44 | 均值 ± SD |
|---|---:|---:|---:|---:|
| chembl | 0.8266 | 0.8256 | 0.8064 | 0.8195 ± 0.0114 |
| legalacts | 0.7698 | 0.7719 | 0.7939 | 0.7785 ± 0.0134 |
| telstra | 0.8143 | 0.8190 | 0.8748 | 0.8360 ± 0.0336 |

LODO AUC：

| 留出库 | seed42 | seed43 | seed44 | 均值 ± SD |
|---|---:|---:|---:|---:|
| chembl | 0.3888 | 0.4213 | 0.4957 | 0.4353 ± 0.0548 |
| legalacts | 0.5713 | 0.6217 | 0.6585 | 0.6172 ± 0.0438 |
| telstra | 0.6626 | 0.6522 | 0.7487 | 0.6878 ± 0.0530 |

---

## 4. 判定（逐条对照 97 §4.3）

| 判据 | 实测 | 门槛 | 结论 |
|---|---:|---|:--:|
| 库内 AUC 三 seed 最小值 | **0.9102**（chembl / seed44） | > 0.85 | ✅ |
| LODO AUC 三 seed 最大值 | **0.6130**（legalacts / seed43） | < 0.65 | ✅ |
| telstra LODO < 0.5 的 seed 数 | **3**（0.3822 / 0.4374 / 0.4721） | ≥ 2 | ✅ |

**总判定：`confirmed`。** 96 号的结论就 seed 覆盖这一条而言不再是 `needs_evidence`，可进论文正文；
库数 n=3、目标库未读等其余边界见 §6。

描述性量（不设门槛）：库内 AUC 的 seed 间样本 SD 为 0.0034 / 0.0052 / 0.0045，
**远小于 seed 间差异本身的量级（0.9102–0.9847）**，说明「极强」不是某个种子的幸运值。
与 94 号测得的「单 seed 筛选噪声 6.28pp」相比，本轮的 seed 间波动小一个量级——
因为这里比较的是**固定数据上的分类器 AUC**，不是被噪声淹没的端到端提升幅度。

---

## 5. 两条必须写进论文的限定

### 5.1 跨库门槛是 HGB 判定，logistic 辅助视图不通过

96 号 §3.7 已经披露「Logistic 在 telstra 上是 0.663，与 HGB 的 0.382 完全相反，说明连『迁移方向』都不稳定」。
本轮把这条量化到三 seed：**logistic 的 telstra LODO 为 0.6626 / 0.6522 / 0.7487，三 seed 均 > 0.5**，
其中 seed44 的 0.7487 **超过 0.65 门槛**。若把 §4.3 的 LODO 门槛套到 logistic 视图，**判定不通过**。

可写的结论：**「跨库不迁移」在两个模型类上的共同部分是 chembl 不迁移（两者均 ≤ 0.5）
且没有任何库出现强迁移；但『telstra 反向』是 HGB 特有的，logistic 下 telstra 是 0.65–0.75 的弱迁移。**
不可写：「方向信号在所有模型类、所有库上都不可迁移」。

### 5.2 「LODO 准确率 > 多数类基线」不是 seed 稳定判据

96 号 §4.1 用「准确率 vs 多数类」作为跨库判据之一。三 seed 逐项核对（HGB）：

| 库 | seed42 | seed43 | seed44 | 均值 |
|---|---|---|---|---|
| chembl | 0.5430 > 0.5109 ✅ | 0.4816 < 0.5510 ❌ | 0.5614 < 0.5645 ❌ | 0.5287 < 多数类 0.5421 |
| legalacts | 0.6358 < 0.6396 ❌ | 0.5754 > 0.5322 ✅ | 0.5694 > 0.5255 ✅ | 0.5936 > 多数类 0.5658 |
| telstra | 0.4450 < 0.5746 ❌ | 0.5706 < 0.6010 ❌ | 0.7592 < 0.7628 ❌ | 0.5916 < 多数类 0.6462 |

多数类基线本身随 seed 变（chembl 0.5109 / 0.5510 / 0.5645），因为方向类别的先验比例依赖各 seed 的
K=0 预测。**chembl 与 legalacts 的「优于多数类」在三个 seed 间都会翻转**，telstra 三个 seed 都不优。
所以这一条只能作描述性观察，**不能作为跨库不迁移的证据**；跨库结论应建立在 AUC 上（§4 的门槛）。

---

## 6. 边界（不能写成什么）

- **本轮只覆盖 seed 覆盖这一条**。96 号原有的其余边界不变：跨库结论的样本单位是「库」，n=3；
  事后方向修正仍是同一批验证查询上的相关性分析，不是精度改进；未读任何目标库。
- **LODO 判据仍是 n=3 库**。本轮提升的是「同一划分下换 seed 是否稳定」，没有增加库的数量；
  97 号任务 B（23 库）仍是把「不迁移」推广成普遍结论的唯一途径。
- **三 seed 都是 baseline arm、4000 步、同一训练配置**，只换了随机种子；不代表其他 arm、
  其他训练预算或 FOMAML 元初始化下的同一结论。
- **审计是源侧的零样本方向信号**，与目标库上的精度改进无关；不要与 94/95 号的 Q95 数字并列。
- 近重复分组的 telstra seed 间 SD 偏大（0.0268，区间 0.8926–0.9452），引用时应给区间而非只给均值。
- 推理耗时不等于数据库端到端加速。

---

## 7. 产物

| 类型 | 路径 |
|---|---|
| per-seed 审计 | `实验文档/results/p2-2-direction-audit-20261020{,-seed43,-seed44}/audit.json` + `protocol.json` |
| 三 seed 汇总 | `实验文档/results/p2-2-direction-audit-20261020-three-seed/summary.json` + `protocol.json` |
| 汇总脚本 | `scripts/three_seed_audit_summary.py` |
| 单测 | `tests/test_three_seed_audit_summary.py`（13 项） |

`summary.json` 含逐 seed 原始指标、每格 mean ± SD（带参与聚合的 seed 数 `n`）、以及 §4 的三条门槛判定。
`protocol.json` 冻结了口径、门槛来源（97 §4.3）与三份输入的 sha256 / content_hash。

---

## 8. 执行记录

**服务器**（`/gly/weilin/spatial-meta-ce-p2-2-three-seed-20261020-code`）：

- 预测导出（GPU）：`export_source_predictions.py` + `configs/eval_p0_3.json`，
  checkpoint 为 seed43 / seed44 的 baseline `last.pt`，各 2,990,080 行。
  seed43 用 GPU 1、seed44 用 GPU 2 并行；PRICE 项目 venv（Python 3.11.4 / torch 2.1.0 / numpy 1.26.1），
  `CUDA_VISIBLE_DEVICES` 显式指定，batch 128。
- 审计（CPU，**本地 macOS**）：seed43 与 seed44 并行，各约 37 分钟。
  审计只用 numpy + scikit-learn，不含 torch；三个 seed 全部在**同一环境**（macOS / Python 3.11.7 /
  numpy 1.26.4 / scikit-learn 1.2.2）下完成，与 96 号 seed42 审计的环境一致，
  避免把 sklearn 版本差异混进 seed 间差异。
- cache 只读 `/gly/weilin/spatial-meta-ce/artifacts/cache/pretrain/`，content_hash 与 96 号一致。
- 导出日志：服务器 `runs/p2-2-seed{43,44}-20261020/export.log`；审计为本地前台运行。
- **未产出结果的旁支检查**：曾尝试在服务器用 `/opt/conda/bin/python`（Python 3.8 / scikit-learn 1.0）
  重算 seed42 审计，作为「换 sklearn 版本是否改变结果」的对照。因主机负载升到 101（共享机器）而中止，
  未产出结果。**不影响本轮结论**：三个 seed 已在与 96 号 seed42 完全相同的环境（macOS / sklearn 1.2.2）下算完，
  需要的是 seed 间可比，而不是跨环境可比。

**验证**：

- 本地 `PYTHONDONTWRITEBYTECODE=1 python3 -B -m unittest discover -s tests`：1406 项，新增 13 项全过
  （均值/样本 SD、目标库拒绝、缺失模型记 null 而非 0、三条门槛的正反例）。
- **全量套件存在既有偶发失败**，与本轮无关：`tests/test_meta65_joint_supervisor.py` 的进程监管时序项，
  两种症状——`os.killpg` 的 `PermissionError`，与 `assertIn(SIGKILL, sent)` 报「9 not found in [15]」
  （子进程在 0.08 s 宽限期内退出，于是只发了 SIGTERM）。该文件**单跑 40 项全过**；
  在父提交 `ebae49f`（不含本轮改动）上 6 次全量复跑有 **3 次**同样失败，
  本分支 5 次复跑有 2 次失败，量级一致。97 §9.7 已记录此现象；本轮未触碰任何监管代码。
- `ruff check scripts tests spatial_meta_ce`：无问题。
- 三个 seed 的 query 数与 ties_excluded 均为 50,000 / 49,504 / 50,000 与 0，与 96 号一致。
