# G0 源数据身份链核验（2026-11-02）

> 状态：**已完成（2026-11-02）**；23 个训练库与 3 个验证库的内容复现均已核对。
> 范围：只读核验；未修改 PRICE 原始统计、未训练、未读取目标库标签、未启动任何正式实验。

## 1. 要回答的问题

联合分布方案的前提是：源库存在**与冻结统计同版本**的原始表快照，且能从该快照重现生成链路中的统计。此前 P2 盘点只确认了 workload 与统计 pickle，`raw_source_table_snapshot_verified=false`、`historical_statistics_to_snapshot_lineage_verified=false`（见 `实验文档/results/p2-source-semantics-20261005-03/g1_inventory.json`）。

## 2. 已确认事实

### 2.1 候选原始快照

- 路径：`/gly/weilin/datasets.tar.gz`
- 大小：3,870,918,542 字节；归档内文件 mtime 为 2024-04-23/24，属主 `ljw`
- SHA256：`b5918f1e1743599bb717dfae15cd1715b4156363d79a2aabd5dfee552f1c62f8`
- 内容：`datasets/<db>/` 共 29 个库（含 23 训练库、3 验证库及 4 个目标库）；本次只解压核验 26 个源库，目标库未解压。
- 每个库带 `postgres_create_<db>.sql`，其中 `\copy <table> FROM '<file>.csv' CSV DELIMITER '|'` 给出表与 CSV 的映射。

### 2.2 冻结统计的生成链路

- 生成脚本：`/gly/weilin/PRICE-master/utils/statistics/gen_{size,histogram,summary,fanout}.py` 与 `tools.py`。
- `tools.load_table_datas` 直接读取 `datas/datasets/<db>/postgres_create_<db>.sql` 引用的 **CSV 文件**（`pandas.read_csv(sep='|', quotechar='"', escapechar='\\')`），不经过 PostgreSQL 导库。
- `PRICE-master` 为 git 仓库，提交 `0a9358eafb332201678e411ac35dc78141f751bd`；`utils/statistics/` 未被列入本地修改（`git status --porcelain` 只显示 finetune/meta_train/model 等 6 项）。
- 脚本哈希：
  - `gen_size.py` `6f2c7e28dadeb020870b2deae64dc58cf6e140415c8c7a2bb88e2a44c8df6819`
  - `gen_histogram.py` `8ad5181892859bc548921db46e21d49d9d9a29c51bb236f3074d3c80c9012cf0`
  - `gen_summary.py` `30ea3df4b26e8cbb3c7e53a33a8b55b3af664bede0be2ff131c2b4e994837a5d`
  - `gen_fanout.py` `dcef7cc56165a7ee61b02dc83ae2d5e2e988689b3c74333f45650e61f57d05ad`
  - `tools.py` `d89ed28db054f78d2d42e5a81435676dfd4b415ef5b1d76d68549e52f87ac307`
- 生成环境：`/gly/weilin/PRICE-master/venv`，pandas 2.1.3、numpy 1.26.1。

### 2.3 缓存 manifest 绑定

对 26 个源库逐一核对：`artifacts/cache/pretrain/<db>/manifest.json` 中记录的 5 个统计文件 SHA256 与当前 `/gly/weilin/PRICE-master/datas/statistics/pretrain/<db>/*.pkl` **全部一致**；记录的 `workload_sha256` 与当前 `datas/workloads/pretrain/<db>/workloads.sql` **全部一致**。

提取器：本地 `vendor/price_features/features_tool.py` 的 SHA256 为 `88a6450d32fabe2ebb9f180697937d1679fe4f5e974b23bab41a36d34a932fe4`，与 manifest `extractor_sha256` 一致；PRICE-master 自带的 `setup/features_tool.py` 为 `f3bc51e6…`，与缓存构建所用版本不同（缓存用的是本项目 vendor 副本）。

### 2.4 复现核验（三验证库，已完成）

按生成脚本同样的读取方式，从 `datasets.tar.gz` 解压的 CSV 重算统计，并与冻结 pickle 逐项比对：

| 库 | size（行/列） | histogram40 | summary40 | fanout40 |
|---|---|---|---|---|
| chembl | 16/16 表一致 | 29/29 一致 | 32/32 一致 | 38/38 键与值一致 |
| legalacts | 5/5 一致 | 10/10 一致 | 10/10 一致 | 16/20 共享键值一致；**4 个冻结键无法由当前 workload 重现** |
| telstra | 5/5 一致 | 6/6 一致 | 4/4 一致 | 20/20 一致 |

### 2.5 全源库结构核验（26 库，已完成）

- 23 个训练库 + 3 个验证库：CSV 逐表行数、列数均与 `size.pkl` 的 `num_rows`/`num_cols` 完全一致（逐库逐一比对，无 mismatch）。
- SQL 表定义与 CSV 表头逐列比对：24 个库完全一致；差异仅两处：
  - `airline.on_time_on_time_performance_2016_1`：CSV 比 SQL 多一列 `uniquecarrier`（83 vs 82）。
  - `tubepricing` 7 张表：CSV 比 SQL 多一列 `unique_feature`。
  - 生成脚本对这两类情况会由 pandas 推断多余列，冻结 `num_cols` 与 CSV 表头一致；属于 SQL 说明文件不完整，不是数据不一致。

### 2.6 legalacts fanout 异常

- 冻结 `fanout40.pkl` 含 `leg_ll.actid1 ↔ leg_s.actid`、`leg_ll.actid2 ↔ leg_s.actid` 四个方向键。
- 当前 `workloads/pretrain/legalacts/workloads.sql` 中不存在这些连接（grep 计数为 0）；其哈希又与缓存 manifest 一致。
- 推断：冻结 fanout 由更早版本的 workload 生成，之后 workload 被替换但 fanout 未重算。数据内容本身（size/hist/summary）不受影响；这些额外键在当前 workload 中不会被查询使用。

### 2.7 训练库内容复现（23 库，已完成）

按与生成脚本相同的读取方式逐库重算，逐库结果见 `summary_server.json`，结构化证据见 `evidence.json`：

- histogram40：**631 个列全部一致**（每库 hist_bad=0）。
- summary40：**854 个离散列全部一致**（每库 summary_bad=0）。
- fanout40：21/23 库键集合与数值完全一致；`sakila` 共享 62 个键值一致，但冻结文件多出 6 个方向键（`skl_cus/skl_stf/skl_str` 三表 `address_id` 两两直接连接），当前 workload 中不存在这些连接。

加上三验证库，26 个源库合计：histogram 676 列、summary 900 列全部复现；fanout 24/26 库完全一致，legalacts 与 sakila 各有 4/6 个只存在于冻结文件的连接键（共享键值均一致，且没有"只在重算结果中出现"的键）。

## 3. 结论：G0 数据可用性判定

**有条件通过（功能等价层面）。**

- 26 个源库的原始 CSV（`datasets.tar.gz`）能够精确重现冻结统计的行列数、histogram40、summary40；fanout 24/26 库完全一致，另外两库的差异是**旧 workload 遗留的多余键**，不影响数据内容。
- 缓存 manifest 中记录的统计文件哈希、workload 哈希与当前文件全部一致；特征提取器哈希也与本地 vendor 副本一致。
- 按 [13 号协议 §6.1](../13-P2源侧语义与真实选择率审计协议.md) 的"等价证明"路径：逐表内容与 schema 已认证（2 处 SQL 说明文件不完整，不影响数据），可以在不替换冻结统计的前提下成立。

**仍未满足的部分：**

- 没有历史生成 manifest 或日志；归档本身无 README。因此这是**复现证明**，不是文档学意义上的"原始版本 → 导入 → 统计"身份链。
- legalacts 与 sakila 的 fanout 与当前 workload 不同版；若后续工作依赖 fanout 的 join 集合，应以本次快照重新生成并记录新 manifest，而不是沿用冻结 fanout 当作精确同版。

对联合分布方案的直接含义：可以基于该快照构建新的、无标签的条件统计（G1），不需要目标库，也不需要重跑 PRICE 统计；但新增统计/缓存/特征仍需单独协议与授权，不能把本核验当作实验许可。

## 4. 限制与操作说明

- 本核验证明的是**功能等价**（同一批 CSV 能精确重现冻结统计），不是文档学意义上的历史生成日志；归档本身没有 README 或生成记录。
- 时间戳（统计与脚本同为 2025-04-30 16:07:26）是打包时间，不能当作生成时间证据。
- 期间一次脚本因路径配置把两个 `fanout_check_*.pkl` 临时写入 PRICE 统计目录（新增文件，未覆盖既有文件）；已立即移出并复核 5 个冻结 pickle 的 SHA256 未变。后续输出全部改到 `/tmp/verify-src-20261102`。
- 目标库 `imdb/stats/genome/ergastf1` 未解压、未参与本核验，也未进入任何训练或选型。
- 核验用的 13 GB 临时解压目录 `/tmp/verify-src-20261102` 已在记录结果后删除；核验脚本为临时脚本，方法（与 `gen_*.py` 相同的 pandas 读取参数、逐列重算）已在本文说明，可据此重建。
- 本报告只陈述可复核事实；没有训练、推理或精度结论。
