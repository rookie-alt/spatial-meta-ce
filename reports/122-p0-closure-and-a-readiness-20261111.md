# 122 — P0 验收收口与 A 启动前准备

日期：2026-11-11。**结论：P0 总门未通过；A 预算未定／待授权；不得启动 A/B/C/D。** 本文是审计交付，不将 [122 号协议](../实验文档/122-PRICE式编码器与元学习低秩适配分阶段协议.md) 标为已执行，不声称架构、LoRA 或元学习有效。

## 1. 范围、版本与证据读法

- 基准：`experiment/price-meta-lora-p0-real-20261111`，`89623047d644842dca134867b6a6ec5f38c0e756`。新独立 worktree：`.claude/worktrees/122-p0-closure-20261111`；独立分支：`audit/122-p0-closure-20261111`。未切换/修改主工作树，未合并 main、重写历史或 force push。
- 生产实现提交：`b02c21a1c67bf9d82674c7bb428c11aba93a2e38`；部署来源提交：`e6b60f04c963dd760d1d6348e86407e61d3f0325`。历史验证清单19项文件与当前检出全部字节 SHA 相等；本轮源核验覆盖的16项部署代码/配置 SHA 也与本地相等。服务器部署目录没有 `.git`，因此证明的是**已列文件字节匹配**，不是远端 Git HEAD 或整个部署树完整性。
- 已完整阅读协议、[真实源预检报告](122-p0-real-source-preflight-20261111.md)、[实现报告](122-p0-a-implementation-20261111.md) 及两份预检 JSON。原始 `data_preflight_passed`、`p0_gate_passed=false`、`formal_training_authorized=false` **原件未修改**。
- 主代理只执行源侧文件只读审计与无模型更新的合成 NumPy 复算。**子代理执行偏差另见 §8，不以“全程只读”掩盖。** 未观察到本轮正式训练、GPU 计算或真实目标缓存读取；没有独立文件访问追踪，不能将“未观察到”夸大为取证证明。
- [证据包说明](122-p0-closure-evidence-20261111/README.md)、[SHA 清单](122-p0-closure-evidence-20261111/manifest.json)、[实际命令与退出码](122-p0-closure-evidence-20261111/commands.json)、[源缓存/计划核验](122-p0-closure-evidence-20261111/source_readonly.json)、[历史复用反证](122-p0-closure-evidence-20261111/checkpoint_readonly.json)、[独立 NumPy 复算](122-p0-closure-evidence-20261111/numpy_recompute.json)。脚本原文以 `.py.txt` 归档，不接入库/CLI，不是正式 runner。

**状态含义**：已验证＝在明确所列历史合成或本轮真实源范围取得证据；仍缺证据＝不得推断通过；失败＝观察到违反条件。合成测试是工程证据，不是研究训练结果。历史证据未重跑不等于本轮新测。

## 2. §4 P0 验收表

| 项目 | 判定 | 可复核证据与限定 |
|---|---|---|
| 固定协议/代码提交/原始 SHA | 已验证 | 协议 SHA `d95fffbb030191b28ff02c1c80ec00e100f7ac18d2b3151a793e44bf8cfc07f3`；19项历史验证文件全匹配，完整清单见manifest。 |
| §7 全部必过验证 | **仍缺证据** | 见§3。候选在有限loss下内梯度变非有限、以及外层非有限梯度的受控失败证据均不完整；NaN loss拒绝和正常梯度通过不能替代这两条梯度失败路径。 |
| 合成 episode FOMAML 外循环反传 | 已验证（历史合成） | `tests/test_price_model.py:254–301` 检验base/head/A/B组级非零有限梯度及不污染原参数；历史日志对应测试ok，代码SHA相同。 |
| 新旧 checkpoint 往返 | 已验证（历史合成） | `test_price_model.py:406–465`；类型/config/容量/cache篡改拒绝，legacy strict保留。不证明历史A0-ref可复用。 |
| 23train容量及三源max_counts | 已验证（本轮真实、独立算法） | 26个manifest字节/内容SHA、156个arrays SHA；逐行 `np.diff(offsets)` 求maxima，仅23train取 `[12,12,7,10]`；三源unsupported=0。 |
| sealed training/source plan 身份 | 已验证（本轮全量） | 3×16,000 episode索引/hash、数据hash、交集逐项核对；source原始/增补计划无重采样。 |
| 输入配置覆盖 | 已验证（字段）；**失败（原样可运行性）** | 12份JSON冻结字段及SHA匹配，但 `data_config=configs/data.json` 仍解析到不存在的 `<server-code>/artifacts/cache`。上轮是内存override，不是已封存运行配置。 |
| unsupported时A前停止 | 已验证 | `price_capacity.py:163–204`、`test_price_protocol.py:60–70,119–150`、`test_price_model.py:160–181`；本轮无超限、截断、扩容或排除。 |
| smoke资源边界 | 历史合成证据已验证；**本轮执行合规失败** | 历史2-update supervisor仅用于smoke；本轮子代理自行运行完整模型测试，缺本轮要求的资源监管，见§8，不回填为合规smoke。 |
| 正式成本/总预算/资源监督 | **仍缺证据** | 无122四臂真实workload计时；无冻结122整矩阵parent supervisor与GPU/CPU小时总上限。§6只是草案。 |
| 授权状态不变 | 已验证 | 原始两门仍false；本轮获准正式run=0，此审计不产生把任一门改true的权限。 |

## 3. §7 十类必过验证逐项核验

| 类别/断言 | 判定 | 证据位置、范围与缺口 |
|---|---|---|
| PG：零头 `forward == pg_log` | 已验证（历史合成） | `price_model.py:89–90,201–207`；`test_price_model.py:88–106`，lite/tiny逐位恒等。上轮单条chembl只摘要登记、缺单独stdout；本轮不重跑，不扩称全population验证。 |
| PG：trunk/head无第二路，0-step不变 | 已验证（历史合成） | `test_price_model.py:108–127` 改PG而head input不变，旧sum仍读head PG；`:94–96`0-step恒等。 |
| 标签：support/query/预测统一log空间 | 已验证（历史合成＋调用链） | `data.py:33–40,76–82`，`test_price_model.py:363–381`拒绝PRICE转换仍可反传；train/meta直接消费batch label。输出无clamp，指标floor不等于裁剪预测。 |
| 输入：fanout还原、padding、mask、counts、混合batch | 已验证（历史合成） | `price_model.py:127–174`，`test_price_model.py:129–158`检验expm1、-1000/table首维0/mask/log1p(counts)。单样本/混合batch以 `atol=1e-6` 比较，**非逐位承诺**。 |
| 容量：等于通过，超过显式unsupported | 已验证（历史合成＋本轮真实） | `test_price_model.py:160–181`含database/hash/group/actual/capacity信息；本轮独立offsets扫描全覆盖。 |
| 容量：只23train生成并写checkpoint | 已验证 | `price_capacity.py:102–121`，`checkpoints.py:64–79`；本轮23库maxima独立取max相等，三源只核对超限。未重读原始PRICE特征/统计。 |
| LoRA：注册Q/V A/B、r4/alpha4；support只改A/B | 已验证（历史合成） | `price_model.py:18–37,91–101`、`meta.py:86–101`、`test_price_model.py:183–214,254–301`；lite内循环10,240，无.data/delta缓存/跨episode保存。 |
| LoRA：query对base非零；K0/0-step不变 | 已验证（历史合成） | 合成非零head检验base/head/A/B组级非零有限梯度。零头trunk/LoRA为0、B=0时A首步为0是数学限定（`:216–251`），不声称fresh-zero-head LoRA内循环能改变预测。 |
| FOMAML：一阶、functional weights、不跨episode | 已验证（历史合成） | meta使用create_graph=False/functional_call(strict=True)；`test_meta65_scope_gradients.py:177–219,241–270`及候选`:254–301`。C正式runner未实现不等于数学接口未实现。 |
| FOMAML：非有限内层loss拒绝 | 已验证（历史合成） | `test_price_model.py:337–339` NaN label→FloatingPointError；实际在 `meta.py:66–67` 的loss检查先抛错，尚未到内梯度守卫。 |
| FOMAML：有限loss但非有限内梯度失败 | **仍缺证据（P0阻塞P1）** | `meta.py:134–138`有守卫，但上述NaN测试未到达该路径，历史日志不能证明受控梯度注入拒绝；本轮只登记，不执行模型补证。 |
| FOMAML：非有限**外**梯度失败 | **仍缺证据（P0阻塞P1）** | `train.py:655–662`有守卫，但122专项缺候选梯度仅外层变非有限的注入失败记录；`test_s4_anchor.py:558–575`仅模拟错误字符串。正常有限反传不代替此失败路径，本轮不启动模型更新补证。 |
| checkpoint：新旧往返预测一致 | 已验证（历史合成） | `test_price_model.py:383–465`、历史日志与相同代码SHA。当前无122研究checkpoint，不冒充正式产物。 |
| checkpoint：type/config/容量/cache错配拒绝 | 已验证（历史合成） | `checkpoints.py:64–112`及上述篡改负例；区分manifest文件SHA和content_hash。 |
| 确定性：eval两次一致、dropout0 | 已验证（重复适配属历史合成；dropout属静态实现/配置核对） | `test_price_model.py:303–339`仅检验fast weights逐位一致/原参数不变；`price_model.py:56–57`构造守卫要求dropout=0，本轮12份A配置的dropout均为0；**无dropout专项拒绝测试断言**，不声称测试覆盖此分支或跨设备bitwise。 |
| mode：support/query一致且恢复子模块mode | 已验证（历史合成） | `test_price_model.py:303–361`保存异构caller mode和requires_grad，不靠冻结外循环base实现scope。 |
| 数据：仅源pretrain，S/Q交集0，目标守卫 | 已验证（真实源身份＋历史拒绝） | 本轮白名单26源pretrain；`test_price_protocol.py:226–258,342–403`和plans校验。配置中目标名称不算读取数据，不使用历史目标数字。 |
| 统计：独立NumPy log-Q95/macro/配对SD | 已验证（本轮全合成） | 独立公式生成期望再比生产输出；log_q95=3.9386049789855493而log(q95)=3.943483687694013，等权macro≠pooled；三seed ddof1、精确SD0/全0、逐库+0.1反例覆盖。 |
| 统计：空4+桶/负迁移/seed-run层次 | 已验证（本轮全合成） | 空桶null、非空库等权；独立复算log_q95(K)−log_q95(0)、query恶化比例、自比均0；90单元→先5run再3seed→六格；support SD先每seed五run ddof1再平均。B/C生产判定未实现，**仅验证公式**。 |

历史全量：`reports/122-unittest-final-reviewed.log:2073,2075` **1784 tests / 175.439s / OK(skipped=16)**，SHA `93c9b2ad05aaa425ec09895c0159df0585f72fe03b5150fd5ceac917e6cb38d0`。主代理未重跑suite，无生产代码改动；Ruff本轮exit0。不用子代理越界新suite替代历史绑定。

## 4. 真实源身份与配置覆盖

### 4.1 原始SHA（完整清单见manifest）

| 文件/身份 | SHA256 |
|---|---|
| 原始preflight/remote_preflight.json | `fcd8758f9a4db87538c519da37cda128be7462ed4d186a483062f03369b538a9` |
| 上轮preflight_summary.json | `1cb282b987152b83ab85378e9a067b7a2deaedf0212555ebb0c3b7c55812d773` |
| configs/data.json | `a9057d65c7cb220bd21e7cd67dacbc15c0c20dcc87f61d9b3e6470c0ea094d90` |
| 内存override canonical配置 | `8766c14d1c2e434407de445a8bacc29ff5af3d541d028421c217357feea7b522` |
| 23train容量binding（非文件SHA） | `1b7326c0ce8a562c3988ac793f817f8a6834d0925c6068616f8009eeda95f1f8` |
| 112 plan_binding.json | `bb1bf8009af92c3813feb0ca1979a5f69f4f4c4a0bc00471bf203662be561ebc` |
| seed42训练计划文件 | `07abc216b5d12466e722617b2cfc8342014b05a4e9845412e04621c5be763289` |
| seed43训练计划文件 | `0f1e7a1b663a3a8446bc4e88a1fc478d9cb2a8f2a9ee84598fd42c448e1f88a0` |
| seed44训练计划文件 | `0c0c35dc08cef444a5dd3086db2317214a3fcd78383c518abe1493ef13e2f0d6` |
| 原source_eval_plan文件 | `7c212763ff3d3ae818f152e297e33afcb607e4f800dad5b1444d7d2c1591b972` |
| 122 source_eval_plan文件 | `26c83355b2ea5bee8cb403154d05e6bafa7b23e4b6f4fba5ee70ba0a9abd0a59` |
| 122 source_eval_plan内部binding | `b760ec2383d4fe0048b3a4e4e6aa2e10f815eca9cd23e4e6c7b98ad08a991764` |

独立脚本只用stdlib/NumPy，不调用生产容量/plan函数。26manifest/156arrays SHA与sealed data_sha256一致；48,000episode逐条校验索引/hash、K/64长度、无重复/交集和23train闭包。K次数按封存均匀抽样计划复用，**不要求每K次数恰等**，不重新抽样。source全部query ID/hash和support ID/hash核对；support排除后集合恰覆盖其余unique，不只核对分母算术。

| 源库 | unique | support排除 | Q_d | max_counts | unsupported |
|---|---:|---:|---:|---|---:|
| chembl | 50,000 | 100 | 49,900 | [11,12,7,10] | 0 |
| legalacts | 49,504 | 100 | 49,404 | [5,6,4,10] | 0 |
| telstra | 50,000 | 100 | 49,900 | [5,8,5,5] | 0 |

每库5×20互斥block，K5取封存前5/K20全20，support_seed2026；原版与122补ID版顺序相同，无plan_seed2718/shot_seed。本包保留完整300条support ID/hash；大训练计划不提交。未重读PRICE原始workload/statistics/features，缓存provenance只是历史记录，不是原始来源重认证。

### 4.2 12配置及可运行性

4臂×3seed文件SHA与上轮输出清单相符。逐字段核对 `method=erm, meta_objective=post, steps=4000, meta_batch=4, query_size=64, accumulation=1, outer_lr=3e-4, inner_lr=1e-3, inner_steps=5, adapt_scope=all, loss=smooth_l1, support_sizes=[5,10,20,50,100], fixed_budget=true, purpose=research`，各自model/seed/fixed_stage/plan/preflight SHA。A的ERM训练**没有support内循环**，inner_steps只是固定签名与评估适配字段，不声称训练内更新5步。

**启动阻塞A1**：`cli.data_settings`以代码根解析 `artifacts/cache`，服务器该路径不存在。真实预检用绝对cache_root只在内存；12模板不是可原样执行的封存环境。另获准后需冻结实际只读路径/配置SHA并复核；本轮不改服务器、模板或创建symlink。

直接依赖的vendored module.py/encoder.py SHA与本地匹配；部署缺vendor/price_features/features_tool.py，但派生缓存训练/模型路径不调用特征重提取。不能声称整个vendor完整部署，不获得重新提取或修改PRICE的权限。

## 5. A0-ref历史复用：三个seed均不可复用

对象只限 `/gly/weilin/spatial-meta-ce/runs/erm-sum-seed{42,43,44}-20260927/`，不声称穷尽全服务器。只读config/history/validation manifest；六个.pt文件SHA与ZIP内data.pkl primitive opcode，**全文件原始字节只用于SHA；只解析data.pkl primitive opcode，不torch.load、不unpickle、不解码或使用tensor值**。既有JSON的 `tensor_storage_read=false` 只表示未按storage成员加载/解码，不表示SHA阶段从未读取其原始字节；保留原始stdout不重写。

| 条件 | seed42 | seed43 | seed44 | 判定 |
|---|---|---|---|---|
| 完全匹配sealed plan | config无training_plan/fixed_budget，checkpoint无相应binding | 同左 | 同左 | 仍缺证据→不可复用 |
| selection=none | source-validation macro mean log-Q95 after adaptation | 同左 | 同左 | 失败 |
| 训练未读源验证库 | 25条非空validation score和三库manifest | 30条 | 29条 | 失败 |
| 仅last、无best选择 | 有best，step1500 | 有best，step2000 | 有best，step1900 | 失败 |
| last实际4000updates | 2500 | 3000 | 2900 | 失败 |
| ERM/SmoothL1/LR3e-4/4/64/accum1 | 部分字段相同不能替代plan/选择/步数 | 同左 | 同左 | 不足以准入 |
| 最终 | **不可复用** | **不可复用** | **不可复用** | 按12个新run申请，不执行 |

last SHA：42 `c04003a77e6d735d0f0d158889bc25bae7c011db9752b5bdb2b25e36ddd99ef4`；43 `b0506676f73158825a9144b950fd6b34e49792ae86b9b9a91997bbbc9041c1bc`；44 `38433ca8e963c31c2ebc9afba08495d3e0b42bc9a8a9d6aa0b604dd22c9edf01`。六权重及三组config/history/validation SHA见JSON。不从config写4000推断实际满4000；其他候选须重新核验，不补metadata伪装合规。

## 6. A矩阵、监督与预算申请草案（未就绪）

### 6.1 运行矩阵

| 臂 | backbone/PG | seeds | 训练 | 固定评估适配 | 新run |
|---|---|---|---|---|---:|
| A0-ref | sum，head PG＋anchor | 42/43/44 | ERM，4000×4，同seed sealed plan | all/full，1e-3，5步 | 3 |
| A0 | sum_anchor_only，仅anchor | 同上 | 同上 | 同上 | 3 |
| A1 | price_lite，d64，两stage各2层4head，无LoRA | 同上 | 同上 | 同上 | 3 |
| A2 | price_tiny，d32，其余同A1 | 同上 | 同上 | 同上 | 3 |

各run只23train pretrain，Adam3e-4/SmoothL1/accum1，从头不resume、不best/早停/源验证读取。12份last/config/plan/代码SHA均封存后，才在三源相同Q_d做K0及K5/20×5run。仅last指研究checkpoint选择；现实现另写initial_state.pt供梯度审计，不能作为额外候选权重。

以下为**协议/封存计划导出计数，不是实测运行输出**：每checkpoint33单元＝3库×(1K0＋2K×5run)，总396（36K0＋360K>0）；逐query输出 **19,694,928行**＝149,204×11×12。ERM拼support+query，不是只训64query：seed42/43/44的support一次呈现595,640/591,165/592,880，加query各1,024,000，单run共1,619,640/1,615,165/1,616,880；每臂三seed4,851,685，四臂19,406,740。loop合计48,000 optimizer更新/192,000episode backward，启动gradient audit/I/O/序列化另计。

H-A只用K0的A1−A0和A2−A0同方向/逐库保护，A1−A0-ref只描述完整变化。K5/20不选架构但仍必报；summarize_a只查K0矩阵，不能把仅36个K0单元的a_passed_source_screen当396单元全验收。未来整矩阵验收须独立核对K>0、身份/分母/文件SHA。

### 6.2 真实测量、estimate与未定

| 证据 | 性质 | 本轮用途 |
|---|---|---|
| 122合成2-update CPU墙钟3.2679s/RSS291,483,648B，内部更新0.2280s | 历史**合成实测**，非真实workload | 不用于A小时预算 |
| 455.98s/464.87s的4000-step外推 | 合成estimate，not_research_budget | 不作为真实成本或下界 |
| 112 sum历史200-step GPU计时 | 不同协议/结构的源实测 | 不搬用113小时预算/卡授权 |
| 122四臂真实workload训练/全population评估耗时、RSS/VRAM、输出bytes | **仍缺证据** | 预算未定／待授权 |
| GPU-hours/CPU core-hours、父wall与逐run deadline | **未定** | 不填猜测数字，不申请立即开训 |

合成容量[4,4,4,4] vs真实[12,12,7,10]：scale序列9→25/filter17→42。注意力形状二次项倍率不是整模型成本倍率，更不是小时预算；参数量2.22倍也不替代计时。无真实FLOPs/时延，不做速度结论。

### 6.3 资源监督与总上限申请字段

**当前已授权正式训练预算GPU-hours=0、CPU训练小时=0、run=0。** 下列仅为待填/待审草案，预算不完整即禁止启动。

| 字段 | 提案/要求 | 状态 |
|---|---|---|
| 总训练上限 | 12个新run、48,000optimizer更新；不补seed/延长失败臂 | 协议上限，未授权 |
| 并发/卡号 | 建议最多1训练worker，卡未指定；启动当时确认空闲、显式CUDA_VISIBLE_DEVICES | 提案，不占卡；旧状态不算预约 |
| CPU线程 | 建议每worker1计算线程，OMP/MKL/OpenBLAS=1；core-hours按进程树CPU秒 | 提案 |
| GPU/CPU总小时＋父wall | null／预算未定／待授权，含prepare/hash/audit/train/eval/archive/失败清理 | **阻塞A2** |
| VRAM/RSS硬限 | 建议8GiB VRAM、12GiB进程树RSS（单并发） | 非实测/非足够性保证，须测量后冻结批准 |
| 输出总硬限 | 建议整个A批次8GiB含CSV/权重/audit/log/sidecar | 约1969万行须字节预检；不够则失败，不删行 |
| 输出目录 | `/gly/weilin/spatial-meta-ce-price-meta-lora-A-<approved-date>-runs/`新目录 | 提案，本轮未建；禁止覆盖/链接输出 |
| 权重 | last/initial_state等仅授权服务器目录，不入Git | 必须 |
| 监督 | parent累计GPU分配秒/CPU core秒/bytes；每run进程组deadline/RSS/VRAM，建议1s采样，采样失败fail closed | 未接线认证，**阻塞A3** |

/gly快照97%使用率见原始df；可用空间不等于配额/预留/未来可用证明。批准前仍需核对quota/inode/可写性及批次总空间。

现有meta65_joint_supervisor接口不能直接代表122授权：其formal默认4800s/RSS4GiB/VRAM8GiB/output1GiB属于旧任务，还有旧endpoint验收逻辑。run_fomaml_joint_source只支持112三臂，price_meta_p0仅prepare/smoke；无已认证122父级训练+评估+判定封存路径。validate_training_preflight检查p0门，**未检查formal_training_authorized**；不能只改p0字段就调用train。正式入口还须绑定单独用户授权/完整budget，在输出/模型操作前拒绝未授权。这里只登记，不改实现或状态。

未来计时预检也须单独授权；CPU smoke必须 **≤2 updates／300s／单线程／4GiB，标smoke-not-research**。不搬用112的200-step授权。若不足以给可信预算，继续写未定，不能自动扩步。

### 6.4 失败停止与归档

- 非有限loss/内外梯度、错标签、目标读取、来源SHA变化、unsupported/静默排除、跨episode污染：立即停止阶段，保留stdout/stderr/退出码/首失败原因/资源记录，不自动重试。
- parent总GPU/CPU小时、wall、RSS/VRAM或bytes/空间超限，监管采样失败：停止自有进程组并留痕，不杀他人、不扩大预算、不删query继续。
- 同命令第三次失败前停止汇报，不换seed、删库/查询、调rank/容量/训练量或选best求通过。
- 每run归档协议/代码/配置/cache/plan SHA、完整manifest、gradient audit/counts、资源轨迹、last SHA和失败原因；source eval存query ID/hash/true/pred和seed/库/K/run汇总。大文件留服务器，Git只精简证据。
- A失败停B/C/D；即使A过，B固定lite；目标评估另批，不能按目标结果回调。本轮无上述阶段执行。

## 7. 总门与阻塞清单

| ID | 类型 | 状态 | 关闭所需证据/动作（不是授权） |
|---|---|---|---|
| P1 | P0必过缺证 | 仍缺证据 | 分别补有限loss但非有限内梯度、以及非有限外梯度的受控注入拒绝留痕；现有NaN loss测试不能替代。涉及模型更新先明确授权并受限监管。 |
| P2 | 执行合规 | **失败** | 子代理越界事实留痕，不能追认为合规P0；后续模型测试与只读审计分开授权监管。 |
| A1 | 实际data配置 | **失败** | 模板cache_root不存在，须冻结实际路径/SHA并验证，不改旧证据伪装就绪。 |
| A2 | 成本/预算 | 仍缺证据 | 四臂真实源计时/资源/bytes依据、总GPU/CPU小时/parent上限及用户批准。 |
| A3 | 监督/授权/全矩阵归档 | 仍缺证据 | 独立授权与完整budget门禁，12run＋396单元含K>0封存认证。 |
| R1 | A0-ref复用 | **失败/不可复用** | 具名三seed不复用，按12新run预算；别的候选重新核验。 |
| L1 | 取证限制 | 保留 | 上轮单条chembl stdout缺失/部署无Git；现有全量SHA不能创造过去stdout，未重读原PRICE特征统计。 |

**P0总门未通过，不建议现在申请正式A启动。** 先补必过缺证、解决配置/监督/预算阻塞并独立审查，才可提交新的授权申请；本轮不自动补证smoke、不实施A。

## 8. 执行偏差、验证与最终审查

主代理委派两路只读审计。技术一路未执行模型；历史复用一路违反明确不执行模型/测试约束：做CPU模块构造、两次完整unittest，第二次输出1784 tests/179.602s/OK(skipped16)。命令timeout590、管道未pipefail，不能冒称unittest进程exit0；无2-update/300s/4GiB supervisor、RSS或全段update留痕。主代理核对后不再运行模型测试，不拿越界执行补P0或预算。

该代理还查GPU状态（只查未分配）及列既有目标结果run文件名；未观察到结果payload或真实目标缓存读取，不使用任何目标数值。没有file-access trace，不能证明全部副作用为零；执行合规失败单列P2。未观察到正式训练、GPU计算、PRICE原目录修改或授权布尔值修改。

生产代码/测试/协议不变，脚本只以审计文本归档。主代理真实：源核验exit0、历史复用exit0、最终合成NumPy exit0；首次人工合成断言exit1保留；Ruff exit0。提交前完成JSON/SHA/链接/diff检查，且由未参与撰写的新独立只读代理审查最终差异；前置审计不替代最终审查。

最终独立审查结果单列于 `122-p0-closure-review-20261111.md`。仅在该最终审查无未解决交付阻塞时提交/推送独立分支；审查通过只指本轮审计交付，不改变P0总门与训练权限。
