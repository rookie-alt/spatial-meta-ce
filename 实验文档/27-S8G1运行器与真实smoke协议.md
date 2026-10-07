# 27 S8 G1：P-only运行器、封存顺序与真实 smoke 协议

冻结日期2026-10-07；基础提交04561e8；分支feat/s8-g1-smoke-runner-20261007。本轮用户要求按计划执行下一步。本协议只执行G1 engineering smoke及资源profile，**不是S8 G2–G4正式研究训练**。24号旧设计与26号数值/访问修订不改写。

## 1. 固定工程范围与状态

- 19库T、4库H、3库V仍由固定身份规则决定，目标imdb/stats/genome/ergastf1绝不读取。旧PRICE目录、权重/数据及历史runs只读。
- 仅CPU最多四线程，launcher在Python导入前固定OPENBLAS_NUM_THREADS/MKL_NUM_THREADS/OMP_NUM_THREADS=4，runner核实际BLAS/OpenMP线程不超过4；若冻结venv无threadpoolctl，则只接受本轮已只读实测的Linux已加载NumPy OpenBLAS64/Torch libgomp直接C计数接口，记录库路径/接口/计数，未知后端失败，不安装/改动PRICE环境；显式CUDA_VISIBLE_DEVICES为空，无GPU任务。表示/端点batch128；原F0前向FP32，矩阵/读出CPU FP64。
- 新F0从seed42重新初始化，sum64/PG input+anchor1；不从旧checkpoint/resume，不读取含H的旧权重。Adam lr3e-4、clip1、SmoothL1、meta_batch4、K5/10/20/50/100均匀、query64、RNG seed43。
- **F0固定2次更新、8个episode/backward**；每episode只从当库1024P身份内无放回抽support/query，ERM合并监督。训练候选按照原cache首次raw行顺序排序后抽样，身份固定P仍保留SHA排序；两者顺序不同须独立复核。
- 探针拟合每库P哈希顺序前64，共1216条，身份选择不看标签或误差。两个表示h64/x774，各Adam2更新及同cutoff LS；这些都是smoke engineering状态，不用于候选或精度结论。
- 完整P19456条在新F0上提取一次表示，用其完整矩阵做**资源only固定cutoff LS/数值认证**；此路径系数不参与预测、不选择模型、不扩大smoke为正式19456训练。每路径含solve_ls+certify参考重解/独立SVD共3次线性代数调用，计入成本。
- 当前保守full_linear_optimality_verified=False，solver pass不推为科学最优或机制定位。无超参数/rcond/loss网格、扩seed或自动重试。

## 2. source与checkpoint认证

开跑前用冻结S6 reference原字节SHA和26×7源文件的before/after SHA认证；T/V全tokenfinite/shape/fanout与重复身份内容核对；H前置只解码identity/count/hash、inspection/provenance内容opaque，只流式读取原字节作来源SHA。沿用O_NOFOLLOW逐目录FD及文件安全读。

P/Q完整身份由s8_contract生成，P每T1024、Q其余。实际store.batch收到positions必须都在P，记录episodes完整sourcehash/support/querypositions；独立复算器从原源数组重算固定P与8episode采样，不能只依赖runner“用过P”的总结。

checkpoint只新目录last.pt，保存bindings signature：schema/mode/purpose、27协议SHA、source binding SHA、P身份digest、T/H/V名单、模型/seed/2步预算、资源字段、实际counts/sample digest。独立reload先核外部固定文件SHA再trusted本次pickleload，拒不同source/P/mode/steps/非FP32/参数/optimizerstates缺失；所有optimizerstep必须2、moment shape/dtype正确、模型tensor指纹匹配。新checkpoint重组模型forward在固定T面板128身份比较raw delta<=1e-4。相同运行器的另一次reload不是第三方科学认证，属于严格工程重载门槛。

## 3. V seal与H访问顺序

本工程smoke不从V指标选择候选。seal固定purpose smoke、decision=smoke_not_selection、candidate=None，**不是no_development_candidate也不是S6 no_candidate的新一次判定**。

seal文件canonical body，O_EXCL一次写/fsync、目录和文件O_NOFOLLOW禁止别名及覆盖。seal创建后从生成payloadcanonical字节计算expected_file_sha并外部锁定，再require_nomination校file/body/signature，不能H前从潜在篡改文件重新算“expected”。

每个H实际语义加载入口均先require_nomination；未封存、改source/mode、篡改文件、未知db/目标均fail closed。完整过程record events，独立复算器核seal fileSHA与各H事件在seal后，但**记录事件顺序不等同独立证明实际I/O顺序**；该约束另由唯一SourceStore入口和未封存负例保障。本轮H标签/token内容认证、smoke前向在smoke seal之后，不能前置到预提名阶段。

V每库固定SHA(`s8-bridge-v1|db|query_hash`)前256；H每库固定SHA(`s8-smoke-h-v1|db|query_hash`)前32，仅工程测试。三个V768、四H128、Pfull19456，共20352端点，不冒充完整V/H或Q结果。正式需要完整Q/V/H另行阶段实现/冻结。

## 4. 端点与独立核验

每端点保存database/cohort/query_hash/FP64true/PG/n_tables、五函数raw_log_pred（四probe及F0），不做新增预测裁剪或PGfallback。所有稀疏行也检查finite、standard Q exponent/rawsquare overflow；不足200的群体只输出n/status，不报告详细精度。标准Q floor不变；辅助指标从各群体完整smoke分母算。

独立verify_smoke不导入生产metrics/probe/train/state：重算P固定hash身份、8episode真实采样、20352身份/配对source字段、全部52 all/4+桶、五函数Q50/Q95/rawMAE/MSE、sealSHA与event序；用独立排序线性插值。任何缺失/重复/错配/非法db/非finite/overflow直接failed不drop。该复算器不重新生成tokens/模型预测，也不证明线性最优、实际solverfit或pickle安全；来源认证/数值solver/actualcall/reload由分立门槛覆盖。

## 5. 资源预算、执行包与停止

- 一次真实smoke，墙钟1800秒（30分钟）硬截止覆盖source认证、F0、P表示、四探针、小矩阵/完整P资源profile、端点和独立复算；Linux地址空间12GiB。这个上限不是预计一定完成时间。
- 新隔离执行包在项目runs/s8-g1-code-20261007路径，文件内容hash与已测试本地版本逐项核对，禁止覆盖旧run或现有项目源码。输出只能新建项目runs/s8-g1-smoke-*直接子目录，绝不写PRICE或随意data_root。
- 运行器默认只打印planned；--execute且CPU env明确才执行。未提供formal4000入口、GPU参数或resume。本轮明确不执行G2–G4。
- 非finite/overflow/数值认证/输入SHA/身份/封存/时限/RSS错误停止，留status failed与已运行证据；不改LR/batch/matrixcutoff/分母凑成功，失败不自动重试。
- 仅清理自有进程。exec运行用OS timeout覆盖；SIGALRM显式转TimeoutError，Python异常/可交付超时有失败状态，SIGKILL/OOM可能不写finally，父监控需保留exitcode而不能误判running/completed。真实子进程会在当前会话监督等待，不自动脱离为长期任务。

## 6. 本轮可观察完成条件与交付

工程通过必须同时有：真实2更新/8episode P-only认证、freshPG起点及初始化指纹、checkpoint/reload桥接、两表示probe有限数值证书、小矩阵与fullP资源计时/字节/峰值RSS、nomination-beforeH入口及记录、20352端点/52桶独立复算、输入和执行源码前后SHA、完成status/summary匹配。

只归档精简contract/preflight summary/verification/probe profile/events摘要/status/源码部署manifest及日志，checkpoint/P矩阵/全端点留服务器，不提交Git。本地报告区保留TDDred/green、失败、审查及实际结果，不graphify。

成功只写completed_smoke / engineering_verified、formal_ready=False，不产生research candidate或改S6 no_candidate。资源测量可帮助下一预算，但2步F0/32H smoke吞吐和完整P求解成本不能直接推出完整1279218查询六小时一定完成；正式成本仍需说明估计假设/瓶颈/安全余量。若门槛全部通过，下一阶段为固定4000F0与完整Q/V/H正式runner及独立认证的审查/预算启动，不自动启动。
