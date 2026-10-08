# S8 G1 v2 A：只读 P 核验本地准备（2026-10-07）

## 当前授权与基线

worktree `/private/tmp/spatial-meta-ce-s7-diagnosis-20261007`，分支 `feat/s8-g1-engineering-v2-20261007`，接手 HEAD `41ccc17a6d6adcc0d70e58ffbadd29353825eec8`，接手干净。本轮仅本地准备/TDD/冻结/独立审查和本地提交；未推送。未连接 myserver、未读真实 P/H/目标/旧 checkpoint、未部署、未 smoke、未正式训练。

指定委派记忆在此 worktree 缺失，主代理明确只读了主工作区最新版 `/Users/williamlin/Downloads/13/spatial-meta-ce/memory/spatial-meta-ce-agent-delegation.md`，没有回主仓库编辑。其余指令及交接/进度42–43/34/final-verification/review/28已读。674/90.850为历史基线，本轮结果另记。

## 最小入口与保持不变事项

新增 `scripts/verify_s8_readonly_p.py`：默认仅 plan；`--execute` 要求 Linux、真实冻结契约及命令行可信 SHA、显式 empty CUDA、新且唯一 output、父监督内部 worker。仅 import 当前已批准的 `spatial_meta_ce.s8_probe`；不 import smoke runner/state/model，不调用 fresh F0、Adam、checkpoint loader 或表示提取。生产 `s8_probe` 本轮未改，rcond/数值阈值/34协议未改，S2 P1未动。

每个 h/x 分别只重算：1216行（每T库旧 P 顺序前64）的 zero-reference LS 诊断；19456行 full-P resource-only LS 工程检查。各自仅从对应 P 面板拟合 FP64 population mean/std、zero/tiny规则与截距，完全采用已验证生产代码。normalizer派生 SHA 绑定当前数组与行选择；不读取历史 normalizer。只返回完整证书/时间/rank，不存 theta、不预测、不选择；计算完立即丢弃资源系数。

小面板结果只是 `diagnostic_not_original_Adam_probe_gate`，不是原两步Adam probe通过，也没有修改旧small pass gate。full-P以 engineering_verified为工程门，rank_stability单列stable/unknown，strict始终unknown、full_linear_optimality=False、formal_ready=False。任一来源或数值失败为failed，不用strict unknown掩盖。

## 身份认证的明确限制

四数组 expected SHA 来自旧 solver-diagnosis，四元数据 SHA 来自旧 controller且与本地归档逐字节一致；新contract可信SHA固定在待批命令。比较旧status=failed、schema1 contract、T库顺序、panel/source digests、19×1024 panels及唯一位置；只比较 preflight 清单的规范JSON digest，**不 dereference其中任何源路径**。旧清单含H/V，不沿用旧183文件全量rehash。

身份范围是 **historical_sha_bound_panel_attestation**。原证据没有四数组逐行query_hash映射；本轮禁止源库/query cache读取，不能独立重验哈希排序或每行SQL身份。历史normalizer也没有认证产物，因此本轮只证明新P-only派生定义一致，不追认历史normalizer。上述缺口已由独立方案代理识别；不能靠checkpoint/表示重生成/H扩读补齐。若未来审批要求独立逐行语义证据而非历史SHA绑定，应停止并另拟来源授权，不把A输出升格。

## I/O与失败留痕

实际未来保存P输入白名单恰为旧run下：`P-h.npy`、`P-x.npy`、`P-label.npy`、`P-pg_log.npy`、`status.json`、`contract.json`、`panels.json`、`preflight.json`。文件及祖先symlink、非普通文件、SHA/shape/dtype/finite/身份不一致拒绝；SHA与load消费同一字节缓冲区，无pickle。禁止额外路径、H/目标/checkpoint，即使只hash也拒绝。

worker的Python audit guard拦截非白名单open、旧输入写入/删除/rename、网络和进程创建；运行库读取仅允许已安装解释器/runtime目录及必要`/proc/self`。profile guard拒绝优化器/训练/加载入口。它们是对审查过的Python入口的防误用防线，**不是针对恶意native库的OS安全沙箱**；可信运行库仍需现场预算校验。source SHA在数值import前、读P前和最终核对。原backend预检的NumPy1.26.1、Torch2.1.0+cu121、OpenBLAS64/libgomp家族和venv来源被冻结，执行时不符即停止；只有历史版本/家族/来源绑定，缺少历史共享库字节SHA，不能声称库二进制逐字节相同。

输出为新proof目录（固定路径，拒绝既有路径/旧failed/PRICE/旧code）。数值证书立即写独立新文件，即使工程失败也留存。worker-proof区分not_run/failed/verified，未执行工程为null、rank为not_run、strict unknown不意味着执行过。父记录exitcode、signal、wall、采样RSS/VSZ，并合并最终proof；子进程没写proof、异常、资源超限不能成为成功。所有写入独占创建，不overwrite、不resume、不retry。

父SIGTERM/SIGINT中断会杀自有子进程组、reap并保存监督证据。父遭不可捕获SIGKILL、主机崩溃或磁盘不可写无法承诺保存新JSON；既有日志/契约仍保留，缺最终proof按未完成/失败处理，绝不能按成功补写。未执行旧last.pt/hash，不能报告该checkpoint本轮unchanged。

## 资源口径

单次attempt、4 CPU线程，父子共享Linux CPU affinity最多4逻辑CPU，运行时检查BLAS/OpenMP线程；empty CUDA。父wall上限900秒，parent与child各自RLIMIT_AS hard/soft均12 GiB（12884901888 bytes）。RSS超过12 GiB停止阈值按0.05秒采样，另记录worker Linux ru_maxrss；采样不是RSS硬上限，AS是虚拟地址空间，不等于RSS。matrix_bytes只指当前design payload；资源预算不是15分钟内完成保证。达到任何停止条件杀自有group，不放宽门槛或重试。

## 本地证据（最终状态由final-verification.json补齐）

- 首轮RED：31项、31失败、exit1，独立入口缺失。
- 第一实现测试含symlink别名及本地观测失败，原日志保留；未把失败改写成通过。
- audit删除拒绝补强RED：1项失败，exit1；修补后通过。
- supervisor SIGTERM补强RED：1项失败，exit1；修补后通过。
- parent预算与backend绑定各1项RED/exit1，随后GREEN。
- 专项43项通过、2.145秒、exit0，全部是synthetic/local fixtures。
- macOS与Linux AS计数不同，本地exit/signal/wall实际子进程测试mock资源计数；RSS/AS stop路径mock。Linux真实预算及full-P资源未执行/未认证。
- changed Python Ruff已通过；完整回归：首次715项87.279秒，沙箱OpenMP SHM导致原3测试失败；沙箱外715项93.705秒通过，但该快照随后新增预算/backend测试，仅作为中途证据，最终快照另记。最终SHA与独立实际diff批准待收尾核实。

## 待批准执行与后续边界

执行包在 `/private/tmp/s8-g1-readonly-p-package-20261007.tar.gz`，不提交二进制；清单及测试/源码SHA入Git。源码包只含只读入口、数值模块、包init、27/34协议、专项测试、contract和schema；不含smoke/train runner、模型、旧checkpoint或真实数组。

只有用户明确写出“批准执行A阶段真实P只读核验”才可连接/新目录部署/单次执行；具体命令见36号准备记录。A输出只能是新资源工程证明，不追认原G1 failed。B fresh F0单次新smoke另授权，正式G2–G4 runner/预算和训练也另授权。S8空间漂移研究目标不变，不继续Fraction、不将A工程证据写为科学结论。

## 独立BLOCK后的修补（等待最终独立复审）

第一次独立实际diff明确BLOCK4项，原记录保留于independent-review-blocked.md，不用旧43GREEN掩盖。首次contract/完整worker/输入输出目录替换/用途与normalizer完整性负例已补入；14项4FAIL为真实RED。proof-completeness-red首次因OMP SHM中断exit134，未称为有效RED；随后字段负例GREEN。

逐级O_DIRECTORY|O_NOFOLLOW目录FD与dir_fd末级打开覆盖输入、JSON与stdout/stderr创建竞态；固定manifest路径在首次读取之前检查；worker校新proof路径、Linux父cmdline执行参数和继承pipe OS归属，不再以可伪造env父PID作为授权；parent应用冻结schema的现有子集并核用途、shape、计时、fit_indices/normalizer SHA、完整证书记录及阶段文件一致。这是结构/旧工程门槛一致性，不重算科学证明。

guard下真实import及合成LS集成测试暴露torch/dill的/dev/null无持久写与tempfile路径查询。两个实际失败依次记录，最小修补仅允许/dev/null并先指定新proof为tempfile.tempdir以避免外部写探测；不放宽数据读取路径。该1项最终1.679秒通过（沙箱外避免OMP），属于本地NumPy1.26.4/Torch2.3.0测试，不冒充服务器1.26.1/2.1.0环境。Linux父握手/AS与实际full-P仍未执行。

### 失败证据分支补强

合法新output内，parent源码SHA拒绝也生成failed schema proof，child尚未启动的阶段仍not_run。数值阶段先写独占started JSON，若child遭SIGKILL且未写worker-proof，parent只恢复新proof目录中已有阶段记录，overall保持failed、输入最终认证False，不能追认成功；无记录且未开始阶段保持not_run。两个新增RED实测后51项GREEN/3.700秒。49/723前快照保留但不冒充本修补最终回归。

## 本轮最终代码与本地验证

最后独立P2：可解析partial stage不可直接认证verified。已实1测5个FAIL子例；抽取现有stage_record_complete供success/partial共同使用，缺字段、负尺寸、错误用途或normalizer/输入绑定会failed/record_unverified并引用原记录，不改原文件。额外1个RED确认LS异常已进入阶段不能标not_run；六行修补只标当前未出证书stage为failed/stage_started，不覆盖已有证书。

最终54项专项3.662秒/exit0；完整728项91.705秒/exit0，空CUDA/单线程/本地合成与夹具，沙箱外避免OpenMP SHM限制。改动Ruff通过；全库110历史问题与原日志字节一致。前中途715/717/723/726/727通过均不冒充最终snapshot。

同一G1 A只读代码审查代理已实核最终源码/test/contract/archive SHA与8成员，批准本地准备，剩余代码阻断0，见independent-final-approval.md；最终728日志再做独立绑定。66个保护文件不变。用户所需前置条件仍为真实Linux握手、版本/家族/线程/AS/RSS现场认证与真实P，尚未发生。自检准确性/完整性/清晰性/可操作性/简洁性各4/5：证据真实且范围清楚，执行命令冻结；主要限制是Linux真实执行未获授权、身份只历史attestation、包装代码较长但均对应访问/资源/状态负例，不扩大重构。

## 2026-10-08 本地收尾

本轮恢复HEAD41ccc17，未提交产物归属均是本任务。当前源码/测试/8包成员/contract/日志SHA与54专项+728最终回归snapshot全部一致，66保护文件不变。没有机械重跑测试，也没有继续扩展核验器。

同一Dalton只读代理完成728最终原始日志的独立读取/哈希绑定，APPROVED、代码阻断0，详见independent-final-log-binding.json。主代理没有把自身hash核对冒充独立批准。进度第44节与本地提交完成后才进入服务器。用户已明确授权A单次执行，授权另存../s8-g1-readonly-p-execution-20261008/authorization.json；原冻结contract中的历史not_authorized条目保持原字节，不是当前权限依据。未推送/不合并main，B与训练仍未授权。
