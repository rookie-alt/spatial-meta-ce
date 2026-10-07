# 31 S8 full-P 秩稳定性：规格与成本审查

日期：2026-10-07（Asia/Shanghai）。接手基准 `8d39dea`，沿用 `feat/s8-numerical-certificate-20261007`。
范围：纯本地公式／源码／合成审查，不连接项目服务器，不读真实P/目标库，不部署、训练、GPU或smoke。30号及全部历史failed证据保持原字节。本页是候选规格审查，不替代30号冻结生产规则。

## 1. 当前阻断究竟是什么

令 rho=1e-12、eps=2^-52、d=max(m,n)、gamma=d eps/(1-d eps)、eta=gamma s1。代码 `_spectrum_checks` 对每个谱自身s1建cutoff区间，对每个sigma建±eta区间；丢弃要求 `sigma+eta < rho*(s1-eta)`（s1>eta时）。

即使sigma精确为0，仍需 `gamma < rho*(1-gamma)`，等价 `d < rho/(eps*(1+2*rho))`。4503可满足、4504起不满足；d=19456的gamma约4.3201e-12，约rho的4.32倍。非零矩阵中任何被该谱自身cutoff丢弃的方向都不可能稳定丢弃；独立SVD这一条件失败已足以阻断合取。整张实际A全零仍适用旧rank0特判，不代表任意零列都可豁免。

全秩非零矩阵不是必然失败：若最小比值 `sigma_min/s1 > rho+(1+rho)*gamma` 且三谱、backend、预测和其他gate全通过，仍可认证。它不是本次未知真实矩阵的已证性质。

- **认证规则保守性（已证）**：用全局维度包络将精确零尾谱也纳入不可分辨区间；可在已知精确秩的高瘦合成矩阵上复现。unknown不等于solver错误。
- **真实数值问题（未证）**：若真实谱靠近cutoff、投影/目标对扰动敏感或backend检查失败，则可能确有数值困难；未读真实full-P，不能诊断。
- **证据不足（当前事实）**：没有本次真实full-P的三谱、边界间隔、残差/Gram的验证性上界、峰值RSS与分阶段时间。旧1216行诊断不能外推19456行谱，普通FP64 residual很小也不是严格误差证明。

这里的rank是**存储FP64设计矩阵在固定cutoff下的数值秩**。不认证真实精确代数秩、原SQL选择率、全线性类最优或空间漂移收益。

## 2. 少量候选，不为旧结果反推阈值

### A. 保持 v1.1，保留 unknown（现在可执行的决策）

对象：当前工程证书及其边界；不新增通过出口。假设和误差口径仍30号。任何旧gate失败或临界区间重叠均停止。可先做纯维度必要条件检查，成本O(1)，但它不能替代真实矩阵认证，且本轮不改runner。

不改科学设定或协议，不获得真实G1通过。局限是不能解锁含丢弃方向的full-P。若无可靠改进方案，停在这里，不无限调常数。

### B. 验证性后验残差／正交性界（优先研究，但尚未批准替换）

对象：同一存储FP64 A的奇异值区间与固定cutoff数值秩稳定性，而非同solver输出相近。保留子空间／预测还须单独验证，不能由奇异值区间自动外推。

候选数学构造：对economy U,S,V，若有**可信向外舍入**的上界
`r >= ||A-USV^T||2`、`eU >= ||U^T U-I||2`、`eV >= ||V^T V-I||2`，且eU,eV<1，令
`delta(e)=e/(1+sqrt(1-e))`，
`beta=r+smax*(delta(eU)+delta(eV)+delta(eU)*delta(eV))`。

推导：将U,V极分解校正为列正交Q_U,Q_V；校正的 `Q_U S Q_V^T` 有同一S。对展开式用谱范数乘积及三角不等式，得到A距该精确正交分解的上界beta；再用奇异值扰动界定位真奇异值。该代数结论是**条件性的数学界**；只有输入上界和计算链被验证后才成为计算机证书。Fro上界可替代谱范数上界，但可能过宽。

不能直接把30号普通FP64测得的residual/Gram代进此式当证明；dot/GEMM、求范数、sqrt与最终beta都需要可靠的舍入误差处理，包含下溢/溢出、IEEE/BLAS行为与向外舍入。数学上beta可由数据决定，不等于任选更小atol。

cutoff仍rho不变；每个算法各自数值cutoff与可信真实谱区间的对应需单独定义，不能将近边界值改判、静默忽略尾谱或省略参考预测/目标/normal/保留投影。beta非finite、e>=1、无法验证算术、边界触碰或两个参考方向不一致仍unknown。即使奇异值区间闭合，保留子空间还需cutoff处分离gap、子空间误差与预测误差的单独界；普通投影比较只保留原工程含义。现阶段没有可信上界计算实现或本地全尺寸成本证据，**不能进入生产TDD替换**。

### C. 独立高精度／不同算法参考（成本更高的后备）

对象仍是同A及同rho，原LS不换solver、不做cutoff搜索；仅加一个独立参考。提高精度或Jacobi可减少部分数值误差，但“两个算法一致”不自动成为严格界。必须固定精度/迭代/硬预算与可信误差证书；不收敛、证书不闭合或gap不足仍unknown，不无限升精度。

一次额外完整分解使标准pair至少4次分解；CPU常数与内存取决于实现/精度，本轮不安装工具，尚未选定/认证验证性高精度实现，不给服务器ETA。候选精度只是设计，不根据旧P挑选。`longdouble`存在不意味着当前NumPy linalg支持相应高精度求解。

QR/TSQR压缩可作为B/C的成本优化，但必须把QR残差和Q正交性计入回传A的总误差；不能把d从19456偷换成775而遗漏压缩误差。不得直接用A^T A做秩判定：它将奇异值比平方，固定1e-12边界变成1e-24量级，需要更强算术证据。

## 3. 初步成本模型（后续待独立核对）

当前m=19456、h设计n=65、x设计n=775（协议尺寸，不是本轮读取实际P）；k=min(m,n)。每份FP64 A或economy U占8mk字节，每份Vh/Gram为8k^2字节。标准pair2次LS+1次economy SVD，认证本身2次分解；不能把次数当完整成本。

显式backend检查另有三次大乘法：U^T U约2mk^2、Vh Vh^T约2k^3、重构约2mk^2，合计**4mk^2+2k^3标量flop模型**；不是BLAS真实指令数或总分解flop。还有norm、拷贝、比较及多次预测/投影GEMV。

现有 `_array64` 必拷贝A/U/Vh，重构还会生成多个mk临时数组；分解workspace、标准化临时数组、Python/Torch/model、allocator/cache同时驻留情况未核峰值。矩阵字节或粗略6–8份mk数组账单不是峰值RSS上界，不据此宣称12GiB预算足够。

B若复用U/S/V不必增分解，但可信残差/Gram验证至少涉及同级大乘法及舍入包络，验证实现可能多遍或高精度；时间/内存倍率未定。若使用QR压缩，至少加1次QR及误差回传工作。C加至少1次高精度完整分解，运算O(mk^2+k^3)，常数和字节数必须按候选实现另测。

本地仅允许有界小尺寸合成时间探针：单线程、最长60秒，最多2048x65和19456x2；不测19456x775或读取真实数据。报告原始wall/RSS/backend和输入尺寸，不将小矩阵时间比例外推为server/full-P ETA。

## 4. 资料依据与禁止的跳步

LAPACK Users' Guide的SVD误差解释包含 `p(m,n)*eps*s1`，p为维度相关函数；示例代码取p=1是近似示例，不能拿来替换当前包络并宣称严格证明。普通dense降双对角的误差不能由双对角阶段相对精度保证省略；Jacobi只是在某些情形更准确，不保证本项目必过。NumPy1.26说明economy SVD形状及gesdd实现。原始官方来源与查阅日期另存sources.json；本轮不安装新工具。

## 5. 初稿审查流程与停止条件（历史阶段）

本草稿先交独立只读代理审规格和成本；不改生产s8_probe、run_s8_smoke、科学gate或实验协议。A是当前保持unknown的安全出口；B/C是待实现/待预算的候选，不预填审定。

若独立审查不能确认可靠且成本可接受的计算证书，则本轮停止在unknown，列出最小缺失证据及用户决策。真实P核验、smoke重试、正式实验分别等待明确授权。本地历史617通过不计作本轮实测，不计作G1通过。


## 6. 本轮已完成的有界合成复核及独立成本核对

[合成探针源码](../reports/s8-full-p-rank-cost-20261007/analysis_probe.py)只调用旧生产LS/认证，不调用Adam/optimizer、runner或训练；它是复核分析脚本，不是新版生产证书，不据此声称TDD实现B/C。父进程60秒timeout，实际backend threadpool全部1线程。输入限n65的m<=2048与n2的m<=19456，未生成19456x775矩阵。完整[结果](../reports/s8-full-p-rank-cost-20261007/synthetic-probe.json)exit0。

存储entries精确0/1、代数rank1的4503x2返回stable/pass；4504x2和19456x2返回unknown/False，明确复现保守性而非真实solver故障。512/1024/2048x65稠密DCT合成例只用于本地小尺寸计时，3次分解/pair与当前源码一致；不能外推真实full-P时间或认证状态。

探针数值阶段总wall（不含导入）0.120729秒；整个进程（含NumPy/Torch导入及所有例）的累计maxRSS约191.74MiB，不是full-P峰值或各例独立峰值。输入shape/backend/阶段原时间留JSON，不做服务器ETA。本轮新增分析脚本Ruff通过。

独立只读成本审查确认：标准full-P pair三次分解、三次backend-check GEMM、12次A GEMV和2次U保留投影GEMV。h/x的单份A分别9.6484/115.0391MiB，单份Gram0.0322/4.5824MiB；认证三GEMM分别0.329356/47.674009Gflop。只计重构阶段显式数组约6D+3S至7D+3S，h约57.99–67.64MiB、x约703.98–819.02MiB，但这不是峰值上下界，未计workspace/标准化/模型/allocator。

成本审查另指出：full-P计时开始在标准化之后，`matrix_bytes`仅一份A；记录的run maxRSS与RLIMIT_AS限制（虚拟地址空间）不是同一口径。旧failed在h处先停，不保证x资源阶段执行。两小面板及两full-P pair都成功才合计8LS/4SVD/12认证GEMM；不是本轮实测，也不是完整F0/Adam/抽取/端点总成本。详见[独立成本日志](../reports/s8-full-p-rank-cost-20261007/independent-cost-review.log)。

极小[longdouble能力测试](../reports/s8-full-p-rank-cost-20261007/dtype-capability.json)显示本地NumPy1.26.4的longdouble SVD抛TypeError；不自动转换成FP64“通过”，不外推服务器。高精度方案不能只改dtype。


候选C并不能在保持v1.1原gamma区间时自行消除结构性unknown：更准确的参考输出只是证据，若没有另经审定的可信误差界及新版本判据，原生产判据仍阻断。普通FP64验证计算的gamma_m项还可能让B的上界过宽；误差验证的算术及精度必须先审，不能先假设beta更小。


## 7. 最终独立审查：停在 A，不替换生产认证

独立只读规格代理批准A的保守停止、4504/19456规则内阻断、B极分解beta的**条件性数学界**及C/QR/Gram的谨慎边界；**不批准B/C实现、生产新通过出口或TDD替换**。成本代理已独立核实际调用和显式数组账单，未认证真实server资源。[规格日志](../reports/s8-full-p-rank-cost-20261007/independent-spec-review.log)、[成本日志](../reports/s8-full-p-rank-cost-20261007/independent-cost-review.log)。代理是独立上下文AI，不是外部科学第三方认证。

B前提补记：U/V必须完整economy形状、S完整非负有序；r/eU/eV必须是未归一范数的可信向外上界。30号 `Gram_F/sqrt(k)` 均方根不是谱范数上界，不能直接代入eU/eV；最后一次nextafter不能弥补此前dot/GEMM/缩放/减法/范数/sqrt计算链的误差。环境、累加/FTZ/下溢/溢出处理未冻结。

对一种通用length-m点积验证路径，即使观测Gram误差0，上界预算仍可能含gamma_m。理想化r=eV=0时，零尾区分仍需 `eU < 2*t-t^2`，t=rho/(1+rho)，阈值约2e-12；m19456时按u=2^-53计算的gamma_m约2.16005e-12，也可能过宽。[精确有理数标量记录](../reports/s8-full-p-rank-cost-20261007/conditional-gram-budget.json)。这只是该验证包络的局限，不是实际Gram误差下界，不排除补偿/更强算术或结构方法，不能预先保证B解锁。

新接受路径仍需定义：各算法自身可信谱与cutoff（单SVD beta不能无依据共用给LS）；严格keep/discard/触边unknown；极分解校正、gap、投影运算与target尺度的误差链。只追加B/C旁证而保留旧gamma合取时同样不能解除unknown。所有科学参数与原门槛本轮不动。

**本轮决策**：没有已闭合且成本已认证的替代规格，保持unknown；新生产规格未审定、新certifier未实现、无新生产TDD。6项本地合成复核和Ruff只是本轮分析验证；617全库通过仍为历史记录。本轮不再做更大矩阵、改常数或升精度重试。

### 最小缺失证据与下一授权

无需运行真实P来确认已经证明的规则性阻断。若希望继续，最小下一授权仅为：**有界本地可信误差算术与接受契约可行性研究**；先固定预算，不实施生产B/C、不安装工具、不联网/连接server、不读取真实P、不benchmark、不训练/GPU/smoke。交付全链验证算法/环境假设、原rho下预算是否可闭合（允许仍unknown）、三谱cutoff及子空间/预测证明范围，经独立审定后另行讨论生产TDD。真实P核验、真实smoke重试、正式实验继续各自待授权。

### 验证与同步口径

生产库/runner/tests、30号和旧报告原字节不变；65历史failed证据SHA检查一致；本轮相对Markdown链接、分析脚本Ruff、合成输出语义及预算复核已检查。没有真实full-P认证，G1仍failed、formal_ready=false。Git同步见99号第39节与本轮最终记录，不预填未发生的提交/推送。
