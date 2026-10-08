# S8 G1 A 握手最小修复：本地与Linux合成验收记录

日期2026-10-08，worktree `/private/tmp/spatial-meta-ce-s7-diagnosis-20261007`，分支feat/s8-g1-engineering-v2-20261007；接手HEAD72eb087、干净。旧36包/contract、37失败proof、旧server目录均保留，本轮不真实P重试/B/训练/推送/merge。

## 根因与修复范围

父argv使用相对contract，child接收绝对contract，旧握手逐字比较拒绝同文件。现在读取真实父cwd并验证父exe/入口/execute身份，再以该cwd规范化contract/output，严格对冻结路径/SHA/pipe/ticket比较。child合同另核固定ROOT/CONTRACT_RELATIVE；不靠env，不删除握手，不按basename接受，不以child cwd解析。路径symlink拒绝/安全读取保留。

新code/proof仅预留`/gly/weilin/spatial-meta-ce/runs/s8-g1-readonly-p-{code,proof}-handshake-v2-20261008`，新contract存本报告目录；原输入旧P路径和八白名单不变。这些是下一次真实P待授权路径，本轮不在其中部署或执行。生产改动仅核验器，未改s8_probe/LS/rcond/数值阈值/34科学协议/S2P1。

## TDD与本地验证

RED检查点7c89b0b：12个mock测试中两合法相对路径正例在旧代码抛Supervisor绑定错误；绝对调用/拒绝负例绿。初次夹具readlink mock误拒解释器symlink单独保存，不冒充生产RED。最小修复后12 GREEN/0.033秒；完整既有只读专项66项/4.361秒/exit0。

新增Linux专项13方法/16握手场景，在macOS全部skip，不能称Linux通过。当前必要完整本地回归753项/96.688秒/exit0：740实际执行、13 Linux skip。5份改动Python Ruff通过。源码/test/log SHA见local-verification-pre-linux.json；若后续审查要求修改，本快照明确历史化，不能沿用为最终证据。

## Linux集成夹具的边界

真实/proc/cmdline/cwd/继承pipe/ticket和自有parent→child监督；原生产源码字节不变复制，ROOT自然来自副本__file__，没有mock/patch函数、常量、proc或返回值。测试父入口是独立临时夹具，不调用production main/worker/dataCLI或任何数值solver；最小synthetic contract只供helper，不能冒充通过真实manifest校验。副本SHA前后核对，numeric imports空。详见linux-handshake-test-design.md。

**这验证真实Linux helper进程握手，不验证完整生产数据CLI、source manifest guard、runtime backend/数值阶段，也不认证真实P/G1。** 本轮用户只授权一次Linux合成suite，1CPU/emptyCUDA/outer总60秒/failfast，新临时目录，失败保存原始日志后停止，不第二次调用。测试实际执行前需本地专项/回归/Ruff和服务器前审查无阻断；最终独立批准还须实际Linux原日志与SHA。

## 当前状态与冻结

本文件建立时服务器测试未执行，独立审查待完成；冻结生成器已编写但新真实P包未冻结。验收通过后生成新包/contract/SHA，保留旧包不覆盖；新真实P执行仍必须另获“新版A单次真实P只读核验”授权。原G1failed/formal_readyfalse/strictunknown始终保持，helper通过不等于P核验通过。

## 服务器前独立审查与envelope中断补强

独立代理实际核source/test/payload、本地66/753日志与Ruff，生产路径修复/helper夹具无新增阻断且无后门；但指出outer timeout不会覆盖start_new_session fixture组，suite默认TERM退出可能跳过finally。主代理先保存1项RED（缺TERM桥接），仅给envelope注册TERM/SIGINT→InterruptedError，unittest记录错误/failfast并展开fixture原有kill/reap finally。实际本地POSIX中断测试0.136秒通过：AST读取真实handler、SIGTERM当前driver、cleanup杀自有group并reap。它不是Linux系统测试，不冒充服务器验收。新增后67/754快照的本地验证正在完成；旧66/753为中途证据。生产helper与Linux测试字节没有改变。服务器仍未执行。

## 本轮最终验收与独立审批

最终67项3.991秒exit0；754项95.377秒exit0，741实际执行/13macskip；Ruff通过。一次真实Linux13项2.234秒/0skip/0fail/0error、outer/SSH0，suite2.2435秒，affinity[0]/emptyCUDA。前后source/testSHA同值，没有server重复调用。外层原日志不包含逐场景inode明细，仅原测试名/metadata/result，不能宣称raw syscall trace；source绑定断言支持16场景。原生TERM测试与Linuxhelper测试的口径分开。

Dalton最终只读审实际diff、local/Linux原日志、源/test/9包成员/contractSHA，APPROVED、阻断0，详见independent-final-review.md；无独立数值复算。新版真P包未server部署或运行，不读H/目标/旧checkpoint/训练状态，不跑smoke/F0/Adam/GPU。旧88文件与36包/contract37失败证据不变，原G1failed/formal_readyfalse/strictunknown保持。冻结新包后停止，等待另行新版A单次授权。

自检Accuracy5（实际日志/SHA/独立复审一致），Completeness4（helper集成不含完整dataCLI，明确未授权未执行），Clarity4（区分mock/macskip/Linux），Actionability4（新包/命令已冻结，但trueP仍待授权），Conciseness4（多数体积是原始测试/日志），平均4.2/5。不以自检替代工程/科学准入。


原始Linux stderr含SSH的CRLF警告，原全库Ruff诊断含历史空白行；全量git diff --check因这些raw证据提示空白，不修饰原始字节。代码/测试/JSON/Markdown/属性文件的限定diff检查单独执行；所有暂存rawlog与记录SHA完全一致。
