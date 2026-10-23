# G1运行器集成独立审查记录

日期2026-10-07。只读AI独立上下文，非科学第三方认证。state实现代理完成自己的标准库模块后承担单次只读runner/checker集成审查；未改主runner、未联网或访问server/GPU、未运行完整suite、未执行真实smoke。

## 首轮发现与核心路径

审查限定实际P-only/source边界、签名/checkpoint、V-seal-beforeH、独立分母与资源路径，不把事件日志当实际I/O独立证明。

- actual P-only：从seed42新模型起点，原权重/resume禁止，seed43排序raw P候选采样，2更新/8backward；store.batch收到positions只P。
- checkpoint核固定SHA后trusted新pickleload，模型配置/来源/P签名、step2、完整optimizerstates/moments、FP32权重和tensor指纹检查；重载forward桥接固定T身份。
- seal期望SHA从返回payloadcanonical字节锁定，不事后读文件重算expected；每H实际semantic load入口先require_nomination。smoke明确不selection，formal4000在本runner拒绝。
- independent重算P19456/V768/H128身份及52桶；complete-only原则，无Q/完整研究V/H结果声明。fullP资源LS系数不会进入预测solvers。
- 实测runner7项/independent23项通过，无新science结果。

阻断：默认SIGALRM直接终止绕过failed状态。真实临时子进程复现returncode -14、run目录有但无status。主代理先负例后加入显式TimeoutError handler/finallycancel。另一启动门槛：Torch线程4不保证NumPy BLAS4；需Pythonimport前env固定并实际线程池核验。

## timeout/线程修订快速复审

agent核最终timeouthandler注册于alarm之前、TimeoutError可进failed写入、finally清alarm。读取实际SIGALRM子进程9项runner/3.221秒通过日志；未再重跑。检查runtime必须发现BLAS、各线程池1–4，缺诊断不静默跳过，summary保存实际后端数据。Pythonhandler在nativeBLAS可能延迟，外层OS timeout/exitcode仍必要，不以handler代替硬停止。

之后server启动前检查发现冻结venv未装threadpoolctl，尚未创建执行包/run或训练。主代理不安装或改PRICE环境，另SSH只读核实加载NumPy OpenBLAS64/Torch libgomp能分别直接给4线程计数，添加有界Linux C-counter路径及对应合成拒绝测试。未知后端/库来源/计数接口失败即blocked，真实summary仍要保存实际计数。此兼容补丁基于实际运行时来源核验，不声称agent曾重新访问服务器或认证新runtime。

已知范围：checker核event顺序不是实际读取顺序；_source读取/入口gate测试是另门槛。资源profile真实尺寸尚待主代理运行；正式fresh4000/完整Q/V/H及研究candidate尚未执行。该审查允许有预算的一次CPU工程smoke，不授予正式GPU或新特征/loss搜索。


## 真实失败诊断只读复核

另一个只读AI代理基于本地原证据确认失败因果链：source认证→F0及reload→P表示→首h certify_pair metadata拒绝→failed/exit1。核六原文件SHA/十四执行源码、contract protocol/source binding/panel/checkpoint跨记录一致。normal/objective/referenceJ/预测均通过，rtol1e-12/atol0对近零singular逐项比较拒绝是实际字段原因；不能推为solverJ、原标签或SQL语义独立正确性。x是后置矩阵诊断非原run已执行。

zeroStd为P面板常量维，不含截距；h秩亏19中18constant，x秩亏189中177constant，另1/12可能依赖或near-cutoff，当前未知，不能称模型失效维。无seal/Hsemantic事件与失败位置支持未进H语义，但预检有H身份/count/SHA字节读取，不能说所有H数据都未读。partial-stage只认证采样和checkpoint字节/原reload记录；optimizer_steps=2在其代码直接赋值，不能宣称它重验optimizerstate。主代理已另存partial-stage-annotation.json并修文档，不修改原proof/status。

复核认为失败停止收尾正确，仍G1未通过；下一任务应事前冻结数值尺度/FP64epsilon/维度的谱误差规则与cutoffboundary稳定性，保留近界unknown、projection/预测/目标和伪造谱负例，不忽略尾谱或追认成功。此代理未联网、未server、未重跑fulltests、未修改文件或训练。
