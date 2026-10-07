# S8 数值谱认证修订：最终工程报告

日期：2026-10-07（Asia/Shanghai）；基准7eb3370；分支feat/s8-numerical-certificate-20261007。
**本轮本地工程修订已独立复审批准；原G1真实smoke仍failed，formal_ready=false。**

## 实现与实际验证

- 事前独立规格审查、TDD最小修复完整谱数值比较，FP64 epsilon/维度/谱尺度工程包络，不忽略尾谱；每谱自身cutoff区间，触边/不一致unknown。
- 独立economy SVD重构、Gram均方根正交性、保留子空间预测；保留原objective/normal/参考预测、rcond1e-12及所有full_linear_optimality=False。不比较秩亏theta，不修改原F0/train/model。
- v1健康合成正例遭Gram门槛拒绝，留原日志后独立审定v1.1，仅修改正交性工程尺度；不能把gamma当普适SVD算法误差界或真实秩证明。
- 初实现615项全套通过后仍被独立反例拒绝：共享cutoff锚点遗漏。新增两个RED负例并修复分别计算自身cutoff，日志保留，不把测试通过替代独立审查。
- 最终全套617项通过、84.862秒、exit0；其中旧probe32、新数值20、新runner拒绝2。改动Ruff通过；全库110问题与基准快照检查相同，本轮不扩大修复范围。沙箱初轮子进程OpenMP/信号失败留痕，获必要权限外重跑，不改测试绕过。
- 最终代码/测试与backend环境SHA绑定到synthetic-certificates-final.json。此前synthetic-certificates.json属于cutoff修复前版本，仅保留历史。独立最终复审日志independent-review-final.log；复审批准仅本地工程，不是外部科学认证。

## 证据、安全与下一步

- 原failed目录、本次冻结27号协议、28号失败记录及执行包不变；核65本地文件SHA和基准字节。没有访问服务器实体，不将本地归档核验冒充远端实时核验。
- 没有读取旧真实P、H语义、目标数据，没有真实F0训练、GPU任务或smoke重启。没有更改rcond/loss/LR/seed/面板/提名门槛，不重建graphify、不覆盖其他窗口内容。
- d>=4504后误差包络可能宽过cutoff；19456行full-P存在丢弃方向时无法认证稳定秩。小面板/合成通过不等于full-P或完整G1通过，仍blocked/unknown。
- 新增重构/Gram/投影有成本；2是认证分解数、标准pair含初始LS为3，不是完整运算计数。正式资源profile与数学误差界审定仍待后续。
- 下一步先审full-P秩稳定性与成本的规格，若必要才只读原SHA绑定P矩阵并另存新proof。新真实smoke和正式GPU分别待明确执行指示，不resume原failed，不复用partial F0作正式初始化。

## 自检

准确性4/5：真实failed与本地通过分开，工程包络不是算法定理；尚未认证远端矩阵。
完整性4/5：完成规格、RED/GREEN、回归及复审；完整G1/full-P仍未解除阻断。
清晰性4/5：最终证明与旧失败/中途证明明确分层，历史日志较多需按入口阅读。
可操作性4/5：提供后续规格与授权门槛；不能直接从本地通过启动训练。
简洁性4/5：主报告汇总、完整真实日志单独留痕。平均4.0；自检不替代实验或独立复审。
