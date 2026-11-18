# 127 独立只读审查记录

代理`/root/review_official127`未参与实现，读取实际diff、上游七文件及验证信息；不SSH、不GPU、不写文件、不递归。基线`254efd946179a7532f15a97d661f979fc96782fc`。

最终审查确认metrics `e273e0d7…c2b2`、native `050d9d3f…8960`、runtime `7199e011…7817`、supervisor `6c67a55a…75c2`、协议`6fdf9419…a0a3`，完整SHA以[当前binding](reselected/binding.json)为准。独立复算器最终SHA`7f28a2b5fb4d755d4a635fa5a90f5990f20dc9dec4c76c3058365b7e64c333ff`。提交前实际逐字节核对。

## 发现与闭环

1. 初报极端Q-error舍入差异，随后审查者确认其对照脚本提前转Python float改变了上游round行为，正式撤回。改用上游原函数，500组随机及604组边界0 mismatch，主代理另跑200组0 mismatch。生产公式没有因此加clip或改变nan_to_num。
2. Genome重叠单位混用，已同时记录164个不同查询/166原始行，协议和binding一致，复审关闭。
3. 复算器all模式把supervisor旁文件当阶段，已限制目录且需completion，旁文件回归通过，复审关闭。
4. 复算器极端数值JSON序列化失败，已转null并明记诊断；不删样本、不改公式，复审关闭。
5. 复算器汇总路径及重复键排序，改为逐实际stage复算，report仅读汇总与完整原格字典比较。排序键包含prediction_file区分K0与微调同模型/库；53格键全部唯一、两种实际顺序重建后完全相等，主代理补53格集成回归，复审关闭。

最终结论：核心及复算器均**通过，无未解决阻塞**。审查覆盖原生padding/变换、累积权重与尾批、每epoch调度、400更新/500万呈现、不同counts严格重载、SHA与资源门、失败停止及输出不覆盖。

审查不是GPU或数据认证。实际CSV逐行、资源实测由主代理完成，见原始JSON/日志；GPU0 probe在worker启动前拒绝，新根/GPU6重选后重新执行K0/probe，再启动微调。微调仍运行，不宣称21格完成。
