# 续做独立复审摘要

只读AI子代理，非外部人员/科学第三方认证；基准8fe6de4，日期2026-10-07。未改工作区文件、未运行完整suite、未联网或访问服务器、未执行新实验或委派。

## S7工程

独立读代码/diff、合成验证和实际proof后确认两项核心修复有效：稀疏非空群体数值先验证；独立重放support/query全部逐query数组并匹配公共hash链。原两项边界（uint顺序检查、boolean accumulation）建议主代理补齐，随后按TDD修复，最新服务器再认证的verifier SHA与主代码一致。

独立角色重放仍共享NumPy RNG API；不声称另一随机数引擎再现。采样角色计数不等于ERM inner-loop监督量，描述性/因果边界不变。

## 论文数字/表/图

独立复核：

- 三目标×六方法×六K共108目标单元完整，14份目标摘要/1260 episode/252初始化单元的五support均值、三seed均值和样本SD正确；PRICE supportSD、PG无SD分离。
- 章节108主表值及SD列与CSV一致，S4/S5/S6逐库Q95/等权macro/B÷A−1一致，no_candidate保留，不pooled或虚构source seedSD。
- S7 seen/unseen库内比中位数与方向计数从generalization独立重算一致。
- 70输入、原36来源、18非manifest输出及章节hash绑定通过；14 CSV所有可解析数值finite。
- 三目标SVG各24点、源SVG12点，逐点/折线/SD帽坐标正确，log轴/分类K轴正确，PG无误差条；两类SD的图例明确。
- 19产物及章节与renderer一致，两次临时目录导出字节相同。原始目标标签、SQL语义与逐查询Q95未重新认证，来源hash一致不扩张为原证据真实性验证。

## 发现的导出路径边界与修复

初审export仅检查末级path/目录，祖先symlink可绕过，tempdir历史target-q95.csv sentinel实际被覆盖；当前交付路径无symlink、历史SHA仍一致，不否定当前科学数字。主代理新增负例先实测failed，随后在mkdir前拒绝所有祖先symlink、写各文件前拒绝末级symlink/hardlink；章节路径在导出之前做同样拒绝。macOS临时测试基目录显式resolve以区分系统路径别名与测试中的不可信别名。15项paper专项通过，当前renderer/来源表/图字节未变。

路径补丁最终复审和最新完整suite见主报告，旧writer14项记录保留为修复前快照，不写成当前全部覆盖。无下一GPU实验或S8执行声明。


## 路径补丁复审（最终版本）

只读复审接受最小path补丁，单独重跑新增测试1项通过；临时探针确认祖先symlink（已有/未建目录）、末文件hardlink拒绝、普通真实目录可写、chapter alias拒绝在export之前。原19产物/章节及18非manifest输出绑定均未变。当前exporter SHA8b74bd99a81cba9e72927352df787f9611cfc69b079b89ac20c63b31987a599f，旧writer代码SHA仍是交付快照。主代理新final-verification绑定当前代码/测试，不改旧记录。
