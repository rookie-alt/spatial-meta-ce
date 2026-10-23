# Plato：A1 只读独立审查及修复复审（主代理归档工具回传）

2026-10-08，基线b3390b7。最终结论：**APPROVED，限A1代码修复及证据一致性，不批准正式实验。** 代理未写文件、未访问服务器/网络、未执行真实数据、未创建子代理。

## 首轮与后续发现

1. P2：true100/model125/PG80与反向，Decimal给两侧Q1.25，但原4eps相对log-Q tie误判为improved/worsened。已先补有效RED（13tests2failures）再改操作数尺度判等；不归零raw mean/median或Q95，既有1e-10改善仍识别。代理独立检查112个精确等比合成案例，全部中性；独立Decimal80位复算9案通过。
2. P3：早期a1-recompute.log与后来JSON源码测试SHA不同，不能当同次输出。已明确历史版本并另建同次after-review JSON/log/binding；最终源码更新后重新真实执行，不手改hash。
3. P2：数值规则修后，生成报告仍声明旧tie定义。已补实际生成report定义RED，同步fraction三个字段，并复算绑定。

## 最终批准核验

- 实际代码evaluate.py的improved/worsened字段明确排除machine-roundoff ties，fraction_ties定义与操作数尺度规则一致。
- 新test_paired测试实际生成报告，覆盖三个字段；a1-definition-red日志1test1failure有效，a1-definition-green-final为30tests/0.193s/OK。
- 最新9-case JSON/log均6174字节、逐字节相同，5个源码SHA均当前；共同SHA=f091150ffebae8cf8a0772cd8fa9dd57407d8506ca633c57b9626987ddd855aa。
- 最终独立执行：新增定义测试文件操作替换为内存fixture，另检查序列化summary定义，1test/0.007s/OK；FORBIDDEN_IO_ATTEMPTS=0，SOURCE_AND_EVIDENCE_HASHES_UNCHANGED=True，exit0。
- 作用域Ruff --no-cache与diff检查均exit0。
- 批准绑定evaluate.py SHA=819cdd51aa500c33e9dfe5c6031cf18ed5f09b6a9111dfddda316af122c1f7d5。

最终批准不认证源侧缺失原始预测、真实Linux、A2/A3或正式科学结论；最后完整回归由主代理执行并以新源码绑定为准。旧A/G1 failed及formal-ready状态不得因APPROVED改变。
