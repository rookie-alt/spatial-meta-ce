# 独立方法与G1边界审查

日期2026-10-07，基准19470ee；只读AI独立上下文，不是外部科学认证。未编辑、提交、联网、访问服务器/GPU或递归委派。代码作者/主代理测试与只读审查的执行记录分别陈述。

## 方法初审

原24号fixed rcond截断LS不能作为原线性类最优证据。标准库两行反例Z=[[-1e-14,1],[1e-14,1]],r=[-1,1]：截断LS/有限Adam J1，合法theta[1e14,0]J0，normal residual约1e-14仍过gate。建议只批准有限G1，并修订zero/tiny mask、批次/后端、checkpoint independent proof、H提名顺序及实际资源成本。H四库身份/历史分母与T/P/Q数目经本地原S2复核一致，四臂参数类别/计算成本混杂已披露；不要求扩容量/loss网格。

主代理随后发现fullrank病态伪造也可过弱gate；数值实现子代理按TDD追加同固定cutoffreference objective/预测核对并保守不发full-linear-optimality标签。只读审查快速概念复核接受“内部数值一致性而非独立最优证明”边界，未冒充重复数值worker所有测试。每路径两次lstsq+一次SVD成本需额外profile。

## source/hook首轮审查

独立只运行contract/preflight十二项（0.070秒）及合成tempdir探针，指出：

1. H manifest含inspection.true_min/true_max，整体json.loads及重序列化digest解析了标签范围，不能严格声称身份/count/SHA-only。没有训练/提名泄漏证据，但旧字段须更正语义。
2. 第一遍hash安全检查后，第二遍读取H未绑定目录；合成目录替换为根外symlink，外部manifest曾被打开。没有真实运行发生替换的证据。
3. 重复H身份表数冲突被首次去重掩盖；真实H各库duplicates=0，不影响既有分母，但需负例。
4. 第二个hook注册失败时第一个handle未清理。

## 修复后快速复审

主代理新增负例并修复后，独立限定contract/preflight十八项（0.124秒）通过：

- H identity_projection只raw_decode rows/count/content_hash，inspection/provenance字节不进入value decoder，也不重算包含标签stats的semantic digest。真实manifest原字节靠冻结reference的file SHA绑定。
- safe_binary以逐目录FD、O_DIRECTORY|O_NOFOLLOW和末文件O_NOFOLLOW打开。原“hash后将accidents改成外部symlink”探针在目录阶段拒绝，外部manifest打开次数0；打开目录后再重命名也保持原目录FD读取，不跟随后续alias。
- T/V也改走secure读数组；Hduplicate tables first/inverse一致性拒绝；hook在try内逐个注册/finally清理。
- 指出一条合法转义JSON夹具实际生成非法JSON；主代理改用json.dumps后重跑S8五十项通过。projection不是完整JSON合法性验证器，未解码段的可信性来自已认证文件SHA和历史有效writer格式；不能外推到任意未知JSON。

独立审查未运行最后新版真实source preflight；该部分由主代理实测，新proof与旧initial分开保存。fresh F0监督/P-only runner、提名锁定、checkpoint及完整端点独立核验、真实smoke/profile仍未完成。本次通过限定G1，formal_ready=false。
