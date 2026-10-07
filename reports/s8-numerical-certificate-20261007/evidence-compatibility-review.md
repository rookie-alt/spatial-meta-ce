# 失败证据与调用方兼容性独立审查

日期：2026-10-07；只读独立 AI 子代理，基准 `7eb3370`。不是外部科学第三方认证。

- status/execution log一致指向h small-panel认证拒绝；x是后置诊断，不是原smoke执行。metadata和reference两处全谱atol0均拒绝，参考objective/预测通过不等于reference_pass。
- 保存的是完整P原始FP32表示/label/PG；small-panel固定1216行才拟合normalizer。失败发生在保存theta/normalizer前，没有可直接加载的已冻结normalizer/design实体；旧诊断从固定索引重新构建。四P矩阵实体与checkpoint留server，本轮未重算其实体SHA。
- 独立核对本地14执行包文件、6归档文件SHA以及诊断status/contract绑定一致；不得把本地归档hash核验外推成远端实时验证。
- 保持pass==solver_verified、rank_limited==numerical_rank_limited、fullrank_numeric_solver条件及full_linear_optimality全False；科学rcond与normal/objective/参考预测门槛不变。
- 两个runner入口均消费pass；新增工作不得遗漏成本披露。当前认证仍一次SVD加一次reference LS，但新增Gram/重构/投影矩阵运算有额外成本，分解计数不能代表完整耗时。
- endpoint独立verifier不认证probe或normalizer；小面板或新版本旧矩阵通过不能解除full-P/nomination/H/完整端点未执行阻断。
- 已由隔离实现代理新增两处synthetic runner拒绝测试，mock真实训练/solver/cache/checkpoint；不落盘相应probe、不进入full-P或seal/H。本轮新增2测试通过及Ruff通过，主代理再纳入完整回归。
