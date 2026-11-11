# 122 P0 收口证据包（2026-11-11）

这是审计证据，不是 A/B/C/D 执行结果或训练授权。基准提交 `89623047d644842dca134867b6a6ec5f38c0e756`，分支 `audit/122-p0-closure-20261111`。

- `source_readonly.json`：本轮独立 stdlib/NumPy 只读核验的服务器 stdout；26 个源 pretrain manifest、156 个数组文件 SHA、48,000 个 episode、三源 population 完整身份与12份未执行配置。远端无写入、无项目导入、无模型运算。
- `checkpoint_readonly.json`：三组历史源侧 ERM-sum 的 config/history/validation manifest SHA，以及六个 checkpoint 的文件 SHA 与 primitive pickle opcode 证据。全checkpoint原始字节仅用于SHA；只解析ZIP的 `data.pkl` primitive opcode，不反序列化、不加载/解码或使用tensor值、不导入torch。原始JSON的 `tensor_storage_read=false` 仅指未按storage成员加载/解码，不表示SHA阶段未读取原始storage字节；原始stdout保持不变。
- `numpy_recompute.json`：独立 NumPy 合成期望值与生产指标的比较。包含 log-Q95、倍数 Q 分位数、等权 macro、三训练 seed 的配对 SD、空4+桶、负迁移90原始单元/六格/support SD层次和自比为零边界。`model_updates=0`，不是 smoke 或研究。
- `source_readonly.py.txt`、`checkpoint_readonly.py.txt`、`numpy_recompute.py.txt`：上述已执行脚本的原文，作为可复核文本，不接入库/CLI，不是正式 runner。
- `numpy_initial_failure.txt`：首次合成检查因错误的人工差值阈值 `>0.01` 失败的 traceback；未改生产函数，改为非相等断言后通过。最终另用二进制精确的 -0.25 检验 SD=0。
- `remote_vendor_storage.txt`：源侧部署 vendor 两个模型模块 SHA 与 `df -Pk` 快照；盘空间是快照，不是配额或预留证明。
- `commands.json`：本轮主代理关键实际命令、退出码和作用域；另独立撰写子代理执行偏差说明，不复制会话原文。
- `manifest.json`：本包文件与本轮原始输入的 SHA256、历史验证代码 SHA 对照。自身不放入自身哈希。

## 解释边界

`manifest_sha256` 是 manifest 文件字节 SHA；`content_hash` 是去除 `content_hash` 字段后的 canonical JSON 摘要；`data_sha256` 是 cards/offsets/group0..3 文件 SHA 构成的 canonical JSON 摘要；`plan_sha256` 不等于计划文件字节 SHA。二者均单列，不混用。

本轮没有重读 PRICE 原始 workload、pickle statistics 或 features40 文件；只复算派生源缓存及计划的现存字节 SHA。`upstream_provenance_recorded_not_reread` 只是缓存记录的上游 provenance，不是原始数据重新认证。真实目标缓存不在主代理的读取白名单内。

独立子代理曾违反只读/不执行模型的委派约束，做本地模型构造与两次完整 unittest；见主报告“执行偏差”。这些测试没有本轮要求的 smoke supervisor 证据，不能列为本轮合规 P0 验收、资源计时或训练授权依据。

所有生成的权重/大数组仍不得入 Git。本证据包不含 checkpoint、标签向量、目标预测或完整大训练计划。
