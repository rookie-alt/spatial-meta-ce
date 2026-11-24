# 执行、历史可比性与失败记录

本次执行四个新增组各三 seed，未新训 A1-PG 基线。历史 122-A1-PG24k 三份权重重新从 GitHub 获取并严格 CPU 加载，24000 步、96000 episodes、294033 参数和来源 SHA 一致；不能追认为 132 新训练完成。

## 历史可比性

- 三 seed 的训练计划 SHA、sealed plan SHA 逐一相同；capacity 文件 SHA 及旧模型 capacity JSON 与新 capacity 相等，包含 `[12,12,7,10]`、transform 和 26 库 cache 绑定。
- data JSON 只有 `/gly/weilin` 到 `/home/liushuaixi/weilin` 的迁移根路径差异。新 mapping 报告 26 库 / 1,300,000 行的 cache_content_hash / cache_manifest_sha256 与旧 capacity 相符。此处核对已有绑定，不冒称重新认证全部数组。
- 共同配方：ERM、SmoothL1、lr3e-4、4 episodes/update、query64、24k updates、96k episodes、26train、validation=[]、last.pt。A1-PG 共同声明 price_lite / 64维 / 2层 / 4头 / dropout0 / LoRA关闭 / PG头输入。
- fixed_stage 从 122-A1-PG 改为 132-module26，validate_every 从 4000 改为 24000，validation_seed 从 1729 改为 2026。当前 fixed_budget 不作验证选择。新 JSON 显式列 optimizer=adam / gradient_clip1.0，旧 JSON 未显式列，字段缺失不能证明实际优化器不同。
- 初始化规则均从头；没有旧 initial_state.pt 来证明逐 seed 初始化字节相同。旧 initial_state_sha256 是文件哈希，不能与新参数哈希直接比较。
- 旧 spec 仅绑定 8 份代码，缺 price_model/data 完整闭包；旧新 train.py SHA 不同。旧完整源码、环境及 CUDA 数值完全相等未证明。

允许报告“相对历史 A1-PG24k 的观察差”，同 seed 配对、四库 query 身份/真值及评估器一致；RelationTrue/Shuffled 是同轮训练对照。三 seed 全披露，不作显著性、因果或未见目标库声明。

## 四卡启动与修复

GPU0=GroupMean、GPU1=GroupAttention、GPU2=RelationTrue、GPU3=RelationShuffled；GPU4–7 不属于本轮。每 run 前重新核验空闲及绑定，任一 lane 失败即停止自有队列，不自动重试。启动监督使用独立 CLI、subreaper、进程身份核验及 pidfd；对 leader 先退、独立后代、日志关闭错误和 Popen 后身份失败已有 14 项 Linux 合成回归通过的记录。

第一轮 `launch-parallel-20261123-2136` 在 optimizer 构造前因缺 `packaging`（pkg_resources 回退也不可用）失败，checkpoint=0、completion=0，全部自有进程退出。仅向专用 venv 补 packaging==23.2；CPU 及 GPU0–3 各两步 Adam 合成更新检查后，用新的 `2236` 目录和输出绑定重启，37 份科研源码不变。首轮失败目录完整保存。

最终 launcher SHA256：`9629e29919bbf80f8de53761a086622fe1592fe2c9745568129d4edeee375d00`；启动 manifest SHA256：`1d54aaa0c94074f38abce76ad5f98365bd4814446c0be73b9f11147a52735f7b`。作为嵌入库且启动前已有子进程树时，subreaper 的旧后代归属存在已记录边界；本次独立 CLI 启用前未创建子进程，不受该边界影响，不作通用无缺陷声明。

## 自动评估与资源范围

用户后续授权“跑完之后直接评估”。独立历史比较控制器等待 12 新 run 全部完成，认证完成记录、预算、checkpoint 后，固定四库 K0 推理一次。正式 132 的全部 15 新完成门禁保留。

helper SHA256：`090873919e8480647aad77d88a0709492585d02ce43e11fb5946478a7c887f0b`；spec SHA256：`9a5855ffc17b0f2edf82b393c7a3fbf61cb266d93d543c3e385e1a4db040f79d`；authorization SHA256：`29cf145c510e67b286919a9c7e3f01d7af3a09af5e8485cc83617f7e808a343f`。authorization 是研究执行记录，不是 GitHub 凭据。工具 11 项 Linux 检查和独立复审通过的原记录在归档内。

worker RSS/VRAM/输出受监督；parent+worker CPU 被认证，外层总墙钟限制 3600 秒。parent RSS/输出未全程在线监督，不能把 worker 上限写成整个流程硬上限。实测 worker 峰值 RSS 970,788,864 bytes / VRAM 54,525,952 bytes，CPU 30.097 核心秒。

## 附件内容与复现边界

完整归档的相对目录保留：`outputs-parallel-20261123-2236/`、`outputs-parallel-20261123-2136/`、两轮 `launch-parallel-*`、`evaluation-tools-20261124/`、`evaluation-after-training-20261124/`、`reference-a1-pg24k-20261123/`、`frozen-training-source/`。15 份终点之外也保留新训练初始状态与审计；37 源码只是运行闭包，数据和环境需另恢复。逐文件清单见 Release 附件 `MODULE26-ARTIFACT-MANIFEST.json`，完整包自身 SHA 和清单 SHA 见 [artifact-index.json](artifact-index.json)。

本轮只归档训练、评估及原有验证，不追加训练或模型改动。132 准备分支与 main 已分叉；发布从 main 新建独立结果分支，仅加本目录及状态入口，保留主代码和其他实验历史。
