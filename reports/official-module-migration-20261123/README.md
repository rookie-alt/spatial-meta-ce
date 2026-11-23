# 官方 26/4 模块实验迁移包

**GitHub 下载已就绪：** [Release 页面](https://github.com/rookie-alt/spatial-meta-ce/releases/tag/module26-migration-20261123) 提供三个字节分卷及 27 个配套附件，共 30 项；公开下载入口和 GitHub CLI 下载已验证。[下载与合并说明](../../docs/module26-github-download.md) 给出完整命令，合并后原包 SHA 不变。每项远端 SHA/大小均与本地一致，见[上传验证](release-upload-verification.json)、[发布记录](github-release-publication.json)与[下载验证](release-download-verification.json)。

用户要求：“那你先帮我把需要迁移的数据打包好，再写个新服务器的部署说明”。本次只打包和准备部署材料，没有研究训练、模型推理、驱动修改或新服务器部署。

## 包和校验

- 服务器目录：`myserver:/gly/weilin/module26-migration-20261123/`。
- 本机交付目录：`/Users/williamlin/Downloads/module26-migration-20261123/`；已下载完成，[本机五文件 SHA 校验](local-copy-verification.json) 全部通过。
- 压缩包：`module26-official26-4-inputs.tar.gz`，**3,049,224,614 bytes（约 2.84 GiB）**。
- SHA256：`9077240a3e1b560aed7eb0366049f6195629f26c647ae04224028c2613aaea73`。
- 原始内容：**634 个文件，17,012,110,734 bytes（约 15.84 GiB）**。归档另外含两份内嵌 metadata。

[文件清单](file-manifest.json) 和 [逐文件 SHA](payload.sha256) 绑定实际来源与包内路径；[压缩结果](package-result.json) 表明打包时全部源文件字节匹配原认证 SHA。[服务器完整回读](readback-verification.json) 重新解压并核对 634/634 文件、总字节、两份内嵌 metadata、tar 成员类型/路径及 gzip 完整性，结果通过。

`package-result.json` 的状态保留打包完成时的 `packaged_not_yet_readback_verified`，不追改历史记录；最终回读状态由 `readback-verification.json` 给出。`SHA256SUMS` 绑定压缩包和四个配套文件；大文件不进入 Git 历史，通过 Release 三个分卷分发，并保留上述原包副本。`release-preparation-validation.json` 和 `release-small-assets-verification.json` 中 draft/pending 字段是上传期间快照；最终发布状态以 `github-release-publication.json` 为准。

## 内容范围

| 内容 | 文件数 | 原始字节 |
|---|---:|---:|
| SQL workload、真实基数与 PG 标签 | 30 | 400,903,943 |
| PRICE features40.pkl | 30 | 7,433,190,703 |
| 直方图、fanout、表规模等统计 | 150 | 9,892,712 |
| 30 库缓存及可选 struct 数组/manifest | 270 | 3,076,730,765 |
| 官方 mapping、逐库 manifest 和根认证 | 61 | 1,706,344,979 |
| 原 source26 mapping 来源 | 52 | 1,698,501,740 |
| 三份固定计划、容量、划分和数据配置 | 6 | 2,686,311,711 |
| 原生产 extractor、builder 与历史 builder 字节 | 3 | 57,045 |
| 逐 run 预算参考 | 1 | 376 |
| 旧 unsigned spec/template/readiness 参考 | 31 | 176,760 |

覆盖 26 个训练库与四个固定 K0 评估库。打包四库文件仅用于迁移和完整性校验，不用于训练或方案选择；训练复验仍只读取 pretrain26。现有五臂 × seed42/43/44、每 run 24k、last.pt、validation=[] 的协议不变。

包内 `rootfs/gly/weilin/...` 保留原认证绝对路径。`references/` 是原准备文件的不可执行参考，不恢复到旧启动目录。包不含旧 signed 授权、启动队列、失败训练输出、checkpoint、Python venv、SSH 配置或凭据，也不含原始 PostgreSQL 表或数据库服务。

## 验证与部署材料

打包/回读工具采用独立脚本，不改动 37 文件冻结训练闭包。[合成校验](synthetic-package-validation.json) 验证正常回读、既有输出拒绝、源 SHA 漂移、symlink 和越界路径拒绝；[内层坏包校验](synthetic-inner-archive-validation.json) 在重算外层 SHA 后仍拒绝内容变化、缺失成员、重复和路径越界。工具 Ruff 0.11.2、格式检查和 CLI 帮助通过。

本机也完成 [634/634 文件完整回读](local-full-readback.json)，压缩包 SHA 与服务器完全一致。[新服务器部署说明](../../docs/module26-new-server-deployment.md) 提供传输、隔离解包、原路径恢复、GitHub 代码闭包验证、独立 Python 环境、CUDA 小运算、15 份未签 spec 准备及 dry-run 步骤。

[prepare_new_server.py](prepare_new_server.py) 不修改生产训练代码，复用已有认证与准备 API，拒绝既有输出根、路径重叠、symlink、`..` 和源码漂移；只生成未签模板。[7 项 helper 回归](helper-green.log) 通过，修复前的三项路径保护失败保留在 [red 日志](helper-red.log)。[全量回归](full-unittest.json) **1935 tests，OK（16 skipped），rc=0**，288 份 Python 源码前后 SHA 相同。Ruff 0.11.2 全仓通过；[8 个 Bash 代码块、2 个内嵌 Python 和 helper CLI 帮助](deployment-syntax-validation.json) 校验通过。

[打包工具独立审查](package-independent-review.json) passed，无未解决工具阻塞；明确保留 GitHub 代码与真实 `/gly/weilin` 路径两项前提。[部署材料最终独立复审](deployment-independent-review.json) **passed，无未解决阻塞**：首次审查发现的工具获取顺序问题已改为先传随包工具，再 clone 训练代码；[路径修复验证](deployment-path-fix-validation.json) 和 7 项 helper 复跑通过。`TOOL-SHA256SUMS` 在交付目录校验部署说明和三个工具/代码清单文件，不在 Git 证据目录直接执行。

本机交付目录已含 `DEPLOYMENT.md`、`PACKAGE-README.md`、三个工具、包/工具 SHA 清单和精简验证记录；旧服务器同步工具后 4/4 SHA 校验通过。原压缩包和 634 文件 metadata 完全未变。本次没有向另一服务器上传；实际新服务器的 CUDA、安装环境、输入认证和执行状态须现场验证。提交/推送状态以本次交付回复及 Git 为准。
