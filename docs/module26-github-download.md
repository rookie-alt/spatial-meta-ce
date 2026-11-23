# 从 GitHub 下载迁移数据包

GitHub Release：<https://github.com/rookie-alt/spatial-meta-ce/releases/tag/module26-migration-20261123>。

仓库在本次发布时为公开仓库。三个分卷、清单和部署工具发布为 Release 资产，不进入 Git 历史。GitHub 每个 Release 附件必须小于 2 GiB，因此原 2.84 GiB 压缩包按原始字节顺序拆为三个文件；分卷不能单独解压。

在新 Linux 服务器上，先让管理员准备真实且可写的 `/gly/weilin` 目录。下面的下载命令不启动训练：

```bash
set -euo pipefail
mkdir /gly/weilin/module26-migration-20261123
gh release download module26-migration-20261123 \
  --repo rookie-alt/spatial-meta-ce \
  --dir /gly/weilin/module26-migration-20261123
cd /gly/weilin/module26-migration-20261123
sha256sum -c PARTS-SHA256SUMS
test ! -e module26-official26-4-inputs.tar.gz
set -C
cat module26-official26-4-inputs.tar.gz.part01 \
    module26-official26-4-inputs.tar.gz.part02 \
    module26-official26-4-inputs.tar.gz.part03 \
  > module26-official26-4-inputs.tar.gz
set +C
sha256sum -c SHA256SUMS
sha256sum -c TOOL-SHA256SUMS
```

GitHub CLI 未安装或未登录时，可以直接从 Release 页面逐个下载附件，也可用该公开地址模板（不需要访问旧服务器）：

```bash
curl --fail --location --continue-at - \
  --output module26-official26-4-inputs.tar.gz.part01 \
  https://github.com/rookie-alt/spatial-meta-ce/releases/download/module26-migration-20261123/module26-official26-4-inputs.tar.gz.part01
```

另外两卷分别是 `part02`、`part03`；还要下载 Release 中的校验清单、metadata、`DEPLOYMENT.md`、`PACKAGE-README.md`、`verify_archive.py`、`prepare_new_server.py` 及其证据文件。使用 `gh release download` 可一次取得全部资产。

合并后的原始归档应为 **3,049,224,614 bytes**，SHA256 为：

```text
9077240a3e1b560aed7eb0366049f6195629f26c647ae04224028c2613aaea73
```

检查通过后，按随包 `DEPLOYMENT.md` **第 4 节**继续完整回读、隔离解包和新服务器部署；不重复执行其中创建下载目录的第 3 节。解包后有 634 个 payload 文件、17,012,110,734 bytes。

数据、冻结计划和 37 文件训练代码闭包均未改变。新服务器继续原五臂 × 三个 seed、每次 24k 的方案；迁移文件和下载命令不构成训练启动，不复用旧服务器的 signed 授权或失败队列。
