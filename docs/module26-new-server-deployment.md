# 官方 26/4 模块实验：新服务器部署说明

日期：2026-11-23。训练代码基准为 `9e40dd571c225269d7f781ca5d019c9fcdf66556`，分支为 `feat/official-module-training-prep-20261118`。本指南和 helper 所在后续文档提交不改动包内绑定的 37 份训练代码。

迁移范围是五臂 A1-PG、GroupMean、GroupAttention、RelationTrue、RelationShuffled × seed42/43/44，共 15 次；每次 24,000 ERM updates、从头初始化、只取 `last.pt`。26 库训练、`validation=[]`，四库仅作固定 K0 终点评估，不参与选择。迁移包和下面的准备命令不运行训练、不签发授权。

## 1. 已准备的包

本机目录：`/Users/williamlin/Downloads/module26-migration-20261123/`。

旧服务器目录：`myserver:/gly/weilin/module26-migration-20261123/`。

主归档 `module26-official26-4-inputs.tar.gz`：**3,049,224,614 bytes（2.84 GiB）**。

SHA256：`9077240a3e1b560aed7eb0366049f6195629f26c647ae04224028c2613aaea73`。

归档包含 **634 个文件、17,012,110,734 bytes（15.84 GiB）**：30 库 workload/真实标签/PG、PRICE 特征和统计、pretrain26 与 test4 缓存、官方及原 source26 mapping、三份固定采样计划和配套容量/划分、少量原生产源码证据。服务器与本机均已完整回读 634/634 文件通过；具体结果见[迁移包记录](../reports/official-module-migration-20261123/README.md)。

`SHA256SUMS` 绑定归档和 `file-manifest.json`、`payload.sha256`、`required-code-sha256.json`、`package-result.json`。额外的部署说明、工具和验证记录不属于 634 个冻结 payload，随交付目录提供。

归档布局为 `rootfs/gly/weilin/...`；`references/` 中的旧 spec/template/readiness 只作记录。包不含训练代码 checkout、旧 signed 授权、旧队列、失败训练输出、checkpoint、Python 环境或 SSH 凭据。代码需从 GitHub 拉取，不需要搬整个 PostgreSQL 数据库或原始表。

## 2. 新服务器要求

- Linux，Python 3.11；至少 32 GiB RAM，建议至少 100 GiB 空闲磁盘，容纳包、staging、数据副本、CUDA 环境和 15 次输出。
- GPU 和驱动须能运行 PyTorch 2.1.0 的 CUDA 12.1 构建；不能只看 `nvidia-smi`，以本指南的实际张量运算为准。更晚的新架构若不被该版本支持，需要单独验证环境变更。
- `/gly/weilin` 及其父目录必须为真实目录，不能使用 symlink。输入认证绑定绝对路径；若只能用其他目录，须重新生成路径相关认证，本指南不做字符串替换或绕过。
- 新服务器的 `/gly/weilin/PRICE-master`、`spatial-meta-ce`、`official-split-20261117`、`module26-prep-20261118` 均应不存在，防止覆盖已有项目。需要管理员提前给部署账号创建 `/gly/weilin` 并设置写权限。

以下步骤在新服务器上按顺序执行。代码块包含 `set -euo pipefail` 的步骤请在 Bash 中运行，任一步非零就停止。

## 3. 传输迁移包和随包工具

迁移交付目录另外提供 `verify_archive.py`、`prepare_new_server.py`、`required-code-sha256.json`、`DEPLOYMENT.md` 和 `TOOL-SHA256SUMS`。先把整个目录传到新服务器，再校验和解包；这一步不依赖 GitHub 上是否已有本轮工具提交。

```bash
set -euo pipefail
mkdir /gly/weilin/module26-migration-20261123
```

将本机整个交付目录上传到新服务器的 `/gly/weilin/module26-migration-20261123/`。使用 SFTP 或 rsync；rsync 用 `--partial` 保留中断内容，不使用 `--delete`。另一种方式是在新服务器配置好旧服务器 SSH 密钥后直接拉取：

```bash
# 仅当新服务器已配置可用的 myserver SSH 别名时使用。
rsync -a --partial -e 'ssh -o ControlMaster=no -o ControlPath=none' \
  myserver:/gly/weilin/module26-migration-20261123/ \
  /gly/weilin/module26-migration-20261123/
```

不要把本机 SSH 私钥或密码放进迁移包/Git 仓库。GitHub 克隆需有私有仓库访问权限。

## 4. 校验、隔离解包、恢复输入

先检查传输完整性并完整回读归档；结果路径使用新的文件名，避免覆盖包内附带的旧服务器证据。

```bash
set -euo pipefail
cd /gly/weilin/module26-migration-20261123
sha256sum -c SHA256SUMS
sha256sum -c TOOL-SHA256SUMS
test -f verify_archive.py
test -f prepare_new_server.py
python3 /gly/weilin/module26-migration-20261123/verify_archive.py \
  /gly/weilin/module26-migration-20261123 \
  --record /gly/weilin/module26-migration-20261123/readback-new-server.json
mkdir /gly/weilin/module26-migration-20261123/staging
tar -xzf module26-official26-4-inputs.tar.gz -C staging
cd staging
sha256sum -c ../payload.sha256
```

应观察到 `state=verified`、`verified_payload_files=634`、原始字节数为 17,012,110,734。检查通过后才复制到最终路径：

```bash
set -euo pipefail
cd /gly/weilin/module26-migration-20261123/staging/rootfs/gly/weilin
test ! -L /gly
test ! -L /gly/weilin
for name in PRICE-master spatial-meta-ce official-split-20261117 module26-prep-20261118; do
  if [ -e "/gly/weilin/$name" ] || [ -L "/gly/weilin/$name" ]; then
    echo "REFUSE: /gly/weilin/$name already exists" >&2
    exit 1
  fi
done
cp -a PRICE-master spatial-meta-ce official-split-20261117 module26-prep-20261118 /gly/weilin/
```

`PRICE-master` 此时只有此次输入所需文件，不含旧 venv。`code-ready` 只有原生产 extractor 和 builder，用来核对历史认证；不要改写它们。`references/` 不复制到旧执行根，不恢复旧队列或授权。

## 5. 安装代码和独立 Python 环境

恢复输入之后才 clone 到最终代码路径，避免与前一步的目标存在保护冲突。

```bash
set -euo pipefail
git clone --single-branch --branch feat/official-module-training-prep-20261118 \
  https://github.com/rookie-alt/spatial-meta-ce.git \
  /gly/weilin/module26-prep-20261118/code-training
python3.11 -m venv /gly/weilin/module26-runtime/venv
/gly/weilin/module26-runtime/venv/bin/pip install \
  torch==2.1.0 --index-url https://download.pytorch.org/whl/cu121
/gly/weilin/module26-runtime/venv/bin/pip install numpy==1.26.1 sqlglot==18.17.0
cd /gly/weilin/module26-prep-20261118/code-training
/gly/weilin/module26-runtime/venv/bin/python -I -B - <<'PY'
import json, pathlib, sys
root = pathlib.Path.cwd()
sys.path.insert(0, str(root))
from scripts.run_official_module_experiment import verify_code_tree
expected = json.loads(pathlib.Path('/gly/weilin/module26-migration-20261123/required-code-sha256.json').read_text())['code_sha256']
verify_code_tree(root, expected)
print('37/37 frozen training files match the migration package')
PY
```

不要直接运行顶层 README 的历史 23/3 流程。本次入口是 `run_official_module_experiment.py`，并使用下节新生成的 spec。

## 6. 验证实际 GPU 计算

示例选择新服务器 GPU0；若选其他卡，统一修改本节 `CUDA_VISIBLE_DEVICES` 和下一节 `--cuda-device`。本节只做小张量运算，进程结束后释放显存。

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONDONTWRITEBYTECODE=1 \
  /gly/weilin/module26-runtime/venv/bin/python -I -B - <<'PY'
import ctypes
lib = ctypes.CDLL('libcuda.so.1')
code = lib.cuInit(0)
assert code == 0, f'CUDA initialization failed: {code}'
import torch
assert torch.cuda.is_available()
a = torch.arange(64, dtype=torch.float32, device='cuda').reshape(8, 8)
actual = float((a @ a.T).sum().cpu())
torch.cuda.synchronize()
b = torch.arange(64, dtype=torch.float32).reshape(8, 8)
assert actual == float((b @ b.T).sum())
print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0), 'CUDA compute passed')
PY
```

该测试通过才表示具备这一次 CUDA 计算能力；不替代实际训练资源验证。失败时先修服务器环境，不继续开训。

## 7. 生成 15 份新服务器未签 spec

下面的 helper 完整复验 26 个训练库、三份固定计划和 37 文件代码闭包一次；然后派生 15 份独立 spec 和 15 份未签模板，逐项验证未签执行被拒绝。它不训练、不预测、不创建任何训练输出。

```bash
set -euo pipefail
cd /gly/weilin/module26-prep-20261118/code-training
timeout --signal=TERM --kill-after=15s 1800s \
  env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' \
  /gly/weilin/module26-runtime/venv/bin/python -I -B \
  /gly/weilin/module26-migration-20261123/prepare_new_server.py \
  --cuda-device 0 \
  --out /gly/weilin/module26-prep-20261118/training-ready-migrated-20261123 \
  --output-root /gly/weilin/module26-prep-20261118/outputs-migrated-20261123 \
  --code-dir /gly/weilin/module26-prep-20261118/code-training
```

`--out` 与 `--output-root` 必须全新、不重叠，不能位于 data/cache/mapping、代码或计划树中。helper 写 `readiness.json`；检查 `prepared=true`、`authorized=false`、`training_run=false`、`model_predictions_run=false`、`target_databases_read_for_training=[]`，且 `runs` 恰为 15。所有模板中的批准人/时间为空。

选择第一份新 spec 做实际 dry-run（不含 `--execute`）：

```bash
set -euo pipefail
cd /gly/weilin/module26-prep-20261118/code-training
module_spec=/gly/weilin/module26-prep-20261118/training-ready-migrated-20261123/A1-PG-seed42/training_spec.json
module_spec_sha=$(sha256sum "$module_spec" | cut -d ' ' -f 1)
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /gly/weilin/module26-runtime/venv/bin/python -I -B \
  scripts/run_official_module_experiment.py run \
  --spec "$module_spec" --spec-sha256 "$module_spec_sha"
```

应返回 `dry_run=true`、`executed=false`，且 `outputs-migrated-20261123` 仍不存在。第一次全量认证及 dry-run 可能用时数分钟，不能把文件生成当成训练通过。

## 8. 正式执行前的交接

到此只完成迁移和未签准备。本指南没有创建或启动执行队列；不要运行仓库中旧服务器的 `run-training-queue.sh` 或旧 signed 授权。

后续将新服务器 GPU/输出/spec SHA、原预算和用户既有授权范围核对并记录到新授权文件，再通过正式 runner/supervisor 逐 run 执行。已经明确授权的同一方案不重复询问；改预算、执行设备矩阵或评估范围时应具体说明变更。新服务器的 15-run 参数、采样计划和目标禁入边界保持原方案。

每 run 冻结上限：墙钟 30,000 秒、单 run 18,000 秒、CPU 30,000 core-seconds、GPU 24,000 allocated-seconds、RSS 24 GiB、VRAM 2 GiB、输出 2 GiB。15 次 GPU 上限合计 100 小时，属于上限加总而非预计耗时。失败停止、不自动扩预算或重试；固定四库评估另有全部 15 checkpoint 封存与评估授权门禁。

本次已验证旧服务器包与本机副本；尚未连接新服务器。新服务器的安装、解包、CUDA 运算、真实输入复验和训练状态均应按现场输出确认。
