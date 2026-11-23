# 旧 A1-PG 24k 权重补充迁移

上一版 `module26-official26-4-inputs.tar.gz` 只包含数据和训练准备输入，没有旧权重。本次补充原 127 号实验的 **seed42、43、44 三个 `last.pt`**，不改变原数据包或新五臂训练方案。

补充包为 `old-a1-pg24k-reference.tar.gz`，**13,725,084 bytes（约 13.1 MiB）**，SHA256：

```text
f082cd5a63f3997ffd2bf01afd92234bd205f750866628e585b5c68a5602dda2
```

文件按独立补充资产加入同一 [GitHub Release](https://github.com/rookie-alt/spatial-meta-ce/releases/tag/module26-migration-20261123)，不覆盖已有 30 个资产。每个 seed 附原配置、训练计数/完成记录、loss/gradient audit 和原训练 spec，共 24 个原始 payload 文件；另含内嵌 manifest 与 CPU 严格加载验证。

| seed | 权重 SHA256 | 实测步数 | 参数量 |
|---|---|---:|---:|
| 42 | `770cd4eb5876a17a37be11ef7c6a87bc43e71ed879f754ca2e554594e55843c5` | 24,000 | 294,033 |
| 43 | `ff9399ab0d778788a27e55df218a93ab83ab333b67e169d04b512a4f501035d7` | 24,000 | 294,033 |
| 44 | `d1aa42d598da195b934836c6ee55b69ad427237d6a425ba08796a842550458a1` | 24,000 | 294,033 |

三个文件均与原训练记录的 SHA 一致，CPU `load_checkpoint()` 严格加载通过，模型类型为 `price_lite_pg_residual`。训练库为官方 26 库，`validation=[]`；未新增训练、适应、推理或目标数据读取。

## 下载与解包

在新服务器创建一个全新目录下载补充材料，无需重下三份大数据分卷：

```bash
set -euo pipefail
mkdir /gly/weilin/old-a1-pg24k-download-20261123
gh release download module26-migration-20261123 \
  --repo rookie-alt/spatial-meta-ce \
  --pattern 'old-a1-pg24k-reference.tar.gz' \
  --pattern 'OLD-A1-PG24K-*' \
  --dir /gly/weilin/old-a1-pg24k-download-20261123
cd /gly/weilin/old-a1-pg24k-download-20261123
sha256sum -c OLD-A1-PG24K-SHA256SUMS
mkdir extracted
tar -xzf old-a1-pg24k-reference.tar.gz -C extracted
cd extracted/a1-pg24k-reference
sha256sum seed42/last.pt seed43/last.pt seed44/last.pt
```

对照上表，三个 checkpoint 分别在 `extracted/a1-pg24k-reference/seed42/last.pt`、`seed43/last.pt` 和 `seed44/last.pt`。补充包只在本独立目录解包，不恢复或覆盖旧服务器执行路径。

## 读取和使用边界

在已经按[新服务器部署说明](module26-new-server-deployment.md)安装好代码与环境后，可以做 CPU 严格加载（不是模型评估）：

```bash
cd /gly/weilin/module26-prep-20261118/code-training
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 \
  /gly/weilin/module26-runtime/venv/bin/python -I -B - <<'PY'
import pathlib, sys
sys.path.insert(0, str(pathlib.Path.cwd()))
from spatial_meta_ce.checkpoints import load_checkpoint
root = pathlib.Path('/gly/weilin/old-a1-pg24k-download-20261123/extracted/a1-pg24k-reference')
for seed in (42, 43, 44):
    model, state = load_checkpoint(root / f'seed{seed}' / 'last.pt', 'cpu')
    assert state['next_step'] == state['extra']['counts']['optimizer_steps'] == 24000
    print(seed, state['model_type'], state['next_step'], sum(p.numel() for p in model.parameters()))
PY
```

旧 checkpoint 的 `metadata.fixed_steps=4000` 和 `selection` 文本是历史遗留字段，原件未改写。实际 `next_step`、`optimizer_steps`、原配置 `steps/preregistered_step_budget`、sealed plan 和完成记录均确认 24,000；具体核对值在 `OLD-A1-PG24K-VALIDATION.json`。

这些是旧 `122-official-26 / 122-A1-PG` 探索性实验的历史权重；其目标结果已经被查看。它们可用于原结果复现或明确标注的历史对照，不能冒充新 `132-module26` 五臂矩阵的完成 checkpoint，也不能据此跳过新矩阵的 15 次训练/封存门禁。新方案依然从头初始化；本次补迁移不授权额外训练或评估。
