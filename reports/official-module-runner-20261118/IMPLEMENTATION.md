# Official module contract and runner implementation

日期：2026-11-18。实现子任务只修改三个新源码文件及本报告目录；未提交、推送、SSH、GPU、研究训练、研究推理或递归委派。分支 `feat/official-module-training-prep-20261118`，基准 `2d1f766`。父任务模型、checkpoint、train 与 mapping 文件不属于本子任务写入范围。

## 交付与验证

- [module_experiment.py](../../spatial_meta_ce/module_experiment.py)：`STAGE='132-module26'`、`PROTOCOL='official-module-26-4-v1'`、五臂、seed 42/43/44、24,000 updates、last.pt 唯一终点。
- [run_official_module_experiment.py](../../scripts/run_official_module_experiment.py)：`prepare/verify/run/worker/evaluate-ready`。
- [test_module_experiment.py](../../tests/test_module_experiment.py)：最终 **21 tests OK**，Anaconda Python 3.11.7；F/E lint（排除 E501）、格式检查、`git diff --check`、五个子命令帮助均退出 0。
- [verification.json](verification.json)、[unittest.log](unittest.log)、[ruff-fe.log](ruff-fe.log)、[ruff-format.log](ruff-format.log)、[diff-check.log](diff-check.log) 保存命令、真实结果和源码前后 SHA；三份源码字节一致。
- 初次 red 是新模块尚不存在导致 ImportError；随后实现并完成 green。

验证均为合成或模拟证据：26 个微型源缓存、三份 2-update sealed plan；未签拒绝、授权绑定、wrong output/data/closure 拒绝；typed aliases；流式 plan canonical digest、support/query ID/hash/完整数组 SHA；worker 调用 train 前注入证据；十五 endpoint 未齐先拒绝；父进程资源认证；CSV ID/hash/truth 配对和独立统计。常量 patch 与 mock 只存在于测试。

本子任务未运行真实 train。父任务另负责五臂实际两步 train→checkpoint 合成整合测试、全量测试和独立只读审查，其结果不记为本子任务执行。

## 父任务 API

```python
model_config(arm, capacity)
mapping_mode(arm)  # None / 'mapping' / 'shuffled_mapping'
training_config(
    arm, seed, capacity, *,
    data_config_binding={'path': '/absolute/data.json', 'sha256': '<64hex>'},
    training_plan_binding={
        'path': '/absolute/plan.json', 'sha256': '<64hex>',
        'plan_sha256': '<64hex>',
    },
    mapping_root=None, module_execution=None,
)
validate_training_config(config, data_config, resume=None, *, output_dir=None)
checkpoint_metadata(config)
validate_checkpoint(state, arm, seed, *, expected_config=None)
```

train 入口必须传 `output_dir=output_dir`；缺失或不等于 spec `outputs[run_key]` 拒绝。worker 在每次 train 调用前注入：

```json
{
  "module_execution": {
    "spec": {"path": "/absolute/training_spec.json", "sha256": "<64hex>"},
    "authorization": {"path": "/absolute/authorization.json", "sha256": "<64hex>"},
    "output_dir": "/absolute/new-output",
    "run_key": "A1-PG-seed42"
  }
}
```

关系配置 `{root, mode, scope:'official_26_4', split:'pretrain'}`；其他臂无 mapping。两关系臂同类型 `price_lite_pg_official_relation_residual`、同模型配置，由 mapping mode 与 metadata 区分真实/打乱。分组与关系不得组合。`checkpoint_metadata` 包含完整 training_signature（含执行证据）；父任务原 metadata 中 mapping_provenance 通过 update 保留。

配置精确固定 ERM/post、SmoothL1、24k、batch4、accum1、Adam 3e-4、clip1、supports 5/10/20/50/100、query64、inner 1e-3/5/all。额外字段、resume、initialization、best、旧 122 preflight、额外步数覆盖均拒绝。

## Spec 与命令

训练 bindings：`code_dir/code_sha256/data_config/capacity/split_manifest/training_plans/mapping_report`，可选 templates。普通文件绑定 `{path, sha256}`；三 seed plan 绑定另含 plan_sha256。可选 templates 必须为本合约 baseline training_config 的精确结果，不需要旧模板。源码闭包必须精确 REQUIRED_CODE，缺项/额外/逃逸/symlink/ROOT 不同/hash 漂移均拒绝。

config_sha256 始终含完整 5×3 配置；execution 为 `{'mode':'matrix'}` 或 `{'mode':'single_run','arm':..., 'seed':...}`。single run 不改变完整 arms/seeds 预声明；实际 outputs 只包含选中项。

独立 purpose：train=`official-module-training`，evaluate=`official-module-k0-evaluation`。授权精确绑定 protocol/purpose/action/spec SHA/budget SHA/output/execution keys/target reads/selection_use；authorized 必须 bool True，相应 formal flag 才可 True；批准人非空，granted_at 为带时区 ISO8601，例如 `2026-11-18T12:00:00+08:00`。prepare 只产生 false/None 模板，拒绝旧 122 授权。

以下仅展示命令形状，未执行：

```bash
python -B scripts/run_official_module_experiment.py prepare \
  --data-config /absolute/data.json --capacity /absolute/capacity.json \
  --split-manifest /absolute/official_split_manifest.json \
  --training-plan 42=/absolute/plan42.json \
  --training-plan 43=/absolute/plan43.json \
  --training-plan 44=/absolute/plan44.json \
  --mapping-report /absolute/mapping/official_module_inputs.json \
  --budget /absolute/explicit-budget.json --cuda-device 1 \
  --output-dir /absolute/new-output --out /absolute/new-preparation

python -B scripts/run_official_module_experiment.py verify \
  --spec /absolute/training_spec.json --spec-sha256 '<actual64hex>'

python -B scripts/run_official_module_experiment.py run \
  --spec /absolute/training_spec.json --spec-sha256 '<actual64hex>'
```

无 execute 为 dry run，授权状态分别报告 not_provided/not_authorized/signed_and_bound，不实例化模型或创建执行输出。执行另需 `--authorization PATH --authorization-sha256 SHA --execute`，仅 Linux；CUDA index 显式绑定，worker 核对可见设备与自身 supervisor parent，无重试。

复用既有 BudgetTracker/CudaMemoryTelemetry/UnitDeadline/supervisor_budget/check_output_path/validate_budget，不修改旧入口。completion 存累计预算和按观测 unit marker 分配的采样秒；同一采样间隔内边界存在归属分辨率，累计计量为权威结果。预算来自调用方 JSON，不估测性能。

## 大计划与独立评估

真实计划可能约 895 MB/seed。runner 以 64 KiB 块和单 episode 有界解码认证，逐条检查身份/唯一性、完整数组 SHA、recipe 和 canonical plan digest，不 read_text/json.loads 整份计划或保留三份 episodes。摘要缓存按完整 SHA 与绑定索引，每次命中先重新读取完整文件 SHA，不用 mtime/stat。CLI __main__ 复用正式模块名，train 门禁复用同进程摘要。single run 深验选中 seed，其他 seed 验文件 SHA/manifest binding；矩阵深验三个 seed。实际 train 仍只按所选 seed 读取一份计划。

evaluate-ready 要求全部十五份记录、checkpoint SHA 和对应父进程 supervisor/budget completion 通过，才读 test4；允许多个 training-records 文件合并，缺项/重复拒绝。评估 spec 另绑定 training_completions 文件 SHA。不会打开训练缓存或用 target 最大 token 数重建 capacity。

report 复用 sibling `verify_input_report`：训练仅 pretrain，评估仅 test。K0 无适应/选择；两种人口 CSV 分别保留 ID/hash/double truth/raw log prediction，不裁剪原输出，异常保留 failure 证据并失败。主口径 unique-query Q95 与四库等权 macro log-Q95，all-rows Q50/Q90/Q95/Q99 辅助；父进程独立复算逐项核对报告及人口真值，提供完整 paired seed delta，无 best 或 significance 判定。

真实输入准备、服务器资源监督/大计划性能、完整四库模型评估未由本子任务执行。独立审查由父任务完成；本报告或测试通过不构成执行授权。
