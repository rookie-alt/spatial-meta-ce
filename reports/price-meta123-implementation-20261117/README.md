# 实验123：原版PRICE元学习实现与隔离部署

日期：2026-11-17。用户授权：“那就按照这个方案开始实现吧，记得新开一个分支，在服务器上新建一个单独的文件夹，不要和其他的产生冲突”。

本轮覆盖方案D1工程实现、隔离目录建立、CPU合成验证与少量真实源输入的前向等价检查。GPU成本探针、LR校准、正式研究训练、源报告及目标评估尚未执行。

## 隔离位置

- 分支：`feat/official-price-meta-123-20261116`，从已审方案提交`c9b55cf`建立。
- 本地worktree：`/Users/williamlin/Downloads/13/spatial-meta-ce/.claude/worktrees/official-price-meta-impl-20261116`。
- 远端独占根：`/gly/weilin/price-meta-123-impl-20261116`，本轮全新建立，权限700。
- 远端子目录：`code/`、`inputs/`、`outputs/`、`evidence/`。代码与证据只写入这个根，不改其他运行目录。
- 初始checkpoint只读路径：`/gly/weilin/122-official-price-seed42-20261115/output-v2/last.pt`，服务器实测SHA与方案`dee83c46…070f`一致；原PRICE项目只读。
- 缓存根：`/gly/weilin/spatial-meta-ce/artifacts/cache`，只读。训练API仅打开严格train白名单，源校准/报告另用source白名单；拒绝数据库软链别名、数组软链及数据库身份错配。

主工作区及其他worktree的未提交内容没有迁入或覆盖。历史23库/26库结果仅作来源核对，不作为本轮研究数字。

## 新实现

| 文件 | 行为 |
|---|---|
| `scripts/run_price_meta123.py` | 独立命令入口；help/bind无模型计算，不自动连续执行阶段。 |
| `spatial_meta_ce/price_meta123_model.py` | 原生PRICE结构/输出和MSE；vendor/checkpoint SHA；functional一阶SGD、模式恢复、非有限拒绝。 |
| `spatial_meta_ce/price_meta123_plans.py` | 固定23/3/4边界；三条8000-task计划；S100/C2000/R去重身份、缓存内容绑定和重认证。 |
| `spatial_meta_ce/price_meta123_results.py` | 不导入模型的原生z解码与Q95复算，完整矩阵/分母校验、有限LR校准、配对查询诊断与预设门槛。 |
| `spatial_meta_ce/price_meta123_runtime.py` | 阶段授权/版本绑定、独占执行锁、成本可行性门、6运行封存、单cell超时、累计预算和研究阶段隔离。 |
| `spatial_meta_ce/price_meta123_supervisor.py` | Linux自有进程组资源监督；监测RSS/VRAM/输出/墙钟，首异常留证、清理与完成认证。 |
| `tests/test_price_meta123_*.py` | 原生等价/完整模型梯度、独立标量梯度、计划隔离、CSV复算、矩阵拒绝、合成6运行封存与Linux清理验证。 |

FOMAML额外支持集审计前向不参与更新，成本分别登记`presentations`、`audit_forward_presentations`和`total_forward_presentations`。后续checkpoint保存plain PRICE state供复核，另存优化器/RNG审计状态；禁止把旧run当resume。

## 阶段使用

先在专用`code`目录绑定已部署代码、数据配置与共同初始checkpoint：

```bash
CUDA_VISIBLE_DEVICES="" /gly/weilin/PRICE-master/venv/bin/python -B \
  scripts/run_price_meta123.py bind \
  --root /gly/weilin/price-meta-123-impl-20261116 \
  --data-config /gly/weilin/122-engineering-closure-20261112/rebound/data_config_122.json \
  --checkpoint /gly/weilin/122-official-price-seed42-20261115/output-v2/last.pt
```

`bind`只核对与保存路径/SHA，不生成真实任务计划或运行模型。代码有修改后旧binding失效，不能修改旧证据来掩盖变更。

每次`run`需要匹配binding的独立授权JSON，包含`schema=1,protocol=123,authorized=true,root,binding_sha256,actions,granted_by,granted_at,budget`；GPU阶段另含`cuda_device`。budget严格字段是`max_wall_seconds,max_rss_bytes,max_vram_bytes,max_output_bytes,poll_seconds`，不得突破123方案上限。此处不提供预填true的未来执行授权。

阶段依赖：`prepare-training → probe → prepare-source → calibrate-p → [confirm-alpha if needed] → train:E/M:42/43/44 → seal-training → calibrate-e → report`。每步独立命令，旧输出或sidecar存在就拒绝，不自动重试；所有计算阶段由自己的Linuxsupervisor启动。主报告只在全部训练与校准封存后读取R做模型前向。

## 已有验证与当前状态

- 最终本地完整回归：1839项，`OK (skipped=22)`，无失败，耗时178.075秒；记录见`local-full-suite-final.log`。不把skip当通过。
- 独立复审本地123专项：83项，`OK (skipped=6)`；6项为Linux专属。
- 服务器Torch2.1/Linux123专项：83项，全部通过、无skip；最终逐文件绑定版本另保存`tests-123-linux-bound-version.log`。
- 真实`accidents/pretrain`两条查询CPU前向：新旧包装器逐位一致，`max_abs_difference=0.0`；参数量10,709,330。这是工程等价验证，不是精度评估。
- `ruff check scripts tests spatial_meta_ce`和新增文件`ruff format --check`通过；最终源码修改后需再核验。
- 源缓存身份实测：23个训练库及3个源验证库，全部26个manifest/来源/row id验证通过，证据`source-store-identity.json`；没有读取目标库。
- 独立初审1重要/2建议已关闭：LR校准强制完整K0+K20的48/144单元，完成证据副本等值校验，脱离session子进程按starttime持续计量并TERM/KILL清理；主自查缓存软链问题已修。最终复审无未解决阻塞或重要问题。

最终校验发现服务器监督文件少了本地已审版本的Python兼容分支，已保留初始binding至远端`evidence/binding-before-final-sync.json`，补传最终代码并重新测试/绑定；初始binding不用于执行。`binding.json`副本绑定33个依赖文件，必须与本地逐项SHA一致，不能以部署成功代替内容核对。

最终binding SHA256：`c990d72a98229fcac5c4f98fc9c2f54036c099aa9a6ae8201dc95618abaf5ed1`。33文件本地/远端SHA逐项匹配；最终服务器83专项耗时21.739秒，全过无skip。

非有限报告单元终止整个阶段并保存`failure.json`，不会输出科学负结果或静默忽略查询；这是方案`invalid`在运行器中的失败形态。资源监督能追踪已采样后代及其reparent/setsid；worker本身不启动子进程，不声称能捕获采样前已完全脱离可观测树的未知进程。

本轮没有占用GPU、没有运行P/E/M研究矩阵、没有读取目标库。下一步是完成最终独立复审与执行包绑定，再按方案准备单卡成本探针的具体授权和预算。源侧工程完成不等于元学习精度改善。
