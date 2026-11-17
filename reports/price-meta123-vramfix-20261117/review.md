# 显存监督与预算修订独立审查

日期：2026-11-17。独立只读审查者：`/root/review_vram_revision`，未参与实现。
基准提交：`f6106365b70eb79a39bb4abf0d8d21d3e98ba915`。

## 代码与修订审查

结论：**通过，无未解决阻塞项或重要级缺陷**。覆盖实际生产 diff、两个测试文件和124修订；不将此结论替代服务器执行认证。

| 文件 | 已审SHA256 |
|---|---|
| `spatial_meta_ce/price_meta123_runtime.py` | `67729224a682e22548d9898cab687328e4073430e0399b16b9a55a542a9d220d` |
| `spatial_meta_ce/price_meta123_supervisor.py` | `21329c12a6f027428c53ae976df9d739bc863d494d0343e478b69e3e4f29edfb` |
| `tests/test_price_meta123_runtime.py` | `f55b49c0f1b3d30a3801ca7a0fcc54b012f28be7f7cccdc3302fbe9af86d27c1` |
| `tests/test_price_meta123_supervisor.py` | `bd6c59bb8356f8440132d9e151e4c3a6c7ca28262eba3fdd5da00b19cf3650e6` |
| `实验文档/124-原版PRICE显存监督与预算修订.md` | `92499705964f20e7ecada182cd21fb1efe82498cb59b0cb8727e947ce087994d` |

核对整卡UUID计量、不依赖容器PID交集、忙卡启动拒绝、CUDA子进程UUID固定、监控异常失败停止、只清理本任务进程、24h/E2h/M6h各门槛及124来源绑定。原123文件SHA未变。旧外部VRAM=0无法通过新的`_cost_gate`。

审查者执行专项测试及Ruff，报告通过；其回传的不同Python测试计数存在口径差异，不作为精确测试总数来源。主代理留存的真实日志是：本地全量1857项，`OK (skipped=22)`；服务器Torch2.1/Linux专项101项，`OK`、无跳过。见本目录两份日志。审查者未独立运行这两套完整命令，也未执行真实GPU探针。

非阻塞限制：整卡其他意外占用会并入本任务资源并导致停止；这属于保守上界，绝不对其他用户PID发信号。基线部分模块不兼容Python3.9，应使用已验证的Anaconda或服务器Python3.11。成本门的supervisor/completion对应关系依赖`_completed`既有封存校验，已核对链条成立。

## 部署核验

新binding `48204dfc5f5911b611ae04a8d2c777baef0ec85d540d7d492fcfe7bee63123bb` 的34个代码依赖与已审本地文件SHA逐项一致。GPU实际显存、D2新投影及源侧执行状态另以真实结果认证；本节不预先宣布通过。

## 实际D2与预算证据复核

GPU6在检查后被其他任务占用，首次root的预检记录为`gpu_not_idle`，`pid=null`、`samples=0`、输出0B、0.203秒，worker从未启动。审查者确认这是正确的启动拒绝；保留旧root后，在新root重新绑定并选择空闲GPU不构成工程失败重试。历史记录不删除或重标。

新root为`/gly/weilin/price-meta-123-vramfix-reselect-20261117`，binding `5d81667b8ad958c92f15fab4b28dbd5c421e7fe8fe02796846298f200d8934f3`，34个依赖SHA与已审版本一致。实际探针在GPU2完成。审查者只读核对本地`reselected-gpu2/`九份证据，以独立公式直接计算，未调用生产`project_budget`：总时20.9865083322h≤24h、E单run0.5301241675h≤2h、M单run5.4815280673h≤6h、单元34.038904s≤120s、输出1.9944GiB≤6GiB，均通过。

外部`device_total`峰值1,409,286,144B=1.3125GiB，明显高于1MiB baseline及16MiB认证线；worker allocator峰值938,945,536B=0.8745GiB，两者口径不同，外部0计量缺口闭合。GPU index2/UUID在授权与记录一致。D2准备加探针504.7844125109秒、532,078,126B均在30分钟/512MiB以内。timing SHA、completion/supervisor副本及binding一致，实际只打开23训练库；无源验证/目标读取。

结论：**实际D2证据复核通过，可进入源侧阶段**。M单run预估离6h上限只剩约8.6%，后续超限仍停止，不自动加预算。审查者未SSH、未重跑GPU；主代理直接从服务器取回这些副本并完成checkpoint/CSV验证，详见`reselected-gpu2/d2-postrun-validation.json`。

## 固定串行脚本审查

同一独立审查者随后只读审查`run_source_continuation.py`，SHA `0f6e1c521a72c9da383f09f52bac44b96fcef3e4bcdb70a62262d55a8a513d14`。结论：**通过，无阻塞，可启动**。脚本固定九阶段，只调用已审运行器，阶段前核对授权/自身SHA；所有输出预先要求不存在，失败即停止。累计24h/6GiB扣减、单卡空闲预检、六个last封存后校准/报告、目标禁入均仍由生产路径执行，脚本不覆盖这些门槛。

主代理执行两项模拟（非研究实验）：九阶段成功顺序返回0；第二阶段返回失败时只调用前两阶段，保存failure并返回1，后七阶段未调用。Ruff和格式检查通过。实际九份授权在服务器创建后取回本目录，包含脚本SHA与单阶段action；主代理核对后于15:18:26（北京时间）启动后台PID7151。审查者未执行模拟、未SSH、未核验服务器授权原件，未把模拟当作正式完成。
