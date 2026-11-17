# 125划分修订与实现独立审查

日期：2026-11-17。基准：`ad8237eb94856f029f2e163eb6756eca744209fc`。用户指令：“不要三源验证了，改成和官方一样的”。

## 方案

独立只读代理`/root/review_official26_plan`确认26训练/0验证/4测试、官方P、固定alpha及目标不参与选择的方案无方法/权限阻塞。三库加入训练后不再称跨库留出；旧123/124保留。

意见处理：新`price_meta125_plans.py`独立处理26/0/4，不调用旧三源角色门或强制非空validation的旧入口。成本必须重做26库探针，不能复用旧D2放行。未采纳探针输出上限增到1GiB的建议：任务数仍固定8000，不随库数按26/23倍增，新探针也不写旧评估CSV；实测准备270,430,806B，剩余266,440,106B给probe，仍按512MiB硬限。审查中暂缺plans的中间报错不当作最终测试结论。

## 代码

未参与实现的只读代理`/root/review_official26_runtime`审查runtime、plans关键约束、入口、测试和supervisor小diff：**通过，无未解决阻塞**。

| 文件 | 已审SHA256 |
|---|---|
| `spatial_meta_ce/price_meta125_runtime.py` | `78d0f99a3184bf4fc123be7839bd59041873cf45af3147e20cd239efc1e4ca79` |
| `spatial_meta_ce/price_meta125_plans.py` | `021203287526f626375480bed43b77f1b0ed4456a200be78330027a1f39f6776` |
| `spatial_meta_ce/price_meta123_supervisor.py` | `cc3411804afc4602b26735305af841620a8725476ff795be1f6daf9832234a24` |
| `tests/test_price_meta125_runtime.py` | `163497aa241d28647de883bf1ec270891b2aeb724634aad666eaa00159904696` |
| `tests/test_price_meta125_plans.py` | `fc8212139bcda641964f167cddb5918b7b70ca0e16bda9c25cd5835e85e9d79f` |
| `实验文档/125-官方26库划分的PRICE元学习实验.md` | `354f1d651b1617beb337468dccd73648e7c101d0564f98a04553e9528e898f92` |

核对官方P固定SHA、目标白名单拒绝、无旧校准gate、固定LR、累计预算、任务S/Q身份和cache SHA复验、六last seal、checkpoint/optimizer step重载、显式125完成协议且123默认不变。opened_databases随机顺序疑点经审查者核实前置verify遍历26库后自行推翻，无需改动。

非阻塞项：Adam betas/eps采用PyTorch默认值，数值与协议完全相同，冻结绑定后不为显式写默认值重改生产文件；`target_databases_read=[]`由训练白名单保证，是范围声明而非文件系统全路径跟踪，限制明确保留。

## 串行脚本

同一审查者另审`run_training_continuation.py`，SHA `8277b6a8e5eac34b68c0189841c570a079b29f71234c130105bb9ff0706da473`，及AGENTS中125例外：**通过**。脚本固定E42/M42/E43/M43/E44/M44/seal，只调用受监督运行器，逐阶段校验授权/自身SHA；首次失败即停、不覆盖、不续训、无目标命令。脚本通过逐阶段授权单独绑定，未混入已冻结代码closure。

主代理模拟（非研究实验）：7stage成功顺序返回0；第2stage失败只调用前2、后5不启动，保存failure返回1。Ruff/format通过。旧123的E42完成、M42用户变更导致parent_signal及cleanup验证的证据已由审查者核对，不能当科学失败。

## 验证与限制

本地全量1886项，239.418秒，`OK (skipped=22)`、exit0。Linux/Torch2.1最终130专项，91.215秒，`OK`无skip。Linux首轮两项旧123测试因初始部署缺历史timing.json报错，补齐只读fixture后复验，首轮日志保留。runtime10项主代理/审查者各跑通过，plans19项实现者/主代理通过，supervisor72项本地通过、6Linux项跳过；不重复相加测试数。

官方权重GitHub blob身份与真实chembl两条原生/PriceAdapter逐位一致由主代理服务器核验并归档；35文件binding与已审本地SHA一致。审查者未SSH、未跑GPU或完整回归；对plans是关键路径定向核验，非逐行形式证明。后续probe/研究训练与最终目标成绩仍以各阶段原始证据为准，审查本身不产生科学结果或目标权限。

## 实际26库D2证据复核

同一审查者独立复算125公式：72,457.76640154627秒=20.1271573338小时，E单run0.7559261119小时、M单run5.8935549060小时，均小于24h/2h/6h；投影输出1,313,330,810B≤6GiB。外部整卡峰值1,627,389,952B=1.515625GiB>baseline1MiB和认证线16MiB、≤12GiB；两臂实际只打开26训练库，validation/target为空。结论：**通过，无确证阻塞**。

D2合计口径是prepare监督117.2959150400秒加probe监督437.3367794470秒，共554.6326944870秒；字节270,430,806+258,845,607=529,276,413B，均低于1800秒/512MiB。arm训练循环wall或checkpoint字节不等于整个监督阶段，不混用。M投影接近6h仍超限即停；主代理检查两臂loss各52行、optimizer52步、finite和重载，见`probe-verification.json`。
