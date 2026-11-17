# 125：官方26训练库，无三源验证

日期：2026-11-17。用户指令：“不要三源验证了，改成和官方一样的”。

**官方26训练/4测试划分已实现，旧任务已停止，新训练正在运行；完整六训练与最终测试尚未完成。** 使用[125协议](../../实验文档/125-官方26库划分的PRICE元学习实验.md)和[独立配置](../../configs/data_price125_official26.json)，旧123/124及结果保留。

## 当前配方

chembl/legalacts/telstra加入26训练库，validation为空；删除三源LR校准、强E校准和源筛选阶段。测试库imdb/stats/ergastf1/genome未读取，新入口无目标评估命令，测试不参与选参。

共同P使用官方发布权重`/gly/weilin/PRICE-master/results/model_params.pth`，SHA256 `4f889c4df269d1bca5a50dadd83de7c801a607fe84228b6c21fb3371ae053f92`，43,092,427B；Git blob `95ad4f8096cf8520ba2b313269799d8914146ae0`与GitHub发布文件一致。10,709,330参数、真实chembl两条查询与PriceAdapter前向逐位相同，差值0，见[来源验证](official-checkpoint-identity.json)。本轮使用官方预训练成果，不声称独立复现150epoch。

E/M各三流42/43/44、每run2000更新、每更新4任务、support20/query64；完整模型及原生MSE保留，alpha固定1e-6、外Adam2.85e-5，固定取last。alpha的旧校准历史在125披露，当前不再校准。新训练全部从官方P重新开始，不接旧23库或探针权重。

## 隔离与旧任务

分支`experiment/price-meta125-official26-20261117`，base `ad8237e`；新服务器根`/gly/weilin/price-meta-125-official26-20261117`。binding `220400d2d1a8797d3124e084caba4f1d7e1baac5ea953274884899c3206f0e17`的35个执行依赖与本地已审SHA一致，见[binding](binding-initial.json)。

旧123在核对PID/starttime/命令后仅向自有supervisor7151发SIGTERM，由监督器清理worker62831；cleanup验证通过、无residual，两个PID均退出。旧E42完成2000步；旧M42最后持久化301/2000，因用户换协议中止，`parent_signal`不作科学失败，其他旧run未启动。见[停止请求](previous123/user-split-change-stop-request.json)、[M监督](previous123/train-M-42.supervisor.json)、[E监督](previous123/train-E-42.supervisor.json)。不删除或覆盖旧结果。

## 真实计划与D2

三流各8000任务，26库1,300,000行、1,279,218唯一查询、maxima `[12,12,7,10]`；24,000任务均逐项验证S/Q身份不交叠、26库闭包，见[计划验证](plan-verification.json)。

| 项目 | 本轮真实值 |
|---|---:|
| CPU准备 | 117.2959秒／270,430,806B |
| GPU探针 | 437.3368秒／258,845,607B；E/M各52步 |
| D2合计 | 554.6327秒／504.7573MiB，低于30分钟/512MiB |
| 外部整卡显存峰值 | 1.515625GiB，baseline1MiB |
| worker allocator峰值 | 0.951410GiB |
| E/M最大更新耗时 | 1.046667／8.160307秒 |
| 六训练成本投影（含30%余量） | 20.1272小时≤24小时 |
| E/M单run投影（含30%余量） | 0.7559／5.8936小时，限2／6小时 |

两臂checkpoint有限、重载相等，optimizer52步、loss各52行，实际只开26训练库；结束GPU2回到1MiB/0%。成本与显存已独立复算通过，见[验证](probe-verification.json)、[投影](budget_projection.json)、[审查](review.md)。M投影距6h约6.4分钟，后续超限仍停止，不扩大预算。当前投影只覆盖训练和seal，未来目标评估未计入也未授权。

## 运行与接手

北京时间**2026-11-17 17:01:50**在GPU2启动PID **87746**（starttime **121177250**），顺序E42→M42→E43→M43→E44→M44→seal-training。[17:03快照](continuation-progress-observation.json)为E42 **51/2000**、身份存活、无failure。六个last尚未完成，尚无元学习精度收益结论。

串行脚本由逐阶段授权单独绑定SHA，每阶段调用已审运行器、重新空闲预检并扣累计剩余资源，第一次失败即停止、不切卡重试/续训。累计24h、E2h/M6h、RSS24GiB、VRAM12GiB、输出6GiB不变。不要重启脚本；本地JSON是取回时快照，后续只读核对服务器：

```bash
ssh -o BatchMode=yes -o ConnectTimeout=12 -o ConnectionAttempts=1 \
  -o ControlMaster=no -o ControlPath=none myserver \
  'cat /gly/weilin/price-meta-125-official26-20261117/evidence/continuation-status.json'
```

完成后须检查七stage完成、六个2000步last及seal哈希、官方P/26库计划一致；失败保留全部证据，不自动重启。最终四库评估另行冻结与明确授权，不按目标结果选配方。

## 验证

本地全量1886项`OK (skipped=22)`，239.418秒、exit0；Linux/Torch2.1最终130专项`OK`无skip，91.215秒。Linux首轮漏带旧123历史timing测试fixture造成2error，补齐只读文件后复验，初始日志保留。Ruff、格式、diff及绑定检查通过。

独立方案、代码/计划关键路径、七阶段脚本、D2证据审查均无未解决阻塞。脚本成功7stage、第二stage失败只调用前2的模拟通过，模拟不算研究实验。大计划/checkpoint/数据留服务器，不提交Git。Git交付状态以实际Git及回复为准，训练仍在执行。
