# 当前模型精度审查与改进建议

日期：2026-10-18（Asia/Shanghai）。范围：代码正确性、表示限制、已有精度证据和新方案可行性。
代码基准：`3dc0b9e452c2d3db5bc43533f8d24cb94d021d7d`。
隔离分支：`docs/model-accuracy-plan-review-20261018`。
本轮没有修改模型实现、访问服务器、执行真实训练/推理或读取目标数据。
CPU合成测试中的少量优化步不是正式训练结果。

## 1. 结论与发现分级

**当前sum/FOMAML主路径：本轮未发现已复现、可以直接解释精度差的P1/P2实现缺陷。**
这不等于证明模型没有bug，也不等于输入充分或方法适合任务。核心专项86项通过，
独立只读检查确认梯度、episode隔离和实际checkpoint路径未出现先前担心的断链。

| 编号 | 类型 | 发现 | 对当前精度的解释强度 |
| --- | --- | --- | --- |
| F1 | P2，旧变体实现问题 | attn_pair的绝对join编号embedding破坏完整pair重排不变性 | 影响旧attn_pair，不解释当前sum |
| L1 | 已知表示限制 | sum没有显式跨组表/列/join关系；边际统计不保证包含条件相关性 | 是能力限制，不是已定位的真实误差主因 |
| E1 | 已证实实验现象 | support MSE改善与query Q95改善脱节，并存在整体/low4取舍 | 是当前最直接的改进依据，机制仍需受控实验 |
| H1 | 方法假设 | mean SmoothL1、统一适应更新与尾部目标/异质查询之间不完全匹配 | 合理假设，不是因果结论或代码bug |
| D1 | 报告范围错误 | 状态页99,777/107.3倍对应set，不是当前sum64 | 影响模型规模叙述，不影响预测数值 |

## 2. F1：旧attn_pair的顺序问题

代码：[model.py](../spatial_meta_ce/model.py) 第62-67行以 `arange(n)//2` 给fanout的
每个完整pair加不同embedding。将同一组pair重新排序，会改变某个pair得到的embedding。
它编码了任意出现位置，而不仅是“两个方向属于同一join”的关系。

独立代理用随机权重合成输入核验，主代理另行执行同一反例复核：只反转完整pair，attn_pair输出差
`0.004822134971618652`，同配置sum为0。未用真实checkpoint，不能推算历史Q95受影响多少。
现有 [test_arch.py](../tests/test_arch.py) 的小读出权重/容差下用例通过，不能保证结构性不变性。

如果将来恢复该变体，应先用可观测的非零读出、多个固定输入做严格pair重排测试，
再用共享pair关系编码替代绝对join位置；本轮只记录，不改实现、不重跑旧实验。
当前sum不走 `_mix()`，不能据此宣称找到了主模型效果差的根因。

## 3. 已排除的常见误诊

| 怀疑 | 实际代码/验证 | 判断 |
| --- | --- | --- |
| FOMAML把梯度完全detach了 | meta.py:57的create_graph=False只截断support梯度的导数；:61保留p-lr*g的初始化通路 | 符合一阶近似，不是完整MAML，但命名准确 |
| head-only使encoder永远不训练 | meta.py:41只限制inner；train.py:185/239的outer对全部参数更新 | 编码器仍获得query梯度 |
| 零末层导致死网络 | model.py:54只将末层置零；support适应后fast末层可非零 | 当次outer即可有encoder信号；合成验证成立 |
| 每个episode沿用上一个适应结果 | meta.py:50重新取共享参数，更新非in-place；部署detach，runner核对state digest | 本轮未发现污染 |
| 使用错误的best checkpoint | run_meta65_screen.py:424-430及run_meta65_holdout.py:102明确载入last.pt | 4000步/100监控=40次，patience41，当前正式路径不早停 |
| PG未接入或重复加锚 | model.py:138-142同时将PG作为head输入和一次输出锚 | 设计如此，无重复锚定；S4已测去锚 |
| 训练评估使用不同变换 | data.py:45/59/61共用fanout log1p和log(max(card,1))；checkpoint验证transform版本 | 本轮未发现双重变换 |
| sum先把每个直方图平均掉 | token先经过MLP，再按组池化 | 保留token内部40-bin输入，不能写成丢掉全部分布 |

独立数值检查：标量w=2、MSE、lr=.1、3步适应，fast w约1.512，FOMAML gradient约1.024，
原w仍为2；完整MAML gradient约.524288但forward loss相同。二阶与一阶差异要比较meta-gradient，
不能比较同一support更新后的forward loss来诊断。

## 4. 为什么效果不理想：已知事实与未知原因

### 4.1 support拟合和query泛化不是同一个目标

Gate 1使用相同R初始化、K20、五run、三源验证库。以下是已有真实结果，本轮没有重新训练。
本对话上一轮从本地15份原始NPZ独立复算60个cell，结果与归档一致；本轮复核其文档和来源。

| 算子 | 4+ Q95 macro | low4 Q95 macro | 解释 |
| --- | ---: | ---: | --- |
| none | 41.440836 | 142.380057 | 共同初始化参照 |
| SmoothL1-last3 | 41.035807 | 146.970635 | 整体改善0.98%，low4恶化3.22% |
| MSE-last3 | 39.323504 | 171.027300 | 整体改善5.11%，low4恶化20.12% |
| proximal ridge | 99.230620 | 3124.871537 | support MSE的15/15改善没有转化为泛化 |

这里low4=`true<100且表数>=4`，只用于评估。实际推理不知道true，也不知道PG究竟高估还是低估，
不能按这些事后分桶选择模型或修正方向。ridge与梯度MSE还同时改变求解器和近端正则，不能单独归因于求解器。

R_head3已适应的历史主值37.8033也优于上述MSE-last3的39.3235；不能把后者相对none改善
包装成已超越既有方法。新21库实验必须使用同次21库基线，不能直接拿23库历史分数作门槛。

### 4.2 方法正在不同群体之间做取舍

80号T_all3相对R_head3在low4三库均改善，却使主4+ macro从37.8033恶化到42.4364。
S6主macro改善9.42%，但legalacts的4+ Q95恶化15.70%；S7还观察到部分已见训练查询退化。
这些不是“只有未见库才出问题”或“只需更多训练”的一致图景。

训练 [meta.py](../spatial_meta_ce/meta.py):8-25 最小化mean SmoothL1(log prediction, log label)，
而主验收是按库聚合的4+ Q95。大残差下SmoothL1对单点的梯度幅值饱和；它不直接优化第95分位数。
这种目标差异真实存在，但S5/S6反证说明简单加权或追尾不必然有效，不能把换loss当已确定修复。
同样，mean/max/sum的尺度、梯度clip、随机support覆盖都是待测变量，不可据负结果宣布其为主因。

### 4.3 输入不保证足以辨别所有查询条件

sum逐组聚合，不接收filter-column到table、fanout到join-column等映射。
但 [12号映射实验](../实验文档/12-映射消融正式协议.md) 已测sum_map：三seed主值均值
sum43.3766、真实映射51.6143、打乱映射59.6171，说明映射含信号但旧实现未胜过sum。
不得把重复接同一个映射或普通attention重新包装成未测试方向。

[features_tool.py](../vendor/price_features/features_tool.py):137/150/375/407提供单列histogram、
未按过滤条件重算的fanout、单列过滤选择率和AVI/MINSEL/EBO。它们不显式包含过滤列与join key的条件联合分布。

一个合成反例（不是实际数据结论）：R的两行分别为`[(k=1,x=1),(2,0)]`或`[(1,0),(2,1)]`，
S有100行key=1和1行key=2。两种R的k/x边际、join图和未过滤join统计相同，
`R JOIN S WHERE x=1`却分别返回100和1行。完整join图也不能凭空恢复缺失的条件相关性。
此例没有证明真实PG标量相同、真实缓存存在相同特征不同标签的碰撞，或该机制解释当前误差比例。

P2已认证冻结编码自洽，不等于真实SQL语义、同版本源表和真实选择率已认证。
未完成的数据来源门槛仍保持blocked，不能用图模型或额外训练绕过。

### 4.4 泛化证据有限，不等于训练本身错误

三验证库多轮参与开发；单seed/五support run不能作为五次独立模型训练。
81号的hepatitis/sakila曾被历史checkpoint训练和S7描述性分析覆盖，仅在新臂训练中隔离，
不应称完全干净holdout。50-cell仍为归档的formal_not_started；本轮未现场检查GPU/进程。

## 5. D1：状态页的模型规模需按变体区分

原主目录状态页把99,777参数、0.40MB、比PRICE小107.3倍与当前sum路线放在同一叙述中。
本轮CPU构造核对：set=99,777，sum64=132,545；sum的纯FP32参数为530,180bytes。
以页面PRICE的10,709,330参数计算，sum约小80.80倍。head分别为74,753和107,521参数，
last均为65；head-only不等于“只更新65参数”。原页面属于其他工作，本轮未改。

## 6. 新方案的先验风险与修改理由

[86号方案](../实验文档/86-低维元适应精度改进方案草案.md) 推荐学习四维任务适应子空间，
保留PG与sum，不是再接一个frozen ridge。它只是假设，不保证超过现有head3。

方案审查中先处理两项风险：

- 双零初始化死分支：先对没有QR约束的线性残差分支检查，直接令P=0/u=0时，三步后u范数和P梯度均为0；非零正交P初始化时，
  合成样例u范数`0.0012360432`、P梯度范数`0.0008447921`。这只检查直接线性基的梯度通路，非精度证据。
- 只比R与新臂无法区分低维约束与显式学习A，因此增加固定A的F对照；只有L是预先指定候选，
  不在失败后改选F。F仍通过encoder学习表示，不是完全不学习的模型。

本轮没有实现新模型。未来还需独立CPU契约测试、完整回归、真实资源profile、新冻结协议和一次性授权。
该基础探针没有验证QR反向；拟议方法的A=0违反满列秩前提，必须拒绝，不能用QR输出冒充有效基。
随后补做固定fast-u的FP64 QR小张量检查：seed1729、A为65x4、support20/query64、三步1e-3，
u范数`0.0005270396`、A梯度范数`4.668343e-5`，正交最大误差`4.440892e-16`。
A[17,2]的解析梯度`4.027640e-9`与固定u的中心有限差分`4.030110e-9`相差`2.469253e-12`，exit0。
这只验证一个非退化合成点的固定u梯度，不认证完整展开导数、FP32训练、QR符号/近秩退化边界或GPU一致性。

## 7. 本轮验证、独立审查与边界

实际CPU专项命令，在本报告所在worktree执行：

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' \
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=tests /opt/anaconda3/bin/python -B -m unittest \
test_core test_arch test_meta65_scope_gradients test_data test_mapping_model test_s4_model test_paired_metrics -v
```

结果：exit0，`Ran 86 tests in 4.015s / OK`；本机Torch2.3.0、NumPy1.26.4，非服务器环境认证。
因本机沙箱已有OpenMP共享内存限制，本轮经审批在沙箱外运行；未伪造沙箱通过状态。
另执行CPU参数计数和纯合成梯度/关系计数探针，均exit0。未运行全量回归或全仓lint，未改生产代码。

两名只读代理独立检查训练正确性与表示信息；第三名未参与编写的只读代理审查实际三文件diff、
相关代码、旧协议及验证证据。第一轮结论为无阻塞/无必修重要发现，可以交付草案。
审查曾把inner head误解为outer head，主代理与梯度审查代理核对train.py:185/239及65号协议后，
审查方正式撤回该误报；没有据此改坏正确的R_head3表格。
采纳的非阻塞补充为：非零梯度限定非退化合成样例；训练前检查V/run的low4计数。
最终补充版本已完成独立只读复核，无新增阻塞或必修重要发现；通过仅表示草案审查，
不等于新模型已实现、方案已冻结或执行就绪。

未解决事项：真实checkpoint上的clip饱和与梯度冲突缺少新测量；真实条件相关性未认证；
旧attn_pair缺陷未修；新方案收益未知。主目录既有未提交文件未触碰。

### 可复现的F1合成反例

在本worktree根目录、前述单线程禁CUDA环境运行以下Python；不加载真实数据或checkpoint。
seed和随机调用顺序固定，输出即第2节数值，两个断言均通过。

```python
import copy
import torch
from spatial_meta_ce.data import batch_samples
from spatial_meta_ce.model import SetEstimator

torch.set_num_threads(1)
torch.manual_seed(7)
s = {"groups": [torch.rand(3, 40), torch.rand(6, 40),
                torch.rand(3, 4), torch.rand(3, 43)],
     "true": 10.0, "pg": 2.0, "id": "synthetic", "hash": "synthetic"}
t = copy.deepcopy(s)
t["groups"][1] = s["groups"][1].reshape(3, 2, 40).flip(0).reshape(6, 40)
m = SetEstimator(arch="attn_pair").eval()
torch.nn.init.normal_(m.head[-1].weight, std=1.0)
torch.nn.init.normal_(m.join_embedding.weight, std=1.0)
p, q = m(batch_samples([s])), m(batch_samples([t]))
print("attn_pair delta", float((p - q).abs()))
assert not torch.allclose(p, q, atol=1e-6, rtol=1e-6)
sm = SetEstimator(arch="sum").eval()
torch.nn.init.normal_(sm.head[-1].weight, std=1.0)
sp, sq = sm(batch_samples([s])), sm(batch_samples([t]))
print("sum delta", float((sp - sq).abs()))
assert float((sp - sq).abs()) < 1e-5
```
