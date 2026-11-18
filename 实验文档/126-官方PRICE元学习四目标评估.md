# 126 官方PRICE元学习四目标评估

日期2026-11-18。用户授权：“那就开始四个目标库的评估吧，看看效果怎么样”。本轮只评估125已封存的共同P与E/M三流last，不训练公共权重、不选参、不延长训练，不改125冻结字节及历史输出。

## 1. 冻结来源与范围

训练根`/gly/weilin/price-meta-125-official26-20261117`，binding `220400d2d1a8797d3124e084caba4f1d7e1baac5ea953274884899c3206f0e17`；seal-training/completion SHA256 `299b0cbf63f41950f54d1abc94ac0d095eaf3fa2fce553b681280d457c7159f4`。所有六个last均2000更新，逐run completion/supervisor、官方初始SHA、训练计划SHA、checkpoint及训练状态重新核验后绑定。

P为官方发布权重SHA `4f889c4df269d1bca5a50dadd83de7c801a607fe84228b6c21fb3371ae053f92`；E42/E43/E44与M42/M43/M44按seal取全部六份，禁止best挑选或丢失失败流。P只有一个公共参考，不当成三次独立预训练。PG按相同unique-query身份及标准Q-error现场复算，四份CSV纳入相同独立复算通道，不引用论文或历史all-rows数字。

测试固定顺序imdb/stats/ergastf1/genome，完整unique-query分母分别521/2603/1258/1699（原始行766/2749/1422/2302）。同hash内容冲突立即停止；代表行按(query_hash,id,source_line)稳定确定，保留全部unique，不筛误差/长度/基数。

## 2. 支持身份与矩阵

四库都运行K0；stats/ergastf1/genome运行K5和K20，各5个支持block。2026-11-18只读检查`artifacts/cache/finetune/imdb`不存在；计划构建再次检查并固化路径/存在状态，IMDB的K>0标记`unavailable_missing_independent_support`，不借test补support、不悄悄变成三库评估。若后续补齐IMDB另立补充协议，不改本次固定population。

三库支持仅来自其finetune，先canonical去重并去除所有test hash。实测独立池stats49959、ergastf149705、genome49609。每库`Generator(PCG64(SeedSequence([20261118,126,db_index])))`对独立池无放回选100，顺序分5个20条block，K5为各block前5条；P/E/M及所有流共用完全相同支持和test身份。block间不重叠，但共用test与权重，分数相关，不当五次独立训练。

固定全参数SGD5步、alpha1e-6，原生MSE；适应与推理eval模式，支持梯度开启，query无梯度，临时参数每格从封存checkpoint重置，不写回公共权重，不跨库/格累积。batch128、FP32，无AMP；不扫LR/步数或利用目标分数做选择。alpha此前校准历史在125披露。

P/E/M共7模型，每模型4个K0加3库×2K×5block=34格，共238格，加四库PG K0=242格。缺IMDB K>0不伪造格；任何其他缺格、非有限、身份/SHA/分母错误均invalid并停止，失败查询不删除或裁剪。

## 3. 长度与指标

训练布局为(12,12,7,10)，stats test maxima(8,12,7,10)、finetune maxima(12,14,8,6)来自真实缓存，不能截断或按长度排除支持。布局不含按位置或容量学习的参数，目标评估以逐库train/test/support maxima逐元素最大值作为PriceAdapter固定padding宽度；stats为(12,14,8,10)，其余沿用(12,12,7,10)。中间batch动态宽度不得超此上界，超长拒绝而不截断。所有模型、K和run在同库用同一布局。执行前核对扩展前后state_dict键/形状相同，对七份权重的完整2603条STATS test验证扩展前后输出在FP32容差内；长support实测加载及完整长度检查另列，不能把短查询兼容当成长输入证明。结构兼容性不等于泛化收益。

逐查询保存ID/hash/真实基数/原生输出及解码log预测，完整CSV压缩。标准Q-error为`exp(abs(max(log_pred,0)-log(max(true,1))))`；先完整查询上计算Q50/Q90/Q95/Q99，Q95为numpy线性分位。各格独立从CSV复算，与在线指标对照；再独立复算聚合，不平均batch分位数。

每模型/流/库/K>0先对5block的Q95算术平均qbar，再对数据库取log等权平均L；报告逐库原始值、block波动与三流值。四库K0与三库K20分开：主效应`exp(mean_s(mean_d log(qbar[M,s,d,20])-mean_d log(qbar[E,s,d,20])))`，d只含三个支持可用库。另报M20/P20、M20/M0、M20/P0、M20/E0及M0/P0、M0/E0，负百分比表示Q95降低。四库K0几何聚合另列，禁止与三库K20交叉相除。

开发性判断沿用5%收益/5%K0保护/10%逐库保护语义，但不冒充原123源门或四库fewshot完整验收：M20/E20、M20/P20、M20/M0、M20/P0、M20/E0均≤0.95，E比较与自身适应至少2/3流改善；四库M0/P0和M0/E0≤1.05；逐库Q50/Q95保护≤1.10。同条件收益、代价与负迁移完整报告，未全过标partial/no gain，不按结果再改配方。目标有历史观察，不称新盲测；不外推到任意数据库或数据库端到端加速。

## 4. 执行与验收

新worktree/分支`experiment/price-meta126-targets-20261118`，服务器新根`/gly/weilin/price-meta-126-targets-20261118`，code/inputs/outputs/evidence独立。旧PRICE、125权重与缓存只读。协议原文、执行依赖、目标计划、训练seal与全部权重SHA重新绑定。

先只读身份和布局兼容性检查，单元/拒绝测试、本地全量与Ruff、Linux专项及独立只读审查通过后，选择一张当前空闲GPU。单卡整卡UUID计量、VRAM12GiB、RSS24GiB、累计墙钟3小时、每格120秒、输出1GiB；CPU计划/绑定检查≤600秒。总墙钟与单格限额同时为硬限，任何一项触顶即停止并标invalid，不按部分格宣称完成，不自动重试或加预算，保存首次失败及只清理自有进程。单格上限不作为每格预估耗时。运行期间不连接PostgreSQL、不生成新标签或缓存。

全部242格、CSV独立复算、完整分母及三流权重/支持配对后才写完成与结论。大权重和全量预测留服务器，精简来源、格级指标、独立验证及结果图回仓库。更新99进度，独立结果审查后提交/同步；不把代码通过或启动成功当研究结果。
