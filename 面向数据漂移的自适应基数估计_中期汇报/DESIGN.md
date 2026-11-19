# 中期汇报设计规范

## 全局视觉约定

- 画布：1280×720，16:9；内容页安全边距60px。
- 结构：内容页采用“页面标题—引导句—主体内容—结论/注释—页脚”的固定层级；标题下使用1px贯通细线作为唯一视觉母题。
- 主色：#1E4FA8；藏蓝：#0E3F8C；中蓝：#3D7BD9。
- 辅色：#E8EFF8、#F0F5FC；底色：#FFFFFF、#F7F9FC；线框：#D6DCE5。
- 正文色：标题#1A2230，正文#4A5568，辅助#8B97A8；少量风险强调#D9534F。
- 中文字体：Source Han Sans SC；西文与数字：Aptos。
- 字号：封面主标52px；页面主标34px；卡片标题24px；正文22px；表格19–20px；注释15–16px；页脚14px。
- 视觉素材：不生成装饰图片；使用原生流程图、结构图、数据表格和时间轴，保证证据清晰、风格统一。
- 页面整体保持蓝白、浅灰、低饱和，无渐变、无装饰图标、无大段深色底。

## 逐页设计

### P1 封面
- title: 面向数据漂移的自适应基数估计方法研究与实现
- type: cover
- role: hero
- rhythm: peak
- layout: 顶部学校与汇报类型，中央论文题目，底部汇报信息
- visual: 以细线和少量浅蓝几何色块构成学术封面，无摄影
- density: 约90字 / 图片0张 / 留白约45%
- anti_pattern: 禁止多栏信息卡、装饰图标和复杂背景
- description: 展示论文题目、东北大学软件学院、软件工程硕士中期汇报、汇报人魏林、导师王蓓蕾与副导师郭朝鹏、日期2026年11月21日。

### P2 数据变化带来的基数估计挑战
- type: content
- role: supporting
- rhythm: valley
- layout: 顶部四步流程，中部空间/时间漂移双栏，底部核心判断
- visual: 原生流程图与两类漂移对照
- density: 约260字 / 图片0张 / 留白约18%
- anti_pattern: 禁止将两类漂移拆成多组散落小卡，禁止长篇文献综述
- description: 说明基数估计对执行计划的作用，界定跨库空间漂移与随增删改发生的时间漂移，得出核心问题是低成本适应分布变化。

### P3 研究问题与总体目标
- type: content
- role: supporting
- rhythm: valley
- layout: 双栏研究问题，底部当前阶段定位
- visual: 两个编号研究模块与目标条
- density: 约300字 / 图片0张 / 留白约15%
- anti_pattern: 禁止堆砌开题背景，禁止将后续时间漂移写成已完成
- description: 分别阐述新数据库少样本适应和数据持续变化时的反馈校准，明确当前重点是轻量编码与元学习初始化。

### P4 总体技术路线与中期进展
- type: content
- role: supporting
- rhythm: valley
- layout: 四步技术路线、状态表与60%进度锚点
- visual: 原生技术路线图与完成状态表
- density: 约230字 / 图片0张 / 留白约12%
- anti_pattern: 禁止仅列计划而无完成状态，禁止把60%拆成未经材料支持的子模块百分比
- description: 展示多源数据库、离线训练、目标域适应、独立测试的路线；列出数据协议、模型训练、跨库评估、方法改进和时间漂移状态，补充总体完成约60%。

### P5 轻量查询编码与模型设计
- type: content
- role: supporting
- rhythm: valley
- layout: 左侧四组输入与编码流程，右侧参数和训练成本数据
- visual: 原生模型结构图与数据锚点
- density: 约210字 / 图片0张 / 留白约12%
- anti_pattern: 禁止直接粘贴模糊报告截图，禁止只放参数量而不解释意义
- description: 说明连接直方图、fanout、表、过滤条件四组变长输入，经独立Token MLP和掩码池化形成定长表示；展示132545参数、约PRICE的1/80及训练成本。

### P6 实验设置与评价方法
- type: content
- role: supporting
- rhythm: valley
- layout: 三项数据库划分KPI，底部基线、样本预算与隔离协议
- visual: 数据划分卡片和实验协议表
- density: 约230字 / 图片0张 / 留白约12%
- anti_pattern: 禁止罗列全部数据库名称，禁止混淆23源训练库与PRICE的26源库口径
- description: 展示23个源训练、3个源验证、4个目标测试库；列明PG、PRICE、ERM、FOMAML，K=0/5/10/20/50/100，unique-query Q95与查询隔离协议。

### P7 跨数据库零样本结果
- type: content
- role: hero
- rhythm: peak
- layout: 左侧结果表，右侧两项结论与实验口径
- visual: 零样本Q95对比表为证据主体
- density: 约180字及16组数值 / 图片0张 / 留白约10%
- anti_pattern: 禁止宣称超过PRICE，禁止用被Genome大数值主导的柱状图替代精确表格
- description: 对比PG、PRICE、ERM-sum在四库的Q95；指出ERM-sum在IMDB、Genome、ErgastF1上优于PG，但四库仍高于PRICE，尾部估计精度需继续提升。

### P8 阶段性总结与待解决问题
- type: content
- role: supporting
- rhythm: valley
- layout: 已完成工作与待解决问题双栏对照
- visual: 两项三行清单与底部研究判断
- density: 约280字 / 图片0张 / 留白约12%
- anti_pattern: 禁止只讲成绩回避负迁移，禁止虚构少样本实验数值
- description: 总结数据评估流程、轻量编码器与FOMAML、结构对照与误差分析；说明Genome改善、STATS负迁移、元学习收益依赖设置、结构改进仍需验证。

### P9 下一阶段计划与预期成果
- type: content
- role: supporting
- rhythm: valley
- layout: 五阶段时间轴与底部成果条
- visual: 2026.11—2027.04工作计划表
- density: 约260字 / 图片0张 / 留白约10%
- anti_pattern: 禁止把算法、实验和论文任务混成无时间顺序的长段落
- description: 安排源侧对照、少样本实验、时序基线、反馈校准、消融、全文整合和送审准备；预期学位论文、可复现实验、对比消融图表并力争发表1–2篇论文。

### P10 总结与致谢
- type: ending
- role: hero
- rhythm: peak
- layout: 左侧三项阶段结论，右侧致谢
- visual: 简洁总结清单与致谢大字
- density: 约170字 / 图片0张 / 留白约35%
- anti_pattern: 禁止再引入新实验数据，禁止烟花、印章和装饰图片
- description: 归纳完成统一实验链路、轻量模型与零样本评估、明确差距和后续路径；以“请各位老师批评指正”结束。
