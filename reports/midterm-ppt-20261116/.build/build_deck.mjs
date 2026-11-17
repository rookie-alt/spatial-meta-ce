import fs from 'node:fs/promises';
import path from 'node:path';
import { execFileSync } from 'node:child_process';
import { Presentation, PresentationFile } from '/Users/williamlin/.cache/codex-runtimes/codex-runtime-install-3jSuSH/payload/codex-primary-runtime/dependencies/node/node_modules/@oai/artifact-tool/dist/artifact_tool.mjs';

const ROOT = '/Users/williamlin/Downloads/13/spatial-meta-ce';
const WORK = path.join(ROOT, 'reports/midterm-ppt-20261116');
const BUILD = path.join(WORK, '.build');
const OUT = path.join(WORK, 'output');
const RENDER = path.join(BUILD, 'slides');
const FONT = 'PingFang SC';
const C = { navy:'#153653', ink:'#233B4D', teal:'#007E84', muted:'#5F7484', light:'#F1F6F7', rule:'#D8E3E8', white:'#FFFFFF', amber:'#AC6B2E', paleTeal:'#E9F4F3' };
await Promise.all([fs.mkdir(OUT, {recursive:true}), fs.mkdir(RENDER, {recursive:true})]);
const p = Presentation.create({slideSize:{width:1280,height:720}});
const sourceSummary = path.join(ROOT,'实验文档/results/p2-2-target-3seed-20261023/summary.json');
const data = JSON.parse(await fs.readFile(sourceSummary,'utf8'));
const reportPath = '/Users/williamlin/Downloads/东北大学学位论文活实践成果中期考核报告/东北大学硕士学位论文或实践成果中期考核报告.doc';
const reportText = execFileSync('/usr/bin/textutil',['-convert','txt','-stdout',reportPath],{encoding:'utf8'});
const person = reportText.match(/学\s*生\s*姓\s*名[：:]\s*([^\n\r]+?)(?:\s{2,}|[\n\r])/u)?.[1]?.trim() || '';
const advisor = reportText.match(/导\s*师\s*姓\s*名[：:]\s*([^\n\r]+?)(?:\s{2,}|[\n\r])/u)?.[1]?.trim() || '';
if (!person || !advisor) throw new Error('Could not extract report name/advisor');
const notes = [];
const layoutBoxes = [];

function text(s, str, x,y,w,h, size=26, color=C.ink, bold=false, opts={}) {
  const name=opts.name || `text-${s.id}-${s.shapes.items.length}`;
  const sh=s.shapes.add({geometry:'textbox',name,position:{left:x,top:y,width:w,height:h},fill:'none',line:{fill:'none',width:0}});
  sh.text=str;
  sh.text.style={typeface:FONT,fontSize:size,color,bold,alignment:opts.align||'left',verticalAlignment:opts.valign||'top',autoFit:'none',wrap:'square',insets:0,...opts.style};
  layoutBoxes.push({slide:p.slides.items.length,name,text:str,x,y,w,h,size});
  return sh;
}
function line(s,x1,y1,x2,y2,color=C.rule,width=1.2,dash='solid') {
  return s.shapes.add({geometry:'line',position:{left:x1,top:y1,width:x2-x1,height:y2-y1},fill:'none',line:{fill:color,width,style:dash}});
}
function node(s,str,x,y,w,h,{fill=C.light,color=C.navy,size=24,stroke=C.rule,bold=true}={}) {
  const sh=s.shapes.add({geometry:'rect',position:{left:x,top:y,width:w,height:h},fill,line:{fill:stroke,width:1}});
  sh.text=str; sh.text.style={typeface:FONT,fontSize:size,bold,color,alignment:'center',verticalAlignment:'middle',insets:12,autoFit:'none',wrap:'square'};
  return sh;
}
function connect(s,a,b,{color=C.teal,from='right',to='left',kind='straight',dash='solid'}={}) {
  return s.shapes.connect(a,b,{kind,fromSide:from,toSide:to,line:{fill:color,width:2,style:dash},tail:{type:'arrow',width:'sm',length:'sm'}});
}
function base(title,section,page,subtitle='') {
  const s=p.slides.add(); s.background.fill=C.white;
  text(s,section,64,30,1000,26,18,C.teal,true);
  text(s,title,64,78,1152,62,44,C.navy,true,{name:`title-${page}`});
  if(subtitle) text(s,subtitle,64,143,1152,38,23,C.muted);
  line(s,64,674,1216,674);
  text(s,'东北大学 · 软件工程硕士学位论文中期汇报',64,687,1000,22,14,C.muted);
  text(s,String(page).padStart(2,'0'),1150,683,66,28,19,C.muted,false,{align:'right'});
  return s;
}
function note(s,spoken,sources=[]) {
  s.speakerNotes.text=`讲稿\n${spoken}\n\n来源\n${sources.join('\n')}`;
  notes.push({page:p.slides.items.length,spoken,sources});
}
function bullet(s,title,body,x,y,w,{size=26,bodySize=24}={}) {
  text(s,title,x,y,w,40,size,C.navy,true);
  text(s,body,x,y+48,w,80,bodySize,C.muted);
}
function nativeTable(s,values,x,y,w,h,widths,{font=23,header=22}={}) {
  const t=s.tables.add({rows:values.length,columns:values[0].length,left:x,top:y,width:w,height:h,columnWidths:widths,values});
  t.styleOptions={headerRow:true,bandedRows:false};
  t.borders.assign({fill:C.rule,width:0.8,style:'solid'});
  for(let r=0;r<values.length;r++) for(let c=0;c<values[0].length;c++) {
    const cell=t.getCell(r,c);
    cell.fill=r===0?C.navy:(r%2?C.light:C.white);
    cell.text.style={typeface:FONT,fontSize:r===0?header:font,bold:r===0,color:r===0?C.white:C.ink,alignment:c===0?'left':'center',verticalAlignment:'middle',autoFit:'none',wrap:'square'};
  }
  return t;
}

// 1. Minimal cover.
{
 const s=p.slides.add(); s.background.fill=C.navy;
 text(s,'东北大学  软件学院',72,54,1000,38,25,'#C4D7E0');
 text(s,'硕士学位论文中期汇报',72,148,1000,40,26,'#81CBC6');
 text(s,'面向数据漂移的\n自适应基数估计',72,226,1120,162,64,C.white,true,{name:'cover-title'});
 text(s,'方法研究与实现',76,410,1000,50,35,'#C4D7E0');
 line(s,76,507,1200,507,'#57778A',1);
 text(s,`汇报人  ${person}       指导教师  ${advisor}`,76,542,1100,42,26,C.white);
 text(s,'软件工程       2026 年 11 月 21 日',76,596,1100,35,22,'#C4D7E0');
 note(s,'各位老师好，我的论文题目是《面向数据漂移的自适应基数估计方法研究与实现》。本次汇报分为研究背景、当前进度和未来规划三个部分。现阶段的实验重点是空间漂移条件下的跨数据库少样本估计。',[reportPath]);
}

// 2. Background and an editable explanatory pipeline.
{
 const s=base('数据变化带来的基数估计挑战','01  研究背景',2,'基数估计预测查询中间结果的行数，为执行计划选择提供依据');
 const labels=['SQL 查询','基数估计','执行计划选择','数据库执行'];
 const ns=labels.map((v,i)=>node(s,v,64+i*298,210,258,68,{fill:i===1?C.paleTeal:C.light,color:i===1?C.teal:C.navy}));
 for(let i=0;i<3;i++) connect(s,ns[i],ns[i+1]);
 text(s,'估计偏差可能影响连接顺序与连接算法，进而影响查询执行效率',64,300,1152,40,24,C.muted);
 line(s,64,368,1216,368);
 text(s,'空间漂移',64,402,520,45,31,C.teal,true);
 text(s,'数据库之间发生变化',64,456,520,37,25,C.navy,true);
 text(s,'表结构、属性分布与查询模式不同\n直接迁移模型可能出现精度下降',64,507,520,100,25,C.ink);
 text(s,'时间漂移',680,402,520,45,31,C.teal,true);
 text(s,'同一数据库随时间变化',680,456,520,37,25,C.navy,true);
 text(s,'数据增删改与业务负载持续变化\n历史统计信息与模型可能逐渐过期',680,507,520,100,25,C.ink);
 note(s,'基数估计是查询优化的基础环节。优化器根据中间结果规模选择连接顺序和算法，估计偏差可能导致不合适的计划。我的研究关注两类变化：空间漂移是数据库之间的结构与分布差异，时间漂移是同一数据库在运行过程中的持续变化。',[path.join(ROOT,'memory/spatial-meta-ce-project.md'),path.join(ROOT,'docs/protocol.md'),reportPath]);
}

// 3. Research questions.
{
 const s=base('研究问题与总体目标','01  研究背景',3);
 text(s,'01',64,188,90,74,52,C.teal,true);
 text(s,'新数据库上的少样本适应',176,190,1020,44,31,C.navy,true);
 text(s,'利用多个源数据库的训练经验，在有限标注查询下适应新数据库，\n降低重新训练的数据需求与计算开销。',176,246,1010,86,27,C.ink);
 line(s,176,355,1216,355);
 text(s,'02',64,390,90,74,52,C.teal,true);
 text(s,'数据持续变化时的反馈校准',176,392,1020,44,31,C.navy,true);
 text(s,'利用近期查询及其真实基数，研究当前分布下的估计修正，\n平衡反馈数量、校准效果与推理成本。',176,448,1010,86,27,C.ink);
 text(s,'当前重点：跨数据库少样本估计',64,591,1152,47,32,C.teal,true);
 note(s,'第一个问题是如何把多个源数据库中学到的经验用于新数据库，并减少所需标注查询。第二个问题是如何利用数据库运行中不断积累的查询反馈。当前先完成跨数据库方法和实验基础，时间漂移校准作为后续研究。目标是改善精度，同时控制标注与适应开销。',[reportPath,path.join(ROOT,'memory/spatial-meta-ce-project.md')]);
}

// 4. Route and progress.
{
 const s=base('总体技术路线与完成情况','02  当前进度',4);
 const ns=[node(s,'多源数据库\n查询与统计特征',64,185,250,92),node(s,'离线训练\n学习模型参数',362,185,250,92),node(s,'少量支持查询\n目标域适应',660,185,250,92),node(s,'独立测试查询\n评估精度与开销',958,185,258,92)];
 for(let i=0;i<3;i++) connect(s,ns[i],ns[i+1]);
 const rows=[['数据与协议','数据库划分、查询去重、统一评价口径','已建立'],['模型与训练','轻量编码器、ERM 与 FOMAML','已实现'],['跨库评估','四库零样本、三库少样本对照','已有结果'],['方法改进','结构对照、适应分析与误差诊断','持续完善'],['时间漂移','上下文学习方案设计，无实现与结果','方案阶段']];
 rows.forEach((r,i)=>{ const y=322+i*61; text(s,r[0],64,y,220,34,25,C.navy,true); text(s,r[1],308,y,704,34,24,C.ink); text(s,r[2],1060,y,156,34,24,i<3?C.teal:C.amber,true,{align:'right'}); if(i<4)line(s,64,y+44,1216,y+44); });
 note(s,'目前已经建立从数据处理、模型训练到独立评估的完整流程。模型方面实现了普通监督训练与FOMAML，评估方面已有四个目标库的零样本结果和三个目标库的少样本结果。现阶段正在完善精度与适应效果，时间漂移部分仍停留在方案设计，尚无实现与精度结果。',[path.join(ROOT,'实验文档/14-离线实验统一对照与论文结论.md'),path.join(ROOT,'实验文档/108-四目标库K0三seed补测.md'),reportPath]);
}

// 5. Editable model diagram and adaptation mechanism.
{
 const s=base('轻量查询编码与少样本适应','02  当前进度',5);
 text(s,'查询表示与估计模型',64,176,830,40,28,C.navy,true);
 const fspec=['连接列直方图','Fanout 特征','表级统计','过滤条件特征'];
 const a=fspec.map((v,i)=>node(s,v,64,228+i*57,220,46,{size:23,fill:C.light}));
 const enc=node(s,'分组 MLP 编码\n掩码池化\nmean / max / sum',366,250,300,176,{fill:C.paleTeal,color:C.teal,size:25});
 a.forEach(n=>connect(s,n,enc,{kind:'elbow'}));
 const head=node(s,'回归头\n预测修正量',736,282,180,110,{fill:C.light,size:25});
 connect(s,enc,head);
 text(s,'132,545',974,244,242,68,46,C.teal,true);
 text(s,'模型参数',974,318,242,36,24,C.muted);
 text(s,'约为 PRICE 的 1/80',974,365,242,70,23,C.navy);
 text(s,'预测对数基数 = PG 对数估计 + 修正量',64,466,1152,48,29,C.navy,true);
 line(s,64,530,1216,530);
 text(s,'FOMAML',64,553,204,36,27,C.teal,true);
 const f1=text(s,'支持集梯度更新',286,551,270,40,26,C.ink);
 const f2=text(s,'查询集计算误差',608,551,270,40,26,C.ink);
 const f3=text(s,'更新共享初始化',930,551,286,40,26,C.ink);
 connect(s,f1,f2); connect(s,f2,f3);
 text(s,'每个任务均从共享初始化出发；同结构 ERM 不进行训练内循环',64,614,1152,33,22,C.muted);
 note(s,'模型将查询分为连接列直方图、fanout、表级统计和过滤条件四组特征，各组使用独立的MLP。sum架构包含均值、最大值和求和三种掩码池化，再输出对PG对数估计的修正量。该版本参数量为132,545，PRICE为10,709,330，比例约1/80.8，这个比较只描述参数数量。FOMAML在每个源数据库任务的支持集上做梯度更新，用适应后的查询集误差更新共享初始化；ERM使用同结构、但不做内循环。评估适应的每个episode独立从初始化开始。',[path.join(ROOT,'spatial_meta_ce/model.py'),path.join(ROOT,'spatial_meta_ce/meta.py'),path.join(ROOT,'实验文档/14-离线实验统一对照与论文结论.md')]);
}

// 6. Experimental settings.
{
 const s=base('实验设置与评价方法','02  当前进度',6,'PRICE 公开数据：SQL 查询、真实基数、PG 估计与统计特征');
 const cols=[{x:64,n:'23',title:'源训练数据库',detail:'训练模型与共享初始化'},{x:466,n:'3',title:'源验证数据库',detail:'Chembl / Legalacts / Telstra'},{x:868,n:'4',title:'目标测试数据库',detail:'IMDB / STATS / Genome / ErgastF1'}];
 cols.forEach(({x,n,title,detail})=>{text(s,n,x,198,330,94,68,C.teal,true);text(s,title,x,304,348,40,28,C.navy,true);text(s,detail,x,357,348,68,22,C.muted);});
 line(s,64,434,1216,434);
 text(s,'对照与预算',64,466,220,40,27,C.navy,true);
 text(s,'PG、PRICE 参考、轻量 ERM、轻量 FOMAML\nK = 0 / 5 / 10 / 20 / 50 / 100',302,466,914,87,26,C.ink);
 text(s,'主评价指标',64,568,220,40,27,C.navy,true);
 text(s,'unique-query Q95，越低越好；兼顾推理与适应开销',302,568,914,45,25,C.ink);
 text(s,'目标库不参与元训练和策略选择；支持集与测试集按查询身份隔离',64,630,1152,28,19,C.muted);
 note(s,'实验使用23个源训练库、3个源验证库和4个目标测试库。源验证库用于配置与策略选择，目标库只用于冻结方案后的评估；目标适应允许读取规定支持集标签，但目标测试结果不用于选型。K表示带标签支持查询数，K=0是零样本。目前四库零样本评估已经完成，少样本覆盖STATS、Genome和ErgastF1，IMDB完整少样本结果仍缺失。主指标为唯一查询的Q95，即Q-error分布的95分位。PRICE官方预训练权重使用26个源库，是补充参考，不能解释为完全匹配训练条件的因果对照。',[path.join(ROOT,'configs/data.json'),path.join(ROOT,'docs/protocol.md'),path.join(ROOT,'实验文档/14-离线实验统一对照与论文结论.md')]);
}

// 7. Native evidence table and editable chart, both based on original JSON.
{
 const s=base('跨数据库零样本结果','02  当前进度',7,'轻量 ERM-sum，K = 0；全量 unique-query Q95，越低越好');
 const keys=['imdb','stats','genome','ergastf1'];
 const names=['IMDB','STATS','Genome','ErgastF1'];
 const num=x=>x.toFixed(2);
 const vals=[['数据库','PG','PRICE','ERM-sum'],...keys.map((k,i)=>{const r=data.databases[k];return [names[i],num(r.pg.q95.mean),num(r.price_k0.q95.mean),`${num(r.ours_k0.q95.mean)} ± ${num(r.ours_k0.q95.std)}`];})];
 nativeTable(s,vals,64,224,682,270,[140,120,120,302],{font:22,header:23});
 s.charts.add('bar',{
  position:{left:784,top:190,width:432,height:308},categories:names,
  title:'Q95 / PG（PG = 1）',titleTextStyle:{typeface:FONT,fontSize:23,bold:true,fill:C.navy},
  series:[{name:'ERM-sum',values:keys.map(k=>Number((data.databases[k].ours_k0.q95.mean/data.databases[k].pg.q95.mean).toFixed(6))),fill:C.navy},{name:'PRICE',values:keys.map(k=>Number((data.databases[k].price_k0.q95.mean/data.databases[k].pg.q95.mean).toFixed(6))),fill:C.teal}],
  barOptions:{direction:'column',grouping:'clustered',gapWidth:90},
  hasLegend:true,legend:{position:'bottom',textStyle:{typeface:FONT,fontSize:17,fill:C.muted}},
  xAxis:{textStyle:{typeface:FONT,fontSize:17,fill:C.muted},line:{fill:C.rule,width:1}},
  yAxis:{min:0,max:1.2,majorUnit:0.4,numberFormatCode:'0.0',textStyle:{typeface:FONT,fontSize:16,fill:C.muted},majorGridlines:{fill:C.rule,width:1}},
  dataLabels:{showValue:false},chartFill:C.white,plotAreaFill:C.white,
 });
 text(s,'IMDB、Genome、ErgastF1 低于 PG；四库仍高于 PRICE',64,529,1152,48,29,C.navy,true);
 text(s,'尾部估计精度仍有提升空间',64,584,1152,42,28,C.teal,true);
 text(s,'ERM-sum：三次训练均值 ± 标准差。PRICE：26 源库预训练参考；本模型：23 源库训练。',64,637,1152,25,17,C.muted);
 note(s,'这张表展示固定历史轻量参考模型ERM-sum的零样本结果，不代表最新PRICE式编码器，也不证明元学习收益。数值从p2-2-target-3seed-20261023/summary.json逐库读取。相较PG，IMDB、Genome和ErgastF1三个库的Q95更低，STATS更高。与PRICE参考相比，四库仍存在尾部精度差距。右侧图为逐库Q95除以该库PG值，1表示与PG相同，越低越好。表中均值和标准差来自三次训练，尤其Genome的离散程度较大，不能只取最优结果。PRICE仅有一个官方预训练checkpoint，训练数据与预算不同，属于补充参考。',[sourceSummary,path.join(ROOT,'实验文档/108-四目标库K0三seed补测.md')]);
}

// 8. Summary with bounded claims.
{
 const s=base('阶段性总结与待解决问题','02  当前进度',8);
 text(s,'已完成的工作',64,185,514,44,31,C.teal,true);
 text(s,'数据与评估流程',64,254,514,40,27,C.navy,true);
 text(s,'完成跨库划分、查询去重和统一指标评估',64,301,514,66,24,C.ink);
 text(s,'模型与适应实现',64,386,514,40,27,C.navy,true);
 text(s,'轻量编码器、FOMAML 与独立支持集适应',64,433,514,66,24,C.ink);
 text(s,'结构对照与误差分析',64,518,514,40,27,C.navy,true);
 text(s,'已开展编码方式、适应效果和运行开销分析',64,565,514,66,24,C.ink);
 line(s,628,184,628,626);
 text(s,'需要解决的问题',686,185,530,44,31,C.amber,true);
 text(s,'适应效果存在数据库差异',686,254,530,40,27,C.navy,true);
 text(s,'既有对照中 Genome 有改善，STATS 仍有负迁移',686,301,530,66,24,C.ink);
 text(s,'元学习收益依赖具体设置',686,386,530,40,27,C.navy,true);
 text(s,'收益随架构和样本预算变化，尚未形成普遍优势',686,433,530,66,24,C.ink);
 text(s,'结构改进仍需验证',686,518,530,40,27,C.navy,true);
 text(s,'PRICE 式编码器初步源侧对照尚未显示总体收益',686,565,530,66,24,C.ink);
 note(s,'当前已形成可用于比较不同方案的模型和实验基础。既有少样本对照表明，Genome在适应后尾部误差下降，但STATS存在负迁移，这一现象的原因尚未被独立分离。FOMAML相对ERM的收益依赖架构和支持样本预算，不能写成普遍有效或无效。最新A0/A1源侧同条件比较是单次训练的描述性结果，尚无正式阶段判定，完整矩阵未完成。已有运行开销分析也提示需要区分模型前向、特征提取与数据库执行。',[path.join(ROOT,'实验文档/14-离线实验统一对照与论文结论.md'),path.join(ROOT,'.claude/worktrees/122-p0-closure-20261111/reports/122-a0-control-20261115/README.md'),path.join(ROOT,'实验文档/110-推理延迟与估计延迟实测.md')]);
}

// 9. Part two opener: temporal drift / in-context learning (design stage only).
{
 const s=base('面向时间漂移的上下文学习在线校准','02  当前进度 · 第二部分',9,'在线阶段不更新参数，用近期查询反馈感知当前分布并校正估计');
 text(s,'两阶段框架',64,196,560,34,26,C.navy,true);
 const off=node(s,'离线阶段\n元学习通用初始化参数',64,240,250,84,{size:22});
 const onl=node(s,'在线阶段\n上下文学习校正估计',352,240,272,84,{size:22,fill:C.paleTeal,color:C.teal});
 connect(s,off,onl);
 text(s,'对应空间漂移',64,330,250,28,21,C.teal,true);
 text(s,'对应时间漂移',352,330,272,28,21,C.teal,true);
 line(s,64,368,624,368);
 text(s,'在线阶段的机制',64,392,560,34,26,C.navy,true);
 text(s,'上下文示例：「近期查询-真实基数」对',64,438,560,26,20,C.ink);
 text(s,'与当前查询一并输入模型',64,466,560,26,20,C.ink);
 text(s,'由注意力机制感知当前分布并校正估计',64,494,560,26,20,C.ink);
 text(s,'全程不更新参数',64,522,560,26,20,C.ink);
 node(s,'两阶段共用同一套查询特征与评估指标',64,556,560,56,{size:22,fill:C.light,color:C.navy});
 text(s,'来源：中期考核报告第二部分与后续工作安排',64,626,560,24,17,C.muted);
 text(s,'完成情况',676,196,540,34,26,C.navy,true);
 text(s,'已完成 · 方案设计，约占该部分 30%',676,232,540,28,21,C.teal,true);
 text(s,'确定「近期查询-真实基数」对',676,272,540,26,22,C.ink);
 text(s,'作为上下文示例的输入形式',676,298,540,26,22,C.ink);
 text(s,'设计按时序截断与回放',676,336,540,26,22,C.ink);
 text(s,'构造训练/测试切分的仿真管道方案',676,362,540,26,22,C.ink);
 text(s,'尚未实现 · 暂无代码与精度结果',676,400,540,28,22,C.amber,true);
 const chips=['上下文池构建','示例选取策略','注意力校准模块'];
 chips.forEach((v,i)=>node(s,v,676+i*184,436,172,42,{size:20,fill:C.light,color:C.ink,bold:false}));
 line(s,676,500,1216,500);
 text(s,'已知边界与风险',676,518,540,30,23,C.amber,true);
 text(s,'示例选取策略与长度决定推理延迟，尚未量化',676,556,540,26,18,C.ink);
 text(s,'只有仿真切分方案，缺少真实时序负载验证',676,584,540,26,18,C.ink);
 text(s,'上下文校准与元学习初始化的耦合待实验确认',676,612,540,26,18,C.ink);
 note(s,'第二部分是面向时间漂移的上下文学习在线校准，目前完成方案设计，约占该部分工作的30%。框架上，离线阶段用元学习在源库上得到通用初始化参数，对应空间漂移；在线阶段把近期查询及其真实基数作为上下文示例，与当前查询一并输入模型，由注意力机制感知当前分布并校正估计，在不更新参数的前提下适应时序漂移，两阶段共用同一套查询特征与评估指标。已确定的是上下文示例的输入形式，以及按时序截断与回放构造训练测试切分的仿真管道方案；上下文池构建、示例选取策略与注意力校准模块尚未实现，暂无代码与精度结果。已知边界有三条：示例选取策略与长度直接决定推理延迟，目前尚未量化；目前只有仿真切分的方案，缺少真实时序负载的验证；两部分共用一个模型时，上下文校准可能干扰元学习得到的初始化，耦合方式仍需实验确认。',[reportPath,path.join(ROOT,'docs/protocol.md')]);
}

// 10. Future research.
{
 const s=base('下一阶段研究内容','03  未来规划',10);
 text(s,'跨数据库估计与适应',64,181,535,44,31,C.teal,true);
 bullet(s,'完善查询表示','加强连接关系与过滤条件的交互表达',64,253,535);
 bullet(s,'分析适应机制','在源验证库比较适应范围与样本预算',64,383,535);
 bullet(s,'完成对照与消融','方案冻结后，再开展目标库评估',64,513,535);
 line(s,628,183,628,630);
 text(s,'时间漂移反馈校准',686,181,530,44,31,C.teal,true);
 bullet(s,'构造漂移场景','明确数据分布的变化内容与发生时刻',686,253,530);
 bullet(s,'利用历史查询反馈','仅使用预测时刻之前已获得的真实基数',686,383,530);
 bullet(s,'比较校准方式','静态模型、滑动窗口与上下文校准',686,513,530);
 note(s,'后续沿两条路线推进。跨数据库方向继续围绕查询表示和适应机制开展源侧对照，所有方案与选择规则冻结后再进入目标评估。时间漂移方向从实际分布变化的场景和简单校准基线开始，不能仅把查询按时间排序就视为数据漂移。所有上下文反馈必须在预测时刻之前已获得。评价同时关注精度、反馈需求、适应开销，并单独检验数据库执行效果。时间漂移属于后续研究规划，当前尚无实现与效果结论。',[reportPath,path.join(ROOT,'docs/protocol.md')]);
}

// 11. Editable schedule table.
{
 const s=base('工作计划与预期成果','03  未来规划',11);
 const rows=[['时间','主要工作','阶段产出'],['2026.11—12','补齐源侧对照与误差分析','实验报告、方法修订方案'],['2027.01','完善少样本实验与时序基线','跨库结果、校准基线'],['2027.02','反馈校准、消融与章节整理','方法与实验章节初稿'],['2027.03','整合全文与主要图表','学位论文完整初稿'],['2027.04','导师意见修改与送审准备','学位论文送审稿']];
 nativeTable(s,rows,64,184,1152,328,[215,510,427],{font:24,header:25});
 text(s,'预期成果',64,542,210,38,27,C.teal,true);
 text(s,'学位论文、可复现实验实现、对比与消融图表',278,542,938,40,27,C.ink);
 text(s,'请各位老师批评指正',64,607,1152,48,34,C.navy,true,{align:'center'});
 note(s,'后续按照跨库方法完善、时间漂移验证和论文整理三个阶段推进。时间安排是当前计划，将结合实验进展和导师意见调整。每阶段分别形成实验报告、结果图表或章节稿，最终完成学位论文及可复现实现。以上是我的中期汇报，请各位老师批评指正。',[reportPath]);
}

if(p.slides.items.length!==11) throw new Error('Expected 11 slides');
await (await PresentationFile.exportPptx(p)).save(path.join(BUILD,'candidate.pptx'));
await fs.writeFile(path.join(BUILD,'notes.json'),JSON.stringify(notes,null,2));
await fs.writeFile(path.join(BUILD,'text-boxes.json'),JSON.stringify(layoutBoxes,null,2));
for(const [i,s] of p.slides.items.entries()) {
 const stem=`slide-${String(i+1).padStart(2,'0')}`;
 const png=await p.export({slide:s,format:'png',scale:1.25});
 await fs.writeFile(path.join(RENDER,`${stem}.png`),new Uint8Array(await png.arrayBuffer()));
 const lay=await s.export({format:'layout'});
 await fs.writeFile(path.join(RENDER,`${stem}.layout.json`),await lay.text());
 console.log(`rendered ${i+1}/11`);
}
console.log(JSON.stringify({draft:path.join(BUILD,'candidate.pptx'),slides:11,renderDir:RENDER}));

if(process.argv.includes('--finalize')) {
 const SKILL='/Users/williamlin/.codex/plugins/cache/openai-primary-runtime/presentations/26.1112.11217/skills/presentations';
 const RUNTIME='/Users/williamlin/.cache/codex-runtimes/codex-runtime-install-3jSuSH/payload/codex-primary-runtime/dependencies';
 process.env.RUNTIME_NODE_MODULES=path.join(RUNTIME,'node/node_modules');
 process.env.RUNTIME_NODE=path.join(RUNTIME,'node/bin/node');
 process.env.RUNTIME_PYTHON=path.join(RUNTIME,'python/bin/python3');
 process.env.RUNTIME_BIN_DIR=path.join(RUNTIME,'bin/override');
 const {finalizePresentation}=await import(path.join(SKILL,'container_tools/artifact_tool_utils.mjs'));
 const result=await finalizePresentation({
  workspaceDir:WORK,candidatePath:path.join(BUILD,'candidate.pptx'),
  finalPath:path.join(OUT,'面向数据漂移的自适应基数估计-中期汇报.pptx'),
  pythonExecutable:process.env.RUNTIME_PYTHON,
  integrityValidatorPath:path.join(SKILL,'container_tools/inspect_presentation_package_integrity.py'),
  layoutValidatorPath:path.join(SKILL,'container_tools/inspect_presentation_layout_geometry.py'),
  layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-bullet-geometry','--validate-heading-fit','--require-native-table-slide','7','--require-native-table-slide','11'],
  explicitTotalSlideCount:11,requiredNativeTableOwnerSlides:[7,11],requiredNativeChartOwnerSlides:[7],
  materializeLiteralChartWorkbooks:true,
  fontPolicy:{basis:'design',families:[FONT]},verifyArtifactToolImport:true,
  receiptPath:path.join(BUILD,'validation.json'),
 });
 console.log('FINALIZED',JSON.stringify({finalPath:result.finalPath||result.path,passed:true}));
}
