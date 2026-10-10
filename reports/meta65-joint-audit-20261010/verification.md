# Meta65联合复核验证记录

日期：2026-10-10。独立worktree `/private/tmp/spatial-meta-ce-meta65-joint-control-20261010`，分支 `audit/meta65-joint-control-20261010`，基准 `a447ac1d7d1cd061beddbd72b55091d5b64d5af6`。最新旧worktree干净；主目录未切换、未改写、未提交他人内容。基准没有 `memory/spatial-meta-ce-agent-delegation.md`，本轮只读主目录已有同名文件并遵循用户给定委派授权，没有迁移或覆盖它。

## 验证命令与真实结果

1. `PYTHONDONTWRITEBYTECODE=1 /opt/anaconda3/bin/python -B -m unittest tests.test_meta65_joint_audit tests.test_diagnose_meta65_telstra -v`：27项通过。新模块先失败于缺少实现；D5两项真实调用回归先失败（0条留一结果/错误键未抛错）后修复；JSON序列化测试先报int64错误后修复。未删除测试、未放宽容限。
2. `PYTHONDONTWRITEBYTECODE=1 /opt/anaconda3/bin/python -B -m unittest discover -s tests -v`：903项、4fail/15skip，macOS沙箱子进程OpenMP共享内存错误。获批沙箱外相同命令：903项、1fail/15skip，既有S8测试发现实际BLAS线程超预算。
3. `PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 CUDA_VISIBLE_DEVICES="" /opt/anaconda3/bin/python -B -m unittest discover -s tests -v`（获批沙箱外）：**903项、100.660秒，OK(skipped=15)**。15项为平台/环境条件skip，不等于Linux集成已验证。
4. `/Users/williamlin/.cache/uv/archive-v0/YGscLdzyjxxZtKYJn0d_H/ruff-0.16.10.data/scripts/ruff check scripts/meta65_joint_audit.py scripts/diagnose_meta65_telstra.py tests/test_meta65_joint_audit.py tests/test_diagnose_meta65_telstra.py`：All checks passed。全仓 `ruff check --no-cache scripts tests spatial_meta_ce --output-format json`：110项、exit1；对基准旧worktree相同命令也是110项；逐文件/规则/消息multiset比较added=[]、removed=[]，不修复无关历史问题。服务器venv无ruff，后用本地缓存二进制，无安装。
5. `PYTHONDONTWRITEBYTECODE=1 /opt/anaconda3/bin/python -B scripts/meta65_joint_audit.py --remote myserver --diagnosis 实验文档/results/meta65-telstra-diagnosis-20261010/telstra_diagnosis.json --decision 实验文档/results/meta65-endpoint-completion-20261009/decision.json --protocol 实验文档/54-Meta65诊断口径复算协议.md --out 实验文档/results/meta65-joint-audit-20261010/joint_audit.json`：最终恢复exit0、69.806986秒。远端单线程timeout300秒、CUDA不可见，230输入SHA前后一致、225预测cells完整、配对身份/真值/表数与完整分母、跨Kquery通过。script/protocol绑定与当前文件SHA一致；原诊断SHA `246f18f1...`、decisionSHA `2ce27117...` 不变。首次复算及诊断执行均暴露JSON int64序列化缺陷，修复回归后成功；一次本地同路径JSON压缩误截空本轮新输出，已从冻结输入恢复为紧凑JSON，不从手抄数值恢复，旧产物未受影响。
6. SSH只读checkpoint完整SHA核验：A=`ae0c386d35ac368964b4160d634ca063d7801c65e5296ac679c459dd01951b12`，B=`a2d777a4dcaad6d23c3f1212929d062768f5157a3fc37bce9909903a22f18db9`，C=`58f2f4db6f54cd4578fd6127d5a4aa350dbdc2268a36a057b56f7161cb6097a4`，各2574664bytes；不反序列化。plan=`ebeb178de15b134722bd9fbc24ff4b4752f795a311580c1a17cb23c550bfd0f0`。
7. 本地从新JSON逐run数值复算4+ macro：A44.49797889568323、B39.000403352286426、C39.01038305004801，与原完整端点一致。原D5留一独立stdlib复算：telstra25组合符号不翻转、删run0排序改变；chembl删K5/run4符号翻转，legalacts未翻转，三库均有K排序变化。不能把telstra结论泛化为三库稳定。

## 独立审查

- 旧统计独立核验（Cicero，只读）：确认D2 all-table/pooled误标，发现D5真实调用未执行、K排序强结论错误；核对原decision与表格比值，53四处两位小数已更正。采纳全部实质问题，冻结52原协议不改，55明确勘误。
- 新产物/代码独立审查（Ampere，只读）：690个aggregate逐项重算0不一致，225cells与完整端点CSV指标逐项匹配；提出并已修正55号的全局Q95差异上界及“实际出现n<30”表述。56方案审查（Carson，只读）：当前不可启动，已采纳独立evaluation-only入口、完整K20计划复用、四桶交互公式、共享身份侧表、FP64真值链路和外部watchdog等阻塞项；`launch_ready=false`保持不变。
- 56修订后的窄复审通过：上述意见已落实，无仍阻塞本轮方案交付的问题；仅确认协议修正，不认证未实现runner或实际资源达标，不构成执行授权。新增/修改Markdown链接检查0缺失、`git diff --check`通过；四个Python AST解析通过，新增行/新文件敏感信息模式扫描通过（历史99号模板不混入本轮新增证据）。

## 边界与自检

新联合结果是seed42/既有开发库上的描述性证据，不是适应算子因果对照。56协议尚无runner、`launch_ready=false`，预算为历史耗时外推，不是实测；不以审查通过作为GPU授权。不自动重启S8，不新增目标库评估。未重建生成图谱；可复查证据均直接SHA绑定。

自检五轴（当前交付）：accuracy4（真实输入绑定/复算，因果结论未测）、completeness4（协议具备有限矩阵，runner按授权边界留待后续）、clarity5（明确pooled/逐run/区组/未知）、actionability4（实施预检后才能执行）、conciseness4（完整JSON较大但正文只列关键量），均分4.2。最高优先改进为实施none与reset防污染测试、冻结launch-binding、在获授权后完成同checkpoint对照；不在本轮越权补跑。

提交前检查链接、空白、绑定、敏感信息与diff，实际提交/推送状态以Git及最终回复为准，不提前写已同步。
