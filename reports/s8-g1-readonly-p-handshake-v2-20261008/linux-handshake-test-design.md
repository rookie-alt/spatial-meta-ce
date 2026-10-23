# 真实 Linux parent→child 握手 unittest：helper 级设计与限制

日期：2026-10-08。工作区：`/private/tmp/spatial-meta-ce-s7-diagnosis-20261007`，接手 HEAD `72eb087`。

## 1. 边界与夹具合规评估

已读取 [37 号单次失败记录](../../实验文档/37-S8G1A单次只读核验握手失败记录.md) 与 [独立实际记录复审](../s8-g1-readonly-p-execution-20261008/independent-record-review.md)。旧 A 的一次授权已消耗；本测试不重试真实 A、不读取旧 P、不追认 G1，也不改旧协议或失败证据。

测试文件：[test_s8_readonly_handshake_linux.py](../../tests/test_s8_readonly_handshake_linux.py)。生产被测 helper：[verify_worker_handshake](../../scripts/verify_s8_readonly_p.py)。主代理负责生产路径规范化及既有测试，本实现不写这些文件。

**临时包方案不修改生产 guard，但证据仅为 helper 级真实 proc/pipe 集成。**

- 每个用例在 `TemporaryDirectory` 中建独立包，将当前生产源码完整 `read_bytes/write_bytes` 复制为 `scripts/_handshake_impl.py`，不抽取、拼接或改写任何生产字节。
- 测试从 AST 读取当前源码的固定 `ENTRYPOINT` / `CONTRACT_RELATIVE` 路径字面量，不设旧路径 fallback。临时父 fixture 写到该包的 `scripts/verify_s8_readonly_p.py`；fixture import 同目录 `_handshake_impl`。其 `ROOT = Path(__file__).resolve().parents[1]` 自然指向临时包。
- parent 与 child 在 import 前独立核对副本 SHA256；用例结束核对副本原字节及当前生产源码 SHA256。没有修改或 patch `ROOT`、常量、生产函数、返回值、`sys.platform`、proc 或 syscall；不使用 mock。
- fixture 的 `--execute` 只是其真实父 cmdline 身份标识。fixture **不调用生产 `main()` / `worker()` / `check_sources()` / `install_read_guard()` / 资源或数值入口**；唯一生产函数调用是 `verify_worker_handshake()`。生产 CLI 的 synthetic `--execute` 拒绝逻辑保持原状，不增加任何生产测试参数或后门。
- contract 是 tempfile 中的最小合成 helper manifest：`synthetic_fixture`、包路径与新 output 路径等。它不声称通过完整 `validate_contract` 冻结 schema。
- 完整实现副本 SHA 与生产源码一致，**不等于 fixture 父脚本 SHA 与生产入口相同**。临时父脚本只满足被测 helper 的路径/cmdline 身份条件；不能用它替代生产父入口源码 SHA 认证。

因此没有为生产执行增加绕过通道，也没有把 fixture 成功冒充完整 data CLI guard 成功。若验收目标是完整生产 CLI，则本夹具不足：必须由主任务在另行批准的真实冻结包、真实授权与预算下执行；不能为此放开 synthetic 生产执行。

## 2. 真实认证链与观察点

拓扑为 unittest → fixture parent → 单个 fixture child。parent 与 child 均实际使用 `sys.executable -B`；parent argv 真实包含 `ENTRYPOINT`、`--execute`、contract/SHA/output 参数。child 通过真实 `/proc/<ppid>/cmdline`、`/proc/<ppid>/cwd`、`/proc/self/cwd` 和双方 fd symlink 记录观测，随后调用原字节 helper。

每个 child 的 cwd 都是包外的 `child-cwd`，与 parent cwd 不同。child contract 始终使用冻结临时包绝对路径（专门的错 child 路径负例除外）。相对 parent contract 应按 **parent cwd** 解释，而不是 child cwd；绝对 parent contract 也应成功。

parent 用真实 `os.pipe()` 和 `os.write()` 发送 ticket，保留 writer 至 child 结束，仅 `pass_fds` 继承 reader。正例同时核：FIFO 类型、双方 `pipe:[inode]` 一致、真实 PPID、ticket 四字段、包 ROOT、contract 字节 SHA、路径和真实 argv；没有替换 helper 返回值。负例要求真实 helper 的 `ValueError` 与对应错误消息，不能把任意 fixture 异常或 timeout 当作拒绝通过。

parent/child 都检查 `sys.modules` 无 `numpy` / `torch` / `spatial_meta_ce`。实现副本仅导入标准库；数值函数虽存在于源码定义中，但本测试不调用，也没有创建模型、solver、数组、checkpoint 或数据读路径。

## 3. 用例覆盖

共 **13 个 unittest 方法，16 次独立 parent→child 握手**（ticket 方法内四个 subTest）；macOS 整类 skip，不创建 fixture 进程。

| 类型 | 场景与要求 |
|---|---|
| 正例 | 相对 contract + 不同 child cwd：实际按 parent cwd 认证 |
| 正例 | 绝对 contract + 不同 child cwd：相同冻结文件认证 |
| 负例 | 错 parent cwd，相对路径落到另一个同字节同 SHA shadow contract；入口仍为真实预期绝对路径，必须拒绝参数绑定 |
| 负例 | parent 另一个 contract 路径，即便字节/SHA 相同也拒绝 |
| 负例 | child 另一个 contract 路径，拒绝冻结路径不匹配 |
| 负例 | parent SHA 错，child 真实 manifest SHA 与 ticket 不变 |
| 负例 | parent output 错，child 冻结 output 与 ticket 不变 |
| 负例 | child output 错，拒绝冻结 output 不匹配 |
| 负例 | output 已有 `worker-proof.json`，拒绝旧 proof 复用 |
| 负例 | 同 fixture 字节却由 `fake_parent.py` 执行，拒绝假父入口身份 |
| 负例 | 两个真实 pipe，advertised parent writer fd 指向另一 inode；FIFO 仍真实，ownership 绑定必须拒绝 |
| 负例 | 继承真实 regular-file fd 而非 pipe，必须拒绝 |
| 负例×4 | 真实 pipe 中分别篡改 ticket 的 parent PID、contract SHA、output、package，必须逐项拒绝 |

这不是所有安全攻击面的穷举；没有声称覆盖真实部署 source manifest、audit guard、预算、运行库来源、数值 stage 或完整 CLI。

## 4. 自有进程预算与回收

fixture parent 的 `communicate(timeout=3)` 监督唯一自有 child；超时立即 kill 并在 0.5 秒内回收，finally 也检查并回收。正常用例要求无 timeout、child 已回收、监督区间不超过 3.5 秒（小于要求的 5 秒），child stderr 为空，child `/proc` 已消失。unittest 再以 5 秒监督其自有 parent；异常时只 kill 该 `start_new_session=True` 建立的进程组并回收 parent，不触及其他进程。

child 是 unittest 的 grandchild，应由 fixture parent 回收；测试自身的 `waitpid` 应得到 `ChildProcessError`，不将 grandchild 当作自己可直接 wait 的 child。正常及握手拒绝路径都实际执行 `communicate` 回收，不重试。

fixture 环境显式 `CUDA_VISIBLE_DEVICES=""`、禁止写 pyc、数值线程变量设为 1，清除 `PYTHONPATH/PYTHONHOME/PYTHONSTARTUP/LD_PRELOAD`。没有 solver、训练、GPU 或数据执行。超时为进程监督界限，非实时系统硬时延证明；整体 60 秒／1 CPU 限制由主代理单次 Linux 合成测试外层监督设置，本测试不擅自启动该服务器任务。

## 5. 本地验证与待执行项

本轮本地平台为 macOS（`sys.platform == 'darwin'`）。专项发现命令：

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -B -m unittest discover -s tests -p test_s8_readonly_handshake_linux.py -v
```

结果：13 项发现，`OK (skipped=13)`；这只是 skip/发现验证，**不是 Linux 集成 GREEN**。测试源及内嵌 fixture 分别以 `ast.parse/compile` 验证语法；Ruff 对测试文件与内嵌 fixture stdin 检查均 `All checks passed!`（先修正测试 import 空行与 fixture 无效 noqa，再复查通过）。AST 静态检查确认 fixture 唯一生产模块函数调用为 `impl.verify_worker_handshake`；Markdown 链接/行尾空白已核对。本轮不生成额外 fixture 文件。

Linux 实际执行留给主代理：完成本地全 tests / Ruff / 审查后，按用户允许的 **一次合成 unittest、60 秒、1 CPU、empty CUDA** 外层运行同一专项命令。不由本实现代理联网，不重复失败命令，不读真实 P/oldrun/H/目标/checkpoint，不运行训练或 GPU。Linux 是否通过必须以那次真实输出为准，本设计没有提前认证。

写集只包含测试文件和本设计文档；不修改主任务进度、证据索引或历史记录，不 commit/push。
