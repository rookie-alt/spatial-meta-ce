# 分支收敛与主干恢复执行方案

> 生成日期：2026-10-24。性质：**待执行的操作手册**，不是实验结果。执行前请先读完「§2 三个决策点」并做出选择。
> 全部现状数字来自 2026-10-24 本机实测（`git for-each-ref` / `git merge-tree` / `git rev-list --count`）。

---

## 1. 现状事实（实测）

### 1.1 主干状态

| 项 | 实测值 |
|---|---|
| `main` tip | `54c502e` **2026-10-03** |
| 当前 HEAD | `a9cbef3` `docs/meta65-scope-holdout-paper-20261017` **2026-10-19** |
| 最新真实工作 | `claude/peaceful-kare-5bb350` = `docs/advisor-report-corrections-20261023` = `feat/arch-compare-paired-20261024` 均指向 **`1892b60` 2026-10-23** |
| `origin/HEAD`（Gitee 默认分支） | `feat/paired-eval-and-exp-docs` @ `2022d86` 2026-10-19 |
| `main` → `1892b60` | **ahead 243 / behind 0** ⇒ `main` 是 `1892b60` 的**严格祖先** |

**关键结论：`main` 是 frontier 的祖先，所以恢复主干是一次 fast-forward，不需要重写历史。** 这是整个方案风险最低的前提，请勿改用任何破坏性操作。

### 1.2 分支账目

| 分类 | 数量 | 说明 |
|---|---:|---|
| 本地分支总数 | **172** | |
| 远端分支总数 | 69 | |
| 已完全合并进 `1892b60` | **165** | 含 74 个自动生成名分支 |
| **未合并进 `1892b60`（有独有提交）** | **7** | 需逐个判定，见 §5 |
| 自动生成名分支（`worktree-agent-*` / `claude/<形容词>-<科学家>-<hash>`） | 74 | 其中 58 个 tip = `54c502e`（零独有提交） |
| 零独有提交、tip 正好是 main | 65 | 纯噪声 |
| tag | **0** | 全仓库没有任何 tag |
| stash | 2 | 见 §8 |
| worktree | 50 项（3 项 prunable） | 占 **3.1 GB**；仓库总计 3.3 GB |

### 1.3 未推送情况（精确版）

只有 **4 个** 分支的 tip 尚未推到 origin：

| 分支 | 状态 |
|---|---|
| `chore/remove-graphify-use-gitnexus` | 无对应远端分支，独有 1 提交 |
| `claude/review-plan-changes-915c63` | 无对应远端分支，独有 1 提交 |
| `docs/meta65-scope-holdout-paper-20261017` | 本地领先 origin 1（= `a9cbef3`） |
| `feat/s8-numerical-certificate-20261007` | 本地领先 origin 2，但**内容已是 frontier 祖先**，可丢弃 |

（其余"未推送"分支 tip 都已包含在 `1892b60` 中，删除无损失。）

### 1.4 工作区当前状态（**不属于本方案**）

```
 M AGENTS.md                                    ← 另一窗口的"独立审查门槛"规则，勿动
 M CLAUDE.md                                    ← 同上
 M memory/spatial-meta-ce-agent-delegation.md   ← 同上（+41 行）
?? reports/figures/                             ← 4.9 MB × 4 张 PNG
?? reports/project-status-20261018.html
```

**因此本方案全部在临时克隆里执行，绝不在当前工作区切分支或 reset。** 见 §4 Phase 0。

---

## 2. 三个决策点（执行前必须先回答）

### 决策 1：主干叫什么？

| 选项 | 做法 | 代价 |
|---|---|---|
| **A（推荐）** | `main` fast-forward 到 `1892b60`，Gitee 默认分支改回 `main` | 无历史重写；需在 Gitee 仓库设置里改默认分支 |
| B | 把 `claude/peaceful-kare-5bb350` 改名为 `main`，旧的 `main` 改名为 `archive/main-20261003` | 全是本地分支改名，仍需改 Gitee 默认分支；旧 main 需单独保留 |
| C | 不动结构，只打 `thesis-final` tag | 5 分钟，但"哪棵树是当前状态"的问题不解决 |

**选 A。** 因为 `main` 是 frontier 祖先，ff 无损，且 `main` 这个名字对 Gitee/答辩最无歧义。

### 决策 2：GitNexus 工具迁移要不要？

现状是**两条互斥的工具链并行存在**：

- `frontier (1892b60)`：仍是 graphify —— 有 `scripts/build_experiment_graph.py`、`tests/test_experiment_graph.py`、`tests/test_graph_build.py`
- `chore/remove-graphify-use-gitnexus` (`2c1494e`)：删掉 `test_graph_build.py` **2199 行**，换成 GitNexus
- `chore/gitnexus-index` (`258f103`)：往 AGENTS.md/CLAUDE.md 加 88 行 GitNexus 配置

两者 add/add + modify/delete **必然冲突**。

| 选项 | 结果 |
|---|---|
| **D1（推荐）** | **不合并迁移**，两条都打 `archive/` tag 后删除。理由：删 2199 行证据图谱测试属于工具偏好，不是研究内容；项目 CLAUDE.md 的按需图谱规则目前是按 graphify 写的，迁移需单独一轮改造 + 测试。 |
| D2 | 执行迁移：需在新分支上手工重做，补 `test_graph_build` 的等价测试，再合入。约半天。 |

### 决策 3：`experiment/target-3seed-20261023-local` 的结果算不算正式结果？

这个分支上有 **`e1bd400`：4 个目标库 × 3 seeds（42/43/44）的完整 summary**，共 12 文件 5330 行，frontier **完全没有收录**（`git ls-tree 1892b60 | grep -c p2-2-target-3seed` = **0**），文档里也**没有任何引用**。

这与项目自己记录的最大方法学缺陷（单 seed、seed 噪声 ~6.3 pp）直接相关——它可能是目前唯一的 3-seed 目标库测量。

| 选项 | 结果 |
|---|---|
| **E1（推荐）** | 合并进主干，并在 `99-进度与下一步.md` 新增一条正式条目说明其状态（pilot 还是 formal），避免它成为"存在但无人引用"的孤儿数据 |
| E2 | 确认它已被后续 formal 实验取代 → 打 tag 归档，不合并 |
| E3 | 暂不合并，保留分支 | ← 不推荐，会再次制造孤儿 |

---

## 3. 安全前提（先读）

1. **全程不需要 `git push --force`。** 任何步骤要求 force，说明前提被破坏，停下重新评估。
2. **只在临时的 second clone 里操作**（§4.1），因为当前工作区有 3 个未提交文件 + 2 个未跟踪项属于另一窗口。
3. **`git branch -d` 只删已合并分支**，这是天然保护；绝不用 `-D` 批量删未合并分支。
4. **未合并的 7 个分支，一律先打 `archive/` tag 再删分支**，保住历史可达性。
5. **dirty worktree 不可恢复**，删除前必须 `git -C <path> status --porcelain` 为空。

---

## 4. Phase 0：零风险备份（约 10 分钟）

**目的：让后面所有操作都可回滚。**

```bash
cd /Users/williamlin/Downloads/13/spatial-meta-ce

# 0.1 确认没有正在跑的 git 操作
git status
ls .git/*.lock 2>/dev/null        # 期望：无输出

# 0.2 全量 bundle 备份（含所有 ref 和 2 个 stash）
mkdir -p ~/Downloads/13/_backup
git bundle create ~/Downloads/13/_backup/smc-full-20261024.bundle --all
git rev-parse --short HEAD > ~/Downloads/13/_backup/HEAD-20261024.txt

# 0.3 校验 bundle 可读
git bundle verify ~/Downloads/13/_backup/smc-full-20261024.bundle

# 0.4 记录当前 ref 清单（回滚用）
git for-each-ref --format='%(objectname) %(refname)' > ~/Downloads/13/_backup/refs-20261024.txt
wc -l ~/Downloads/13/_backup/refs-20261024.txt
```

**验收**：`git bundle verify` 输出 `The bundle records a complete history.`；`refs-20261024.txt` 行数 ≈ 172 + 69 + 2。

**注意**：`git bundle --all` 不包含 stash。若 §8 的 stash 重要，额外 `git stash show -p` 导出 patch。

---

## 5. Phase 1：确定唯一主干（约 30 分钟）

### 5.1 在临时克隆里操作（不动当前工作区）

```bash
cd /tmp
rm -rf smc-consolidate
git clone --no-hardlinks /Users/williamlin/Downloads/13/spatial-meta-ce smc-consolidate
cd /tmp/smc-consolidate
git remote set-url origin https://gitee.com/william-forest/spatial-meta-ce.git
git fetch --all --tags --prune
```

### 5.2 fast-forward `main`

```bash
git switch main
git merge-base --is-ancestor main 1892b60 && echo "OK: main 是祖先，可 ff" || echo "STOP: 非祖先，勿继续"
git merge --ff-only 1892b60
```

**验收**：`git log -1 --oneline` 显示 `1892b60`；`git rev-list --count main..1892b60` = 0。

### 5.3 打论文交付 tag（无论选哪个选项都建议打）

```bash
git tag -a thesis-20261023 -m "论文交付快照：含 100-106 号文档（IMDB 对照、误差结构诊断、表示探针）" 1892b60
git tag -a pre-consolidation-20261024 -m "分支收敛前快照：main=54c502e, 172 本地分支" 54c502e
```

### 5.4 推送

```bash
git push origin main
git push origin thesis-20261023 pre-consolidation-20261024
```

**验收**：`git ls-remote --heads origin main` 返回 `1892b60`。

---

## 6. Phase 2：处理 7 个未合并分支（约 1.5 小时）

逐个按表执行。**顺序不能乱**：先合干净的，再逐个人工解冲突。

### 6.1 处置决策表（已用 `git merge-tree` 预演过冲突）

| # | 分支 | 独有 | merge-tree | 处置 | 理由 |
|---|---|---:|---|---|---|
| 1 | `claude/project-progress-review-94eba1` | 1 | ✅ clean | **合并** | 新增 `实验文档/77-4000步探针因共享盘写满中断.md`（17 行），纯历史记录，无风险 |
| 2 | `claude/review-plan-changes-915c63` | 1 | ✅ clean | **合并**（合并后必须 diff 复核） | 修 `CLAUDE.md` 里 `--semantic <semantic.json>` → `graphify-out/semantic.json` 并补前置说明。⚠️ 该分支基于旧版 CLAUDE.md，合并后要确认 graphify 按需规则**没有被删**（见 6.4 验证） |
| 3 | `docs/meta65-scope-holdout-paper-20261017` | 1 | ✅ clean | **合并** | 独有内容只有 `a9cbef3` 往 `.gitignore` 加 `.gitnexus` 一行，无副作用 |
| 4 | `experiment/target-3seed-20261023-local` | 1 | ❌ 冲突（仅 `99-进度与下一步.md`） | **按决策 3 处理**；若 E1 则合并 | 5330 行 3-seed 结果，见 §2 决策 3 |
| 5 | `audit/cache-integrity-20261016` | 1 | ❌ 冲突（`data.py`/`features.py`/`test_input_audit.py`） | **先人工 diff，默认归档不合并** | 见 6.2 |
| 6 | `chore/gitnexus-index` | 1 | ❌ 冲突（AGENTS.md/CLAUDE.md add/add） | **按决策 2** | 决策 2 选 D1 ⇒ 归档 |
| 7 | `chore/remove-graphify-use-gitnexus` | 1 | ❌ 冲突（modify/delete `test_experiment_graph.py`） | **按决策 2** | 决策 2 选 D1 ⇒ 归档 |

### 6.2 分支 5（cache-integrity）的判定方法

frontier 的 `data.py` **已经**有内容哈希校验和绑定：

```bash
git show 1892b60:spatial_meta_ce/data.py | grep -nE 'content hash|bind_cache|counts mismatch'
# 实测命中：
#   112:        from .manifests import digest
#   115:        if digest(payload) != self.manifest.get("content_hash"):
#   116:            raise ValueError("Cache manifest content hash mismatch")
#   140:        self.mapping.bind_cache(self)
#   172:            raise ValueError(f"Mapping/cache counts mismatch at row {index}")
```

但 `tests/test_cache_integrity.py`（225 行测试）在 frontier 中**不存在**（`git ls-tree 1892b60 -- tests | grep -i cache` 无输出）。

```bash
# 逐文件看真实差异，判断 frontier 是否已独立实现了同样的修复
git diff 1892b60 2998807 -- spatial_meta_ce/data.py spatial_meta_ce/features.py
git show 2998807:tests/test_cache_integrity.py | head -40
```

判定：
- 若 frontier 的校验**不弱于**该分支 ⇒ 只 **cherry-pick 测试文件**，分支本身归档
- 若 frontier 缺少其中某项强制（例如 strict reuse policy）⇒ 开新分支 `fix/cache-integrity-reapply-20261024`，手工重做该增量 + 跑测试

### 6.3 执行合并（分支 1/2/3）

```bash
cd /tmp/smc-consolidate
for b in claude/project-progress-review-94eba1 claude/review-plan-changes-915c63 docs/meta65-scope-holdout-paper-20261017; do
  echo "=== merging $b ==="
  git switch main
  git merge --no-ff "$b" -m "merge: 收敛 $b"
done
```

### 6.4 合并后必须做的复核（**这一步不能省**）

```bash
# a) graphify 按需规则是否还在
git show main:CLAUDE.md | grep -c '默认不运行 graphify'      # 期望 1，不能是 0
git show main:CLAUDE.md | grep -n 'graphify-out/semantic.json'

# b) 无产物混入
git ls-tree -r --name-only main | grep -E '\.(pt|pth|npy|pkl)$'   # 期望：无输出

# c) 文件数没有异常减少（基线 1807，只应增不应减）
git ls-tree -r --name-only main | wc -l

# d) .gitnexus 已进 .gitignore
git show main:.gitignore | grep -c '^\.gitnexus$'                 # 期望 1
```

### 6.5 处理分支 4（若选 E1）

```bash
git switch main
git merge --no-ff experiment/target-3seed-20261023-local
# 冲突只会在 实验文档/99-进度与下一步.md —— 手工保留双方内容：
#   保留 frontier 的 10-23 入口层，把 3-seed 结果作为新的一节追加，不要覆盖
git diff --name-only --diff-filter=U
# 编辑后：
git add 实验文档/99-进度与下一步.md
git commit
```

然后在 `99-进度与下一步.md` 顶部入口层补一行，说明这批结果的性质（pilot/formal、是否 `formal_verified`、SHA 绑定在哪）。**不要让它成为无引用孤儿。**

### 6.6 归档被否决的分支

```bash
# 每个不合并的分支都打 tag，再删分支（tag 保留全部历史可达性）
for b in audit/cache-integrity-20261016 chore/gitnexus-index chore/remove-graphify-use-gitnexus; do
  short=$(echo "$b" | tr '/' '-')
  git tag -a "archive/$short-20261024" -m "归档：未合并，原因见 reports/branch-consolidation-plan-20261024.md" "$b"
done
git push origin --tags
```

---

## 7. Phase 3：清理 165 个已合并分支（约 40 分钟）

### 7.1 先确认 frontier 已在 main 上

```bash
cd /tmp/smc-consolidate
git switch main
git rev-list --count main..1892b60      # 必须是 0
```

### 7.2 删除已合并本地分支

```bash
# 先干跑，看要删什么
git branch --merged main --format='%(refname:short)' | grep -vE '^(main|master)$' > /tmp/to_delete.txt
wc -l /tmp/to_delete.txt     # 期望 ≈ 165
grep -E '^(main|master)$' /tmp/to_delete.txt && echo "STOP: 含主干" || echo "OK"

# 分批删除（-d 是安全的：只删已合并的）
git branch -d $(cat /tmp/to_delete.txt)
```

> **注意**：`git branch -d` 对未合并分支会拒绝，这正是我们要的保护。若个别分支因 worktree 占用无法删，先做 §8。

### 7.3 删除远端同名分支（约 60-100 个）

```bash
# 找出远端已合并的
git branch -r --merged main --format='%(refname:short)' \
  | grep '^origin/' | grep -vE '^origin/(HEAD|main)$' | sed 's|^origin/||' > /tmp/to_delete_remote.txt
wc -l /tmp/to_delete_remote.txt

# 分批推送删除（每批 20 个，便于中断）
split -l 20 /tmp/to_delete_remote.txt /tmp/rdel_
for f in /tmp/rdel_*; do
  xargs -a "$f" -I{} git push origin --delete "{}" || echo "skip {}"
done
```

**验收**：`git ls-remote --heads origin | wc -l` 从 69 降到个位数。

---

## 8. Phase 4：worktree 与磁盘清理（释放约 3 GB）

### 8.1 先 prune

```bash
cd /Users/williamlin/Downloads/13/spatial-meta-ce
git worktree prune -v
git worktree list | grep -c prunable      # 期望 0
```

### 8.2 分类处理仍在使用的 worktree

当前 50 项中：

| 类型 | 数量 | 处理 |
|---|---:|---|
| `~/.codex/worktrees/*` | 5 | **先确认无未提交改动**，再 `git worktree remove` |
| `.claude/worktrees/agent-a*`（tip=54c502e 的自动 worktree） | ~24 | 全部是可重建的 agent 沙箱，直接 remove |
| `.claude/worktrees/<有意义名>`（docsync-104-105、peaceful-kare、fervent-shamir 等） | ~20 | 逐个 `git -C <path> status --porcelain`，非空则先处理 |

```bash
# 安全模板：只删工作区干净的
for wt in $(git worktree list --porcelain | awk '/^worktree /{print $2}'); do
  [ "$wt" = "/Users/williamlin/Downloads/13/spatial-meta-ce" ] && continue
  if [ -z "$(git -C "$wt" status --porcelain 2>/dev/null)" ]; then
    git worktree remove "$wt" --force && echo "removed: $wt"
  else
    echo "SKIP (dirty): $wt"
  fi
done
```

**验收**：`du -sh .claude/worktrees` 从 3.1 GB 降到 < 200 MB；`git worktree list | wc -l` < 10。

> ⚠️ `.claude/worktrees/peaceful-kare-5bb350` 指向 `1892b60`，是 frontier 的检出。若不确定里面有没有未提交内容，先 `git -C .claude/worktrees/peaceful-kare-5bb350 status` 看一眼。

---

## 9. Phase 5：stash 处置（约 20 分钟）

```bash
cd /Users/williamlin/Downloads/13/spatial-meta-ce
git stash list
```

| stash | 日期 | 规模 | 判定 |
|---|---|---|---|
| `stash@{1}` `meta65-scope-wip-99855` | 2026-10-11 | 1 文件 +10 行（`99号`） | 内容大概率已被后续文档覆盖 → `git stash show -p stash@{1}` 看过后 `drop` |
| `stash@{0}` `refmt-pending-1792426421` | 2026-10-20 | **69 文件 +10692 / −5429** | 大规模格式化，涉及 `tests/test_s8_*`。**不要 `drop`**，先导出 |

```bash
# 大 stash 转成真实分支（推荐：保留完整历史 + 可 apply）
git stash branch archive/refmt-pending-20261020 stash@{0}

# 或只导出 patch
git stash show -p stash@{0} > ~/Downloads/13/_backup/stash0-refmt-20261020.patch

# 小 stash 看完再决定
git stash show -p stash@{1}
git stash drop stash@{1}     # 确认已被 99号 覆盖后执行
```

**验收**：`git stash list | wc -l` ≤ 1；`archive/refmt-pending-20261020` 分支存在。

---

## 10. Phase 6：改 Gitee 默认分支（约 10 分钟，必须手动）

命令行改不动 Gitee 的服务端默认分支，需在网页做：

1. 打开 `https://gitee.com/william-forest/spatial-meta-ce/settings`
2. 「仓库设置」→「默认分支」改为 `main`
3. 保存

然后本地同步：

```bash
git remote set-head origin main
git symbolic-ref refs/remotes/origin/HEAD    # 期望 refs/remotes/origin/main
git remote show origin | grep 'HEAD branch'  # 期望 HEAD branch: main
```

**验收**：`git symbolic-ref refs/remotes/origin/HEAD` 输出 `refs/remotes/origin/main`。

---

## 11. Phase 7：文档入口唯一化（约 1 小时，纯文档）

这是与 172 个分支同源的第二个病：**任何时刻都不止一个"当前状态"入口**。

### 11.1 `README.md`

- 日期从 2026-10-05 更新到 2026-10-23
- 「当前保留 sum」→ 补上 102 号（IMDB 已作为第四目标库）
- 删除「IMDB 阻塞」的过时表述，改为指向 `实验文档/102-*.md`
- 把「最新状态看 99 号」改成「**唯一入口：`实验文档/99-进度与下一步.md` 顶部第 0 节**」

### 11.2 `实验文档/99-进度与下一步.md`（当前 1761 行 / 275 KB）

现在顶部叠了 **4 层"最新入口"**（10-23 / 10-21 / 10-19 / 10-18），互相都说"历史入口仍有效"。改成：

```
## 0. 当前状态（唯一入口）
   - 日期、HEAD、主干 tag
   - 研究结论一句话 + 关键数字表（指向 102/103/106）
   - 未完成 / 未兑现项清单
## 1. 历史入口（按时间倒序，一层一条链接，不再展开）
   - 2026-10-21 摘要 → 见 §81
   - 2026-10-19 摘要 → 见 §80
   ...
```

**规则：同一时刻只允许一层"最新"**，其余全部降级为带日期的历史链接。

### 11.3 `实验文档/00-实验总览.md`

顶部状态从 2026-10-11 更新为 2026-10-23，参数表补一行说明经独立重算：

- set = 99,777；sum = 132,545；PRICE = 10,709,330；比值 107.33× / 80.80×

### 11.4 归档建议（可选）

`99号` 超过 1500 行后可把 §0–§40 拆到 `实验文档/archive/`，主文件只保留近 30 天 + 索引。**不在验收标准内。**

---

## 12. Phase 8：补规则，防止再次退化（约 15 分钟）

在项目 `CLAUDE.md`「工作流」一节追加（**需用独立分支提交**，别混进当前窗口的改动）：

```markdown
- **分支必须有终点**。分支完成后二选一并当天执行：
  (a) 合并回 `main` 并删除分支；
  (b) 打 `archive/<branch>-<date>` tag 后删除分支，tag message 写清不合并的原因。
  任何存活超过 7 天且未合并的分支，必须在上条 (a)/(b) 中做出选择。
- **主干纪律**：`main` 永远是最新可跑状态。合并进 `main` 前必须本地跑通
  `python3 -B -m unittest discover -s tests`，否则主干的"绿"等于无效信息。
- **文档入口唯一**：`实验文档/99-进度与下一步.md` 顶部只允许一层「当前状态」，
  历史一律降级为带日期的链接。
- **自动生成分支不入库**：`worktree-agent-*` / `claude/<随机名>` 类型的临时分支用完即删，
  不以任何理由保留。
```

同时建议在 `AGENTS.md` 工作流节镜像同一段（两个入口要一致）。

---

## 13. 回滚手册

| 阶段 | 回滚命令 |
|---|---|
| Phase 1（ff main） | `git switch main && git reset --hard 54c502e && git push --force-with-lease origin main` |
| Phase 2（合并） | `git switch main && git reset --hard thesis-20261023` |
| Phase 3（删分支） | 用 `_backup/refs-20261024.txt` 里的 sha 逐个 `git branch <name> <sha>` 重建 |
| Phase 4（worktree） | worktree 内容可从 `main` 重新 `git worktree add` 重建；**未提交内容不可恢复** ⇒ 务必先 `status --porcelain` |
| Phase 5（stash） | `git stash apply < patch 文件` |
| **全程灾难** | `git clone ~/Downloads/13/_backup/smc-full-20261024.bundle recovered/` 得到完整仓库 |

**git 的不可逆点只有一个：`git push --force`。** 本方案全程不需要 force（ff 是 fast-forward）。

---

## 14. 完整命令清单（可顺序复制）

```bash
# ===== Phase 0 备份 =====
cd /Users/williamlin/Downloads/13/spatial-meta-ce
mkdir -p ~/Downloads/13/_backup
git bundle create ~/Downloads/13/_backup/smc-full-20261024.bundle --all
git bundle verify ~/Downloads/13/_backup/smc-full-20261024.bundle
git for-each-ref --format='%(objectname) %(refname)' > ~/Downloads/13/_backup/refs-20261024.txt

# ===== Phase 1 恢复主干 =====
cd /tmp && rm -rf smc-consolidate
git clone --no-hardlinks /Users/williamlin/Downloads/13/spatial-meta-ce smc-consolidate
cd /tmp/smc-consolidate
git remote set-url origin https://gitee.com/william-forest/spatial-meta-ce.git
git fetch --all --tags --prune
git switch main
git merge-base --is-ancestor main 1892b60 && echo OK || echo STOP
git merge --ff-only 1892b60
git tag -a thesis-20261023 -m "论文交付快照：含 100-106 号文档" 1892b60
git tag -a pre-consolidation-20261024 -m "分支收敛前快照" 54c502e
git push origin main
git push origin thesis-20261023 pre-consolidation-20261024

# ===== Phase 2 合并干净分支 =====
for b in claude/project-progress-review-94eba1 claude/review-plan-changes-915c63 docs/meta65-scope-holdout-paper-20261017; do
  git switch main && git merge --no-ff "$b" -m "merge: 收敛 $b"
done
# 复核
git show main:CLAUDE.md | grep -c '默认不运行 graphify'    # 期望 1
git ls-tree -r --name-only main | grep -E '\.(pt|pth|npy|pkl)$' || echo "无产物 OK"
git ls-tree -r --name-only main | wc -l                      # ≥ 1807

# ===== Phase 2b 归档否决分支 =====
for b in audit/cache-integrity-20261016 chore/gitnexus-index chore/remove-graphify-use-gitnexus; do
  git tag -a "archive/$(echo $b | tr '/' '-')-20261024" -m "归档：未合并" "$b"
done
git push origin --tags

# ===== Phase 3 清理已合并分支 =====
git branch --merged main --format='%(refname:short)' | grep -vE '^(main|master)$' > /tmp/to_delete.txt
wc -l /tmp/to_delete.txt
git branch -d $(cat /tmp/to_delete.txt)

# 远端（分批）
git branch -r --merged main --format='%(refname:short)' | grep '^origin/' \
  | grep -vE '^origin/(HEAD|main)$' | sed 's|^origin/||' > /tmp/to_delete_remote.txt
split -l 20 /tmp/to_delete_remote.txt /tmp/rdel_
for f in /tmp/rdel_*; do xargs -a "$f" -I{} git push origin --delete "{}" || echo "skip {}"; done

# ===== Phase 4 worktree =====
cd /Users/williamlin/Downloads/13/spatial-meta-ce
git worktree prune -v
for wt in $(git worktree list --porcelain | awk '/^worktree /{print $2}'); do
  [ "$wt" = "/Users/williamlin/Downloads/13/spatial-meta-ce" ] && continue
  if [ -z "$(git -C "$wt" status --porcelain 2>/dev/null)" ]; then
    git worktree remove "$wt" --force && echo "removed: $wt"
  else echo "SKIP dirty: $wt"; fi
done

# ===== Phase 5 stash =====
git stash branch archive/refmt-pending-20261020 stash@{0}
git stash show -p stash@{1}
git stash drop stash@{1}

# ===== Phase 6 默认分支（Gitee 网页手改后） =====
git remote set-head origin main
git symbolic-ref refs/remotes/origin/HEAD
```

---

## 15. 总验收清单

执行完勾选，全部为 ✅ 才算完成：

- [ ] `git log -1 --oneline main` = `1892b60`（或含其后的收敛提交）
- [ ] `git rev-list --count main..claude/peaceful-kare-5bb350` = 0
- [ ] `git symbolic-ref refs/remotes/origin/HEAD` = `refs/remotes/origin/main`
- [ ] `git branch | wc -l` ≤ 20（从 172）
- [ ] `git ls-remote --heads origin | wc -l` ≤ 10（从 69）
- [ ] `git branch --no-merged main | wc -l` = 0（未合并的都已转成 archive tag）
- [ ] `git tag | wc -l` ≥ 5（thesis + pre-consolidation + ≥3 archive）
- [ ] `git worktree list | wc -l` ≤ 10，`du -sh .claude/worktrees` < 200 MB
- [ ] `git stash list | wc -l` ≤ 1
- [ ] `git ls-tree -r --name-only main | grep -cE '\.(pt|pth|npy|pkl)$'` = 0
- [ ] `git show main:CLAUDE.md | grep -c '默认不运行 graphify'` = 1（工具链未被误删）
- [ ] `实验文档/99-进度与下一步.md` 顶部只有一层「当前状态」
- [ ] `README.md` 日期 = 2026-10-23 之后
- [ ] 在 `/tmp/smc-consolidate` 里 `python3 -B -m unittest discover -s tests` 通过（或明确记录环境性失败）
- [ ] 项目 `CLAUDE.md` 已加入「分支必须有终点」等 4 条规则

---

## 16. 预计耗时与工作量

| 阶段 | 内容 | 耗时 | 风险 |
|---|---|---:|---|
| 0 | 备份 | 10 min | 无 |
| 1 | 恢复主干 + tag + push | 30 min | 低（ff） |
| 2 | 7 个未合并分支逐个判定 | 1.5 h | 中（3 处冲突需人工） |
| 3 | 清 165 本地 + ~60-100 远端分支 | 40 min | 低（`-d` 有保护） |
| 4 | worktree / 3 GB 清理 | 30 min | 中（dirty worktree 不可恢复） |
| 5 | 2 个 stash | 20 min | 低（先存 patch） |
| 6 | Gitee 默认分支（网页） | 10 min | 无 |
| 7 | 文档入口唯一化 | 1 h | 无 |
| 8 | 规则补丁（需独立分支提交） | 15 min | 无 |
| | **合计** | **≈ 5 小时** | |

---

## 17. 与本研究主线的关系（重要）

这次收敛**不解决研究问题**，只解决交付问题。执行后：

- **主干上能看到 102/103/106**（IMDB 对照、误差结构诊断、表示探针）——这是论文真正的结论所在
- **`experiment/target-3seed-20261023-local` 不再是孤儿**——若它确实覆盖 4 目标库 3 seeds，它是目前对抗"单 seed、6.3 pp 噪声"缺陷的唯一实证
- **`feat/meta65-source-holdout-20261018` 会作为已合并历史留在主干**——其 formal 状态仍是 `not_started`，不因收敛而改变

**不要在收敛过程中顺手跑任何实验。** 这个方案是纯 git / 文档操作，全程不需要 GPU、不需要连服务器。

---

## 附：本次实测数据来源

| 数字 | 命令 |
|---|---|
| 172 本地分支 / 69 远端 | `git branch \| wc -l`、`git branch -r \| wc -l` |
| 165 已合并 / 7 未合并 | `git branch --merged 1892b60 \| wc -l`、`git branch --no-merged 1892b60 \| wc -l` |
| main ahead/behind = 243/0 | `git rev-list --count main..1892b60` |
| 冲突预演 | `git merge-tree --write-tree 1892b60 <branch>` |
| worktree 3.1 GB / 50 项 | `du -sh .claude/worktrees`、`git worktree list \| wc -l` |
| frontier = 1892b60 | `git branch --contains` 交叉验证 3 个分支指向同一 tip |
| 1807 文件 / 530 篇实验文档 | `git ls-tree -r --name-only 1892b60 \| wc -l` |
