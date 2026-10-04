# 07 迁移操作手册：拆成 vcc-data / vcc-bench / vcc-exps

撰写：2026-10-04，Claude。依据你的目录决定。本手册中的**目录位置**以这里为准，取代 06 §2 里的位置描述；06 中的仓库职责和内容去向表仍然有效。

## 0. 目标与约定

```
NAS  /mnt/projects/virtual-cell-challenge/
├── vcc-data/      数据仓库：git 管代码、登记、清单；raw/ models/ assessments/ releases/ 不入 git
├── vcc-bench/     评测、诊断、提交库（git）
├── vcc-exps/      实验仓库的 NAS 副本（由你的同步脚本从本地同步过来）
└── vcc-legacy/    旧仓库，冻结（原来 NAS 根目录下的全部内容挪到这里）

本地 ~/Downloads/virtual-cell-challenge/
├── vcc-exps/      计算都在这里跑；未进版本控制的文件由你用脚本同步到 NAS 的 vcc-exps/
└── vcc-legacy/    （可选）旧仓库本地副本，NAS 校验完成后可删除
```

有一点我先按如下理解，你如果不是这个意思告诉我就行：**`vcc-exps/` 是一个容器目录，里面每个实验是一个独立的 git 仓库**，例如 `vcc-exps/transfer/`、`vcc-exps/setmodel/`，这样对应你说的“多个完全不同的实验仓库”。如果你希望 `vcc-exps` 本身就是一个仓库，只有 P3.4 和 P5.3 这两步会变。

每一步都标了执行人：**你**，或者 **Claude**（你确认后执行）。所有移动都是同一 NFS 挂载内的重命名，可以逆向恢复；**整个过程不删任何东西**，删除只放在最后的可选步骤 P8。

---

## P1 冻结旧仓库（约 15–20 分钟）｜你，或 Claude 代做

```bash
cd ~/Downloads/virtual-cell-challenge
git status --short            # 应只看到 ?? docs/agent/ 和 ?? docs/dataset_overview_v3.pdf
git add docs/agent docs/dataset_overview_v3.pdf
git commit -m "Freeze legacy: agent analyses, restructure plan and dataset overview v3"
#   pre-commit 会跑 research.py check --staged，大约 10 分钟（就是 02 审计 C4 说的重哈希问题），等它跑完
git tag -a legacy-final-2026-10-04 -m "Legacy repo frozen before vcc-data/vcc-bench/vcc-exps split"
git push origin master --follow-tags
```

**检查点**：`git status` 是干净的；GitHub 上能看到这个 tag。

**从这一刻起，旧仓库不再新增任何工作。**

---

## P2 停掉 CD4，最后同步一次旧仓库｜CD4 已停；同步由你执行

**CD4 已按你的要求在 2026-10-04 约 15:47 UTC 停止。**

- 先停收尾进程 1929551，再停下载器进程组 1929550（含 2 个 curl）。都是 SIGTERM，正常退出，没有残留进程。
- 停止时的状态：

  | 项 | 数值 |
  |---|---|
  | 已完成文件 | 55/57 个，共 1,646,103,209,448 字节 |
  | `D1_Stim48hr.assigned_guide.h5ad` | 停在 56,251,482,112 字节 |
  | `D4_Stim48hr.assigned_guide.h5ad` | 停在 62,044,413,952 字节 |
  | 剩余待下 | 约 197.6 GB |

- 状态文件 `acquisition.json` 仍是 `waiting_for_download`，没有写入错误。
- 两个未完成文件会用 `curl -C -`（带 `If-Match` ETag）从当前字节续传。停止前的文件大小已记录，续传时用来核对。

**续传放在 P6**：整理完成后，从 `vcc-data` 里用迁移后的下载工具恢复，目标为 `vcc-data/raw/zhu2026_cd4`；见 P6 的 CD4 一行。

**最后一次旧仓库同步**：仍然用原来的目标，把 10-03 之后的提交、kNN 官方产物和 `docs/agent` 带到 NAS。

```bash
cd ~/Downloads/virtual-cell-challenge
/home/jy001/micromamba/envs/virtual-cell/bin/python scripts/sync_to_nas.py --dry-run
/home/jy001/micromamba/envs/virtual-cell/bin/python scripts/sync_to_nas.py
```

**检查点**：同步日志以 `completed` 结束。估计 0.5–1 小时，是增量同步。

**注意：此后不要再用默认目标运行这个脚本。** 它的默认目标就是 NAS 根目录，重组之后再跑，会把旧仓库的结构写回根目录。

---

## P3 建新仓库骨架（现在就可以做，不碰旧路径）｜Claude 做，你确认远端

### P3.1 在 GitHub 建私有远端

只放代码、登记和清单，**不放任何数据**（Xaira 是 NC-SA 许可）。

```bash
GH=/home/jy001/micromamba/envs/llm-ft/bin/gh
$GH repo create yjcyxky/vcc-data  --private
$GH repo create yjcyxky/vcc-bench --private
$GH repo create yjcyxky/vcc-exp-transfer --private   # 第一个实验仓库
```

### P3.2 vcc-data 骨架

```bash
NAS=/mnt/projects/virtual-cell-challenge
mkdir -p $NAS/vcc-data && cd $NAS/vcc-data && git init -b main
mkdir -p registry manifests src/vccdata docs tests raw models assessments releases
printf 'raw/\nmodels/\nassessments/\nreleases/\ncache/\n.venv/\n__pycache__/\n' > .gitignore
#   导入登记：data/sources.json、registry.lock.json、MANIFEST.tsv、selections/ → registry/（复制，不移动）
#   导入工具和文档，按 06 §3 的去向表；README 和 AGENTS.md 重写
git add -A && git commit -m "Initialize vcc-data: registry, acquisition tools and data docs from legacy-final-2026-10-04"
git remote add origin git@github.com:yjcyxky/vcc-data.git && git push -u origin main
```

### P3.3 vcc-bench 骨架

```bash
mkdir -p $NAS/vcc-bench && cd $NAS/vcc-bench && git init -b main
#   pyproject 钉 cell-eval2@5e64833518a6603a0301cbe28185d49c30f4a986；
#   src/vccbench 和测试的迁移在 P6 完成
```

### P3.4 vcc-exps 容器与第一个实验仓库

```bash
mkdir -p $NAS/vcc-exps
#   第一个实验仓库先在 NAS 上初始化并推到 GitHub；P5 再 clone 到本地开始计算
```

**检查点**：
- 三个目录里 `git status` 都是干净的，远端可以 push；
- NAS 根目录暂时会同时有旧内容和 `vcc-*`，这是过渡状态。

---

## P4 NAS 重组（P2 的最后同步完成后，约 30 分钟）｜Claude 执行，你逐步确认

### P4.1 把旧内容整体挪进 vcc-legacy

只挪下面这 14 个具名条目，不用通配，免得误伤 `vcc-*`：

```bash
NAS=/mnt/projects/virtual-cell-challenge
cd $NAS && mkdir vcc-legacy
for e in AGENTS.md data docs experiments knowledgebase models scripts src requirements-lock.txt \
         .git .gitattributes .githooks .gitignore .nas-sync; do
  mv -- "$e" vcc-legacy/ && echo "moved $e" >> vcc-legacy/MOVE_LOG.txt
done
ls -A $NAS        # 应只剩：vcc-bench vcc-data vcc-exps vcc-legacy
```

### P4.2 vcc-data 接管原始数据、评估报告和模型

先在 `vcc-data` 里做实体重命名，再在旧位置留相对软链接，旧仓库内部的路径仍然能用：

```bash
cd $NAS/vcc-legacy/data/raw
for s in *; do
  mv -- "$s" ../../../vcc-data/raw/"$s" && ln -s ../../../vcc-data/raw/"$s" "$s"
done
mv $NAS/vcc-legacy/data/assessments $NAS/vcc-data/assessments \
  && ln -s ../../vcc-data/assessments $NAS/vcc-legacy/data/assessments
for m in $NAS/vcc-legacy/models/*; do
  n=$(basename "$m"); mv -- "$m" $NAS/vcc-data/models/"$n" && ln -s ../../vcc-data/models/"$n" "$m"
done
```

（`mv` 前要先删掉 P3.2 里建的空目录 `raw`、`assessments`，或者改成往里面逐个移动。到时我会按实际状态给出精确命令。）

**检查点**（Claude 用脚本核对，不重算哈希；重命名不会改动内容）：
- `registry.lock.json` 里的每个文件在 `vcc-data/raw/` 下都存在，大小一致；
- 17 个来源的文件数与重组前一致；
- `vcc-legacy/data/raw/<src>` 都能解析到 `vcc-data/raw/<src>`；
- 用 `vcc-legacy/MOVE_LOG.txt` 可以逐条逆向恢复。

---

## P5 本地重组（P2 完成后，约 10 分钟）｜你，在 Claude 会话之外执行

当前 Claude 会话的工作目录就是这个旧仓库。**请退出会话后再执行这一步。**

```bash
cd ~/Downloads
mv virtual-cell-challenge vcc-legacy-local            # 旧仓库整体挪开，此时没有进程在里面跑
mkdir virtual-cell-challenge
mv vcc-legacy-local virtual-cell-challenge/vcc-legacy  # 可选：保留在新容器里；不想保留就放到别处
mkdir -p virtual-cell-challenge/vcc-exps
cd virtual-cell-challenge/vcc-exps
git clone git@github.com:yjcyxky/vcc-exp-transfer.git transfer
```

**说明**

- 以后**从 `~/Downloads/virtual-cell-challenge` 启动 Claude Code**。这个路径没变，Claude 的跨会话记忆还能接上；从 `vcc-exps/transfer` 里启动则会换成一套新的记忆。
- 旧仓库挪动后，里面的 `.venv` 入口脚本、指向旧 NAS 根目录的大文件软链接都会失效。旧仓库已冻结，不再运行；以 NAS 上的 `vcc-legacy` 为准。

**检查点**：`~/Downloads/virtual-cell-challenge` 下只有 `vcc-exps/`（以及可选的 `vcc-legacy/`）。

---

## P6 填充新仓库｜Claude（就是 04 路线的第 0 轮，按冲刺契约每步先对齐）

| 仓库 | 首批内容 | 验收 |
|---|---|---|
| vcc-data | 获取和校验工具、`io`（取自 `rna.py`）、PDF v3 流水线 → `releases/vcc-pdf-v3-r1`，清单进 `manifests/`；`vccdata pull` 和带缓存的校验 | 与 PDF“处理后”的数字逐项对照；清单哈希 |
| vcc-data：CD4 续传 | 工具迁入、登记改写为 `vcc-data/raw/zhu2026_cd4` 之后，用断点续传补齐 2 个未完成文件（约 197.6 GB）；校验后建索引；处理后发 `vcc-pdf-v3-r2` | 续传起点 ≥ P2 记录的停止大小；最终大小与 lock 一致；新的收尾记录写入新的 assessment 目录，旧记录保留 |
| vcc-bench | 契约、评分（cell-eval2 固定）、无信息下限、预测诊断、统一发射器、提交（必须带 `--approved`）、run 记录、新实验模板；评测集作为数据发布物 `vcc-evalsets-r1` | 对同一份预测文件，新评分与旧仓库的已知结果逐项一致；打 tag `v0.1.0` |
| vcc-exps/transfer | 1 页的 AGENTS.md（冲刺契约 + 每轮 grill-me）；`pyproject` 依赖 `vcc-bench`（钉 tag）；`run.sh`；`results.jsonl`；`docs/`（含 03、04 和历史证据摘要） | `run.sh` 从拉数据到评分全程跑通一个 Shared 基线，并写入 `results.jsonl` |

**数据缓存**：本地放在 `~/Downloads/virtual-cell-challenge/vcc-exps/.data-cache/<release_id>/`，各实验仓库共用。**你的同步脚本要排除这个目录**，它只是 NAS 发布物的副本。

---

## P7 调整你的同步方式｜你

每个实验仓库单独同步。脚本要求源目录是 git 仓库的根，目标不能与源互相嵌套：

```bash
# 示例：沿用 sync_to_nas.py（P5 之后在 vcc-legacy/scripts/ 下，P6 会复制一份到实验模板里）
python sync_to_nas.py \
  --source      ~/Downloads/virtual-cell-challenge/vcc-exps/transfer \
  --destination /mnt/projects/virtual-cell-challenge/vcc-exps/transfer \
  --dry-run
```

- 已跟踪的文件走 git push；未跟踪的 `runs/` 等由你的脚本同步。
- 排除 `.data-cache/`。
- 不要再对 `/mnt/projects/virtual-cell-challenge`（根目录）运行任何同步。

---

## P8 收尾与可选清理｜你决定

1. **核对清单**：
   - NAS 根目录只有 4 个 `vcc-*` 目录；
   - 三个新仓库的远端都已推送；
   - `vcc-data` 的数据完整性检查通过；
   - 实验仓库端到端跑通。
2. **可选，单独确认后才做**：确认 NAS 上的 `vcc-legacy` 和 `vcc-data` 完整之后，删除本地 `~/Downloads/virtual-cell-challenge/vcc-legacy`，可以释放本地约 250 GB。NAS 副本保留。

---

## 时间线（估计）

| 时间 | 步骤 |
|---|---|
| 今天 | P1 冻结 → P2 最后同步（CD4 已停）→ P3 骨架 → P4 NAS 重组 → P5 本地重组 |
| 10/05–10/07 | P6，即第 0 轮：数据发布、bench、第一个实验仓库；P7 同步调整 |
| 之后 | 按 04 路线进入第 1 轮 |

## 现在需要你确认的

1. `vcc-exps` 是“容器 + 每个实验一个仓库”（我的理解），还是一个仓库？
2. 三个 GitHub 私有远端（`vcc-data`、`vcc-bench`、`vcc-exp-transfer`）可以建吗？
3. P1 冻结旧仓库，可以现在就做吗？由我代做，还是你自己做？
