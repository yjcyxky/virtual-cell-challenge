# 06 仓库改造方案：数据仓库 + 实验仓库

撰写：2026-10-04，Claude。**方案草案，供你审核，没有做任何改动。**
目标：用更干净的方式重新开始实验。数据独立成一个仓库并放在 NAS 上；实验放到另外的仓库，以后可以并行开多个完全不同的实验仓库。

---

## 1. 现状事实（只读核实）

| 项 | 实测 |
|---|---|
| 当前仓库 | `/home/jy001/Downloads/virtual-cell-challenge`，348 次提交，GitHub `yjcyxky/virtual-cell-challenge`，与 `origin/master` 同步 |
| NAS 镜像 | `/mnt/projects/virtual-cell-challenge` 是整个仓库的完整副本：`sync_to_nas.py` 在 10-03 同步了 719,683 个文件、约 3.44 TB；本地大于 1 GB 的未跟踪文件已替换成指向 NAS 的软链接 |
| 原始数据 `data/raw` | 17 个来源在 NAS 镜像上都有完整副本。Xaira、CD4、Jost 只在 NAS 上，本地是目录级软链接。本地另有一些小于 1 GB 的分片文件的副本：scBaseCount 187 GB、Tahoe 28 GB、scPerturb 13 GB 等 |
| `data/assessments` | 已经是指向 NAS 的软链接 |
| `models/` | 本地 ESM-2 和 SE-600M 的权重（大文件已软链到 NAS） |
| 进行中的任务 | CD4 下载进程（PID 1929550）和收尾进程（PID 1929551）。它们按旧仓库的路径写入 `/mnt/projects/virtual-cell-challenge/data/raw/zhu2026_cd4`；收尾进程用的是旧仓库里 `experiments/init-linear/.venv` 的 Python |
| 文件系统 | `/mnt/projects` 是同一个 NFS 挂载（`nas.mycluster.on:/volume1/dgx-spark-01/projects`），同一挂载内的 `mv` 是重命名，不会拷贝数据 |
| 本地盘 | `/` 剩余 2.6 TB，NVMe；NAS 剩余 5.1 TB |
| 历史校验 | 71 个已执行节点都有 `code_snapshot_commit`；旧仓库冻结后，历史结果仍可按提交追溯 |

---

## 2. 目标结构

### 2.1 推荐三个角色

你提的是“数据仓库 + 实验仓库”。我建议把**评测与提交工具**单独做成一个小仓库 `vcc-bench`，理由见 §6 的决策 1。如果你希望严格只有两类仓库，可以把它并进 `vcc-data`。

```
NAS: /mnt/projects/vcc-data/        ← 数据仓库（git 工作区在 NAS 上；git 只管代码、登记和清单）
GitHub: vcc-bench                   ← 评测/诊断/提交库（小，按 tag 发版）
本地: ~/Projects/vcc-exp-<名字>/    ← 实验仓库，可以有多个，各自独立
旧仓库 virtual-cell-challenge       ← 冻结归档，只读
```

依赖方向是单向的，任何地方都不改 `sys.path`：

```
vcc-exp-*  ──(pyproject 依赖，钉 tag)──▶  vcc-bench  ──(按 release id + 清单哈希读数据)──▶  vcc-data 发布物
     └──────────────(按 release id + 清单哈希读数据)──────────────────────────────────────▶  vcc-data 发布物
```

### 2.2 `vcc-data`：数据仓库（NAS）

```
/mnt/projects/vcc-data/
├── README.md               现状、如何取用发布物
├── AGENTS.md               数据仓库规则：raw 和 release 都不可变；这里不训练
├── registry/               sources.json、registry.lock.json、MANIFEST.tsv、selections/   （入 git）
├── raw/<source_id>/        原始数据，不入 git；从旧镜像原地重命名接管
├── models/<model_id>/      第三方权重（ESM-2、SE-600M），不入 git，按来源登记
├── releases/<release_id>/  不可变的处理产物，不入 git；例如 vcc-pdf-v3-r1/
├── manifests/<release_id>.json   入 git：逐文件 sha256、行列数、schema、来源哈希、处理代码 commit、参数、seed
├── assessments/            历史数据评估报告，不入 git；从旧镜像接管
├── src/vccdata/
│   ├── acquire/            fetch、lock、verify、finish（来自 scripts/ 顶层的数据工具）
│   ├── io.py               H5AD/Parquet 只读读取（来自 scripts/dossier/rna.py）
│   ├── process/pdf_v3/     严格按 dataset_overview_v3.pdf 处理：各来源适配器、DE 筛选、对照截取、基因筛选、schema
│   ├── release.py          发布、校验、拉取（pull 到本地或云端缓存，带哈希缓存）
│   └── assess/             旧 dossier 评估代码，冻结，只用于复现历史报告
├── docs/                   数据卡、PDF v3 规格、缺口分析（更新后）、获取记录
├── tests/
└── pyproject.toml / uv.lock
```

**发布物的约定**

- `releases/<id>/` 一经发布就不再修改；有改动就发新 id，例如 `vcc-pdf-v3-r2`。
- 清单入 git，消费方同时钉住 `release_id` 和 `manifest_sha256`。
- 训练从**本地 NVMe 缓存**读数据，用 `vccdata pull <id>` 拉取并校验，不直接从 NFS 读。之前 NFS 上的 D 状态和慢读问题就是这么来的。
- 校验结果按 `(path, size, mtime, inode)` 缓存。这吸取了 `research.py` 每次全量重哈希 136 GB 的教训。
- 云端用同一个 `pull`，也可以用你自己的同步方式，到达后再按清单校验。

**第一批发布**

- `vcc-pdf-v3-r1`：严格按 PDF 处理。第 1 批是 H1 和 5 个公开背景加 Adamson、GSE132080；第 2 批是 Xaira；CD4 等下载完成后单独发 r2。
- `vcc-official-val1`：官方 A/B/C 输入，即现有的 `arc_vcc2026_controls`。10/22 之后另发 `vcc-official-final1`。

### 2.3 `vcc-bench`：评测、诊断与提交库

```
vcc-bench/
├── src/vccbench/
│   ├── contract.py     官方基因轴、靶点 panel、映射、输入/评分 NTC 隔离（来自 challenge.py 的契约部分）
│   ├── evalsets.py     评测集定义：PDF H1 测试集（25+75 靶 × 400 细胞）、稳健性留出背景
│   ├── official.py     固定 cell-eval2@5e648335 的 reference、anchors 和评分（来自 challenge.py、bounded_bundle、frozen_scoring）
│   ├── floors.py       零响应、靶点错配、源模板等无信息对照（来自 capability_controls）
│   ├── diagnostics.py  AGENTS.md 规定的训练后预测诊断（来自 analyze_submitted_counts、prediction_diagnostics）
│   ├── emit.py         参考发射器，只保留一个统一接口（取代三套实现）
│   ├── submit.py       生成 360k 细胞、vcc prep、提交与恢复；必须显式加 --approved 才会提交
│   └── record.py       run 记录：commit、配置哈希、数据发布 id、bench 版本、seed、六项 raw/norm、W&B
├── docs/               指标说明、r4 边界、榜单解读、KnowGraph 注意力索引
├── tests/
└── pyproject.toml / uv.lock（钉住 cell-eval2 commit）
```

- 构建出的评测 reference 和 anchors 很贵，历史上发生过 OOM。它们作为数据发布物放进 `vcc-data/releases/vcc-evalsets-r1/`，与 bench 版本绑定，各实验仓库共用，不用每个仓库重建。
- 版本规则：bench 改变评分口径就升主版本；不同实验仓库只有在 bench 主版本相同时，分数才能直接比较。

### 2.4 `vcc-exp-<名字>`：实验仓库（本地，可以有多个）

由 bench 提供的模板一键生成，例如 `vccbench new-exp transfer`：

```
vcc-exp-transfer/
├── AGENTS.md         冲刺契约（1 页）：每轮先 grill-me 对齐；干净工作区才允许启动；预测先诊断再评分；官方提交必须你批准
├── pyproject.toml / uv.lock   依赖 vcc-bench（钉 tag），模型代码自由
├── configs/          完整配置（YAML）
├── src/<pkg>/        模型代码
├── run.sh            唯一入口：环境同步 → pull 数据发布 → 训练 → 预测 → 诊断 → 评分 → 记录
├── runs/             不入 git，放本地 NVMe；完成后大产物归档到 /mnt/projects/vcc-runs/<repo>/<run_id>/
├── results.jsonl     入 git：每个 run 追加一行（commit、配置哈希、数据 release、bench 版本、seed、分数、W&B 链接）
├── champion.json     当前冠军及其各级评估分数
└── docs/PLAN.md、docs/rounds/round-N.md   每轮的假设、事先写下的预测、结论与决定
```

- **不再使用** Method Space、DAG、Ledger 三件套和逐 run 提交。保留下来的核心可复现要素：commit、完整配置、数据发布哈希、bench 版本、seed、环境锁、W&B。
- 允许并行跑多个 run。
- W&B：沿用 `yjcyxky/virtual-cell-challenge` 项目，group 用仓库名，run id 用 `<repo>-<run_id>`。
- 第一个实验仓库是 `vcc-exp-transfer`，承接 04 路线的第 1、2 轮。大模型线以后另开 `vcc-exp-setmodel`，与它互不干扰。

### 2.5 旧仓库的去向

- 先把 `docs/agent/` 提交上去，打 tag `legacy-2026-10-04`，然后冻结：不再新增 run，也不再做清理。05 清单里的旧仓库内项目全部作废，不用执行。
- 有价值的内容**复制**到新仓库，旧仓库保持原样，可以追溯：
  - `03_alignment.md` 和 `04_roadmap.md` 进 `vcc-exp-transfer/docs/`；
  - 数据文档和 PDF v3 进 `vcc-data/docs/`；
  - 指标文档和 KnowGraph 进 `vcc-bench/docs/`。
- 旧仓库里 Method Space、DAG、Ledger 中的结论（例如 Shared 是强参照、DepMap 有信号但没有晋级），以一页“历史证据摘要”的形式带进新实验仓库的 PLAN，注明来自哪个旧提交。

---

## 3. 内容去向表

| 旧位置 | 新位置 | 处理方式 |
|---|---|---|
| `data/sources.json`、`registry.lock.json`、`MANIFEST.tsv`、`selections/` | `vcc-data/registry/` | 复制；README 重写成现状 |
| NAS `…/virtual-cell-challenge/data/raw/<src>`（CD4 以外） | `vcc-data/raw/<src>` | NAS 上原地重命名；旧路径留兼容软链接 |
| NAS `…/data/raw/zhu2026_cd4` | 同上 | **等下载和收尾都结束后**再迁移 |
| NAS `…/data/assessments` | `vcc-data/assessments/` | 重命名，加兼容软链接 |
| `models/` | `vcc-data/models/` | 重命名或复制，并登记来源 |
| `scripts/fetch_data.py`、`build_lock.py`、`fetch_vcc_controls.py`、`finish_crispri_acquisition.py`、`audit_data_inventory.py` | `vcc-data/src/vccdata/acquire/` | 迁移；硬编码路径改为可配置 |
| `scripts/dossier/rna.py` | `vcc-data/src/vccdata/io.py` | 迁移并补测试 |
| `scripts/dossier/*`（其余） | `vcc-data/src/vccdata/assess/` | 原样冻结，只用于复现历史报告 |
| `scripts/dossier/challenge.py` | `vcc-bench/contract.py` 和 `official.py` | 拆分迁移 |
| `src/vcc_task/*`（评分、诊断、发射、提交相关） | `vcc-bench` | 重构迁移，合并成一套生命周期 |
| `scripts/analyze_submitted_counts.py` | `vcc-bench/diagnostics.py` | 统计函数迁移，CLI 变成薄壳 |
| `src/vcc_mechanism/*`、`experiments/*`、`scripts/research.py`、三份研究对象 | 留在旧仓库 | 冻结，不迁移 |
| `docs/research/challenge_2026/*`（基因轴、映射、划分） | `vcc-bench`（契约）+ `vcc-data`（审计） | 复制 |
| `docs/dataset_overview_v3.pdf`、v3 缺口分析、`docs/datasets/*` | `vcc-data/docs/` | 复制；状态段落更新 |
| 指标、r4、榜单解读文档，KnowGraph | `vcc-bench/docs/` | 复制；9/29 的时效数字加日期说明 |
| `docs/agent/01–06` | 留在旧仓库；03、04 复制到 `vcc-exp-transfer` | — |
| `scripts/status.sh`、`requirements-lock.txt`、`scripts/sync_to_nas.py` | 不迁移 | 新仓库不再需要 |

---

## 4. 迁移步骤

| 步骤 | 内容 | 验证 | 耗时（估） |
|---|---|---|---|
| M0 冻结 | 旧仓库提交 `docs/agent/`，打 tag `legacy-2026-10-04` 并 push | tag 存在，工作区干净 | 10 分钟 |
| M1 建 `vcc-data` | 在 `/mnt/projects/vcc-data` 执行 `git init`；导入登记、工具、文档；新建 GitHub 私有远端（只放代码和清单） | 测试通过；`vccdata status` 能列出全部来源 | 2–3 小时 |
| M2 接管 raw | NAS 上把 16 个来源（CD4 以外）、`assessments`、`models` 重命名进 `vcc-data`；旧路径留兼容软链接 | 逐文件比对大小和 inode（重命名不改内容）；抽样复算 sha256，与 `registry.lock` 对照 | 30 分钟（加后台抽样校验） |
| M3 建 `vcc-bench` | 拆分迁移契约、评分、诊断、发射、提交和记录模块，并补测试；钉住 cell-eval2 | 用旧仓库的一个已知结果做回归：对同一份预测文件，新旧评分器给出的六项分数逐项一致（例如 `masked-response-qc-s01` 的本地 H1 分数） | 0.5–1 天 |
| M4 发数据 | `vccdata` 的 PDF v3 流水线 → `vcc-pdf-v3-r1`（第 1 批，再第 2 批） | 对照 PDF 里各数据集“处理后”的数字；清单和哈希 | 约 1 天（同 04 路线的 R0.2；DE 计算是大头） |
| M5 评测集 | `vcc-evalsets-r1`：H1 PDF 测试集和稳健性留出背景的 reference 与 anchors；无信息下限；噪声基线 | 零响应 ≈ 0；真实复制 > 1；Shared > 零响应可以复现 | 约 0.5 天（同 R0.3、R0.4） |
| M6 实验仓库 | 用模板生成 `vcc-exp-transfer`，写入 03、04 和历史证据摘要；跑通一个 Shared 基线 run | `run.sh` 从数据拉取到评分全程跑通，`results.jsonl` 有记录 | 2–3 小时 |
| M7 交云 | 你把 `vcc-pdf-v3-r1` 同步到云端；在云端执行 `vccdata pull --verify` | 清单校验通过 | 取决于带宽（压缩后约 30–60 GB） |
| M8 CD4 | 下载和收尾完成后，把 CD4 迁入 `vcc-data`，处理后发 `vcc-pdf-v3-r2` | 同 M2、M4 | 下载完成后约 0.5 天 |

**对 04 路线时间表的影响**【推断】

- 04 里的 R0.1（修补 `research.py`、改 AGENTS.md）被 M1、M3、M6 取代，**不再做**。
- R0.2–R0.4 原样进入 M4、M5。
- 第 0 轮总体大约多出 0.5–1 天，主要是 M3 的拆分迁移和回归验证。10/12 达到 0.155 的里程碑变紧，但仍然可行。

---

## 5. 风险与应对

| 风险 | 应对 |
|---|---|
| CD4 下载还在写旧路径，收尾进程依赖旧仓库的环境 | CD4 目录和旧仓库都先不动，等两个进程退出后再迁（M8） |
| 重命名后，旧仓库的软链接、Ledger/DAG 里的路径失效 | 旧位置留兼容软链接；旧仓库已冻结，不再运行 `check`。历史仍可通过 tag 和兼容链接读取 |
| 拆分迁移引入评分偏差 | M3 用旧仓库已知结果做逐项回归，不一致就不进入 M5 |
| NFS 读得慢、卡在 D 状态 | 训练一律从本地 NVMe 缓存读；NAS 只用于存放和归档 |
| 新仓库重新长成一个庞大的协议体系 | 实验仓库的 AGENTS.md 限 1 页；登记只保留 `results.jsonl`、`champion.json` 和每轮文档 |
| Xaira 的 NC-SA 许可 | 数据仓库的 GitHub 远端设为私有，不放任何数据文件 |
| 迁移占用冲刺时间 | M1–M3 与 M4 的流水线开发可以并行；M3 只迁移冲刺需要的模块，其余留在旧仓库 |

---

## 6. 需要你决定的事项

| # | 决策 | 推荐 | 备选 |
|---|---|---|---|
| 1 | 评测库放在哪里 | **单独建 `vcc-bench` 仓库**：评测口径独立发版，多个实验仓库钉同一版本，分数才可比；云端可以直接从 GitHub 安装 | 并进 `vcc-data`，严格只有数据和实验两类仓库 |
| 2 | 新仓库的 git 历史 | **全新初始化**，附一个 `PROVENANCE.md` 记录旧仓库 tag 和对应 commit | 用 `git filter-repo` 把相关路径的历史抽出来带过去（更慢，历史也更乱） |
| 3 | 远端托管 | **三个仓库都建 GitHub 私有远端**，只放代码、登记和清单 | `vcc-data` 只在 NAS 上做 bare 仓库，不上 GitHub |
| 4 | raw 数据接管方式 | **在 NAS 上重命名进 `vcc-data`，旧路径留兼容软链接** | 不移动，`vcc-data/raw` 用软链接指向旧位置（零风险，但所有权不清） |
| 5 | 实验仓库的管理方式 | **轻量：`results.jsonl` + `champion.json` + 每轮 PLAN/REPORT** | 把 `research.py` 和三件套移植过来（更严格，但慢） |
| 6 | 实验仓库和 run 产物的位置 | **仓库放 `~/Projects/vcc-exp-*`；run 放本地 NVMe；完成后归档到 NAS `/mnt/projects/vcc-runs/`** | 全部放 NAS（简单，但慢） |
| 7 | 命名 | `vcc-data`、`vcc-bench`、`vcc-exp-<名字>` | 你指定 |
| 8 | 旧仓库本地的数据副本（scBaseCount 187 GB 等） | **NAS 上校验通过后再决定是否删除**，本次不删 | 立即清理 |

你批准或修改这份方案后，我从 M0 开始执行。每一步按冲刺契约先和你对齐关键事项；凡是建 GitHub 仓库、在 NAS 上重命名、删除这类操作，都会单独确认。
