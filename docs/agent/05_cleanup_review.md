# 05 清理审核清单：docs / data / scripts / src

撰写：2026-10-04，Claude。范围按你的要求限定为 `docs/`、`data/`、`scripts/`、`src/`，不含 `experiments/`。**只列不改**，每一项等你批准后再动。

## 处置代号

| 代号 | 含义 |
|---|---|
| DEL | 删除 |
| ARC | 移到 `docs/archive/2026-10-04/`，加一行“已被 X 替代”横幅 |
| BAN | 原地顶部加一行横幅：截至何时过时，现行内容见哪里 |
| REV | 修订相关段落 |
| KEEP | 保留不动 |
| OBJ | 研究对象条目：标注已替代，或关闭为“不再执行”（按冲刺契约走） |
| REF | 代码重构 |

## 引用约束

- **[H]**：该文件被 Ledger 或 DAG 用 sha256 绑定。原地修改会让 `research.py check` 失败。
  - 默认做法：不改原文，只在 `docs/README.md` 新增的“过时段落索引”里登记；
  - 如果一定要改原文，先按 RESEARCH.md 的流程把这些引用固定到 git commit。
- **[P]**：只有固定到 git commit 的引用，可以直接改。
- **无标记**：没有被引用，可以直接处置。

所有引用数都来自本次对三份 JSON 的统计。

---

## 1. `src/` 与 `scripts/`：为什么并存，哪里不合理

### 1.1 现状

| 目录 | 实际承担的角色 | 主要文件 |
|---|---|---|
| `scripts/` 顶层 | 数据获取与登记 | `fetch_data.py`、`build_lock.py`、`finish_crispri_acquisition.py`、`audit_data_inventory.py`、`fetch_vcc_controls.py`、`sync_to_nas.py` |
| `scripts/` 顶层 | 研究管理 CLI，**同时被训练代码当库导入** | `research.py` |
| `scripts/` 顶层 | 提交诊断 CLI，**同时被 `src` 当库导入** | `analyze_submitted_counts.py` |
| `scripts/dossier/` | 80 个文件、1.1 万行。其中 75 个只服务于 09-19 的数据评估报告（`data/assessments`） | `profile_*`、`compose_*`、`render*` 等 |
| `scripts/dossier/` | 另外 5 个模块被 `src/` 和实验**当库导入** | `rna.py`（被导入 23 处）、`challenge.py`（13 处，数据与评测契约）、`profile_responses.py`（13 处）、`heterogeneity*.py` |
| `src/vcc_mechanism` | 09-28 机制整包的运行框架；只服务 5 个已关闭的实验 | `runner`、`population`、`learning`、`diagnostics` |
| `src/vcc_task` | 09-30 之后的观测、评估、DepMap 与提交框架 | `run_context`、`response_experiment`、`population_emission`、`prediction_diagnostics` 等 |
| `experiments/init-linear/src` | **事实上的共享库**：发射器、评估、提交 IO。被 6 个其他实验和 `src/vcc_task/depmap_submission.py` 导入 | 不在本次清理范围，但影响结构判断 |

依赖方向是反的：

```
experiments/* ──sys.path──▶ src/ ──sys.path──▶ scripts/、scripts/dossier/
      │                     │
      └──────sys.path──────▶ experiments/init-linear/src ◀── src/vcc_task/depmap_submission.py
```

【已核实】`sys.path` 注入分布：`src` 里 2 处；实验里至少 8 处；`scripts/tests` 里约 30 处。命令：`grep -rn "sys.path" src scripts experiments/*/src`。

### 1.2 为什么会同时存在

代码是按阶段堆积起来的，每个新阶段都把共享代码放在当时顺手的位置，再用 `sys.path` 引用前一阶段的代码，从来没有回头整理。

【已核实，`git log` 首次提交日期】
- 09-19：数据获取与 `rna.py`，进了 `scripts/`。
- 09-27：`research.py`。
- 09-28 同一天：
  - `challenge.py`（放在 `scripts/dossier`，因为它复用 `rna.py` 和那个锁定了 cell-eval2 的 uv 项目）；
  - `init-linear/src/evaluation.py`；
  - `src/vcc_mechanism`。
- 09-30：`analyze_submitted_counts.py` 和 `src/vcc_task`。

AGENTS.md 写的是“src 服务正式训练/评估，scripts 只放有持续用途的通用工具”，实际做法与之不符。

### 1.3 不合理之处

| # | 问题 | 证据 | 风险 | 建议 |
|---|---|---|---|---|
| U1 | 分层倒置：库（`src`）依赖工具（`scripts`）和某个实验的代码；全靠修改 `sys.path`；共享代码没有作为依赖写进各实验的 `pyproject`，`uv.lock` 覆盖不到它，只能靠 `code_refs` 哈希补救 | 见 1.1 | 高：任何改动都会牵连所有实验，环境复现也不完整 | REF：见 1.4 |
| U2 | “实验相互独立”在代码层面不成立：`init-linear/src` 是隐性共享库 | `experiments/{masked-response,composition-response,conditional-cell-response,distribution-response,gene-conditioned-response,response-transport}/src/runtime.py`；`src/vcc_task/depmap_submission.py` 第 56–57 行 | 中：修改 init-linear 会改变其他实验的行为 | REF：共享函数迁进 `src`；旧实验按各自快照提交复现 |
| U3 | 两套运行生命周期，三套发射器，两套 DE 诊断 | 发射器：`init-linear/src/evaluation.py`（log-shift 加随机取整）、`vcc_mechanism/population.py`、`vcc_task/population_emission.py`。DE 诊断：`vcc_mechanism/diagnostics.py`（Welch）和官方路径 | 中：新实验不知道该用哪一套，官方提交之间生成器也不一致（02 审计 H6） | REF：统一成一个生命周期、一个发射器接口；`vcc_mechanism` 冻结为 legacy |
| U4 | 数据处理没有统一产物：`data/processed` 是空的；每个实验从 raw 重新扫描（首次约 14 分钟），在自己的 cache 里重建。数据定义分散在 `challenge.py`（映射、NTC 池、S0–S4）、`vcc_task/observations.py` 和各实验 cache 中。接下来的 PDF 流水线会成为第三套定义 | 目录检查；02 审计 H3 | 高：训练集口径会不一致 | REV：明确分工——`data/processed/<子目录>` 是训练语料（PDF）；`challenge.py` 只保留评测契约（基因轴、NTC 池、scorer） |
| U5 | 训练代码在运行时下载外部数据：`full_reference.py` 第 19–34 行从 figshare 把 **DepMap 24Q2** 下载进 run cache，绕过 `sources.json` 和 `registry.lock`；已登记的 `data/raw/depmap_24q4` 反而没有用到 | 代码；`experiments/depmap-response/configs/*.json` 的 `external_prior` 字段 | 中：两个 DepMap 版本并存，数据血缘断开 | REV：把 24Q2 正式登记为数据源，或者新 run 统一改用 24Q4 |
| U6 | 工具和库混在同一个文件里：`research.py`（CLI，同时提供 bind/digest/execution_metadata 给训练代码）、`analyze_submitted_counts.py`（CLI 只能扫已发布提交，统计函数却被 `src` 复用）、`profile_responses.py`（名字是 profile 脚本，实际被 13 处当库用） | import 统计 | 中：改 CLI 就会动到训练路径 | REF：库函数迁入 `src`，`scripts` 只保留 CLI 外壳 |
| U7 | 硬编码的环境路径，有一个已经失效 | `scripts/research.py` 第 1 行 shebang、第 34 行 `ARTIFACT_MIRROR`；`sync_to_nas.py` 第 25 行；`dossier/publish_assessment.py` 第 18 行（用的是 llm-ft 环境里的 gh）；`dossier/convert_seurat_cache.py` 第 19 行（biominer 环境的 R）；`scripts/status.sh` 调用的 `.venv/bin/python` 已不存在 | 低到中 | REV 或 DEL：见第 4 节 |
| U8 | 已知运行缺陷 | `src/vcc_task/common.py` 第 18–27 行，NAS 软链接会导致 `relative_to` 报错；`src/vcc_task/run_context.py` 第 58 行，commit 在 run 完成时才读取 | 高：复用迁移到 NAS 的产物或恢复训练会失败 | REV（已列入路线 R0.1） |
| U9 | `scripts/dossier` 里 75 个一次性评估复现模块和 5 个核心库模块混放；整个 dossier 是一个独立的 uv 项目 | 1.1 | 低：主要是可读性 | 核心模块迁出后，dossier 冻结为“历史数据评估复现代码” |

**有利条件**【已核实】：71 个已执行节点**全部**设置了 `code_snapshot_commit`。所以重构 `src` 和 `scripts` 不会破坏历史 run 的校验，它们会到当时的提交里读代码。

### 1.4 建议的目标结构（待批准，不实施）

- **`src/vcc/` 合成一个包**，按职责分模块：

  | 模块 | 内容 |
  |---|---|
  | `io` | `rna` 读取器 |
  | `contract` | 官方基因轴、映射、NTC 池、scorer 配置（取自 `challenge.py`） |
  | `data` | 读取 `data/processed` |
  | `eval` | 官方评分、bundle、诊断统计（取自 `analyze_submitted_counts.py` 和 `prediction_diagnostics`） |
  | `emit` | 统一的发射器 |
  | `models` | 模型 |
  | `run` | 单一运行生命周期，是对 `research.bind` 的薄封装 |
  | `submit` | 导出与 `vcc prep` |

- **`scripts/` 只放 CLI**：数据获取与登记、research CLI、新的 PDF 数据处理 CLI、dossier（冻结）。
- **新实验**通过 `pyproject` 的路径依赖安装 `src` 包，不再改 `sys.path`。
- **旧实验**不动，按各自的快照提交复现。
- **范围建议**：冲刺期只做最小重构，即新包骨架，加上 U4、U5、U7、U8 的修复。legacy 代码原地冻结，不搬。

---

## 2. `docs/` 清理清单

### 高风险：会直接误导下一轮的口径或优先级

| # | 位置 | 过时或冲突的内容 | 依据 | 处置 |
|---|---|---|---|---|
| D1 | `AGENTS.md` 第 28 行（闭环第 1 步）、第 74 行 | 逐 run 登记、`node_id = run_id`、每次拟合一个节点 | 已被 Q2 冲刺契约替代 | REV（属于 R0.1，你审 diff） |
| D2 | `AGENTS.md` 第 83 行 | “共享 data/ 只读” | Q6 授权写入 `data/processed/<子目录>` | REV：写明例外 |
| D3 | `AGENTS.md` 第 68 行 | “五背景轮换与研究留出分别报告”作为正式评估 | 已被 Q7（H1 PDF 测试集为主，加留出背景）替代 | REV |
| D4 | `docs/RESEARCH.md` 第 27 行 [P] | “Pseudo-VCC 先冻结五背景轮换留出及全局靶点留出……” | Q7 | REV |
| D5 | `docs/RESEARCH.md` 第 31 行 [P] | “每次独立拟合占一个 run……ready 前须按外层划分展开节点” | Q2 | REV |
| D6 | `docs/RESEARCH.md` 第 57 行 [P] | “DepMap 监督对齐基线作为下一轮建模候选，仍为未验证设计” | DepMap 42 个 fit 已于 10-01 关闭 | REV |
| D7 | `docs/research/task_constraints_review.md` 第 84–90 行 [H] | “下一轮资源顺序……本轮训练与 leaderboard 提交预算为零……DepMap 首轮” | 三项都已完成；读起来像当前队列 | 登记进过时索引（或固定引用后加横幅） |
| D8 | `docs/research/data_evaluation_decision.md` 第 38、89 行 [H] | “不启用按效率硬筛选”“正式评价使用五背景 S2 LOCO、S3……S4” | 与 Q6（严格按 PDF 按 DE 数筛扰动）和 Q7 冲突 | 同上，并在冲刺契约里写明被替代 |
| D9 | `docs/research/data_evaluation_decision.md` 第 123–129 行 [H] | 状态表：“NTC 总体校准 UNTESTED”“全折 bundle 尚未生成”“DepMap 尚未完成” | 三项分别已是 signal、supported、closed | 同上 |
| D10 | `docs/research/challenge_protocol.md` 第 25 行 [H] | “不使用响应强度……筛选本轮任务” | 与 Q6 的 PDF 筛选冲突；它仍然是**评测**契约，训练语料改用 PDF | 同上，并在冲刺契约里写明分工（U4） |
| D11 | `docs/dataset_overview_v3.pdf` | **未被 git 跟踪**，但它现在是数据处理规格（Q5、Q6） | `git status` | 入库（git add） |

### 中风险：状态过时或是旧版本，可能被当成现状引用

| # | 位置 | 内容 | 依据 | 处置 |
|---|---|---|---|---|
| D12 | `docs/README.md` 第 5–18 行 | “当前研发入口”没有列 `docs/agent/`，也没有冲刺契约 | — | REV，并新增“过时段落索引”一节 |
| D13 | `docs/research/repository_analysis_inventory_2026-10-03.md` 第 119–121 行 | `status` 退出码 1；`check` 有 33 处路径错误，“仍需另行处理” | `f64177e` 已修复；今天 `status` 退出码为 0 | BAN（指向 `docs/agent/01`） |
| D14 | `docs/research/dataset_overview_gap_analysis.md` | 针对旧版 PDF 的缺口分析，写着 Xaira 等“缺失” | 已被 `dataset_overview_v3_gap_analysis.md` 替代 | ARC |
| D15 | `docs/research/dataset_overview_source_review.md` | 旧版 PDF 的来源核查 | 同上 | ARC（来源核查结论如仍有效，可保留链接） |
| D16 | `docs/dataset_overview.pdf` | 旧版 PDF | 已被 v3 替代 | ARC（被 `docs/datasets/dataset-overview-audit-2026-09-30.json` 按哈希记录，不是三对象引用） |
| D17 | `docs/research/dataset_overview_v3_gap_analysis.md` 第 3、20、27、55、81 行 | “GSE132080 完全缺失”“CD4 当前未运行”“若下一步补数据……” | Jost 已于 10-03 入库；CD4 已在 10-02 恢复下载，目前 85% | REV（这份是对接 PDF 的现行文档） |
| D18 | `docs/research/leaderboard_prediction_counts_REPORT.md` 第 124–128 行 [H] | “下一步维持当前 Ledger 的三个研究问题” | 三个问题都已关闭 | 登记进过时索引 |
| D19 | `docs/experiments/leaderboard-score-interpretation.md` | 9/26 的榜单快照（第 1 名 0.3579）和旧的改进建议（CVAE 时期） | 今天第 1 名 0.4301；分数解释部分仍然有效 | BAN（快照数字以 `docs/agent/01` §1.4 为准） |
| D20 | `docs/datasets/new-crispri-acquisition-2026-09-30.md` 第 3 行 | “文档创建时下载仍在进行” | Xaira 已完成；CD4 仍在下载 | BAN |
| D21 | `docs/datasets/arc_vcc2025_h1.html` | 唯一细胞数 376,531、重复 23%（出现 3 处） | 正确值是 414,694 和 15.5%；`docs/datasets/README.md` 已注明它是含错计数的历史报告 | ARC 或 DEL |

### 低风险：已标注为历史，或只是日期性陈述

| # | 位置 | 说明 | 处置 |
|---|---|---|---|
| D22 | `docs/research/dataset_foundations.md` 第 33、41、60、77–83 行 [H] | “重复表达是否一致尚未验证”“尚未做 HGNC 别名归一”“协议冻结前仍需完成的审核”。这些后来都由 `challenge_protocol` 完成了 | 登记进过时索引 |
| D23 | `docs/research/next_mechanism_sources.md` 第 3 行 [H] | “当前条件细胞 VAE 由主任务完成”（9/28 的状态） | KEEP（State/CellOT 的来源核查对路线 B1 仍有用） |
| D24 | `docs/research/knowledge_graph.attention.json` | 16 个节点含 9/29 的时效数字，例如 VCC-LEADERBOARD 写第 1 名 0.4185、VCC-TIMELINE 写“距 10/22 还有 23 天”、VCC-SUBMIT 写 CLI 0.2.2。这些节点都注明了“核实于 2026-09-29” | KEEP；可选择只刷新 VCC-LEADERBOARD 和 VCC-TIMELINE 两个节点 |
| D25 | `docs/experiments/` 下的 exp001、exp005、exp006、exp007 分析文档 | 目录 README 已注明是历史分析 | KEEP（exp006 里的 THP 停滞诊断对运维仍有用） |
| D26 | `docs/ideas/`（三篇） | 早期讨论，有推测内容；`docs/README.md` 已注明“不代表已验证结论” | KEEP 或 ARC |
| D27 | `docs/datasets/` 下的 09-18、09-19 审计文档 | 都已有“历史核查快照”横幅 | KEEP |
| D28 | `docs/operations/`、`docs/maintenance/` | 操作记录和工具说明，仍然准确 | KEEP |

**仍然有效、保留不动**：
- `docs/research/r4_anchor_audit.md`、`literature_foundations.md`、`leaderboard_prediction_metric_foundations.md`、`perturbation_decomposition_review.md`；
- `docs/research/challenge_2026/*`（评测契约产物，被引用 83 处以上）；
- `docs/research/experiment_dag.md`（生成视图，已核实与 DAG 同步，104/104 个节点）；
- `docs/agent/*`。

---

## 3. `data/` 清理清单

| # | 位置 | 问题 | 依据 | 风险 | 处置 |
|---|---|---|---|---|---|
| A1 | `data/README.md` 第 68 行 | “新的预处理……写入对应 Experiment 的 cache/，不写回 data/interim/、data/processed/” | 与 Q6 冲突 | 高 | REV：写入 `data/processed/<子目录>` 的规格（PDF 规则、清单、哈希、不含软链接） |
| A2 | `data/README.md` 第 13–35、121 行 | 09-18 的状态快照：scBaseCount“下载中”、Jiang“尚未转换”、Tahoe“未统计”、“processed 为空” | 09-19 的报告已完成这些；后来又新增了 Xaira、Jost、CD4 | 中 | REV：现状放在最前面，旧快照移到附录 |
| A3 | `.gitignore` 关于 `data/interim/`、`data/processed/` 的说明（“派生语料入库，>10MB 走 LFS”） | 和 pre-commit 一起，会把数百 GB 的标准化数据推进 LFS | 与 Q6 的体量冲突 | 高 | REV：忽略 `data/processed/<子目录>/`，只把清单和哈希入库（已列入 R0.1） |
| A4 | `data/profiles/arc_vcc2025_h1.json` | `unique_cells = 376531`（错误）；09-14 用旧 profiler 生成，profiler 已于 09-19 重写 | 正确值 414,694，现行报告在 `data/assessments/h1-structure-20260919-v3` | 中 | DEL 或 ARC（与 D21 一并处理） |
| A5 | `data/MANIFEST.tsv` | 追加式日志：同一路径重复 65 次，7 行 sha 为空 | 管理审计子代理 | 低 | KEEP 日志原样，README 里说明“以 registry.lock 和各 SOURCE 为准” |
| A6 | `data/sources.json` 的 `why` / `caveat` 字段 | 含历史建模判断 | README 已注明“不能当作证据” | 低 | KEEP |
| A7 | `data/raw/depmap_24q4` 和 run cache 里的 24Q2 | 两个 DepMap 版本并存，见 U5 | — | 中 | 二选一：把 24Q2 正式登记为来源，或者新 run 改用 24Q4 |
| A8 | `data/assessments`（指向 NAS 的软链接） | 文档里大量链接依赖 NAS 挂载 | 设计如此 | 低 | KEEP |

---

## 4. `scripts/`、`src/` 文件级清单

| # | 位置 | 问题 | 处置 |
|---|---|---|---|
| C1 | `scripts/status.sh` | 调用 `.venv/bin/python`，而仓库根目录已没有 `.venv`，**无法运行**；只被一份 09-18 的旧审计文档提到；内容是旧的“下载进度一览” | DEL（或改成调用基础环境 python） |
| C2 | `requirements-lock.txt`（仓库根目录） | 没有任何地方引用；内容过时（含 `cupy-cuda12x`） | DEL |
| C3 | `scripts/scbasecount_tools/` 及其本地 `.venv`（139 MB，未跟踪） | exp001 时期的 scBase 准备与审计工具，只被它自己的测试引用 | 代码 KEEP 或 ARC；本地 `.venv` DEL |
| C4 | `src/vcc_task/common.py` 第 18–27 行 | NAS 路径缺陷（U8） | REV（R0.1） |
| C5 | `src/vcc_task/run_context.py` 第 58 行 | commit 在完成时才读取（U8） | REV（R0.1） |
| C6 | `src/vcc_task/full_reference.py` 第 19–34 行 | 运行时下载 DepMap 24Q2（U5） | REV（A7 定下来后改） |
| C7 | `src/vcc_task/depmap_submission.py` 第 15、56–57 行 | 导入 `scripts/`、`scripts/dossier/` 和 `experiments/init-linear/src`（U1、U2） | REF |
| C8 | `src/vcc_mechanism/*` | 只服务 5 个已关闭实验，与 `vcc_task` 职责重复（U3） | 冻结为 legacy，在包内 README 注明；不删，保证复现 |
| C9 | `scripts/research.py` 第 1、34 行 | 硬编码 shebang 和 NAS 镜像路径；兼作库（U6、U7） | REV（R0.1：路径改为可配置，库函数拆出） |
| C10 | `scripts/dossier/publish_assessment.py` 第 18 行；`convert_seurat_cache.py` 第 19 行 | 依赖其他 micromamba 环境（llm-ft、biominer）的绝对路径 | 在 dossier README 注明外部依赖（dossier 冻结，不改代码） |
| C11 | `scripts/dossier/{rna,challenge,profile_responses,heterogeneity,heterogeneity_cells}.py` | 核心库放在评估脚本目录里（U6、U9） | REF：功能迁到 `src`。旧路径保留，保证 dossier 复现 |
| C12 | `scripts/analyze_submitted_counts.py` | CLI 只能扫已发布提交；统计函数被 `src` 复用（AGENTS.md 也提到） | REF：统计函数迁到 `src/vcc/eval`；CLI 改为调用 `src` |

---

## 5. 研究对象中的过时条目（`docs/research/*.json`）

以下条目受 `research.py check` 约束，已关闭的证据不能覆盖。处置方式统一为 OBJ：新增“已替代”标注或替代条目，原文不改。

| # | 对象 | 条目 | 问题 | 处置 |
|---|---|---|---|---|
| O1 | Ledger | 7 条 pending 实际已被替代：`E-LOCAL-EVAL-SUPPORT`、`-R2`、`-R3`；`E-TASK-DATA-GENERATION`；`E-LOCAL-CAPABILITY-EVALUATOR`、`-R2`；`E-CELL-VAE-PACKAGE` | 后续 R2/R3/R4 或 aligned 版本已经闭环，它们却仍然算在 27 个 pending 里，`status` 会列出来 | 标注已替代，关闭为“不再执行” |
| O2 | Ledger | `E-QC`、`E-QC-WEIGHT`、`E-QC-CONFIRM` | 与 Q6“除 GSE132080 外不再 QC”冲突 | 关闭为“不再执行（冲刺决策）” |
| O3 | Ledger | 16 条 init 时期的 pending：`E-TASK`、`E-BALANCE`、`E-MOMENTS`、`E-SET`、`E-PRETRAIN`、`E-SCFOUNDATION`、`E-CAPACITY`、`E-LOSS`、`E-CONTEXT`、`E-DECOMPOSITION`、`E-OPTIMIZATION`、`E-TRANSPORT-CONTEXT`、`E-TRANSPORT-CAPACITY` 等 | 基于旧的 S2/线性头框架，不在新路线里 | 关闭为“不再执行”，保留问题原文。以下三条可以**转入新路线**：`E-SET-DISTRIBUTION-PACKAGE`→B1；`E-TARGET-PRIOR`、`E-SHARED-TARGET-PRIOR`→E10；`E-GENERATION`→E4、B1 |
| O4 | Ledger | `initialization.policy`（“本轮仅初始化，不执行训练”）和 `resource_allocation` 的 `unit` / `third_slot` 文字 | 描述的是 10-03 之前的状态 | 第 0 轮开始时改写为冲刺第 1 轮的分配 |
| O5 | DAG | 33 个 draft 节点、20 个 draft 比较（22 个 `init-*`、4 个 `confirm-*`、4 个 `target-response-*`、2 个 `transport-*`、`set-distribution-s01`） | 都从未推进；`status` 每次都会列出它们的启动缺项 | 标注已替代（新增字段），或移到“已撤销”状态 |
| O6 | DAG `protocols` | `DATA-EVAL-DEFAULTS-20260930`（draft）、`PSEUDO-VCC-INIT`（ready） | 与 Q6、Q7 冲突，但历史 run 绑定着它们 | 保留，供历史使用；**另外新登记**冲刺协议（PDF 数据加 H1 测试集） |
| O7 | Method Space | 31 个 UNTESTED 方法，大多来自 init 时期 | 不在路线内 | KEEP，作为候选库；可选择加上“不在当前路线”的标签 |

---

## 6. 需要你决定的事项

1. **批准粒度**：逐条批准，还是按类别批准（例如“D 类高风险全部按建议处理”）。
2. **[H] 文件的处理方式**：
   - (a) 不改原文，只在 `docs/README.md` 增加“过时段落索引”（推荐：成本最低，不碰哈希）；
   - (b) 先把引用固定到 git commit，再在原文加横幅。
3. **归档位置**：`docs/archive/2026-10-04/`，可以吗？
4. **代码重构范围**：
   - 冲刺期只做最小重构：新包骨架，加上 U4、U5、U7、U8 的修复（推荐）；
   - 还是把 1.4 的完整迁移也一起做（约 1–2 天，会挤占第 1 轮时间）。
5. **DepMap 版本**：登记 24Q2，还是改用已登记的 24Q4？
6. **删除项**：C1、C2、C3 的 `.venv`，以及 A4、D21。确认后删除，都有 git 历史可以找回，`.venv` 除外。

你批准后，我先执行清理，再开始第 0 轮的 grill-me 对齐。
