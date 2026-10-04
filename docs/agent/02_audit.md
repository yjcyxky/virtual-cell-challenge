# 02 实验管理审计

撰写：2026-10-04，Claude。只读审计，所有建议**均未实施**。

**方法**

- 我自己读了协议、三份研究对象、代码和 git 历史。
- 另派三个只读子代理分别审计管理设施、评测与泄漏、算力成本，并做了一次外部调研。子代理没有改仓库，也没有跑训练或 `research.py` 的写操作。
- 引用的关键结论我做了抽查，抽查的项注明“抽查”。

**标注**：【已核实】有文件、行号或命令输出支持；【推断】是推理。

**严重程度定义**

| 级别 | 含义 |
|---|---|
| Critical | 直接威胁 11/5 的 final 成绩，或使模型选择不可信；应在下一轮实验前处理 |
| High | 显著拖慢迭代，或系统性地偏置结论 |
| Medium | 有界的可复现或追溯缺口 |
| Low | 卫生问题 |

---

## 0. 总评

这套体系**在“不做错误主张”上非常强**：

- 配置、哈希、seed、数据登记、W&B、失败留档首尾闭合。
- 输入 NTC 与评分 NTC 物理隔离，没有发现标签泄漏。

**它在“尽快找到更好的模型”上很弱**：

- **决策依据**：几乎全部比较都基于单折（S2-H1）、单 seed，阈值也没有用噪声标定。
- **本地与官方脱节**：本地分数到官方的换算只有约 0.51 倍斜率，另有两把不一致的本地标尺。
- **吞吐量**：门禁每次重新哈希约 136 GB 历史产物；每个 run 必须串行，且要单独提交一次；协议正文和必读材料合计约 6.6 万字。
- **没有终局计划**：final 阶段（D/E/F）既没有计划，也没有全量重训流程；榜单只被用来“核查”开发折上的拟合。
- **不适合 agent 迭代**：代码没有稳定的模型接口，agent 没法只改一小块就迭代。

距截止还有 32 天，需要从“审计优先”调整为“在可信评估下快速搜索”。只有准备对外宣布结论或提交最终模型时，才走完整协议。【推断】

---

## 1. 问题清单

### Critical

#### C1 研究体系与比赛终局脱节

**现象**：三份研究对象和协议文档里，都没有 final 阶段的目标、时间表、全量重训或提交策略。官方提交全部是开发折拟合的事后“核查”。

**证据**【已核实】

- 在 Ledger、DAG、`docs/RESEARCH.md`、`AGENTS.md` 和各 PLAN/REPORT 中搜索 `D/E/F|10-22|11-05|final`，没有任何计划条目。只有 KnowGraph 和 09-26 的 `docs/experiments/leaderboard-score-interpretation.md` 提到过 final。
- 最近 4 次官方提交（Linear、Shared、Shared-QC、kNN）都来自 S2-H1 折。这个折排除了 H1 标签，官方靶点只有 K562 的监督，覆盖 7,680 个读出基因（`docs/research/leaderboard_prediction_counts_REPORT.md` 第 68 行；Ledger `E-*-OFFICIAL`）。
- Ledger `resource_allocation.exploration_slots = []`、`queue = []`；`status` 输出里也没有下一步行动（见 01 附录 A）。
- 规则：final 只计**最后一次**提交，期间不显示榜单，每天 2 次（【外部】[rules](https://virtualcellchallenge.org/rules)，10-04 读取）。

**影响**

- final 的 panel 会换。没有提前演练的“推断 → 打包 → 提交”流水线，没有冻结的“选型规则 + 全量重训配方”，也没有兜底版本。
- 现有协议把“官方提交”定义为已完成 fit 的追加评估。这意味着**榜单从未见过用全部可用数据训练的模型**。

**建议**

- 写出 final 目标与里程碑。
- 定义“final 配方”：先在多背景 LOCO 上选型，再用所有允许的源全量重训，最后冻结导出。
- 10/22 前在 A/B/C 上把这套配方完整演练一次。演练本身就是验证榜提交。
- 预先规定 final 窗口的兜底提交，以及“最后一次提交”的冻结规则。

#### C2 模型选择基于单折、单 seed，阈值没有用噪声标定

**证据**【已核实；评测子代理统计，我抽查了 DAG 的 fit_scope 分布】

- 19 条已关闭的本地实验结论中：
  - 14 条只基于 S2-H1；
  - 19 条全部单 seed。
- 60 个预测器拟合中：
  - 25 个是 S2-H1；
  - 35 个是 DepMap 的 S3 五折；
  - S2 的另外 4 个背景没有任何模型拟合，S4 也没有。
- `min_effect: 0.02` 在 DAG 中出现 17 次，是事先拍定的启发值。这与 `docs/research/data_evaluation_decision.md` 第 113 行“最小效应与容差由训练/校准范围冻结”的要求不一致。
- 噪声的已知情况：
  - 训练 seed 的方差从未测过。
  - 生成 seed 只有旧协议下的一个数字：SD 0.0011（`experiments/exp003-context-module-cvae/REPORT.md` 第 322 行）。
  - 背景之间配对效应的 SD 为 0.043–0.166（`experiments/depmap-response/REPORT.md` 第 11–14 行）。
- 有几条结论正好卡在门槛附近：
  - ridge 强度 +0.0174；
  - DepMap 残差相关增益 0.0155 / 0.0187，门槛 0.02。

**影响**：“Shared 是强参照”“停止 QC”“停止补全”这些结论可能只在 H1 上成立。好的方向可能被误杀，也可能把噪声当成收益。

**建议**

- 正式比较至少使用 4–6 个留出背景，各背景平均和最差背景都报告。Xaira 的 HCT116/HEK293T 可以作为新增留出背景。
- 每个配置至少跑 3 个训练 seed × 2–3 个生成 seed。
- 阈值取 k × SD，并在候选运行前冻结。

#### C3 本地分数对官方的预测效度没有建立，而且存在两把不一致的本地标尺

**证据**【已核实；评测子代理整理，我抽查了 4 对分数和 FES 的 NTC 数】

1. **本地与官方的换算**：当前协议下有 4 对“本地 vs 官方”分数，排序一致（Spearman 1），但官方比本地低 0.127–0.163。拟合关系约为 `官方 ≈ −0.058 + 0.51 × 本地`。本地分差到官方被压缩到 0.31–0.63 倍；kNN 与 linear 这一对反而放大了 2.33 倍。
2. **两把本地标尺**：

   | | init-linear 面板 | full-eval-support 面板 |
   |---|---|---|
   | NTC（输入 / 评分） | 18,400 / 19,104 | 2,048 / 2,048 |
   | 深度 | H1 原生，中位约 54k | 抽稀到约 20k |
   | 参考细胞 | 沿用 `tasks.csv` 冻结身份 | 用 seed 930 重抽 |

   来源：`experiments/task-observation/protocol.json` 第 15 行；`src/vcc_task/full_reference.py` 第 63–65 行。官方是 18,400 个 NTC、约 20k 深度、每靶 400 个细胞（`data/raw/arc_vcc2026_controls/manifest.json` 第 18–19 行）。两个面板都不完全等于官方条件。
3. **没有靶点信息的对照也有不低的分数**：在 S2-H1 上，靶点错配对照得 0.099，源模板得 0.071（`experiments/full-eval-support/REPORT.md` 表 1；`experiments/depmap-response/REPORT.md`）。
4. **线上 baseline 配方未知**：线上 r4 的 baseline 构建方式无法核实。仅把本地 baseline 从 dispersed 换成 tile，Overall 就移动 0.16–0.18（`docs/research/r4_anchor_audit.md`；`experiments/init-linear/REPORT.md` 第 273 行）。
5. **面板的支持结构不同**：本地 280 靶中有 99 个有四个背景的支持；官方的 272 个已见靶点只有 K562 支持。linear 对 shared 的 PDS 差在两类子群中分别是 −0.023 和 −0.230（`experiments/init-linear/REPORT.md` 第 91 行）。

**影响**：本地排序可以用来做粗筛，但不能用来估计收益大小。本地 +0.02 大约只对应官方 +0.006–0.010。本地 Shared 的 0.21 中，真正的靶点特异部分大约只有 0.11【推断】。

**建议**

- 建一个“官方仿真”的本地面板：18,400 / 18,400 个 NTC、约 20k 深度、每靶 400 个细胞，靶点支持结构接近官方，并重建 anchors。
- 每次评估都同时报告靶点错配和源模板两个“无信息”下限，以及 raw 六项。
- 本地分数只用于筛选，不用于预测榜单。验证榜作为低频校准仪器。

#### C4 门禁每次都重新哈希约 136 GB 历史产物

**证据**【已核实；管理子代理统计，我抽查了代码和实际耗时】

- 代码：`scripts/research.py` 第 132–143 行，对每个未跟踪的引用按 16 MB 分块做全量 SHA-256。
- 规模：49 个大引用，共 136.4 GB。其中 133 GB 来自 `full-eval-support-s04` 和 `init-linear-s01` 的 `evaluation_repair.preserved_refs`，有 33 个经 NAS 读取。
- 跳过这些引用后，全量校验只要 3.5 秒，0 个错误。
- 调用频次：一次 `execute` 校验 4 次，`close` 2 次，pre-commit 1 次。
- 实测：今天 `status` 从 09:08 跑到约 09:19，大部分时间处于 NFS 的 D 状态。
- 这是 **10-02 之后才出现的回归**：DepMap 批次期间，每个 run 的门禁只要约 0.7 分钟（`experiments/depmap-response/outputs/depmap-mlp-shuffled-s2-h1-s930/diagnostics/batch-runtime.json`）。

**影响**【推断】：现在每个 run 的门禁开销约 40–50 分钟，超过多数模型的训练时间（2–13 分钟），而且会随历史累积继续线性增长。agent 迭代的吞吐量因此塌掉。

**建议**

- 按 `(path, size, mtime, inode)` 缓存已验证的哈希。
- 全量复核改为 `--deep` 选项。
- `gate` 只校验本节点及其依赖。

### High

#### H1 结构性串行，每个 run 都要一次提交

**证据**【已核实，管理子代理】

- gate 要求三份对象与 HEAD 一致（`scripts/research.py` 第 636–641、711、718 行），而 record 会改写 DAG（第 870 行）。所以下一个 run 启动前，必须先提交上一个 run 的记录。
- 每个 Experiment 一把非阻塞排他锁（第 880–887 行），同一路线不能并行。
- 比较的各臂都完成但证据没关闭时，`staged_errors` 会拒绝提交（第 624–632 行）。
- DepMap 的 42 个 run 全部串行，产生了 41 个只改 `experiment_dag.json` 的提交。
- 全仓 348 次提交中，113 次是登记、回收、关闭一类的记账提交（`git log | grep -ciE "Recover completed|Record|Close|Register|Freeze"`）。

**影响**：无法并行搜索；跨 Experiment 并发时，一个 run 的 record 会让另一个正在 bind 的 run 门禁失败【推断】。

**建议**

- 执行记录改为每个 run 追加写的独立文件，不放进受 Git 约束的 DAG。
- gate 只比对该节点的登记哈希。
- 按内存预算允许 N 个 run 并发。

#### H2 没有登记工具，DAG 只能手改

**证据**【已核实，管理子代理】

- CLI 只有 check / status / graph / gate / execute / record / retry / close，没有写节点的命令。
- DAG 有 1.26 MB、29,016 行；1,686 条 `code_refs` 只对应 136 个路径。
- DepMap 那次登记提交改了 56 个文件、新增 24,156 行。
- 连一个 4 行的修复（`e46b4e8`）都要手改 DAG 里的哈希。

**影响**：每次会话都要写临时脚本去改大 JSON，容易出错；对 agent 来说 token 成本也高。

**建议**：增加 `register`、`clone-run`、`rehash`、`supersede` 子命令；增加“批次 / sweep”节点类型，一次登记一组 fit。

#### H3 最相关的数据没进模型，也没有统一的训练语料

**证据**【已核实】

- Xaira HCT116/HEK293T：覆盖 300/300 个官方靶点、18,401/18,533 个官方基因，10-01 已完成索引，但在 `experiments/*/src`、`configs` 和 `src/` 中 grep 不到任何使用。
- H1 标签不在任何一次官方提交的训练数据里。
- CD4 还在下载。
- `data/processed/` 是空的；每个 Experiment 在自己的 cache 里重建训练统计。

**影响**：官方靶点的监督只覆盖 K562 的 7,680 个读出。用了这些数据的公开方案能到 0.155–0.180（【外部】AtlasShift，见 01 §1.4）。

**建议**：一次性构建一个有版本的“源响应库”，供所有实验只读复用：
- 维度：源 × 靶点 × 基因；
- 内容：pseudobulk 计数和、mean CPM、细胞数 n、guide/batch、测量掩码、剂量估计。

#### H4 协议和文档负担过重，而且变化频繁

**证据**【已核实】

- `AGENTS.md` + `docs/RESEARCH.md` 共 21,592 字符；必读的链接文档另有约 44,700 字符。
- `research.py` 里约有 150 条契约检查。
- docs 和 experiments 下的 Markdown 共约 57 万字符。
- 过去两周的几次大动作：
  - exp008：一份计划做了约 30 次文档提交，最终没有执行；
  - 9/27：重新初始化，丢弃了全部历史证据；
  - 9/30：再次转向。
- KnowGraph 有 213 个节点，也被要求作为注意力输入。

**影响**：每个 agent 会话要把大量上下文花在协议上，决策变慢。AGENTS.md 自己警告过的“停留在协议审计或文档维护”，确实发生了。

**建议**：为 10/04–11/05 写一页“冲刺契约”，取代逐 run 的仪式；完整协议只用于最终模型和对外主张。**这需要你批准**，因为要改 AGENTS.md。

#### H5 代码结构不适合 agent 迭代

**证据**【已核实，管理子代理】

- 没有稳定的“模型接口 + 固定评估器调用”。
- `src/vcc_mechanism` 和 `src/vcc_task` 各有一套 run 生命周期。
- 重复代码：`wandb.init` 写了 18 份，`Tee` 类 7 份；代码通过 `sys.path` 跨实验导入。
- 环境：20 个 `.venv`，最大的 6.5 GB，但多个 `uv.lock` 之间只差几行。
- DepMap 主流程 `src/vcc_task/response_experiment.py`（543 行）没有直接测试。

**影响**：每加一个机制，就要新建目录、新建环境、新建登记。agent 没法在一块固定区域里做小改动、快速评估。

**建议**：做一个包，提供“任务契约”：

```
load_fold(split) -> SourceStore, TargetContext
Model.fit(sources)
Model.predict_delta(ntc, targets) -> Δ（bulk 空间 + 均值 CPM 空间）
emit_counts(Δ, ntc, seed)
score(fold) -> raw 六项 + normalized + 诊断
```

评估做成分级的：快速代理 → 完整六项 → 多背景确认。冲刺期间共用一个环境。

#### H6 生成器在官方提交之间不一致，对 DE 指标的影响没有系统隔离

**证据**【已核实】

- 第 9 次提交用固定 log-shift 加随机取整；第 10 次用群体校准发射器（`experiments/masked-response/OFFICIAL-PLAN.md`；`experiments/depmap-response/REPORT.md`）。
- 群体校准是 signal，但五个背景的 DE Jaccard 全部下降（Ledger `E-POPULATION-EMISSION`）。
- 【外部】85% 的队伍 fidelity 为负，前 100 名中只有 8%（外部调研子代理统计榜单 JSON）。

**影响**：官方比较混入了生成器的差异。六项中有 4 项是 DE 类指标，它们的上限受发射器制约。

**建议**

- 每个冲刺阶段冻结一个发射器。
- 用固定响应（零响应、真值 oracle）在官方仿真面板上做发射器对比，单独报告 DE 指标。

#### H7 运行库仍有 NAS 路径缺陷，代码溯源不封闭

**证据**【已核实，管理子代理；我抽查了第 1 条】

- `src/vcc_task/common.py` 第 18–27 行的 `ref()` 用了 `resolve().relative_to(ROOT)`。对已软链到 NAS 的产物（例如 depmap-knn 的 `model.pt`、`counts.h5ad`），这一行会抛 ValueError。恢复分支（`response_experiment.py` 第 204–206、341–343 行）会走到这里。
- gate 不拒绝脏工作区；`code_refs` 只覆盖 `experiments/<exp>/src`。例如 `scripts/dossier/rna.py` 经 `challenge.py` 被导入，但不在 `code_refs` 里。
- execution commit 在 run 完成时才读取（`src/vcc_task/run_context.py` 第 58 行）。`population-emission-s01` 15:58 启动，记录的却是 16:01:59 才出现的 `e80947d`。

**影响**：迁移到 NAS 的 run，恢复和复评会失败；run 记录的 commit 不一定是实际运行的代码。

**建议**

- 复用 `research.artifact_path` 的路径处理。
- 启动时记录 commit。
- 从干净的 worktree 快照运行，并拒绝脏树。

### Medium

#### M1 数据血缘：DepMap 先验其实是 24Q2，而且不在登记链里

**证据**【已核实，我抽查】

- 42 个 DepMap 配置的 `prior_ref` 指向 `experiments/full-eval-support/outputs/full-eval-support-s01/cache/CRISPRGeneEffect.csv`，这是一个**失败 run 的 cache**。
- `external_prior` 字段写的是 “DepMap 24Q2 Public version 1”，而仓库登记的是 `data/raw/depmap_24q4/`。
- `data/MANIFEST.tsv` 是追加式日志，有 65 个重复路径、7 行 sha 为空（管理子代理统计）。

**建议**：新 run 改用已登记的 24Q4，或把 24Q2 正式登记为一个来源；MANIFEST 提供去重后的视图。

#### M2 环境记录不全

**证据**【已核实，管理子代理】

- 基础环境（Python 3.14.7 / uv 0.12.16）没有导出入库。
- `requirements-lock.txt` 没有任何地方引用，里面还写着 `cupy-cuda12x`。
- run 配置只记录了 6 个包的版本，没有 Python、CUDA、驱动、GPU 信息。
- MLP 没有开启确定性算法。

**建议**

- 每个 run 的配置里写入 `micromamba env export` 摘要、`torch.version.cuda` 和 `nvidia-smi` 信息。
- 开启确定性开关，或者明确记录不确定性。
- 删除过时的锁文件。

#### M3 `status` 给不出可执行的下一步

**证据**【已核实】

- 27 个 PENDING 中包含已被替代的旧问题（`E-LOCAL-EVAL-SUPPORT` R1–R3、`E-TASK-DATA-GENERATION` 等）。
- 64 个 completed 节点一律显示 `await result/evidence`。
- 22 个 `init-*` draft 从未推进。

**建议**：`status` 只输出前 3 个问题和各自的下一步；被替代的 pending 自动标注为已替代。

#### M4 Overall 的截断和构成，让关键改进看不见

**证据**【已核实】

- 本地和官方的 Expression 标度分**一律为 0**，而 raw 在 1.47–11.97 之间差别很大。
- S3-K562 的 Jaccard 标度分在 −2.5 到 −2.8，主导了这个背景的 Overall（`experiments/depmap-response/REPORT.md` 第 223–229 行）。
- K562 只测了 7,680 个基因，跨背景平均实际上混用了不同的标尺。

**建议**：把 raw 表达误差（以“不预测任何变化 ≈ 1.0”为参照）设为主诊断；逐背景报告；跨背景时同时给出中位数。

#### M5 验证榜的使用方式

**证据**【已核实 + 外部】

- 10 次提交的结果都参与过决策。
- 榜单只显示最近一次提交，所以 0.049 的条目已被 kNN 的 0.0115 覆盖，现在排第 802 名。
- A/B/C 与 D/E/F 不同。

**建议**

- 把验证榜当作有固定预算的校准仪器，例如每天不超过 1 次，且只提交愿意被看见的候选。
- 登记每次提交的目的。
- 采用 Ladder 式规则：新分数超过门槛才替换当前版本。

#### M6 本地参考细胞数不足，DE 后端没有和官方核对

**证据**【已核实，评测子代理】

- S2-H1 有 42/280 个靶点不足 400 个细胞，其中 8 个少于 100。
- 其余背景每靶中位只有 45–178 个细胞。
- 本地 DE 用 GPU（gpudge）计算，是否与官方后端一致没有验证。

**建议**：官方仿真面板只纳入 n≥400 的靶点；做一次 CPU 与 GPU 的 DE 等价性检查。

#### M7 文档漂移

**证据**【已核实】

- `docs/research/repository_analysis_inventory_2026-10-03.md` 第 119–121 行仍写着 NAS 路径问题“仍需另行处理”，但提交 `f64177e` 已经修复。
- `data/README.md` 的 09-18 快照部分过时。

**建议**：有日期的文档保留原样，但加上“已被 X 替代”的指针；当前状态以工具生成的页面为准。

### Low

| # | 问题 | 证据 |
|---|---|---|
| L1 | `ARTIFACT_MIRROR` 是硬编码路径；NAS 未挂载时有 33 个引用会校验失败 | `scripts/research.py` 第 34 行 |
| L2 | DAG 和 Ledger 写出时权限为 0600 | 管理子代理 |
| L3 | 没有 CI。79 个治理测试从不自动运行；在临时仓库里跑，14 秒全部通过 | 管理子代理 |
| L4 | Ledger 的 `E-SHARED-QC` 缺少 `kind` 字段 | 评测子代理 |
| L5 | `docs/dataset_overview_v3.pdf` 未跟踪；`docs/README.md` 索引没有列出 `docs/agent/` | `git status` |
| L6 | DepMap 特征包含全部约 1,150 个细胞系（含 K562/HepG2/Jurkat）。这不是标签泄漏，但属于背景旁信息；essential 文库本身按 DepMap 选靶，存在选择偏倚 | `experiments/depmap-response/PLAN.md` 第 57 行；评测子代理 |

---

## 2. 四个维度的结论

| 维度 | 判断 | 主要条目 |
|---|---|---|
| 可复现性（seed / 环境 / 数据版本 / 配置） | **强**。4 个抽样 run 的 expected_config、configs、config.yaml、metrics、W&B 及配置/锁文件哈希逐项一致；种子从单一主种子派生并逐靶记录；数据有 sources + lock + SHA-256 | 缺口：环境导出（M2）、NAS 路径（H7）、DepMap 24Q2 血缘（M1）、MLP 非确定性 |
| 实验追踪（结果 → 代码 / 配置） | **强**。记录的 74 个 commit 都存在于 HEAD 历史中；失败身份完整保留 | 缺口：commit 在完成时才记录、允许脏树、`code_refs` 覆盖不全（H7） |
| 评测（统一、可信、泄漏、划分） | **实现正确，决策效度弱**。NTC 隔离和训练边界无泄漏；scorer 固定在 `5e648335`，anchors 按参考群体重建 | 单折单 seed（C2）、两把本地标尺和 0.51 倍换算（C3）、截断（M4）、参考 n 不足（M6） |
| 结构（支撑 agent 自动迭代） | **不支持** | 门禁重哈希（C4）、串行且每 run 一次提交（H1）、手改 DAG（H2）、没有模型接口（H5） |

## 3. 应当保留的做法

1. **配置冻结链与哈希绑定**：任何结果都能追溯到配置、代码和数据。
2. **种子派生与记录**：评分 seed 冻结在 `scorer.json`。
3. **数据契约**：H1 物理去重、输入/评分 NTC 隔离、官方 18,533 基因轴加测量掩码、缺测不补零。
4. **失败不覆盖**：失败身份、supersede、evaluation repair 都保留原始绑定。
5. **训练后预测诊断**：这套诊断确实发现过真问题，例如未见靶点完全回退到 NTC、旧模型把敲低乘数硬编码为 0.1。
6. **证据纪律**：区分外部证据和本地证据；预登记判据；记录负结果。

## 4. 改进建议（未实施，按优先级）

| 优先级 | 建议 | 对应问题 | 预估成本 | 需要你批准吗 |
|---|---|---|---|---|
| P0 | 定义 final 目标、final 配方（LOCO 选型 → 全量重训 → 冻结导出）和 10/22 前的演练 | C1 | 0.5 天（文档）+ 首次演练 | 是（目标与提交策略） |
| P0 | 门禁加哈希缓存，全量复核改为 `--deep`，gate 只查本节点 | C4 | 2–4 小时 | 是（改 research.py） |
| P0 | 评估器 v2：官方仿真面板、多留出背景（加 Xaira 两个背景）、≥3 seed 噪声基线、k·SD 阈值、无信息下限 | C2、C3 | 1–2 天 | 方案需要批准 |
| P0 | 冲刺契约：批量登记、run 记录移出 DAG、允许并行、每轮最多 3 个问题 | H1、H2、H4 | 0.5–1 天 | 是（改 AGENTS.md / research.py） |
| P1 | 源响应库（K562、RPE1、HepG2、Jurkat、H1、Xaira×2，CD4 下载完成后加入） | H3 | 0.5–1 天计算 | 否（只读数据，写入新 cache） |
| P1 | 任务契约包加模型接口，冲刺期间共用一个环境 | H5 | 1–2 天 | 是（结构调整） |
| P1 | 冻结发射器，做以 DE 为导向的发射器对比 | H6 | 0.5 天 | 否 |
| P1 | NAS 路径修复，启动时记录 commit，拒绝脏树 | H7 | 2–3 小时 | 是（改共享代码） |
| P2 | 环境导出、DepMap 血缘、`status` 输出精简、CI、文档索引 | M1–M3、M7、Low | 0.5 天 | 否 |

所有建议**都没有实施**。其中改 AGENTS.md、research.py、共享代码或提交策略的，会在阶段 3 逐项问你。
