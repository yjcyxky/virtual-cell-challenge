# 01 项目理解：VCC 2026 virtual cell 模型

撰写：2026-10-04，Claude（只读调研，未改代码、未训练、未提交）。
范围：仓库 `master@1a5edde`、三份研究对象（Method Space / Experiment DAG / Evidence Ledger）、各 Experiment 的 PLAN/REPORT/metrics、git 历史、KnowGraph 注意力索引，以及公开榜单与官方规则页面。

**标注约定**

- 【证据】结论有仓库文件或命令输出支持，后附来源（路径相对仓库根）。
- 【外部】来自官方网页、公开 API 或第三方仓库，附 URL 与读取时间。
- 【推断】我的推理或综合判断，未经实验验证。
- 【待核实】目前没有足够来源。

---

## 0. 结论摘要

1. **任务**：给定一个未见细胞背景的 18,400 个 NTC 细胞和 300 个 CRISPRi 靶点，为每个“背景 × 靶点”生成 400 个细胞，每个细胞覆盖官方 18,533 个基因，输出非负整数 counts。评分用 cell-eval2 `vcc2026` preset 的六项指标：每项先按“背景平均扰动响应（oracle，记 0）”和“真实 split-half 重复（记 1）”标度，再取等权平均得到 Overall。【证据】`docs/research/challenge_protocol.md`、`docs/experiments/leaderboard-score-interpretation.md`
2. **决定胜负的是 final 阶段**：D/E/F 三个新背景加一套新的 300 靶 panel，10/22 发布，11/5 23:59 UTC 截止。每天最多提交 2 次，只计**最后一次**提交，final 期间不显示榜单。今天是 10/04，离 final 数据发布还有 18 天，离截止还有 32 天。【外部】KnowGraph `VCC-TIMELINE`、`VCC-FINAL`（据 virtualcellchallenge.org/rules，核实于 09-29）
3. **SOTA**：2026-10-04 13:13 UTC，验证榜第一名 Illumina AI（iCell）的 Overall 为 **0.4301**，第 10 名 0.282，中位数 0.0788，共 1,304 队。公开可复现的 AtlasShift 多源迁移方案得分 0.1546，作者新版本 0.1804。【外部】[公开 API](https://virtualcellchallenge.org/api/leaderboard?get_final=false)，见 §1.4
4. **我们**：
   - 10 次官方提交，最好的是 Shared-QC，**0.0501**，低于中位数。
   - 榜上当前显示的是最近一次提交，DepMap kNN，**0.0115，排第 802 名**。
   - 本地 S2-H1 最好约 0.21，但到官方普遍缩水 0.13–0.16。
   - 我们所有条目的 raw 表达误差（4.5–7.9）都比原样输出 NTC（约 1.02）还差。
   - 来源：【证据】`docs/research/leaderboard_prediction_counts_scores.csv`、`experiments/depmap-response/REPORT.md`
5. **已被否定的方向**（结论只针对各自的具体实现）：条件线性、NTC 程序/PCA 表示、缺测补全、额外 QC、神经迁移、组成 logFC、NB 分布生成、跨基因解码器、条件计数 VAE、DepMap Ridge/MLP/kNN 整包，以及历史上的 XGB/CVAE 路线。**在匹配对照下，没有候选超过 Shared（同靶平均响应迁移）**。【证据】`docs/research/evidence_ledger.json` 中已关闭的 18 条本地结论
6. **最大的未用杠杆**【推断，§3.5 给出依据】：
   - Xaira HCT116/HEK293T 覆盖全部 300 个官方靶点、18,401/18,533 个官方基因，已入库，但从未进过任何模型。
   - 全部官方提交都来自 S2-H1 折的拟合，这个折排除了 H1 标签。官方靶点的直接监督只来自 K562，而 K562 只覆盖 7,680 个读出基因。
   - 因此，榜单上从未出现过“用全部可用相关数据训练”的模型。

---

## 1. 任务本身

### 1.1 输入与输出

| 项 | 内容 | 来源 |
|---|---|---|
| 输入 | 每个验证背景（A/B/C）18,400 个 NTC 细胞（46 个 NTC guide × 400），18,533 个基因的原始 counts；`pert_counts.csv` 列出 300 个靶基因，`gene_names.csv` 给出基因轴及顺序 | 【证据】`data/README.md` §2026 官方验证输入；`data/raw/arc_vcc2026_controls/manifest.json` |
| 输出 | 360,000 个细胞（3 背景 × 300 靶 × 400），18,533 个基因，非负、有限的整数 counts，不能含 NTC 行；先用 `vcc prep` 打包成 `.vcc` 再提交；有单细胞和稀疏存储上限 | 【证据】`docs/research/challenge_protocol.md` §官方目标；【外部】[VCC CLI 指南](https://vcc-cli-wiki.virtualcellchallenge.org/#submission-requirements-2026) |
| 没有提供的东西 | 2026 年没有训练标签；A/B/C 的细胞系身份不公开；final 换成另外三个细胞系和另一套 300 靶 panel | 【证据】`docs/ideas/challenge-overview.md` 第 3–13 行；【外部】KnowGraph `VCC-PANELS` |
| 测量化学 | 2026 与 2025 H1 同为 10x Flex；A/B/C 的 NTC 文库中位数约 20k（20,109 / 19,946 / 20,034） | 【证据】`docs/ideas/challenge-overview.md` 第 42–43 行；`data/README.md` §总计数 |
| 敲低强度 | 官方材料称所有靶点的 on-target 敲低 >80% | 【证据】`docs/ideas/challenge-overview.md` 第 29–30 行（对官方材料的摘要）；【待核实】未在官方页面原文复核 |

本质上，这是一个“未见细胞背景 × 基因扰动 → 单细胞分布生成”的零样本任务。真正要预测的，是每个靶点**相对该背景平均响应**的扰动特异变化。【推断】依据见 §1.3。

### 1.2 可用数据与实际使用情况

| 数据 | 背景 | 化学 | 规模 | 官方 300 靶中有监督 | 官方 18,533 基因中实测 | 是否进过模型 |
|---|---|---|---|---:|---:|---|
| VCC 2025 H1 | H1 hESC | 10x Flex | 414,694 条唯一记录，300 靶 | 25 | 18,074 | 是，但主要作为 S2-H1 评估折；S3 的其他折拿它当训练源 |
| Replogle K562 GWPS | K562 | 10x 3′ | 1,989,578 细胞，9,866 靶 | 272 | 7,680 | 是，官方靶点的主要监督来源 |
| Replogle RPE1 | RPE1 | 10x 3′ | 247,914，2,393 靶 | 0 | 8,260 | 是 |
| Nadig HepG2 | HepG2 | 10x 3′ v3 | 145,473，2,393 靶 | 0 | 9,024 | 是 |
| Nadig Jurkat | Jurkat | 10x 3′ v3 | 262,956，2,393 靶 | 0 | 8,283 | 是 |
| **Xaira HCT116** | HCT116 | FiCS 固定细胞 | 3,409,169 细胞，NTC 165,777 | **300**（≥50 个支持细胞的 263） | **18,401** | **否** |
| **Xaira HEK293T** | HEK293T | 同上 | 4,534,299，NTC 218,838 | **300**（≥50 的 276） | **18,401** | **否** |
| Jost GSE132080 | K562，guide 活性滴定 | 10x 3′ | 23,633 细胞，25 靶 | — | — | 否，仅完成入库校验 |
| Zhu 2026 CD4+ T | 原代 T 细胞，4 供体 × 3 状态 | GEM-X Flex | 约 22M 细胞（PDF 口径） | 【待核实】 | 【待核实】 | 否，仍在下载 |
| DepMap 24Q4 | 1,178 个细胞系 | — | 依赖性 + 表达 | 先验 | — | 是，`depmap-response` 用作先验 |
| 其他 | Jiang、McFaline、scPerturb、Tahoe 子集、scBaseCount 子集、LINCS、STRING/Reactome/GO、ESM-2、SE-600M | — | — | — | — | 只用过 Reactome（程序表示）；其余只做了描述统计，或未使用 |

来源：

- 【证据】`docs/research/challenge_protocol.md` 表 1；`docs/research/dataset_foundations.md`
- 【证据】`docs/research/dataset_overview_v3_gap_analysis.md` §Xaira；`docs/datasets/acquisition-resume-2026-10-03.md`
- 【证据】`grep -ril "xaira|hct116|hek293" experiments/*/src experiments/*/configs src` 无结果
- 【证据】进程表显示 CD4 下载器 PID 1929550 自 10-02 起一直在运行
- 化学一栏：【外部】KnowGraph `DATA-XATLAS` 称 Xaira 用 10x 5′ 化学、中位约 16k UMI；PDF 写作 10x Chromium/Cell Ranger 8。本地抽查的一个分片，每细胞 counts 中位数为 18,364（`docs/datasets/new-crispri-acquisition-2026-09-30.md`）

关键事实：

1. **官方靶点的监督几乎只来自 K562**。在当前冻结的映射下，RPE1/HepG2/Jurkat 覆盖 0 个官方靶点；Shared 对官方靶点的 272 个监督全部来自 K562，只覆盖 7,680 个读出基因。【证据】`docs/research/leaderboard_prediction_counts_REPORT.md` 第 68 行
2. **五个背景只来自 3 个研究组**（Arc H1；Replogle 的 K562/RPE1；Nadig 的 HepG2/Jurkat），背景与 assay、时间点、效应器相互混杂。【证据】`docs/research/dataset_foundations.md` §来源与独立性
3. **每靶真实细胞数差别很大**：中位数 H1 1,071、K562 178、RPE1 72、HepG2 45、Jurkat 83。17,322 个结构合格任务中，只有 806 个达到 400 个细胞。【证据】`docs/research/data_evaluation_decision.md` 表 1

### 1.3 评测指标

| 指标 | 含义 | 原始方向 | 标度后的截断 |
|---|---|---|---|
| PDS | 扰动身份的余弦判别；排除整个 panel 的靶基因；无信息时为 0.5 | ↑ | 不截断 |
| Expression MSE（`expr_mse_unbiased_capped_norm`） | 去噪后的 pseudobulk 平方误差，除以真实效应尺度（ratio of sums） | ↓ | 截断到 [0,1] |
| LFC NMAE | 真实 DE 基因上 LFC 的归一化误差；零 LFC 预测约为 1 | ↓ | 下限 −6 |
| Direction fidelity | DE 方向的保真度与产出量 | ↑ | 不截断 |
| Direction reach | 真实显著基因池中，方向可靠的覆盖深度 | ↑ | 不截断 |
| Jaccard | 显著基因集合的重叠 | ↑ | 不截断 |

- 标度方式是 `s = (u − b)/(r − b)`，Overall 是 3 个背景 × 6 项的等权平均。【证据】`docs/experiments/leaderboard-score-interpretation.md` §零分表示什么
- b 是被评估背景**真实扰动细胞**的平均响应，即 oracle 模板；r 是真实数据的 split-half 重复。参赛者拿不到 b，所以“只输出模板”最多得 0 分。【证据】同上
- **A/B/C 上 MSE 的 b 约为 0.986–0.992**。这说明背景平均响应只占去噪后效应能量的约 1%，正分几乎全部来自扰动特异信号。【外部】KnowGraph `EVAL-MSE`、`EVAL-SCALING`（cell-eval2 vcc2026 metrics brief §8，rule_version 3；与线上 r4 是否一致【待核实】）
- 评分器版本：仓库冻结了 cell-eval2 commit `5e648335` / v0.16.0 / `vcc2026`。线上 anchors 是 `vcc2026-val{A,B,C}-r4`，它的构建配方无法从包内默认值认证。【证据】`docs/research/challenge_protocol.md` §官方 cell-eval2 接入；`docs/research/r4_anchor_audit.md`

### 1.4 SOTA 的定义与当前数值

**定义。**

- 项目协议规定：SOTA 主张必须有当前官方数据，或有严格匹配的竞争方法作为证据；单个背景上的本地涨分只算线索。【证据】`AGENTS.md` §研发目标
- 在操作层面，最终标尺是 **final 阶段 D/E/F 上的官方 Overall**。它在评奖时才公布，而且只看最后一次提交。验证期唯一可以和别人同标尺比较的，是 A/B/C 公开榜。验证分与 final 分官方明确不可比。【外部】KnowGraph `VCC-FINAL`、`VCC-PANELS`

**数值。**

| 时点 | 条目数 | 第 1 名 | 第 2–3 名 | 中位数 | 来源 |
|---|---:|---|---|---:|---|
| 2026-09-26 | 1,174 | Illumina AI / PerturbationAI 0.3579 | cqawesome 0.3183；Testh 0.3014 | 0.0698 | 【证据】`docs/experiments/leaderboard-score-interpretation.md`（公开 API 快照） |
| 2026-09-29 | 1,211 | Illumina AI 0.4185（55 次提交） | 前 50 名在 0.224–0.419 之间 | 0.073 | 【外部】KnowGraph `VCC-LEADERBOARD` |
| **2026-10-04 13:13 UTC** | **1,304** | **Illumina AI / iCell 0.4301** | cqawesome 0.3289；Testh 0.3076；第 4–10 名 0.282–0.294 | **0.0788**（正分 835 个，64%） | 【外部】[公开 API](https://virtualcellchallenge.org/api/leaderboard?get_final=false)，curl 抓取全量 JSON，副本在会话 scratchpad `ext/lb_false.json` |

前 10 名的标度分 / raw（10-04 抓取）：

| # | 队伍 / 模型 | Overall | PDS | MSE | NMAE | Fid | Reach | Jac |
|---|---|---:|---|---|---|---|---|---|
| 1 | Illumina AI / iCell | 0.4301 | 0.878 / 0.896 | 0.680 / 0.342 | 0.373 / 0.776 | 0.148 / 0.555 | 0.431 / 0.460 | 0.070 / 0.057 |
| 2 | cqawesome / m0957f | 0.3289 | 0.825 / 0.872 | 0.460 / 0.551 | 0.289 / 0.826 | 0.084 / 0.538 | 0.290 / 0.336 | 0.025 / 0.040 |
| 3 | Testh / test | 0.3076 | 0.803 / 0.862 | 0.435 / 0.574 | 0.247 / 0.851 | 0.057 / 0.529 | 0.286 / 0.333 | 0.017 / 0.037 |
| 10 | SQ W / sq-1004a | 0.2820 | 0.807 / 0.864 | 0.375 / 0.631 | 0.193 / 0.883 | 0.050 / 0.527 | 0.260 / 0.310 | 0.007 / 0.033 |
| 802 | **OpenProphetDB / depmap-knn-s2-h1-s930-knn** | 0.0115 | 0.148 / 0.568 | 0 / 7.907 | −0.041 / 1.023 | −0.069 / 0.492 | 0.048 / 0.122 | −0.017 / 0.024 |

我们在榜上的情况：

- **当前公开显示的是我们最近一次提交，也就是 kNN，0.0115，排第 802 名。** 榜单按“最近一次提交”排名，所以之前 0.049 的 Shared 条目已经被覆盖；按当前分布，0.049 大约在第 728 名。【外部】同上
- final 榜接口（`get_final=true`）返回 HTTP 400 “Leaderboard is not available yet”。【外部】同上，13:14 UTC

**和第一名的差距，主要在表达幅度（MSE）和 PDS**：

- 第一名 MSE 标度分 0.68（raw 0.342），我们是 0。我们最好条目的 raw MSE 为 4.72，比“原样输出 NTC”（raw 约 1.02）还差。
- 在 0.049 条目与第一名之间 0.381 的总差距中，MSE 一项贡献 0.113，是最大的一项。
- 来源：【外部】同上；【证据】`docs/research/leaderboard_prediction_counts_scores.csv`

公开方法的参考点：

- **AtlasShift / Simple Atlas Transfer**（[`kaipengm2/Virtual-Cell-Challenge-2026`](https://github.com/kaipengm2/Virtual-Cell-Challenge-2026)，MIT）：
  - 0.1546 已被 **5 个不同条目复现**，它们分数完全相同，排第 363–368 名，raw MSE 0.943。作者的新版本为 0.1804（第 239 名）。
  - 做法：以 K562 GWPS、Xaira HCT116/HEK293T、H1、CD4（DE 统计）为源加权迁移；先减去各源的共同响应，再把幅度收缩到 0.6/0.3；做启动子邻近校正；最后用匹配两阶矩的整数计数生成器出 counts。
  - 【外部】同上；代码副本在 scratchpad `ext/atlasshift/`
- **参赛者自述，未复现**：
  - MMatinGerami：多源迁移把分数从 0.069 提到 0.137；A/B/C 的基础表达更接近 Flex 化学测的 H1/CD4。
  - yashnil：基于注释的先验迁移后基本失效；改用直接实测的响应后，从 −0.062 升到 0.139。
  - 第 19 名（0.2673）：用 K562、X-Atlas、CD4、iPSC、L1000 多源迁移。
  - 提到 State/scGPT/Stack 的条目，最好的只在第 275–296 名（0.162–0.170）。
  - 来源：【外部】各自 GitHub README 与榜单 description 字段
- **前 6 名都没有填方法描述**，也没找到公开的代码或论文。【外部】同上

### 1.5 规则约束

- 验证榜和 final 都是**每天最多 2 次计分提交**，UTC 午夜重置；同一时间只能有 1 个提交在评分。验证榜 10/22 截止，按每队**最近一次**提交排名，不取最好的一次。【外部】[rules](https://virtualcellchallenge.org/rules)、[FAQ](https://virtualcellchallenge.org/faq)，10-04 读取；仓库记录 9/28 的第 3 次提交被拒（`experiments/masked-response/outputs/masked-response-qc-s01/predictions/official-abc/linear/submission-blocked.json`）
- final 期间不显示榜单，只有每队**最后一次** final 提交参与评奖；入围者的分数会被复核。【外部】同上
- Arc 10-01 的数据说明：
  - 所有细胞系用同一批 sgRNA；
  - 平台为 10x Flex，中位约 20k UMI/细胞；
  - 入选靶点要求中位敲低 ≥80%。

  来源：【外部】[Arc：Behind the data 2026](https://arcinstitute.org/news/behind-the-data-virtual-cell-challenge-2026)
- cell-eval2 已在 10-01 发布 0.18.0，changelog 称不改变已有分数；仓库冻结的是 0.16.0 / `5e648335`。【外部】子代理读取 changelog；逐条一致性【待核实】
- 9/16 修订的规则：允许用实验或文献的实测结果训练、微调模型；**不允许**把实测结果放进提交，也不允许用它们修改模型的预测。不经学习、直接把源细胞系 Δ 搬过来的 atlas 迁移处在灰区，稳妥做法是把权重和缩放当成在公开数据上拟合的模型参数。【外部】KnowGraph `VCC-RULES-0916`（转述，非规则原文）
- 进入决赛的队伍需要提交一份公开的方法描述（训练数据、处理、模型、非学习组件），不要求代码。【外部】KnowGraph `VCC-FINAL`
- Xaira 数据的许可是 CC BY-NC-SA 4.0，State 代码也是非商业许可，参赛身份是否商业需要确认。【外部】KnowGraph `DATA-XATLAS`、`VCC-RULES-DATA`

---

## 2. 已经发生的事

### 2.1 时间线

| 日期 | 阶段 | 要点 | 来源 |
|---|---|---|---|
| 09-13 ~ 09-19 | 数据获取与描述统计 | 237 GB+ 原始数据入库；H1、Replogle、Nadig、Jiang、McFaline、scPerturb、scBase、Tahoe、官方 NTC、先验都做了完整描述统计（9/19 一天 96 次提交） | 【证据】`git log`；`docs/research/repository_analysis_inventory_2026-10-03.md` §data |
| 09-19 ~ 09-27 | 历史路线 exp001–exp008 | XGB pair/residual/log2FC、shared shrink、module CVAE、conditional shared；6 次官方提交，分数在 −0.303 到 −0.031 之间 | 【证据】同上 §九条历史路线 |
| 09-27 夜 | 研究管理重新初始化 | 引入九轴 Method Space、Experiment DAG、Evidence Ledger；**历史结果不导入**，方法全部重置为 UNTESTED | 【证据】提交 `b989bdd`、`7093f8c`；`docs/RESEARCH.md` §初始化依据 |
| 09-28 | 首轮当前协议路线 | init-linear、masked-response、QC、三机制整包、跨基因解码器、细胞 VAE；官方 Linear 0.0338、Shared 0.0490、Shared-QC 0.0501 | 【证据】Ledger `E-BASE` … `E-CELL-VAE-ALIGNED-PACKAGE` |
| 09-30 | 用户纠偏，转向任务约束 | 重新设计数据、生成、评估；完成 leaderboard counts 诊断、task-observation、local-capability、population-emission | 【证据】`docs/research/task_constraints_review.md`；`docs/RESEARCH.md` 第 57 行 |
| 10-01 ~ 10-03 | 评估支持与 DepMap | full-eval-support 经过多次 OOM 后完成；DepMap 7 方案 × 6 折 = 42 个拟合；kNN 官方 0.0115 | 【证据】`experiments/full-eval-support/REPORT.md`；`experiments/depmap-response/REPORT.md` |
| 10-01 ~ 10-04 | 新数据 | Xaira 已获取并建索引；Jost 已入库；CD4 下载中 | 【证据】`data/README.md` 10-03 更新 |

### 2.2 当前协议下的 12 条路线

数值是本地 S2-H1 主面板 Overall（通常为 280 靶）。H1 是 Flex 化学，训练用的是其余四个背景；这些数值不能和官方 A/B/C 的分数直接比较。

| 路线 | 规模 | 结果 | 结论（Ledger 状态） | 来源 |
|---|---|---|---|---|
| init-linear | 7 个 run | Shared 0.2109；条件线性 0.1611；去掉条件 +0.0075；α 1→0.1 后 0.1861；真实程序 0.1410，不如 raw 0.1659、PCA 0.1659、随机程序 0.1748 | not_supported（E-BASE、E-CONDITION-ABLATION、E-RIDGE-STRENGTH、E-PROGRAM） | `experiments/init-linear/REPORT.md` |
| masked-response | 6 个 run | 缺测补全：SVD 0.2090、真实程序 0.2099、随机 0.2105，都没有超过 Shared 0.2109；QC 0.2128 对比匹配随机删减 0.2123，差 +0.0004 | not_supported（E-MASKED-RESPONSE、E-SHARED-QC） | `experiments/masked-response/REPORT.md` |
| response-transport | 1 | MLP 迁移 0.0899 | not_supported（E-MECHANISM-PORTFOLIO） | `experiments/response-transport/REPORT.md` |
| composition-response | 1 | logFC + 组成约束 0.1922 | 同上 | `experiments/composition-response/REPORT.md` |
| distribution-response | 1 | 均值/离散度 + Gamma-Poisson −0.1223 | 同上 | `experiments/distribution-response/REPORT.md` |
| gene-conditioned-response | 1 | 跨基因解码器 0.1712 | not_supported | `experiments/gene-conditioned-response/REPORT.md` |
| conditional-cell-response | 1（另有 1 个失败） | 条件计数 VAE 0.0803；零响应时基线漂移，辅助 BH 检出 13,697 个 DE | not_supported | `experiments/conditional-cell-response/REPORT.md` |
| task-observation | 1（另有 1 个失败） | 观测/生成接口 16/20 项通过 | not_supported（R2） | `experiments/task-observation/REPORT.md` |
| local-capability | 1（另有 2 个失败） | 评估器接口 87/90 项通过 | inconclusive | `experiments/local-capability/REPORT.md` |
| population-emission | 1 | 总体校准后独立抽样；群均值方差比 0.83–1.05，原 IPF 为 0.13–0.38；但 DE Jaccard 在五个背景都下降 | signal | `experiments/population-emission/REPORT.md` |
| full-eval-support | 1（另有 3 个失败） | 15 个面板的六项和 Overall 全部可定义 | supported | `experiments/full-eval-support/REPORT.md` |
| depmap-response | 42 | 真实 DepMap 映射优于匹配置乱：MLP 的 PDS +0.083、Overall +0.098，五个 S3 背景 5/5 同向；残差相关增益 0.0156/0.0187，未达 0.02 门槛；S2-H1 上 Shared 0.2162，MLP 0.0796 | not_supported（整包不晋级） | `experiments/depmap-response/REPORT.md` |

来源汇总：【证据】Ledger `evidence` 各条目；`docs/research/repository_analysis_inventory_2026-10-03.md` 表“12 条当前路线”。DAG 当前有 104 个节点（64 completed / 7 failed / 33 draft）、46 个比较。【证据】用 Python 读取 `docs/research/experiment_dag.json` 统计

### 2.3 历史路线（重新初始化前，按协议不作为当前方法证据）

| 路线 | 结果 | 官方分 |
|---|---|---:|
| exp001 context-pair XGB | 共同轴 MSE：static 0.004047，pair 0.004073，relation 0.004106，都不如 shared-shrunk 0.003654 | −0.3032 |
| exp002 shared response shrink | 跨研究收缩相对零响应，MSE 降 8.6%–9.0% | −0.1978 |
| exp003 module CVAE | 真实/随机/无先验三种响应 MSE 都差于零响应（0.0184 / 0.0178 / 0.0168 vs 0.0072）；H1 折 cycle4 | −0.0359 |
| exp004 shared conditional | 消融矩阵未完成；导出两个 checkpoint | −0.0311 / −0.0570 |
| exp005 / exp006 | 中断或未完成，不构成方法结论 | — |
| exp007 log2FC XGB | H1 0.0204，低于同靶转移 0.1868 | −0.1278 |
| exp00701 | 连续响应 / +DEG / +模块，H1 最佳 0.127 / 0.111 / 0.104 | — |
| exp008 | 只有 PLAN（9/27 约 30 次文档提交），未执行 | — |

【证据】`docs/research/repository_analysis_inventory_2026-10-03.md` §九条历史路线；`docs/experiments/leaderboard-score-interpretation.md`

### 2.4 全部官方提交（同一 panel `vcc2026-val-1`，anchors r4）

| # | 模型 | 来源 run / 折 | entry | Overall | PDS（标度） | MSE（标度） |
|---|---|---|---|---:|---:|---:|
| 1 | XGB-pair | exp001 / 全背景 | BekjoBqdOayh8j6X61RP | −0.3032 | 0.008 | 0 |
| 2 | Shared-shrink | exp002 | LS34Z7nIuD3njQC4D4TM | −0.1978 | 0.232 | 0 |
| 3 | Module-CVAE | exp003 H1 折 cycle4 | wDKL5Vvgw5wf2LuqfdUG | −0.0359 | 0.002 | 0 |
| 4 | Cond-CVAE-H1 | exp004 | payUWYmujElaWQXgnrgl | −0.0311 | −0.001 | 0 |
| 5 | Cond-CVAE-K562 | exp004 | 8RFfav38cQ1NftnO8EFb | −0.0570 | −0.010 | 0 |
| 6 | XGB-logFC | exp007 | VxirEGSQBl0HhgKDxIYj | −0.1278 | 0.000 | 0 |
| 7 | Linear | init-linear-s01 / S2-H1 | BpAGRy6zY8YFxKlLvmua | 0.0338 | 0.174 | 0 |
| 8 | Shared | init-linear-s01 / S2-H1 | vWd1Z4K9tJ2tjfns2gSj | 0.0490 | 0.371 | 0 |
| 9 | **Shared-QC** | masked-response-qc-s01 / S2-H1 | due9As7B4FXieRh6I8No | **0.0501** | 0.376 | 0 |
| 10 | DepMap kNN | depmap-knn-s2-h1-s930 / S2-H1 | VbwIq7s66YimJ1CgCfT8 | 0.0115 | 0.148 | 0 |

【证据】`docs/research/leaderboard_prediction_counts_scores.csv`；`experiments/depmap-response/REPORT.md` §官方已发布结果

- 十次提交的 MSE 标度分**全部是 0**：raw 在 1.47–11.97 之间，都高于 b≈0.99。【证据】同上
- 第 7–10 次都是已有 S2-H1 拟合的“官方核查”，没有一次专门为提交重新训练。【证据】Ledger `E-BASE-OFFICIAL`、`E-SHARED-QC-OFFICIAL`、`E-DEPMAP-KNN-H1-OFFICIAL`
- 当前协议下有 4 对“本地 S2-H1 vs 官方”的分数，排序完全一致（Spearman = 1），但官方比本地低 0.127–0.163。线性拟合约为 `官方 ≈ −0.058 + 0.51 × 本地`，即本地 +0.02 大约只对应官方 +0.006–0.010。【证据】由上表与各 REPORT 的本地分数计算（评测子代理复算，我抽查了 4 对数值）
- 第 9 和第 10 次的生成器不同：第 9 次用固定 log-shift 加随机取整，第 10 次用群体校准发射器。因此两者的差异混合了模型差异和生成器差异。【证据】`experiments/masked-response/OFFICIAL-PLAN.md`；`experiments/depmap-response/REPORT.md` §S2-H1 kNN 官方核查

### 2.5 已证伪或未获支持的方向

**范围限定**：每条结论只针对当时的具体实现和预算，大多数还是 S2-H1 单背景、单 seed。协议文本明确说，这些不等于该类方法普遍无效。【证据】Ledger 各条 `claim` 的限定语

| 方向 | 当前证据 | 边界 |
|---|---|---|
| NTC 条件项（背景条件化） | 去掉条件后 +0.0075，未达 +0.02 | 只测了 H1 单折、线性 ridge |
| NTC 表示：Reactome 程序、PCA、随机程序 | 真实程序不如随机程序 | 只测了 rank32 / 注入 NTC 侧 |
| 缺测读出的低秩或程序补全 | 三种都没有超过零回退 | rank32、α=0.1 |
| 低深度 QC 过滤 | 相对匹配随机删减只有 +0.0004 | H1 单折 |
| 神经迁移、组成 logFC、NB 分布、跨基因解码、条件 VAE | 都低于 Shared，部分为负分 | 每种只有一个配置、一次完整训练 |
| DepMap 功能映射（Ridge/MLP/kNN） | 有信号（比置乱好），但整包不晋级 | 单 seed；Ridge 未收敛 |
| 逐组 IPF 计数生成 | 不作为默认（4 个背景重建误差大） | — |
| 历史：XGB 系列、module CVAE、shared shrink | 官方全部为负 | 协议不同，没有导入当前证据 |

### 2.6 保留下来的正向能力

1. **同靶平均响应迁移（Shared）**：官方 PDS 标度 0.37（raw 0.67），明显好于无信息水平，而且排除靶基因后仍然成立。【证据】`docs/research/leaderboard_prediction_counts_REPORT.md` §3
2. **DepMap 功能先验对靶点区分有信息**：五个背景 5/5 优于置乱，可作为未见靶点的候选来源。【证据】`experiments/depmap-response/REPORT.md`
3. **群体校准 + 独立抽样的计数生成器**：工程上 15/15 项通过，已用于 kNN 的官方导出。【证据】Ledger `E-POPULATION-EMISSION`
4. **本地评估基础设施**：15 个面板（S2/S3/S4）都能可靠算出六项。【证据】Ledger `E-LOCAL-EVAL-SUPPORT-R4`

### 2.7 明确的能力短板

【证据】`docs/research/leaderboard_prediction_counts_REPORT.md` §4；`experiments/depmap-response/REPORT.md` §弱响应

- **表达幅度和 DE 都弱**：MSE raw 远大于 1，NMAE 接近 1，Jaccard 约 0.02，fidelity 低于基线。
- **全局未见靶点没有能力**：Shared 对 28 个未见靶点完全回退到 NTC 重采样。
- **未监督的读出基因**：10,853 个官方基因上没有任何直接响应。
- **靶点自身敲低偏保守**：残留 65%–68%，而官方材料称敲低 >80%。
- **背景特异修正**：目前没有可归因的稳定收益。

### 2.8 尚未尝试的方向

- **多源 atlas 迁移**：以 Xaira 两个背景加 H1 作为源，配合源模板去除和幅度校准。这是 KnowGraph 里的“里程碑④ M0”，仓库从未执行。【证据】KnowGraph `EXP-M0`；代码 grep 无结果
- Ledger 中 27 个 pending 问题和 DAG 中 33 个 draft，包括集合编码器、预训练表示（scGPT/scFoundation/SE）、GO 靶点先验、FiLM、shared+context 分解、两阶段训练、NB/flow 生成。【证据】Ledger `state=pending`；DAG `status=draft`
- STATE（ST）或 OT 集合分布整包：已有来源核查，没有实现。【证据】`docs/research/next_mechanism_sources.md`

---

## 3. 当前状态

### 3.1 最好的模型

| 口径 | 模型 | run | 分数 |
|---|---|---|---|
| 官方 A/B/C | Shared + 低深度 QC | `masked-response-qc-s01`（S2-H1 折，seed 1） | Overall **0.050079**；PDS 0.376，MSE 0，NMAE 0.047，fidelity −0.197，reach 0.105，Jaccard −0.031 |
| 官方（同族） | Shared | `init-linear-s01` 的 shared 臂 | 0.048975 |
| 本地 S2-H1 | Shared | `masked-response-shared-s01` / `depmap-shared-s2-h1-s930` | 0.210866 / 0.21621 |

两个本地数字来自**两套不同的本地标尺**，不能直接比较：

| | init-linear / masked-response 面板 | full-eval-support 面板（DepMap 系使用） |
|---|---|---|
| NTC（输入 / 评分） | 18,400 / 19,104 | 2,048 / 2,048 |
| 深度 | H1 原生，中位约 54k | 抽稀到约 20k |
| 参考细胞 | 沿用 `tasks.csv` 冻结身份 | 用 seed 930 重新抽取 |

来源：【证据】`experiments/task-observation/protocol.json` 第 15 行；`src/vcc_task/full_reference.py` 第 63–65 行（子代理核实）

同一 S2-H1 面板上，“靶点错配”对照（有响应幅度、没有靶点信息）也能得 0.099。所以本地 Shared 的 0.21 中，真正来自靶点特异信息的部分大约只有 0.11。【证据】`experiments/full-eval-support/REPORT.md` 表 1；后半句为【推断】

**Shared 的具体做法**【证据】`experiments/masked-response/OFFICIAL-PLAN.md`；`docs/research/leaderboard_prediction_counts_REPORT.md` §2：

- 对训练背景里见过的每个靶点，取训练背景平均的 log 空间 Δ。在官方靶点上，这个 Δ 实际只来自 K562。
- 把 Δ 加到目标背景的输入 NTC 上，再用固定 log-shift 和“保持文库的随机取整”生成计数。
- 未见靶点和未测基因都取零 Δ（等价于 NTC 重采样）。
- QC 版只在训练端额外删掉一部分低深度细胞。

### 3.2 位置

- 代码：`experiments/masked-response/src/`（复用 `experiments/init-linear/src/` 的数据、评估和提交函数）。
- 产物：`experiments/masked-response/outputs/masked-response-qc-s01/`，包括 `config.yaml`、`metrics.json`、`checkpoints/`、`predictions/official-abc/`（VCC 与完整回执）。
- W&B：`yjcyxky/virtual-cell-challenge`，run id 为 `masked-response-qc-s01`，group 为 `masked-response`。
- 【证据】目录列表；`metrics.json` 中 `research.git_commit = 9de9157`

### 3.3 怎么复现

```bash
# 训练 + 本地正式评分（经 research.py execute 门禁）
cd experiments/masked-response && ./reproduce.sh --run-id masked-response-qc-s01
# 官方导出：只生成并校验
cd experiments/masked-response && ./submit.sh
# 官方导出并提交（会消耗每日 2 次额度）
cd experiments/masked-response && ./submit.sh --submit
```

注意事项：

- 协议规定“标准入口不恢复已完成节点”。所以要从头重跑一次，需要按协议登记新的 run_id；直接重跑会被门禁拒绝。【证据】`docs/RESEARCH.md` 第 146 行
- 10-03 盘点时，`research.py status/check` 因 NAS 软链接路径（“reference escapes repository”）失败。【证据】`docs/research/repository_analysis_inventory_2026-10-03.md` 第 119 行。本次 `status` 用约 10 分钟以退出码 0 跑完（见附录 A）；`check` 本次没有运行。

### 3.4 正在进行的事

- **没有训练在运行**。
- CD4 数据的下载和收尾进程在后台运行（PID 1929550 / 1929551，自 10-02 起）。【证据】`ps`
- Ledger 的 `resource_allocation.exploration_slots` 是空的。下一轮方向只有文字描述：保留同靶监督响应、用功能映射处理未见靶点、校准弱响应和背景强度、跨基因共享输出，**尚未登记**。【证据】`docs/research/evidence_ledger.json` 的 `resource_allocation`

### 3.5 依据：为什么说最大杠杆在数据覆盖

全部为【推断】，依据如下：

1. 官方 PDS 主要来自同靶响应迁移，而官方靶点的直接监督只来自 K562 一个背景、7,680 个读出基因（【证据】§1.2）。Xaira 两个背景能为全部 300 个官方靶点提供近乎全轴的监督（18,401 个基因），文库深度也接近 2026 的约 20k。
2. 9/16 规则允许用这些数据训练和拟合。公开的 atlas 方案正是以 HCT116/HEK293T/H1/K562 为源，自报约 0.155，大约是我们最好成绩的 3 倍（【外部】§1.4）。
3. final 的 panel 会换，但 Xaira 是全编码基因筛选（18,903 个基因），所以对新 panel 的覆盖预计也接近完整。这一点【待核实】，要等 10/22 的 panel 发布。
4. 反方证据：同靶响应在不同背景之间的相关中位数只有 0.0035，绝对表达相关却有 0.761（【证据】`docs/research/repository_analysis_inventory_2026-10-03.md` §同靶响应迁移性）。所以多源平均未必有效，需要用留一背景实验检验，还要做幅度收缩。

---

## 附录 A：`research.py status` 运行记录

- 按 AGENTS.md 要求，于 2026-10-04 09:08 启动，约 10 分钟后以**退出码 0** 结束；运行期间长时间处于 NFS 的 D 状态。10-03 盘点时它以退出码 1 失败，这次通过了，原因【待核实】。
- 首行汇总：`Research: 9 axes, 59 methods, 33 draft nodes; 27 source records, 27 pending hypotheses, 19 local conclusions`。
- 输出共 134 行，其中：
  - 列出 27 个 PENDING 问题，包括已被后续 R2/R3/R4 替代、实际已闭环的旧问题，例如 `E-LOCAL-EVAL-SUPPORT`、`E-TASK-DATA-GENERATION`；
  - 33 个 draft 节点的启动缺项；
  - 64 个 completed 节点**一律**显示 `await result/evidence`，即使对应比较已经在 Ledger 关闭。
- **输出里没有“当前最高优先问题”**：Ledger 的 `queue` 和 `exploration_slots` 都为空。
- 结论：`status` 目前给不出可执行的下一步。这一点在 02 审计中展开。

## 附录 B：本文没有做的事

- 没有重算任何大文件哈希，没有重跑统计、训练或评分，没有访问 W&B 线上数据。
- 历史文档里写的“当前”“正在运行”，只描述当时的状态。本文以 DAG/Ledger 状态和实际进程为准。
