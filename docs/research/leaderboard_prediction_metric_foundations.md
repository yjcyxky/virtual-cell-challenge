# 已提交预测的 counts 与能力诊断：指标依据

日期：2026-09-30。本页是本轮分析前确定的诊断设计，不包含尚未计算的模型结论。依据是 [KnowGraph 注意力索引](knowledge_graph.attention.json)、[五背景原始数据事实](dataset_foundations.md)、[文献卡片 L01/L04/L09/L10](literature_foundations.md) 与下面列出的原始来源；图谱是检查提示，不是模型有效性证据。分析对象限定为保存了 leaderboard 评分回执、能够关联原预测文件的模型；同一预测的重复 entry 不增加模型数。

## 先固定解释范围

令同一背景、同一靶点的生成计数矩阵为 $Y\in\mathbb N_0^{n\times G}$，目标背景公开 NTC 为 $X$。$L_i=\sum_gY_{ig}$，$C_{ig}=10^6Y_{ig}/L_i$。零文库另列，不让其悄悄进入 CPM 分母。所有模型使用同一基因轴、同一目标集合和同一 NTC 参照。官方 A/B/C 只有公开 NTC，没有本地可读的真实扰动标签，因此 **接近 NTC 不是扰动正确，远离 NTC 也不是错误**。预测分布诊断与 leaderboard 真实标签评分回答不同问题。[本地输入事实](dataset_foundations.md)

来源五背景是三个研究组，测量化学、基因覆盖及时间点不同；本地真实扰动仅在其真实测量的官方子轴比较，分母也限定该轴，不能用未测补零。A/B/C 完整提交则在完整 18,533 轴检查。图谱 `WL-FLEX` 关于交集的建议不能覆盖此契约。[数据契约](challenge_protocol.md)

本轮不更改官方六项评分。若复用其结果，记录原 run、entry、预测哈希、panel、anchor_version、raw/normalized 和有效数。固定 scorer 为 [scorer.json](challenge_2026/scorer.json) 所记 `5e64833518a6603a0301cbe28185d49c30f4a986`。上游 metric brief 的示例 b/r 属于 `0.15.0/rule_version 3`，不能拿来替代某次榜单回执的尺度。[官方指标来源](https://github.com/ArcInstitute/cell-eval2/blob/5e64833518a6603a0301cbe28185d49c30f4a986/docs/vcc2026_metrics/vcc2026-metrics-brief.md)；[r4 证据边界](r4_anchor_audit.md)

## 注意力节点转成检查项

| 直接节点 | 本轮采用的检查项 | 必要一跳补充 |
|---|---|---|
| `WL-UMI`、`VCC-OUTPUT` | 整数、有限、非负、轴/标签/数量，以及稀疏存储 | `WL-DEPTH`、`ST-COUNT-MODELS` |
| `WL-DEPTH`、`ST-COUNT-MODELS` | 文库、检出基因数、零值和均值–方差须联合看 | `ST-SAMPLING-NOISE`、`EXP-EMITTER-V0` |
| `WL-NTC`、`EVAL-DE-PIPELINE` | 同背景 NTC 参照；独立 NTC null 与生成器空响应检查 | `EVAL-REF-CONTROL`、`ST-WILCOXON`、`ST-NORMALIZE` |
| `WL-KD-EFFICIENCY` | 靶基因相对抑制及可估计覆盖；与下游能力分开 | `EVAL-PDS` |
| `EVAL-MODE-COLLAPSE`、`EVAL-PEARSON-DELTA` | 跨靶响应区分度、共享模板占比、去模板残差 | `EVAL-MSE`、`MOD-MEAN` |
| `EVAL-BIO-FIDELITY`、`ST-DISTANCES` | 细胞内联合结构、状态异质性，补充官方边际评分 | `EVAL-AGGREGATE`、`DG-DIST-LOSS` |
| `MY-EFFECT-EMITTER` | 区分已学习响应与固定/继承的细胞生成规则 | `EXP-EMITTER-V0` |

上述节点的 `vcc_implication` 以 description 中“对 VCC 2026 意味着什么”段落读取。未将图谱的实验排期、外部案例或推断当成本轮必过阈值。

## 可计算指标与判读

### 1. 文件与计数合法性：硬约束

统计非有限/负值/小数个数，零文库数、最大 $L_i$、显式零数、实际非零数和存储元素数；核对 context/target 集合、每组 n、基因集合及顺序。当前上传要求每组 400 个细胞、完整三个背景、18,533 基因、无 NTC 行、每细胞总数不超过 $10^6$、总细胞不超过 400,000、存储元素不超过 $4.75\times10^9$。整数允许以 float dtype 保存；存储上限不等于“生物零比例必须多少”。包评分接口与 CLI 上传规则不能混为一谈，上传规格以 [官方 CLI Guide](https://vcc-cli-wiki.virtualcellchallenge.org/#submission-requirements-2026) 为依据。

### 2. 文库与稀疏性：经验性诊断

对每个 context×target 报告 $L_i$、检出数 $D_i=\sum_g1(Y_{ig}>0)$ 的 p05/p25/p50/p75/p95、均值、CV；报告零比例 $1-\mathrm{nnz}(Y)/(nG)$、每细胞 singleton 比例、最高表达 10 个基因的 counts 占比。与同背景真实 NTC 的等 n 抽样分布比较，可附一维 Wasserstein 距离 $W_1$，保留其单位，并按深度分位层比较 $D_i$，避免将测序深度变化误判为表达复杂度变化。

约 20k 是官方参考的中位深度条件，不是要求每个细胞都等于 20k，也不能推出扰动组方差必须等于 NTC。高零率可能来自低深度、组成变化或少数细胞强表达；低零率可能来自均值平铺或真实广泛表达，需与下一组指标联合判断。[官方参考数据条件](https://github.com/ArcInstitute/cell-eval2/blob/5e64833518a6603a0301cbe28185d49c30f4a986/docs/vcc2026_metrics/vcc2026-metrics.md)

### 3. 均值–方差与零值耦合：经验性诊断

逐基因计算 raw-count 均值 $\mu_g$、样本方差 $s_g^2$、零率 $z_g$、Fano $s_g^2/\mu_g$（$\mu_g>0$）；CPM 另报均值、方差和生成/NTC 方差比。按 NTC 表达量分层，报告方差比中位数、低于 0.5/高于 2 的比例及分母有效数。0.5/2 是描述性切点，不是已校准合格线；NTC 方差为零的基因不强行加常数构造比值，另列。

可计算 Poisson 的 $e^{-\mu_g}$ 或矩估计 NB 的 $(\theta_g/(\theta_g+\mu_g))^{\theta_g}$ 与实测零率的残差；需要先按深度处理，且只作为模型假设诊断。计数测量噪声与真实表达异质性必须分开解释，NB 并非所有 assay 的生物真理。[Sarkar & Stephens 原论文](https://pmc.ncbi.nlm.nih.gov/articles/PMC8370014/)；[Svensson 原论文](https://www.nature.com/articles/s41587-019-0379-5)

**不能把 Fano<1 判为硬违规**：固定文库、固定组成的多项分布有 $\mathrm{Var}(Y_g)=Lp_g(1-p_g)<\mu_g$。同样，NTC 方差偏离可能是真实扰动异质性；只有与真实扰动、零响应输出或已知生成机制结合，才可归因于生成器。各模型跨不同靶点合池的方差不能冒充单靶点细胞间方差。

### 4. 目标抑制与极端表达：生物预期检查

对靶基因可测且 NTC 平均 CPM>5 的任务，计算 $\mathrm{KD}_{t}=1-\bar C_{t,t}/\bar C_{0,t}$ 及 $\mathrm{LFC}_{t,t}=\log_2[(\bar C_{t,t}+10^{-9})/(\bar C_{0,t}+10^{-9})]$，报告 KD 分位数、下调比例、KD≥0.8 的比例、反向上调任务和不可估计数。CRISPRi 预期是转录抑制，不要求每个细胞必为零；Replogle 原研究中敲低强度也存在差异。[Replogle 2022 原论文](https://pmc.ncbi.nlm.nih.gov/articles/PMC9380471/)

0.8 作为图谱 `WL-KD-EFFICIENCY` 提示的目标面板经验参照，不能替代隐藏参考的逐靶 KD 真值，也不是提交合法性门槛。若模型代码直接固定靶点残留表达，KD 合格证明规则被执行，不证明模型学会敲低。官方六项排除自身靶点（PDS 排除全部面板靶点），因此必须单独诊断下游响应。[官方评分基因限制](https://github.com/ArcInstitute/cell-eval2/blob/5e64833518a6603a0301cbe28185d49c30f4a986/docs/vcc2026_metrics/vcc2026-metrics.md)

对 NTC>5 CPM 的基因另报预测全零比例、极端 LFC 比例和上下调基因数。图谱 `ST-LFC` 的“不能整体归零”不是硬约束：真实强抑制可以归零；应检查是否大面积、跨靶点、集中于训练未测基因，以及是否由取整/截断系统性造成。

### 5. 学到模板还是靶点差异：预测内部结构

使用 $b_{t,g}=\log(1+50000\sum_iY_{tig}/\sum_{ig}Y_{tig})$、$\Delta_t=b_t-b_{NTC}$。用于跨靶点结构分析时统一去除所有 300 个靶基因，防止手工敲低制造区分度。每个背景独立计算：

- 响应强度 $\|\Delta_t\|_2$；同时报告逐细胞 CPM 算术均值上的 LFC，不能把这两种量互换。
- 共享能量比例 $Q=P\|\bar\Delta\|_2^2/\sum_t\|\Delta_t\|_2^2$，分母为零时未定义。$Q$ 接近 1 说明预测以同一偏移为主，**不说明该偏移正确或错误**。
- 去共享矩阵 $R_t=\Delta_t-\bar\Delta$ 的奇异值能量、首成分占比及参与率 $r_{eff}=(\sum_j\sigma_j^2)^2/\sum_j\sigma_j^4$；成对余弦的中位数/分位数、完全相同/近似相同预测数。
- 将每组 400 细胞分为两个不交叉半组，比较去共享响应的一致性；或对照等 n NTC 伪靶点的残差能量。跨靶点差异小于细胞抽样波动时，不能声称获得稳定靶点信息。固定总和约束会压低此种波动，必须结合生成机制说明。

这些是本轮提出的描述性指标，受 Systema 区分共同偏移与特异响应的思路启发；高秩、高区分度也可能只是随机噪声。只有真实扰动标签上的 PDS、标签置乱对照、去模板预测–真值相关或方向/幅度评分才验证正确性。[Systema 原论文及 reference-sensitive metric 限制](https://pmc.ncbi.nlm.nih.gov/articles/PMC13271886/)

### 6. DE 负担与伪 DE：必须分清检验对象

从冻结官方流程取得预测 vs NTC 的 DE 表：CPM、对照>5 CPM 过滤、Wilcoxon、每扰动 BH；报告每任务显著基因数、上下调数、显著集合的跨靶点重合、单基因在多少靶点显著。该结果称“预测 DE 负担”，**不能直接称假阳性**，因为扰动本应造成变化。[冻结官方配置](challenge_2026/scorer.json)

伪 DE 需两类 null：不交叉的真实 NTC 子集相互比较；同一生成器输入零响应、与独立评分 NTC 比较。固定 n、参考规模、guide/批次分层、seed 和同一检验基因集，比较显著基因数与 LFC/AUC 分布。经验 null 的漂移/宽度是参照，不能简单要求“5% 基因显著”：BH 的 FDR 定义不是显著基因比例。重复 null 的抽样不增加独立背景；若没有零响应输出，本轮只能结合已登记 null 证据讨论生成器风险。

### 7. 细胞联合结构与有限群体抽样：补充能力证据

在预先固定且只用允许 NTC/训练数据拟合的基因或 PC 空间中，比较去组均值后的 covariance trace、特征谱/有效秩、选定基因对相关结构；可用样本量匹配的能量距离或 MMD 衡量分布差异。对 NTC 的距离只刻画状态变化，不是扰动误差。没有真实扰动时，不能认证新亚群、基因调控网络或细胞状态迁移是否正确。[Replogle 的细胞状态/扰动分布分析](https://pmc.ncbi.nlm.nih.gov/articles/PMC9380471/)；[MMD 原始定义](https://jmlr.org/papers/v13/gretton12a.html)

若存在同一条件多次独立生成，比较重复群均值方差与独立细胞抽样预期：CPM 均值有 $\mathrm{Var}(\bar C_g)\approx s_g^2/n$；相关/固定总和生成需说明预期不同。每组列总和被精确锁定时，单细胞可以有方差而群均值不再有应有的随机波动。**一个提交文件无法直接估计重复生成方差**；从文件 bootstrap 只能估计“把这些细胞当 iid 时”的变异，不能证明生成器实际 iid。该项需代码审查和已登记重复生成证据支持。

## 可支持的结论边界

本轮优先回答三个问题：输出是否是合法且统计形态合理的计数；不同靶点是否产生超出模板/抽样噪声的变化；已有真实标签分数是否支持这些变化的方向、幅度和响应基因选择。不能仅凭“像 NTC”、一个 UMAP、绝对表达高相关或总体榜单分，声称学到了扰动机制。

最终结论应拆为“学习得到的平均下游响应”“生成规则继承的 NTC/测序形态”“固定规则施加的靶点敲低”“尚无证据的扰动后异质性/协方差”。若某能力来自规则或复制，不把它算作学习能力；若隐藏真值或实际诊断未覆盖，则明确未验证，不从缺证据推成失败。

## 九份提交的源码归因边界

本节在计数扫描期间补充实现依据，不改变前述诊断判据。逐份读取原 run 的 `config.yaml` / `export-identity.json`，再用 `git show <记录版本>:<源码路径>` 核对，未运行模型、改预测或使用当前源码替代旧版本。表中版本是生成代码版本；有独立训练版本时同时列出。`submitted: false` 是部分 export identity 生成时的状态，不覆盖随后保存的 published 回执。

| 提交 | 原训练/导出身份 | 学到或估计的量 | 固定规则、继承与外推边界 |
|---|---|---|---|
| exp001 `20260923-a` | pipeline `4094b3b2227f53f0e2b1c50e8b1bd4b7d75263db`；[config](../../experiments/exp001-context-pair-xgb/outputs/20260923-a/config.yaml) | 27 棵树的 context-relation XGBoost：由基因对先验、目标背景 NTC 特征及关系特征预测平均 log 响应 | 无未见 target 的硬性零响应回退；训练未测的 400 个 readout 响应强制为零。按基因缩放 NTC 模板；靶点乘数强制 0.1，不能归为学习到 90% 敲低。 |
| exp002 `20260922-d` | pipeline `e28c792e6a1b5d3f63105bc3c9cc4d69b4151d08`；[config](../../experiments/exp002-response-transfer-validation/outputs/20260922-d/config.yaml) | 各背景观测到的同靶点平均 log 响应，加通过背景留出拟合的收缩系数：一个 donor 为 0.25，多个 donor 为 0.5 | 没有 donor 的 target×readout 为零响应；无 target donor 的 28 靶下游回退 NTC，但仍强制自身靶点乘数 0.1。响应没有目标背景条件项，输出背景差异来自 NTC。 |
| exp003 H1 holdout / cycle 4 | train `b16daa16aa9b28e3d5ad34c678d66b978cadffbd`；export `49e00ea1cae827af83a06147f8909e0d8c44c67b`；[identity](../../experiments/exp003-context-module-cvae/outputs/20260925-lodo-response-s17/predictions/leaderboard-holdout-H1-cycle-0004-seed-101/export-identity.json) | 条件潜状态的均值/尺度/混合权重，模块与残差的均值响应，条件 NB 离散度，以及可正可负的 on-target head | NTC 提供 raw 均值、矩估计离散度与 PCA/KMeans 状态起点；不复用单细胞模板。未见 target 的 ID embedding 归零，仍通过模块先验/target encoder 外推；7,632 个未监督 readout 仍原样生成，无 NTC 回退。 |
| exp004 H1 holdout / cycle 11 | train `be92538b80b67745d92878002966afa7b2157875`；export `40f68a1ef910d2351d4ac42a7684bc53c5073e67`；[identity](../../experiments/exp004-shared-response/outputs/20260925-exp004-conditional-s17/predictions/leaderboard-holdout-H1-cycle-0011-seed-101/export-identity.json) | exp003 的条件生成框架，加共享 gene-feature readout/dispersion/gating 与 context-dependent relation operator | 7,632 个训练未监督 readout 通过共享生物特征参数外推；未见 target 也有共享特征 embedding；无固定敲低率或未测基因零响应回退。 |
| exp004 K562 holdout / cycle 1 | 同一训练版本；export `2b73c4dcf5bc99ba8e83f79f6bbc73d0436cd7ee`；[identity](../../experiments/exp004-shared-response/outputs/20260925-exp004-conditional-s17/predictions/leaderboard-holdout-K562-cycle-0001-seed-101/export-identity.json) | 同一架构，独立背景划分和训练 checkpoint | 因训练包含 H1，未监督 readout 为 401 个；不同来源、周期和权重，不能把与上一行的差异单独归因于 readout 覆盖。 |
| exp007 round 512 | train `49e2faf2aef2dd9552968c080366d9717c682efb`；export `138fb832513170a1e7210c41c3046038ea876ed2`；[identity](../../experiments/exp007/outputs/20260927-exp007-h1-log2fc-s17/predictions/leaderboard-round-0512-seed-101/export-identity.json) | 基因对先验、NTC 和自靶点指示特征的 XGBoost，预测逐细胞 CPM 算术均值的 log2FC | 全官方 readout 由共享树模型外推，没有未测 readout/未见 target 的零回退或强制 0.1。NTC 重采样后 IPF 校准到预测的均值组成，细胞深度沿用模板。 |
| init-linear `linear` | train `26a03062011fe902ee4a2c41cbdc287858ce3561`；export `3998aed3854f1ab1bcfc91ff87b185402024f58a`；[identity](../../experiments/init-linear/outputs/init-linear-s01/predictions/official-abc/export-identity.json) | mask-aware ridge 的靶点效应 $a_t$ 与 NTC 条件项 $x_{c,t}\beta$，监督为平均逐细胞 log1p(CP10k) 响应 | 未见 target 没有 $a_t$，但仍输出 $x_{c,t}\beta$，不是零响应；训练并集外 readout 置零 Δ。无强制自身靶点抑制。 |
| init-linear `shared` | 同一 checkpoint/导出版本 | 同靶点来源响应的 cell-weighted 均值；无目标背景响应条件项 | 未见 target 为零 Δ，未测 readout 为零 Δ；目标背景形态由 NTC 提供。与 `linear` 共享生成器与配对 seed。这里 shared 指同靶跨来源共享，不是所有靶点共享一个向量。 |
| masked-response `qc` | train `9de9157b4b6fb75571b897c86b5e61e9cc3bf645`；export `387d59f862cf52a6f9cc78bed51584ad6e888ead`；[identity](../../experiments/masked-response/outputs/masked-response-qc-s01/predictions/official-abc/export-identity.json) | low-depth 训练筛选后的 cell-weighted 同靶共享响应 | 实际 `representation.kind=shared`，没有 PCA/通路补全；未见 target、无 target×readout 监督的坐标均保留零 Δ。接口 arm 叫 `linear` 不代表它拟合了 ridge 背景条件项。生成器与 init-linear 相同。 |

### 直接决定 counts 形态的实现

**exp001/exp002：模板复制、乘性响应和人工敲低。** 每背景只选一次 guide 平衡的 400 个真实 NTC，随后全部 300 靶点复用这批细胞。`gene_factors` 把期望 log 响应反变换为乘数，缺 donor 的乘数设 1、上限设 10，然后覆写自身靶点为 0.1；`emit_counts` 按原模板文库重缩放并独立随机取整。因此精确的最终 CPM 敲低率会被文库重缩放影响，不必等于 90%；原模板为零的坐标不能被乘法生成诱导表达。文库、稀疏支持和很大部分相关结构来自 NTC；两者都没有学习扰动特异细胞方差。[exp002 冻结生成循环](https://github.com/yjcyxky/virtual-cell-challenge/blob/e28c792e6a1b5d3f63105bc3c9cc4d69b4151d08/experiments/exp002-response-transfer-validation/src/vcc_submission.py#L114)、[固定乘数和取整](https://github.com/yjcyxky/virtual-cell-challenge/blob/e28c792e6a1b5d3f63105bc3c9cc4d69b4151d08/experiments/exp002-response-transfer-validation/src/vcc_baseline.py#L209)、[exp001 readout 回退](https://github.com/yjcyxky/virtual-cell-challenge/blob/4094b3b2227f53f0e2b1c50e8b1bd4b7d75263db/experiments/exp001-context-pair-xgb/src/training.py#L279)

**init-linear 两臂/ masked-qc：NTC 上逐细胞加 log 偏移。** 每靶有放回抽 NTC，计算 `expm1(max(log1p(raw*10000/library)+delta,0))`，按模板文库重缩放并随机取整。正 Δ 会让原模板零坐标产生非零期望，负 Δ 可使一整段低表达细胞截为零；这是生成规则的数学后果，不是细胞响应异质性被单独学到。即使某基因 Δ=0，其他基因变化引起的共同归一化也可改变该基因的最终 counts/CPM，所以“零 Δ”不等于输出逐基因原样复制。完全零 Δ 的整组则退化为 NTC 重采样。三者均没有条件方差参数、混合状态权重或逐细胞响应头。[固定生成器](https://github.com/yjcyxky/virtual-cell-challenge/blob/3998aed3854f1ab1bcfc91ff87b185402024f58a/experiments/init-linear/src/evaluation.py#L25)、[linear/shared 未见 target 分支](https://github.com/yjcyxky/virtual-cell-challenge/blob/3998aed3854f1ab1bcfc91ff87b185402024f58a/experiments/init-linear/src/model.py#L92)、[masked 实际预测分支](https://github.com/yjcyxky/virtual-cell-challenge/blob/387d59f862cf52a6f9cc78bed51584ad6e888ead/experiments/masked-response/src/model.py#L135)

**exp007：固定均值组成、继承细胞模板。** 先将预测 log2FC 投影为总和为 1 的合法表达组成；按 batch/guide 比例为每靶抽模板，交替归一化行列，使未加权单细胞比例均值等于该组成，再恢复模板深度并随机取整。模板袋未检出的基因若预测组成为正，会先给所有细胞相同的比例起点。该 IPF 规则会压低不同模板袋带来的均值波动，不能把稳定群均值解释为更准的概率预测。条件细胞形态并未单独训练；模型学的是 LFC，IPF/抽样/取整是固定发射器。[冻结生成器完整定义](https://github.com/yjcyxky/virtual-cell-challenge/blob/138fb832513170a1e7210c41c3046038ea876ed2/experiments/exp007/src/generation.py#L5)、[预测全轴](https://github.com/yjcyxky/virtual-cell-challenge/blob/138fb832513170a1e7210c41c3046038ea876ed2/experiments/exp007/src/submission.py#L93)、[自靶点特征](https://github.com/yjcyxky/virtual-cell-challenge/blob/138fb832513170a1e7210c41c3046038ea876ed2/experiments/exp007/src/features.py#L72)

**exp003/exp004：确有可学习的分布参数，但不等于已学对。** 公开 NTC 给定 raw mean `base`、$\theta_0=\mathrm{clip}(\mu^2/\max(v-\mu,0.01),0.1,1000)$、四个潜状态中心/尺度/比例；在训练冻结的 16 维 PCA 坐标中，导出时对目标 NTC 重新估计 KMeans 统计。网络再条件化改变各状态中心、尺度和权重，解码均值与 $\theta$，最终 Gamma–Poisson 抽样。总文库并未逐细胞固定为 NTC，也没有生成后裁剪/重采样来修复非法细胞。训练包含 NB ELBO、响应/NTC 与文库矩辅助目标；因此它们“有能力学习分布”有源码支持，“正确重建扰动后异质性”仍须数据证据。[NTC 统计公式](https://github.com/yjcyxky/virtual-cell-challenge/blob/49e00ea1cae827af83a06147f8909e0d8c44c67b/experiments/exp003-context-module-cvae/src/state.py#L11)、[条件先验与 NB 生成](https://github.com/yjcyxky/virtual-cell-challenge/blob/49e00ea1cae827af83a06147f8909e0d8c44c67b/experiments/exp003-context-module-cvae/src/model.py#L90)、[训练目标边界](https://github.com/yjcyxky/virtual-cell-challenge/blob/49e00ea1cae827af83a06147f8909e0d8c44c67b/experiments/exp003-context-module-cvae/src/objectives.py#L17)

这三份 VAE 提交的 `on_target` 是无符号限制的可训练 head，不是强制负向敲低。未见 target 仅屏蔽未训练的 ID embedding，仍保留功能先验；exp004 还添加共享生物特征 embedding。exp003 的 gene-specific decoder 参数存在未被 readout 标签直接监督的行；exp004 删除这些自由基因行，改为共享特征解码，但未监督 readout 的外推可靠性仍待验证。导出对完整官方轴传入全 1 mask，`trained_readouts` 用于记录覆盖而非屏蔽输出。故不能将这些未监督基因的输出称为实测支持，也不能把它们与 init-linear 的零 Δ 回退混为一类。[exp003 target/decoder 参数](https://github.com/yjcyxky/virtual-cell-challenge/blob/49e00ea1cae827af83a06147f8909e0d8c44c67b/experiments/exp003-context-module-cvae/src/model.py#L54)、[exp004 共享读出与 target 编码](https://github.com/yjcyxky/virtual-cell-challenge/blob/40f68a1ef910d2351d4ac42a7684bc53c5073e67/experiments/exp004-shared-response/src/shared_model.py#L82)、[官方全轴 mask](https://github.com/yjcyxky/virtual-cell-challenge/blob/49e00ea1cae827af83a06147f8909e0d8c44c67b/experiments/exp003-context-module-cvae/src/submission.py#L43)

因此，本轮 counts 统计若发现“接近 NTC 的协方差/深度”，应优先结合模板型生成器解释；若发现 VAE 方差或 readout 覆盖异常，应区分条件分布未校准、无监督外推与真实响应变化，不能仅由模型名称裁决。以上是实现约束和可检验机制解释，不替代九份预测的实际结果。
