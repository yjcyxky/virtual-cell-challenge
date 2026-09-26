# EXP005：靶点覆盖与生物先验的证据边界

调查日期：2026-09-26。本文服务于训练前需求对齐，记录资料核查与方法建议；不代表已批准扩大训练组合，也不宣称先验已经带来改进。

## 当前判断

RPE1、HepG2、Jurkat 与官方 300 个靶点没有交集，会加强“让模型在不同靶点之间共享信息”的动机，但不能推出所有官方靶点都没有直接监督，更不能证明 PPI 或 TF 先验一定提高分数。需要同时区分：在某个背景没有该靶点的扰动记录、在所有训练背景都没有记录、基因作为表达 readout 是否被测量。这三个条件不同。

VCC2026 官方任务是未见扰动响应的细胞背景泛化；官方只给新背景 NTC 与靶点列表，没有声明所有靶点在外部数据库中均未被扰动过。当前公告也没有提供足以解释特定外部数据集零交集的详细靶点选择规则，不能把本地交集结果解释成官方刻意排除某类 essential genes。[Arc 2026 任务说明](https://arcinstitute.org/news/virtual-cell-challenge-2026)

## 原始覆盖核查

本轮只读审计读取 `obs` 元数据，没有读取表达矩阵。查询集合是 [官方靶点表](../data/raw/arc_vcc2026_controls/pert_counts.csv) 的 300 个唯一 symbol。H1 使用 `obs/target_gene`，Replogle/Nadig 使用 `obs/gene`，并以 `gene_id` 与 HGNC 作独立核对；不把 `sgID_AB` 构建体身份直接当 gene symbol。

| 原始来源 | 原始扰动靶点数 | 与官方 300 的交集 |
| --- | ---: | ---: |
| H1 三个原始 split 合并 | 300 | 25（Training 13、Validation 4、Test 8） |
| K562 GWPS | 9,866 | 272 |
| RPE1 | 2,393 | 0 |
| HepG2 | 2,393 | 0 |
| Jurkat | 2,393 | 0 |

输入分别是 `data/raw/arc_vcc2025_h1/adata_{Training,Validation,Test}.h5ad`、`data/raw/replogle2022/K562_gwps_raw_singlecell_01.h5ad`、`data/raw/replogle2022/rpe1_raw_singlecell_01.h5ad`、`data/raw/nadig2025/GSE264667_{hepg2,jurkat}_raw_singlecell_01.h5ad`。上述审计来源并集覆盖 **272/300**，H1 的 25 个全部已被 K562 覆盖；因此不能把三背景零交集称为“300 个官方靶点全部未见”。这也是为什么 K562 直接响应转移是有意义的参照。

例如 ACLY、ADNP、AKT2、BRPF1、HDAC1 在 K562 的 `obs/gene`、`sgID_AB` 前缀和对应 ENSG 上一致，后三个背景没有这些 intervention。相反，`var/gene_name` 是表达 readout：官方靶点作为 readout 在 K562/RPE1/HepG2/Jurkat 的字面 symbol 覆盖分别为 231/236/254/255。因此“未在该背景做该基因扰动”不等于“没有测到该基因表达”。HGNC 映射优先 approved symbol，其次只接受唯一匹配的别名；例如 AARS→AARS1。冲突不能任取首项。

精确 symbol、不区分大小写、HGNC canonical 映射得到同样交集。官方 `TMEM104` 是当前 `SLC38A12` 的历史名。`HOMEZ` 在 K562 的旧 ENSG 为 `ENSG00000215271`，当前 HGNC 对应 `ENSG00000290292`；不作历史桥接的当前 ENSG 精确比较会把 K562 错降为 271。Ensembl archive 提供后者为前者的候选替代（score 0.99），EBI Expression Atlas 将旧 ID 明确标为 HOMEZ。该桥接须显式记录，而非静默忽略 ID 不一致。[Ensembl archive](https://rest.ensembl.org/archive/id/ENSG00000215271)、[EBI HOMEZ 页面](https://www.ebi.ac.uk/gxa/genes/ENSG00000215271)

当前被审计旧实验合格缓存的交集为 H1 25、K562 258、其余 0，并集 260。它仅说明 **raw coverage 不等于训练 QC 后可用监督**，不是 EXP005 已决定采用的任务集。后续正式输入校验仍需输出自身每折合格覆盖。

### H1 三面板与 2026 原始包的再次独立复核

用户提出两年的 300 个靶点应一致后，本轮重新从 Arc 官方地址读取 [2025 Training CSV](https://storage.googleapis.com/arc-institute-virtual-cell-atlas/virtual-cell-challenge/2025/train/pert_counts_Training.csv)、[Validation CSV](https://storage.googleapis.com/arc-institute-virtual-cell-atlas/virtual-cell-challenge/2025/validation/pert_counts_Validation.csv)、[Test CSV](https://storage.googleapis.com/arc-institute-virtual-cell-atlas/virtual-cell-challenge/2025/test/pert_counts_Test.csv)。三者分别有 150、50、100 个唯一靶点，当前远端内容均与本地文件逐字节一致，且各自集合与对应 H5AD 的 `obs/target_gene` 完全一致。三个集合合并为 300，与 2026 面板的交集分别为 13、4、8，合并交集为 **25**；两边各有 **275** 个独有靶点。

2026 侧直接读取原始 ZIP 内的 `pert_counts.csv`，与解压文件逐字节相同，SHA-256 为 `f57edd7b912ebd718efc7ee9d0f334772513e7cc418d133ce525470e373b3276`，匹配 `SOURCE.json`。ZIP manifest 为 `season=2026`、`panel_id=vcc2026-val-1`。本轮未重新下载认证服务端的 2026 ZIP，因此此结论对应已登记的官方验证快照。

具体反例：`TMSB4X` 存在于 H1 Training，且有 `TMSB4X_P1_A|TMSB4X_P1_B` guide 标识，却不在 2026 列表；`ABCD1` 存在于 2026 列表，却不在 H1 三面板。完整的 25 个共同靶点为：ACLY、ADNP、AKT2、ANKZF1、BRPF1、HDAC8、HSBP1、MED13、MED15、MED25、MTA1、NFE2L1、PLAGL2、RNF2、SHPRH、SIN3B、SLIRP、SMARCA5、STAT6、TARBP2、TRAPPC6A、TWF2、ZFP62、ZNF32、ZNF714。

原始并集缺少的 28 个官方靶点为：ABCD1、CAPRIN2、EPHB2、HEATR6、KIF21B、MAPK7、MLKL、NICN1、PARP3、PBLD、PDK3、PHF11、PSMB9、RAB11FIP5、SEMA4F、SLC44A1、SNN、SOCS5、SPATA20、TAF4、TBC1D19、TDRD7、TESK2、TMEM104、TMEM79、TTBK2、TXNDC16、ZC2HC1A。未逐一核验这些靶点的网络覆盖，不宣称先验已经补齐它们。

### 2025 与 2026 面板的独立复核

针对“两届是否为相同 300 靶点”的疑问，重新从 Arc 官方 GCS 读取 [2025 Training CSV](https://storage.googleapis.com/arc-institute-virtual-cell-atlas/virtual-cell-challenge/2025/train/pert_counts_Training.csv)、[Validation CSV](https://storage.googleapis.com/arc-institute-virtual-cell-atlas/virtual-cell-challenge/2025/validation/pert_counts_Validation.csv)、[Test CSV](https://storage.googleapis.com/arc-institute-virtual-cell-atlas/virtual-cell-challenge/2025/test/pert_counts_Test.csv)，三者均与本地原始文件字节相同，分别含 150/50/100 个靶点。独立审计又确认对应三个 H5AD 的 `obs/target_gene`（排除 non-targeting）与各自 CSV 集合完全一致。

2026 侧直接读取原始 ZIP 中的 `pert_counts.csv`，与解压文件字节相同，SHA256 为 `f57edd7b912ebd718efc7ee9d0f334772513e7cc418d133ce525470e373b3276`，符合 [SOURCE.json](../data/raw/arc_vcc2026_controls/SOURCE.json)；ZIP manifest 指定 `vcc2026-val-1`。因此本结论对应这个已登记的官方 2026 快照，不声称重新下载了当前服务端 2026 ZIP。

两届各有 300 靶点，交集为 13+4+8=25，各自独有 275。`TMSB4X` 仅在 2025，`ABCD1` 仅在 2026。完整交集为：ACLY、ADNP、AKT2、ANKZF1、BRPF1、HDAC8、HSBP1、MED13、MED15、MED25、MTA1、NFE2L1、PLAGL2、RNF2、SHPRH、SIN3B、SLIRP、SMARCA5、STAT6、TARBP2、TRAPPC6A、TWF2、ZFP62、ZNF32、ZNF714。

文献能解释为什么背景之间的靶点集合不相同，但不能代替逐文件交集：Replogle 的 K562 genome-wide screen 包含表达基因、TF 与 common essentials，RPE1 面板主要来自 common essentials 及部分人工选择基因。Nadig 的跨细胞研究也显示 perturbation 响应并非完全跨背景一致；这些发现支持做背景条件化，而不是无条件平均所有来源。[Replogle 原始研究与 Library design](https://pmc.ncbi.nlm.nih.gov/articles/PMC9380471/)、[Nadig 2025 原始研究](https://www.nature.com/articles/s41588-025-02169-3)

## 哪类先验提供什么信息

以下优先级是针对 residual XGBoost 特征设计的判断，不是数据库之间已经验证的性能排名。

| 信息 | 对 EXP005 的候选用途 | 必须保留的限制 |
| --- | --- | --- |
| 有向、有符号 TF→gene regulon，例如 CollecTRI / DoRothEA | 直接边及方向、调控模式、共享上游 TF、少量有向路径特征；结合背景 NTC 推断的 regulon 活性代理 | 符号描述已报道调控关系，不保证任意背景 CRISPRi 后最终表达的方向或幅度；非 TF 靶点不一定有直接出边 |
| STRING 功能关联或物理网络 | 靶点与 readout 的连接强度、邻域相似性、邻居 NTC 表达、与已测靶点的相似性 | 默认 functional association 不是纯物理结合；普通边权不是激活符号、响应强度或因果效应 |
| Reactome / GO 模块 | 靶点和 readout 的共同功能、模块成员关系、NTC 模块表达或活性代理；帮助跨靶点共享统计量 | 同通路不意味着同方向变化，通路成员表本身不提供调控符号 |
| 固定蛋白/基因表示 | 给没有扰动标签的靶点提供连续功能特征，避免只有不可泛化的 target ID | 序列或功能相似不等于响应相似；应固定版本并核对缺失覆盖与表示来源 |

CollecTRI 的官方接口明确提供 TF、target、正/负调控权重与可用时的文献证据；它扩展了 DoRothEA。可优先调查其覆盖，但先验仍应是模型可以忽略的软特征。[CollecTRI 官方接口](https://decoupler.scverse.org/en/latest/api/generated/decoupler.op.collectri.html)

STRING 明确区分 functional 与 physical networks；前者涵盖共同生物过程，后者也可能表示同一复合体而非直接结合。不能给普通 combined score 人为加上因果符号。若使用新版另外提供的有向调控信息，必须明确具体字段和版本，不能将已有无方向网络重新解释为有符号图。[STRING 网络与证据说明](https://string-db.org/help/scores/)、[STRING 2025 原始论文](https://pmc.ncbi.nlm.nih.gov/articles/PMC11701646/)

Reactome 的核心是经整理的反应与通路；现有 GMT 的成员关系是对这些机制的简化，不足以决定 knockdown 的净效应。项目已经登记 STRING / Reactome / GO，优先核查并复用现有版本。[Reactome 数据模型说明](https://reactome.org/what-is-reactome)、[本地登记](datasets/data-inventory-audit-2026-09-18.md)

## 背景信息与因果解释

建议的建模表达是 `delta(c,t,g) = f(NTC背景特征, 靶点特征, readout特征, 靶点–readout关系)`。这是候选接口，不决定多输出策略或训练配置。NTC 可以提供均值、检测率、变异、模块状态、TF regulon 活性代理；cell-context ID 只表身份，无法告诉模型匿名 A/B/C 的生物学状态。对已见靶点，还应保留直接测得的跨背景 response 信息作为强比较基线，而不是用先验替换实测信号。

RNA 表达不等于 TF 蛋白活性；低表达或 dropout 也不足以把一条边硬删为无作用。regulon 活性估计是代理量。CellOracle 展示了结合 promoter/motif、可及染色质及表达来构造背景相关 GRN 的可能性，也给出相同 TF 表达不代表相同背景作用的例子，但其结果不是 VCC2026 分数保证。[CellOracle 原始研究](https://www.nature.com/articles/s41586-022-05688-9)

当前官方新背景输入是 NTC RNA，不能冒充拥有这些匿名背景的匹配 ATAC、ChIP 或蛋白活性测量。外部 promoter/染色质知识可以作为静态候选关系；是否在 A/B/C 活跃必须保留不确定性。NTC 共表达也不能自行证明调控方向。

## 有用的证据与尚未证明的内容

GEARS 通过 GO 关系与共表达图对未见靶点共享信息，在其论文的单背景任务得到改进。这支持先验值得检验，但论文单基因模型分别在各背景训练，不能证明相同设计能同时解决未见背景和未见靶点，也不能把其 top-DE 指标当本届六项官方评分。[GEARS 原始研究](https://www.nature.com/articles/s41587-023-01905-6)

另一方面，2025 年 Nature Methods 的专门比较发现，所评估的多种深度模型没有超过设计得当的简单基线；Arc 2025 总结也报告蛋白表示和 residual 建模被成功团队使用。这些证据同时支持“考虑生物特征”和“保留直接转移/简单回归比较”，没有推出越多先验越好。[简单基线基准](https://www.nature.com/articles/s41592-025-02772-6)、[Arc 2025 总结](https://arcinstitute.org/news/virtual-cell-challenge-2025-wrap-up)

## 后续需要检验的假设

实施前应把每折靶点分为训练已见和全局未见，并另报 readout 可测覆盖；纯留一背景总体分可能主要反映已见靶点迁移，掩盖冷靶点失败。只有确认需要检验先验增益时，再在相同数据、生成方式与官方评分口径下比较无先验、真实先验及合适的随机先验对照；同时保留零响应与简单跨背景直接 response 转移参照。先验构造与特征选择不能使用外折扰动标签，来源与版本应可追溯。

上述是区分效应来源所需的未来验证设计，不是本轮已获批准的额外 run 清单。零交集本身既不要求新增复杂图模型，也不能作为忽略 XGBoost 无先验基线的理由。
