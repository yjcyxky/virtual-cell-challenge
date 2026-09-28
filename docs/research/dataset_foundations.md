# 五背景数据基础：前瞻初始化依据

核查日期：2026-09-27。范围限定为 H1、K562 GWPS、RPE1、HepG2、Jurkat 五个扰动背景及本地 2026 官方输入。没有读取本项目训练结果、实验 PLAN、实验缓存或模型；本页不能作为任何候选方法在本项目有效的证据。

本次直接以只读方式读取七个原始 H5AD 的 `obs`、`var` 和矩阵形状，以及官方 CSV；没有读取 `X` 表达值、重算表达统计或改写原始数据。文献/发布方声明、当前本地元数据事实、待验证的研究设计分别列出。文件 SHA-256 引用已有 SOURCE 记录，本次没有重新扫描大文件哈希。旧 `data/sources.json` 的 `why`、`caveat` 含建模判断，不作证据。

## 来源与独立性

| 数据背景 | 来源 ID | 发布方/论文声明 | 本地原始输入（相对于仓库根目录） |
|---|---|---|---|
| H1 hESC | DS-H1 / SRC-ARC-H1 | 双 guide CRISPRi，10x Flex；300 靶覆盖不同响应强度。细胞类型和测量化学均不同于常用 3′ 公共数据。[Arc 数据生成说明](https://arcinstitute.org/news/behind-the-data-virtual-cell-challenge)、[官方数据发布](https://github.com/ArcInstitute/arc-virtual-cell-atlas/blob/main/virtual-cell-challenge/README.md) | `data/raw/arc_vcc2025_h1/adata_Training.h5ad`、`adata_Validation.h5ad`、`adata_Test.h5ad`；[SOURCE](../../data/raw/arc_vcc2025_h1/SOURCE.json) |
| K562 GWPS | DS-K562 / SRC-REPLOGLE-2022 | Replogle 等，Cell 2022；K562 全基因组规模 CRISPRi，转导后第 8 天；双 guide，10x 3′。[原论文](https://doi.org/10.1016/j.cell.2022.05.013)、[作者数据发布](https://plus.figshare.com/articles/dataset/20029387) | `data/raw/replogle2022/K562_gwps_raw_singlecell_01.h5ad`；[SOURCE](../../data/raw/replogle2022/SOURCE.json) |
| RPE1 | DS-RPE1 / SRC-REPLOGLE-2022 | 同一论文中的 RPE1 essential-scale 筛选，转导后第 7 天；CRISPRi 效应器与 K562 不同。[原论文](https://doi.org/10.1016/j.cell.2022.05.013)、[作者数据浏览器](https://gwps.wi.mit.edu/) | `data/raw/replogle2022/rpe1_raw_singlecell_01.h5ad`；同上 SOURCE |
| HepG2 | DS-HEPG2 / SRC-NADIG-2025 | Nadig 等，Nature Genetics 2025；与 Jurkat 平行进行，以 DepMap common-essential 为主的双 guide CRISPRi，转导后第 7 天、10x 3′ v3。[论文](https://doi.org/10.1038/s41588-025-02169-3)、[原始 GEO GSE264667](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE264667) | `data/raw/nadig2025/GSE264667_hepg2_raw_singlecell_01.h5ad`；[SOURCE](../../data/raw/nadig2025/SOURCE.json) |
| Jurkat | DS-JURKAT / SRC-NADIG-2025 | 与 HepG2 同一研究、同一发布批次，并非独立论文复现；各自有独立细胞背景。[GEO 设计](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE264667) | `data/raw/nadig2025/GSE264667_jurkat_raw_singlecell_01.h5ad`；同上 SOURCE |

这里是 **5 个背景、3 个来源研究组**，不是五项独立研究；Replogle 与 Nadig 还共享部分研究团队和实验技术体系。背景、研究、技术、时间点及靶点库存在混杂，五折 leave-one-context-out 的差异不能自动解释成纯细胞类型效应。生物独立重复的数量尚未核实；`batch`、`gem_group`、guide 对、NTC bag 或随机种子均不自动构成独立生物重复。

K562 essential day-6 版本、本地 bulk 文件及 scPerturb 处理副本没有纳入这五背景定义。它们与原研究的关系必须另行审计；同一细胞系、同一实验的不同处理版本不增加背景数。[Replogle 发布说明](https://plus.figshare.com/articles/dataset/20029387)

## 当前本地元数据事实（SRC-LOCAL-METADATA-20260927）

以下计数取实际观测行，不只数 categorical 的类别表。`non-targeting` 是七文件共同的 NTC 标签；所读靶点与 guide 分类编码没有缺失值。guide 数指联合 guide 对 ID，包含 NTC，不能当作单 guide 数或重复数。

| 背景 | 记录数 | NTC 记录 | 非 NTC 靶点 | 读数列 / 唯一原始 symbol | guide 对 ID（其中 NTC） | batch/GEM ID |
|---|---:|---:|---:|---:|---:|---:|
| H1 三文件合并后唯一记录标识 | 414,694 | 38,176 | 300 | 18,080 / 18,080 | 348（31） | 48 |
| K562 GWPS | 1,989,578 | 75,328 | 9,866 | 8,248 / 8,246 | 11,187（514） | 267 |
| RPE1 | 247,914 | 11,485 | 2,393 | 8,749 / 8,748 | 2,662（113） | 56 |
| HepG2 | 145,473 | 4,976 | 2,393 | 9,624 / 9,623 | 2,679（130） | 56 |
| Jurkat | 262,956 | 12,013 | 2,393 | 8,882 / 8,881 | 2,679（130） | 55 |

H1 的 Training / Validation / Test 分别为 221,273 / 98,927 / 170,846 行和 150 / 50 / 100 个非 NTC 靶。三个文件共享相同的 38,176 个 `(batch, barcode)` 标识；直接拼接有 491,046 行，重复 76,352 行。当前验证的是记录身份，重复表达值是否完全一致尚未验证。其他四文件各自的 `(gem_group, cell_barcode)` 标识无重复，不等于跨版本去重已经完成。

H1 使用 `obs/target_gene`、`guide_id`、`batch`；其他四背景使用 `obs/gene`、`sgID_AB`、`gem_group`，并有 `gene_id`、`transcript`、`gene_transcript`、`UMI_count`、`mitopercent` 等字段。其他四背景的读数索引是 Ensembl ID，symbol 位于 `var/gene_name`，分别有 2 / 1 / 1 / 1 个重复 symbol 列。必须锁定 ID/别名/重复列策略，不能仅将索引改成 symbol 后直接拼接。

非 NTC 靶点细胞数的中位数分别是 K562 178、RPE1 72、HepG2 45、Jurkat 83；最少分别为 2、2、2、1。H1 三子集的中位数分别是 1,045、1,090.5、1,141.5。上述是筛选前本地记录覆盖，不是有效样本数、质量评分或精度保证。作者发布的 Replogle `raw_singlecell_01` 已限定于平均表达高于 0.01 UMI/cell 的读数基因；名字中的 raw 不表示未筛选的全基因组矩阵。[作者发布](https://plus.figshare.com/articles/dataset/20029387)

### 靶点覆盖与测量覆盖分开管理

以下交集只比较原始 symbol 字符串，**尚未进行 HGNC 别名归一、Ensembl 对齐或表达阈值筛选**。不能将它们写成已冻结训练面板。

| 靶点交集 | H1 | K562 GWPS | RPE1 | HepG2 | Jurkat |
|---|---:|---:|---:|---:|---:|
| H1 | 300 | 280 | 99 | 99 | 99 |
| K562 GWPS | 280 | 9,866 | 2,390 | 2,390 | 2,390 |
| RPE1 | 99 | 2,390 | 2,393 | 2,392 | 2,392 |
| HepG2 | 99 | 2,390 | 2,392 | 2,393 | 2,392 |
| Jurkat | 99 | 2,390 | 2,392 | 2,392 | 2,393 |

五背景非 NTC 靶点交集为 **99**、并集为 **9,889**；读数 symbol 交集为 **6,123**。共同靶点面板便于配对比较，但会偏向被多来源共同选择的靶点，必须同时报告各背景可用全集，不能只保留交集后声称泛化到所有扰动。

本地 [2026 官方输入](../../data/raw/arc_vcc2026_controls/SOURCE.json) 的 `vcc2026-val-1` 面板有 18,533 个读数基因、300 个目标靶点；A/B/C 是 NTC 输入，没有扰动响应标签。2026 官方任务是仅以未扰动背景及靶点列表预测未见背景的 CRISPRi 响应；六背景分为三个验证背景和三个最终测试背景。[SRC-ARC-2026：官方任务说明](https://arcinstitute.org/news/virtual-cell-challenge-2026)

| 与当前官方面板的原始字符串交集 | H1 | K562 GWPS | RPE1 | HepG2 | Jurkat |
|---|---:|---:|---:|---:|---:|
| 300 目标靶点中已有监督标签 | 25 | 272 | 0 | 0 | 0 |
| 18,533 读数基因中本地有测量列 | 18,077 | 7,681 | 8,260 | 9,024 | 8,284 |

五背景靶点并集只字面覆盖官方 272 靶；其余 28 靶是**待别名审计的未见候选**，不能立即断言生物学上未被扰动。由覆盖推导出的设计含义是：K562 提供大部分直接目标监督，其他背景主要支持跨背景响应关系的学习；是否能迁移仍待实验。不能由“多 essential-gene 数据”推断已经覆盖本届靶点。缺读数列、实测零值及非显著响应分别管理，缺测不能补零作为负标签。

## 由覆盖推导的前瞻划分（候选设计，不是完成的实验）

| 研究问题 | 可实施划分 | 必须固定/报告的边界 |
|---|---|---|
| Context OOD：同一扰动能否迁移到未见背景 | 五背景逐一留出全部扰动标签，允许预声明的留出背景 NTC 适配；只在训练背景出现过的靶点上计算 context-only 分项 | 分别报告五背景、三研究组、共同 99 靶与每折可用全集；所有响应表示与超参拟合排除留出背景标签 |
| Perturbation OOD：靶点知识是否支持新扰动 | 全局按靶点或预先冻结的功能家族划分，留出靶点从全部背景监督中删除；在有其他训练扰动的背景评价 | 随机靶点划分与功能家族外推不能混称；图/模块/响应嵌入不得间接包含留出响应 |
| Joint OOD：未见背景且未见靶点 | 同时留出背景和一组全局靶点；比较无先验、真实先验、匹配随机先验/等维表示 | 必须人工构造全局靶点 holdout；自然 LOCO 的“未见靶点数”在各折极不均衡 |
| Study OOD：能否跨来源研究迁移 | 一次留出 Arc H1、Replogle 两背景、或 Nadig 两背景；留出 NTC 的许可单独声明 | 只有三个研究组，且实验体系有共享；该结果不能视为丰富独立研究验证，也不能剥离背景/化学/时间点混杂 |
| QC 是否有收益，收益来自哪里 | 正确性处理固定后，比较基础可用集、额外 QC、匹配随机删减或等覆盖重加权 | 配平细胞数、靶点/guide/批次覆盖；报告保留率、目标敲低及响应强度；不为对照保留已知错误 |
| NTC 表示与分布生成是否可靠 | 输入 NTC 与评分 NTC 按物理记录隔离；固定响应模型比较生成器，或固定生成器比较表示 | guide/GEM分层与批次漂移诊断；NTC 对 NTC 伪扰动检验；生成细胞与种子不增加生物独立性 |

自然 leave-one-context-out 中，留出背景靶点被其余四背景覆盖的数量为 H1 280、K562 2,572、RPE1/HepG2/Jurkat 各 2,393；对应自然未见数量为 20、7,294、0、0、0。因此不拆 context-only / joint 分项会把不同任务混进一个平均分。上述原始标签统计应在别名映射和覆盖阈值冻结后重新生成。

TRADE 原论文报告跨细胞类型效应具有部分一致性，也报告剂量变化可改变响应性质；这支持检验“共享响应 + 背景修正”及剂量/guide 分层，不能证明固定共享响应或统一删除弱响应就是正确策略。[SRC-NADIG-2025：论文摘要与正式 DOI](https://pubmed.ncbi.nlm.nih.gov/40259084/)

## 协议冻结前仍需完成的审核

1. **记录身份与处理版本**：核对 H1 重复记录表达；追溯各文件和 scPerturb/bulk 等副本关系。输入/评分 NTC 分离到物理记录，留出边界写入可重建 split。
2. **测量和靶点映射**：固定 Ensembl/HGNC 版本、重复 symbol 处理、目标别名、测量掩码；重算官方覆盖、背景×靶点覆盖及 panel 版本。6,123 原始交集只是核查值。
3. **实验单位**：将 guide、GEM/batch 映射到转导、培养、生物重复和建库单位；字段不足时明确 unknown，不能补造独立重复数。补充时间点、效应器、细胞选择和 assay 信息。
4. **质量与不确定性**：在正式预处理内只读计算计数深度、敲低、guide 一致性、NTC 批次漂移、响应信噪比及有效覆盖。阈值在训练边界拟合；保持真实低/高响应及细胞状态变化的可辨性。
5. **评估范围**：冻结真实可评分读数、scorer、baseline/anchors、NTC 池和统计单位。完整官方轴上的补全策略单独验证；局部可测轴的成绩不能等同完整官方评分。

初始化账本可引用本页的**数据可用性事实和限制**，以及上述文献的外部研究结论；候选方法在 VCC 2026 的本地有效性均为未验证。下一步应先生成可审核的映射/划分/覆盖对象，再进行配对归因实验。
