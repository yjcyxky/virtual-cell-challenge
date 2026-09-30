# dataset_overview.pdf 的原始来源核查

核查日期：2026-09-30。本文核验 PDF 的来源身份、公开数据形态与训练/评估用途；本地文件是否已经存在由[综合库存与缺口分析](dataset_overview_gap_analysis.md)另行核对。未下载大型表达矩阵，未启动训练，不代表新增数据的本地方法效果或已完成接入。

参考本地[五背景事实](dataset_foundations.md)以及 KnowGraph 的 `DATA-XATLAS`、`DATA-CD4` 与直接相邻的跨背景评估、敲低效率检查项。图谱对源权重、筛选及迁移收益的建议仍是待检验假设，不能成为采用结论。

## PDF 的内容与来源纠错

PDF 列出五组来源：Xaira Orion 的 HCT116/HEK293T、VCC 2025 H1、Replogle 2022 K562 GWPS、GSE264667 的 Jurkat/HepG2/HeLa，以及原代 CD4+ T。它提出除 H1 外用于训练；H1 选与本届重叠的 25 靶加随机 75 靶作为测试，每靶抽 400 细胞。非 H1 扰动须至少 50 细胞，再满足 DE 数阈值或属于比赛 300 靶之一；Xaira/CD4 用 DE 数前 5%，K562/Jurkat/HepG2/HeLa 用 DE 数至少 50。NTC 最多保留 30,000 细胞，读出轴限制到比赛基因表。以上是 **PDF 所提方案**，不是作者数据默认处理，也不是已核实可复现的本地划分。

有两处明确的来源错误：

1. **GSE264667 不含 HeLa。** GEO 的研究设计仅写 Jurkat 与 HepG2；本次读取全部 224 个 GSM 的 SOFT 元数据，`cell line` 只有这两种，全文没有 HeLa。Series 的 processed H5AD 也仅列这两背景。因此不能把 PDF 的 HeLa 项当作已确认、可直接下载的新增训练背景。这里不否认其他研究存在 HeLa 数据，只否定该 accession 的归属。[GEO Series](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE264667)、[全体样本 SOFT](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE264667&targ=gsm&form=text&view=full)
2. **PMC3312336 不是 CD4 Perturb-seq 的来源。** 该链接是 2012 年的 CD4 功能综述。与 PDF 的四供体及 Rest/Stim8hr/Stim48hr 文件命名匹配的是 Zhu、Dann 等的 Cell 2026 研究、GEO **GSE314342** 与 CZI Virtual Cells Platform 的 **Primary Human CD4+ T Cell Perturb-seq**。[错误链接的实际文章](https://pmc.ncbi.nlm.nih.gov/articles/PMC3312336/)、[Cell 论文记录](https://pubmed.ncbi.nlm.nih.gov/42664972/)、[GSE314342](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE314342)

## 候选一：X-Atlas/Orion

**来源与数据形态。** Xaira 作者发布的两个人细胞系 HCT116、HEK293T 全基因组 CRISPRi 筛选，论文/官方卡描述约 800 万细胞、18,903 个设计靶基因，中位深度约 16,000 UMI/cell。这是两背景、同一研究来源，不是数百万独立生物重复。官方 HF 提供按背景分片的 Parquet，`gene_expression` 明确为非零基因的 raw counts，`gene_token_id` 经独立 metadata 映射 Ensembl/symbol；支持 streaming。不要将论文总规模当作拟选训练集 QC 后的细胞数。[作者 HF 数据卡](https://huggingface.co/datasets/Xaira-Therapeutics/X-Atlas-Orion)

**对照、质量与强度。** Figshare 的 H5AD 为经过双 guide 配对筛选的细胞，带 `sample`、`guide_target`、`gene_target`、UMI、线粒体比例及筛选字段。小型 guide 库有 `Non-Targeting` 构件；实际每批 NTC 细胞支持量仍需从 obs 核验。论文报告中位 on-target KD 为 HCT116 75.4%、HEK293T 51.5%，并发现 sgRNA 丰度能区分 KD 强度；这是推荐保留 guide/剂量元数据的依据，不是支持对弱 KD 直接线性放大的证据。[作者 Figshare 发布](https://doi.org/10.25452/figshare.plus.29190726.v3)、[原论文](https://doi.org/10.1101/2025.06.11.659105)

**成本与获取。** 2026-09-30 查询 Figshare API 得到版本 3（2025-11-12）。两个 H5AD 合计 **559.52 GB，约 521.09 GiB**；不能把网页用二进制换算的 521.09 与 GB/GiB 混用。Parquet streaming 可降低本地落盘需求，不保证服务端自动只传所需靶点。[Figshare API](https://api.figshare.com/v2/articles/29190726)

| 文件 | 文件 ID | 字节数 | 用途 |
|---|---:|---:|---|
| `HCT116_filtered_dual_guide_cells.h5ad` | 55021257 | 209,354,246,272 | HCT116 细胞 counts 与元数据 |
| `HEK293T_filtered_dual_guide_cells.h5ad` | 55074802 | 350,164,035,901 | HEK293T 细胞 counts 与元数据 |
| `guide_library.csv` | 57368587 | 3,648,784 | 靶点/guide/Ensembl 映射与设计覆盖 |
| `HCT116_filtered_guide_calls_per_cell.csv.gz` | 59490731 | 69,606,079 | 细胞 guide calls |
| `HEK293T_filtered_guide_calls_per_cell.csv.gz` | 59490734 | 90,886,568 | 细胞 guide calls |
| HF `metadata/gene_metadata.parquet` | HF revision 见下 | 881,915 | 稀疏 gene token 到基因轴的映射 |

许可证为 **CC BY-NC-SA 4.0**，无需申请受控队列即可访问上述公开文件；具体用途须符合非商业、署名及相同方式共享条件。[作者发布的许可](https://figshare.com/articles/dataset/29190726)

### 本次实算的设计库覆盖

只读取 3.65 MB 的[官方 guide 库](https://ndownloader.figshare.com/files/57368587)，用 `target_gene_name` 原始字符串与本地 `data/raw/arc_vcc2026_controls/pert_counts.csv:target_gene` 做集合交集，未作别名修正，未读取表达值：

- 20,890 行 guide pair 记录，18,331 个唯一标签，其中包含 `Non-Targeting`；NTC 构件行数为 1,026。
- **官方 300 靶的设计库覆盖为 300/300**，缺失集合为空。
- 这只能证明库里有这些靶的设计，不能证明 HCT116 与 HEK293T 各自都有足够通过 QC 的细胞、有效敲低或可用下游响应。也不能把 18,330 个非 NTC 唯一 symbol 与论文 18,903 的原设计基因数直接视为同一统计口径；版本、注释和设计单位仍需接入时对齐。

内容指纹与版本：

| 项目 | 固定引用/内容指纹 |
|---|---|
| Guide CSV SHA-256 | `1841320684c72f4a6c5f51a98ebaab0054711367a9ee81cc2167dd3fbbd6381d` |
| HF repository revision | `53a5bc98d49247bcf967500292575c3d3602de31` |
| Gene metadata SHA-256 | `33900a0dbaeafb84601c7c8ba7c6642c440774fc3f49613dab11cf1f2bf0fdbf` |
| 固定 gene metadata URL | [revision 下的 Parquet](https://huggingface.co/datasets/Xaira-Therapeutics/X-Atlas-Orion/resolve/53a5bc98d49247bcf967500292575c3d3602de31/metadata/gene_metadata.parquet) |

**用途判断（推断）。** 在现有五背景之外，两种新细胞背景和全基因组设计库能直接检验“目标响应目前过度依赖 K562”是否属于监督覆盖限制；也可留一整个细胞系或整个 Xaira 研究做外部评估。HCT116 可作为更高 KD 的第一接入背景，HEK293T 可专门检验强度迁移，但先后顺序应由官方靶的实际支持量和分布诊断决定。两者同时用于训练后，不能又将其随机细胞留出称为 context OOD。

## 候选二：原代 CD4+ T 全基因组 Perturb-seq

**正确身份。** Zhu et al., *Cell*, DOI **10.1016/j.cell.2026.08.002**（2026-08-28）；预印本 DOI **10.64898/2025.12.23.696273**。研究在四个供体的原代人 CD4+ T 细胞中进行 CRISPRi，约 2,200 万细胞，覆盖静息、再刺激 8 小时和 48 小时三个状态。GEO 为 **GSE314342**，2026-02-06 公开。[论文记录](https://pubmed.ncbi.nlm.nih.gov/42664972/)、[GEO 研究设计](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE314342)

**可用层次。** 作者公开每个 donor×condition 的 `D*_*.assigned_guide.h5ad`；`.X` 是单细胞 UMI counts，带 NTC/targeting guide 类型、guide ID、lane、QC 与靶点注释。另有按 guide×donor×condition 聚合的 raw-count pseudobulk（18,129 读出基因），及 DESeq2 DE 汇总（33,983 扰动×条件行、10,282 读出基因）。细胞级 target 注释与后续校正注释需要对齐；DE 汇总不能反推出单细胞分布。[作者数据结构说明](https://raw.githubusercontent.com/emdann/GWT_perturbseq_analysis_2025/master/metadata/data_sharing_readme.md)

**平台与访问。** CZI 作者数据卡标示 MIT、公开 CLI/S3 获取；按供体和状态分 12 个 cell-level 文件。GEO 原始样本说明提供细胞表达与 CRISPR guide UMI 的 H5 counts，以及 Cell Ranger 9.0.1/2024-A probe set 的处理信息，可用于追溯探针测量轴。不要因为同属探针测量就假定与 VCC 的面板/化学/深度完全一致。[CZI 官方获取页](https://virtualcellmodels.cziscience.com/dataset/genome-scale-tcell-perturb-seq)、[GEO 示例样本及计数文件](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSM9393898)

**用途判断（推断）。** 这是新增原代背景、供体差异和状态变化监督的候选；适合测试背景条件项是否学到随状态变化的下游响应，而不只是共享的增殖/应激方向。可设计供体留出、刺激状态留出和整研究留出，三者回答不同问题。四供体×三状态可产生 12 个任务上下文，但仍是 **4 个供体、1 个来源研究、1 类细胞的3个状态**；不能称为12项独立研究。PDF 只取 D1 会放弃验证跨供体稳健性的机会。

若为了低成本先研究均值响应，可先用保留 donor/guide 维度的 pseudobulk；要评估稀疏性、细胞方差、协方差和分布生成，则必须拿 cell-level counts。不能用全供体或全状态的 DE 文件筛选一个随后声称完全留出的 donor/condition；这会使用留出响应标签。官方靶与读出轴的最终有效覆盖仍需按同一映射和 QC 规则实算。

## 已有来源的身份边界与筛选方案的可用性

H1 对应 Arc VCC 2025，K562 GWPS 对应 Replogle 2022；GSE264667 的 Jurkat/HepG2 对应 Nadig 2025。它们与当前五背景来源身份相同，重新获取别的 processed 版本通常不增加独立背景；PDF 没有列出本地基础中已有的 RPE1。数据形态或版本差异仍可能需要审计，不能只因文件名不同就当新研究。[Arc 数据发布](https://github.com/ArcInstitute/arc-virtual-cell-atlas/tree/main/virtual-cell-challenge)、[Replogle 作者发布](https://doi.org/10.25452/figshare.plus.20029387.v1)、[GSE264667](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE264667)

PDF 的筛选可以作为一个待比较的训练配置，不能直接作为通用有效数据定义：

- DE 数依赖细胞数、深度、对照池、scorer 版本与阈值；“前 5%”和“至少50个DE”不是跨研究统一的生物效应门槛。只选强响应会改变训练与评估难度分布，应同时保留未经效应筛选的覆盖统计，并将强度分层作为诊断。
- “属于官方300靶”仍必须满足 `n≥50`；数据接入前不能据设计库300/300声称筛完后也完整覆盖。
- 基因轴可限制到官方子轴，但必须保留测量掩码和源矩阵归一化分母。源中缺列不是实测零；只剩交集后不能称为完成18,533基因监督。
- 每靶抽400细胞要明确不足400时的处理；重复抽样不会制造独立真值细胞。NTC应按批次和输入/评分用途隔离，30,000上限不等于实际有效支持量。
- H1的25+75划分需要固定靶列表、seed、抽样文件与NTC物理记录。如果已经用于模型选择，只能称开发评估，不能重新命名为未见最终测试。

以上用途与筛选判断是基于数据结构和本地协议的研究建议；需要独立、登记的增量数据比较才能判断其是否改善 VCC 泛化。
