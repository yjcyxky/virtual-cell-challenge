# PDF 数据方案总结与本地缺口

核查日期：2026-09-30。结论：**PDF 中可确认、本地尚未具备且值得接入的新增扰动来源是 Xaira X-Atlas/Orion（HCT116、HEK293T）和原代 CD4+ T 全基因组 Perturb-seq。HeLa 的来源标注错误，不能列为 GSE264667 的下载缺口。** 以下是数据可用性与用途判断，不是新增数据已经改善模型的实验结论，也不变更当前训练队列。

依据：[原 PDF](../dataset_overview.pdf)、[原始来源核查](dataset_overview_source_review.md)、[本次机器可读库存及覆盖结果](../datasets/dataset-overview-audit-2026-09-30.json)。本次读取 PDF 全文和链接、当前登记/锁/清单、实际目录、七个核心 H5AD 的形状和观测靶点标签，以及发布方小型元数据。大矩阵只核对文件大小与已有 SOURCE，不重新计算完整哈希，也没有扫描表达值或下载新的大型表达矩阵。

## 1. PDF 提出的数据与划分

PDF 是一份数据选择方案，共五组来源，包含以下背景：

| 来源 | PDF 指定背景 | PDF 的扰动筛选 |
|---|---|---|
| Xaira X-Atlas/Orion | HCT116、HEK293T | 每靶至少 50 细胞，且 DE 数位于前 5% **或**属于比赛 300 靶 |
| VCC 2025 | H1 | 不作上述筛选 |
| Replogle 2022 GWPS | K562 | 每靶至少 50 细胞，且 DE 数至少 50 **或**属于比赛 300 靶 |
| GSE264667 | Jurkat、HepG2、HeLa（HeLa 归属有误） | 同 K562 |
| Primary CD4+ T | 只用 Donor 1 的 Rest、Stim8hr、Stim48hr | 同 Xaira |

非 H1 来源的 NTC 保留至 30,000 个；K562/GSE 项明确不足时不下采样。读出基因限制到比赛基因表。训练用除 H1 以外的来源；H1 取与比赛重叠的 25 靶及另外随机 75 靶，共 100 靶，每靶抽 400 个细胞作测试。

PDF 没有给出筛选后的真实规模、每靶支持量、随机种子、DE 配置、NTC 划池或固定的 100 靶列表。因此它可以指导候选数据选择，但尚不是可复现的完整实验协议。

## 2. 本地实际覆盖

| PDF 数据项 | 当前本地情况 | 判定 |
|---|---|---|
| Xaira HCT116、HEK293T | 当前 `data/sources.json`、lock、MANIFEST 未登记该来源，原始目录中无对应 Orion 表达数据 | **真实缺口，优先补充候选** |
| H1 | `data/raw/arc_vcc2025_h1/` 的 Training/Validation/Test 三个 H5AD 均存在，大小与 SOURCE 一致；已有训练路线使用 | 已有，不需重复下载 |
| K562 GWPS | `data/raw/replogle2022/K562_gwps_raw_singlecell_01.h5ad` 存在，1,989,578 × 8,248；已有训练路线使用 | 已有，不需重复下载 |
| HepG2、Jurkat | `data/raw/nadig2025/` 两个 raw single-cell H5AD 存在；分别 145,473 × 9,624 和 262,956 × 8,882；已有训练路线使用 | 已有，不需重复下载 |
| HeLa / GSE264667 | 该 GEO Series 的全部 224 个 GSM 仅列 HepG2/Jurkat，发布的 processed H5AD 也仅这两背景 | **PDF 来源错误，无法据此确定可下载的 HeLa 数据** |
| CD4 四供体×三状态数据 | 未登记该研究，原始数据目录无其 `D*_*.assigned_guide.h5ad` 或 GWCD4i 数据 | **真实缺口，补充训练与评估候选** |

H1 三文件共 491,046 行，包含重复 NTC；不能把该总行数当作独立细胞数。身份去重见[五背景基础](dataset_foundations.md)。本次重读实际靶标签确认：官方 300 靶在 H1 中有 25 个、K562 中有 272 个，RPE1/HepG2/Jurkat 中均为 0，五背景并集仍为 272 个。这是**原始 symbol 字面匹配和观测标签覆盖**，不是 QC 后有效监督量。后面新来源的增量也使用相同口径。

本地覆盖比 PDF 更广：RPE1、K562 essential、Jiang、scPerturb 和 Tahoe 子集已在库。尤其 **scBaseCount 人类子集现在已存在**：1,808 个 H5AD，表达文件约 200 GB；9 月 18 日的“缺失”结论已过时。其选择清单按未扰动类元数据筛选，HeLa/HEK293 或 CD4 细胞名称出现并不意味着拥有 PDF 所指研究的 CRISPRi 配对监督。普通细胞图谱不能填补这里的扰动来源缺口。[当前选择清单](../../data/selections/scbasecount_2026_01_12_human.json)

## 3. 优先候选及本次实算覆盖

### Xaira：优先补充跨细胞系响应监督

官方公开两背景的单细胞 raw counts、guide/batch 元数据和 NTC 构件。它新增两个细胞系和一个研究来源；相比继续增加 K562 细胞，能够直接检验模型是否受限于同靶跨背景监督不足。这个判断是数据价值假设，需通过增量数据比较验证。[作者数据卡](https://huggingface.co/datasets/Xaira-Therapeutics/X-Atlas-Orion)

本次读取 Figshare v3 的 3.65 MB guide 库和固定 HF revision 的基因目录，实算得到：

| 覆盖层次 | 结果 | 解释边界 |
|---|---:|---|
| 官方靶点在设计库中 | **300/300** | 当前五背景缺少标签的 28 靶全部在设计库中 |
| 官方读出 symbol 在 HF 基因目录中 | **18,106/18,533（97.70%）** | 目录覆盖；尚不是逐背景 QC 后的实际测量覆盖 |
| 实际逐背景每靶细胞数、NTC 数、KD 与有效响应 | 未核查 | 需读取表达/obs 后确认，不能把设计覆盖写成有效监督覆盖 |

427 个读出名字未字面匹配，包含旧符号；应按稳定 Ensembl ID/HGNC 处理，不能直接认定这 427 基因都没测到。目录还有重复 symbol，映射须显式处理。计算所用固定 URL、SHA-256 和缺失集合保存在[审计 JSON](../datasets/dataset-overview-audit-2026-09-30.json)。

用途建议：优先接入 HCT116 的匹配子集，HEK293T 作为后续来源或预先锁定的外部背景；也可将整个 Xaira 研究作为研究留出。若两个背景都参与训练/选模，后续只能通过新的严格划分评估泛化。guide 与敲低强度应保留为质量/条件信息，不能假设弱敲低响应可直接线性放大。

成本：两个 Figshare H5AD 共 **559.52 GB**，HCT116 209.35 GB、HEK293T 350.16 GB；可考虑 HF 分片/streaming，实际传输量取决于分片布局。许可 **CC BY-NC-SA 4.0**。先登记版本和实际选择范围，避免为拿 300 靶而无条件下载全量。[作者 Figshare 发布及文件清单](https://api.figshare.com/v2/articles/29190726)

### CD4：补原代、供体与刺激状态泛化

PDF 的 `PMC3312336` 是[2012 年 CD4 综述](https://pmc.ncbi.nlm.nih.gov/articles/PMC3312336/)。与所述文件匹配的正确来源是 **Zhu et al., Cell 2026 / GSE314342**，数据名 **Primary Human CD4+ T Cell Perturb-seq**。四供体、Rest/Stim8hr/Stim48hr 三状态，提供单细胞 UMI counts、NTC、guide/lane/QC 注释；CZI 页面标示 MIT。[正式数据页](https://virtualcellmodels.cziscience.com/dataset/genome-scale-tcell-perturb-seq)、[GEO](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE314342)

本次读取作者固定 Git commit `aa5c84a973c0e1a090b0072dc5b080bf7fbbed38` 的小型补充表，重新计算：

| 覆盖层次 | 官方靶点覆盖 | 能补五背景现缺的 28 靶 |
|---|---:|---:|
| 校正后的 guide library | **297/300** | 27/28，缺 EPHB2 |
| 作者发布的 DE 统计行，三个状态取并集 | **291/300** | **24/28** |
| DE 统计行：Rest / Stim8hr / Stim48hr | **283 / 287 / 284** | 分状态明细见 JSON |

“有 DE 统计行”表示作者发布了该靶的响应估计，**不表示响应显著**，也不保证 D1 单独有足够细胞。上述 DE 表汇总多个供体；不能用于宣称 D1 覆盖，或用于选择一个随后声称完全未见的供体/状态。库覆盖与 DE 并集不一致，已经说明“设计过该靶”和“有可用估计”必须分开。[固定 guide 元数据](https://raw.githubusercontent.com/emdann/GWT_perturbseq_analysis_2025/aa5c84a973c0e1a090b0072dc5b080bf7fbbed38/metadata/suppl_tables/sgrna_library_metadata.suppl_table.csv)、[固定 DE 元数据](https://raw.githubusercontent.com/emdann/GWT_perturbseq_analysis_2025/aa5c84a973c0e1a090b0072dc5b080bf7fbbed38/metadata/suppl_tables/DE_stats.suppl_table.csv)

样本表有 12 行、4 个 donor，建库字段均为 `GEMX_flex_v1`，因此图谱中“是否为 Flex”的疑点获得了直接元数据支持；同属 Flex 不等于与 VCC 轴、深度、时间点相同。[固定样本元数据](https://raw.githubusercontent.com/emdann/GWT_perturbseq_analysis_2025/aa5c84a973c0e1a090b0072dc5b080bf7fbbed38/metadata/suppl_tables/sample_metadata.suppl_table.csv)

用途建议：相比照抄“仅 D1”，更有信息量的选择是预先保留一个供体或一种状态用于评估。供体留出检验同类细胞的个体泛化，状态留出检验刺激依赖，整套 CD4 留出检验从细胞系到原代细胞的跨研究迁移；分别报告，不能混称 context OOD。四供体×三状态是 12 个任务上下文，仍只有四供体、一项研究。训练均值响应可先使用保留 donor/guide 维度的 pseudobulk；验证 counts 稀疏性、协方差、异质性与生成分布必须使用 cell-level counts。

## 4. PDF 中哪些规则应调整

1. **DE 筛选作为待比较的训练配置。** DE 数受 n、深度、对照和阈值影响，只选前 5% 会偏向强响应。应保留全部结构合格任务的覆盖统计，报告强弱响应分层；正式留出评估遵守当前 DAG 的面板，不能用留出效应或 anchor 表现挑“好评”的靶。
2. **50 细胞准入与 400 细胞评估分开。** n≥50 不保证可无放回抽 400 个真实细胞；重复抽样不增加独立支持量。需要记录每个背景×靶点×guide 的实际 n，并按冻结协议处理低 n。
3. **固定 NTC 与测量分母。** NTC 上限 30,000 是成本选择，不是生物约束；输入/评分 NTC 分离，保留 batch/guide。官方基因子轴必须携带缺测掩码，归一化分母遵守当前协议，不能补零监督。
4. **H1 只能作为当前开发评估。** 本仓库已反复使用 H1 进行方法选择，PDF 的 25+75 即便重新采样也不能恢复盲测独立性。保留新背景/新研究作为外层确认更有价值，且必须在看响应结果前固定选择与隔离边界。

本轮实际参考 KnowGraph：`DATA-XATLAS`、`DATA-CD4` 及一跳 `EXP-LOCO`、`WL-MODALITIES`、`MOD-ATLAS`、`EXP-AGENT`。来源与广覆盖机会被官方元数据支持；固定源权重、强响应筛选或自动迁移收益均未获本地实验验证。图谱的全背景交集、baseline 接近某数值等旧建议不覆盖当前官方轴/测量掩码及冻结面板协议。

## 5. 接入顺序建议

**第一优先：Xaira。** 从设计上可补齐当前缺失的 28 靶，并增加同靶的跨细胞系监督；先核对实际细胞支持、NTC、靶点与读出映射，再决定 HCT116/HEK293T 的训练和外层评估分工。

**第二优先：CD4。** 新增原代与刺激/供体维度；优先利用明确分层的元数据与 counts，保留真正未参与选择的供体或状态。297 个设计靶、291 个具有汇总估计的靶只表示接入潜力；分供体实际可用监督仍需核验。

**不列入新增下载：H1、K562、HepG2、Jurkat；暂不列 HeLa。** 前四者已经在库，HeLa 必须先有正确 accession 和可用表达/扰动元数据。新增来源的有效性仍须用登记的数据增量比较验证；每轮训练完成后依照 AGENTS 的新流程，先做预测统计诊断，再评分与决定是否提交 leaderboard。
