# dataset_overview_v3.pdf 与当前仓库的数据缺口

核查日期：2026-10-02（美东）。**按 PDF 的 11 个数据项计，9 项已有对应的单细胞表达来源；GSE132080 尚未登记或落盘；Primary CD4+ T 已登记但获取未完成，当前没有完整的单细胞文件。** PDF 的最终筛选产物与原始来源分开判断，不能因缺少同名 `*_filtered.h5ad` 就认定来源缺失。

依据：[PDF](../dataset_overview_v3.pdf)、[来源登记](../../data/sources.json)、[文件锁](../../data/registry.lock.json)、[清单](../../data/MANIFEST.tsv)、实际目录及已有获取/索引报告。PDF SHA-256：`3fdb9252dcdfb116d374b7f9f9b141dd75d3ddc26059486ee977dbeed748e22a`。本次核对文件存在性、锁定大小、SOURCE 与 H5AD 形状，复读 Xaira 的已完成扫描结果；**未重算全部大文件哈希、未扫描全部表达值、未下载大型矩阵或修改训练队列**。这里的“来源齐备”指作者发布的表达矩阵/元数据，不包括原始 FASTQ，也不等于已纳入正式训练。

## 逐项核对

| PDF 数据项 | 当前实际文件或目录 | 本地状态 |
|---|---|---|
| VCC 2025 H1 | `data/raw/arc_vcc2025_h1/`，Training/Validation/Test 为 221,273 / 98,927 / 170,846 × 18,080 | **已有**，该来源登记的 6/6 文件大小匹配，有 SOURCE |
| HCT116 | `data/raw/xaira_orion/data/HCT116_Batch*.parquet`，109 个表达分片 | **已有**，已完成获取与索引 |
| HEK293T | `data/raw/xaira_orion/data/HEK293T_Batch*.parquet`，223 个表达分片 | **已有**，已完成获取与索引 |
| K562 GWPS | `data/raw/replogle2022/K562_gwps_raw_singlecell_01.h5ad`，1,989,578 × 8,248 | **已有** |
| K562 essential | `data/raw/replogle2022/K562_essential_raw_singlecell_01.h5ad`，310,385 × 8,563 | **已有**，不是 GWPS 的同名替代品 |
| Jurkat | `data/raw/nadig2025/GSE264667_jurkat_raw_singlecell_01.h5ad`，262,956 × 8,882 | **已有** |
| HepG2 | `data/raw/nadig2025/GSE264667_hepg2_raw_singlecell_01.h5ad`，145,473 × 9,624 | **已有** |
| RPE1 essential | `data/raw/replogle2022/rpe1_raw_singlecell_01.h5ad`，247,914 × 8,749 | **已有**原研究来源，未声称与 PDF 的再处理版本逐字节一致 |
| Adamson UPR | `data/raw/scperturb/AdamsonWeissman2016_GSM2406681_10X010.h5ad`，65,337 × 32,738 | **已有**对应 UPR 实验 |
| GSE132080：Jost 滴定 sgRNA | 来源登记、锁、清单、选择清单均无对应记录；文件名扫描无 GSE132080/Jost | **完全缺失** |
| Primary CD4+ T / GWCD4i | `data/raw/zhu2026_cd4/` | **已登记、部分落盘、未完成**；12 个单细胞文件均未完整取得 |

Replogle 的 6/6、Nadig 的 2/2、scPerturb 的 54/54 登记文件均在盘且大小匹配，各有 SOURCE。表内 H5AD 形状直接从本地文件只读核对；尺寸差异反映源文件与 PDF 筛选后的测量轴/细胞集合不同，不能据此推断数据损坏。

## 两项实际缺口

### GSE132080：尚未获取的独立实验

这是 Jost 等的 *Titrating gene expression using libraries of systematically attenuated CRISPR guide RNAs*，DOI `10.1038/s41587-019-0387-5`。GEO 明确对应中间活性 sgRNA 的 Perturb-seq；PDF 所述 25 靶、不同 guide 活性档位属于这项 K562 剂量实验。它增加的是 **guide 活性/敲低剂量维度**，不增加新细胞背景。[原论文](https://pmc.ncbi.nlm.nih.gov/articles/PMC7065968/)、[GEO 原始记录](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE132080)

官方 Series 提供以下五件套，足以建立表达矩阵、细胞到 guide 的映射及活性注释；本次仅核实列表，没有下载矩阵：[GEO 文件清单](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE132080)

- `GSE132080_10X_matrix.mtx.gz`
- `GSE132080_10X_genes.tsv.gz`
- `GSE132080_10X_barcodes.tsv.gz`
- `GSE132080_cell_identities.csv.gz`
- `GSE132080_sgRNA_barcode_sequences_and_phenotypes.csv.gz`

当前登记的 scPerturb Zenodo 版本 `13350497` 不含 Jost/GSE132080 文件；拥有 scPerturb 全套不代表已覆盖该来源。不能将其按基因合并后的标签直接解释为相同强度的完全敲低。[scPerturb 发布方文件清单](https://zenodo.org/records/13350497)

### Primary CD4+ T：已有登记，表达文件尚未完整

当前登记来源是 `zhu2026_cd4`，对应 Zhu、Dann 等的 **GSE314342 / Primary Human CD4+ T Cell Perturb-seq**，覆盖四供体和 Rest、Stim8hr、Stim48hr 三状态。作者现已公开 12 个 `D*_*.assigned_guide.h5ad`，内含原始 UMI counts、guide 身份和 QC 注释。因此 PDF 的“GEO 未给 guide 分配，需自行 calling”只适用于其选用的 GEO 入口，不能推广为作者没有发布带注释的单细胞文件。[GEO](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE314342)、[作者数据结构说明](https://raw.githubusercontent.com/emdann/GWT_perturbseq_analysis_2025/master/metadata/data_sharing_readme.md)

| 锁定的 57 个对象 | 当前盘点 |
|---|---|
| 文件大小匹配 | 41 个，共 177,343,982 bytes；为元数据、补充表和说明材料 |
| 文件存在但短于锁定大小 | 8 个，共 112,915,021,824 bytes；4 个 DE/pseudobulk 文件和 4 个单细胞文件 |
| 文件完全不存在 | 8 个，均为单细胞文件 |
| 完整单细胞文件 | **0/12** |
| 已落盘字节 / 锁定总字节 | 113,092,365,806 / 1,843,656,060,436，约 **6.13%**；该比例不是完整可用数据比例 |

四个未下载完整的单细胞文件是 `D4_Rest`、`D4_Stim8hr`、`D2_Stim48hr`、`D3_Stim48hr`（均以 `.assigned_guide.h5ad` 结尾）。八个完全缺失的是 `D1_Rest`、`D1_Stim8hr`、`D1_Stim48hr`、`D2_Rest`、`D2_Stim8hr`、`D3_Rest`、`D3_Stim8hr`、`D4_Stim48hr`。

[获取状态](../../data/assessments/new-crispri-20260930/zhu2026_cd4/acquisition.json)停留在 `waiting_for_download`，最后记录于 2026-10-01 02:35 UTC；无 SOURCE、无完成的索引报告，核查时无对应 fetch/finish 进程。准确说法是**已登记且获取未完成、当前未运行**，不是“正在下载”或“已经可训练”。实际落盘字节采用本次盘点，未沿用旧状态中的进度数字。已有小型 DE 表也不能替代缺失的单细胞分布。

## 容易误判为缺失的三处

**Xaira 两背景已经齐备。** 338/338 个锁定对象全部在盘，合计 126,424,871,412 bytes；已有 [SOURCE](../../data/raw/xaira_orion/SOURCE.json)、完成的[获取记录](../../data/assessments/new-crispri-20260930/xaira_orion/acquisition.json)和[索引报告](../../data/assessments/new-crispri-20260930/xaira_orion/index/report.json)。原始扫描记录为 338 files、0 issue files，索引于 2026-10-01 00:01 UTC 完成。本库采用作者 HF 的 raw-count Parquet 发布，未重复获取两个大型 Figshare H5AD；这是存储格式选择，不能列为数据来源缺口。[作者 HF 说明](https://huggingface.co/datasets/Xaira-Therapeutics/X-Atlas-Orion)

| 既有索引结果 | HCT116 | HEK293T |
|---|---:|---:|
| 单细胞数 | 3,409,169 | 4,534,299 |
| NTC 数 | 165,777 | 218,838 |
| 实际观察到的官方靶点 | 300/300 | 300/300 |
| 至少 50 个来源支持细胞的官方靶点 | 263/300 | 276/300 |
| 映射后的实测官方读出基因 | 18,401/18,533 | 18,401/18,533 |

以上是复读已完成索引的结果，不是本次重新计算。`source_supported` 指来源质量/guide 分配通过且原生与映射轴 counts 为正；不证明有效 KD、DE 门槛通过或已形成训练划分。PDF 的“Xaira 未下载”和[旧版缺口报告](dataset_overview_gap_analysis.md)中的同类判断属于较早快照，已被后续获取结果更新。

**Adamson UPR 已有正确实验。** PDF 对应 `GSM2406681 / 10X010`，本库该文件确实存在。旁边的 `GSM2406675 / 10X001` 是 Pilot TF experiment，`GSM2406677 / 10X005` 是 epistasis experiment；三者不能混认。本次“已有”依据是 10X010 本身，而非作者名称相同。[GEO Series 的样本对应](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE90546)、[scPerturb 的转换源码](https://raw.githubusercontent.com/sanderlab/scPerturb/master/dataset_processing/scripts/AdamsonWeissman2016.py)

**K562 essential 与 RPE1 已有原研究文件。** 作者区分 day 8 的 K562 GWPS、day 6 的 K562 essential、day 7 的 RPE1 essential；K562 essential 是独立实验但仍为同一细胞系、同一来源研究。PDF 使用的再处理版本或筛选版本仍需单独复现，不能当新增研究来源。[Replogle 作者发布说明](https://doi.org/10.25452/figshare.plus.20029387)

## PDF 筛选产物的单独缺口

本次全仓文件名盘点未发现 PDF 七个派生文件：`k562gwps_filtered.h5ad`、`k562ess_filtered.h5ad`、`jurkat_filtered.h5ad`、`hepg2_filtered.h5ad`、`rpe1_filtered.h5ad`、`adamson_filtered.h5ad`、`gse132080_filtered.h5ad`；也未发现其 `04_filter`、`06_schema` 或 `dataset_schema` 配套产物。仓内 `NormanWeissman2019_filtered.h5ad` 属于另一研究。新来源的 NAS 链接目录另按锁逐文件核对，未因常规扫描不跟随链接而漏判。

因此，“复现 PDF 的最终筛选包”仍缺这七份派生数据及明确处理实现；其中六份已有对应上游表达来源，只有 GSE132080 连上游来源也缺。PDF 的 DE 选择、读出轴裁剪、对照下采样属于特定处理方案，不是所有来源的默认可用性定义。筛选后覆盖不能用原始库存覆盖代替，原始表达文件也不应被覆盖修改。

若下一步补数据，需要完成的获取工作只有：登记并获取 GSE132080；恢复 `zhu2026_cd4` 的未完成获取并完成校验/索引。其余九项无需为来源覆盖重复下载。若目标是严格复现 PDF，则另行取得或实现其处理流程与配置。KnowGraph 参考 `DATA-XATLAS`、`DATA-CD4` 及一跳 `WL-MODALITIES`、`MOD-ATLAS`、`EXP-LOCO`：来源覆盖不等于有效敲低，剂量不等于统一完全敲低，原代细胞的多个状态不增加独立供体数；本次不据此新增实验或采用结论。
