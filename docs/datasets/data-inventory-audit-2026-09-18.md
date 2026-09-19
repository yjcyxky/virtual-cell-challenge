# VCC 2026 推荐数据集入库核查

核查日期：2026-09-18，America/New_York。内容校验执行时间为 21:20–21:23，耗时 187.8 秒。

依据：[challenge 摘要](../README.md)、[训练协议](../../CLAUDE.md)、[数据源登记](../../data/sources.json)、[文件锁](../../data/registry.lock.json)、[落盘清单](../../data/MANIFEST.tsv) 和各数据源的 `SOURCE.json`。

**结论：尚不能说推荐数据集都已规整完成。已登记的下载范围全部完整；推荐资源中 scBaseCount 缺失、Tahoe 仅选取了部分分片；跨来源的统一训练语料尚未生成。**

## 1. 对本次 challenge 的理解

任务是根据目标细胞背景的未扰动表达和 CRISPRi 靶基因，预测该背景的扰动后单细胞表达。2026 不提供目标背景的扰动训练标签；A/B/C 是验证背景，最终 D/E/F 是另外三个背景。训练语料的主要作用，是支持扰动知识向未见背景迁移，而非按公开细胞系名称查找响应。[Arc 官方任务说明](https://arcinstitute.org/news/virtual-cell-challenge-2026)

因此，资源应按用途理解：

- **正式任务输入：**2026 controls、基因轴、扰动清单，不是监督标签。
- **跨背景 CRISPRi 监督：**2025 H1、Replogle、Nadig，以及完成条件整理后的 Jiang。
- **细胞状态表示的候选语料：**scBaseCount 和 Tahoe 的适当子集；observational 或药物处理数据不能直接当作目标 CRISPRi 真值。
- **关系和表示先验：**STRING、HGNC、DepMap、冻结模型等，需要与任务输入分开登记。

本地官方 `manifest.json` 已明确本轮为 `vcc2026-val-1`，含 A/B/C，每个背景 18,400 个 NTC 细胞、300 个待预测靶基因、18,533 个基因，要求每个扰动产生 400 个预测细胞。最终背景尚未到计划发布日期，当前没有 D/E/F 归档不算本次下载遗漏。[官方时间表](https://arcinstitute.org/news/virtual-cell-challenge-2026)

## 2. 核查范围与证据

本次重新读取 **415 个登记文件、263,437,480,419 字节**，逐文件计算 SHA-256，与 `SOURCE.json` 和 `MANIFEST.tsv` 的记录比较，并检查锁定大小及可用的上游 MD5/SHA-256。

| 范围 | 数据源数 | 登记文件数 | 字节数 | 结果 |
|---|---:|---:|---:|---|
| `data/raw/` | 11 | 407 | 255,149,382,737 | 全部存在、大小/哈希通过、登记文件全部只读 |
| `models/` | 2 | 8 | 8,288,097,682 | 全部存在、大小/哈希通过；文件仍可写 |
| 合计 | 13 | 415 | 263,437,480,419 | 0 个完整性异常文件 |

所有来源都有 `SOURCE.json`；`data/raw/` 下的这些来源记录也均只读。本次未改动原始文件、数据登记表或其历史条目。重新哈希证明文件与冻结记录一致；对于没有上游 checksum 的来源，不能据此额外声称获得了新的上游独立认证。

机器可读的逐文件结果见 [完整核查记录](data-inventory-audit-2026-09-18.json)。该记录包含输入文件哈希、逐文件重新计算的 SHA-256、锁定 checksum 验证结果、权限和来源匹配方式。

复查脚本：[audit_data_inventory.py](../../scripts/audit_data_inventory.py)。使用一个新的输出文件名，避免覆盖历史审计：

```bash
micromamba run -n virtual-cell python scripts/audit_data_inventory.py \
  --workers 2 \
  --output docs/datasets/data-inventory-audit-next.json
```

此次没有直接运行全量 `fetch_data.py --verify-only`，因为其公共数据分支仍会重写 `SOURCE.json` 和追加 MANIFEST，不符合本次只读核查目的。

## 3. docs/README.md 推荐资源逐项核对

下表的“齐备”只指当前登记选择的原始文件，不代表上游所有版本、所有附件或统一训练产物均已齐备。大小使用十进制 GB，压缩包和解压文件同时保留时均计入。

| 推荐资源 | 本地位置 | 原始归档情况 | 训练整理状态 |
|---|---|---|---|
| **2026 官方验证输入** | `data/raw/arc_vcc2026_controls/` | 7/7，1.32 GB；ZIP、三个 H5AD、两个 CSV、官方 manifest 全部校验通过 | 正式轴与任务输入已具备；不是扰动训练集 |
| **2025 VCC H1** | `data/raw/arc_vcc2025_h1/` | 6/6，34.36 GB；train/validation/test 各有 H5AD 和任务 CSV | 原始数据齐备；仍需控制去重、批次处理、基因映射和实验划分 |
| **Replogle 2022** | `data/raw/replogle2022/` | 6/6，85.74 GB；K562 GWPS、K562 essential、RPE1，各有 bulk 和 single-cell 版本 | 原始数据齐备；标识、metadata、控制定义和测量覆盖尚未统一 |
| **Nadig 2025** | `data/raw/nadig2025/` | 2/2，14.98 GB；HepG2、Jurkat raw single-cell | 原始数据齐备；仍需映射与和 scPerturb 副本去重 |
| **Jiang 2025** | `data/raw/jiang2025/` | 9/9，20.14 GB；5 个 Seurat 对象、3 个基因列表、readme | **尚未转换**为本项目可训练格式；刺激/时间/细胞系条件仍需分别整理 |
| **Tahoe-100M** | `data/raw/tahoe100m/` | 306/306，32.22 GB；6 个元数据文件 + 300 个表达分片 | **选定子集齐备，全量未下载**；控制/药物条件筛选与表达统一未完成 |
| **scBaseCount** | 无 | **未登记、未下载**；`sources.json`、lock、`data/raw/` 均无该来源 | 不存在可用的本地子集 |

`Arc Virtual Cell Atlas` 是资源集合，不是还需额外下载的一份独立数据。本地已经有其中的 VCC 2025 和 Tahoe 子集，但缺 scBaseCount，不能据此声称整个 Atlas 已入库。[官方 Atlas 说明](https://arcinstitute.org/tools/virtualcellatlas)

### Tahoe 的范围需要写清楚

注册配置明确为 `n_shards=300`、`shard_stride=11`；所选文件名来自 `train-xxxxx-of-03388.parquet`。因此这是 3,388 个分片中的 300 个，而不是完整 Tahoe-100M。约 8.85% 是分片数量比例，**不是已核实的细胞覆盖比例**。

此次检查了这 306 个文件的大小和完整哈希，没有解码全部 Parquet 并统计其中的细胞数、DMSO 数量或细胞系分布；不能把“分片存在”直接等同于“已经拿到了足够的跨细胞控制数据”。

## 4. 额外已入库的资源

| 资源 | 本地位置 | 校验范围 | 使用边界 |
|---|---|---|---|
| scPerturb | `data/raw/scperturb/` | 54/54，43.04 GB | 文件集合混合不同模态与干预方式；必须按训练目的筛选 |
| DepMap 24Q4 | `data/raw/depmap_24q4/` | 5/5，1.87 GB | 必需性、表达、模型元数据等先验；不是目标背景 CRISPRi 表达真值 |
| STRING / Reactome / GO / HGNC | `data/raw/networks/` | 6/6，0.13 GB | 原始先验已到位；特征和命名映射仍需在实验内生成 |
| LINCS L1000 | `data/raw/lincs_l1000/` | 5/5，21.34 GB | 平台不同，包含推断表达；不能直接混作单细胞 raw-count 真值 |
| 2025 基因轴 | `data/raw/vcc_gene_axis/` | 1/1，116,023 字节 | 保留旧版；本届正式轴在 `arc_vcc2026_controls/gene_names.csv` |
| ESM-2 650M | `models/esm2_650m/` | 5/5，2.61 GB | 第三方模型位置正确；尚无本轮按基因对齐的蛋白表示产物 |
| Arc SE-600M | `models/arc_se600m/` | 3/3，5.68 GB | 第三方模型位置正确；尚无本轮细胞表示产物 |

第三方权重按 `CLAUDE.md` 放在 `models/`，无需搬到 `data/`。当前权重文件仍可写，后续实验应记录内容哈希和固定版本，避免依赖可变权重。

## 5. 原始归档距离可训练语料还有哪些步骤

### 5.1 尚无统一化产物

核查时 `data/interim/` 和 `data/processed/` 为空，`experiments/exp001-context-pair-xgb/` 中只有 `PLAN.md`。未发现统一 metadata 表、跨来源去重结果、2026 轴对齐的表达矩阵、pseudobulk、控制池划分或训练 split。

这些历史全局目录为空并不是要求往其中写入数据。按照当前 `CLAUDE.md`，新预处理产物应写入 `experiments/<id>/outputs/`。`data/README.md` 中关于把统一产物写回 `data/interim/processed` 和自训练 checkpoint 放入 `models/` 的旧文字与当前协议不一致。

### 5.2 H1 去重数字已重新核对

此次直接读取三份 H5AD 的 `batch`、barcode 和 `target_gene`：

- 总行数：221,273 + 98,927 + 170,846 = **491,046**。
- 各文件 NTC 行数均为 **38,176**，其 `(batch, barcode)` 集合完全一致。
- 三文件的 `(batch, barcode)` 并集为 **414,694**，等于 `491046 - 2 × 38176`。

旧 profile/数据说明中的 **376,531** 与本次原始 metadata 核查不一致，不应继续作为唯一细胞数引用。本次没有写出去重矩阵，也没有逐一比较重复记录的整行表达计数；训练前仍应检查相同记录的计数一致性，再执行去重。

### 5.3 基因标识和测量覆盖尚未统一

Replogle/Nadig 的变量索引与正式 symbol 轴直接相交为零，是因为索引使用不同标识，并不意味着没有共有基因。读取其 `var/gene_name` 并处理旧式 categorical 编码后，可得到如下**字面 symbol 匹配**；这些数字尚未包含完整 HGNC 别名解析：

| 单细胞文件 | 测量列数 | 唯一 symbol 数 | 与 2026 轴字面匹配数 |
|---|---:|---:|---:|
| K562 essential | 8,563 | 8,561 | 7,940 |
| K562 GWPS | 8,248 | 8,246 | 7,681 |
| RPE1 | 8,749 | 8,748 | 8,260 |
| HepG2 | 9,624 | 9,623 | 9,024 |
| Jurkat | 8,882 | 8,881 | 8,284 |

存在重复 symbol，必须结合稳定 ID 和计数语义处理，不能只修改列名。未测量基因需要监督掩码，不得补零后当作真实零表达。

H1 的 18,080 基因与 2026 轴共有 18,077 个；2025 独有 3 个、2026 独有 456 个。官方 A/B/C 的 18,533 基因及顺序则已经一致。

### 5.4 scPerturb 不是 54 份可直接合并的 CRISPRi RNA 数据

文件和结构抽查确认其中包括：

- `FrangiehIzar2021_protein.h5ad`：24 个蛋白特征。
- `PapalexiSatija2021_eccite_protein.h5ad`：4 个蛋白特征。
- `TianKampmann2021_CRISPRa.h5ad`：CRISPRa 文件。
- `SrivatsanTrapnell2020_sciplex3.h5ad`：药物扰动数据，具有 dose、cell line 等字段。

另外已有 Replogle 和 Nadig 的处理版本，与独立来源存在研究级重叠。完整入库不代表这些文件全部进入首轮训练；需选择 RNA、单基因 CRISPRi、匹配对照及正确细胞背景，并处理跨来源重复。

### 5.5 Jiang 转换流程尚未落地

目前是五种 pathway/stimulation 对应的 Seurat `.rds`，不是六个已统一细胞系的训练表。仓库中未发现登记说明提到的 `scripts/convert_seurat.R`，也没有转换后的 H5AD 或 Parquet 产物。转换时需保留 raw counts、细胞系、刺激条件、时间和 guide 信息；不能只导出归一化表达矩阵。

### 5.6 Tahoe 的 SOURCE 路径记录需要改进

历史 fetcher 在 `SOURCE.json` 中只保存文件 basename，丢失了 `metadata/` 和 `data/` 前缀。本次 306 个 basename 均唯一，且可用 basename + 原始 URL 与 lock 唯一匹配，所有重新计算的哈希也通过，因此不是 306 个文件缺失。

后续登记程序应保留来源根目录内的相对路径。本次只记录问题，没有改写这份已冻结的历史 SOURCE；当前 lock 与 MANIFEST 已保留完整相对路径。

## 6. 下一步优先级

1. **先完成 EXP001 的数据准备。**H1、K562/RPE1、Jurkat/HepG2、官方轴及 STRING/HGNC 的原始文件已具备。按实验计划做来源适配、控制拆分、基因映射、去重、监督掩码与跨背景划分，完成任务覆盖和质量检查后再训练。
2. **再整理 Jiang 和 Tahoe 的任务相关子集。**Jiang 可补充跨背景扰动监督；Tahoe 先核实选定分片内的控制、细胞系和处理条件覆盖。按明确实验需求扩充，避免把数据体积当作训练收益。
3. **scBaseCount 是推荐资源覆盖的真实缺口，但不是 EXP001 原始文件的缺口。**如果后续要训练广覆盖细胞状态表示，应固定 release、quantification 和人类相关样本范围，再登记和下载选定子集。官方当前访问路径位于 `gs://arc-institute-virtual-cell-atlas/scbasecount/`，涉及 Marketplace 订阅和 Requester Pays 配置；应按实际访问说明规划，不能使用已经撤下的旧桶地址。[官方数据访问说明](https://github.com/ArcInstitute/arc-virtual-cell-atlas#accessing-the-data)

本轮完成的是只读核查和报告：没有启动训练、转换全部语料或额外下载 scBaseCount/Tahoe 全量数据。原始文件的完整性检查已通过；模型可用性仍需执行上述实验级数据准入检查。
