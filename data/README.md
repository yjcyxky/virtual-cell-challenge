# VCC 2026 数据目录

这里保存已登记的外部数据、官方任务输入和来源记录，**不是已经统一好的训练集**。2026 官方不提供 challenge-specific 训练集；参赛者根据未扰动背景和 CRISPRi 靶基因预测扰动后表达，官方对照不是扰动监督标签。[Arc 官方任务说明](https://arcinstitute.org/news/virtual-cell-challenge-2026)

本页以 [CLAUDE.md](../CLAUDE.md)、实际文件和核验记录为准。`sources.json` 中的 `tier`、`why`、`caveat` 含有历史判断，不能直接当作质量认证或已验证的建模结论。

## 当前登记与下载范围

状态快照：**2026-09-18 22:43，America/New_York**。大小为十进制 GB，仅统计 lock 中的文件；不含额外的 `SOURCE.json`。已完成表示当前登记选择齐备，不表示上游所有版本、SRA 测序 reads 或训练派生物均已下载。下载中的进度会继续变化。

| 资源 | `data/raw/` 下的目录 | 登记范围 | 状态 / 大小 |
|---|---|---|---|
| 2025 VCC H1 | `arc_vcc2025_h1/` | Training、Validation、Test 的 H5AD 和计数 CSV，共 6 个文件 | 已入库，34.36 GB |
| Replogle 2022 | `replogle2022/` | K562 GWPS、K562 essential、RPE1，各有 bulk 和 single-cell 版本，共 6 个 H5AD | 已入库，85.74 GB |
| Nadig 2025 | `nadig2025/` | HepG2、Jurkat 的 2 个 raw single-cell H5AD | 已入库，14.98 GB |
| Jiang 2025 | `jiang2025/` | 5 个 Seurat RDS、3 个基因集 RDS、1 个说明文件 | 已入库，20.14 GB；尚未转换 |
| McFaline-Figueroa 2024 | `mcfaline_figueroa2024/` | GSE225775 的全部 31 个补充文件和 2 个 GEO 元数据文件，覆盖四组筛选 | 已入库，13.39 GB |
| scPerturb 合集 | `scperturb/` | 54 个 H5AD；包含不同测量模态和干预方式 | 已入库，43.04 GB；需按任务筛选 |
| Tahoe-100M | `tahoe100m/` | 3,388 个表达分片中的 300 个，另有 6 个元数据文件 | **子集**已入库，32.22 GB |
| scBaseCount | `scbasecount_2026_01_12_human/` | 2026-01-12 发布的人类数据，选定 1,808 个 H5AD 和 2 个元数据文件 | **下载中**；目标 200.00 GB |
| LINCS L1000 | `lincs_l1000/` | GSE92742 Level 5 GCTX 及 4 个注释文件 | 已入库，21.34 GB |
| DepMap 24Q4 | `depmap_24q4/` | 基因依赖性、gene effect、表达和模型/profile 元数据，共 5 个文件 | 已入库，1.87 GB |
| STRING / Reactome / GO / HGNC | `networks/` | 网络、通路、基因注释与映射，共 6 个文件 | 已入库，0.13 GB |
| 2026 VCC 验证输入 | `arc_vcc2026_controls/` | 官方 ZIP、3 个 H5AD、2 个 CSV、manifest，共 7 个文件 | 已入库，1.32 GB；不含扰动标签 |
| 2025 基因轴 | `vcc_gene_axis/` | 旧版 `gene_names.csv`，1 个文件 | 已入库，116,023 字节；不是本届正式轴 |

**Srivatsan et al., 2020 已包含在 scPerturb 中**：`SrivatsanTrapnell2020_sciplex2.h5ad`、`SrivatsanTrapnell2020_sciplex3.h5ad`、`SrivatsanTrapnell2020_sciplex4.h5ad`，合计 2.93 GB。它不是缺失的数据源，也不应再与 scPerturb 的体积相加。scPerturb 还收录了 Replogle 与 Nadig 的处理版本；这些是研究级重叠，后续需核对是否存在相同细胞和测量记录。

scBaseCount 在上述快照时有 **1,432/1,810 个文件达到登记大小**，另有 4 个部分下载文件，合计落盘约 159.84 GB；尚无完整的 `SOURCE.json`，文件也尚未冻结。这不是已完成的内容审计。选择规则见 [selection manifest](selections/scbasecount_2026_01_12_human.json)：`Homo sapiens`、`GeneFull_Ex50pAS`，根据元数据筛选候选 control/untreated 样本。**这些标签不证明每个细胞都是合格的未扰动对照**。200 GB 是本地选择范围，不是全量 scBaseCount 的大小。

Tahoe 使用 `n_shards=300`、`shard_stride=11`。300/3,388 只是分片数量比例；本次未统计所选表达分片实际覆盖多少细胞、细胞系或 DMSO 对照，不能据此声称已覆盖完整 100M 细胞或全部背景。

第三方权重单独保存在 `models/`：

| 资源 | 目录 | 已下载范围 |
|---|---|---|
| ESM-2 650M | `models/esm2_650m/` | 5 个文件，2.61 GB |
| Arc SE-600M | `models/arc_se600m/` | 3 个文件，5.68 GB；包含上游 `se600m_epoch4.safetensors` |

这些是第三方输入，**不是本项目自训练产物**。核查时这 8 个文件仍有写权限；按协议应作为固定版本输入使用，实验 manifest 必须记录内容哈希。自训练模型只能写入本轮实验的 `outputs/`。

## 目录与写入约定

```text
data/
├── sources.json       数据选择、来源、用途与历史判断
├── registry.lock.json 逐文件 URL、大小及可用的校验和
├── MANIFEST.tsv       下载结果、实际 SHA-256、状态与来源地址
├── selections/        版本化的数据子集选择清单
├── raw/<source_id>/   原始输入；完成校验后冻结
├── logs/              数据获取日志和获取工具的工作文件
├── profiles/          历史 profile；当前有 H1 JSON，部分结论已过时
├── interim/           历史目录，目前为空；不是新实验的输出位置
└── processed/         历史目录，目前为空；没有现成的统一训练语料

models/<model_id>/                  第三方模型与权重
experiments/<id>/inputs/            对固定输入的引用
experiments/<id>/outputs/           解压、预处理、特征、切分、模型及训练日志
docs/datasets/                     数据说明与审计报告
```

按照 `CLAUDE.md`，实验只读 `data/`，不得覆盖、删除或修改输入；**新的预处理、缓存、切分和增强结果写入各实验 `outputs/`**，不写回 `data/interim/`、`data/processed/` 或软链接目标。数据获取和注册是独立的入库步骤，不是实验预处理的输出。

已完成的原始数据文件和其 `SOURCE.json` 已去除写权限；scBaseCount 下载中的文件除外。`chmod a-w` 只是辅助保护，不等同于不可变存储或备份，不能宣称它能阻止所有误删、替换或权限变更。

## 2026 官方验证输入与基因轴

当前归档的官方面板为 **`vcc2026-val-1`**，位于 `raw/arc_vcc2026_controls/`。文件内容与 [官方 manifest](raw/arc_vcc2026_controls/manifest.json) 一致：

- A/B/C 各有 18,400 个 NTC 细胞、18,533 个基因；各背景包含 46 个 NTC guide ID。
- `gene_names.csv` 含 `gene_name` 表头，给出 18,533 个唯一基因及顺序。
- `pert_counts.csv` 含 `target_gene` 表头，给出 300 个唯一靶基因，均在正式轴上。
- 每个 context × perturbation 要求 400 个预测细胞；对照是模型输入，不能作为扰动训练标签。
- 原始计数的有限性、非负整数性质、基因顺序及目标覆盖已验证，记录在 [SOURCE.json](raw/arc_vcc2026_controls/SOURCE.json)。

保留的 ZIP 为 662,118,680 字节，下载时通过官方 CRC32C 检查。快照版本为 `vcc2026-val-1-sha256-329a22bc29cac4f4`，ZIP 的 SHA-256 为 `329a22bc29cac4f43ad0b846b9bb1a383f7a945c83a8cb0450c465f6dc5cc4b5`。1.32 GB 的归档体积同时计入 ZIP 与解包原件，不是两个独立的数据集。

2025 H1/旧基因轴有 18,080 个基因，与 2026 轴共有 **18,077** 个；2025 独有 **3** 个，2026 独有 **456** 个。净维度差 453 不等于只需增加 453 列。RNA 输入应先确认标识体系、别名、重复 symbol 和测量覆盖，再按正式轴映射；未测量基因不能补零后当作真实零表达监督。蛋白特征不能直接按 RNA 基因轴合并。

这个面板是当前验证快照，不能自动代表未来面板。后续官方包若改变，应以新来源/快照登记，不能覆盖现有输入。

## 已核实的结构问题与训练前待办

### H1 的重复记录

三份 H5AD 分别有 221,273、98,927、170,846 行，共 **491,046** 行。各自含相同的 **38,176** 个 NTC `(batch, barcode)` 标识；三文件的标识并集为 **414,694**，冗余行数为 `2 × 38176 = 76352`，约占直接拼接行数的 **15.55%**。

因此旧版的“376,531 个唯一细胞、23% 重复”不正确。这里核实的是**唯一记录标识**；尚未逐一比较重复记录的整行表达是否一致。训练前需要核对表达、去重，并保证相同对照记录不跨训练和验证划分。

### 总计数差异不等于强制的下采样规则

本次重新流式读取 `X`，计算每行总计数，中位数如下：

| 输入 | 每细胞总计数中位数 |
|---|---:|
| 2025 H1 Training | 53,912 |
| 2025 H1 Validation | 54,312 |
| 2025 H1 Test | 54,125.5 |
| 2026 NTC A | 20,109 |
| 2026 NTC B | 19,946 |
| 2026 NTC C | 20,034 |

A/B/C 对照的中位数接近 20k，但每个细胞并非恰有 20k 计数；这些对照统计量也不能推出隐藏扰动真值的深度分布。旧 profiler 的 `EVAL_DEPTH = 20000` 是硬编码参照值，不能作为官方规则的证据。是否下采样、采用什么目标分布、如何保留生物学上的总 RNA 差异，都应在实验计划中定义并验证；不再规定所有输入必须统一降到 20k。

### 不同资源不能直接拼接

- **scPerturb：**54 是 H5AD 文件数，不是 54 份同质 CRISPRi RNA 数据。已核实包含 24 或 4 个蛋白特征的文件、CRISPRa 和药物扰动文件。观测字段的部分统一不代表基因轴、数值语义、对照定义或批次效应已统一。
- **Replogle / Nadig：**本地文件的基因索引使用 Ensembl ID，symbol 位于 `var/gene_name`；还存在重复 symbol。不能将 ID 与 symbol 的零交集误判为无共有基因，也不能仅靠改列名完成映射。raw/bulk/处理版的使用取决于具体任务，不规定“伪批量只能用 scPerturb”。
- **Jiang：**5 个表达对象按 IFNB、IFNG、INS、TGFB、TNFA 命名，不是六个已分开的细胞系训练文件。当前没有转换产物，也没有登记说明曾提到的 `scripts/convert_seurat.R`。需要在实验内实现并锁定转换流程。
- **McFaline-Figueroa：**四组筛选同时包含遗传与药物条件，原始计数文件没有 Matrix Market 标准表头或维度行，并含 Monocle/CDS RDS 对象。因此“Jiang 是唯一含 RDS、需要格式适配的来源”不成立。详见 [数据说明](../docs/datasets/mcfaline-figueroa2024.md)。
- **Tahoe / scBaseCount：**背景表示学习是候选用途，收益尚未验证。需实际筛选条件、检查元数据、质控和跨来源重复，不能把所有细胞都视为 untreated control。
- **DepMap：**同时包含表达矩阵与依赖性数据，所以“C 层不是表达数据”的说法不成立；用它推断背景相似性是建模选择，不保证能识别匿名背景身份。
- **LINCS L1000：**本地 gene-info 有 978 个 landmark 和 11,350 个非 landmark 基因。Level 5 是合并重复后的 MODZ 差异表达签名，不是单细胞 raw counts；用途可以包括签名监督或先验，但不能混作本任务的原始计数真值。[CLUE 数据层级说明](https://clue.io/connectopedia/data_levels)

截至本次核查，`data/interim/`、`data/processed/` 为空，未发现 `experiments/*/outputs/`。统一 metadata、跨来源去重、基因映射、监督掩码、控制池及训练 split 均不能标记为已完成。

## 登记分组与溯源边界

目录由 `<root>/<id>` 决定；`root` 默认是 `data/raw`，两个第三方权重源使用 `models`。`tier` 只是下载筛选字段，当前登记如下：

| tier | 当前成员 |
|---|---|
| a | 2025 VCC、2026 官方验证输入、scPerturb |
| b | Replogle、Nadig、Jiang、McFaline-Figueroa |
| c | DepMap、networks、2025 基因轴、ESM-2、SE-600M |
| d | Tahoe 子集、LINCS L1000、scBaseCount 子集 |

这些分组不证明来源同分布、质量更高、模态一致或已完成预处理。修改 tier 不需要移动数据，但 `sources.json` 与 lock 应保持一致，因为 fetcher 读取的是 lock。

溯源链为 `sources.json` → `registry.lock.json` → 各来源 `SOURCE.json` 与 `MANIFEST.tsv`。完成入库的来源有 SOURCE；仍在获取中的来源可能尚无完整记录。

- **大小相同不等于内容相同。**审核应比较实际 SHA-256、SOURCE、MANIFEST 和可用的锁定校验和。不是每个上游都发布 checksum；没有上游 checksum 时，本地 SHA-256 证明的是与已记录内容一致。
- **URL 不一定不可变。**部分历史条目使用 `main`、`current` 或未带 generation 的地址；重新解析上游可能改变文件集合或版本。实验必须固定所用内容和获取/处理代码，不能只记录一个可变 URL。
- **元数据并非备份。**哈希可发现变化，但不能保证上游仍提供相同内容，也不能单独恢复文件。
- **历史字段可能不完整。**例如 Tahoe 的旧 SOURCE 只保存 basename；本次审计可通过唯一 basename + URL 与 lock 对齐。后续 fetcher 已保留相对路径，但未改写被冻结的旧 SOURCE。
- **McFaline 来源单独明确：**10 个补充文件直接来自 GEO，21 个通过固定 commit 的镜像获取；31 个文件均与镜像 LFS SHA-256 一致。镜像校验和不是 NCBI 发布的校验和，原始 GEO 地址也已保留。

## 核查和获取命令

以下全局获取/核查工具通过已有的 `virtual-cell` 基础环境执行，不依赖仓库根目录 `.venv`；实验依赖仍按 `CLAUDE.md` 使用各实验独立的 uv 项目。

```bash
# 只读内容审计：每次使用新的报告文件名；--only 可排除正在下载的来源
micromamba run -n virtual-cell python scripts/audit_data_inventory.py \
  --only arc_vcc2025_h1,mcfaline_figueroa2024 --workers 2 \
  --output docs/datasets/data-audit-manual.json

# 官方验证快照的专用只读校验
micromamba run -n virtual-cell python scripts/fetch_vcc_controls.py --verify-only

# 查看某一来源的下载计划与磁盘预算，不执行下载
micromamba run -n virtual-cell python scripts/fetch_data.py --only tahoe100m --dry-run
```

注意现有工具的实际行为：

- 公共数据的 `fetch_data.py --verify-only` **会重写 SOURCE 并追加 MANIFEST**，且会跳过缺失文件；它不是严格只读的完整性审计，也可能因 SOURCE 已只读而失败。日常复核使用上方 `audit_data_inventory.py`。
- `fetch_data.py --freeze` 会遍历整个 `data/raw/`，**不会按 `--only` 缩小范围**。有 scBaseCount 等活动下载时不要将它作为通用收尾命令。
- 下载器支持 curl 续传、按登记大小跳过已到位的传输，并默认计算校验和；但写入被冻结的 SOURCE 仍可能失败。“可续传”不意味着对任意冻结快照重跑都安全。
- `build_lock.py --only <id>` 用于有意登记/解析特定来源，不是核查命令；不要为了查看状态而全量重建 lock，或把上游新版本覆盖到旧快照。
- 旧 `scripts/dossier/profile_arc_vcc2025_h1.py` 会写 `data/profiles/`，含硬编码的评测深度；旧 HTML 仍有过时结论。当前只有 H1 的交互档案，不能声称每个数据集都有档案、每个判断都来自原始扫描。完成输出位置和证据校正前，不将该流程作为新实验的准备入口。

## 核查证据

- [本次 README 核查说明](../docs/datasets/data-readme-audit-2026-09-18.md)：逐项列出问题、证据和修正。
- [本次只读测量与目录快照](../docs/datasets/data-readme-evidence-2026-09-18.json)：H1 标识并集、六份矩阵的行和分布、基因轴、LINCS 注释和目录状态。
- [较早的完整文件审计](../docs/datasets/data-inventory-audit-2026-09-18.json)：当时 415 个文件的大小/哈希通过；范围不含随后登记的 scBaseCount 和 McFaline，不能将对应旧报告的下载状态当作当前状态。
- [McFaline 下载审计](../docs/datasets/mcfaline-figueroa2024-download-audit.json)与 [gzip 检查](../docs/datasets/mcfaline-figueroa2024-gzip-check.json)：33 个文件内容一致、32 个 gzip 完整性通过。
