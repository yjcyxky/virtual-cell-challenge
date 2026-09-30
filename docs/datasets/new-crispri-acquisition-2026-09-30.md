# Xaira Orion 与 Zhu 2026 CD4+ T 入库

本次按 [PDF 与本地缺口核查](../research/dataset_overview_gap_analysis.md) 接入两个缺失的 CRISPRi 来源。**文档创建时下载仍在进行，不能标记为全部入库或训练可用。**机器内盘不足以容纳全量数据，新增原件使用现有 NAS；没有删减供体、条件、靶点或细胞。

## 固定来源与范围

| source_id | 实际获取范围 | 文件数 | 登记字节数 |
|---|---|---:|---:|
| `xaira_orion` | 官方 HF 发布的两个背景 raw-count Parquet、基因元数据、许可和说明，另附 Figshare v3 的 guide 设计表与两份细胞 guide 注释 | 338 | 126,424,871,412 |
| `zhu2026_cd4` | 官方公开 S3 `marson2025_data/` 完整文件集合：12 个供体×状态 H5AD、pseudobulk、DE、Croissant 元数据和补充表 | 57 | 1,843,656,060,436 |

总计 **1,970,080,931,848 字节**（十进制 1.970 TB）。这指作者公开的表达 counts 与配套分析材料，不包含 SRA 测序 reads，也不重复下载同一数据的另一种序列化格式。Xaira 的完整 Parquet 发布已经包含稀疏原始计数、细胞和 guide 标签，因此没有再复制约 559 GB 的 Figshare H5AD。

- Xaira：固定 [HF revision `53a5bc98d49247bcf967500292575c3d3602de31`](https://huggingface.co/datasets/Xaira-Therapeutics/X-Atlas-Orion/tree/53a5bc98d49247bcf967500292575c3d3602de31)；配套注释固定 [Figshare article 29190726 v3](https://api.figshare.com/v2/articles/29190726/versions/3)。表达分片按 HF LFS SHA-256 校验，Figshare 按其 MD5 校验。许可为 CC-BY-NC-SA 4.0，保留原件。
- CD4：数据入口由[作者分析仓库](https://github.com/emdann/GWT_perturbseq_analysis_2025)提供；[官方数据卡](https://virtualcellmodels.cziscience.com/dataset/genome-scale-tcell-perturb-seq)和所附 Croissant 标注 MIT。固定 S3 key、大小、ETag 和 version_id。该桶匿名 `?versionId=` GET 返回 403，因此采用公开 URL 加 `If-Match: ETag`，对象变化时请求失败。多段上传 ETag **不是完整文件 MD5**；仅单段对象用 ETag 校验 MD5，其余记录完整本地 SHA-256。Croissant 的全零 MD5 占位符不作为校验依据。

逐文件登记见 [`sources.json`](../../data/sources.json)、[`registry.lock.json`](../../data/registry.lock.json)；上游枚举和访问边界见[来源快照](../../data/selections/new-crispri-20260930.json)。未改写此前来源的 lock 条目。

## 位置、运行和状态

原件实际位于 `/mnt/projects/virtual-cell-challenge/data/raw/{xaira_orion,zhu2026_cd4}`，仓库 `data/raw/` 下同名软链接提供统一入口。下载器按软链接真实目标所在文件系统检查剩余空间，支持部分文件续传。

下载日志：

- `data/logs/fetch-xaira-orion-20260930.log`
- `data/logs/fetch-zhu2026-cd4-20260930.log`

每个来源由独立收尾工作器等待正在运行的 downloader，随后核对完整来源记录、仅冻结该来源、复读所有字节校验 SHA-256／可用上游校验和，最后扫描所有 counts 并生成索引。不会启动第二个并发下载器，也不会调用会冻结整个 raw 目录的全局 `--freeze`。

实时状态与收尾日志：

- `data/assessments/new-crispri-20260930/xaira_orion/acquisition.json`
- `data/assessments/new-crispri-20260930/zhu2026_cd4/acquisition.json`
- `data/logs/finish-xaira-orion-20260930.log`
- `data/logs/finish-zhu2026-cd4-20260930.log`

`stage=waiting_for_download` 只表示下载中；`files_at_expected_size` 不是内容校验。只有 `stage=completed` 且所指 inventory、`index/report.json` 均完成，才表示获取与本次整理全部完成。任何失败写入 JSON 并停止，不将短文件、缺失 SHA 或失败审计当成成功。代码或当前来源登记变化也会停止，避免几天后的任务悄悄使用另一套逻辑。

初次获取或未冻结来源的同条件续传：

```bash
/home/jy001/micromamba/envs/virtual-cell/bin/python scripts/fetch_data.py --only xaira_orion --jobs 8
/home/jy001/micromamba/envs/virtual-cell/bin/python scripts/fetch_data.py --only zhu2026_cd4 --jobs 8
```

不要在现有 downloader 活动时再运行上述命令。已冻结来源不重跑 fetch；只读核验用 `audit_data_inventory.py`。

下载结束后可重复执行以下收尾命令；等待现有下载进程时增加 `--wait-pid <下载器PID>`。工作器使用源级排他锁。示例中的 Python 是现有只读诊断环境，未修改基础环境或安装依赖；索引报告记录实际依赖版本。

```bash
/home/jy001/micromamba/envs/virtual-cell/bin/python scripts/finish_crispri_acquisition.py \
  --source-id xaira_orion \
  --analysis-python experiments/init-linear/.venv/bin/python \
  --output data/assessments/new-crispri-20260930/xaira_orion
```

CD4 对应替换两个 `xaira_orion` 为 `zhu2026_cd4`。完成的索引按代码、输入内容、官方轴和 inventory 哈希恢复；不匹配时使用新的 assessment 目录，保留旧结果。工作器退出后重跑收尾不会自动重启失败下载，应先查看日志并恢复对应 fetch。

## 整理结果与使用边界

`index/files/<原文件名>/` 为每个原件保留：

- `cells.parquet`：全部原始 obs（`native__` 前缀）、source/context、donor/state、batch/lane、barcode、guide、显式 NTC、标准化靶点、原始行号、原件 SHA-256 和物理记录 ID；附原计数总量、官方可测轴总量、检测基因数及来源 QC/guide/support 注释。
- `genes.parquet`：完整原始基因轴、每基因总 counts 与检出细胞数。
- `gene_mapping.parquet`、`official_axis.parquet`：沿用 `ChallengeIdentity` 的 symbol/Ensembl/HGNC 映射，保留官方 **18,533** 个位置与测量掩码。多义映射、重复特征、冲突和未测量位置不作为测得零。
- `tasks.parquet` 与 `completed.json`：任务规模、NTC、批次、完整 counts 有限非负整数检查、原注释行和核对、输出哈希。

上下文级 CSV 单列官方 300 靶点的实际细胞覆盖和来源质量/guide/非空条件下的覆盖，50/400 细胞阈值仅作支持量诊断，不宣称扰动有效。检查文件内和同背景跨文件的物理标识重复；不把 donor×state 当作 12 个独立供体。

**本次不删细胞、不归一化、不按 DE 筛靶点、不拟合表示、不创建训练/评估切分。**CD4 保留原始靶点注释，同时依据作者修订的 guide 表提供候选标准身份；multi/no-guide 与低质量标记仍保留，后续实验必须明确实际使用规则。配套 pooled DE、pseudobulk 与单细胞可能来自相同细胞，不能用留出供体/状态的汇总标签泄漏评估真值。训练矩阵、NTC 输入/评分隔离、研究/供体/状态/靶点留出以及跨来源重复检查仍由正式 Experiment 登记执行。

## 已执行的首个分片验证

`HCT116_Batch103.parquet` 已与固定 LFS SHA-256 对齐，并全量扫描；这是单文件验证，**不能代表两个新来源整体完成**。

| 项目 | 实测 |
|---|---:|
| 细胞 / NTC | 30,925 / 1,560 |
| 原始特征 / 可无歧义映射的官方测量位置 | 38,606 / 18,401 |
| 扫描非零 counts | 171,198,188 |
| 有限、非负、整数 / 原注释总 counts 不一致 | 全部通过 / 0 |
| 每细胞总 counts 中位数 / 检测基因数中位数 | 18,364 / 5,371 |
| 本分片实际观察到的官方靶点 | 208 / 300 |
| 未解析靶点细胞 | 6：DUSP29 三个、EPRS1 两个、TARP 一个 |

DUSP29/EPRS1 涉及官方旧名槽位多义，TARP 未解析；保留原因，不任意合并官方位置。完整分片索引与结果位于 `data/assessments/new-crispri-20260930/pilot-xaira-HCT116-Batch103/`。此处实际覆盖与先前仅按 gene catalog 名称交集得到的数字不同，不能混用。

新增行为测试覆盖 Arrow 切片/非连续 gene token、重复稀疏项、非法计数、NTC、CD4 guide 修订与多重分配保留、官方缺测掩码，以及短文件/缺来源哈希阻止收尾。
