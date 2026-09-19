# McFaline-Figueroa et al., 2024：sci-Plex-GxE

状态：**33/33 个上游文件下载完成，完整性校验通过，连同 SOURCE.json 共 34 个文件已冻结为只读。** 登记 ID：`mcfaline_figueroa2024`；落盘目录：`data/raw/mcfaline_figueroa2024/`。

论文：McFaline-Figueroa et al., *Multiplex single-cell chemical genomics reveals the kinase dependence of the response to targeted therapy*, Cell Genomics 4(2):100487 (2024)，[DOI: 10.1016/j.xgen.2023.100487](https://doi.org/10.1016/j.xgen.2023.100487)。

来源：[GEO GSE225775](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE225775)、[作者分析仓库](https://github.com/cole-trapnell-lab/sci-Plex-GxE)。作者仓库确认该系列包含以下四组实验。

## 下载范围

下载 GEO `filelist.txt` 列出的全部 **31 个补充文件**，另保存 `filelist.txt` 和 `GSE225775_family.soft.gz`，共 **33 个上游文件、13,392,023,219 字节（13.39 GB / 12.47 GiB）**。`SOURCE.json` 是额外的本地来源记录。直接下载各成员文件，未重复保存包含同一批文件的 `GSE225775_RAW.tar`；本次范围为表达数据和注释，不含 SRA 原始测序 reads。

| GEO 样本 | 实验 | 补充文件数 | 压缩文件总字节数 |
|---|---|---:|---:|
| GSM7056148 / sciPlexGxE_1 | HPRT1 与错配修复相关基因的遗传及药物联合筛选 | 8 | 348,995,319 |
| GSM7056149 / sciPlexGxE_2 | 激酶组遗传及药物联合筛选 | 11 | 11,152,123,813 |
| GSM7056150 / sciPlex_3 | 胶质瘤干细胞化学基因组筛选 | 6 | 420,901,709 |
| GSM7056151 / sciPlex_4 | 联合用药化学基因组筛选 | 6 | 1,469,994,829 |

每组都包含计数矩阵、细胞注释、基因注释、hash 计数与样本表、预处理 RDS 对象。前两组还包含相应的 guide 计数、序列或白名单。

## 来源与固定版本

GEO 直接下载完成的 10 个补充文件保留 GEO 下载地址。其余 21 个补充文件使用 [VirtualCell2025/datasets 镜像](https://huggingface.co/datasets/VirtualCell2025/datasets/tree/abee070ba0d9066d3d5a7315de3ca4aabe5eca9d/public/GSE225775-sciPlexGxE-McFalineFigueroa_et_al_2024_20250802/data)，固定 commit 为 `abee070ba0d9066d3d5a7315de3ca4aabe5eca9d`。GEO 的 filelist 和 SOFT 元数据仍直接从 NCBI 获取。

全部 31 个补充文件的名称和精确大小已与 GEO 清单交叉核对，SHA-256 固定到镜像 LFS 对象哈希；10 个 GEO 原件的完整 SHA-256 与镜像一致。镜像是第三方分发渠道，LFS 哈希不是 NCBI 发布的校验和。`origin_url` 保留每个文件的原始 GEO 地址，`checksum_source` 说明哈希来源，下载地址使用固定版本。

注册链路为 `data/sources.json` → `data/registry.lock.json` → `data/raw/mcfaline_figueroa2024/SOURCE.json` 与 `data/MANIFEST.tsv`。现有其他数据源的锁定条目保持原样。

GEO SOFT 元数据未声明独立的数据许可；作者代码仓库采用 MIT，不能据此推定 GEO 数据也采用 MIT。

## 完整性与使用说明

核验结果：33 个文件大小与 lock 一致；31 个补充文件的完整 SHA-256 与固定镜像哈希一致；32 个 gzip 文件完整解压校验 CRC；SOURCE、MANIFEST 与实际内容一致；新增输入冻结为只读。两个 GEO 元数据文件也与登记时读取的响应 SHA-256 一致。

[逐文件下载审计](mcfaline-figueroa2024-download-audit.json)读取了全部 13,392,023,219 字节，**异常文件数为 0**，审计期间登记输入未发生变化。[gzip 完整性报告](mcfaline-figueroa2024-gzip-check.json)验证了全部 32 个 gzip，流式读取的解压内容共 61,137,185,920 字节，没有将解压副本写入原始目录。

- 四个 `*_UMI.count.matrix.gz` 实际以三列坐标记录开头，**没有 Matrix Market 标准表头或维度行**，不能直接假定 `scipy.io.mmread` 可读取。构建矩阵时需要结合细胞、基因注释及作者处理代码核对轴与索引。
- RDS 对象属于作者的 Monocle/CDS 分析流程；本次保留原件，没有转换为 H5AD。
- 数据含 CRISPRi、CRISPRa、药物单独处理及联合处理。用于 CRISPRi 基线时，须保留并筛选扰动方式、细胞背景、药物、剂量和批次，对照也须匹配条件。
- 原始条形码或坐标记录的数量不等于通过质控的细胞数；尚未执行训练前质控、去重或 2026 基因轴映射。
- 按 `CLAUDE.md`，解压、转换、筛选和训练派生物应写入各实验 `outputs/`，不回写冻结输入。

重新核验时使用新的报告文件名：

```bash
micromamba run -n virtual-cell python scripts/audit_data_inventory.py \
  --only mcfaline_figueroa2024 --workers 2 \
  --output /tmp/mcfaline-figueroa2024-audit.json

gzip -t data/raw/mcfaline_figueroa2024/*.gz
```

在全新检出中重新下载的入口为 `micromamba run -n virtual-cell python scripts/fetch_data.py --only mcfaline_figueroa2024 --jobs 8`。现有公共 fetcher 会重写 SOURCE；冻结快照的日常复核应使用上方只读审计入口。
