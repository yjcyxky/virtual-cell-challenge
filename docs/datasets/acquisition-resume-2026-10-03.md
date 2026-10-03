# GSE132080 入库与 CD4 下载恢复

本次按用户要求继续把缺失的公开数据写入 `/mnt/projects/virtual-cell-challenge/data/raw/`。本地 `data/raw/` 下通过对应目录软链接访问原件。新增 GSE132080，恢复原 CD4 下载任务；已完成的 Xaira 保持原登记和原件。

## GSE132080 已完成获取与校验

来源 ID 为 `jost2020_gse132080`，登记在 [sources.json](../../data/sources.json)，由 `build_lock.py --only jost2020_gse132080` 生成新锁条目；原有 18 个来源锁条目完全保持不变。

[GEO 原始发布目录](https://ftp.ncbi.nlm.nih.gov/geo/series/GSE132nnn/GSE132080/suppl/)的五个文件已全部下载到 `/mnt/projects/virtual-cell-challenge/data/raw/jost2020_gse132080/`：表达矩阵、genes、barcodes、cell identities，以及 sgRNA 序列/活性表，总计 **352,976,976 字节**。保留原始压缩格式，没有过滤细胞、合并 guide、裁剪基因或归一化。

校验完成于 2026-10-03 04:03 UTC：5/5 文件，0 个完整性问题；另对全部五份 gzip 执行完整解压校验并通过。Matrix Market 头为 33,694 genes × 23,633 cells、112,125,300 个存储项；这只是输入形状核对，不是 QC 后有效监督量或完整生物质量审计。源文件与 SOURCE 已去除写权限。

- [来源与本地 SHA-256](../../data/raw/jost2020_gse132080/SOURCE.json)
- [逐文件完整性结果](../../data/assessments/jost2020-gse132080-20261003/inventory.json)
- [可读校验报告](../../data/assessments/jost2020-gse132080-20261003/inventory.html)
- 下载日志：`data/logs/fetch-jost2020-gse132080-20261002.log`

GEO 文件列表未提供上游完整校验和；本次核对官方 HEAD 大小并计算本地 SHA-256，不能把本地哈希写成作者提供的校验和。文件公开可下载，未发现该数据集单独声明的许可证，未自行补写 CC 许可。

## CD4 已恢复原任务

`zhu2026_cd4` 沿用原来的 57 个锁定对象、S3 版本/ETag 条件、NAS 目标和 `new-crispri-20260930` 获取状态。恢复前确认没有原 downloader/finisher 进程，且来源和全部收尾代码哈希仍与原绑定相符。

恢复时启动 downloader PID `1929550` 和收尾 PID `1929551`，均脱离当前终端会话、追加原日志。PID 只描述启动时点，后续状态应读状态文件并核对进程命令，不能把仍存在的同号 PID 当作原任务。

2026-10-03 04:04 UTC 状态仍为 `waiting_for_download`，已有约 118.82 GB / 1,843.66 GB 字节落盘；41/57 个配套小文件完整，单细胞矩阵尚未完整。字节持续增长已核实，**尚未宣称整个来源完成或可用于训练**。下载结束后原收尾工作器会校验、冻结该来源、复读全部内容并建立索引；失败会写入状态并停止。

- [获取状态](../../data/assessments/new-crispri-20260930/zhu2026_cd4/acquisition.json)
- 下载日志：`data/logs/fetch-zhu2026-cd4-20260930.log`
- 收尾日志：`data/logs/finish-zhu2026-cd4-20260930.log`
- 来源范围、同条件恢复与完成判据见[原入库说明](new-crispri-acquisition-2026-09-30.md)。有活动下载器时不要重复启动。

## 手动同步整个仓库

见[同步说明](../maintenance/nas-sync.md)。同步脚本只替换**未纳入版本控制且严格大于 1 GB**的文件，并且先确认 NAS 副本与本地 SHA-256 一致。已有 NAS 数据目录链接会被识别和保留，重复同步不会把原件目录覆盖成自指链接。
