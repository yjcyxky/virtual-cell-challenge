# data/README.md 结论核查与修正

核查日期：2026-09-18，America/New_York。本次修改数据目录说明，不执行训练、格式转换、原始文件重写或数据源重新登记。

证据优先级：`CLAUDE.md` 的项目约定 → 当前原始文件及官方面板 manifest → 可核验的来源记录和官方资料 → 旧 README、profile 和人工判断。旧档案中的文字不是独立证据。

## 核查方法与实际范围

1. 读取 `CLAUDE.md`、README、sources、lock、来源记录、下载器、审计器及 dossier 代码。
2. 核对全部 15 个登记来源的目录、文件大小、权限与 SOURCE 存在性；其中 13 个在 `data/raw/`，2 个在 `models/`。scBaseCount 持续下载，因此目录状态带独立的观察时间。
3. 对 2025 H1 的三份 H5AD 重新读取 batch、barcode、target_gene，计算记录标识并集和 NTC 重叠；没有逐行比较重复记录的表达向量。
4. 流式读取三份 H1 和三份 2026 NTC 的完整 `X/data`，通过 CSR indptr 计算每行总计数；没有调用会写回 `data/profiles/` 的旧 profiler。
5. 重查基因轴、Replogle/Nadig 的基因标识、scPerturb 的蛋白/CRISPRa/药物文件，以及 LINCS gene-info 标记。
6. 核对 [Arc 2026 官方说明](https://arcinstitute.org/news/virtual-cell-challenge-2026)和 [CLUE 数据层级说明](https://clue.io/connectopedia/data_levels)。

机器记录见 [data-readme-evidence-2026-09-18.json](data-readme-evidence-2026-09-18.json)。六份矩阵行和与元数据扫描约 9.9 秒；这是一次结构和语义核查，**不是再次对全目录执行完整 SHA-256 扫描**。完整性事实引用此前的 [415 文件审计](data-inventory-audit-2026-09-18.json)及 [McFaline 33 文件审计](mcfaline-figueroa2024-download-audit.json)，明确各自的范围。

## 逐项修正

| 原说明的问题 | 核查依据 | 修正后的表述 |
|---|---|---|
| 将整个目录称为已组装训练语料 | interm/processed 为空，没有实验 outputs；官方包只有 NTC 和任务输入 | 区分原始输入入库与训练语料准备 |
| 预处理往 data/interim、processed 写，可随时删除重建 | `CLAUDE.md` 要求实验 outputs 独占，data 只读 | 派生物写入本轮实验 outputs；历史空目录不作为新输出位置 |
| models 可写并存放自己训练的 checkpoint | 协议规定 models 为固定第三方输入；上游提供了 SE epoch4 权重 | 自训练模型只进实验 outputs，记录现有权重文件仍有写权限的事实 |
| scPerturb 的 54 份都是同质 RNA / 可直接使用 | 本地包含 24/4 维蛋白特征、CRISPRa 和药物扰动 | 写为 54 个不同模态/干预的 H5AD，需筛选与映射 |
| A 层代表同分布或已统一，C 层不是表达数据 | a 混含 NTC 和蛋白文件；DepMap 含表达 CSV | tier 仅解释为当前下载分组，不等价于模态、质量或训练准备程度 |
| 缺少 scBaseCount 当前范围与状态 | 已登记 1,808 H5AD + 2 元数据，获取进程仍运行 | 加入带时间戳的下载快照，明确约 200 GB 人类子集 |
| Tahoe 数量可能被理解为完整数据或全部细胞背景 | lock 明确 300/3,388 分片，另有 6 个元数据文件 | 标注子集，分片占比不等于细胞或对照覆盖率 |
| Srivatsan 的位置不清，合集可能重复计数 | scPerturb 中的三个 Srivatsan 文件确实存在 | 明列在合集内，共 2.93 GB，不与合集重复相加 |
| H1 唯一细胞数 376,531、重复率 23% | 原始 `(batch, barcode)` 并集 414,694，总行数 491,046 | 改为 414,694 个唯一标识、76,352 冗余行、约 15.55%；保留整行表达未比对的限制 |
| H1 中位数笼统为 53k，2026 固定 20k，必须统一下采样 | 六份矩阵重新计算；NTC A/B/C 中位数 20,109/19,946/20,034，存在广泛分布 | 明列每个文件的中位数；测量事实与实验采样策略分开 |
| 将旧 profiler 的 EVAL_DEPTH 常量当作官方证据 | `EVAL_DEPTH = 20000` 是脚本硬编码，没有记录来源 | 不将其用作官方要求；不能从对照推断隐藏扰动真值深度 |
| gene axis 可按差额补列或只改 symbol | 18,080 与 18,533 的交集 18,077；旧独有 3、新独有 456；外部数据有 Ensembl、重复 symbol 和未测量列 | 强调 ID/别名/覆盖/重复处理与监督掩码，蛋白不按 RNA 列合并 |
| 伪批量用 scPerturb、原始版只供采样器 | 未完成效果比较，且存在研究级重叠 | 不规定未经验证的来源用途分配；按任务验证数值语义与重复 |
| Jiang 是唯一需要 R 格式转换的来源，转换脚本似乎已存在 | 本地无 convert_seurat.R，McFaline 也有 CDS/RDS | 明确未转换，RDS/格式适配不止一个来源 |
| LINCS 只能作先验，以及未交代 Level 5 数值语义 | gene-info：978 landmark、11,350 非 landmark；CLUE：Level 5 为重复合并的 MODZ | 说明不是 raw counts，可用于其他签名任务，但不混作单细胞计数真值 |
| 每个来源均有 SOURCE，每个数据集均有 HTML，所有数字都来自扫描 | scBaseCount 尚无 SOURCE；仅有 H1 HTML；模板/脚本有常量与过时结论 | 对完成状态和历史档案适用范围加限定，优先链接新的证据 |
| `fetch_data.py --verify-only` 是只读且检查完整集合 | 代码重写 SOURCE、追加 MANIFEST，并跳过缺失文件 | 改用 audit_data_inventory.py，保留具体工具限制说明 |
| `--freeze` 是可安全通用调用的收尾命令 | 分支在 --only 过滤前执行，遍历整个 RAW | 说明不能在其他来源下载中随意调用，--only 不限制冻结范围 |
| chmod 能防止所有误操作，哈希清单足以复现上游 | 文件权限不是不可变存储；旧 URL 有 main/current；不是每项都有上游 checksum | 描述权限、内容哈希、固定版本和备份各自的边界 |
| 根目录 Python/.venv 可直接用于所有命令 | 协议要求已有 virtual-cell 基础环境和独立实验 uv 项目；status.sh/dossier 硬编码根 .venv | README 全局工具示例改用 micromamba，实验依赖仍保持隔离 |

## 关键测量

H1 三份文件分别为 221,273、98,927、170,846 行；每份 NTC 38,176 行，NTC 标识集合一致，任意两份文件的 `(batch, barcode)` 交集恰为 38,176。因此总计 491,046 行，标识并集 414,694。这个结果不等同于已经写出去重矩阵。

| 输入 | X 每行计数和中位数 | 最小值 | 最大值 |
|---|---:|---:|---:|
| H1 Training | 53,912 | 19,981 | 431,281 |
| H1 Validation | 54,312 | 19,982 | 338,047 |
| H1 Test | 54,125.5 | 19,982 | 346,239 |
| NTC A | 20,109 | 3,275 | 52,420 |
| NTC B | 19,946 | 710 | 42,666 |
| NTC C | 20,034 | 3,447 | 50,965 |

2025 独有的三个字面基因名为 `HSPA14-1`、`TBCE-1`、`TMSB15B-1`。这只是现有轴的集合比较，不是已经完成 HGNC 别名合并。

## 修正范围与剩余限制

本次重写 `data/README.md`，添加本核查说明和机器证据，并记录纠错经验；未修改来源登记、lock、原始数据、权重、实验计划或 dossier 实现。已有 scBaseCount 下载独立继续，其进度文件会正常变化。

旧 sources 中的部分 `why/caveat`、dossier 模板/JSON/HTML 仍有历史假设。本次在 README 明确它们的适用范围，不把重跑旧工具描述成安全的准备流程。后续若使用这些工具，应先修正输出位置、冻结实现版本，并复核其统计逻辑和依据。

文档验证：README 与本报告的本地链接均存在，关键数值与 JSON 测量结果一致，`git diff --check` 通过。实际运行 `fetch_vcc_controls.py --verify-only`，7 个官方锁定文件全部通过；运行 Tahoe 的 `fetch_data.py --dry-run`，计划为 306 个文件且不下载。sources 与 lock 的 SHA-256 和核查开始时一致。
