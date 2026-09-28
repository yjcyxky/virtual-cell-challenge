# 实验存储清理记录（2026-09-28）

**状态：用户批准「执行B」，B 已完成；A 未批准、未执行。** 本文是存储维护记录，不是新的实验列表或研究证据。

原审核提案共 **41 个文件、约 347.51 GiB**。A 组拟删除可精确重建的临时矩阵；B 组已去重，保留每个原路径为指向同一份只读内容的硬链接。

| 组 | 操作 | 文件数 | 空间 | 状态 |
|---|---|---:|---:|---|
| A | 保存恢复配方后删除临时 baseline | 3 | 预计 132.49 GiB | 未批准，未执行 |
| B | 完全相同副本改为只读硬链接，保留路径 | 38 | 已释放 215.03 GiB | 已完成 |

逐文件路径、SHA-256、原因和保留来源见 [files.csv](files.csv)；原审核提案见 [manifest.json](manifest.json)。原清单的条目和审核条件保持不变，CSV 仅将换行规范为 LF；manifest 原字节及哈希不变。其中未批准/未执行字段描述的是提案生成时的状态；B 的当前状态以 [execution-B.json](execution-B.json) 为准。

## B 执行结果

- 执行前重新完整计算 38 个副本和 12 个保留源的 SHA-256，全部与原清单一致；重新扫描 2970 份文档/元数据的完整路径引用。核验记录见 [preflight-B.json](preflight-B.json)。
- 先在全部 10 个受影响 run 的 `cache/storage-retention.json` 保存来源、原文件身份和权限、恢复办法；再把保留源设为只读，逐文件使用同目录临时硬链接原子替换副本。所有 38 个原路径继续可读。
- 替换后重新计算 12 个共享内容的完整 SHA-256，并逐一检查 38 个路径的 inode、只读权限及链接数；原内容哈希、保留源 inode 均不变。硬链接共享保留源的 mtime，目标原 mtime 已留存。
- 实际释放原副本独占数据块 **230,881,767,424 字节（215.03 GiB）**。磁盘可用空间变化同时受其他活动 run 写入影响，前后读数和差额单独记入回执。
- A 的三个文件身份、大小、mtime 和权限未变。受影响 run 中其他 1944 个文件的身份、大小、mtime、权限也未变，其中 1128 份文档/元数据另通过内容哈希核验；研究 `check` 通过（33 个节点）。
- 活动 `init-program-s01` 及其复用来源不在清理范围；每次替换前检查可访问进程的文件句柄、内存映射、命令和工作目录，未发现 B 文件使用者。部分系统进程的 `/proc` 信息不可读取，限制已在核验记录中保留。

需要写入任何已去重文件时，先复制为独立 inode、校验哈希，再原子替换该路径并恢复该路径记录的权限。不能直接对共享 inode 增加写权限后修改内容。只读使用无需生成或恢复任何文件。

## 纳入标准

- 不重新训练、不重新生成模型预测、不重新运行官方评分，也不依赖外网下载。
- 重复文件必须全文件 SHA-256 相同，保留源不在删除清单；不依据相同名称、尺寸或抽样判重。
- 临时 baseline 必须有精确恢复配方，已经逐块重放整个数组，重放 SHA-256 与原文件相同。
- 原始数据、唯一模型/预测、回执、研究账本、指标、anchors、配置、数据身份、训练/失败日志均保留。
- init-linear 路线的活动 run、复用来源及所有输出整体排除。执行前重新确认是否又有旧 run 启动。

## A：三个可重建的临时 baseline

这些文件都没有对应的完成 bundle manifest。原实现仅把它们作为建立评分参照的中间矩阵，并在成功后 unlink；中断使文件留存。不是已完成模型预测，也不是评分结果。

| ID | run / 文件 | 大小 | 完整重放校验耗时 |
|---|---|---:|---:|
| A001 | `experiments/exp003-context-module-cvae/outputs/20260925-lodo-priordiff-s17/cache/holdout-K562/official/baseline-counts.npy` | 37.22 GiB | 78.1 秒 |
| A002 | `experiments/exp003-context-module-cvae/outputs/20260925-lodo-response-s17/cache/holdout-K562/official/baseline-counts.npy` | 37.22 GiB | 84.3 秒 |
| A003 | `experiments/exp006/outputs/20260926-exp006-loco-residual-s17/cache/official/K562/mean-response-baseline.npy` | 58.04 GiB | 114.9 秒 |

删除前必须在同目录保存 `.recovery-profile.npy` 和 `.recovery.json`：精确 float32 响应向量、原 NPY header、NTC 行身份来源、reference 来源/哈希、原文件哈希及恢复说明。当前尚未往历史 run 写入这些资料。

恢复算法：按原 header 创建临时 NPY；非 NTC 行重复已保存响应向量，NTC 行复制保留 reference 的相同行并转换为原 float32；流式写出、fsync、检查全文件 SHA-256，最后原子改名。无需调用模型、重新计算响应 profile 或评分。此算法已对全文件逐字节验证，但未再落盘生成 132 GiB 的副本。

本机 1 GiB 复制+fsync 抽测 1.86 秒（551 MiB/s），不能把小样本测速当完整重建计时。结合上表的全文件流式校验，预计单文件恢复为分钟级、通常约 2–5 分钟；并发 I/O 可能延长，未承诺秒级。若审核要求已实测的完整落盘时间，则先不批准 A。

## B：完全相同副本去重，保留原路径

38 个候选来自 12 个完整 SHA-256 相同的文件组。优先保留 exp007 已完成 run 的 counts，以及 exp004 的评分 reference/真实侧 DE 缓存；exp004 H1 来源被 exp00701 明确引用，保留为主副本。

B 不让旧路径消失：目标路径改为相同内容的只读硬链接。各 run 自己的细胞索引、基因轴、统计、配置、reference.json、评分 bundle 和来源记录保持原样，字节内容与旧哈希不变。也不共享 Python 环境。

**附带变化：** 保留源和新链接去除写权限，并在 run 的存储记录中说明共享来源。恢复或开发若要写 counts，必须先复制为独立 inode，不能修改共享文件。拆链回滚只需本地复制并恢复记录的权限，无需任何模型计算。只允许同文件系统、无活动写入者且哈希不变时执行。

| 受影响 run | 文件数 | 预计释放 |
|---|---:|---:|
| `exp003-context-module-cvae/outputs/20260924-lodo-fixedlr-s17` | 2 | 1.48 GiB |
| `exp003-context-module-cvae/outputs/20260924-lodo-s17` | 2 | 1.48 GiB |
| `exp003-context-module-cvae/outputs/20260924-lodo-v2-s17` | 2 | 1.48 GiB |
| `exp003-context-module-cvae/outputs/20260925-lodo-priordiff-s17` | 4 | 22.62 GiB |
| `exp003-context-module-cvae/outputs/20260925-lodo-response-s17` | 4 | 22.62 GiB |
| `exp005/outputs/20260926-exp005-nested-residual-s17` | 8 | 55.12 GiB |
| `exp006/outputs/20260926-exp006-loco-residual-s17` | 8 | 55.12 GiB |
| `exp006/outputs/20260927-exp006-dispersed-s17` | 8 | 55.12 GiB |

## 明确保留的主要大项

| 保留项 | 原因 |
|---|---|
| init-linear 全部输出，包括约 15.91 GiB 的 interrupted-predictions | 活动路线；中断预测另有恢复审计引用，不能按名字认定为垃圾。 |
| exp00701 约 357 GiB 预测，以及其他 Experiment 的 predictions | 各检查点/协议的正式生成结果；未证明快速且精确再生，不纳入。 |
| exp004 K562 的 37.22 GiB prediction-seed-101.npy | 需核实 checkpoint、生成状态和恢复耗时，当前证据不足。 |
| exp006 dispersed K562 的 CSR baseline/reference，约 123 GiB | 存储格式优化和随机 emission 重建另需验证，不借本次清理修改。 |
| exp003 最初计数缓存、exp007/exp004 保留源 | 被其他 run 复用，保留至少一份完整内容。 |
| 所有 checkpoint、正式 VCC/H5AD 预测、回执、模型及结果 Artifact 记录 | 维持复现、恢复和结果身份。 |
| 所有 PLAN/REPORT、配置、日志、metrics、raw 分项、anchors/bundle、细胞/基因身份表和研究三对象 | 保证 agent 能继续理解实验，不改历史结论。 |

## 原提案的执行约束

1. 重新读取 status、活动进程、文件句柄/映射及 DAG 来源；任何新活动引用、目标变更或源哈希变化都跳过相关项。
2. 先写每个受影响 run 的 `cache/storage-retention.json`，记录条目 ID、旧路径、大小、哈希、动作、保留来源/恢复方法和这份清单位置；不改旧 metrics、REPORT 或 Ledger。
3. A 先落地恢复材料并再次完成全流哈希验证；B 先固定只读源，再逐个原子替换。任何一步失败都保留原目标。
4. 只按 manifest 中的明确文件执行，不用目录级 rm 或通配删除。完成后逐项验哈希、核对研究 check，保存实际执行回执和磁盘释放量。
5. 所有“计划”和“已执行”状态严格分开；审批前不创建恢复资料、不改变实验文件、链接或权限。

核验范围限制：完整路径引用扫描覆盖 2903 份本地元数据/文档，不能识别所有动态路径。因此 B 保留路径，A 仅纳入代码明确定义的临时量；执行前仍需重新确认当前状态。

原盘点 Git：`73f0c85891ded1f6c372252b65bfb4cc5d767a38`。原活动 run 快照见 manifest；B 执行时的复核见 preflight 和执行回执。A 如另行获准，仍需重新复核。
