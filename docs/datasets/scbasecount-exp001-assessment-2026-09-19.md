# scBaseCount 对 EXP001 的适用性评估

评估日期：2026-09-19。**下载完整，数值层面可读取；不能把所选 scBaseCount 整体作为未扰动对照或 CRISPRi 监督训练集。原 PLAN 的 HepG2 任务覆盖门槛仍未通过，尚未启动新的模型训练。**

本次只进行数据评估和准备，没有创建本地 `outputs/runs/`、W&B run 或模型 checkpoint。没有恢复用户移除的历史 preflight runs，没有修改共享 `data/`。原始 PLAN 从提交 `177f53c` 恢复；该历史版本仅实现过 S0 元数据审计，并没有可直接启动的 S1–S5 基线训练程序。

## 1. 本地完整性与表达矩阵

| 项目 | 本次实测 |
|---|---:|
| 发布/选择范围 | 2026-01-12、Homo sapiens、GeneFull_Ex50pAS 的本地选择 |
| 登记文件 | 1,810：1,808 个 H5AD、README、样本 metadata parquet |
| 总大小 | 200,001,038,092 字节，约 200 GB |
| 文件校验问题 | 0；逐文件 SHA-256 对齐 SOURCE/MANIFEST，同时核对 lock 大小和可用的上游校验和 |
| H5AD 总细胞行数 | 13,255,146；与选择 metadata 一致，不代表跨文件唯一物理细胞数 |
| X 结构 | 全部 CSC，36,601 个 Ensembl 特征，同一基因轴 |
| X 数值 | 全量扫描有限、非负整数；无零文库细胞；UMI 总和与 obs 注释一致 |
| 逐细胞 guide/target/perturbation 字段 | 0/1,808 文件 |
| 正式轴的原始 symbol 交集 | 18,533/18,533；不构成物种或文库类型合格的证据 |
| 重复 symbol | 10 个 symbol 各出现两次，其中 6 个在正式轴上 |
| 文件中位 UMI <500 | 308 个文件、2,014,499 个细胞；不能只按“已有 ≥500 细胞”接纳文件 |

`X` 对应 Unique counts；`UniqueAndMult-EM`、`UniqueAndMult-Uniform` 是另外的层，未混入本次计数审计。[Arc 数据说明](https://github.com/ArcInstitute/arc-virtual-cell-atlas/blob/main/scBaseCount/README.md)

所有文件仍有写权限。本次没有修改其权限或数据；训练前仍需对实际消费文件再次校验哈希并建立每个 run 的固定输入清单，不能把“本次哈希通过”理解为永久不可变。

## 2. 全量上游溯源揭示的问题

已通过 ENA 原始实验、样本和研究记录，对本地 1,808 个 accession 逐项解析，取回 1,805 份实验记录、1,725 份样本记录和 578 份研究记录。原始 XML、获取时间和内容哈希已保存在评估目录。

下列类别可重叠，不能相加当作排除总数。自动标记并隔离的并集是 **267 个文件、1,573,767 个细胞**；其余也没有自动认证为可训练。

| 风险 | 文件数 | 意义 |
|---|---:|---|
| 来自现有监督研究 | 48 | 45 个属于 Replogle 的 SRP376262，3 个属于 Nadig 的 SRP501831；不能当作独立新增背景或对照 |
| 标题/文库名称提示非 GEX | 220 | 涉及 sgRNA、ADT、TCR、BCR 等；这是保守隔离标记，需样本级解释，不声称 220 个均已人工判定错误 |
| 上游物种为 Mus musculus | 4 | 与本地 Homo sapiens 分类冲突，禁止作为人类训练输入 |
| 非 RNA strategy | 1 | AMPLICON，与常规 GEX 输入语义不同 |
| 上游身份未完整解析 | 4 | 2 个 NRX、1 个实验记录缺失、1 个样本记录缺失；NRX 应另走其 CELLxGENE 来源，不能猜测 |

34 个上游样本 ID 被多个本地 experiment accession 共享，涉及 113 个文件。这是技术重复/多文库核查线索，**不是已经证明 113 个文件含重复物理细胞**。仅按 SRX 或 barcode 去重不足以保证研究与细胞独立。

几个直接影响原设想的例子：

- **SRX24301895**：本地 metadata 为 HepG2、`perturbation=none`；上游明确为 HepG2 第 53 个 gemgroup 的 Perturb-seq sgRNA 文库。其 X 中位总计数仅 28。既不是新的匹配 NTC，也不能把该文库的表达矩阵当完整 RNA。对应 [ENA 实验记录](https://www.ebi.ac.uk/ena/browser/api/xml/SRX24301895)。
- **SRX24258243**：本地标为 Jurkat 无扰动，上游为 Nadig 研究中的 Jurkat Perturb-seq mRNA。没有逐细胞 guide 标签，不能将整个文件视为对照。对应 [ENA 实验记录](https://www.ebi.ac.uk/ena/browser/api/xml/SRX24258243)。
- **SRX15542286**：本地标为 RPE1 无扰动，上游为 Replogle RPE1 Perturb-seq；并入外部 PCA 可能引入本轮验证背景的扰动细胞。对应 [ENA 实验记录](https://www.ebi.ac.uk/ena/browser/api/xml/SRX15542286)。
- **SRX8094420**：另一个本地 HepG2 候选来自 HyPBase single-cell calling-cards/scRNA 研究，不是 Nadig 匹配 GEM 批次的新增 NTC。对应 [ENA 实验记录](https://www.ebi.ac.uk/ena/browser/api/xml/SRX8094420)。
- **SRX8983885**：上游是 Dcx-DsRed 小鼠小脑。即使本地有完整人类基因列，也不能据此判为人源表达。对应 [ENA 实验记录](https://www.ebi.ac.uk/ena/browser/api/xml/SRX8983885)。

所有观察来自当前本地选择及其来源记录；不能外推为整个 scBaseCount 的错误率。

## 3. 可供背景表示学习的候选子集

从 metadata 候选中，按确定性哈希次序审阅了 32 个原代、名义正常/无干预样本的原始研究和样本说明。**最终仅将 10 个不同研究、明确人源 RNA 的样本列入候选白名单**。其他文件保持排除或待复核，不为了扩大规模自动接纳。

以下 QC 是辅助数据候选配置：Unique UMI ≥2,000、检测基因 ≥1,000。它不修改原监督数据的资格规则。样本层面的正常组织/无干预说明也不保证所有细胞具有同一状态，不能把原代混合细胞等同于纯培养细胞系 NTC。

| Accession | 组织 | 原始细胞 | QC 候选细胞 |
|---|---|---:|---:|
| SRX10675020 | 正常真皮 | 3,848 | 2,603 |
| SRX22558193 | 正常气管 | 2,114 | 965 |
| SRX13441694 | 匹配正常声门下黏膜 | 4,095 | 3,551 |
| SRX25183950 | 出生后胸腺、明确 GEX/RNA 文库 | 15,845 | 10,162 |
| SRX18517380 | 正常血管区半月板 | 8,563 | 8,351 |
| SRX24084583 | 正常牙龈 | 10,988 | 9,724 |
| SRX14335977 | 正常角膜 | 4,156 | 2,561 |
| SRX25852478 | 未处理表皮 | 1,597 | 1,560 |
| SRX25495573 | 自然月经周期对照子宫内膜 | 8,974 | 7,954 |
| SRX23810064 | 正常卵巢 | 11,026 | 8,850 |
| 合计 | 10 个研究 | **71,206** | **56,281** |

候选白名单包含逐文件 SHA-256、固定 generation 恢复 URL、study/sample ID、人工接纳理由、QC 细胞身份哈希和明确的 `representation_only` 角色。候选文件重新核对了原始计数得到的 UMI/检测基因数与 obs 注释的一致性。

保守排除所有重复 symbol 后，候选文件可用的正式轴唯一 symbol 为 **18,527** 个；缺失的 6 个不能补零当作测量值。后续 PCA 只使用其已覆盖特征，按原规则进行缺失处理和覆盖检查。

候选加入方式：保留原来仅由本折训练背景 NTC 选择 PCA 基因的规则，每个辅助研究最多稳定抽样 500 个 QC 细胞，共最多 5,000 个，参与 PCA 拟合；不参与 XGBoost 的 Δ 监督、不并入 `C_feat/C_ref`，不贡献匹配控制或任务数量。该选择不涉及任何模型测试成绩。其收益未知，原代组织与细胞系有明显分布差异。

**这仍然需要明确修订 PLAN §4.1/§5.2 的 PCA 拟合范围。** 原 PLAN 只允许本折训练背景 NTC；不能在加入外部 scBaseCount 后仍声称严格沿用了原范围。

## 4. 上一轮的数据不足是否解决

重新从当前 H1、Replogle、Nadig metadata 计算，未读取扰动效应或模型分数：

| 每批次两池各至少 | HepG2 可用批次 | HepG2 ≥100 细胞候选任务 | H1 已见任务 | HepG2 已见任务 | Jurkat 已见任务 |
|---|---:|---:|---:|---:|---:|
| 原 PLAN：50 | 20/56 | 32 | 43 | **19** | 353 |
| 候选新 run：40 | 44/56 | 180 | 43 | **128** | 397 |

表中任务数均是**完整原始计数 QC、guide 审核前的元数据上界**，并已考虑原来的训练目标哈希分组和每背景 800 个任务上限；不是最终训练/评估任务数。原 50 规则即使取消 800 上限，也只有 20 个 HepG2 已见任务，仍达不到 30。40 规则下 HepG2 严格未见任务上界为 22；原规则为 4。

scBaseCount 增加的是可供状态表示学习的数据，**没有修复同批次 NTC 和有效 CRISPRi 标签的缺口，也没有增加独立的监督细胞背景**。借用其他研究、其他 GEM 或已被扰动细胞充当匹配 NTC 会破坏原比较设计。

可执行方向是同一 EXP001 的新 run：预先登记每批次 50→40 和辅助 PCA 范围的变更，其余每任务 ≥100 细胞、每背景每池 ≥400、每主测试背景 ≥30 任务、3 折/5 模型组/3 模型种子/18 次拟合及原成功标准保持。代价是少量匹配对照均值更不稳定；不得宣称这一修订必然提高效果。最终仍需完整 QC，通过后才能训练。

本次已提出范围选择，尚未收到用户决定。按原 PLAN §2.3、§3.2、§8.2，不能默认降低门槛并启动。现有仓库及历史提交也尚无 S1–S5 训练/评估实现，确认新 run 范围后仍需补齐、验证并冻结这些程序，不能把 `assess_data.sh` 当作训练入口。

## 5. 证据与复现

- [完整性报告](../../experiments/exp001-context-pair-xgb/preparation/scbasecount-20260919/integrity.json)
- [全量结构与计数摘要](../../experiments/exp001-context-pair-xgb/preparation/scbasecount-20260919/structure-summary.json)；[逐文件记录](../../experiments/exp001-context-pair-xgb/preparation/scbasecount-20260919/structure-counts.jsonl)
- [上游溯源与 XML 哈希](../../experiments/exp001-context-pair-xgb/preparation/scbasecount-20260919/ena/provenance.json)
- [全部文件接纳状态](../../experiments/exp001-context-pair-xgb/preparation/scbasecount-20260919/admission.tsv)
- [辅助候选清单](../../experiments/exp001-context-pair-xgb/preparation/scbasecount-20260919/auxiliary-selection.proposed.json)
- [原门槛覆盖](../../experiments/exp001-context-pair-xgb/preparation/scbasecount-20260919/coverage-ntc50.json)；[40 门槛候选覆盖](../../experiments/exp001-context-pair-xgb/preparation/scbasecount-20260919/coverage-ntc40.json)
- [新 run 修订草案](../../experiments/exp001-context-pair-xgb/RUN_PROPOSAL.md)；[候选配置](../../experiments/exp001-context-pair-xgb/configs/baseline-scbasecount.proposed.yaml)

评估依赖已按原 `uv.lock` 安装到 EXP001 独立环境，未修改基础 `virtual-cell`。Python 3.14.7/aarch64，XGBoost 3.3.0，cell-eval2 固定提交 `5e64833518a6603a0301cbe28185d49c30f4a986`；导入检查通过。计数、稀疏偏移、重复身份、轴顺序等 6 项边界测试通过。

重复评估创建新的准备目录，不创建训练/W&B run，不覆盖已有报告：

```bash
cd experiments/exp001-context-pair-xgb
./assess_data.sh
```

ENA 属于可更新的外部来源，重取结果可能变化；本次结论以已保存的 XML 快照及其哈希为准，不以未来在线内容替换本次证据。

原始 XML 集中保存为评估目录中的 `upstream-xml-snapshots.tar.gz`，避免将数百页上游 XML 展开混入代码 diff；本地展开文件不单独提交。在该评估目录执行 `tar -xzf upstream-xml-snapshots.tar.gz` 可恢复原始路径，逐项 SHA-256 见 `assessment-identity.json` 和 `ena/provenance.json`。归档本身的 SHA-256 见 `verification.json`。
