# exp008：2026 基因轴与数据清洗（需求对齐草案）

日期：2026-09-27。状态：只读调查与需求对齐，**尚未对齐具体方案或开始实现**。
本文件区分用户已指定的范围、已核实事实和待选择方案；不是可启动训练的正式配置。

## 已指定范围

- 在独立 worktree 开展 exp008。分支 `exp008`，起点为 exp007 的
  `b70b485d24d5abc19c15292769ae07d891be96f5`。
- 采用 exp007 的模型设计，官方评分部分采用 exp004 的设计。
- 所有数据集需要预处理，基因限定于 2026 challenge 列出的基因。
- 结合既往实验的问题和各数据集特点确定清洗方案；先调查、grill 对齐，再实施。
- 遵守仓库训练协议：共享输入只读，正式派生产物进入同一 run 的 cache，
  新实验独立锁定环境，历史 run 与原始评分不改写。

当前只建立 worktree 和本文档，没有创建 exp008 环境、训练 run、W&B run 或派生矩阵。

## 已核实的模型与实验问题

### exp007 的继承对象

共享 XGBoost 标量回归器，70 维特征，包括靶点/readout 的 STRING 表示、功能/物理关系、
Reactome、CollecTRI 和 NTC 表达统计、全局投影、regulon 代理。
预测逐细胞 CPM 算术均值的 log2FC，epsilon 为 1e-9，使用加权平方误差。
生成器从 NTC 模板出发校准组成并生成非负整数 counts。
现有训练为 K562 GWPS、RPE1、HepG2、Jurkat，H1 留出用于选择检查点。

exp007 的 H1 最佳 Overall 为 0.020419，同靶点响应转移为 0.186772；
同一第 512 轮 checkpoint 的官方 A/B/C Overall 为 -0.127798344。
H1 已用于开发与选模，不能称为独立测试；H1 与 A/B/C 绝对分数不能直接换算。
这些结果没有证明失败主要由脏数据造成，也没有证明清洗必然提高成绩。

来源：[exp007 PLAN](../exp007/PLAN.md)、[REPORT](../exp007/REPORT.md)、
[模型特征](../exp007/src/features.py)、[评分协议审计](../../docs/exp007-official-scoring-protocol-audit.md)。

### 现有预处理的边界

exp007 主要执行基因身份映射、非负整数/非空验证、H1 重复 NTC 去重、每靶点至少 50 个细胞。
它没有形成统一的细胞质量、guide 可靠性和条件匹配清洗流程。
H1 三个原始 split 的 38,176 个 NTC 重复出现；历史审计确认总行数 491,046，
按 batch/barcode 的身份并集 414,694。9/19-v3 审计已核对全部重复组的计数与标签一致、
冲突为零，可去掉 76,352 个冗余行；正式 run 仍需验证所消费的输入版本与去重身份。

exp007 全局基因集合为 19,300，官方基因只占前缀，后面追加其他基因。
实际训练在各背景原生可靠测量轴上归一化，和严格使用 2026 轴不同。
本次读取 exp007 历史 statistics.npz 得到的可靠测量基因与官方内部身份轴交集如下，
不是新一轮映射/清洗后的最终覆盖：

| 背景 | 与 2026 轴交集 | 18,533 槽位中缺测 |
|---|---:|---:|
| K562 | 7,670 | 10,863 |
| RPE1 | 8,251 | 10,282 |
| HepG2 | 9,013 | 9,520 |
| Jurkat | 8,271 | 10,262 |
| H1 | 18,008 | 525 |

统一列顺序并不能补回未测量基因。缺测必须与实测零表达区分，不能补零后作为监督或真值评分。
19,300 个全局节点中有 767 个不在官方轴上；严格限制靶点和先验图也会改变训练任务集合。
exp007 的四个训练背景分别有 328/155/100/180 个合格扰动靶点位于官方基因轴之外。
这些数字按背景计数，不能相加解释为唯一靶点数。

极端 log2FC 也需要独立诊断：四个训练背景均有扰动均值接近零造成的负向极端标签，
不能笼统归因于低 NTC 表达。H1 的低表达标签问题很强，但 H1 没有进入本次训练。
全标签平方量统计不等于实际采样、加权后的训练 loss 贡献。
直接删弱响应、按差异显著性挑细胞或裁剪 LFC 都会改变监督问题，不能当作无影响修复。

来源：[exp007 data.py](../exp007/src/data.py)、
[历史输入审计](../../docs/datasets/data-inventory-audit-2026-09-18.md)，以及共享数据区
`data/assessments/h1-structure-20260919-v3/report.json`（更新后的重复计数证据）。
缓存证据位于 exp007 原 worktree 的
`experiments/exp007/outputs/20260927-exp007-h1-log2fc-s17/cache/data/`，未复制或修改。

## exp004 评分设计需要拆开的选择

两者使用相同的 cell-eval2 固定 commit
`5e64833518a6603a0301cbe28185d49c30f4a986`，六项 raw 指标定义相同。
但 baseline 构造、对照池和参考样本不同，不能只复用公式就称为同口径。

| 项目 | exp004 H1 | exp007 H1 |
|---|---|---|
| 输入/评分 NTC | 两个独立池，各 3,072 | 完整池 38,176，输入与评分重合 |
| 评分 baseline | 平铺常量 tile | dispersed NTC 重采样与逐基因缩放 |
| baseline profile 排除自身靶基因 | false | true |
| 每靶点真实参考细胞上限 | 128 | 400 |
| H1 评分基因数 | 18,005 | 18,008 |
| 预测采样 | 匹配参考数量，三个 seed | 400 个，单 seed |

该排除开关属于 baseline profile，并非六项指标的 target exclusion 定义变化。
官方固定版本已明确指出 counts 下 tile 的已知偏差，并保留它用于历史复现；
官方公开说明则要求平台评分使用 held-out real controls。
用户已在逐题对齐 Q1 选择 A：恢复 exp004 的输入/参照 NTC 隔离，
评分基线采用官方支持的 dispersed 构造，不继承历史 tile 基线。
本方案使用新协议身份，不能称为完整复刻 exp004。
NTC 分池比例、参考样本量、生成种子和 baseline profile 的自身靶基因排除设置仍待细化。
包的默认值也不能证明线上 r4 bundles 使用了同样的构建配置。

更换 NTC 池必须同步重算输入特征、监督统计、生成模板及 reference/anchors；
不能只更换 scorer 的 control。新数据、新轴、新面板须新建评分基准，不复用历史 anchors。
保留零响应与同靶点响应转移作为候选强对照，最终训练组合与选模方案仍待确定。

来源：[本地两实验逐项审计](../../docs/exp007-official-scoring-protocol-audit.md)、
[官方 baseline 固定源码](https://github.com/ArcInstitute/cell-eval2/blob/5e64833518a6603a0301cbe28185d49c30f4a986/src/cell_eval2/baseline.py)、
[官方评分与 held-out controls 说明](https://github.com/ArcInstitute/cell-eval2/blob/5e64833518a6603a0301cbe28185d49c30f4a986/docs/vcc2026_metrics/vcc2026-metrics.md#7-submission-file-requirements)。

## 基因对齐与清洗候选原则（未定稿）

固定输入为已登记 `vcc2026-val-1` 的 `gene_names.csv`：18,533 个唯一基因，
SHA-256 `25bfa66715e186bebabce7ac788bbcea47e2bf59ca70be1f8f3a06f2f0e47201`。
`pert_counts.csv` 的 300 个唯一靶点是另一张清单，全部在基因轴内；两者不能混淆。
本次只读核对 A/B/C 均为 18,400 × 18,533，基因顺序与该 CSV 相同。
H1 原始轴有 18,080 列，和官方表字面交集 18,077；这是映射前数字，
不能与上表经过身份检查的 exp007 覆盖混用。

1. 先核对来源、测量层、原始 QC 注释、细胞/guide/批次/处理身份，再生成限定官方基因的训练视图。
   如需要轴外基因计算原始细胞 QC，只保留 QC 标量和来源记录，不把轴外表达送入模型。
2. 官方槽位及顺序为外部身份；HGNC/Ensembl 用于显式映射，不得擅自改写官方 gene_names。
   TIAF1 与 MYO18A 同时在官方表中，不能按现代别名无条件合并；TMEM104 是官方槽位，
   SLC38A12 不是。多对一映射须区分合法计数合并与身份冲突，不能一律求和或任取首项。
   较早的单侧 canonical 映射会额外丢掉 559 个本来字面匹配的 H1 基因；另一份双侧映射
   又把官方自身过滤为 18,431，均不能作为最终输出契约。HSPA14/MSANTD7、LRTOMT/LRRC51
   等原始 symbol/Ensembl 冲突需要独立裁决，不能仅凭匹配到某个字符串就视为解决。
3. 每来源保留 measurement mask；不插补监督，不为了拼出完整矩阵制造零标签。
   归一化必须声明使用轴及分母；共同列名不意味着不同测量面板的组成量已经同分布。
4. 按来源/条件进行技术 QC 与匹配对照，先处理确证重复、无效身份、错误模态、数值异常。
   阈值应结合各来源的测序深度和注释证据，不直接给所有来源同一 UMI/线粒体阈值。
5. 保留可能的真实强扰动、细胞周期和应激响应；不把模型难以预测的细胞自动视为脏数据。
   knockdown、guide 一致性、双细胞和异常深度需要区分元数据证据与表达推断。
6. 数据清洗规则不根据留出扰动预测误差选择；reference 的准入面板、NTC 隔离和覆盖均固定记录。
   官方 A/B/C controls 的处理另作决定，不能无记录丢弃输入细胞或伪造扰动标签。

## 各数据源的候选处理与用途

这里的“全来源处理”首先意味着完整性、模态、身份、条件和角色都有记录。
不同模态不能用一条 counts 转换规则强行合并；是否进入 exp008 训练仍须对齐。
旧 `data/README.md` 的下载/转换状态已落后，不能据此重新下载或重做全部适配器。

| 来源 | 已知特点或问题 | 候选清洗与用途 |
|---|---|---|
| H1 2025 | 三个 split 共享 NTC；部分 symbol/Ensembl 冲突 | 去确证重复，保留 batch/guide；独立 NTC 分池；潜在监督或留出背景 |
| Replogle / Nadig | Ensembl 与 symbol 并存，重复 symbol；native 与 scPerturb 存在研究重叠 | 优先明确唯一原始版本；同 study/细胞身份去重；按 GEM/guide 匹配对照；缺测 mask |
| scPerturb | 54 文件混合 RNA/蛋白、CRISPRi/a/KO、药物与组合干预 | 逐文件核实数值层、干预方式、条件和对照；合格单基因 CRISPRi RNA 才是直接监督候选 |
| Jiang | 已有五个 verified counts.h5ad 转换；6 细胞系×5刺激形成30条件 | 保留细胞系和刺激；按 Batch_info/bc1_well/sample_ID 匹配同条件 NTC；固定后拆分的 Rep1/2 不当作独立培养 |
| McFaline | 四组筛选已有转换；混合 CRISPRi/a、药物及联合条件，原始 barcode 大量未成为合格细胞 | 保留 line/drug/dose/guide/hash-well；优先作者 cell-calling 与可核实注释，unassigned guide 不补标签；不同干预分开准入 |
| Tahoe | 本地是300个表达分片；药物、多 plate/line；存在特殊 CLS token | 按格式显式解码特殊token；匹配同plate/line DMSO；不把药物响应当单基因CRISPRi标签 |
| scBaseCount | 1,808个H5AD已到位；存在非GEX、物种/文库冲突，缺逐细胞guide标签 | 按上游研究与文库身份准入、研究级去重；确认无干预的人RNA才可考虑背景表示，不能整体当NTC |
| LINCS / DepMap | Level5 MODZ、bulk表达/依赖性等语义，与单细胞counts不同 | 分别做基因映射和版本登记；如使用仅作明确的先验/辅助信息，不混作counts监督 |
| HGNC / STRING / Reactome / CollecTRI | 固定外部身份与关系先验；节点变化会改变谱表示 | 固定版本，映射到官方槽位，记录不解析/冲突/丢边；不能把关联边当作有符号因果效应 |
| 2026 A/B/C | 官方提供的是NTC输入，没有扰动真值 | 验证完整性、轴和计数；处理规则单独固定，不作为扰动监督 |

scPerturb 较新审计区分 51 RNA/3 protein、44 human/10 mouse；RNA 文件中 47 个
数值 count-compatible、4 个连续转换尺度。count-compatible 本身不证明原始 UMI 语义。
scBaseCount 较新语义分类为 1,560 文件支持 human RNA、219 个靶向 guide/receptor/ADT、
21 个来源歧义、4 个物种冲突、4 个 genomic assay；human RNA 分类也不等于逐细胞已确认 NTC。

较新历史审计实测 Jiang 合计 1,628,476 细胞；Tahoe 本地子集 8,467,330 细胞、
50 个细胞系、14 个 plates。这些是本次读取既有审计的数字，不是本次重扫全部矩阵所得。
McFaline GxE2 的 43,209,765 个 barcode 不能作为合格细胞数；作者 CDS 细胞为 989,299，
另有 62,906 个阈值候选，需要独立准入，不能直接相加当作训练细胞。
已有转换缓存只有在确认内容哈希、转换语义和适用性后才能固定引用；已有探索报告本身不是清洗后训练集。

证据在原共享数据区的 `data/assessments/`，包括
`h1-structure-20260919-v3/`、`jiang-dossier-20260919-v2/`、
`mcfaline-gxe1-dossier-20260919-v2/`、`mcfaline-gxe2-dossier-20260919/`、
`tahoe-dossier-20260919-v2/`、`scbase-dossier-20260919-v2/overview.json`、
`scperturb-reviewed-20260919-v2/verification.json`、
`cross-source-coverage-20260919-v4/panels.json`。正式模块应复用相应解析逻辑并去掉一次性报告职责，
所有本次派生文件仍写入 exp008 的 run；不延续历史共享区的探索输出模式。

可复用代码包括 `scripts/dossier/rna.py`、`convert_seurat_cache.py`、`prepare_mcfaline.py`、
`mcfaline.py`、`gxe2_records.py`、`gxe2_capture.py`、`gxe2_context.py`、
`scperturb_audit.py`、`scperturb_design.py`、`tahoe.py`、`tahoe_counts.py`、
`prepare_tahoe.py`、`cross_source_identity.py` 与 `cross_source_coverage.py`。
Jiang 五份 counts 缓存的身份位于 `jiang-cache-20260919/<stimulation>/identity.json`，
记录计数精确保留。旧 R 转换器硬编码其他环境，需要整理为符合实验协议的路径或引用已核验的
固定转换产物；McFaline 部分大矩阵标记为 omitted/rebuildable，不能把 identity.json 的存在
当作全部缓存仍在磁盘的证据。上述为复用线索，未复制代码或执行转换。

## 逐题需求对齐

用户要求每次只问一个问题，收到答案后再推进；不一次发出整轮问题。

已确认 Q1（2026-09-27）：选择 A，恢复 exp004 输入/评分 NTC 隔离，
采用 dispersed 评分基线并登记为新协议。此次确认不代表其余方案已定稿或已授权开始实现。

已确认 Q2：选择 A，固定 exp007 的 70 维特征设计、逐细胞 CPM 算术均值的
log2FC 标签（epsilon=1e-9）、加权平方损失和 NTC 模板计数生成器。
因数据清洗、基因轴和 NTC 分池变化重新计算特征与标签，不复用旧统计值。
不将改变 epsilon、截断标签、调整损失或更换生成器归入本次清洗；极端标签单独诊断。
具体训练超参数、采样配置和结束条件仍需后续对齐，此项不预先确认它们。

待对齐的根决策：

- 所有数据的范围：全来源审计与适配、按证据分配用途，还是要求每个来源必须进入训练？
- 基因范围：18,533 个表达/先验/靶点身份是否统一限制；训练靶点是否仍允许官方 300 以外的轴内基因？
- 清洗目标：保守去技术错误并保留生物响应，还是显式构建只含高置信有效扰动的监督集？
- 研究优先级：隔离数据清洗带来的收益，还是在固定模型下优先整合合格来源提高迁移能力？

上述答案决定下一轮：各来源准入/用途、条件粒度、QC 阈值、NTC 分池与标签参照、
训练权重、留出设计、控制实验数量、官方提交时机、训练结束条件和资源安排。
全部对齐后再完成正式 PLAN、配置与实现。

## 当前执行边界

本次未重扫全部表达矩阵，历史审计不能冒充本次清洗结果；正式 run 仍需固定并核验实际输入。
`micromamba run -n virtual-cell` 当前出现 libmamba JSON parse_error，基础环境自身 Python 可用于只读调查。
未修改或重建基础环境；正式实施时需先恢复符合协议的启动路径，再创建 exp008 的独立 uv 环境。
未使用其他实验环境执行 exp008 训练。
