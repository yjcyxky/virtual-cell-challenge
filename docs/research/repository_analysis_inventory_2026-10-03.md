# 仓库数据统计与实验完成情况：2026-10-03 只读快照

本次根据现存机器可读报告、评估产物与研究登记核对完成情况；没有重跑大矩阵统计、下载或训练。这里是既有结果的导航与分析，不新增研究管理对象，不替代 Method Space、Experiment DAG 或 Evidence Ledger。旧报告的 `completed` 表示其登记计算已完成，不表示每个统计量可估计，也不表示满足后来新增的模型训练后诊断协议。

核查摘要：数据已完成结构/QC、响应/DE、重采样/null、注释/状态、跨来源去重与响应异质性等多类分析；新增来源的完成度不同。`experiments/` 有 21 条路线目录，当前 DAG 的 104 个节点中 64 个完成、7 个失败、33 个草案；历史九条路线另核实 24 条完成记录，包含统计验证与固定权重重评，不能直接相加称为模型训练次数。模型结果支持保留同靶点共享响应基线；DepMap 功能映射出现正向信息，但当前完整方案未达晋级要求。

## data：实际已有的统计分析

统计成果主要在 [`data/assessments/`](../../data/assessments/)；该目录是指向 `/mnt/projects/virtual-cell-challenge/data/assessments` 的软链接。`data/interim/`、`data/processed/` 当前为空，没有一个可直接代表全部来源的统一训练矩阵。`data/profiles/` 只有 09-14 的旧 H1 档案；当前事实应优先读取 09-19 的完成报告及其后新增结果。

### 已完成到什么程度

以下采用最终 release 或有效完成版本，避免把缓存、重跑版本和同源处理副本重复计数。“任务”依来源可为靶点、构件或条件，不能直接相加为独立生物实验。

| 数据 / 范围 | 已完成统计与实际覆盖 | 完成产物 |
|---|---|---|
| 入库完整性（09-19 原有范围） | 逐字节 SHA-256、登记大小与可用校验和、来源记录、审计期间变更检查；2,258 个登记文件，476.83 GB，问题文件 0。它只证明当时登记选择完整，不覆盖后来新增的来源。 | [inventory](../../data/assessments/inventory-20260919/report.json) |
| H1 三个公开 split | 全量 counts/QC、基因轴/官方覆盖、靶点/guide/batch/NTC 支持、重复细胞；300 个响应任务、靶 RNA 比值、效应大小、DE/BH/BY、20 次重采样及深度敏感性。DE 299 个可估计、1 个不可估计；另完成细胞类型和 12 类状态代理、组成变化。 | [结构](../../data/assessments/h1-structure-20260919-v3/report.json)、[响应](../../data/assessments/h1-response-20260919/report.json)、[注释](../../data/assessments/h1-annotation-20260919/report.json) |
| Replogle：K562 essential / GWPS / RPE1 | 2,547,877 条单细胞记录，15,398 个构件任务均完成响应估计；DE 13,299 个可估计、2,099 个不可估计。包含靶 RNA、guide/分层一致性、重采样、NTC null、深度敏感性、类型/状态和 pseudobulk 与单细胞均值核对。 | [完整报告](../../data/assessments/replogle-release-20260919/report.html) |
| Nadig：HepG2 / Jurkat | 408,429 条记录，5,098 个构件任务均完成响应估计；DE 3,610 个可估计、1,488 个不可估计。结构、质量、响应、NTC、细胞类型/状态均已做。 | [完整报告](../../data/assessments/nadig-release-20260919/report.html) |
| Jiang | 5 个 RDS 已完成只读转换与逐值核对，并分析全部 1,628,476 条记录、6 细胞系 × 5 刺激；1,626 个响应任务，DE 1,613 个可估计、13 个不可估计。涵盖刺激内 NTC、来源标签冲突、条件/guide/批次、作者通路签名与状态代理。 | [完整报告](../../data/assessments/jiang-release-20260919/report.json) |
| McFaline GxE1 | 18,588 条 RNA 记录、24 个声明条件；遗传响应 58/64 可估计，固定 genotype 药物响应 70/77 可估计；guide/hash 赋值核对、CRISPRi/CRISPRa 分离、组成/状态、重采样均已做。 | [完整报告](../../data/assessments/mcfaline-gxe1-release-20260919/report.json) |
| McFaline GxE2 | 扫描 43,209,765 条原始 barcode 记录，其中作者 CDS 细胞 989,299；不能把原始记录数当作细胞数。14,121 个遗传任务中响应 14,109、DE 4,956 可估计；12,576 个药物任务中 12,568 可估计。另有 raw/CDS 一致性、guide/hash 复核、候选细胞、状态及空参照支持量分析。 | [完整报告](../../data/assessments/mcfaline-gxe2-release-20260919/report.json) |
| McFaline chemical3 / chemical4 | 分别 179,576 / 266,662 条 RNA 记录，72 / 1,116 个声明条件；匹配 vehicle 后全基因响应、类型/状态、20 次重采样及条件资格统计。chemical4 还做了 528 个组合减单药描述性差分，其中 513 个可估计；不是药理协同证明。 | [chemical3](../../data/assessments/mcfaline-chemical3-release-20260919/report.json)、[chemical4](../../data/assessments/mcfaline-chemical4-release-20260919/report.json) |
| scPerturb | 54 文件完成模态/来源审核，其中 51 个 RNA、3 个蛋白文件；8,802,191 条 RNA 记录。25,991 个深度响应任务中 24,787 个响应、19,464 个 DE 可估计；做了人/鼠参考、全部细胞 QC/类型/状态、对照资格和原始来源副本核验。 | [完整报告](../../data/assessments/scperturb-dossier-release-20260919/report.json) |
| scBaseCount 本地选择 | 全部 1,808 个文件、13,255,146 条记录、580 个研究标签已完成来源和表达统计。扫描 X 及两种多重比对层；做了深度/检测、重复候选、类型/谱系/12 类状态、源 QC 对照和共享样本关系。1,560 文件获得 human RNA 来源支持，219 是受体/guide/抗体文库，21 来源有歧义，4 物种冲突，4 属基因组测量。不能将全部记录视作合格 NTC。 | [完整报告](../../data/assessments/scbase-release-20260919/report.html) |
| Tahoe 本地 300 分片 | 8,467,330 条记录、50 个细胞系、14 个 plate、65,576 个本地条件；完整 counts 与 CLS 格式检查、重复/QC、vehicle 覆盖、类型/状态、药物响应和稳定性。63,864 个药物条件、1,366 个 vehicle 比较可估计；57,205 个条件可做稳定性分析。还比较了作者指定 plate6/14 重复。 | [完整报告](../../data/assessments/tahoe-release-20260919/report.json) |
| 2026 官方 A/B/C 输入 | 每背景 18,400 NTC × 18,533 基因；完整 counts/顺序/靶点覆盖、深度与检测、宽类型/状态、背景差异，以及 138 个 NTC guide 的同背景比较和不交叉 null。没有本地隐藏扰动真值。 | [完整报告](../../data/assessments/official-controls-20260919/report.json) |
| LINCS / DepMap / STRING / Reactome / GO / HGNC / 权重 | 完整数值/缺失值审计、基因映射、官方轴与靶点覆盖、来源及语义边界。包括 LINCS 473,647 签名 × 12,328 特征、DepMap 1,178 模型的适应度矩阵、1,690 个表达 profile；检查 ESM/SE 权重全部 tensor 的非有限值。这里的权重检查不是推理、训练或已对齐特征。 | [完整报告](../../data/assessments/priors-release-20260919-v4/report.html) |
| Xaira HCT116 / HEK293T（新增） | 两背景全部 332 表达分片已有完整 counts/QC、细胞/guide/batch/NTC 索引、重复检查、官方轴映射与靶点支持量。分别 3,409,169 / 4,534,299 细胞，NTC 165,777 / 218,838，均观察到 300 官方靶点、测得 18,401 官方基因位置。当前没有对应的完整响应/DE/异质性 release。 | [索引报告](../../data/assessments/new-crispri-20260930/xaira_orion/index/report.json) |

这些分析的常见统计包括：逐细胞文库大小/检测基因数分位数、逐基因检测/均值、基因身份与测量掩码；同背景/同来源分层参考下的 logCP10K 均值差、靶 RNA 残留比、排除靶基因的下游响应 RMS、BH/BY 差异表达；半样本稳定性、不交叉 NTC null、预设深度 thinning；保守宽类型组成和 12 类表达状态代理。各来源的实际适用项及不可估计项均在报告保留，并非每个来源使用完全相同的实验设计。

### 已完成的跨来源、跨背景分析

| 分析 | 实际结果与含义 | 证据 |
|---|---|---|
| 技术副本和来源身份 | H1 三 split 共享 38,176 个完全一致 NTC，合并时多出 76,352 行；491,046 存储行扣除已证实副本后为 414,694 条记录。scPerturb 的 Replogle/Nadig 五文件共 2,956,306 条记录被证实是原始来源副本，不增加独立实验。 | [跨来源报告](../../data/assessments/cross-source-release-20260919/report.html) |
| scBase 与监督来源重叠 | 检查 641 个共享样本文件对、48 个作者关联文库（39 RNA / 9 guide）；49,476 条 RNA 记录匹配到原来源 capture/barcode，但没有完全相同的原生 RNA 面板计数记录。关联、重处理和相同细胞不能混为一谈。 | [身份 summary](../../data/assessments/cross-source-identities-20260919-v2/summary.json) |
| 统一身份下的覆盖 | 1,931 个来源面板 × 18,533 官方基因、300 靶点；55 个原生映射轴、62,675 个源构件任务；区分字面覆盖、无歧义映射、冲突/未解析、实测零和缺测。该历史映射下 91 个官方标签未解析。 | [覆盖报告](../../data/assessments/cross-source-coverage-20260919-v4/report.json) |
| 同靶响应迁移性 | 37,845 个身份合格任务、249,071 个同靶比较对，其中 247,581 对可估计。相同有效对上绝对终点表达相关中位数 **0.761**，相对 NTC 的响应相关中位数仅 **0.00354**。绝对表达相似不能证明扰动响应可迁移；全局中位数还混合弱响应、不同来源与复用对照，不能作为来源质量排名。 | [异质性报告](../../data/assessments/response-heterogeneity-release-20260919/report.html) |
| 终点组成与状态内分解 | 按终点推断类型和周期 RNA 三分位划分，共 75,690 个任务分区，66,403 可估计；复算 37,633 个完整源响应，并验证“组成 + 状态内 = 共同支持总差”的数值恒等式。另分析 7,260 对基线，6,105 对可估计。它是条件性描述，不识别因果中介。 | [异质性完成摘要](../../data/assessments/response-heterogeneity-release-20260919/report.json) |
| 真实作者重复的绝对表达 vs 响应 | Tahoe plate6/14 全部 4,738 个匹配候选条件中，两项相关均有效的 4,104 对：绝对表达相关中位 **0.963**，各自扣除 DMSO 后响应相关中位 **0.241**。该差异再次说明基线与响应须分别评估。 | [同一有效分母比较](../../data/assessments/response-heterogeneity-release-20260919/report.json) |

### 完成边界与仍缺少的分析

- **新增 Jost GSE132080 只有入库校验完成**：5/5 文件、352,976,976 字节，问题 0；报告明确 `response_statistics=not_computed`。尚未发现它的细胞级完整 QC/官方覆盖、guide 活性梯度、响应/DE 或跨背景分析产物。[入库证据](../../data/assessments/jost2020-gse132080-20261003/inventory.json)
- **CD4 尚无完整本地索引或统计**：读取时收尾状态为 `waiting_for_download`（2026-10-03 19:55 UTC），46/57 文件达到登记大小。已下载的作者 pseudobulk/DE 是外部配套分析，不等于本仓库已按供体/刺激/guide 完成独立统计；完整量化仍待 counts 下载与校验完成。[获取状态](../../data/assessments/new-crispri-20260930/zhu2026_cd4/acquisition.json)
- **Xaira 目前止于入库与结构/覆盖统计**：两个背景的 300 靶点“观察到”不等于都有足量可靠监督。按当前来源质量/guide/非空规则，≥50 支持细胞的官方靶点分别 263/276；≥400 的分别 9/33。这些只是支持量诊断，不是已经验证的扰动效应或训练资格。[完整支持量定义](../../data/assessments/new-crispri-20260930/xaira_orion/index/report.json)
- **历史失败目录不应算未闭环全局失败，也不应算成功结果**：`jiang-dossier-20260919`、早期 cross-source、heterogeneity 等保留 `failure.json`；后续版本已有完成报告。scBase 早期表达扫描失败，但 `scbase-expression-20260919-v2/failures.json` 为 `[]`，最终 release 覆盖全部 1,808 文件。[失败列表](../../data/assessments/scbase-expression-20260919-v2/failures.json)
- **旧说明存在时效差异**：`data/README.md` 的 09-18 快照仍写 scBase 下载中、Jiang 未转换、Tahoe 未统计；09-19 完成报告已给出对应结果。09-14 H1 profile 的 `unique_cells=376531` 也不是现行去副本分母，按新版身份验证应为 414,694。[旧档案](../../data/profiles/arc_vcc2025_h1.json)、[纠正审计](../datasets/data-readme-audit-2026-09-18.md)
- **这些主要是描述统计与来源诊断**：缺少独立培养重复、未校准类型/状态、未测量基因和源标签冲突仍保留；细胞数、重采样次数、guide 数不能自动变成独立生物重复。现有大规模端点统计已经接触公开标签，不能据此宣称后续使用这些标签是未触碰验证，也不能替代每个模型 run 的冻结预测诊断和正式评分。

对应可复用实现主要在 [`scripts/dossier/`](../../scripts/dossier/)，尤其 `profile_responses.py`、`profile_crispri.py`、`profile_jiang.py`、`profile_tahoe.py`、`profile_response_heterogeneity.py` 和 `profile_endpoint_decomposition.py`；本次“已完成”判断使用上表结果文件，而非脚本是否存在。

## experiments：当前登记的完成实验

### 核实口径

- 当前共有 12 条已执行路线，另有 9 条历史路线目录。以 [DAG](experiment_dag.json) 的节点状态、实际 `metrics.json` 和 [Ledger](evidence_ledger.json) 的比较结论联合判断，不把目录、PLAN 或训练日志中的分数当作完整结果。
- 64 个 `completed` 节点的 metrics **64/64 存在，64/64 SHA-256 与 DAG 相符**，均记录 `status=completed`、`evaluation_completed=true`；其 64 个 checkpoint 引用与 64 个 predictions 引用均存在。本次没有重算全部大型 checkpoint/预测文件哈希；预测引用通常是 manifest，存在性不等于逐文件重新验真。
- 按用途分，64 个完成节点包括 **60 个预测器/基线拟合节点，以及 4 个观测、生成器或评估器诊断节点**。诊断节点的 checkpoint 可以是解析生成器或评估器规格 JSON，并非训练权重。
- Ledger 中有 **18 条已关闭本地比较结论**：13 条 `not_supported`、3 条 `supported`、1 条 `signal`、1 条 `inconclusive`；另有 27 条 `pending`。`E-SHARED-QC` 未显式填 `kind`，这里依据其比较与 `state` 计入本地结论。计算完成、判据通过、方法值得采用是不同状态。
- 下表早期方法的数值是匹配 S2-H1 主面板 Overall，通常为 280 个靶点，并有 25 个官方重叠靶点副面板；不能跨本地协议或与线上 A/B/C 的 Overall 直接排序。各来源使用实测官方基因子轴，缺测没有补零评分。

### 12 条当前路线

| 路线 | 已完成范围 | 主要结果及解释 | 正式报告 |
|---|---|---|---|
| `init-linear` | 7 个 run；基线、背景条件消融、正则强度、raw/PCA/Reactome/随机程序表示；另完成原 run 的官方导出与 anchor 诊断 | Shared **0.210866**，原 Linear **0.161135**；关闭背景条件后的匹配增益仅 +0.007511，未达门槛。降低正则后为 0.186094，仍未超过 Shared。真实程序表示 0.140997，低于 raw 0.165858、PCA 0.165860、随机程序 0.174793。否定的是当前具体实现的晋级主张，不是背景信息或通路先验普遍无用。 | [REPORT](../../experiments/init-linear/REPORT.md) |
| `masked-response` | 6 个 run：Shared/SVD/真实程序/随机程序四种缺测响应补全，再加 QC 与匹配随机删减 | 四种主分数分别 **0.210866 / 0.208995 / 0.209851 / 0.210535**，补全未胜出。QC 0.212761、随机删减 0.212329；QC−随机仅 +0.000432，重叠面板为 −0.000622，未支持 QC 特异收益。 | [REPORT](../../experiments/masked-response/REPORT.md) |
| `response-transport` | 1 个完整 MLP 响应迁移 run，4,096 steps | **0.089932**，低于匹配 Shared 0.210866；训练内 MSE 改善未转化为正式响应/DE 评分改善。其余两个局部变体仍为 draft，已列入停止分支。 | [REPORT](../../experiments/response-transport/REPORT.md) |
| `composition-response` | 1 个 logFC + 组成约束/IPF 完整方案 | **0.192196**；重叠面板 0.203138，对照 0.204838。没有超过共享响应，不晋级。 | [REPORT](../../experiments/composition-response/REPORT.md) |
| `distribution-response` | 1 个均值/离散度 + Gamma–Poisson 计数方案 | **−0.122329**；重叠面板 −0.017875。该分布生成整包未达要求，不能只归因于某一个损失或分布族。 | [REPORT](../../experiments/distribution-response/REPORT.md) |
| `gene-conditioned-response` | 1 个跨基因功能/NTC 条件解码器，16,384 steps | **0.171242**；重叠面板 0.162387，均低于 Shared。跨基因输出方案尚未证明能解决未监督读出。 | [REPORT](../../experiments/gene-conditioned-response/REPORT.md) |
| `conditional-cell-response` | 对齐输入后的 CVAE 1 个完整 run，8,192 steps；原 run 失败另保留 | **0.080311**；重叠面板 0.035926。零响应任务出现明显背景漂移，辅助检验约 13,697 个 DE，而 NTC 重采样为 0；提示生成器/基线保持问题。原 run 因 80 个输入基因在 H1 未测量而于训练前失败，不是方法阴性结果。 | [REPORT](../../experiments/conditional-cell-response/REPORT.md) |
| `task-observation` | `s02` 完成五背景观测轴、guide×batch 剂量可估计性及生成接口诊断，无模型拟合 | **16/20** 预设检查通过；零响应恒等和文库保持通过，但 4 个非 H1 背景的已知 bulk 重建未达要求。剂量不可估计常由细粒度分组过稀引起，不能标为无效扰动。 | [REPORT](../../experiments/task-observation/REPORT.md) |
| `local-capability` | `s03` 完成 15 个 S2/S3/S4 面板、60 组工具对照评分，无模型拟合 | **87/90** 检查通过；3 个面板因真实参考支持不足而 Overall 不可定义。诊断已完成，但当时评估器支持主张为 inconclusive；后续全靶面板由 `full-eval-support` 另行验证。 | [REPORT](../../experiments/local-capability/REPORT.md) |
| `population-emission` | 5 背景 × 4 靶点 × 2 生成器 × 20 独立生成，共 800 个非零群体，另有 null | **15/15** 检查通过；群体校准后的均值方差比约 0.834–1.048，原逐批 IPF 为 0.133–0.381。可采用其独立采样条件，但 DE Jaccard 中位数在五背景均下降；不是完整分布学对的证明。 | [REPORT](../../experiments/population-emission/REPORT.md) |
| `full-eval-support` | `s04` 完成 13 个外层划分、15 面板、30 组真实复制/靶点错配工具评分，无模型拟合 | **15/15** 面板六项归一化和 Overall 可用，已达到预设支持门槛。真实复制是评估接口检查，使用真值且存在自比较，不能当成模型或可达到的性能上界。S4-K562 排除全部靶点后只剩 263 个读出，仍有限制。 | [REPORT](../../experiments/full-eval-support/REPORT.md) |
| `depmap-response` | 7 方案 × 6 外层拟合 = **42 个 run**：Shared、源模板、kNN、Ridge/置乱、MLP/置乱；五个 S3 背景和 S2-H1，seed 930 | 正式生成 1,008 万细胞，42/42 完成诊断和六项评分。五背景真实功能对应关系优于匹配置乱：MLP 的 PDS/Overall 平均增益 **+0.08285 / +0.09779**。但相对 Shared/源模板的残差相关增益 0.01555/0.01871 未达 0.02，弱响应误差仍差；整包不晋级、不提交榜单。12 个 Ridge 拟合均跑完 200 次预算但未达到求解容差，不能称已收敛饱和。 | [REPORT](../../experiments/depmap-response/REPORT.md) |

七个失败节点为 `cell-cvae-s01`、`task-observation-s01`、`local-capability-s01/s02`、`full-eval-support-s01/s02/s03`。这些保留了输入/索引/评估支持/容量与执行故障，之后已有完成的替代身份；失败本身不提供方法无效证据。33 个 draft 是登记设想，不代表已训练、正在训练或仍应机械执行。当前 Ledger 的 `resource_allocation` 已明确 population-emission、DepMap 42 fits、full-eval-support 15 面板全部闭环，未启动新训练。

### 已完成的跨 run 诊断与官方核查

1. **固定预测下的 baseline/anchor 敏感性**：`init-linear` 完成 2 面板 × 4 条件 × 4 输出臂 = 32 组评分，检查 tiled/dispersed 生成及靶点排除。固定预测的 Overall 可因参照变化在约 0.160486–0.179025 间移动，Shared 优于 Linear 的排序稳定。它证明分数依赖评估参照；不能认证线上 r4 的构建配方，也不是 32 次新训练。[正式报告](../../experiments/init-linear/REPORT.md)、[r4 证据边界](r4_anchor_audit.md)
2. **九份已提交预测的全量分布诊断**：9 × 3 背景 × 300 靶点 × 400 细胞，共 324 万细胞、8,100 个群体；检查 counts/基因轴、文库、零率、Fano、方差、协方差、重复行、敲低、共享响应能量、有效秩与监督覆盖。九份硬约束全部通过；三份旧 CVAE 所检协方差相关约 −0.002–0.024，NTC 模板型约 0.970–0.984。后者主要继承 NTC，不能据此宣布学会扰动分布。[完整报告与 raw/normalized 表](leaderboard_prediction_counts_REPORT.md)
3. **实际监督与 unseen 回退核查**：上述 Shared/QC 的官方 272 个 seen 靶点均仅有 K562 的同靶标签，读出只覆盖 7,680/18,533；28 个全局 unseen 靶点的全部 33,600 预测细胞/提交是输入 NTC 行重采样，即零 Δ 回退。新增来源的覆盖没有自动进入这些已冻结模型。[监督核查](../../experiments/init-linear/outputs/init-linear-s01/cache/counts-audit-20260930/supervision-coverage.json)
4. **已有官方回执**：九份历史/当前提交中，该次审计最高为 Shared-QC Overall **0.050079**，Shared **0.048975**，Linear **0.033805**；其余六份为负。QC 相对 Shared 仅 +0.001104，没有匹配重复证据支持稳定提升；所有提交的 raw 表达误差对应 normalized 分项均触底，效应幅度和 DE 恢复仍弱。这只是仓库已有回执的比较，不是最新全榜排名或 SOTA 认证。[回执汇总](leaderboard_prediction_counts_scores.csv)

## experiments：九条历史路线的完成边界

历史产物不直接并入当前初始化方法有效性；不同 NTC、基因轴、模型选择与评分条件须分别保留。以下 24 条完成记录 = 1 + 3 + 13 + 1 + 6，其中包括统计验证和固定权重重评，不能称为 24 次独立新模型训练。

| 历史路线 | 已完成与未完成范围 | 实际结果 / 证据 |
|---|---|---|
| `exp001-context-pair-xgb` | `20260923-a` 完成：5 背景 × 3 XGB = 15 次外折拟合，再拟合最终 context-relation 模型并提交官方 | 共同轴 MSE：static 0.004047、context-pair 0.004073、context-relation 0.004106，Shared-shrunk 0.003654。背景关系未超越共享转移；官方 Overall −0.303153。无当前 PLAN/REPORT，以 [result.json](../../experiments/exp001-context-pair-xgb/outputs/20260923-a/result.json) 和 [evaluation.json](../../experiments/exp001-context-pair-xgb/outputs/20260923-a/evaluation.json) 为据。 |
| `exp002-response-transfer-validation` | b/c 两轮统计验证、d 共享收缩训练与官方提交，共 3 个完成记录；a 中止，native-count-transfer 未训练完成 | b/c 完成响应几何、拆半一致性、象限/等价分类、跨背景预测、收缩/错误靶点/模块随机对照。c 的跨研究 CRISPRi 收缩相对零响应降 MSE 8.62%–9.02%，d 离线降 8.48%，官方 Overall −0.197838。[b](../../experiments/exp002-response-transfer-validation/outputs/20260922-b/report.json)、[c](../../experiments/exp002-response-transfer-validation/outputs/20260922-c/report.json)、[d](../../experiments/exp002-response-transfer-validation/outputs/20260922-d/report.json) |
| `exp003-context-module-cvae` | 历史完整矩阵 **13 runs**：真实/随机/无先验各 3 seeds，另 4 消融；共 65 外折 + 13 最终拟合。后续 LODO、loss 和 prior-diffusion 分支无完整五折结果 | 旧共同 6,114 基因协议的响应 MSE：真实 0.018352、随机 0.017846、无先验 0.016840，均差于零响应 0.007235。不是官方六项模型结果。后续 response 分支仅 H1 cycle4 checkpoint 已完成官方导出（−0.035900），不代表该分支五折完成。[REPORT](../../experiments/exp003-context-module-cvae/REPORT.md) |
| `exp004-shared-response` | 计划 27 runs/135 fits 未完成。仅 conditional-s17 实际有产物，`failure.json` 为 incomplete / KeyboardInterrupt，无最终 metrics；另外 26 个计划入口为断软链接 | 已完成 H1 cycle11、K562 cycle1 两个 checkpoint 的官方导出，Overall −0.031130 / −0.056964。不能计完整消融矩阵；REPORT 中“仍在运行”是历史状态。[REPORT](../../experiments/exp004-shared-response/REPORT.md)、[失败记录](../../experiments/exp004-shared-response/outputs/20260925-exp004-conditional-s17/failure.json) |
| `exp005` | 1 个 failed run，无完整训练/评分 | XGBoost pseudobulk 残差与嵌套跨背景选择路线，KeyboardInterrupt，不是方法阴性结论。[REPORT](../../experiments/exp005/REPORT.md) |
| `exp006` | 2 个未完成 run：旧 tiled 协议主动 stopped；修正 dispersed 协议停留在 cross_validation，0/35 checkpoints 完成评分 | 旧 run 的部分 H1 分数不构成修正版对照；残留 cross_validation 字段不证明现在仍运行。[REPORT](../../experiments/exp006/REPORT.md) |
| `exp007` | **1 个 completed run**：512 轮、7/7 H1 checkpoint、两个对照与官方提交 | log2FC XGB 的旧 H1 Overall 0.020419，低于同靶转移 0.186772；官方 −0.127798。本地曾复用输入/评分 NTC，H1 也用于选模，不能当作当前独立验证。[REPORT](../../experiments/exp007/REPORT.md) |
| `exp00701` | **3 个训练 run + 3 个固定权重重评 run** 均完成；每训练 512 轮×7 checkpoints，重评各 3 生成 seeds | 连续响应/加 DEG/加模块旧 H1 最佳 Overall 0.127174/0.110784/0.104006，均低于 Shared。采用隔离 NTC 等条件重评为 −0.100006/−0.119251/−0.164621；同时变更多个评估条件，降分不能只归因于 NTC。不是 6 次新训练。[REPORT](../../experiments/exp00701/REPORT.md) |
| `exp008` | **只有 PLAN 草案**，无 run、环境、W&B 或 REPORT | 官方轴清理、同名列合并、基础/附加 QC、隔离 NTC 和 2,048 轮 XGB 都是计划，不能算已完成。[PLAN](../../experiments/exp008/PLAN.md) |

数据分析还有一部分保存在历史实验输出内：尤其 exp002 的统计验证和 exp003 的数据/构件支持、批次与方差、模块可观测性分析。因此仅扫描 `data/profiles/` 会严重低估仓库已有工作。

## 从现有证据可以得出的判断

1. **数据描述分析已经较充分，新增来源仍有缺口。** 原有主体来源已完成多层统计；Xaira 需要从结构/支持量推进到响应与可靠性分析，GSE132080 尚未做响应统计，CD4 仍需完成下载与索引。下载完成、统计完成、具备监督资格和已用于模型是四件事。
2. **已有可保留能力是同靶平均响应迁移。** Shared 的实测优势主要来自可用来源的同靶响应；复杂条件模型、模块先验、缺测补全、CVAE 等当前包尚无匹配的整体胜出证据。多数早期结论限于单 seed/H1 开发背景。
3. **DepMap 有正向信号，完整响应恢复仍未解决。** 真实功能映射优于匹配置乱，但弱响应、幅度/DE、未监督读出和完整晋级判据仍是瓶颈。生成计数合法、NTC 形态相似或有非零随机性，都不能替代响应准确性。
4. **评估支持已从部分面板推进到全量完成。** 应以 full-eval-support 最终报告及 Ledger 为准；DepMap REPORT 中“先完成剩余面板”的旧下一步已被后来的完成结果覆盖。它解决了本地评分可执行性，未证明线上标尺一致，更未证明达到 SOTA。

## 本次核查的限制与管理工具状态

按协议首先运行 `scripts/research.py status`，退出码为 1；交付前运行 `scripts/research.py check` 同样失败，报告 33 处同类路径错误：NAS 软链接使 `init-linear-s01` 和 `full-eval-support-s04` 保留的预测路径经 `Path.resolve()` 后位于仓库外，被 `reference escapes repository` 拦截。[路径限制实现](../../scripts/research.py)、[NAS 同步约定](../maintenance/nas-sync.md)

这些路径实际可读；本次独立读取 DAG/Ledger 并逐个核验 metrics 内容哈希和 checkpoint/预测引用存在性。本文与文档索引的 84 个本地链接均存在，`git diff --check` 通过。上述统计不是成功执行 `status` 的输出，也不声称研究门禁已通过；没有绕过门禁提交。未为本次盘点修改校验器、数据、模型或研究三对象；兼容 NAS 外部产物的路径策略仍需另行处理。

核查时 DAG SHA-256 为 `df36d1c9b93173596e02eb00e9a41396582fbfa17ea6e0ec7af2f9c26654192e`，Ledger 为 `1dc1dbcd4d632ec6ad1314f4454475a4d4541d478d675c837e1bfeeb84ef1420`。动态下载状态只描述本次读取时点；本地文件链接依赖工作区和 NAS 产物，Git 检出本身不包含大型结果。
