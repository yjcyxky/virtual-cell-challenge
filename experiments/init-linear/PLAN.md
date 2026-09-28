# 初始线性响应基线

本 Experiment 管正则化线性响应路线；首个 run 为 `init-linear-s01`，比较 `C-BASE`，方法 `M-BASE`，证据问题 `E-BASE`。全部生效条件在 configs/s2-h1-s01.json 与 DAG 冻结，执行产物进入 outputs/init-linear-s01/。

## 问题与范围

先验证有效简单对照，分辨共享响应、拟合能力与扰动特异误差。首个独立拟合固定 S2-H1 / context-seen / seed=1：训练 K562、RPE1、HepG2、Jurkat 的所有允许靶点；H1 扰动标签完全留出，只允许 H1 输入 NTC。H1 与训练的两项研究来源分离，测量轴覆盖广，且有 25 个本届官方靶点可单独检验，因此先选此折；没有参考任何旧实验分数。

局部全目标面板 280 靶、官方重叠面板 25 靶分别评价，各自重建官方 anchors。该折不是五背景轮换、全局未见靶点或完整官方榜单；剩余折及 S1/S3/S4 必须另建 run。生物独立重复数未知，NTC bags、生成细胞、随机种子都不增加背景数。

## 输入、表示与拟合

- 直接重建已冻结 challenge_2026 映射、物理身份及 NTC 池，并逐输入核验 SHA-256、逐背景核对审计对象。训练只读 data/；缓存归本 run。
- 每背景在自身实测官方基因子集上，逐细胞归一化到 10,000，再 log1p；靶点 pseudobulk 为这些值的细胞均值。响应 Δ 是扰动均值减该背景输入 NTC 均值。不同测量面板导致归一化分母不同，作为该初始基线的明确局限保留。
- 输入含靶点 one-hot、NTC 均值的线性表示、NTC 靶基因值及测量标记。NTC 表示只使用四个训练背景共同实测的基因；此交集只限表示的可比坐标，训练输出和评分轴仍保留各背景完整测量掩码。四个训练 NTC 向量的全部 SVD 方向仅用于精确重参数化；不搜索秩、不用响应拟合表示。
- 响应头为 `a[target,gene] + z[context,target] @ beta[:,gene]`，包含截距，无 target×context 非线性交互。所有系数 L2 正则 alpha=1；每任务权重为细胞数/100，相当于 cell-weighted Δ-MSE。数据量和来源混杂保留为下一轮可归因问题。
- 按输出基因测量掩码求精确岭回归解；缺测不作为零监督。全部训练来源都没测到的基因固定 zero-Δ。分析解完成即正常训练结束，没有超参数搜索、验证早停或看分数重拟合。保存全部模型参数、表示状态、配置摘要及 analytic_fit_complete 状态；恢复时加载此完整状态。分析解提交前中断可重做确定性线性代数，不混入不同条件历史。

## 生成及参照

所有臂使用相同输入 NTC、每靶 400 个细胞、配对随机流（seed+固定靶点序号）。在 log1p(CP10K) 空间加 Δ，截断负表达，逆变换后恢复所采 NTC 的文库深度，以随机舍入生成非负整数 counts。zero-Δ 精确退化为原 NTC 重采样。

参照：zero 为零响应；shared 为只用训练背景、按细胞数加权的逐靶点共享 Δ；source 为预定 K562 逐靶 Δ，来源缺靶/缺基因固定零响应。source 的选择来自官方靶点覆盖事实。linear 为上述模型。这里的共享响应参照与评分器使用真实评估标签建立的 oracle baseline 各有用途，后者只能在 evaluator 内使用。

预测按官方 18,533 基因顺序保存。H1 输入没有测到的基因采用零基线/零响应完成输出，并在预测清单显式列出；这些值不作为观测真值，评分只选 H1 实测 18,074 基因。完整官方 A/B/C 输入上的提交属于后续独立任务。

## 官方评估与预定判据

固定 cell-eval2 `5e64833518a6603a0301cbe28185d49c30f4a986`、vcc2026 preset、官方 dispersed baseline 和五次 split-half anchors，运行配置 CUDA / gpudge、8 threads、pert_chunk=16。输入 NTC 与评分 NTC 物理隔离；所有臂与所有后续比较固定真实评价群体。每个面板重新建立匹配 bundle；不删退化指标、不替换锚点、不通过改群体挽救分数。

完成要求为两面板、四臂各有 raw 六指标、官方 normalized 六分项和 Overall；保存 raw per-target 结果、aggregate、run_meta、bundle、模型及完整预测。任一官方六分项无效视作评估阻塞，不能写成方法阴性结果。

主指标为全 280 靶的 S2.Overall。事前将 **0.02 normalized Overall** 定为值得进一步复验的实用差异，非估计的统计显著性阈值。linear 比三个参照中的最好者高至少 0.02、且官方重叠面板相对该参照未降超过 0.02，才记正向筛选线索；明显不满足则保留有效基线，搁置该实现的优越性主张。报告各分项/靶点原始配对差异；一个背景一个种子不能确认模型普遍优越。

本轮无论相对名次如何，只有全部训练与评分有效才能关闭 E-BASE；结论限于已实际执行的范围。下一步先查分项、测量覆盖、来源支持和效应强度，再决定 QC/数量覆盖控制或追加背景/种子；不得直接把小涨分当作换复杂模型的依据。

## 执行

`./reproduce.sh` 在环境同步前执行 research execute；`./reproduce.sh --resume` 恢复同一 run。W&B entity=yjcyxky、project=virtual-cell-challenge、group=init-linear、id=init-linear-s01。上传模型及结果清单，原始细胞数据不上传。依赖和源码提交后才正式启动。

## 本轮官方核查：C-BASE-OFFICIAL

用户于 2026-09-28 授权基于当前模型生成 VCC 并提交 leaderboard。追加同一 `init-linear-s01` 的推断/评估阶段，冻结 `checkpoints/ridge.npz`，不重新训练、不加入 H1 扰动监督、不校准响应幅度。完整条件在 `configs/official-s01.json`，DAG 节点的 `official_evaluation` 锁定代码和配置，账本问题 `E-BASE-OFFICIAL`。

两臂为当前 linear 与 checkpoint 内的 shared；A/B/C 分别使用全部官方 NTC 计算与训练相同的逐细胞 log1p(CP10K) 均值，映射到已训练表示。18,533 个基因按官方顺序保留，300 靶每背景每靶 400 细胞，共 360,000 细胞/臂。沿用 `predict` 和 `generate_counts`；每背景使用 seed=1+官方 CSV 靶点序号，跨臂配对。未见靶 linear 仅条件项、shared 零响应；所有训练背景均缺测的输出基因为零响应，官方 NTC 的真实基线保留。导出记录实有监督覆盖，不根据官方结果改变规则。

入口 `./submit.sh --submit` 在环境调用和输出前检查已完成模型、登记与已提交哈希；共用原 `.venv` 且 uv --locked --no-sync。复用已有来源校验、CSR 写入与全文件审计；外部官方 CLI 0.2.1 负责 prep/submit/status，不改训练环境。串行提交，保存文件哈希和 entry_id，失败沿用同一 entry 恢复，避免重复提交。同一 W&B run 的 official/linear 与 official/shared 分开记录，产物存于原 run 的 predictions/official-abc/。

完成条件为两臂官方 published 且六分项、Overall 有限，panel=vcc2026-val-1，保存实际 anchor_version、全部回执与版本化 Artifact。报告 H1 280 靶和官方重叠 25 靶与 A/B/C 的六项、Overall 和有符号差值；事前以绝对 Overall 差值<=0.02描述“接近”，同时报告排序，不作为统计等价检验。评价背景、靶点、实测基因轴和真实 anchors 同时变化，不能凭差距归因于实现错误，也不能凭接近证明协议等价；官方未提供 raw 时明确不可取得，不能反推冒充原始值。A/B/C 反馈只算开发证据。

## 固定预测的 baseline 归因：C-ANCHOR-AUDIT

沿用已完成 `init-linear-s01` 的评估身份，入口 `./anchor-audit.sh`；不重新训练。完整配置 `configs/anchor-audit-s01.json` 冻结两面板 reference、四臂预测与 raw aggregate/run_meta、原 bundles、源哈希和评分器。四条件为 `dispersed/tile × exclude_target_gene=true/false`，仅改变评分 baseline 的生成；这个排除开关不改变指标本身的 target exclusion。各条件 profile 使用该面板全部冻结扰动，不增加筛选。

每面板首个条件从相同 reference 通过官方接口重建 baseline 与五次 replicate anchors，后续通过官方严格内容缓存复用该 reference 的 anchors、分别重算 baseline。以官方 `score_metrics(real_bundle=..., user_meta=...)` 对原四臂 raw 结果统一缩放。原条件必须以绝对误差 1e-8 复现历史分项与 Overall；reference fingerprint、anchor 语义、seed、rule digest 必须一致，全部输入产物执行前后哈希不变。两个面板×四条件×四臂均取得六项有效分数才完成，失败不作方法阴性结论。

报告各条件 baseline raw、normalized 六项、Overall、有效数、linear−shared 差距，以及固定另一个因素的两项效应和二阶交互。实际官方分差不作为配方选择目标；[公开来源核查](../../docs/research/r4_anchor_audit.md)未找到 r4 构建记录，诊断仅证明本地标尺敏感性。敏感或不敏感均保留原默认协议，除非另有直接官方来源支持新配方；完成后转入已登记条件项消融。

登记沿用官方导出的追加阶段机制：DAG 中 `node.anchor_audit` 冻结配置、源码和前置证据，并由入口检查提交版本；比较对象在阶段执行完成后置 ready 并关闭，避免管理器把已完成拟合误认为本次追加评分已完成。
