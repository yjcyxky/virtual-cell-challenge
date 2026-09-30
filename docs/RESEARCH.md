# VCC 2026 研究管理

本仓库使用 **Method Space + Experiment DAG + Evidence Ledger** 驱动研究。规则在 [AGENTS.md](../AGENTS.md)，数据对象由 Git 管理，训练记录和 Artifacts 继续使用 W&B。

## 初始化依据与研究假设

2026-09-27 的初始化依据是[分享对话](https://chatgpt.com/share/6ab9c3e3-56cc-83e9-bbef-b1f196a8e83c)中的九轴框架、[五背景原始数据审计](research/dataset_foundations.md)、[已发表原始论文](research/literature_foundations.md)和官方任务说明。方法候选不从已有实验反推，初始图不导入旧实验，初始账本没有本项目模型效果。旧文件和结果保留在原处及 Git；后续若要纳入，须另作明确的回顾性审计。

分享的核心问题是：独立背景很少，如何用可信数据、可检验先验和有效表示约束假设空间，使模型学到能迁移的响应。`Baseline + Δ_shared + Δ_context + residual` 是可拆解的主干假设；加性空间、逆变换和生成计数必须明确，不能直接把原始 counts、log 表达与 logFC 相加。先验错误或压缩过度也可能降低表现，因此保留绝对表达、简单线性、无先验和匹配随机表示等反证对照。

分享中的方法效果数字、模型排名及“主干”组合均为说明性例子，不是研究结果。九轴完整保留，但各轴不一定正交：例如 CVAE 与 Flow 往往联动架构、损失及采样，须登记整包比较；V 是冻结评价契约。机制整包搜索按明确资源配额先行，是否扩展调参或拆解由其实际价值决定，不把归因证据作为所有新机制的准入条件。

| 轴 | 初始覆盖与关键反证 |
|---|---|
| T 任务 | 绝对表达、平均差、细胞差、logFC、条件分布、平均响应加残差；目标和逆变换联动须明示 |
| D 数据 | 单来源、合并、背景/靶点平衡、伪配对、课程；新增可靠性过滤、等量随机删减、软权重，区分质量与覆盖 |
| R 表示 | NTC 均值/方差、集合编码、预训练、基因程序、多视图；保留 PCA、随机程序、随机初始化与扰动基因注释消融 |
| A 架构 | 线性、小 MLP、Transformer、CVAE、Flow、响应头加生成器；零/共享响应作为同次评估的参照 |
| L 目标 | count likelihood、响应 MSE、方向、logFC、分布距离、多目标；只在训练内计算响应及权重 |
| O 优化 | 单阶段、预训练微调、两阶段、课程、困难样本、多来源加权；先固定预算，后研究优化瓶颈 |
| V 验证 | ID、扰动 OOD、背景 OOD、联合 OOD、研究留出构成 Pseudo-VCC，随机细胞划分只作诊断 |
| I 推断 | 确定性、NB/Flow 采样、方差/文库校准、集成；固定响应模型再比较生成策略 |
| G 泛化 | 无条件、背景拼接、FiLM、不变学习、仅 NTC 适配、检索/专家；禁止使用留出背景的扰动标签 |

五背景来自三个研究组：H1；K562 GWPS/RPE1；HepG2/Jurkat。原始元数据说明细胞量、靶点面板及测量基因轴严重不平衡，不能把五背景视作五份独立研究，也不能把更多 bags/种子当更多背景。所有覆盖数字的映射口径及局限见数据审计；初始优先级来自这些设计约束与文献争议，不来自模型分数。

Pseudo-VCC 先冻结五背景轮换留出及全局靶点留出；每折只能看目标背景 NTC，拟合表示、QC 阈值和超参数只用训练边界。H1 三个文件中重复 NTC 必须按物理身份去重后再划池；输入和评分 NTC 分离。研究留出按三个来源组整体移除，同组两个背景不能冒充独立跨研究验证。2026 官方基因轴与靶点面板是任务目标；每背景按官方顺序保留全部已测基因及缺测掩码，局部评分覆盖单列。五背景共同交集仅可作为明确登记的诊断，不替代官方任务轴；未测量不补成监督零或评分真值。协议、数据映射与划分的当前状态以 DAG 及其冻结引用为准；训练须另行满足节点配置和执行门禁。

2026 官方任务使用未知背景的 NTC 和 CRISPRi 靶点；验证与最终背景不同。当前提交每背景每靶点 400 个细胞，覆盖官方 18,533 基因轴并输出非负整数 counts；六项 normalized 指标等权聚合。本地须同时保存 raw 分项、逐背景/靶点结果及 reference/scorer 身份，局部基准分数不能直接当榜单分数；细胞采样波动与训练随机性分别记录。[官方任务](https://arcinstitute.org/news/virtual-cell-challenge-2026)、[官方 CLI 与评分要求](https://vcc-cli-wiki.virtualcellchallenge.org/#submission-requirements-2026)

初始 DAG 只预留有具体对照的 draft 槽位；零响应等无需训练的参照随基线执行评估。**每次独立拟合占一个 run，同路线共用 Experiment**：ready 前须按外层划分、全局靶点分区及 seed 展开实际节点，填写 `design.fit_scope` 和一致的 `expected_config.fit_scope`。一个节点不能装入五折训练；比较层汇总各折，控制按相同 fit_scope 和 seed 配对。

QC 的配对 seed 2/3 确认分支已预留，但只有首轮 signal/supported 才能进入；其余确认与交互等待具体证据后展开。`design` 是待落实方案，不能冒充完整 `expected_config`。单 seed 只产生线索，优化稳定性和跨背景一致性需各自的重复与分层证据。

Set Encoder 初筛同时引入可训练表示与非线性容量，只能归因于表示整包；有信号后须配容量匹配的统计量 encoder。FiLM 与 shared+context 分解分别登记比较；后者还要固定共享项与残差的可辨识约束。生成器可在背景调制没有正收益时继续成为合理问题，不能把某一结构获胜设为所有后续研究的通行证。

## 2026 官方任务对齐

本轮数据契约配置见 [challenge_protocol.json](research/challenge_protocol.json)。处理代码复用共享 H5AD 读取工具，直接核验原始输入，不读取历史实验结果。具体映射、测量掩码、物理 NTC 划池、S0–S4 划分及官方 scorer 配置通过 DAG 的 `binding` 引用；训练实际配置必须携带同一 `benchmark`。

官方输出始终以完整基因轴和当前面板为目标。本地公开真值不全时，使用该背景真实测量的官方子轴并报告覆盖；其归一化分数只具有本地参考尺度。实际评分使用固定版本 cell-eval2 的六指标、baseline 和 anchor 接口；阈值不随模型成绩调节。参考和 anchors 在对应 Experiment 内按冻结配方生成并保存哈希；数据审计完成不代替实际评分或模型结论。

## 已完成模型的官方核查

已完成拟合的 A/B/C 推断与远端评分属于原 run 的评估阶段。保留原 metrics、研究绑定和已关闭证据；在节点的 `official_evaluation` 中登记独立阶段配置、代码引用、checkpoint 来源和 `protocol_audit` 比较，账本先登记 pending 问题。阶段入口在写产物和连接 W&B 前核验来源哈希、Git 已提交内容、环境和生成规则；这不是另一次拟合，也不重新调用训练入口。

当前实现为 `cd experiments/init-linear && ./submit.sh --submit`，配置由 `configs/official-s01.json` 和 DAG 绑定；不带 `--submit` 仅生成及校验。该入口复用原模型/生成函数、全文件审计和官方 CLI，沿用原 W&B run。两臂顺序执行，同一 VCC 的断点恢复查找既有 entry，不重复创建提交。只有官方发布有效六分项及 Overall 后才关闭核查证据；上传成功、launching 或评分失败均不能当作完成评分。完整回执与 VCC 保存为版本化 Artifact。

本地 H1 与官方 A/B/C 的背景、靶点、测量轴、真实群体及 anchors 不同，核查输出同时报告六项 raw、normalized、Overall、分数差和排序。回执未提供的逐背景数据、有效数和精确 anchors 明确为未知。更新 REPORT 时将此前关闭证据的 REPORT 引用固定到旧 Git 版本，保留旧结论；官方反馈作为新的开发证据入账，不覆盖原本地结论。

固定预测的 baseline 敏感性诊断使用 `cd experiments/init-linear && ./anchor-audit.sh`。节点的 `anchor_audit` 登记完整配置、代码、来源和前置证据；执行前检验提交版本，恢复时核对阶段身份，结果追加到原 run。比较对象在追加评分完成后置 ready 并 close，不能把已完成 fit 当成本次评分完成。当前管理器尚不原生区分 fit 与追加阶段，阶段门禁由上述入口执行；直接管理命令仍需人工核对阶段结果，不能把这一边界说成全路径强制保护。

入口固定原 raw aggregate/run_meta，调用官方 real-bundle 和 score 接口；原条件须复现，两个面板分别重建匹配 anchors，之后仅通过官方严格内容缓存复用同一 reference。所有对照统一缩放，原指标保留。核查线上 baseline 的构建条件先读 [r4 来源卡片](research/r4_anchor_audit.md)，采用新默认配方需要直接来源，不能按与榜单的距离挑选。

## 机制整包搜索与资源分配

2026-09-30 的用户修正将本轮优先级转向[任务约束、数据处理与本地能力评估](research/task_constraints_review.md)：先让训练观测、计数生成和官方评价相匹配，并证明预测具有可迁移的扰动特异信息，再做榜单核查。弱模型的结果限定为当前实现记录；不据此认定生物瓶颈。当前队列以 Ledger 为准，集合分布 draft 暂不分配训练预算；DepMap 监督对齐基线作为下一轮建模候选，仍为未验证设计。

寻找 SOTA 包含两个不同决策：先判断哪类机制值得投入，再判断该机制中哪些因素带来收益。用户明确允许不同机制的整包探索；数据与官方评价正确性已成立后，不把先在线性头上获胜、逐因素拆完或证明所有先验作为复杂模型的准入条件。

每轮在三对象中登记最多三个优先机制问题，明确每条路线改变的 T/D/R/A/L/O/I/G 条件及共同的 V、数据/划分/输出轴。多个联动变化用 `integrate`，结论限于完整方案。首轮给每条一个有明确结束条件的完整训练配置和相同正式评估，不按第一条的成绩调整其余候选。候选都完成后，按冻结门槛、主/官方重叠表现及成本分配下一轮：有价值的整包优先独立背景/种子确认及必要消融；无信号则更换机制或重审瓶颈，避免自动转入该路线的参数网格。完整训练预算是实验条件，不能以预检、smoke 或中途停止替代。

下一轮比较应同时写出机会成本：该路线能改变哪些预测、覆盖多少官方靶点、与其他路线有何机制差异，以及为何值得本轮计算。结果可能支持任务、架构、目标与生成器的联动；拆解可以后置，不将整包涨分归因于其中某一个因素。旧的条件化/表示阴性只约束其实现；既有 draft 是候选库，当前队列可以根据用户指示和新证据重新排序。

## 先看当前决策

在仓库根目录执行：

```bash
./scripts/research.py status
./scripts/research.py check
./scripts/research.py graph
```

`status` 区分外部来源、待检验问题和本地闭环证据，并给出优先问题与启动缺项；`check` 校验对象；`graph` 从登记节点和关系生成 Mermaid 图。
[当前 DAG 图](research/experiment_dag.md) 是生成视图，修改对象后用 `./scripts/research.py graph > docs/research/experiment_dag.md` 更新，不手工维护连线。

## 三个实际对象

| 对象 | 内容与写入时机 |
|---|---|
| [method_space.json](research/method_space.json) | T/D/R/A/L/O/V/I/G 九轴的明确候选、方法坐标和先验；设计方案时登记 |
| [experiment_dag.json](research/experiment_dag.json) | 精确实验节点、方法、协议、实际配置、控制/来源关系、结果引用，以及带判据的比较；训练前登记，执行后回填 |
| [evidence_ledger.json](research/evidence_ledger.json) | 文献/数据/协议来源与本地实验分类型登记；保存适用范围、下一步与重开条件，维护最多三个优先问题 |

Method 的 `local_status=UNTESTED` 表示在本轮协议上尚未验证。账本 `literature/external` 是作者原任务的观察；`dataset/observed` 是输入事实；`protocol/constraint` 是任务约束；`experiment/pending` 是待检验问题。只有完成本地比较后才产生 `signal / supported / not_supported / inconclusive`，外部论文不能替代一次本地确认。

DAG 的 `controls` 表示科学比较，`sources` 表示复用来源，`requires_evidence` 表示决策前置；`evidence_conditions` 进一步限定允许的结论状态。外部证据和 pending 均不能放行，确认通常需要 signal/supported，交互需要 supported。普通对照边本身不强制串行执行。
节点与配置的哈希引用保留追溯，研究结论只在 Ledger 维护。REPORT 保存详细分析，本文件不复制一份证据排行榜。

## 建立一次比较

1. 在 Method Space 选择已有坐标或添加有明确差异的候选，写清先验注入位置与消融。
2. 在 DAG 建立 comparison：假设、用途、控制/候选节点、变化轴、实际配置变化字段、主指标/方向/实际意义阈值、适用范围，以及支持/反对/证据不足各自的下一步。
3. 为每次独立训练登记新 run 节点。draft 可不完整；准备启动时补齐 `expected_config`、配置/代码引用、冻结协议和 PLAN，改为 ready，提交这些条件。
4. 运行 `check`、`gate <node_id>`。相同方法标签不能掩盖实际配置改变；单因素比较须通过实际差异校验。协议不同的比较登记为 `protocol_audit`，不解释为模型增益。

`expected_config` 是训练解析默认值与所有覆盖后的完整配置，而非少数手工摘录字段。`changed_fields` 使用 dotted paths，例如 `model.max_depth`；单因素的 control/candidate 训练种子匹配。配置中的身份字段变化不算方法改动。
代码引用覆盖该 Experiment 的源码、入口、依赖声明与锁文件，共享依赖另外显式引用。运行入口检查相关文件已提交且内容与登记一致。

比较类型：`screen` 发现线索，`confirm` 验证预设范围，`interaction` 检查四臂协同，`integrate` 只评价组合，`protocol_audit` 检查协议敏感性，`baseline` 建立对照。
V 轴是冻结评估定义，不作为可刷分参数；真实/随机/无先验与等维表示用于区分结构信息、压缩和容量作用。

## 接入真正的训练入口

新 Experiment 的 `reproduce.sh` 使用 `execute` 包住完整环境检查、输入处理、训练和评估命令；它内部先运行 gate，持有实验锁，退出后读取结果回填 DAG。
`execute` 后的命令不得再次调用同一个 `reproduce.sh`，避免递归。已有环境启动代码仍负责使用 virtual-cell 与该 Experiment 的锁定 uv 环境。

```bash
# 结构示例，不是新增实验配置；<node_id> 必须与实际登记一致。
"/home/jy001/micromamba/envs/virtual-cell/bin/python" \
  ../../scripts/research.py execute <node_id> -- <完整流水线命令及参数>
```

训练模块解析全部配置后、创建输出和 W&B 之前执行：

```python
# ROOT 指仓库；config 是已解析全部默认值/覆盖的实际配置。
sys.path.insert(0, str(ROOT / "scripts"))
from research import bind
research_metadata = bind(ROOT, run_id, config, resume=resume)
frozen_config["research"] = research_metadata
# 将同一 frozen_config 写入 outputs/<run_id>/config.yaml 并传给 wandb.init。
```

完成时 `outputs/<run_id>/metrics.json` 保存下列字段，以及比较声明的有限数值主指标和其他科学指标：

```json
{
  "status": "completed",
  "research": "这里实际保存 bind 返回的对象",
  "evaluation_completed": true,
  "checkpoint_ref": {"path": "相对仓库的实际模型文件", "sha256": "实际哈希"},
  "predictions_ref": {"path": "相对仓库的预测文件或内容清单", "sha256": "实际哈希"}
}
```

这只是字段示意，字符串占位符不会通过结果校验。预测为多个大文件时引用带内容哈希的清单，不复制或逐项上传缓存。
`execute` 不会凭退出码 0 宣称完成：缺有效评估、模型、预测或绑定身份会拒绝完成登记。失败记录执行失败；科学结论仍由证据关闭步骤给出。
中断后显式传 `--resume`，训练命令也必须接入已有完整 checkpoint 恢复。管理器不加载模型/优化器，不替代训练状态恢复。
`bind` 在 `outputs/<run_id>/cache/research-binding.json` 留下绑定标记；绑定后失败即使没有最终指标，`execute` 也会保存带身份的失败记录，不能按一次新训练覆盖。修复故障、清除已解决的 blocker 并提交后，再恢复原训练状态。
若环境/输入检查在 `bind` 前失败，可运行 `retry <node_id>`，提交后重新执行同一入口；它只接受条件未变且没有绑定标记或训练产物的执行，并保留失败记录。改变实验条件仍须新建身份。

未登记的实验源码、配置和入口变更会被提交检查拦截。历史恢复从其记录的代码版本进行；初始化不修改旧训练源码、输出或 W&B 记录。

同一 Experiment 增加新 run、演进共用源码时，已完成节点可设置 `code_snapshot_commit` 为其 metrics 中的实际 execution commit（未记录 execution 时用 research commit）。校验在该提交读取旧 `code_refs`，并核对已执行登记与原 research 绑定；不修改旧引用、metrics 或科学条件。尚需恢复的未完成节点不能使用快照；放弃条件并换独立 run 的失败执行按下一节归档。标准入口不恢复已完成节点。历史推断/评估需检出各阶段记录的代码提交；新 run 仍须绑定当前源码、配置并通过全部启动门禁。

### 失败执行归档

修正科学条件需新 run，而同一路线仍共用 Experiment 源码/环境。保留失败节点的状态、metrics、原配置、code_refs 和 research 绑定；登记 `superseded_by`（同 Experiment 的不同新 run）、具体 `superseded_reason`，并以失败记录的实际 execution/research commit 设置 `code_snapshot_commit`。新 run 使用独立比较和完整配置；原比较及待检验账本问题保留失败原因与替代入口，不关闭为方法阴性。方法本身可不变，但变更的输入边界和预算必须重新冻结；不能声称修复版与旧版同时冻结。

校验只接受有完整原绑定且 `evaluation_completed=false` 的 failed/interrupted 记录，核对同路线替代节点、原提交引用和无替代环；门禁拒绝旧身份的 restart/resume。普通未完成 run 没有替代登记仍须恢复完整 checkpoint。这个机制用于保存失败历史并推进新条件，不把失败改成完成，也不允许在修复登记中改写旧科学条件。

### 评估接口修复

完整训练和预测已保存、评估因接口故障未完成时，可沿用原 run 修复适配代码。先保存 `status=failed/interrupted`、`evaluation_completed=false` 的失败快照；在节点登记 `evaluation_repair`：`kind=evaluation_only`、原因、原提交的 `original_code_refs`、准确的 `changed_paths`、快照 `failure_ref`，以及模型、预测清单、每个预测文件和已完成参考 bundle 的 `preserved_refs`。更新当前 `code_refs`，清除已解决 blocker，通过校验并提交后执行 `--resume`。

门禁校验原科学登记、配置、环境和入口不变，逐项核对保留产物哈希。训练的 `research` 仍绑定原提交；修复入口把 `execution_metadata(root, node_id)` 写入最终 metrics 的 `execution` 及本地/W&B config 的执行历史，回收时核验修复在该实际提交上已登记。模型与扰动预测不重算，评分器、reference、anchors、阈值保持冻结。代码审查仍须确认改动确属接口修复；该机制不自动证明算法等价，也不适用于更换评分公式或改写已有有效结果。

## 关闭证据并决定下一步

自动回收使用 `record` 的同一核验路径；已有符合新结果格式的执行也可手动导入：

```bash
./scripts/research.py record <node_id> --metrics <实际metrics.json>
./scripts/research.py close --file <本次证据JSON>
./scripts/research.py status
```

证据文件包含 `id`、`comparison_id`、`node_ids`、`state`、`claim`、`limitations`、`source_refs`、`next_action`、`reopen_when`；证据正文存入 Ledger，临时输入文件使用系统临时目录并清理。
本地结果状态为 `signal / supported / not_supported / inconclusive`。可用同一 ID 将 pending 问题关闭成结果，已经关闭的证据不可覆盖。`close` 检查执行完成、引用及必要决策字段；不会自动将三次生成采样认定为三个生物重复，也不会自动判定科学支持成立。
失败执行先修复或如实保留失败原因，不能用它关闭“方法无效”的证据。证据不足是一种有下一步的结论。
自动回收不冻结正在撰写的 REPORT；可在训练后完善分析再关闭证据。已入账的报告引用须保留原版本，后续修改前将旧引用固定到含相同内容的 Git commit；不要改写旧证据来迎合新结论。

配对效应、逐背景/靶点子集、raw/normalized 六项指标、实际独立样本数与不确定性在 REPORT 中展开。更换评估协议须重新建立可比对照；旧分数保留。

## 提交检查与约束范围

`.githooks/pre-commit` 在现有 LFS 检查前调用 `check --staged`；检查 Git index 中的对象，不因工作区有未暂存修复就放行。
本仓库通过 `.git/hooks/pre-commit` 链接启用，保留 git-lfs 的其他 hooks；新克隆可执行：

```bash
ln -s ../../.githooks/pre-commit .git/hooks/pre-commit
```

提交检查约束实验的登记、标准入口与训练模块接入；完成全部比较臂却没有本地证据及下一步决策时也拒绝提交。启动门禁进一步检查实际配置、哈希和前置证据类型/结论。
这些是标准路径的技术约束，不是针对恶意绕过的安全沙箱。它们能阻止漏登记和不一致，不能证明一个生物假设正确。未接入的新实验不能被标为 ready。
来源笔记和本地结果按哈希验证；已经入账的内容更新时保留原引用版本。本工具不自动下载产物或改写线上 W&B。

来源阅读顺序：设计方法先读九轴与上述边界；涉及数据/拆分时读 [dataset_foundations.md](research/dataset_foundations.md)；采用论文方法或先验时读 [literature_foundations.md](research/literature_foundations.md) 对应条目，再登记可证伪对照。

## Experiment 与 run 的当前身份

按用户当前协议，Experiment 是稳定研究路线，DAG 节点是一次独立 run（node_id = run_id）。同路线各 run 共用源码、配置、锁文件和环境，执行产物只写 `experiments/<experiment_id>/outputs/<run_id>/`，W&B group=experiment_id、id=run_id。旧节点没有 run_id 时按历史根目录布局只作兼容，历史结果不迁移。独立 fit_scope 不得混入同一 run。
