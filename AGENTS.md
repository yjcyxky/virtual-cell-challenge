# VCC 2026 研发与训练协议

## 研发目标与执行入口

- 由目标背景的 NTC 细胞群和扰动信息预测扰动后表达分布，泛化到未见背景与扰动组合。明确 `Baseline + Δ` 的表达空间；共享响应、背景修正和功能模块都是待检验先验。
- 小样本指独立背景、研究及扰动组合不足。先修复数据/评估正确性，再研究数据质量、先验和表示；模型容量、目标函数和优化由已识别瓶颈驱动。
- **初始化依据**：本轮方法空间与前瞻 DAG 以[分享九轴方案](https://chatgpt.com/share/6ab9c3e3-56cc-83e9-bbef-b1f196a8e83c)、五个原始数据集和已发表原始研究建立，不由现有实验、分数或既定模型路线填充。初始方法均为 `UNTESTED`、执行节点均为 `draft`；之后通过实际归因实验积累本地证据。旧结果保留，回顾性纳入另作审计并注明用途。
- **设计前必读**：[研发依据与轴边界](docs/RESEARCH.md#初始化依据与研究假设)；涉及输入/划分时读[五背景数据事实](docs/research/dataset_foundations.md)，采用论文方法/先验时读[文献卡片](docs/research/literature_foundations.md)。H1、K562 GWPS、RPE1、HepG2、Jurkat 是五背景、三来源研究；元数据覆盖不等于 QC 后监督量。
- **2026 官方对齐**：输入映射、模型输出和提交以已登记官方 `gene_names.csv` 的完整基因轴、顺序及 `pert_counts.csv` 为目标。各来源保留测量掩码；本地只在该背景真实测得的官方轴子集评分，报告覆盖，不把缺测补零作为监督或评分真值，不默认将任务缩为五背景交集。辅助靶点可用于训练，官方面板覆盖须单列。
- **官方评估**：使用 DAG 冻结的官方 `cell-eval2` commit 和 `vcc2026` preset，调用其 baseline、real-bundle/anchor 和评分接口。每个参考群体重新生成匹配 anchors；未定义指标与退化参照按官方规则处理，不能自行改公式、删指标或借用旧 anchors。实际配置的 `benchmark` 必须匹配 DAG 协议的 `binding`，否则不能启动。协议冻结不等于已取得六项有效分数。
- **开始任何研发任务，先执行** `/home/jy001/micromamba/envs/virtual-cell/bin/python scripts/research.py status`，按输出处理未闭环比较或最高优先问题。命令与接入示例见 [RESEARCH.md](docs/RESEARCH.md)。
- **三份对象是研究管理的唯一事实源**：[Method Space](docs/research/method_space.json) 定义九轴候选及方法坐标；[Experiment DAG](docs/research/experiment_dag.json) 登记实际执行、对照、来源、协议和比较；[Evidence Ledger](docs/research/evidence_ledger.json) 保存有来源的结论、决策与下一步。不要另维护手工实验列表或仅在文字里声明已登记。
- **证据类型有边界**：`literature/external`、`dataset/observed`、`protocol/constraint` 提供来源依据；`experiment/pending` 是问题，均不代表本地方法有效。只有真实完成的比较可关闭为本地证据；文献、待检验假设不能满足训练前置。`evidence_conditions` 指定继续分支所需结论，失败/证据不足按预定分支处理。

## 必须完成的研究闭环

1. **登记**：先选可证伪问题。在 Method Space 定义方法，在 DAG 建立比较与实验节点；写清对照、全部实际变化、评估协议、完整生效配置、代码/配置引用、种子、结束条件和结果判据。草案 `design` 不能代替 `expected_config`；按外层划分×靶点分区×seed 为每次独立拟合登记唯一 `run_id`，冻结一致的 `fit_scope`；同一研究路线共用 Experiment 代码和环境。PLAN 引用 `comparison_id`；初始 draft 不创建训练目录或 W&B run。
2. **校验**：执行 `scripts/research.py check`。单因素比较必须同时通过方法坐标和实际配置差异检查；评估协议不同只作协议诊断。对照关系不等于数据/权重来源，也不自动成为执行前置；需要先有结论时显式声明 `requires_evidence`。
3. **启动**：新 `reproduce.sh` 通过 `scripts/research.py execute <node_id> -- <训练命令>` 启动，在环境同步/训练前检查登记、冻结引用、已提交代码及前置证据。训练模块解析全部配置后、创建输出/W&B 前调用 `bind(root, node_id, actual_config)`，把返回的 `research` 同时写入实际 config 和 W&B config；CLI 覆盖和默认值不得偏离登记。
4. **回收**：`execute` 自动核验并登记对应 `outputs/<run_id>/metrics.json` 的状态与结果引用。完整训练结果须包含 `research`、有效主指标、`evaluation_completed`、`checkpoint_ref` 和 `predictions_ref`。训练失败与未完成不能成为方法阴性证据；绑定前失败用 `retry` 检查同条件重试，绑定后用 `--resume` 恢复完整训练状态，管理命令不能代替 checkpoint 恢复。
5. **结论与决策**：REPORT 给出比较、配对效应、波动与限制，再用 `scripts/research.py close --file <证据JSON>` 将证据和采用/确认/补对照/搁置决定入账。账本包含下一步与重开条件；`status` 将未关闭比较列为待办。工作区或W&B有分数但未入账，不算研究闭环完成。
6. **提交**：执行 `check --staged`；仓库 pre-commit 同样检查暂存区三对象和新增训练入口。校验失败须修正对象/入口，不能绕过 hook 交付。历史训练代码保持冻结，恢复从其记录的 Git 版本进行；未完成评估的接口修复按 [评估修复登记](docs/RESEARCH.md#评估接口修复) 保留原训练绑定、模型和预测哈希，另记实际执行版本。改变科学条件或使已有有效结果失效仍须新 run。

上述约束覆盖标准启动和提交路径，不能替代科学判断，也不声称能阻止故意绕过入口执行任意代码。维护协议或管理工具时同步更新必要行为测试。

## 归因与小样本边界

- 九轴 T/D/R/A/L/O/V/I/G 是方法覆盖图，不遍历笛卡尔积。V 是冻结协议；先验注明假设、注入位置、来源版本、匹配消融与失效条件。
- 单因素按实际条件差异判断；联动无法拆开时只归因于整包。交互用 `Base/A/B/A+B`；集成只证明整体。正确性修复先建立有效对照，不为消融保留已知错误。
- 记录独立背景、研究、靶点、guide/生物重复与细胞数。NTC bags、生成细胞及种子不增加独立生物背景；少背景展示逐背景结果，不以细胞级显著性外推跨背景泛化。
- 区分正确性处理与可选 QC/重加权；报告保留率、覆盖和响应强度分层。缺测、实测零与非显著分别处理；guide/batch 差异不默认全部是技术噪声。
- 真实先验与无先验、匹配随机先验或等维表示比较。响应特征排除被预测背景标签；表示拟合遵守训练边界，留出 NTC 适配预先声明。
- 比较核对数据/划分、基因/靶点面板、输入/评分 NTC、reference、预处理、scorer、baseline/anchors 与采样。新本地基准隔离输入/评分 NTC；换协议须重建对照，保留历史结果。
- 同时报 raw 六指标、normalized 分项、Overall、配对效应和有效数；区分训练与生成随机性、已见与全局未见靶点、context/perturbation/joint OOD。五背景轮换与研究留出分别报告；任何已用于选择的本地背景或 A/B/C 只能称开发证据。
- 固定响应比较生成器，或固定生成器比较响应模型；检查效应偏移和 NTC 伪 DE。分布校准涨分不等于生物响应更准。
- 单次小涨分仅为线索，未支持只限当前实现与范围。现有 PLAN 的用户约定优先，不自动扩大实验范围；smoke test 不代替训练。每轮最多三个优先问题，以决策影响、信息增量和成本排序。

## Experiment、run 与训练协议

- **Experiment 管研究路线，run 管一次完整训练。** 核心建模方法、架构或训练算法重大变化新建 Experiment；调参、工程修复和数据选择留在原 Experiment。从头重训或改变影响结果的条件，创建新的 run_id。DAG 节点对应独立 run，node_id = run_id。
- 代码、配置模板、PLAN.md、REPORT.md、pyproject.toml、uv.lock、.venv/ 和 reproduce.sh 放在 `experiments/<experiment_id>/`。所有执行产物放在 `outputs/<run_id>/`：config.yaml、train.log、metrics.json、checkpoints/、predictions/、cache/、wandb/。run 不另建源码、环境、PLAN、REPORT 或阶段项目。
- 输入校验、预处理、训练和评估共用一个 run、输出目录和 W&B run。阶段切换、临时故障重试和同条件恢复沿用原身份，追加日志；恢复完整 checkpoint 训练状态。改变数据、模型、训练配置或使已有结果失效的修复，新建 run。历史结果与评估口径保持不变。
- 旧目录与旧记录保持原身份。兼容旧入口不授权新任务使用旧的一次 Experiment = 一次 run 规则。

## 环境、版本与代码清理

- 每个 Experiment 使用一套 uv 管理的 `.venv/`，本路线各 run 共用，不同 Experiment 不共享。基础 Python、uv 和原生库使用已有 micromamba `virtual-cell` 环境；不删除、重建或改变基础环境。
- uv 经 `micromamba run -n virtual-cell` 调用，显式绑定其 Python，使用 `--no-python-downloads`。通过 uv add/remove/lock 声明依赖，执行只用 `uv sync --locked`、`uv run --locked`。有活动 run 时不修改其代码、输入或环境。
- 正式训练前提交代码、配置和锁文件，自动记录 Git commit、锁文件哈希及运行时版本。共享 data/ 只读；来源缺失或不匹配时修复来源。缓存归本次 run，复用产物记录固定来源和哈希，不覆盖历史产物。
- 修改前先找已有实现。src/ 服务正式训练/评估；scripts/ 只放有持续用途的通用工具。一次性查询用命令行或系统临时目录。每次交付前主动删除本次临时代码、重复实现、废弃分支及失效配置/测试，验证并更新引用；其他任务、登记数据和历史产物不在清理范围。
- `.venv/`、缓存、大型产物不入 Git，密钥、凭据及未经授权的原始数据不上传。

## 完整执行与交付

1. PLAN 写清问题、数据/划分、基线、训练方案、评估和结束条件；run 差异以完整配置表达，登记到 DAG。
2. `cd experiments/<experiment_id> && ./reproduce.sh` 贯通环境、输入、预处理、训练和评估；通过 run_id 选择/恢复，先执行研究门禁再启动环境。
3. 实际 config.yaml 保存种子、数据和划分引用、Git commit、锁文件哈希、运行时版本、来源 run 和 research 绑定；与 W&B 配置一致。
4. 保存评估模型、预测、指标与必要完整训练状态；按预定方案训练到完成并评估。REPORT 比较 runs、记录状态、局限和 W&B 链接；测试反馈参与选择时标为开发证据。
5. 关闭 Ledger 证据，记录下一步与重开条件，完成清理、验证和提交。低于基线也是有效结果；预检、环境就绪、smoke 或耗尽 Agent 自设预算不算完成。确实不能继续时如实记录失败/阻塞，不写方法阴性结论。

## W&B

- `entity="yjcyxky"`、`project="virtual-cell-challenge"`、`group=experiment_id`、`id=run_id`；名称含 run_id，本地与线上一一对应。阶段用指标前缀或字段区分。
- 记录完整配置、版本、训练/验证/测试指标、step、状态和资源使用；本地文件写入该 run 的 wandb/。断网本地记录，恢复后同步并说明状态。
- 关键模型和结果上传版本化 Artifacts；复用固定版本。普通缓存不逐项归档，原始细胞数据不作为公开 artifact 上传。
