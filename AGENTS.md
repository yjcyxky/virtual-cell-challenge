# VCC 2026 研发与训练协议

## 研发目标与执行入口

- 由目标背景的 NTC 细胞群和扰动信息预测扰动后表达分布，泛化到未见背景与扰动组合。明确 `Baseline + Δ` 的表达空间；共享响应、背景修正和功能模块都是待检验先验。
- 小样本指独立背景、研究及扰动组合不足。先修复数据/评估正确性，再研究数据质量、先验和表示；模型容量、目标函数和优化由已识别瓶颈驱动。
- **开始任何研发任务，先执行** `/home/jy001/micromamba/envs/virtual-cell/bin/python scripts/research.py status`，按输出处理未闭环比较或最高优先问题。命令与接入示例见 [RESEARCH.md](docs/RESEARCH.md)。
- **三份对象是研究管理的唯一事实源**：[Method Space](docs/research/method_space.json) 定义九轴候选及方法坐标；[Experiment DAG](docs/research/experiment_dag.json) 登记实际执行、对照、来源、协议和比较；[Evidence Ledger](docs/research/evidence_ledger.json) 保存有来源的结论、决策与下一步。不要另维护手工实验列表或仅在文字里声明已登记。

## 必须完成的研究闭环

1. **登记**：先选可证伪问题。在 Method Space 定义方法，在 DAG 建立比较与实验节点；写清对照、全部实际变化、评估协议、完整生效配置、代码/配置引用、种子、结束条件和结果判据。PLAN 解释理由并引用 `comparison_id`。新实验一训练一身份；历史节点标记 retrospective/legacy，保留原始目录与 W&B ID。
2. **校验**：执行 `scripts/research.py check`。单因素比较必须同时通过方法坐标和实际配置差异检查；评估协议不同只作协议诊断。对照关系不等于数据/权重来源，也不自动成为执行前置；需要先有结论时显式声明 `requires_evidence`。
3. **启动**：新 `reproduce.sh` 通过 `scripts/research.py execute <node_id> -- <训练命令>` 启动，在环境同步/训练前检查登记、冻结引用、已提交代码及前置证据。训练模块解析全部配置后、创建输出/W&B 前调用 `bind(root, node_id, actual_config)`，把返回的 `research` 同时写入实际 config 和 W&B config；CLI 覆盖和默认值不得偏离登记。
4. **回收**：`execute` 自动核验并登记根目录 metrics 的状态与结果引用。完整训练结果须包含 `research`、有效主指标、`evaluation_completed`、`checkpoint_ref` 和 `predictions_ref`。训练失败与未完成不能成为方法阴性证据；绑定前失败用 `retry` 检查同条件重试，绑定后用 `--resume` 恢复完整训练状态，管理命令不能代替 checkpoint 恢复。
5. **结论与决策**：REPORT 给出比较、配对效应、波动与限制，再用 `scripts/research.py close --file <证据JSON>` 将证据和采用/确认/补对照/搁置决定入账。账本包含下一步与重开条件；`status` 将未关闭比较列为待办。工作区或W&B有分数但未入账，不算研究闭环完成。
6. **提交**：执行 `check --staged`；仓库 pre-commit 同样检查暂存区三对象和新增训练入口。校验失败须修正对象/入口，不能绕过 hook 交付。历史训练代码保持冻结，恢复从其记录的 Git 版本进行。

上述约束覆盖标准启动和提交路径，不能替代科学判断，也不声称能阻止故意绕过入口执行任意代码。维护协议或管理工具时同步更新必要行为测试。

## 归因与小样本边界

- 九轴 T/D/R/A/L/O/V/I/G 是方法覆盖图，不遍历笛卡尔积。V 是冻结协议；先验注明假设、注入位置、来源版本、匹配消融与失效条件。
- 单因素按实际条件差异判断；联动无法拆开时只归因于整包。交互用 `Base/A/B/A+B`；集成只证明整体。正确性修复先建立有效对照，不为消融保留已知错误。
- 记录独立背景、研究、靶点、guide/生物重复与细胞数。NTC bags、生成细胞及种子不增加独立生物背景；少背景展示逐背景结果，不以细胞级显著性外推跨背景泛化。
- 区分正确性处理与可选 QC/重加权；报告保留率、覆盖和响应强度分层。缺测、实测零与非显著分别处理；guide/batch 差异不默认全部是技术噪声。
- 真实先验与无先验、匹配随机先验或等维表示比较。响应特征排除被预测背景标签；表示拟合遵守训练边界，留出 NTC 适配预先声明。
- 比较核对数据/划分、基因/靶点面板、输入/评分 NTC、reference、预处理、scorer、baseline/anchors 与采样。新本地基准隔离输入/评分 NTC；换协议须重建对照，保留历史结果。
- 同时报 raw 六指标、normalized 分项、Overall、配对效应和有效数；区分训练与生成随机性、已见与未见靶点、context/perturbation/joint OOD。H1 或 A/B/C 用于选择后只能称开发证据。
- 固定响应比较生成器，或固定生成器比较响应模型；检查效应偏移和 NTC 伪 DE。分布校准涨分不等于生物响应更准。
- 单次小涨分仅为线索，未支持只限当前实现与范围。现有 PLAN 的用户约定优先，不自动扩大实验范围；smoke test 不代替训练。每轮最多三个优先问题，以决策影响、信息增量和成本排序。

以下训练协议管理执行、环境和产物；以上研究闭环管理为何做、结果说明什么及接下来做什么。

原则：**一次 Experiment 就是一次完整训练试验，也是唯一的一次 run。** Experiment 统一管理该次训练的研究方案、代码、配置、环境、训练过程和结果；Git 管版本，W&B 管训练记录；交付包含有效训练、评估、结果总结和代码清理。

## Experiment

- **Experiment** 是一次独立、完整、可复现的训练试验，使用唯一的 `experiment_id`。一个 Experiment 对应一个本地训练目录和一个 W&B run，不再在 Experiment 下设置多个 runs。
- 输入校验、预处理、训练和评估是同一 Experiment 的连续步骤，共用同一输出目录和 W&B run，不为不同阶段创建额外 Experiment。
- 阶段切换、同配置下的校验修复、临时故障重试和断点续训沿用原 Experiment，追加日志并保留已有记录。恢复训练须恢复 checkpoint 中完整训练状态，不能只恢复 W&B 日志或模型权重。
- 从头重训，或改变数据、模型、训练配置等会影响实验结果的条件时，应新建 Experiment。修复会使已有训练结果失效的代码后，也应创建新的 Experiment 重新训练，不得将不同实验条件的训练历史混在同一个 Experiment 中。
- 如果只是不会影响已有结果有效性的工程修复，可以继续原 Experiment；若无法明确判断，应优先新建 Experiment，以保证结果可追溯。
- 历史 Experiment 的结果及其评估口径保持不变。比较不同 Experiment 时，应明确说明数据、划分、指标、模型和训练条件等差异。独立复现已有结果时创建新的 Experiment，并记录来源 Experiment。

## 目录职责

```text
data/                         # 已登记的共享数据，训练期间只读
models/                       # 固定版本的第三方模型及权重
scripts/                      # 有持续用途的通用数据处理与维护工具
docs/                         # 数据说明、调查结论和跨实验文档

experiments/<experiment_id>/
  PLAN.md                     # 研究问题、方法、基线和评估方案
  REPORT.md                   # 本次实验结果、结论与局限
  src/                        # 正式训练、评估流程所需模块
  tests/                      # 必要的功能验证
  configs/                    # 本 Experiment 使用的配置
  pyproject.toml
  uv.lock
  .venv/                      # 本 Experiment 独立环境
  reproduce.sh                # 完整训练入口，支持恢复

  config.yaml                 # 实际生效配置及数据、代码、环境版本引用
  train.log
  metrics.json
  checkpoints/
  predictions/
  cache/                      # 本次预处理结果及临时文件
  wandb/
```

目录和产物按需创建。Experiment 根目录同时承担代码、配置、依赖、研究文档和本次训练产物的管理。

禁止在 Experiment 内再建立 `runs/`、`outputs/<run_id>/`、`preflight/`、`preparation/` 等第二套实验或阶段管理目录。输入校验、预处理、训练和评估只是同一 Experiment 的流程阶段，而不是独立的 run。

## 代码保留与主动清理

- **先复用，再新增**：修改前查找已有实现，优先更新所属模块。新文件应承担当前需要的独立职责；调参用配置，修复更新原实现，避免积累替代版本。按职责组织模块，不按操作次数或调查步骤拆文件。
- **保留当前需要的代码**：`src/` 中的模块须服务于正式训练或评估流程，包括必要的数据处理与校验。脚本之间相互引用不构成保留理由，应追溯到实际功能。`scripts/` 同样要求明确的持续用途，不能用来转存一次性脚本。
- **临时代码临时执行**：一次性查询、排障和探索使用命令行或系统临时目录，用后清理。需要持续复现的数据转换整理为正式模块；调查结论记入文档。
- **主动删除**：每次交付和提交前，检查本次新增、修改及被替代的代码，删除临时脚本、重复实现、废弃分支，以及随之失效的配置和测试；同步更新引用与文档。清理范围同时包含本次产生的临时代码和本次修改导致失效的旧实现。历史实现由 Git 保存。
- **验证清理结果**：保留内容必须对应当前功能、必要测试或明确的复现需求；“以后可能有用”不是保留理由。完成与改动影响相匹配的验证后再交付。其他 Experiment 的文件、已登记数据和历史实验产物不属于此次清理范围。

## 环境与版本

- 每个 Experiment 使用一套 uv 管理的 `.venv/`，不同 Experiment 不共享虚拟环境。
- 基础 Python、uv 和原生库使用已有 micromamba 环境 `virtual-cell`，保持其版本稳定，不删除或重建。
- uv 经 `micromamba run -n virtual-cell` 调用，显式绑定该环境的 Python，并使用 `--no-python-downloads`。不回退到系统 Python，不启用系统 site-packages，不把 `UV_PROJECT_ENVIRONMENT` 指向基础环境。
- Agent 自行通过 `uv add/remove/lock` 声明和锁定实验依赖，执行时使用 `uv sync --locked`、`uv run --locked`；不绕过声明安装依赖，不在执行过程中自动修改 lock 文件。
- 有活动训练时，不修改该 Experiment 正在使用的代码、输入或环境。若需要进行会改变训练条件的修改，应停止原实验并创建新的 Experiment。
- 代码、配置模板、`pyproject.toml`、`uv.lock` 和研究文档由 Git 管理，按可独立理解的逻辑变更主动提交。
- 正式训练使用已提交的代码和锁文件，并自动记录 Git commit。
- 无需为 Experiment 再复制一套源码快照；历史复现通过对应 Git commit、依赖锁和数据版本重建运行条件。

## 完整训练流程

1. **明确方案**
   在 Experiment 的 `PLAN.md` 中确定研究问题、数据与划分、基线、训练方案、评估指标，以及训练结束条件，例如既定轮数或正常早停。该 Experiment 的实验条件通过配置明确表达。

2. **统一入口**
   ```bash
   cd experiments/<experiment_id>
   ./reproduce.sh
   ```
   `reproduce.sh` 应完成环境检查、输入校验、预处理、训练和评估，并支持从该 Experiment 的 checkpoint 恢复，不要求手工激活环境。

3. **自动记录**
   在 `config.yaml` 中保存实际生效配置、种子、数据及划分的固定版本引用、Git commit、依赖锁版本、关键运行时版本，以及来源 Experiment（如适用）。本地配置与 W&B 使用一致的 Experiment 身份和配置。

4. **输入与输出归位**
   共享数据只读；本次 Experiment 产生的预处理结果、切分、特征和缓存写入当前 Experiment 的 `cache/` 或相应正式输出目录。

   复用其他 Experiment 的模型、预测或预处理产物时，应引用明确版本并记录来源，不覆写被引用产物。输入缺失或版本不符时修复来源，不静默替换数据或放宽评估标准。

5. **训练与恢复**
   正常训练过程中持续保存日志、指标及必要 checkpoint。若支持断点训练，checkpoint 必须保存恢复训练所需的完整状态，包括模型、优化器、scheduler、训练步数以及其他必要状态。

6. **评估与结论**
   保存用于正式评估的模型、预测和指标，按照 `PLAN.md` 中预先确定的划分和评价方法完成评估，并与基线或来源 Experiment 比较。

   将实验状态、主要结果、比较结果、结论、局限和 W&B 链接整理到 `REPORT.md`。

   如果验证集、测试集或 leaderboard 反馈被用于后续模型选择，应明确记录其开发用途，避免混淆最终评估和模型开发过程。

Experiment 完成要求：按照既定方案完成有效训练与评估，保存可追溯结果，更新 `REPORT.md`，并完成代码清理。

效果低于基线也属于有效实验结果。预检通过、环境就绪、仅完成 smoke test 或耗尽 Agent 自设预算均不算 Experiment 完成。确实无法继续时，应如实记录失败状态和阻塞原因，不宣称实验完成。

## W&B 与产物

- 每个 Experiment 对应且仅对应一个 W&B run。
- 统一使用：
  - `entity="yjcyxky"`
  - `project="virtual-cell-challenge"`
  - `group` 可用于标记更高层次的研究方向或实验系列
  - `id=experiment_id`
  - W&B run 名称包含 `experiment_id`
- 不再设置独立 `run_id`。
- 输入校验、预处理、训练、验证和测试通过指标前缀、step 或字段区分，而不是创建多个 W&B runs。
- 记录配置、版本、训练/验证/测试指标、step、运行状态和资源使用。
- W&B 本地文件写入当前 Experiment 的 `wandb/`。
- 断网时使用本地记录，恢复网络后同步，并在 `REPORT.md` 中说明同步状态。
- 保存用于正式评估的模型；支持断点训练时保存最近的完整训练状态。
- 关键模型和结果上传为版本化 W&B Artifacts。复用其他 Experiment 的 Artifact 时引用固定版本，并记录来源 Experiment。
- 普通缓存无需逐项归档。
- `.venv/`、缓存、大型训练产物不进入 Git；密钥、凭据和未经授权的原始数据不得上传。

## Experiment 身份原则

为了避免 Experiment 与 run 两层概念造成管理歧义，本项目统一采用：

**1 Experiment = 1 完整训练 = 1 W&B run = 1 独立结果单元**

只有不会改变实验有效条件的恢复、重试和执行阶段才继续原 Experiment；任何需要重新从头训练、改变结果生成条件或形成新的独立比较结果的情况，都创建新的 `experiment_id`。
