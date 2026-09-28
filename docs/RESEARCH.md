# VCC 2026 研究管理

本仓库使用 **Method Space + Experiment DAG + Evidence Ledger** 驱动研究。规则在 [AGENTS.md](../AGENTS.md)，数据对象由 Git 管理，训练记录和 Artifacts 继续使用 W&B。

## 先看当前决策

在仓库根目录执行：

```bash
./scripts/research.py status
./scripts/research.py check
./scripts/research.py graph
```

`status` 给出当前未关闭比较、优先问题、每个实验缺少的条件及下一步；`check` 校验对象；`graph` 从实际节点和关系生成 Mermaid 图。
[当前 DAG 图](research/experiment_dag.md) 是生成视图，修改对象后用 `./scripts/research.py graph > docs/research/experiment_dag.md` 更新，不手工维护连线。

## 三个实际对象

| 对象 | 内容与写入时机 |
|---|---|
| [method_space.json](research/method_space.json) | T/D/R/A/L/O/V/I/G 九轴的明确候选、方法坐标和先验；设计方案时登记 |
| [experiment_dag.json](research/experiment_dag.json) | 精确实验节点、方法、协议、实际配置、控制/来源关系、结果引用，以及带判据的比较；训练前登记，执行后回填 |
| [evidence_ledger.json](research/evidence_ledger.json) | 结论、证据文件哈希、适用范围、下一步与重开条件；比较结束后关闭证据，维护最多三个优先问题 |

历史节点已经从实际 config、metrics、停止记录及 REPORT 回填。`legacy=true` 与回顾性登记保留历史身份，不能把历史方案补写成预注册。
零响应和共享响应若是同次执行内的评估基线，保留为该节点结果，不虚造独立训练节点。
exp008 只有需求草案，登记为 draft 并列出缺项；当前机制不擅自补选其尚未确定的对照，也不创建或启动训练。

DAG 的 `controls` 表示科学比较，`sources` 表示数据/模型/代码来源；`requires_evidence` 表示启动前必须有结论。三者不同，候选不因为对照尚未训练就被自动串行化。
节点与配置的哈希引用保留追溯，研究结论只在 Ledger 维护。REPORT 保存详细分析，本文件不复制一份证据排行榜。

## 建立一次比较

1. 在 Method Space 选择已有坐标或添加有明确差异的候选，写清先验注入位置与消融。
2. 在 DAG 建立 comparison：假设、用途、控制/候选节点、变化轴、实际配置变化字段、主指标/方向/实际意义阈值、适用范围，以及支持/反对/证据不足各自的下一步。
3. 为每次独立训练登记新节点。draft 可不完整；准备启动时补齐 `expected_config`、配置/代码引用、冻结协议和 PLAN，改为 ready，提交这些条件。
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
research_metadata = bind(ROOT, experiment_id, config, resume=resume)
frozen_config["research"] = research_metadata
# 将同一 frozen_config 写入 Experiment 的 config.yaml 并传给 wandb.init。
```

完成时 `metrics.json` 保存下列字段，以及比较声明的有限数值主指标和其他科学指标：

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
`bind` 在 `cache/research-binding.json` 留下绑定标记；绑定后失败即使没有最终指标，`execute` 也会保存带身份的失败记录，不能按一次新训练覆盖。修复故障、清除已解决的 blocker 并提交后，再恢复原训练状态。
若环境/输入检查在 `bind` 前失败，可运行 `retry <node_id>`，提交后重新执行同一入口；它只接受条件未变且没有绑定标记或训练产物的执行，并保留失败记录。改变实验条件仍须新建身份。

历史入口已纳入冻结检查，不原地改成新身份。历史恢复从其记录的代码版本进行；本次未修改旧训练源码、输出或 W&B 记录。

## 关闭证据并决定下一步

自动回收使用 `record` 的同一核验路径；已有符合新结果格式的执行也可手动导入：

```bash
./scripts/research.py record <node_id> --metrics <实际metrics.json>
./scripts/research.py close --file <本次证据JSON>
./scripts/research.py status
```

证据文件包含 `id`、`comparison_id`、`node_ids`、`state`、`claim`、`limitations`、`source_refs`、`next_action`、`reopen_when`；证据正文存入 Ledger，临时输入文件使用系统临时目录并清理。
状态为 `signal / supported / not_supported / inconclusive`。`close` 检查执行完成、引用及必要决策字段；不会自动将三次生成采样认定为三个生物重复，也不会自动判定科学支持成立。
失败执行先修复或如实保留失败原因，不能用它关闭“方法无效”的证据。证据不足是一种有下一步的结论。
自动回收不冻结正在撰写的 REPORT；可在训练后完善分析再关闭证据。已入账的报告引用须保留原版本，后续修改前将旧引用固定到含相同内容的 Git commit；不要改写旧证据来迎合新结论。

配对效应、逐背景/靶点子集、raw/normalized 六项指标、实际独立样本数与不确定性在 REPORT 中展开。更换评估协议须重新建立可比对照；旧分数保留。

## 提交检查与约束范围

`.githooks/pre-commit` 在现有 LFS 检查前调用 `check --staged`；检查 Git index 中的对象，不因工作区有未暂存修复就放行。
本仓库通过 `.git/hooks/pre-commit` 链接启用，保留 git-lfs 的其他 hooks；新克隆可执行：

```bash
ln -s ../../.githooks/pre-commit .git/hooks/pre-commit
```

提交检查约束新实验的登记、标准入口与训练模块接入，并防止修改被冻结的历史训练入口；完成全部比较臂却没有证据及下一步决策时也拒绝提交。启动门禁进一步检查实际配置、哈希和前置证据。
这些是标准路径的技术约束，不是针对恶意绕过的安全沙箱。它们能阻止漏登记和不一致，不能证明一个生物假设正确。未接入的新实验不能被标为 ready。
历史报告按固定 Git 版本读取，本地结果按哈希验证；迁移工作区后须先从登记的固定 Artifact 恢复所需历史文件。本工具不自动下载产物或改写线上 W&B。

方法论依据：[原分享对话](https://chatgpt.com/share/6ab9c3e3-56cc-83e9-bbef-b1f196a8e83c)；VCC 任务依据与历史评分限制保留在 Ledger 的固定来源和各 Experiment REPORT。
