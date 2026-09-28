# 文档索引

研发和训练约定以 [AGENTS.md](../AGENTS.md) 为准。设计新实验从当前研发入口开始；历史分析保留当时的判断和结果，纳入本轮证据前须另作审计。

## 当前研发入口

| 需要做什么 | 阅读位置 |
|---|---|
| 理解九轴、归因设计及研究管理命令 | [RESEARCH.md](RESEARCH.md) |
| 核对五背景、三来源研究及原始数据边界 | [数据依据](research/dataset_foundations.md) |
| 使用已核验的 2026 基因轴、NTC 划分和官方评分契约 | [challenge 数据与评估契约](research/challenge_protocol.md) |
| 查验方法和先验的论文依据 | [文献卡片](research/literature_foundations.md) |
| 登记候选方法 | [Method Space](research/method_space.json) |
| 登记执行、对照与前置关系 | [Experiment DAG](research/experiment_dag.json) |
| 查看依赖图 | [DAG 自动生成视图](research/experiment_dag.md) |
| 查验来源、记录实验结论及下一步 | [Evidence Ledger](research/evidence_ledger.json) |

`research/` 中的三份 JSON 是研究管理事实源；Markdown DAG 由工具生成。固定来源引用包含内容哈希，调整文档位置时须核对引用。

## 按用途查找

| 目录 | 内容与使用边界 |
|---|---|
| [datasets/](datasets/README.md) | 数据登记说明、下载审计及有日期的核查快照；原始输入和登记链路见 [data/README.md](../data/README.md) |
| [experiments/](experiments/README.md) | 历史实验设计、评分分析和故障诊断；其结果尚未自动纳入本轮账本 |
| ideas/ | [早期挑战摘要](ideas/challenge-overview.md)、[模块建模设想](ideas/initialized-thought.md)、[响应变量分析](ideas/variations.md)；保留的讨论材料，其中的推荐和推测不代表已验证结论或当前执行方案 |
| operations/ | [worktree 整合记录](operations/worktree-consolidation.md)；维护操作及当时路径 |

## 归档约定

- 单次训练的方案和结果放在 `experiments/<experiment_id>/PLAN.md` 与 `REPORT.md`；跨实验分析归入本文对应目录。
- 配套 CSV、JSON 与分析文档相邻保存；迁移时一起移动并修复调用方链接。
- 历史文档中的“当前”、运行状态、路径、数据量及权限只描述记录时点。历史审计 JSON 的原始路径和哈希保留；当前检出缺失的产物须明确标记。
- 新增或移动文档后更新相关索引并检查本地链接；当前研究来源变更还须通过 `scripts/research.py check`。
