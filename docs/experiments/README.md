# 历史实验分析

本目录保留旧实验的设计判断、评分口径、结果分析和诊断快照。文中的方法建议、运行状态与目录约定属于记录时点；新研究从 [RESEARCH.md](../RESEARCH.md) 和 [AGENTS.md](../../AGENTS.md) 开始。旧结果经单独审计后才能纳入当前 Evidence Ledger。

| 文档 | 内容 |
|---|---|
| [EXP001 计划分析，2026-09-19](exp001-context-pair-xgb-plan-analysis-2026-09-19.md) | 历史设计评审，包含固定版本的原计划 |
| [EXP005 官方评估笔记](exp005-official-evaluation-notes.md) | 固定评分器版本、内存和执行边界 |
| [EXP005 靶点覆盖与先验](exp005-target-coverage-and-priors.md) | 当时官方面板与本地数据的覆盖核查 |
| [EXP006 K562 评估停滞诊断](exp006-k562-evaluation-stall.md) | 故障观测、诊断与当时的恢复验证状态 |
| [EXP007 跨背景 DE 分析](exp007-cross-context-de.md) | 配套 [统计与来源目录](exp007-cross-context-de/) 保留原始结果和哈希 |
| [EXP007 官方评分协议审计](exp007-official-scoring-protocol-audit.md) | 本地与线上评分条件、可比性及控制池审计 |
| [Leaderboard 分数解释](leaderboard-score-interpretation.md) | 2026-09-26 榜单快照和分数解释 |

单次实验的正式 PLAN、REPORT 和训练入口仍放在仓库根目录的 [experiments/](../../experiments/)。本目录中的旧产物链接依赖对应的本地历史输出；仅检出 Git 时可能不包含这些大型文件。

返回 [文档索引](../README.md)。
