# 已提交预测的 counts 分布与能力诊断

2026-09-30；comparison_id: `C-LEADERBOARD-COUNTS-AUDIT`。
KnowGraph attention: WL-UMI, WL-DEPTH, WL-KD-EFFICIENCY, ST-COUNT-MODELS, ST-LOG1P, ST-SAMPLING-NOISE, EVAL-MODE-COLLAPSE, EVAL-SYSVAR, EVAL-BIO-FIDELITY, PAP-EXP-ATTRIB。

本轮是用户请求的固定预测回顾分析。只读全部九份 published 回执所对应的预测，不训练、不重新生成、不重新提交、不改变官方评分或 anchors。当前登记模型为 init-linear-s01 的 linear/shared 与 masked-response-qc-s01；另外六份历史提交只作保留身份的回顾性描述，不回填初始化方法空间或声称单因素因果效应。沿用已有方法坐标与 run；DAG 在已完成节点追加分析阶段，无独立拟合、无新增训练 run。

三个问题：①计数合法且保留细胞异质性吗？②主要生成 NTC/共同偏移，还是存在扰动特异响应？③结合已有官方六项评分，哪些能力得到支持，哪些仍弱？

## 计算方案（读取预测矩阵前固定）

- 输入核验：重新计算 h5ad 与 VCC SHA-256，对照原生成/打包记录；保存 entry_id、panel、anchor 与全部 raw/normalized 原回执。原数据只读。
- 全量扫描：9提交×3背景×300扰动×400细胞×18,533基因。逐组统计有限/负数/非整数、空细胞、>1e6总数、存储元素数≤4.75e9、顺序及分组细胞数。硬约束任一不满足单列失败。
- 边际：文库与检出基因数1/5/50/95/99分位、CV、零率、深度—检出相关；NTC Wasserstein/均值。逐基因原始 Fano、CP10k及log1p(CP10k)方差比，CP10k参照均值≥0.1、方差>1e-8；<0.1/ >10只是极端诊断区间，不是生物学拒绝阈值。
- 异质性：组内完全重复行率、与输入NTC完全相同行率；128个NTC最高log方差且排除全部300靶基因的协方差向量相关和相对误差。高相似支持继承NTC结构，不能证明学到扰动后协方差。
- 响应：同时报告平均log表达差及先汇总CP10k再log2比（伪计数0.01 CP10k=1 CPM）；两者不可混用。排除全部300靶基因后，计算共享centroid能量占比、去centroid参与比有效秩、第一PC占比与靶间余弦；几何量未去噪，不能当作可重复生物信号。
- 靶基因：NTC≥5CPM为可估计组，报告剩余表达比、中位数、下降/≤0.5/上升>1.2比例；低表达单列。CRISPRi敲低是经验性方向预期，不要求所有细胞或所有靶点必然下降。人工敲低先验不得解释为学习结果。
- 对照与抽样：每背景全量18,400输入NTC；seed=20260930+背景索引，30次各抽取两个不相交400细胞子样本，计算同n诊断和独立split RMS。重复抽样不是生物重复。正式预测使用全体输入NTC，因此这些只是经验噪声标尺，不是独立隐藏评分NTC。
- 解释：原六项官方raw/normalized与Overall全部保留。NTC偏离不等于错误，预测vsNTC的DE不能叫伪DE。仅零响应生成/NTC null可检验伪DE，本轮不重生成旧模型；真实DE能力使用既有官方回执。

预算为所有九份完整矩阵一次扫描、三个NTC池与30次null；按target流式处理，不建立新环境。使用init-linear已有uv锁环境。产物保存于原提交目录的 `counts-audit-20260930/`；聚合保存在原init-linear run cache。所有提交都保留，不能按分数删选。结束条件：全九份扫描成功、哈希通过、逐背景及逐靶产物完备、解释边界与研究登记关闭、研究check通过。缺失/失败如实报告；不以子样本扫描代替全量合法性检查。

执行：`micromamba run -n virtual-cell uv run --locked --project experiments/init-linear -- python scripts/analyze_submitted_counts.py`。分析脚本具有后续提交复用价值，不是新增训练入口；不修改任何冻结训练实现。
