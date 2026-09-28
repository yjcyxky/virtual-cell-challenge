# exp007：跨 context 差异表达与两跳知识覆盖

日期：2026-09-27。仅作真实数据的探索性分析；没有训练、修改模型或提交评分。H1 及其余背景已用于开发，本分析也属于开发反馈。

## 方法与来源

- H1 使用 exp007 run `20260927-exp007-h1-log2fc-s17` 的真实 DE 缓存；K562 使用 exp006 run `20260927-exp006-dispersed-s17`；HepG2/Jurkat 使用 exp005 run `20260926-exp005-nested-residual-s17`。
- 逐背景核验旧缓存对应的 native genes、合格细胞数/靶点数/NTC 数、来源文件、evaluation-rows.npy 和 cells.parquet 与 exp007 一致，四份 DE 参数完全相同。旧实验的模型或基线评分不参与本分析。
- RPE1 没有已有 DE 缓存。本次从 exp007 的固定 evaluation-rows.npy 与 counts.npy 取 206,279 个细胞、8,684 个原生基因，调用固定版本 cell_eval2.de_compute.compute_de：backend=gpudge、reference=non-targeting、mean_calc=arithmetic、epsilon=1e-9、input_type=counts、target_sum=1e6、clip_value=None、filter_gene_min_cpm_cell=5、fdr_scope=per_pert、threads=6、device=cuda。得到 13,955,832 行 DE 表。
- 所有背景均为逐细胞 CPM 算术均值 log2FC、Wilcoxon、每扰动 BH 校正、p_adj<0.05、NTC 平均 CPM>5；排除自身靶基因。附加效应门槛为 |log2FC|>=0.5。
- 每个扰动最多 400 个真实细胞，NTC 使用该背景全部合格细胞。细胞数、NTC 池、测序深度、原生基因面板及实验来源不同，不能将检出差异完全归因于 context 生物学。
- 两两比较只使用共同合格靶基因及双方都进入 DE 检验的 readout 基因。沿用各背景原生基因面板内的 BH 结果，不在共同子集重新校正；CPM 也沿用各背景原生基因轴。
- Jaccard=交集/并集；小集合覆盖=交集/min(DEG_A,DEG_B)。双方空集时 Jaccard 记缺失，不记为 100%；小集合为空时小集合覆盖记缺失。表中 Jaccard、小集合覆盖为逐扰动中位数。方向一致率在双方均显著的所有 perturbation-gene 对上合并计算。
- 随机重合预期：对每个扰动，E=|A|*|B|/共同受检基因数，再跨靶点相加。未控制表达量、网络度数或基因间依赖，不作因果证明。

## 各原生面板的 DEG 数量

各行靶点组成不同，仅作数据描述，不是控制变量后的背景比较。

| Context | 靶点数 | 受检基因中位数 | 扰动细胞数中位数 | DEG 中位数 | DEG 比例中位数 | 加绝对 log2FC≥0.5 后 DEG 中位数 |
|---|---:|---:|---:|---:|---:|---:|
| H1 | 297 | 10758 | 400 | 716 | 6.656% | 31 |
| K562 | 9319 | 8134 | 185 | 1 | 0.012% | 1 |
| RPE1 | 1608 | 8678 | 97 | 313 | 3.607% | 150 |
| HepG2 | 1059 | 9478 | 74 | 183 | 1.931% | 89 |
| Jurkat | 1811 | 8754 | 99 | 83 | 0.948% | 32 |

## 同一靶基因的两两跨背景重合

| Context 对 | 共同靶点 | 共同 readout | Jaccard 中位数 | 小集合覆盖中位数 | 交集方向一致率 | 重合/随机预期 |
|---|---:|---:|---:|---:|---:|---:|
| H1–K562 | 266 | 7093 | 0.48% | 55.05% | 76.91% | 1.22 |
| H1–RPE1 | 80 | 7557 | 4.18% | 57.48% | 74.81% | 1.23 |
| H1–HepG2 | 37 | 7983 | 1.18% | 68.48% | 77.02% | 1.34 |
| H1–Jurkat | 83 | 7547 | 1.45% | 60.00% | 73.73% | 1.17 |
| K562–RPE1 | 1527 | 7061 | 5.26% | 43.08% | 89.31% | 1.94 |
| K562–HepG2 | 994 | 7373 | 5.35% | 43.25% | 91.20% | 1.91 |
| K562–Jurkat | 1733 | 7293 | 8.33% | 42.55% | 93.43% | 2.35 |
| RPE1–HepG2 | 836 | 7686 | 6.31% | 56.07% | 92.19% | 1.66 |
| RPE1–Jurkat | 1377 | 7329 | 7.92% | 46.41% | 88.76% | 1.78 |
| HepG2–Jurkat | 905 | 7575 | 8.02% | 43.16% | 88.09% | 1.73 |

Jaccard 低不等于完全不共享响应。H1–K562 共同面板上，逐靶点 DEG 中位数分别为 523.5 和 5；较小集合覆盖中位数为 55.05%，但 Jaccard 中位数只有 0.48%。检测功效及集合大小差异显著影响该指标。
加绝对 log2FC≥0.5 后，两两 Jaccard 中位数约 1.56%–4.58%；双方强效应 DEG 的合并方向一致率约 86.56%–96.97%。不同 context 对的共同靶点集合不同，表格不能解释为严格的 context 相似性排名。

## 五背景共同面板

五个背景共有 32 个合格靶点、5946 个共同受检 readout。靶点列表：ARPC2, DNAJA3, DNMT1, DPH2, EHMT2, EIF4B, ELAC2, GSK3B, HIRA, HMGCS1, HSP90B1, JAZF1, MBTPS1, MED19, METTL3, NCAPH2, NELFE, NISCH, PGAM1, PHF10, PIAS1, PTPN1, SDC1, SETDB1, SMAGP, SMARCB1, TADA1, TAF11, TAF13, TFAM, THAP11, USF2。

- 至少一个 context 显著的 perturbation-gene 对：77,708。
- 仅一个 context 检出显著：59,241（76.24%）。
- 五个 context 都显著：511（0.66%）。
- 加绝对 log2FC≥0.5 后，仅一个 context 检出占 81.11%，五者皆检出占 0.43%。
- 这里“只检出于一个 context”不等于证明其余 context 的真实效应为零。共同 32 靶点的扰动细胞数中位数：H1 400、K562 280.5、RPE1 140.5、HepG2 70、Jurkat 142.5。

## 同一共同面板上的两跳知识覆盖

复用 exp007 的固定 STRING>=700、Reactome、CollecTRI 缓存，所有来源组成联合图，允许异类边跨两跳连接。STRING 无向，CollecTRI 沿出边；Reactome 共享通路作一步，未推断因果方向。中间节点允许当前 19,300 基因轴中的全部基因，不要求在目标 context 表达。两跳包含一跳，排除靶基因自身。

在同一靶点和同一 readout 轴上，静态知识可达集合跨 context 完全相同。本共同面板平均每靶点覆盖 4,772.5 个基因，占可评估空间 80.28%；以下差别来自真实 DEG 集合，不来自知识边随背景变化。

| Context | 实际 DEG 两跳覆盖 | 随机预期覆盖 | 实际/预期 |
|---|---:|---:|---:|
| H1 | 83.71% | 82.38% | 1.016 |
| K562 | 88.24% | 84.74% | 1.041 |
| RPE1 | 88.15% | 83.46% | 1.056 |
| HepG2 | 88.00% | 82.16% | 1.071 |
| Jurkat | 88.33% | 84.99% | 1.039 |

这里使用共同 32 靶点、5,946 readout；此前 H1 的 76.98% 使用全部 297 个 H1 靶点与 H1 原生受检面板，两个比例的分母不同。

## 结论与建模含义

静态两跳知识提供较宽的候选集合，不能直接决定同一靶点在不同背景下实际响应的基因。跨背景显著集合重合有限，但共同检出的 DEG 方向通常一致，支持存在可迁移的响应成分，同时需要背景信息校准响应强度与范围。显著性受到样本量与检验功效影响，不能将非显著视为精确零标签。

## 产物与复算

- [原生背景统计](exp007-cross-context-de/context-summary.csv)
- [两两重合统计，两种效应门槛](exp007-cross-context-de/pair-summary.csv)
- [逐靶点重合明细](exp007-cross-context-de/per-target-overlap.csv)
- [共同面板知识覆盖，五种知识来源、一跳/两跳](exp007-cross-context-de/knowledge-common-panel.csv)
- [共同靶点与 readout 清单](exp007-cross-context-de/common-panel.json)
- [来源、固定参数、校验记录与结果哈希](exp007-cross-context-de/sources.json)

复算先按上述来源读取 DE 表，以原始缓存的 target/feature 对齐；过滤 feature==target 后按 p_adj<0.05（可追加绝对 log2FC 门槛）构造集合，再按上面的分母与公式聚合。知识矩阵使用 Boolean 可达性 A∨A²；联合图的 Reactome 分量通过 P·Pᵀ 构建，其中 P 为 gene-pathway membership。一次性分析代码与 RPE1 中间 DE 文件位于系统临时目录，汇总后清理；没有向正式训练源码添加分析脚本，也没有改动历史 run 产物。
