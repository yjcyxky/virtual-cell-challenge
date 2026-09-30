# Perturbation decomposition v1：原文核验与 VCC 适用边界

日期：2026-09-30。本文是原始文献与代码核验记录；不是新模型实验结果，不替代 Method Space、Experiment DAG 或 Evidence Ledger。

**决策摘要：需要改变现有实现。最优先的模型候选是 DepMap 监督响应映射；必须落实的基础改动是观测空间与计数生成一致、完整轴测量掩码、严格背景/靶点留出。保留已见靶点的 Shared 作为对照，不再把它与论文模板等同，也不把旧 Ridge/程序/CVAE 的失败当成该论文路线已被否定。** §7–9 给出本地比较、改动边界和现有队列的具体调整。

## 1. 阅读范围与来源

- 论文：Alexis Molina、Xinyi Zhang，*Perturbation response decomposition enables biologically aligned generalization to unseen perturbations and cellular contexts*，bioRxiv v1，2026-07-27，DOI `10.64898/2026.07.24.740459`，未经同行评议。标题、作者、日期、版本由 PDF 首页及 [bioRxiv API](https://api.biorxiv.org/details/biorxiv/10.64898/2026.07.24.740459)交叉确认。
- 已成功下载并读取 [44 页官方 PDF](https://www.biorxiv.org/content/10.64898/2026.07.24.740459v1.full.pdf)，包括 Methods 和 Supplementary Notes；视觉检查 Fig. 1、2、4。最初标准 URL 返回 403/429，官方 early 路径及加空查询串的 PDF 请求成功。临时本地 PDF 为 `/tmp/perturbation-decomposition-v1.pdf`；SHA256：`d34e545d0f50ad7e4369126f50dad8965be5a0dad2ac4e0e62c76261b8eb6f5c`。页码以下均为 PDF 页码。
- 辅助核验 [作者仓库](https://github.com/xinyizhanglab/perturbation-decomposition/tree/a15214780619736d393f40240e56ba992fd416a3)，固定 commit `a15214780619736d393f40240e56ba992fd416a3`。原文与代码不一致处分别记录；不假定仓库每个脚本都是 v1 图表最终运行版本。
- 关键位置：主结果 pp. 2–4；Fig. 1–4 pp. 5–8；讨论 p. 9；预处理 pp. 10–12；模型 pp. 13–15；分解 pp. 19–22；归因 pp. 26–28；补充 pp. 30–44。

## 2. 核心思路：先区分预测的响应成分，再判断输入能支持什么

### 2.1 三个容易被混淆的操作

论文的响应是每个背景、靶点的 **mean(log1p(CP10K)) 差值**：

\[
\delta_{c,p,g}=\operatorname{mean}_{i\in(c,p)}\log(1+10^4x_{ig}/L_i)
-\operatorname{mean}_{i\in(c,NTC)}\log(1+10^4x_{ig}/L_i).
\]

它不是先汇总 counts 再 log 的 pseudobulk 组成差，也不是整数细胞矩阵。原文把这个均值称为 pseudobulk。方法引用：[§4.1.8–4.2.1，p. 11](https://www.biorxiv.org/content/10.64898/2026.07.24.740459v1.full.pdf#page=11)。

**A. 平衡 ANOVA 分解**：在四背景共有的 638 个扰动、共同 2,000 基因上，

\[
\delta_{c,p}=\mu+\alpha_c+\beta_p+\gamma_{c,p}.
\]

其中 \(T_c=\mu+\alpha_c\) 是该背景跨扰动均值，\(\beta_p\) 是跨背景保守的靶点效应，\(\gamma_{c,p}\) 是背景与靶点交互。平衡设计下四个展开张量在 Frobenius 内积上正交；这不是说每个靶点的四个基因向量都正交。作者用真实标签做事后分解，没有把这些含目标标签的分量输入预测器。[§4.8.2，p. 20；§4.10.2，p. 27](https://www.biorxiv.org/content/10.64898/2026.07.24.740459v1.full.pdf#page=20)。

**B. 模板方向投影**：用训练扰动均值 \(T_c\)，对每个响应计算

\[
\tau_{c,p}=\frac{\delta_{c,p}^{\top}T_c}{T_c^{\top}T_c},\qquad
\epsilon_{c,p}^{\perp}=\delta_{c,p}-\tau_{c,p}T_c.
\]

这会去掉各扰动在同一模板方向上的不同振幅。它与 A 的减去固定 \(\mu+\alpha_c\) 不同；投影残差也不能直接叫作 ANOVA 的 \(\gamma\)。[§4.8.1–4.8.3，pp. 19–21](https://www.biorxiv.org/content/10.64898/2026.07.24.740459v1.full.pdf#page=19)。

**C. Perturbed-reference Pearson**：

\[
r_{pref}=\operatorname{corr}_g(\widehat\delta_p-T_{ref},\delta_p-T_{ref}).
\]

它减去同一固定质心，没有执行 B 的逐响应投影。因此 `pref` 为正不等于正交残差恢复为正；三种口径必须分列。原文 p. 20 明确区分了 B/C。[§4.2.3，p. 12](https://www.biorxiv.org/content/10.64898/2026.07.24.740459v1.full.pdf#page=12)。

### 2.2 哪些发现可作为本地实验的先验

| 原文发现 | 精确证据 | 可借鉴的实验，而非直接结论 |
|---|---|---|
| 模板方向占 full-response energy 的 23%–65%，背景差异明显 | Fig. 1d p. 5；Table 2 p. 35 | 同时评价完整响应、模板振幅和正交残差；不能用相似总体应激代替靶点特异恢复 |
| 跨背景全张量分解为 template 27.8%、保守扰动 29.4%、交互 23.5%、noise 19.3% | Fig. 1c p. 5；§4.8.4 pp. 21–22 | 用可重复性校正诊断，不将观察到的交互残差全当生物信号；这些百分比不能移植到 H1/GWPS/官方全基因轴 |
| 原始 DepMap cosine 对 full Δ 相似性有一定相关，对去模板相似性接近零 | Fig. 2b p. 6；Tables 11–13 pp. 37–38 | 原始 kNN 作为对照；优先验证从 DepMap 到响应的监督映射，不把先验邻居当调控关系 |
| DepMap→Ridge/MLP 在同背景五折中与 MORPH 接近；MLP 跨背景通常优于 Ridge | Fig. 2a/3a；Table 14 p. 38、Table 18 p. 40 | 同一输入先验、覆盖、训练边界、生成器上比较 Ridge 与小 MLP；不从结果推出更大模型永久无效 |
| 有效信息并非都在 DepMap PC1；20–50 维附近收益趋缓 | Fig. 2d p. 6；Table 15 p. 38 | 将低秩监督对齐作为低成本完整路线；50 维是候选，不是已验证的普遍最优维数 |
| NTC 前 5 PCs 可覆盖模板能量 72.3%–88.8%，对正交残差覆盖低得多 | Fig. 1i p. 5；Table 6、Note 3 p. 36 | 允许 NTC 提供状态子空间，但仍须学习模板方向的系数、符号和振幅；子空间包含不等于无需监督即可估计模板 |
| 部分跨背景收益来自保守扰动成分，测试模型未恢复目标交互 | Fig. 4 p. 8；Tables 25–29 pp. 43–44 | 分开报告 seen-target/context-OOD 与 global-unseen/joint-OOD；用有限目标监督作为 oracle/few-shot 能力参考，不混入比赛 zero-shot |
| 通路诊断揭示 MLP 对部分应激、免疫、信号通路的收益 | Fig. 3f p. 7；§4.7.6 p. 19 | 固定 Hallmark/Reactome 诊断，报告覆盖；通路相关性、效应幅度和模板移除口径分别记录 |

以上均为[论文 v1 原始结果](https://www.biorxiv.org/content/10.64898/2026.07.24.740459v1.full.pdf)，尚非本地有效证据。

## 3. 数据与评估：不能直接照搬的处理

### 3.1 论文任务与 VCC 输出不相同

论文主分析先将四背景测量轴取交集（6,642 基因），统一逐细胞 CP10K/log1p，再选 2,000 HVG。无 log 后逐基因 scaling；没有额外 QC/batch correction；不同 guide 按 target 合并；每个 target 少于 30 细胞剔除。K562 主分析是 focused essential screen；GWPS 另用于 essential/nonessential 的专门分析，不能把二者当同一个任务分布。[§4.1.1–4.1.2 p. 10；§4.1.8 p. 11](https://www.biorxiv.org/content/10.64898/2026.07.24.740459v1.full.pdf#page=10)。

**对 VCC 的推断**：保留原始 counts 与测量掩码；模型输出仍须恢复官方完整轴。论文中的均值 log 响应可作辅助表示或诊断，但不能未经校准反变换后就视作 bulk composition 或细胞级 DE 的正确预测。论文没有验证 20k UMI 深度对齐、400 细胞整数生成、零率、Fano、逐细胞 DE 或我们的六项官方指标，因此不能为任何特定计数生成器背书。CP10K 改成 CP50K 也不是 log 后整体乘 5，必须从同一底层组成重算。

论文的通用 Ridge/MLP MSE 监督及 `std_r/pref` 对完整输入基因向量计算，没有按当前 perturbation 排除靶基因自身的代码。因此靶基因自身若在 2,000 HVG 中，可贡献这些指标；未在 HVG 中则自然缺席。我们应保留官方靶点排除口径，同时在诊断中单列自身敲低与下游恢复，不能将论文相关性直接当作下游预测分数。[`models/ridge.py:93`](https://github.com/xinyizhanglab/perturbation-decomposition/blob/a15214780619736d393f40240e56ba992fd416a3/models/ridge.py#L93)、[`evaluation/metrics.py:13–34`](https://github.com/xinyizhanglab/perturbation-decomposition/blob/a15214780619736d393f40240e56ba992fd416a3/evaluation/metrics.py#L13)。

### 3.2 合并数据选 HVG 包含留出表达信息

原文明确在全部四背景 log-normalized 表达合并后一次选 2,000 HVG，无背景 balancing/downsampling，无 batch key。这包括后续留出的背景与靶点表达，属于表示阶段的 transductive 信息使用；它没有证明目标标签进入回归拟合，但不满足我们严格 LOCO 的表示边界。[§4.1.8 p. 11](https://www.biorxiv.org/content/10.64898/2026.07.24.740459v1.full.pdf#page=11)。

**必须修正的迁移做法**：HVG/表达 PCA/损失权重只由当前训练数据与预先许可的目标 NTC 拟合；不使用留出扰动表达选基因，不把论文 2,000 维设成 VCC 的评分范围。固定官方轴及先验基因集合属于元数据使用，和使用留出表达统计要区别。

### 3.3 DepMap PCA 与目标质心必须分别判断

监督模型的 DepMap PCA 在训练扰动内拟合，跨背景使用 source-training descriptors，joint-OOD 删除全部来源中的留出靶点后重拟合；这部分方法遵守训练边界。源码没有额外 StandardScaler，PCA 自身中心化。[§4.3.1/4.3.3 pp. 12–13；§4.5 pp. 15；`models/ridge.py:20–26,76–78`](https://github.com/xinyizhanglab/perturbation-decomposition/blob/a15214780619736d393f40240e56ba992fd416a3/models/ridge.py#L20)。外部固定先验能查询未见靶点并不等于读到了其表达标签；但输入缺失填充及外部版本仍应冻结。

论文 zero-shot 的 `T_ref` 是**真实目标扰动质心，仅用于推断之后的评分**，未用于训练或 checkpoint 选择。代码 `T_tgt=tD.mean(0)`，预测只用 `sV,sD,tV`，然后调用 `pref(tD,yp,T_tgt)`。这是使用真值定义的诊断标尺，不能直接称训练泄漏，也不能把该质心提供给 VCC 预测器。[§4.5.1 p. 15；`cross_cl.py:62–88`](https://github.com/xinyizhanglab/perturbation-decomposition/blob/a15214780619736d393f40240e56ba992fd416a3/benchmarks/cross_cl.py#L62)。

**对本地评估的要求**：正式六项仍用冻结官方 scorer；论文 `pref`、正交残差及真值 ANOVA 都作额外诊断，并注明哪些基底仅在评分时可用。不同质心产生不同问题，不应声称目标质心必然使分数变高或完全消除了模板误差。

### 3.4 论文还没有解决效率、批次和长尾弱响应

按 guide pooling/minimum n 处理没有显式建模敲低效率；essential-enriched 筛选、DepMap 缺失靶点排除及共有靶点选择改变了覆盖和响应分布。论文的 23%–65% 模板占比、低秩最优范围和训练收益不保证适用于我们的全 GWPS/H1/官方 unseen 靶点。原文 p. 9 主动承认 perturbation efficiency、时间、调控状态和单细胞异质性可能提供当前表示之外的信息。[Discussion p. 9](https://www.biorxiv.org/content/10.64898/2026.07.24.740459v1.full.pdf#page=9)。

**对 VCC 的推断**：未知 KD 效率保留 unknown，不按弱响应/非显著一律删样本；批次匹配 NTC、guide/target 汇聚及效率不确定性需要本地比较。核心覆盖与先验覆盖单列，不用只剩强响应的子集替代完整测试。

## 4. 不可把经验发现扩大成的信息论结论

### 4.1 “NTC 有用/无用”都需要限定

NTC PCs 覆盖模板方向是事后投影实验，需要真实模板才能计算覆盖。它没有给出无需监督从 NTC 唯一识别模板符号/系数/强度的算法。原文的简单 control-descriptor 消融也只用了**靶基因自身**在 NTC 中的 mean、std、median、zero fraction、q25、q75；靶点不在 HVG 中时用六个零。该实现不能排除全基因 NTC 协方差、集合编码、已知调控状态或更丰富背景信息的增益。[`fig4_supplementary.py:118–141`](https://github.com/xinyizhanglab/perturbation-decomposition/blob/a15214780619736d393f40240e56ba992fd416a3/figures/fig4_supplementary.py#L118)。

文中存在两种不同的 control-space 分析：Fig. 1i 图注明确是 NTC 单细胞 PCA，对应 `figures/fig1_e.py:41–48` 实际筛 `non-targeting` 后拟合 PCA；Methods §4.9.6 p. 25 的另一分析则用 energy-test `p>=0.5` 的非显著扰动响应作为 control-like 代理。后者不能重命名为真实 NTC 单细胞分布，两者也不能合并成同一个证据。该节的 Marchenko–Pastur 阈值用 `σ≈1/sqrt(ncells)`，未全面建模基因异方差、基因相关和不等 n，是近似结构诊断，不能作为 VCC 通用噪声上界。[NTC PCA 代码](https://github.com/xinyizhanglab/perturbation-decomposition/blob/a15214780619736d393f40240e56ba992fd416a3/figures/fig1_e.py#L41)、[Methods p. 25](https://www.biorxiv.org/content/10.64898/2026.07.24.740459v1.full.pdf#page=25)。

因此可借鉴“先验负责靶点功能，NTC 提供背景状态”的可检验假设；不能写成“评分不奖励周期结构，所以周期应回归掉”或“目标 NTC 不含可迁移信息”。固定实验快照中的周期/应激变化可能正是标签。

### 4.2 交互项不可预测不是已证明的普遍定理

全文报告所测试方法在四个背景的 zero-shot 下未恢复 \(\gamma\)，未提供所有模型的不可预测性定理。作者也指出数据缺少可能有用的状态与动态测量。随机效应、ANOVA 正交或在某个有限样本中的负相关，均不推出任意输入特征下条件期望为零。

**本次数学推断，非作者结论**：在论文平衡 ANOVA 中，\(\sum_c\gamma_{c,p}=0\)。若预测器直接使用其它 \(C-1\) 个背景同靶点的均值，

\[
\overline\delta_{-c,p}
=\mu-\frac{\alpha_c}{C-1}+\beta_p-\frac{\gamma_{c,p}}{C-1}.
\]

因此源均值的组成天然包含目标交互的反向项；其与目标 \(\gamma\) 的负相关不能独立证明“交互无信息”。需要 source-only 可计算参考、重复性估计和实际预测实验共同判断。[分解定义依据：§4.8.2，p. 20](https://www.biorxiv.org/content/10.64898/2026.07.24.740459v1.full.pdf#page=20)。

### 4.3 “四象限/稳定性门控”仍是我们的假设

v1 没有给出按 `(靶点,读出基因)` 稳定性划分四象限并训练 MoE 门控的验证。其 ANOVA/模板分析可启发这个假设，但稳定性标签应只从来源背景、匹配 n/深度和可重复响应估计；再在留出背景上验证稳定性是否可预测。不能用目标真值确定门控，不能把未显著当稳定无效应。

### 4.4 “低秩够用”有任务边界

作者的 Ridge/MLP 直接预测 full \(\delta\)，没有先把训练标签投影残差再训练；分解主要是事后诊断。跨背景 Ridge/MLP没有显式背景 descriptor，学习的是来源平均映射。因此论文并没有验证“显式分解头必胜”“MoE 必胜”或“NTC 条件化已经被所有方案排除”。[§4.3.3 p. 13；§4.5.1 p. 15](https://www.biorxiv.org/content/10.64898/2026.07.24.740459v1.full.pdf#page=15)。

## 5. 原文及冻结代码存在的可复现性疑点

这些问题影响如何复用配方，不足以单独否定文章的核心思想。真正复现须明确采用哪个配置，并重新跑匹配比较。

1. **MLP 学习率不一致**：Methods p. 13 为 `5e-5`；`models/ridge.py:52–54` 和 `configs/datasets.yaml` 为 `5e-4`。[原文](https://www.biorxiv.org/content/10.64898/2026.07.24.740459v1.full.pdf#page=13)、[代码](https://github.com/xinyizhanglab/perturbation-decomposition/blob/a15214780619736d393f40240e56ba992fd416a3/models/ridge.py#L52)。不能称直接跑默认值为精确复现。
2. **scGPT 与其它模型的划分不完全匹配**：Methods p. 13 明确 scGPT 为单次约 85/15 perturbation split × 5 seeds；Ridge/MLP/MORPH 为 matched five-fold。Fig. 2a 图注的笼统匹配五折说法需按 Methods 收窄；不要将 seeds 与 folds 视为相同配对单位。[§4.4.1 p. 13](https://www.biorxiv.org/content/10.64898/2026.07.24.740459v1.full.pdf#page=13)。
3. **组合任务图文不一致**：Fig. 2c 的实际图及图注明确画出 Add+Resid；Methods §4.6.3 p. 16 却说没有训练 double 同时拥有两个单扰动标签，未评估 additive-plus-residual。当前版本不能据此确认该基线的确切训练/计算条件。[Fig. 2 p. 6](https://www.biorxiv.org/content/10.64898/2026.07.24.740459v1.full.pdf#page=6)、[Methods p. 16](https://www.biorxiv.org/content/10.64898/2026.07.24.740459v1.full.pdf#page=16)。
4. **细胞阈值默认不一致但未证实实际违约**：文稿统一 `n>=30`，`data/process.py:22` 默认 5，部分图脚本 5 或 10。这些脚本读取已处理 h5ad，源文件可能早已过滤 30；只能说公开入口本身不足以保证论文条件，不能断言作者实际用了小于 30 的任务。[代码](https://github.com/xinyizhanglab/perturbation-decomposition/blob/a15214780619736d393f40240e56ba992fd416a3/data/process.py#L22)。
5. **噪声分解需要独立 NTC 和正确分母**：Methods §4.8.4 用两个 half 的平均观测能量为分母，`figures/fig1_f.py:83,168–190` 却用 full-data response energy；脚本两个 half 也减同一个完整 NTC 均值（lines 121–127）。共享 NTC 误差不会因 perturbation split-half 消失，尤其会进入模板；不能直接将 cross-half reproducibility 全称为独立生物可重复性。究竟哪份代码生成 v1 图未被确认。我们应独立划分 NTC/真实重复、明确分母，不把这套百分比直接充当 VCC 噪声上限。[Methods pp. 21–22](https://www.biorxiv.org/content/10.64898/2026.07.24.740459v1.full.pdf#page=21)、[代码](https://github.com/xinyizhanglab/perturbation-decomposition/blob/a15214780619736d393f40240e56ba992fd416a3/figures/fig1_f.py#L115)。
6. **部分强表述超过实际指标**：主文 p. 4 用 `r=0.25–0.39` 描述 conserved alignment，Table 25 p. 43 对应数值列是 `R²(β)`；该表相关系数列约 0.184–0.221（MLP）。引用结果应注明具体 metric/面板，不能混用 r 与 R²。[Table 25 p. 43](https://www.biorxiv.org/content/10.64898/2026.07.24.740459v1.full.pdf#page=43)。

## 6. 论文最值得落实的三项方向

1. **固定输入、监督边界与生成器，完成 DepMap 监督对齐 Ridge/小 MLP 的真实比较**，同时保留无先验、匹配随机先验、原始相似度迁移对照；全局未见靶点必须单列。
2. **把分解诊断放在官方六项之外**：模板振幅、正交残差、通路恢复、标签置乱和可重复性；明确每个基底的训练来源与评价专用真值使用。不能通过删模板/改公式替换官方分数。
3. **保留 VCC 的计数分布任务**：论文指导平均响应表征和输入先验选择，细胞生成仍须自己的整数/深度/稀疏/重复生成/NTC-null 验收。新模型必须先完成预测统计诊断，再进行正式评分。

本节是由原文推导的路线建议，不声明任何本地新方法已经通过比较。

## 7. 与本仓库已完成实验逐项对照

本地核验快照为 `fef68a3`。依据 Ledger 已闭环证据、各 Experiment REPORT 和实际源码；不将未执行的设计算作实验。下表预测模型的主比较均限 **S2-H1、已见靶点、seed 1**，Overall 属该本地参考群体，不能与论文相关性或官方 A/B/C Overall 横比。

| 我们实际做过什么 | 本地结果/代码事实 | 与论文的关系及应改变的判断 |
|---|---|---|
| 靶点 one-hot + NTC 特征的线性模型；每靶源响应均值 Shared | `E-BASE`：Linear 0.16114，Shared 0.21087；`model.py` 的 `shared=right/mass` 为同一靶点跨来源加权均值 | **不是 DepMap Ridge，也不是跨扰动模板**。需要保留同靶迁移基线，再比较有功能描述符的外推；不能说“我们已经试过论文的线性方案” |
| NTC 条件项消融和正则调整 | `E-CONDITION-ABLATION` 0.16865；`E-RIDGE-STRENGTH` 0.18609；仍低于 Shared | 初始背景编码来自四个 source NTC 均值，中心化有效秩最多 3；它不能检验所有 NTC 集合/交互模型。移除条件项小收益不能证明背景信息有害 |
| NTC PCA/Reactome/random program 表示 | `E-PROGRAM`：PCA32 0.16586，真实程序 0.14100，匹配随机 0.17479 | 实际 PCA 候选用训练 NTC 细胞拟合，但下游消费各背景均值的投影。这里压缩的是 **背景输入**；论文压缩的是 **靶基因 DepMap 描述符**，两种假设不同 |
| 缺测响应 PCA/程序补全 | `E-MASKED-RESPONSE`：0.20900/0.20985/0.21054，均未胜 Shared | rank32 输出补全未获支持，不等于 DepMap 对未见靶点无效；也未解决完整官方轴上的靶点×读出监督稀疏 |
| 低深度 QC，配匹配随机删除 | `E-SHARED-QC`：QC 0.21276、随机 0.21233、原始 0.21087，未达门槛 | 只限制本次低深度硬筛选。没有测试 guide 效率软权重/条件化，不支持“所有 QC 无用”或继续扩大硬筛强度 |
| 非线性 transport、组成 LFC/IPF、Gamma–Poisson 整包 | `E-MECHANISM-PORTFOLIO`：0.08993/0.19220/−0.12233 | 联动改变响应、损失和生成器，不能把失败归因于某一分布族或“非线性无效”。新先验比较应固定生成器，生成器比较应固定响应 |
| 全基因 decoder 与条件 cell VAE | `E-GENE-DECODER-PACKAGE` 0.17124；`E-CELL-VAE-ALIGNED-PACKAGE` 0.08031；后者零响应 mean-log RMS 0.06127，NTC 抽样 0.01163 | 当前 decoder 的零响应漂移是可修的具体缺口；尚不支持架构普遍无效。其辅助 Welch/BH 13,697 vs 0 不能冒充官方 Wilcoxon 结果 |
| 五背景观测/计数校准和能力评价器 | `E-TASK-DATA-GENERATION-R2` 16/20；`E-LOCAL-CAPABILITY-EVALUATOR-R3` 15 面板、60 次评分、87/90 检查 | 是真实接口/生成诊断，**不是 15 个跨背景拟合模型**。逐组 IPF 抵消抽样波动；3 面板 normalized 参照未定义。五背景/joint-OOD 模型泛化尚未完成 |

源码定位：[线性模型](../../experiments/init-linear/src/model.py)、[NTC 表示](../../experiments/init-linear/src/representation.py)、[旧数据入口](../../experiments/init-linear/src/data.py)、[旧计数生成](../../experiments/init-linear/src/evaluation.py)、[当前计数工具](../../src/vcc_task/counts.py)、[能力诊断](../../src/vcc_task/capability.py)。本地结论以 [Evidence Ledger](evidence_ledger.json) 中同名证据为准。

### 7.1 最新官方输出审计揭示了具体的覆盖瓶颈

[九份固定提交审计](leaderboard_prediction_counts_REPORT.md)（`E-LEADERBOARD-COUNTS-AUDIT`）比“模型都只会应激”提供了更准确的判断：

- 当前 Shared/QC 的跨扰动质心能量只占预测 mean-log 响应的 **2.6%–3.9%**，官方 PDS 为 **0.66906/0.67137**。这支持有限的已见靶点下游区分能力；不能因旧 XGB/CVAE 模板占比高，就把当前 Shared 也叫作纯模板。该能量不是论文逐响应方向投影比例，二者不可直接相减。
- 当前 272 个官方已见靶点的直接训练响应 **全部仅来自 K562**，其官方读出覆盖 **7,680/18,533**。剩余 **10,853** 个读出没有这些靶点的直接监督。四 source 的基因并集与任务总数，不能冒充每个官方靶点的跨背景/全基因监督覆盖。
- 28 个未见靶点上，Shared/QC 每份提交的 **33,600 个细胞全部是输入 NTC 的原行**：代码零 Δ 回退。它们的预测多样性来自抽样，不能说明已学会未见靶点响应。
- 最好 QC 的官方 raw MSE **4.72351**、LFC NMAE **0.97075**、Jaccard **0.01892**；MSE normalized 为 0，Fidelity/Jaccard normalized 为负。PDS 能区分部分靶点，尚不足以正确恢复效应幅度和 DE 集合。

**由此作出的研发判断**：论文最有价值的补充是给靶点增加可外推的功能表示，并把响应幅度及细胞分布分别验收。继续微调 one-hot Ridge，或只让零率更像 NTC，都没有正面解决这两个缺口。DepMap 输入能共享辅助靶点的信息，但不自动补齐所有未测读出；后者仍依赖观测掩码和可验证的输出映射。

## 8. 必须落实的修正与仍需比较的选择

### 8.1 正确性和能力声明：不以消融保留已知不一致

1. **把预测目标、观测映射和最终计数连起来。** 旧 `init-linear` 学的是 mean(log1p(CP10K)) Δ，生成时逐细胞加 Δ、截零、expm1、重分配文库、独立随机取整；这个链条不保证恢复训练均值，更不保证官方 bulk/DE 统计。新缓存已经保存 bulk CP50K、mean CPM、mean-log 多视图，但旧训练器没有因此自动切换。新 run 必须从最终 counts 同时验证 bulk 组成及逐细胞 CPM/检出率/LFC，并量化逆变换偏差。不是把 `target_sum` 从 10k 改为 50k 就完成对齐。
2. **测量掩码进入归一化分母。** 公共来源 NTC 的已测官方轴计数只占 native counts 中位数约 68%–74%；只在 MSE 上屏蔽缺测值仍不足。完整轴预测先投影到该来源实测集合 M，再在 M 内重算 bulk/细胞统计，与真值比较；缺测不补零监督。正式输出不缩成共同 6,642 或 2,000 基因。
3. **严格按可用信息拟合。** 旧数据/评估入口硬编码 H1，需在新的实现/run 中支持真实五背景轮换、全局靶点留出和研究留出；HVG、PCA、效率估计、模板和超参遵守 source/inner-fold 边界。论文真实目标质心/ANOVA 只可用于标明 oracle 的评分诊断。不得以五背景 evaluator 接口通过代替五背景模型比较。
4. **把“合法整数”与“响应/分布正确”分开验收。** 九份历史提交已是合法整数，不需要修复不存在的小数问题。实际要修的是零响应漂移、期望响应偏移及逐组列和固定导致的抽样方差不足。要求总体校准后独立抽样的候选通过固定 Δ 比较；不能把“目前还未验收”写成“新方案已最优”。文库和稀疏度是联合统计约束，不能强制每细胞 20k 或每扰动等于 NTC 零率。
5. **主响应目标和诊断服从靶点排除。** 旧 Ridge 的响应 loss 未按当前靶点排除自身；论文也没有该排除。新主要下游损失/指标须使用声明的官方掩码，PDS 排除整个面板靶基因，其他项按冻结规则排除当前靶点。靶基因仍参与合法 counts 和组成分母，可作训练侧效率观测；不能先删列再归一化。邻近基因和周期/应激后果属于标签，不因为“间接”就删掉。
6. **把弱模型现象与生物归因分开。** Shared 不是模板；旧程序实验不是 DepMap 实验；NTC 简单拼接失败不是 NTC 无信息；相关性高不是机制正确；方向 Fidelity 也不是纯方向正确率。固定论文/官方/本地三种不同指标的名称和有效数，保留截断前 raw 结果。

本节沿用并落实[数据处理与评估默认方案](data_evaluation_decision.md)，不是另建一套矛盾协议。历史 run、计数、scorer、绑定和分数保持可追溯，新条件必须新 run。

### 8.2 不应伪装成正确性修复的建模选择

- **效率**：值得用 source 端估计剂量和不确定性作候选，但 unknown 不是敲低失败，弱响应不是劣质样本；不能硬筛到只剩强响应，也不能推断时偷看目标自身表达。选择硬筛/软权重/条件化仍需真实比较。
- **深度与细胞数**：20k 是目标深度分布的中位量级；低深度原始数据不能“下采样到更深”，不足 400 个真实细胞不能复制成 400 个独立真值。当前仅 806/17,322 结构合格任务具有 n≥400，统一筛选将只留约 4.7%。按深度生成与规模归一化也不能消除 3′/Flex 的基因捕获差异。
- **响应权重**：source/context/target 等权、可靠性加权、弱响应收缩都值得比较；旧按细胞数加权确会改变目标人群，但它不是数学错误，改权重的收益不能预设。
- **模板模型**：先采用训练 source 定义的模板/残差诊断，不把强应激模板无条件加给目标。不从线上归一化基准的几项数值反推出目标真实模板只占 1%，也不把论文模板与官方 count-space baseline 当作同一对象；线上 r4 构建配方仍有[证据边界](r4_anchor_audit.md)。
- **稳定性 MoE/复杂交互**：保留后续候选。先测来源数据估计的稳定性是否在留出背景有预测力，区别真实重复性与细胞采样噪声；不将真值分型变成推断特征。论文不要求为了采用 DepMap 必须先开发 MoE。

## 9. 下一轮只推进现有三个问题

本次核验加强已有候选的来源依据，**不新增第四条实验队列、不关闭任何 pending 方法比较**。DAG 仍是 draft，实际独立 run/config/完整预算须在启动前冻结；本次没有训练预算消耗。

| 已有比较 | 本文带来的具体落实 | 晋级/停止依据 |
|---|---|---|
| `C-POPULATION-EMISSION` | 固定同一响应和 NTC 群体，对比当前逐组 IPF 与总体校准后独立抽样；共同检查零响应、非零均值偏差、深度×检出率、重复生成方差及 n 匹配 DE | 先在训练校准范围冻结容差。不能靠钉死基因列和通过均值验收；完成有限批次后记录适用范围，不进入无限生成器调参 |
| `C-LOCAL-EVAL-SUPPORT` | 一批全折真实参考与 anchors；正式六项外增加 source-only 模板投影、残差幅度/误差、通路恢复、靶标签置乱；ANOVA/split-half 另标评分诊断，处理独立 NTC 及生物重复边界 | 真实 n、轴覆盖、官方面板覆盖、context/joint/study OOD 单列；不能定义的 Overall 保留未定义，预定有效范围仍可继续开发，不追面板/seed |
| `C-DEPMAP-RESPONSE-ALIGNMENT` | 同一数据/生成器上的 Ridge 与小 MLP 完整方案，保留 Shared、无先验/匹配随机描述符/原始 kNN；默认低秩候选不自动扩全 rank 网格；同时提供完整官方轴观测映射 | 首轮看未见靶点与已见靶点的配对增益、完整六项和去模板下游能力；若只增模板/自身下降、仅强响应或高 n 子群获益，则不认定可迁移响应改进；有价值再分配独立折/seed 确认与归因预算 |

执行所需的正确性条件落实后，建模预算优先给第三项；不能将“所有公开来源都获得有效 normalized 分数”扩张成无期限的模型启动前置。当前 H1 和反复使用的五背景属于开发证据，不能改称未触碰测试集。

**KnowGraph attention**：`PAP-DECOMP`、`PAP-LIMITS`、`PAP-F-CONTROL`、`PAP-F-GAMMA`、`PAP-F-DEPMAP-GEOM`、`BIO-COESSENTIAL`、`BIO-PATHWAY`、`ST-ANOVA`、`ST-WILCOXON`，并检查直接相连的 `MY-SHARED-CTX`、`MY-TEMPLATE-NTC`、`MY-QUADRANTS`。监督对齐、残差及通路诊断获得原文支持；“γ 普遍不可学”“NTC PC 自动给出目标模板”“低秩线性普遍饱和”和“源 Shared=模板”被收窄或否定。图谱仍是注意力索引，本轮未修改用户提供的索引文件。

交付范围：完成全文/源码核验、已有实验对照与研究记录更新；**没有声称上述新管线已经实现，没有新训练、新评分或 leaderboard 提交**。本地有效性仍由登记的三个实际比较决定。
