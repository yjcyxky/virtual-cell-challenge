# VCC 2026 初始化：文献依据与待证伪假设

核对日期：2026-09-27。本文只使用已正式发表的原始研究；研究问题来自[用户分享的九轴方案](https://chatgpt.com/share/6ab9c3e3-56cc-83e9-bbef-b1f196a8e83c)，不读取或采用本仓库历史实验结果来筛选候选。分享中的示例分数、模型优势和推荐主干均不作为已获证据。

下列 `L01–L10` 是**外部文献依据**，支持建立候选和匹配对照，不代表方法已在本项目五个 CRISPRi 背景 H1、K562、RPE1、HepG2、Jurkat 上通过验证。相应的项目内方法效应仍为未知。文献状态与本地实验状态须分开登记；文献不能满足“已完成本地比较”的执行前置条件。

轴沿用分享定义：T 任务、D 数据、R 表示、A 架构、L 损失、O 优化、V 评估、I 推断、G 泛化。V 冻结为 S0 已见条件、S1 扰动 OOD、S2 背景 OOD、S3 联合 OOD、S4 研究/数据集 OOD 的协议集合；V 不参与选优。以下“项目检验”是据文献提出的研究设计，而非论文已经证实的结论。

## 文献卡片

### L01 — 数据质量与真实背景差异：Replogle 等

**正式版本**：Replogle JM et al. *Mapping information-rich genotype-phenotype landscapes with genome-scale Perturb-seq*. **Cell 185**, 2559–2575.e28 (2022). DOI: [10.1016/j.cell.2022.05.013](https://doi.org/10.1016/j.cell.2022.05.013)。[作者论文全文](https://pmc.ncbi.nlm.nih.gov/articles/9380471/)；[作者数据发布](https://plus.figshare.com/articles/dataset/_Mapping_information-rich_genotype-phenotype_landscapes_with_genome-scale_Perturb-seq_Replogle_et_al_2022_processed_Perturb-seq_datasets/20029387)。

- **原研究事实**：多重 CRISPRi 与直接 guide 捕获支持大规模测量；K562 与 RPE1 的细胞背景、采样时间、CRISPRi 效应器及靶点面板并非完全相同。研究执行了 guide 身份过滤、敲低效率检查，并从扰动表达中发现功能关系。
- **适用范围/局限**：支持核验实际 guide、实验条件、target 覆盖与 knockdown；不能据此假定五背景可无差别合并，或把背景差异全解释为技术批次。
- **轴**：D、R、V、G。
- **项目检验**：在固定有效输入上比较 pooled 与背景/扰动平衡采样；质量筛选另外配同细胞数、同靶点覆盖的随机删减对照。逐背景、guide/生物重复和响应强度分层报告。背景与 study 完全混杂时明确不可辨识，不能声称已隔离跨研究效应。

### L02 — 有效扰动识别的价值及选择偏差：Mixscape

**正式版本**：Papalexi E et al. *Characterizing the molecular regulation of inhibitory immune checkpoints with multimodal single-cell screens*. **Nature Genetics 53**, 322–331 (2021). DOI: [10.1038/s41588-021-00778-2](https://doi.org/10.1038/s41588-021-00778-2)。[作者论文全文](https://pmc.ncbi.nlm.nih.gov/articles/PMC8011839/)。

- **原研究事实**：Mixscape 用转录组扰动特征区分有可检测响应与逃逸细胞；论文明确指出，若真实功能扰动没有可检测的转录变化，也可能被标为 non-perturbed。
- **适用范围/局限**：原研究是特定免疫检查点 CRISPR 敲除场景。转录响应分类不是 CRISPRi 效率的无偏真值；按响应幅度筛选会改变预测目标的人群。
- **轴**：D、L、V。
- **项目检验**：必要身份/双细胞校验之外，将额外 QC、软重加权与原样保留视为候选；配随机删减及强弱响应分层，并检查弱效应丢失。训练 QC 不得使用留出背景的扰动结果；评分群体固定，不能筛掉难样本后重新定义成功。

### L03 — 简单、低秩及迁移基线：Ahlmann-Eltze 等

**正式版本**：Ahlmann-Eltze C, Huber W, Anders S. *Deep-learning-based gene perturbation effect prediction does not yet outperform simple linear baselines*. **Nature Methods 22**, 1657–1661 (2025). DOI: [10.1038/s41592-025-02772-6](https://doi.org/10.1038/s41592-025-02772-6)。[作者论文全文](https://pmc.ncbi.nlm.nih.gov/articles/PMC12328236/)。

- **原研究事实**：在其单/双扰动预测任务中，所测深度模型没有稳定胜过简单基线；双线性预测器可组合 gene 与 perturbation 表示，且来源扰动数据的表示迁移有帮助。预训练表示与预测器应分别检验。
- **适用范围/局限**：主要是条件平均表达及指定数据/基因面板的预测；不是五背景 NTC 条件分布生成的判决，也不证明复杂模型无效。
- **轴**：T、R、A、O、G。
- **项目检验**：保留 zero-Δ、训练内共享响应、收缩/均值迁移、ridge/低秩双线性和小 MLP。共享响应必须只用允许的来源背景；S1/S3 未见靶点不能查到其训练响应。固定同一预测头比较训练数据表示、外部预训练表示、等维随机表示；固定表示比较线性与非线性预测头。

### L04 — 共享偏移与扰动特异响应需要分开诊断：Systema

**正式版本**：Viñas Torné R et al. *Systema: a framework for evaluating genetic perturbation response prediction beyond systematic variation*. **Nature Biotechnology 44**, 1050–1059 (2026；2025-08-25 在线发表). DOI: [10.1038/s41587-025-02777-8](https://doi.org/10.1038/s41587-025-02777-8)。[期刊全文](https://www.nature.com/articles/s41587-025-02777-8)；[期刊 PDF](https://www.nature.com/articles/s41587-025-02777-8.pdf)。

- **原研究事实**：十个数据集的平均扰动—对照偏移会影响常用指标；改变参考可突出扰动特异成分。系统变化可能来自生物学、设计或混杂，并非全部是应删除的噪声。
- **适用范围/局限**：替代 reference 同样有强度、弱效应和参考选择方面的限制；该研究不是 VCC 官方评分规则。
- **轴**：T、L、V、G。
- **项目检验**：冻结正式评分，同时附加共享响应与靶点特异响应诊断。比较共享响应、共享+背景修正和直接条件预测，检查增益是否仅来自共同偏移。评分端可以用真实扰动作诊断 reference，但不得将这些标签或其 centroid 输入模型、适配过程或调参流程。

### L05 — GO/共表达图是可测试先验：GEARS

**正式版本**：Roohani Y, Huang K, Leskovec J. *Predicting transcriptional outcomes of novel multigene perturbations with GEARS*. **Nature Biotechnology 42**, 927–935 (2024；2023-08-17 在线发表). DOI: [10.1038/s41587-023-01905-6](https://doi.org/10.1038/s41587-023-01905-6)。[期刊全文](https://www.nature.com/articles/s41587-023-01905-6)。

- **原研究事实**：GEARS 将 GO 靶点关系与表达共变图注入表示，通过图网络预测未见单/多基因扰动。单基因 K562/RPE1 评估分别训练，不能等同于跨背景迁移。
- **适用范围/局限**：GO 通路相近不保证响应相同；共表达关系具有背景和测量依赖。原结果未证明这些先验在五背景 S2/S3 必然有效。
- **轴**：R、A、G；先验作用位置必须单独记录。
- **项目检验**：冻结预测器，以真实图/模块、无先验、保持度数或模块大小与覆盖率的随机先验作匹配比较；图表示与网络架构替换分开。训练数据派生的图只在训练边界拟合，外部 GO 版本、覆盖缺失与未见靶点映射冻结。模块压缩同时配等维 PCA/随机表示，以辨别生物知识与降维收益。

### L06 — 预训练表示候选：scGPT

**正式版本**：Cui H et al. *scGPT: toward building a foundation model for single-cell multi-omics using generative AI*. **Nature Methods 21**, 1470–1480 (2024). DOI: [10.1038/s41592-024-02201-0](https://doi.org/10.1038/s41592-024-02201-0)。[期刊论文](https://www.nature.com/articles/s41592-024-02201-0)。

- **原研究事实**：以大规模单细胞预训练产生 gene/cell 表示，并通过任务适配展示包括扰动预测在内的下游用途。
- **适用范围/局限**：适配后效果不能直接说明冻结表征或新背景零样本预测有效；不能从细胞类型分离能力推出因果扰动表示质量。
- **轴**：R、O；若同时换预测器则另涉及 A。
- **项目检验**：先在同一低容量预测头下比较 NTC mean、mean+variance、等维训练内降维与冻结 scGPT 表示；gene、cell、population pooling 分开定义。记录权重版本、基因覆盖、预处理、预训练数据重叠。只有表示有信号后再将冻结与微调作独立比较。

### L07 — 预训练表示与基因面板兼容性：scFoundation

**正式版本**：Hao M et al. *Large-scale foundation model on single-cell transcriptomics*. **Nature Methods 21**, 1481–1491 (2024). DOI: [10.1038/s41592-024-02305-7](https://doi.org/10.1038/s41592-024-02305-7)。[期刊论文](https://www.nature.com/articles/s41592-024-02305-7)。

- **原研究事实**：大规模转录组预训练产生细胞和基因上下文表示，展示包括扰动预测和模块推断的用途。
- **适用范围/局限**：预训练的输入基因、测序深度和数据处理假设需要满足；缺测不等于实测零，不能靠无记录补零宣称面板兼容。论文结果不预先决定本任务与 scGPT 的排序。
- **轴**：R、O、D。
- **项目检验**：与 L06 使用同一训练边界、同一下游头和对照；分别报告可映射基因比例及缺测策略。冻结表示与训练内响应表示是不同候选，不能因权重可下载就默认优先。需要改面板时整套比较重新冻结，而不只替换候选侧输入。

### L08 — 零样本表征的反证边界：Kedzierska 等

**正式版本**：Kedzierska KZ, Crawford L, Amini AP, Lu AX. *Zero-shot evaluation reveals limitations of single-cell foundation models*. **Genome Biology 26**, 101 (2025). DOI: [10.1186/s13059-025-03574-x](https://doi.org/10.1186/s13059-025-03574-x)。[期刊全文](https://link.springer.com/article/10.1186/s13059-025-03574-x)。

- **原研究事实**：在研究所测零样本表征任务中，指定版本的 Geneformer/scGPT 未稳定胜过简单方法；细胞生物信息与批次混合应分别衡量。
- **适用范围/局限**：不是五背景 CRISPRi 效应预测实验；不否定任务微调、后续版本或特定 gene 表示。其零样本表征概念与 S2“新背景只有 NTC”也不能混用。
- **轴**：R、V、O。
- **项目检验**：使用 L06/L07 的相同预测头对照，并单独做 NTC 表征诊断。表征聚类图只作为辅助诊断，不替代扰动效应与 OOD 指标；微调需另立方法坐标和预算。

### L09 — 潜在空间均值转移：scGen

**正式版本**：Lotfollahi M, Wolf FA, Theis FJ. *scGen predicts single-cell perturbation responses*. **Nature Methods 16**, 715–721 (2019). DOI: [10.1038/s41592-019-0494-8](https://doi.org/10.1038/s41592-019-0494-8)。[期刊论文](https://www.nature.com/articles/s41592-019-0494-8)。

- **原研究事实**：VAE 与潜在向量平移结合，在指定刺激/感染数据中研究跨细胞类型、研究和物种的响应预测。
- **适用范围/局限**：可迁移潜在偏移是一项结构假设；原研究不证明任意未见 CRISPRi 靶点的响应可由同一偏移得到，也不保证细胞异质性校准。
- **轴**：T、R、A、G、I。
- **项目检验**：先比较表达空间共享 Δ 与相同来源条件估计的潜在共享 Δ；共享项与背景修正分开。固定 mean response 比较 NTC 重采样、参数计数采样和条件残差生成；避免把均值模型和生成器同时替换后归因于分布学习。

### L10 — 因子化条件生成及其可组合边界：CPA

**正式版本**：Lotfollahi M et al. *Predicting cellular responses to complex perturbations in high-throughput screens*. **Molecular Systems Biology 19**, e11517 (2023). DOI: [10.15252/msb.202211517](https://doi.org/10.15252/msb.202211517)。[作者论文全文](https://pmc.ncbi.nlm.nih.gov/articles/PMC10258562/)。

- **原研究事实**：CPA 学习 basal state、扰动及协变量的可组合潜在表示，经非线性解码预测未测组合、剂量等条件；论文也指出训练中缺乏某成分的组合信息可能产生不可靠外推。
- **适用范围/局限**：未见条件组合不等于完全未见基因身份；论文对新药使用化学先验的扩展不能自动变成 CRISPRi 靶点能力。
- **轴**：T、R、A、L、G。
- **项目检验**：CVAE/因子化条件生成可作为候选，对照匹配的 shared+context 模型与同条件计数生成器。S1/S3 必须明确未见靶点编码；若只能学习离散 ID，该实现只参加允许已见靶点的比较。比较均值、方差和扰动特异指标，并在固定响应后单测采样/校准。

## 据此初始化的决策边界

以下是项目设计推论，不冒充已发表结论或本地效果：

1. **先冻结可识别的问题。** 五背景不是五份同分布重复；先登记背景—study—guide—target 面板与实际缺失模式。在可用交集上建立 S2，在真实未见靶点上建立 S1/S3；S4 只有独立 study 可留出且设计可辨识时才能独立解释。不能将 NTC bags 或随机种子当成更多背景。
2. **首轮用于建立对照和判定瓶颈。** 数据完整性/有效评估 → 强简单对照 → 质量与数量/覆盖归因 → 先验及表示 → 背景条件化 → 容量/目标/优化。顺序是成本与识别性的安排，不是预先裁定方法优劣。
3. **保持九轴开放。** 本组十篇没有直接验证 Flow Matching、Set Encoder、FiLM、NTC 适配或混合专家在五背景上的优势；这些保留为分享驱动的未验证候选，不能借相关论文包装成已支持。架构、损失、优化联动无法拆开时标为整包比较。
4. **正反证共存。** GEARS/scGPT 原论文与 L03/L04/L08 的批判性研究说明任务、表示、指标及比较器会影响结论。因此账本应保留双方适用范围，并导出匹配对照，不把任何一方变成全局排除规则。
5. **外部依据只生成研究债务。** 首次初始化的本地执行节点和本地效应证据应为空；DAG 可先登记有前置条件和失败分支的研究计划。只有冻结数据/协议/配置并实际执行后，才产生本地比较证据及方法采用/搁置决策。
