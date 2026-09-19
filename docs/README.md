这份材料的核心是 **2026 Arc Virtual Cell Challenge 的任务定义、数据格式和可用训练资源**。如果只提炼真正影响建模策略的信息，可以压缩为下面这些：

1. **2026 是一个真正的 zero-shot perturbation prediction 任务。**
   官方**不提供 2026 训练集**。模型只获得某个未知 cell context 的未扰动 non-targeting control 表达谱，以及需要进行 CRISPRi knockdown 的 target gene，然后预测 perturbation 后的单细胞表达谱。可以自行使用 2025 VCC 数据以及任何合法的外部数据训练。

2. **最大的泛化难点是 cell context 完全未知。**
   2026 共使用 **6 个来自不同组织的 cell lines**，但真实细胞系身份不会公开。Validation 是 A/B/C，final test 是另外三个完全不同的 D/E/F。模型只能通过该 context 的 **unperturbed expression profile** 推断它是什么样的细胞状态。
   因此本质上不是简单的：
   **gene perturbation → expression**，
   而是：
   **baseline cellular state + perturbation → perturbed cellular state**。

3. **Validation 和 Final Test 是真正的跨细胞背景泛化。**
   Validation A/B/C 在 2026 年 8 月 20 日发布；Final test D/E/F 在 10 月 22 日发布，而且**最终排名完全由 final test 决定**。Validation 和 test 使用不同 cell lines、不同 perturbation panels，所以两个阶段的绝对分数也不能直接比较。

4. **每个阶段的数据规模非常明确。**
   每轮包括：

   * 3 个 cell contexts；
   * 每个 context 预测相同的 **300 个 gene perturbations**；
   * 每个 perturbation 需要生成 **400 cells**；
   * 总计 **360,000 predicted cells**；
   * 每个 context 提供 **18,400 个 control cells**；
   * 共 **18,533 genes**。

5. **Control 数据实际上承担了“识别 cell context”的作用。**
   每个 context 有 46 个 non-targeting guides，每个 guide 400 cells。所有 control cell 的 `target_gene` 都是 `non-targeting`，但保留 `ntc_id`，因此可以利用不同 NTC guide 之间的 variation。官方明确指出，这些 control cells 是模型输入，用于定义/识别 cellular context，不能放入预测结果。

6. **目标 perturbation 都是高质量 CRISPRi knockdown。**
   所有 target genes 的 on-target knockdown 均 **>80%**，这意味着比赛希望模型预测的是明确、有较强干预效应的 perturbations，而不是弱干预噪声。

7. **输出要求不是均值，而是完整的单细胞分布。**
   最终需要输出一个 AnnData H5AD，其中每个 target gene × context 都需要 **400 个 predicted cells**，总共 360,000 cells × 18,533 genes。输出必须是 **raw counts、非负整数、finite**，并有严格的 sparse-storage 限制。
   这意味着模型不仅要预测：

   $$
   E[X\mid context, perturbation]
   $$

   还必须能够生成合理的 **cell-to-cell variability / expression distribution**。

8. **2025 VCC 数据可能是最直接的基础训练资源。**
   2025 数据来自 **H1 human embryonic stem cells**，同样使用 CRISPR perturbation、深度单细胞测序，而且使用与 2026 相同的 **10x Flex chemistry**。它包含 training、validation 和 held-out test perturbations，因此是官方明确推荐的训练来源。

9. **官方尤其推荐大规模外部 perturbation 数据来解决 zero-shot 泛化问题。**
   Arc Virtual Cell Atlas 总体包含 **超过 6 亿 cells**；其中：

   * **scBaseCount**：>502M cells、27 organisms、75 tissues，主要提供广泛 observational context；
   * **Tahoe-100M**：100M cells、约 60,000 个 drug perturbation experiments、50 个 cancer models、1,100+ drugs；
   * **Replogle 2022**：>2.5M cells，K562/RPE1 genome-scale CRISPRi，而且 K562 perturbations 与 VCC targets 有大量重叠；
   * **Nadig 2025**：Jurkat + HepG2 CRISPR screens；
   * **Jiang 2025**：A549、MCF7、HT29、HAP1、BxPC3、K562 六种 cancer cell lines 的 Perturb-seq。

10. **从建模角度，这个挑战真正考的是三个能力：**

**① Cell-state representation**
从大量 control cells 中建立未知 cell context 的 latent representation。

**② Perturbation transfer**
从其他 cell types / datasets 学到“敲低一个基因会怎样改变系统”，再迁移到从未见过的 context。

**③ Distribution generation**
不只是预测平均差异，而是生成 400 个统计上合理的单细胞 profiles。

因此，我认为这份规则里**最关键的一句话**其实是：

> **2026 VCC = unseen cell context × gene perturbation → zero-shot single-cell state generation。**

这也意味着，你此前提出的 **“先利用大规模多细胞背景学习基础 biological representation，再学习 perturbation dynamics，最后根据目标 control expression 做 context adaptation”**，在任务定义上是非常对路的。尤其值得重视的是：**2026 并不是要识别 A/B/C 分别是什么细胞，而是要让模型仅凭 baseline expression 就能够条件化 perturbation response。**

你更想继续看这份材料的建模方案，还是先整理成一页式比赛要点？
