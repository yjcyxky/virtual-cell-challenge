# S2-H1 kNN 官方核查方案

## 2026-10-03 用户指定的 kNN 官方核查

用户明确要求使用 H1 折 kNN 生成 VCC 并提交 leaderboard，并已选择 **S2-H1：`depmap-knn-s2-h1-s930`**。本次按该指令执行一次固定模型的官方测量，保留原批次“不晋级”的方法结论，不追加训练或调参。在对应原节点追加 `official_evaluation`，以 `C-DEPMAP-KNN-H1-OFFICIAL` 登记；S3-H1 不生成或提交。

导出读取原 checkpoint 内的 Shared 响应、支持掩码、DepMap 原始余弦描述符和源模板；不重算来源统计、不拟合 PCA、不重训。先逐元素重放原 H1 面板保存的全部 latent 响应（S2 为 280 靶、S3 为 70 靶），要求完全一致。保持 k=5、原响应中心化、全轴未监督读出零 latent、固定自身 bulk 残留 20%、CP50K 组成解码、原 population emitter 及 seed 930。原始训练代码通过执行提交快照保留；导出代码、输入和配置须另冻结并提交。

只使用完整官方 A/B/C NTC，每背景 18,400 个细胞和完整 18,533 基因轴，预测官方 300 靶点各 400 细胞，总计 360,000 个整数 counts 细胞。每靶 seed 延续原函数 `stable_seed(930, outer_split, context, target, 'prediction')`，context 使用官方 A/B/C 原标签。两折不混合、不集成。官方输入文件、checkpoint、配置、代码、运行时及预测哈希共同约束恢复，沿用原 W&B run 和同一上传 entry。

评分前诊断覆盖每背景×全部靶点的 counts/深度/检出/零率、raw 均值/方差/Fano、CP10K 方差、重复与 NTC 行一致率、输入 NTC 上选取的 128 个非面板靶基因协方差、自身残留、上下调比例、低表达覆盖；分别保存平均 log 响应、先均值后 log 的 LFC、共享能量与去共享有效秩。报告同靶 seen/unseen、真实邻居与背景来源、无先验模板回退、全局监督/未监督读出；固定敲低和继承 NTC 形态不计作学习能力。

每背景做 30 次两个不交叉 n=400 NTC 子样本的统计参照。零响应诊断把公开 NTC 固定分为 1/2 生成输入、1/4 DE 参考、1/4 real-null 池，三次重复，核验实际解码/生成器与 NTC 重采样逐元素恒等，并调用冻结官方 DE 比较 zero/input/real-null；新增伪 DE 应为零。正式预测仍消费全部官方 NTC，上述子划分仅用于诊断。每背景另对固定身份哈希选出的 4 靶做 3 次独立非零生成，保存群均值波动；不把抽样重复视为生物重复。

完成新预测诊断且 counts、基因顺序、上下文和细胞数硬约束通过后，记录“按用户请求提交”的判断，执行官方 `vcc prep` 并提交一次。统计异常完整保留，不通过删靶、改先验、调生成器或选生成 seed 改善外观。诊断失败则暂停提交并修复；旧本地完整六项不重算。保存 VCC SHA-256、entry_id、panel/anchor_version、全部 raw/normalized 六项和 Overall，以正式 published 回执为完成条件；评分失败或尚未发布不能当作模型阴性结论。

本次前瞻问题是固定 kNN 的官方 Overall 是否大于 0，即在本次官方 normalized 标尺上是否高于其聚合 baseline；同时描述完整六项及与原 S2-H1 的差异。大于 0 只支持该次单文件、单 seed、官方验证面板上的测量判断，不能撤销原本地整包未晋级结论；小于等于 0 不支持这个有限主张。隐藏真值、逐背景有效数及线上参照配方未公开时保留未知，不把本地/官方的同时变化归因为单个因素。

执行入口：`cd experiments/depmap-response && ./submit.sh --config configs/official-knn-s2-h1-s930.json --submit`；不带 `--submit` 则完成诊断、VCC 打包与校验。所有新产物位于原 run 的 `predictions/official-abc-knn/`。

KnowGraph attention: `DATA-DEPMAP`、`ST-WILCOXON`、`MY-EFFECT-EMITTER`。本次检查先验覆盖和回退、均值与分布诊断、同 n 参照；不改变原模型，也不根据本地与榜单的数值接近认证线上 r4 配方。隐藏扰动真值缺失，预测相对 NTC 的 DE 不称伪 DE；官方反馈属于开发证据，不作 SOTA 或跨背景稳定改进主张。
