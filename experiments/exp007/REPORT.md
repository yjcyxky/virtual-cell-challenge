# exp007 结果

Run `20260927-exp007-h1-log2fc-s17`，状态：completed。

W&B：https://wandb.ai/yjcyxky/virtual-cell-challenge/runs/20260927-exp007-h1-log2fc-s17

目标：逐细胞 CPM 算术均值的 log2 fold change（epsilon=1e-9），加权平方误差。

划分：K562 / RPE1 / HepG2 / Jurkat 训练，H1 唯一留出背景；一次拟合至 512 轮。

H1 Overall 选择检查点，因此 H1 是开发验证，不是独立测试；原始训练阶段不做五折、全背景重训或榜单提交。训练完成后，用户追加授权直接导出第 512 轮并提交官方 A/B/C 评分。

评分：固定官方六项指标、dispersed 基线、排除自身靶基因；本地原生面板分数不等同 A/B/C 榜单。

已完成 H1 checkpoint 评分：7/7。

| 轮数 | Overall | 靶点数 | 训练中已见 |
|---:|---:|---:|---:|
| 1 | -0.063721 | 297 | 269 |
| 16 | -0.024825 | 297 | 269 |
| 64 | -0.003451 | 297 | 269 |
| 128 | 0.001575 | 297 | 269 |
| 256 | 0.011650 | 297 | 269 |
| 384 | 0.016700 | 297 | 269 |
| 512 | 0.020419 | 297 | 269 |

对照 zero_response：Overall -0.062623。

对照 shared_response：Overall 0.186772。

选定轮数：512；H1 Overall：0.020419。

相对 zero_response 的 Overall 差值：+0.083043。

相对 shared_response 的 Overall 差值：-0.166353。

选定模型：`checkpoints/HepG2+Jurkat+K562+RPE1/round-0512.ubj`；预测：`predictions/HepG2+Jurkat+K562+RPE1/H1/model-0512/`。

W&B Artifact：`yjcyxky/virtual-cell-challenge/exp007-20260927-exp007-h1-log2fc-s17:v0`。

W&B 日志同步：online；Artifact 同步：uploaded。

与 exp006：沿用特征、采样、树参数和官方评分；标签改为 log2FC，生成器相应校准非深度加权的逐细胞均值。只比较相同 H1 面板与评分协议，不能把单背景选模分数与五折均值直接比较。

限制：H1 曾用于项目开发且本次用于选轮数；均值与 NTC 模板不能完整模拟新细胞状态。独立基因 LFC 经总量约束投影后会变化，保存逐靶点生成前后 LFC 误差诊断；低表达/零表达 LFC 可能较大，仍保留可测基因标签，不静默截断。NTC bag 不是新背景，未测基因不作零标签。
