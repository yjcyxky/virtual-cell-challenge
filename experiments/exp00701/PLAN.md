# exp00701：跨背景实测响应与 DEG 特征

研究问题：在 exp007 的 p/g/知识/context 特征之上加入其他背景同靶点的实测响应、DEG 方向证据与整体响应表示，能否改善 H1 留出扰动特异性和方向指标，同时避免幅度指标退化？

## 对照与训练

- 历史对照：exp007 run `20260927-exp007-h1-log2fc-s17`，已完成 512 轮和七个 H1 检查点。直接引用其已保存结果与 zero/shared-response 基线，不重复拟合。数据、采样、种子、树参数和生成器完全保持一致；固定引用及 SHA-256 在 `configs/source-reference.json`。
- 三个新增独立 runs，依次使用 `configs/continuous.yaml`、`configs/deg.yaml`、`configs/modules.yaml`；每组从头初始化 XGBoost，512 轮，不提前以成绩决定结束。检查点仍为 1/16/64/128/256/384/512。
- 四训一验：K562、RPE1、HepG2、Jurkat 训练；H1 唯一留出与选轮数。H1 是开发验证，不是独立测试。不开展完整五折或全背景拟合，不在本任务提交官方榜单。
- 标签保持全部合格细胞的逐细胞 CPM 算术均值 log2FC，epsilon=1e-9；原加权平方误差、depth=6、eta=.05、lambda=10 等参数不变。

## 特征与泄漏边界

每个接收 context 的特征只能使用其余训练 context 的响应；H1 永远不是响应来源。训练侧排除当前 context 是特征构造规则，不是新增五折模型训练。数据保留缺测标记；非显著不等于零响应，不按 DEG 数筛除靶点。

1. continuous：在原 70 维特征上增加 7 维：其他背景同 `(p,g)` 的 full-cell LFC 均值、标准差、最小/最大值、平均绝对值、可用背景数、同靶点可用背景数。无来源时数值填 0 并保留支持数 0，区别于实测零。
2. deg：增加 10 维：被 DE 检验的来源数，上/下调支持数及其比例，净方向，方向冲突，平均/最大截断 `-log10(q)`（上限 12），来源扰动细胞数的 log1p 均值。显著性 BH<.05，方向取来源 DE 表的 log2FC。
3. modules：再增加 82 维。固定 Reactome membership 中大小 10–300 的通路经列大小归一化、16 维确定性 TruncatedSVD 得到基因模块坐标；坐标不使用任何扰动标签。同靶点来源上/下调列表分别投影、按列表长度平方根归一化后跨来源平均，附加 log1p 列表长度；加入 readout 坐标和上/下调投影与 readout 的逐维交互。不同 context 共享同一固定坐标，避免独立拟合旋转不一致或当前背景标签进入模块字典。

这是固定通路坐标上的实测响应表示，并非端到端学习响应编码器。该选择让消融聚焦在特征增益上。连续幅度与 DE 证据的来源细胞数不同：连续幅度来自 full-cell statistics；DE 沿用固定 <=400 扰动细胞/全部 NTC 的官方 Wilcoxon、CPM、每扰动 BH。两种信息均显式记录，不将未检出解释为零。

DE 仅处理四个训练 context。K562/HepG2/Jurkat 引用已验证的真实 DE 表；核对来源数据、参数与 SHA-256。RPE1 在对应 run 内从固定 evaluation rows 计算；所有派生产物在本 run 的 cache。被继承的源 data/priors/H1 reference 只读引用，scorer 会写入的 real-cache 与 bundle 复制到当前 run，避免改动历史产物。

## 验证、评估与结束

- 必须验证：修改 H1 或接收 context 的响应/DE 不改变相应特征；来源响应会改变特征；缺测与实测零不同；方向冲突保留；模块坐标不含响应标签；XGBoost 完整状态恢复与正式官方评分仍通过。
- 每组完成全部七个 H1 官方方法六指标评估，保存模型、预测、原始和归一化指标。按相同 H1 Overall 选轮数，另比较固定第 512 轮，避免仅看各自最优。
- 重点报告 PDS/Fidelity/Reach/Jaccard，同时报告 MSE/NMAE；整体增益不能仅由某一指标换取。H1 269 个训练已见靶点与 28 个未见靶点分别报告辅助误差与可用的逐靶点指标；不得将分层指标当作新的官方总分。
- 检查预测/生成后 LFC 幅度误差和训练曲线，说明连续响应、DE 证据、模块各自的增量。源靶点覆盖不均、四个训练背景有限、NTC 与显著性检验功效差异作为局限。
- 每个 run 独立 W&B，entity=yjcyxky、project=virtual-cell-challenge、group=exp00701。保存完整训练 checkpoint、特征身份、模型与必要响应证据，并上传版本化 Artifacts。三个 runs 均有效训练/评估结束并汇总 REPORT 后任务完成，效果低于基线也是有效结果。

入口：`cd experiments/exp00701 && ./reproduce.sh --config configs/continuous.yaml --run-id RUN_ID`；其余配置类同。恢复使用 `./reproduce.sh --resume RUN_ID`，完整恢复训练状态并继续尚未完成的评分，禁止只恢复模型权重或混入改变后的条件。
