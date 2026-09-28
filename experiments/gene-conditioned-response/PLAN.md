# C-GENE-DECODER-PACKAGE：跨基因条件响应解码

本路线检验共享功能解码能否扩展 target×gene 预测并提升官方评价。方法为 `M-GENE-DECODER-PACKAGE`，独立拟合 `gene-decoder-s01`，证据 `E-GENE-DECODER-PACKAGE`；与 `cell-cvae-s01` 在任何新结果出现前共同冻结。两者各一个完整配置，不按第一条结果改变第二条。资源为两个完整训练和四次正式评分；第三槽保留给达标方案的新背景确认。若两者均失败，另选机制而不自动追加 H1 网格；下一机制需要单独冻结，不能在本轮结果出来前假定其参数。

比较是整包 integrate，允许任务、表示、架构、采样、目标和泛化联动；不声称单因素收益，也不要求先通过线性头或完整先验消融。该方案能改变已有来源响应，也能对没有来源测量的 readout 给出有约束的预测，因而值得一个探索槽。[L03](../../docs/research/literature_foundations.md#l03--简单低秩及迁移基线ahlmann-eltze-等) 和 [L05](../../docs/research/literature_foundations.md#l05--go共表达图是可测试先验gears) 仅是方法假设依据；这不是 GEARS 复现。

## 固定数据与来源

S2-H1/context-seen/seed1。K562、RPE1、HepG2、Jurkat 四背景、两个来源研究，17,040 背景×靶点任务、9,867 靶点、2,541,598 扰动细胞。H1 只用 18,400 个 input NTC，独立 score NTC 与扰动真值只进入正式评分及明示的事后诊断。训练内实际测量并集 10,916；输出官方完整 18,533 轴，本地 H1 真实测量 18,074。缺测不作监督零。主面板 280、官方重叠 25；H1 已用于选择，仅为开发证据。

复用 init-linear 的 35 个输入/参考 bundle 文件，逐项哈希核验；控制为独立的 `masked-response-shared-s01`，其 config/checkpoint/prediction/两个面板结果均核验。不得使用 exp007 分数。官方 scorer 固定 cell-eval2 `5e64833518a6603a0301cbe28185d49c30f4a986` / vcc2026，六项 raw、normalized、Overall 和有效数原样保存，不裁剪负分、不换 anchors。每靶生成 400 细胞，seed+冻结靶点索引；`linear` 只是兼容评分接口的候选槽名。

## 表示、解码与训练

外部功能来源为 2026-09-14 下载且 SHA256 固定的 Reactome GMT 与 HGNC；文件未声明 release number，不伪造版本。完整官方轴上映射人类模块，保留 5–500 基因模块并去除同成员重复模块：10,567 基因、2,003 模块、87,672 边，其中 3,694 基因无训练背景直接测量。度数/大小归一化的 incidence 乘固定 seed1729 的 64 维 Gaussian sketch，再逐基因 L2 归一。未注释基因使用零功能向量、显式覆盖标记及 NTC 状态；覆盖不是有效证据。

基因共享特征为 64 维功能、注释标记、五背景 input NTC 均值/测量标记，共 75 维。允许 H1 input NTC 的无监督转导状态描述，没有 H1 扰动标签或梯度适配。背景特征为训练共同实测基因 NTC 均值的固定 32 维投影。靶点 query 联合上述基因特征、背景投影、当前 NTC 靶基因表达/测量标记和已见靶点 64 维 ID；readout key 联合基因特征和当前 NTC 表达/标记。两个共享 MLP 隐层128、SiLU、输出64，点积乘 `0.1/sqrt(64)`；没有每个未测输出基因的自由参数。

训练监督为各背景平均细胞 log1p CP10k 响应。每步均匀抽背景，再独立均匀抽靶点/实测 readout，共4096标量；16,384步，最后 checkpoint，不根据 H1 或内部指标选步数。令 t=真值/0.1、p=预测/0.1，损失为 `mean((1+4*tanh(abs(t)))*(p-t)^2)` 加 0.02 倍活跃坐标的 `softplus(-p*sign(t))` 均值，活跃门槛为原始 |Δ|≥0.05；所有权重只依赖训练标签。AdamW(lr.001,wd.0001,betas.9/.999,eps1e-8)、梯度裁剪1、float32、确定性算法。每512步保存网络、optimizer、采样 RNG、CPU/CUDA RNG、步数和完整配置哈希。

预测为 `f(H1,target,gene)+weighted_mean_source(Δsource-f(source,target,gene))`，权重为原始细胞数、只在各来源真实测量坐标上聚合。无来源坐标直接使用共享函数 f(H1)；全局未见靶点使用功能/NTC query 和零 ID 向量。最终未测输入基因保持显式零完成，不声称这是观测真值。生成器沿用固定 NTC 重采样、log响应平移、文库恢复和随机取整，因而方法 I 轴冻结为 I6。零扰动生成应回到原 NTC 重采样；独立 score NTC 上做辅助 Welch/BH 漂移诊断，不替代官方 DE 指标。

失效条件包括功能关系不代表干预关系、缺注释回退不能区分基因、H1 NTC 状态外推失真，以及来源残差在新背景不稳定。整包若有价值，再做匹配随机/无功能先验与残差锚定归因；不把这些作为本轮启动前置。

## 结束、决定与复现

完整训练后生成全轴预测并完成两面板官方评分。相对 shared 主 Overall≥+0.02 且重叠≥−0.02 才记开发 signal；单 seed/背景不确认泛化。不达标只否定该实现和配置。联合报告两条候选的 raw/normalized 六项、配对靶点差及波动、来源覆盖分层、NTC 诊断、实际耗时/内存。新覆盖基因的误差可事后分层诊断，但不能据此调整本轮配置。

入口 `./reproduce.sh --run-id gene-decoder-s01`，同条件完整状态恢复加 `--resume`。execute 在环境同步前门禁；train 解析全配置后 bind，再创建输出/W&B。代码/锁文件/全部配置共同提交后才正式训练。保存模型与结果 Artifacts，REPORT 完成分析后 close E-GENE-DECODER-PACKAGE，更新 Method Space、DAG、Ledger 和资源选择。
