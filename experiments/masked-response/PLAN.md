# C-MASKED-RESPONSE：保持已监督 shared 的响应补全

## 问题与来源

在强 shared 控制上，只补全训练缺少 target×gene 监督的位置，能否提高冻结的官方评分？真实通路结构是否比等维低秩或匹配随机通路更有效？这是新的响应空间假设，不能由前轮 NTC 表示阴性推断缺测是其失败原因。

依据为初始化文献卡 L03（简单/低秩响应）与 L05（匹配真实/随机先验的研究问题）；本实现不是论文模型复现。Reactome GMT 来自 [官方数据接口](https://reactome.org/download-data)，本地 2026-09-14 来源记录没有可靠 release 号，按完整文件 SHA 冻结，不猜版本。程序投影、未中心化响应 SVD 和掩码重构的具体组合是本项目待检验设计。

## 数据与边界

S2-H1/context-seen/seed1。四训练背景 K562/RPE1/HepG2/Jurkat，来自两个研究；H1 为留出背景，已用于模型选择，所有结果属于开发证据。主面板 280 靶；官方重叠 25 靶。训练响应只读允许的四背景统计量，H1 NTC 用于固定生成器；H1 扰动真值只由评分器读取。

沿用 PSEUDO-VCC-INIT 的完整 benchmark binding、物理隔离 NTC、log1p CP10k 表达空间、官方 18,533 输出轴与 H1 实测评分轴。四背景共 10,916 个测量基因；Reactome 按训练测量并集映射，10–500 基因的人类模块、重复成员集合按 stable ID 去重，覆盖 6,832 基因、1,372 模块、65,816 边。这些数来自无模型效果的输入覆盖检查。训练靶点并集 9,867、四背景共同靶 2,387；低秩实际可用行以本次完整掩码为准，不靠硬编码名单。

输入/bundles 从 init-linear-s01 的固定来源文件逐项核验并复制到各 run/cache；不复用模型权重。新 shared 逐靶点、逐测量模式重新计算，且所有臂与原 checkpoint 的 shared/targets/genes/trained_mask 要逐值相等，否则停止。原 zero/shared/source 参考分数只有通过完整来源审计后才可复用；四个本轮预测均重新生成、重新评分。

## 四臂与唯一实际配置差异

| run_id | representation.kind | 缺失坐标处理 |
|---|---|---|
| masked-response-shared-s01 | shared | 零 Δ 回退，新拟合基线 |
| masked-response-pca-s01 | pca | 32 维训练响应 SVD 的掩码 ridge 重构 |
| masked-response-program-s01 | program | 32 维真实程序基的相同重构 |
| masked-response-random-s01 | random_program | 32 维匹配随机程序基的相同重构 |

全部直接监督 shared 坐标保留，零响应是真实观测时也保留。仅填补上述 6,832 基因中当前靶点没有任何训练背景监督的位置。无通路覆盖、所有训练背景均缺测、全局未见靶点都保持原 shared 的零回退。不缩小输出或评分轴。

shared 与原实现一致：各背景平均细胞 log1p CP10k 响应，以扰动细胞数/100 加权，缺测背景排除；无背景条件特征。每个靶点重构解 `argmin_z sum_observed (B_g z - shared_g)^2 + 0.1 ||z||²`，列正交基 B、无中心项、无截距、观测基因等权。该目标只用于补全，重构值不会覆盖直接监督。

低秩：仅对该覆盖轴上没有任何缺测坐标的训练靶点 shared 行做**未中心化**截断 SVD（保留零响应原点，不是中心化 PCA），不将缺测补零后拟合。每靶点等权；固定 seed=1729、rank32、oversample16、2 次 power iteration；不做维数或正则搜索。只用完整行限制了学习人群，结果不能归因为纯数学降维优势。

真实/随机：归一化二部图 `D_gene^-1/2 B D_module^-1/2`，同 seed 的高斯 sketch32 后 QR。随机图每边 10 次成功 double-edge swap，逐基因度数、逐模块大小、覆盖和秩均保持；保存混合次数及边重叠。它检验此通路投影中的成员结构，不代表所有生物先验。

R 轴描述整个输出表示/补全选择；shared→低秩/程序引入表示和相应重构约束，不能拆称纯表示维度因果。真实→随机保持相同求解、正则、覆盖、秩及图度数，可检验结构贡献。config 除身份外只变 representation.kind。

## 生成、评估和预定判据

复用冻结的 NTC 重采样→log1p 加 Δ→非负截断→恢复文库→随机取整生成器；seed+靶点固定索引，400 细胞/靶。**保持已监督 Δ 不保证这些基因最终计数不变**，因为文库恢复跨基因耦合。固定 cell-eval2 commit 5e64833518a6603a0301cbe28185d49c30f4a986、vcc2026 preset、原 reference/baseline/anchors。两个面板报告六项 raw/normalized、Overall、有效数、配对靶点效应和波动；主面板区分 181 个 K562-only 与 99 个四背景支持靶。预期后者补全掩码为空，可作为不变性核验。

在查看任何本轮分数前冻结：候选相对新 shared 主面板 Overall 至少 +0.02，且官方重叠面板至少 −0.02，才记录候选开发 signal。真实通路的结构性支持还须主面板比 max(PCA,random) 至少 +0.02，且重叠面板相对该最强控制至少 −0.02。不满足阈值不能写成零效应；单 seed、单背景不确认泛化或 SOTA。shared 新旧主/重叠 Overall 应在 1e-8 内，且原始 shared 响应严格相同；若不复现先排查正确性，不能以错误控制闭环。

有信号：优先到新背景或官方面板确认，不追加 H1 维数/正则搜索。无信号：保留 shared，结束这次输出补全筛选，推进预留的 QC/匹配随机数据质量问题。真实通路输给随机只否定当前注入方案。无效执行或缺少有效分数：恢复/修复，不作方法阴性。

## 执行与结束

本 Experiment 独立 uv 环境和锁文件，四个 run 顺序执行、共用代码。正式入口 `./reproduce.sh --run-id <ID>` 在同步环境前通过 research execute，训练在写入输出/W&B 前 bind。预处理及固定评估模块通过 src/runtime.py 复用已冻结的 init-linear 实现，DAG code_refs 全覆盖，不改历史代码。解析完成的 shared、mask、basis、completed effects、config 哈希完整存到 response.npz；`--resume` 只接受完整匹配状态，已完成预测/评分由哈希恢复。

预计每 run 一个解析拟合、两个完整官方评分；无模型搜索。资源上限按现有设备：CPU 8 线程、单 GPU，顺序释放内存，四 run 约 60–100 GB 新产物，可用盘约 260 GB；预算不是提前停止条件。全部四臂保存 checkpoint、全轴预测及清单、原始评分、实际配置/运行版本、W&B 版本化结果，REPORT 分析完成并 close E-MASKED-RESPONSE、更新下一步、提交后才闭环。
