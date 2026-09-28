# exp005：跨 context 的 XGBoost residual response

## 已确认的问题与数据单位

以 H1 hESC、K562 GWPS、RPE1、HepG2、Jurkat 五个完整 context 为单位，学习
`delta(c,t,g) = log1p(CP50K(sum counts of perturbation t)) - log1p(CP50K(sum NTC counts))`。
预测用目标 context 的完整 NTC baseline 加 residual，再生成单细胞整数 counts。
本路线改变 exp001 的响应算子、NTC 使用、完整靶点评估、内层选择规则和调控先验；旧结果不改写。

训练使用每个训练 context 全部合格扰动靶点及全部已纳入细胞计算监督统计。
最低 50 个原始细胞；输入须为非负整数 counts，映射无歧义，有可测表达及 NTC。
采用各 context 的可靠原生基因轴；缺测不补零作为监督。不存在额外 20% 靶点保留。
H1 三个历史 split 合并，重复 NTC 只计一次。同 context 内不拆训练/验证或 NTC 输入/参照池。
技术 batch 与 guide 用于抽样分层，不能当独立 context。

原始数据、HGNC、STRING、Reactome 从登记来源验证 SHA-256；CollecTRI 由固定 SHA-256 快照登记。
旧 HOMEZ Ensembl ID 的显式映射有官方历史依据。TIAF1 与 MYO18A 在官方轴中均存在，
保留两个独立列，不能依据新版 HGNC 别名将它们合并；来源 gene ID 冲突排除并记录。
数据事实见 `docs/experiments/exp005-target-coverage-and-priors.md`。2025 H1 与 2026 面板均为 300，交集只有 25。

## 模型、增强与对照

一个跨 context / target / readout 共享的 XGBoost 标量回归器，输入完整 NTC 或 NTC bag 的
表达、检测率、二阶统计，target/readout 的 STRING embedding、功能/物理关联、Reactome 共同通路、
CollecTRI 有向激活/抑制关系及 NTC regulon 表达代理。先验不含本次验证或测试扰动标签。
不把 RNA 表达等同蛋白活性，不把网络权重当作 delta 幅度，不硬编码靶基因固定 knockdown。

固定完整 NTC 监督 baseline；另外生成 8 个各 400 细胞的分层 bag 作输入增强。
每个背景→靶点→bag→readout 等权。每个 target / bag 均匀选择 512 个可测 readout，
固定 hash 种子；9 个视图含完整 pool，全部 target 每次拟合都纳入。此为基因对监督采样，不是靶点划分。
超参数固定在配置，CPU hist / depth 6 / eta .05 / lambda 10；行列采样均为 1，便于完全恢复。

生成：每 target 分层抽 400 个对应 context 的 NTC 模板；把 `B_full + delta` 投影到合法 CP50K
轮廓，通过逐基因调整和迭代行列校准结合模板表达异质性及 library sizes，再随机舍入。
缺少直接测量的输出基因仍以 baseline 与先验特征预测。均值模型不能完整模拟新的细胞状态。
基线为相同生成协议下的零响应，以及仅用当前训练 contexts 的同靶点平均 response 转移。

## 嵌套留一、评分与结束条件

外层完整留一 context 为 test。其余四个 context 内做四折，每次三训一验；
内层 validation 与外层 test 均使用该背景全部合格靶点，不取交集、不限制官方 300。
模型选择同时包含已见与未见靶点；另外保存 target support 分层，外层结果绝不参与选轮数。

固定候选 checkpoint：1、16、64、128、256、384、512。三 context 模型均训练至 512，
每个 checkpoint 即计算两个未参与拟合背景各自的官方分数；三 context 组合只有 10 个，
复用模型不混用不同外层的选模分数。每个外层按其四个内层 validation 官方 Overall 等权均值
选轮数，平分取较早者，再从头用对应四个 context 训练到该轮数。途中 checkpoint 同样报告 test。
外层训练终点由内层预先确定，不以 test 早停。最终全五 context 重训取五个内层选定轮数的中位数。
共 10 个内层、5 个外层、1 个最终拟合；训练结束要求全部拟合、评估、导出完成，不以 smoke test 替代。

评分固定 cell-eval2 commit `5e64833518a6603a0301cbe28185d49c30f4a986` 的 vcc2026 preset，
CUDA/gpudge 后端固定。六项 raw / scaled / Overall 均由官方包计算，包括五次拆半 replicate anchors，
seed 0，及官方 mean-response baseline。官方锚点内部的拆半仅用于评分参照，不是模型 train/val 划分。
本地 reference 每 target 固定分层抽最多 400 个真实细胞（不足保留全部，不重采样补真值），
保留完整 NTC pool；所有 checkpoint 复用同一 reference、anchors 和预测 RNG。
训练统计仍来自全部合格原始细胞。这与旧 128-cell 缓存不同，不复用旧缓存或旧评价口径。
本地原生 gene / target 面板不同，分数不是官方 A/B/C leaderboard 的绝对尺度；PDS 不逐块独立排名。
官方拆半实现需要复制矩阵；进入 anchor 构建前按两份 reference 矩阵加 12 GiB 余量检查可用内存，
资源不足则同一 run 等待并记录原因，不缩减靶点、不改变指标、不干预其他活动实验。

## 运行、恢复与交付

`./reproduce.sh --config configs/default.yaml`；`./reproduce.sh --resume RUN_ID`。
一个 run 覆盖输入验证、预处理、全部嵌套拟合、评估和最终预测；一个 W&B ID，
entity yjcyxky / project virtual-cell-challenge / group exp005。本次状态和所有缓存位于 outputs/RUN_ID。
正式运行前提交源码、配置和锁文件；记录 commit、锁及实际运行时身份。恢复核对身份，
使用 Booster 完整内存快照、迭代位置、固定数据和采样身份及 RNG，不把仅模型权重加载冒充恢复。

输出模型、逐 checkpoint 官方指标与 response 预测、最终外层细胞预测、内层选轮数证据、
全量最终模型和 A/B/C 的官方 300 targets × 400 cells × 18,533 genes 提交文件。
统一入口默认准备并验证提交文件；显式 `--submit` 使用同一 run 发布并归档官方结果。
最终模型/指标上传版本化 W&B Artifacts；原始训练与 NTC 数据不上传。
REPORT 汇总逐背景、已见/未见靶点覆盖、基线、限制及 W&B 链接；完成代码清理后交付。
