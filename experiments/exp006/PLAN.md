# exp006：跨 context 的 XGBoost residual response

## 已确认的问题与数据单位

以 H1 hESC、K562 GWPS、RPE1、HepG2、Jurkat 五个完整 context 为单位，学习
`delta(c,t,g) = log1p(CP50K(sum counts of perturbation t)) - log1p(CP50K(sum NTC counts))`。
预测用目标 context 的完整 NTC baseline 加 residual，再生成单细胞整数 counts。
本路线沿用 exp005 的模型、数据和生成算法，按用户要求新建 Experiment，改用单层留一背景开发验证；旧结果不改写。

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

## 留一背景选模、全量重训与结束条件

固定五折：每折完整留出 H1、K562、RPE1、HepG2、Jurkat 之一作开发验证，
其余四个背景训练。每折验证该背景全部合格靶点，不取交集、不限制官方 300。
模型选择同时包含训练中已见与未见靶点，保存 target support 分层结果。

固定候选 checkpoint：1、16、64、128、256、384、512。五个四背景模型均训练到 512，
每个 checkpoint 计算留出背景官方 Overall。按五背景等权平均选出一个全局轮数，
平分取较早轮数；禁止先按折挑最好分数再汇总，不按细胞数或靶点数加权。
全部五背景从头训练到该统一轮数，用最终模型生成 A/B/C 并提交 leaderboard。
五个 CV 模型加一个最终模型，共六次拟合；35 次模型背景评分和10次对照评分。
各折保存 checkpoint、完整训练恢复状态、response 预测和官方指标；最终保存提交用细胞预测。

本地留一成绩是用于选模的开发验证结果，不宣称独立无偏测试成绩。
Leaderboard 反馈不参与本 run 轮数选择；若后续参考它改模，明确记为开发用途。
结束要求六次拟合、全部评估、最终导出、官方提交及评分返回、Artifact 归档完成。
无法提交或官方评估失败须如实报告阻塞，不把本地预检或启动称为完成。

评分固定 cell-eval2 commit `5e64833518a6603a0301cbe28185d49c30f4a986` 的 vcc2026 preset，
CUDA/gpudge 后端固定。六项 raw / scaled / Overall 均由官方包计算，包括五次拆半 replicate anchors，
seed 0，及官方 mean-response baseline。基线只通过官方 `generic_response_profile` 与
dispersed emission kernel 构造：`exclude_target_gene=True`、传递官方 target_gene_map、
`emit="dispersed"`、固定 `baseline_seed=0`；不手写平铺基线。
评估协议标识为 `cell-eval2-dispersed-exclude-target-v1`，随 run 配置、reference、bundle 基线记录及评分落盘。
官方锚点内部的拆半仅用于评分参照，不是模型 train/val 划分。
本地 reference 每 target 固定分层抽最多 400 个真实细胞（不足保留全部，不重采样补真值），
保留完整 NTC pool；所有 checkpoint 复用同一 reference、anchors 和预测 RNG。
训练统计仍来自全部合格原始细胞。这与旧 128-cell 缓存不同，不复用旧缓存或旧评价口径。
本地原生 gene / target 面板不同，分数不是官方 A/B/C leaderboard 的绝对尺度；PDS 不逐块独立排名。
reference 与本地预测按小块写成数值无损的文件映射 CSR，保留完整行列轴、细胞顺序与 counts dtype。
官方 dispersed 基线使用固定版本的 `_emission_scale` 与 `_emit_scaled_resample` 分块写盘，
沿用完整 NTC 池、官方排序和一个连续 RNG；逐元素及诊断字段对照 `build_baseline_prediction` 验证。
不复制公式，不修改官方库。这样避免公开全量构造器的多份全尺寸临时数组。
官方拆半实现需要复制矩阵；两份半矩阵合计一份完整矩阵，另留半份切片复制瞬时空间。
进入 anchor 构建前按 1.5 份 reference 的实际稀疏存储字节数加 12 GiB 余量检查可用内存，
资源不足则同一 run 等待并记录原因，不缩减靶点、不改变指标、不干预其他活动实验。
超过整机容量时直接报错，避免永久等待。该检查是预估，不等于保证整个官方构造的峰值；完整运行仍须实测。
启动器在导入 NumPy/CUDA 前以进程级 `PR_SET_THP_DISABLE` 禁用透明大页，子进程继承；
运行身份记录实际 THP 状态及 NumPy 大页设置。主机全局设置和锁定依赖不变。

## 运行、恢复与交付

`./reproduce.sh --config configs/default.yaml`；`./reproduce.sh --resume RUN_ID`。
一个 run 覆盖输入验证、预处理、全部留一拟合、评估和最终预测；一个 W&B ID，
entity yjcyxky / project virtual-cell-challenge / group exp006。本次状态和所有缓存位于 outputs/RUN_ID。
正式运行前提交源码、配置和锁文件；记录 commit、锁及实际运行时身份。恢复核对身份，
使用 Booster 完整内存快照、迭代位置、固定数据和采样身份及 RNG，不把仅模型权重加载冒充恢复。

输出模型、逐 checkpoint 官方指标与 response 预测、统一选轮数证据、
全量最终模型和 A/B/C 的官方 300 targets × 400 cells × 18,533 genes 提交文件。
默认配置 `submit: true`，统一入口准备、验证、发布并归档官方结果；恢复沿用同一 run 和同一提交 entry。
最终模型/指标上传版本化 W&B Artifacts；原始训练与 NTC 数据不上传。
REPORT 汇总逐背景开发验证、已见/未见靶点覆盖、统一选模、基线、榜单、限制及 W&B 链接；完成代码清理后交付。

来源：exp005 run `20260926-exp005-nested-residual-s17` 与代码 `18fe6e2`，仅继承方法和源码。
exp006 使用自己的锁文件、虚拟环境及新 run，从登记原始输入重新预处理，不复用旧模型或 run 缓存。
相对 exp005：四背景训练替代嵌套内层三背景训练，估计目标改变，成绩不作同口径独立测试对比。

2026-09-26 用户要求停止并修正基线。旧 run `20260926-exp006-loco-residual-s17` 已停止，
其自建 tile/不排除靶基因的基线及既有 H1 分数按原口径保留，不能用于新协议选模。
修正后必须新建 run、重新完成五折选模与全量重训；禁止复用旧评分或旧基线锚点。
2026-09-26 用户授权重新训练并验证修复。新 run 从头运行，先执行 K562 留出折，
以尽早检验完整规模的官方基线、锚点和首个 checkpoint 评分；其后执行 H1、RPE1、HepG2、Jurkat。
只改变折的执行顺序，五折定义、候选轮数和等权选模规则不变。配置记录被替代的旧 run 与重启原因。
