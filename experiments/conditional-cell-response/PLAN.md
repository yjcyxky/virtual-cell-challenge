# C-CELL-VAE-PACKAGE：NTC 条件计数 VAE

本路线检验逐细胞生成与响应目标的整包价值。方法 `M-CELL-VAE-PACKAGE`，run `cell-cvae-s01`，证据 `E-CELL-VAE-PACKAGE`；与跨基因解码器共同冻结后顺序运行，各一个完整配置。共同数据、控制、scorer、两个面板、+0.02/−0.02门槛和第三槽资源规则见[同轮计划](../gene-conditioned-response/PLAN.md)。全部实际差异登记为 integrate，不要求先在线性头证明。[scGen/CPA 文献卡](../../docs/research/literature_foundations.md#l09--潜在空间均值转移scgen) 支持条件潜变量假设，本实现不是两篇论文的复现。

## 输入与表示边界

基础允许细胞、物理划分、完整官方轴、各来源测量掩码及评估均匹配 shared。复用 composition-lfc-s01 的固定算术 CP10k 一二阶矩作为输入统计，逐项验证哈希、标签、计数、基因位置及已完成来源；不复用该阴性模型权重。H1 只读 input NTC 矩。辅助响应损失与来源偏移使用全部允许细胞的统计。

逐细胞似然池来自四个训练背景原始 counts。每背景 seed=2718+背景索引；各训练扰动均匀无放回最多64细胞、input NTC最多4096。按冻结物理身份、原始行号和官方映射提取，不根据质量或效应筛选，所有训练任务均有池。保存精确 uint16 counts、身份表和来源哈希，超过表示范围或非整数立即失败。这个预算采样不同于全细胞最大似然，报告实际池细胞数；不将更多抽样次数视为独立重复。评分 NTC、H1 扰动细胞及不允许靶点无法进入池。

每步均匀抽训练背景；128细胞各以0.1概率选 NTC，否则均匀选该背景训练靶点，再均匀选池内细胞。不把随机 NTC–扰动配对当纵向真值，模型只条件于群体 NTC 统计。背景 encoder 使用训练共同实测基因上的 NTC mean-logCPM，MLP hidden128→64；target embedding32。H1 只把同一共同基因的 input NTC 均值送入训练好的函数，不用 H1 后验。

## 条件 VAE 与联合目标

训练 posterior 输入单细胞 logCPM−该背景 NTC 均值及96维条件，MLP hidden128→32，输出16维均值/log方差；log方差截在[−8,6]。标准正态 prior；posterior仅训练使用。

均值 offset 为来源细胞数加权的 `log((meanCPMpert+0.1)/(meanCPMntc+0.1))`，按每来源实测掩码逐基因聚合；无来源坐标为零。条件 mean head 与 latent residual head 各 hidden128→rank64；通过各自共享 readout 矩阵输出。mean head末层初始零，readout矩阵初始 N(0,.01²)。latent residual 使用 `head(z,condition)-head(0,condition)`。NTC 靶点强制零均值响应修正，但仍学习/生成异质性。

计数均值为细胞实际文库×softmax(`log(NTC_meanCPM+.1)+shared_LFC+mean_adjustment+latent_residual`)；总 log shift截在±8。NB方差 `μ+φμ²`，φ从各背景 NTC 矩估计起步，乘学习的全局基因 offset；log offset截[−5,3]，φ截[.0001,20]。缺测基因没有似然或辅助损失。H1 中没有训练测量的基因在生成时强制 learned mean/residual/logφ adjustment 为零，只保留 H1 NTC 统计；避免未经训练的自由输出参数影响结果。

损失为 mean-gene NB NLL + 0.1×响应辅助项 + β×每细胞16维KL之和的均值。β在1024步线性升至.01。辅助目标是全训练群体算术 LFC；scaled MSE量纲.5，幅度权重 `1+4*tanh(abs(truth/.5))`，方向 softplus权重.02、活跃门槛|LFC|≥.05。辅助只约束 latent=0 的响应，不声称精确约束非线性采样后的群体均值；实际偏移必须由评分/诊断检验。

固定8192步，batch128；AdamW(lr.001,wd.0001,betas.9/.999,eps1e-8)、clip1、float32、确定性算法。每256步保存网络/optimizer/采样器/CPU与CUDA RNG/步数/配置哈希及背景步数。选择固定最后 checkpoint，不用 H1 或第一条候选成绩选超参。固定预算就是本轮完整训练条件，训练曲线若未收敛只记录限制，不在看见评价后延长。

## 生成、风险与闭环

每靶 seed+冻结索引，重采样 H1 input NTC 文库，采样16维标准正态 z，解码条件均值与φ，Gamma-Poisson生成400细胞。完整18,533输出轴、H1真实18,074评分轴；缺输入轴显式零完成。全局未见靶点没有合法ID，回退零Δ NTC重采样并标明能力边界；本地context-seen可用，不能声称已解决官方28个无来源靶点。

这个机制能改变均值、异质性和基因相关潜变量，区别于上一轮解析NB矩模型。似然族错配、潜变量抑制响应、来源LFC失配、NTC矩与latent重复解释方差、池样本不足和H1背景外推均可能失败。完整方案有signal后再匹配均值与生成器做归因。

正式评分接口与anchors固定；NTC靶点调用同一VAE prior生成器做独立辅助伪DE诊断，不借用旧生成器结果。完整训练+全轴预测+两面板评分+模型Artifact才算执行完成；REPORT给出配对效应/波动、全部六项与有效数、实际池/全群体监督量和资源。最后关闭 E-CELL-VAE-PACKAGE 并统一比较两条机制，停止未达标配置的局部网格。

入口 `./reproduce.sh --run-id cell-cvae-s01`；同条件完整恢复加 `--resume`。所有实际配置、代码及独立uv锁文件先提交；execute和bind门禁贯穿标准入口。
