# exp007：log2FC 监督与 H1 留一验证

## 研究问题与范围

检验与 exp006 相同的 p / g / p–g 交叉 / NTC context 特征，改为预测
逐细胞 CPM 算术均值的 log2 fold change，是否改善 H1 上的扰动响应预测。
来源 exp006 commit `2a2a740`、run `20260927-exp006-dispersed-s17`；仅继承方法与源码，
不复用模型、训练状态或 run 缓存。exp006 的活动代码、输入、环境和历史产物保持只读。

初始训练阶段用户指定：只留出 H1。训练背景固定 K562 GWPS、RPE1、HepG2、Jurkat；
只拟合一个四背景模型，不进行五折、五背景全量重训或 A/B/C 提交。
H1 标签仅用于评估与选择检查点；H1 NTC 可用于输入特征与生成基线。
H1 已用于项目开发，本次分数也是开发验证，不能宣称独立无偏测试。

## 数据与监督

使用各背景全部合格扰动（原始细胞数至少 50）及其全部已纳入细胞。
H1 历史三个 split 合并，重复 NTC 按 batch/barcode 对齐并核对表达后只计一次。
采用每背景可靠原生测量基因轴；无歧义身份映射，未测基因不补零监督。
原始 counts 要求非负整数。所有原始来源和 HGNC / STRING / Reactome / CollecTRI
核对登记 SHA-256。官方 gene_names 只用作先验节点固定轴；不读取 A/B/C 细胞或扰动标签。

对每个细胞 i，先在该背景原生轴归一化：

`CPM(i,g) = 1e6 * counts(i,g) / sum_g counts(i,g)`

分别求扰动组与完整 NTC 池的算术均值，标签为

`log2FC(c,p,g) = log2((mean_CPM(c,p,g) + 1e-9) / (mean_CPM(c,NTC,g) + 1e-9))`。

这对应固定 cell-eval2 版本的 DE 效应定义，不是 group-sum 的 log1p(CP50K) 差，
也不是逐细胞 log 表达的均值差。不截断有限 LFC，不使用 H1 响应决定训练基因权重或筛选。
完整可测轴均匀采样；官方 DE 的 CPM>5 与显著性筛选仅在评估中使用。
零表达标签可能很大，这是本次平方误差监督的已知限制，结果中需关注低表达效应。

## 模型、采样与 loss

沿用共享 XGBoost 标量回归器，输入：p/g 的 STRING embedding，NTC 表达、检测率、二阶统计，
STRING 功能/物理关系、Reactome 通路交集、CollecTRI 有向激活/抑制，关系×表达交叉项，
NTC 全局表达投影与 regulon 表达代理。只使用基线与固定知识先验，不使用 H1 扰动标签作特征。
NTC 表达输入仍为原先的 log1p(CP50K) 统计，以保持特征方案稳定。

监督为加权平方误差 `reg:squarederror`，日志 `rmse` 的单位为 log2FC。
四个背景等权，每背景 target 等权，再按 bag/readout 等权；总样本权重为 1e6。
完整 NTC 输入加 8 个各 400 细胞的分层 bag，标签始终锚定完整 NTC 池。
每个 target/bag 固定种子均匀抽取最多 512 个可测 readout，所有合格 target 都参与训练。
模型 CPU hist，depth 6，eta .05，lambda 10；其余超参数见配置。

## 生成与评估

完整 NTC 的 mean CPM 与模型输出恢复期望组成：

`desired_CPM = max(0, (baseline_CPM + 1e-9) * 2**predicted_log2FC - 1e-9)`。

独立基因预测未必满足 CPM 总和 1e6，所以先投影至合法组成。
每 target 分层抽取 400 个 NTC 模板，在每细胞概率空间进行行列校准：
每行总量 1，每列均值为期望组成，然后恢复模板各自 library size 并随机舍入为整数 counts。
最多 128 次校准，行和偏差 <1e-6；不收敛显式失败。
模板完全缺失的基因可按目标组成引入表达；模板异质性不等价于新扰动状态。
该生成改动是目标口径所需，不能把最终差异单独归因为 log2FC 标签。
保存原始预测、投影/生成后的 LFC 误差与每个检查点的预测 counts。
不额外拟合细胞方差、零率或新状态分布。

H1 reference 为完整 NTC 池和每 target 固定分层抽最多 400 个真实细胞（不补采样），
沿用 exp006 相同种子、目标集合与原生基因轴。固定检查点 1、16、64、128、256、384、512。
一次拟合训练至 512 轮，七次 H1 模型评分；另有零响应、仅训练背景同靶点平均 LFC 两个对照。
两种对照用同一生成器。未有训练支持的 target/readout 转移响应为零，记录支持分层。

官方 cell-eval2 commit `5e64833518a6603a0301cbe28185d49c30f4a986`、vcc2026 preset，
CUDA/gpudge 后端。六项 raw/scaled/Overall、五次拆半 replicate anchors（seed 0）均由官方包计算。
官方 mean-response 评分基线使用 `generic_response_profile` 与 dispersed emission，
`exclude_target_gene=True`、官方 target_gene_map、baseline seed 0。
协议 `cell-eval2-dispersed-exclude-target-v1`；所有模型与对照复用同一 H1 reference / anchors。
流式基线与官方公开构造逐元素验证，不改写指标或删减面板来绕过资源限制。
本地原生面板分数不等同 A/B/C 榜单。若比较 exp006，仅使用相同 H1 面板、种子和新评分协议，
不能将 exp007 单 H1 最优分数与 exp006 五折均分直接比较。

按 H1 Overall 最高选定一个检查点，平分取较早轮数。报告各检查点、六项分数、
两项对照、训练中已见/未见靶点，以及 CPM>5 的预测、投影和生成误差。

## 运行、恢复与交付

`cd experiments/exp007 && ./reproduce.sh --config configs/default.yaml`
或 `./reproduce.sh --resume RUN_ID`。
Experiment 独立 uv 环境，固定 virtual-cell 基础 Python；使用 micromamba 调用 uv，
锁定依赖、禁止 Python 下载。环境共享锁覆盖整个 run；运行期间不修改环境或执行代码。
输入校验、预处理、训练、评估均处于同一 run，输出在 `outputs/RUN_ID/`。
正式启动前提交代码、配置、PLAN 与 uv.lock，记录 Git commit、锁、运行时和数据身份。
恢复完整 Booster 内存状态、迭代、RNG 与输入身份，先补做未完成检查点评分再继续训练。

W&B entity `yjcyxky`、project `virtual-cell-challenge`、group `exp007`，id 与本地 run 一致。
记录配置、版本、loss、评分、状态和资源；断网本地保留并注明待同步。
关键模型、预测 LFC 和评分上传版本化 Artifacts，不上传包含原始 NTC 的 counts 文件。
预测 counts 本地保留，最近 checkpoint 同时保存完整训练恢复状态。

结束条件：512 轮有效训练、七个 H1 检查点及两个对照评分、选模证据、模型/预测/指标保存、
W&B 归档或明确待同步状态、REPORT 结论和代码清理。启动或 smoke test 不算完成。
资源不足等待并记录，不能干预 exp006；无法继续须记录真实失败或阻塞。

## 2026-09-27 追加：第 512 轮官方提交

用户指定将现有第 512 轮 checkpoint 生成官方 `.vcc` 并提交获取评分。
此为同一 run 的后续评估，复用原 run_id 与 W&B run，不重训、不改变已有训练和 H1 评估结果。
原流程完成后，为该 Experiment 锁定新增 `vcc-cli==0.2.1`；记录训练与导出各自的 commit/锁，
并验证训练时的数值运行时版本和特征、预测、生成实现未改变。

读取已登记、校验哈希的 A/B/C NTC，只构建基线输入与模板；不读取官方扰动标签。
沿用 log2FC 预测、组成校准和随机舍入，每背景 300 靶点各 400 细胞，官方顺序 18,533 基因。
预处理写入当前 run 的 `cache/official-export-data/`；预测、校验、`.vcc`、提交回执写入
`predictions/leaderboard-round-0512-seed-101/`。保留 checkpoint 与输入/输出哈希，
复用已有全文件 audit、官方 CLI prep 和同 entry 断点上传流程，不绕过任何官方校验。

入口：`./reproduce.sh --resume 20260927-exp007-h1-log2fc-s17 --export-checkpoint 512 --submit`。
不传 `--submit` 则只导出。官方成绩与本地 H1 口径分开保存，反馈用于开发；
提交预测及回执归档为版本化 W&B Artifact，REPORT 记录分项、Overall 和限制。
