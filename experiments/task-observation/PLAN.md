# 五背景观测与计数生成校准

Comparison: `C-TASK-DATA-GENERATION`。本路线执行无预测器拟合的真实数据诊断；核心是能否形成可靠的观测统计，并使 NTC 条件计数生成满足零响应恒等、文库守恒和可实现效应重建。

KnowGraph attention: `WL-KD-EFFICIENCY`, `WL-DEPTH`, `ST-NORMALIZE`, `ST-WILCOXON`, `MY-EFFECT-EMITTER`。邻近响应不删除；低深度不补高；弱响应不筛除；剂量不确定性不是生物重复置信区间。

## 数据与信息边界

五背景全部原始计数重新读取、全文件校验哈希，复用 `init-linear-s01` 的冻结物理身份和现有官方基因映射。统计所有合格扰动及输入 NTC；评分 NTC 不进入基线统计。保存 native / 已测官方轴 / metadata UMI 的区别、逐 guide×batch 的靶点下降及近似不确定性。训练消费这些统计之前必须按外层 context、study、global-target 划分限制读取。

任务匹配计数视图采用每背景固定的二项 UMI 抽稀率 `min(1, 20000 / median(native input NTC depth))`，仅从输入 NTC 估计。它不是每细胞固定 20k，也不是缺测全轴补全。低深度来源维持原深度。所有 raw 统计保留；没有实施基于响应或效率的额外过滤。

每个 S2/S3/S4 背景评价面板按靶点身份哈希预取 16 个结构合格靶点、每靶最多 400 个唯一真实细胞，用于后续 evaluator 验收；这是有限面板开发诊断，不称完整竞赛面板成绩。每背景保留输入/评分 NTC bank 各最多 2,048 个。大库训练统计仍使用全部有效真实细胞。

## 生成和评价

以 NTC 经验细胞为模板；非零 bulk log-CP50K 响应先投影至组成单纯形，再用最多 64 次行列缩放匹配文库与组总量，随机守恒取整。零响应严格返回同一随机流抽到的真实 NTC。独立逐元素随机取整是非零效应的匹配诊断对照。

每背景 5 个 null 重抽样，400 细胞/群；评分 NTC bank 分为互斥 control 和 real-null 池。比较 real-null、input-resample、zero-emitter；不得要求真实 null 必须没有显著基因。固定前 4 个哈希靶点提供可实现的 oracle bulk 效应，比较生成器实现误差与官方 Wilcoxon DE 集合。oracle 输入仅属于生成器验收，不能称预测方法。所有 DE 使用冻结 cell-eval2 vcc2026 参数及 gpudge，不运行替代检验。

主要验收是 20 个预定检查的通过率：五背景各检查零响应精确恒等、相对输入重采样新增伪 DE 为零、非零响应逐行总量严格守恒、已知 bulk 生成 RMS ≤ 0.01。该阈值是计数接口的工程精度要求，不是生物响应泛化门槛。任何失败完整保留，修复另登记条件；DE 恢复及 real-null 抽样范围单报，即使工程检查通过也不能宣布分布恢复成功。

## 完整预算与结束条件

单一无预测器拟合 run，seed 930；五背景全扫描、全部统计产物、5×5 null 诊断、5×4 已知效应的两种取整、官方 DE 明细和 W&B artifact 完整完成。预计主要成本为约 120 GB 原始 I/O 和几十次 GPU DE；不以部分背景或预检代替结束。恢复校验已完成阶段的哈希，未完成阶段从原始输入重建，不延长统计预算追逐通过。

该 run 不产生 normalized Overall；退化 null 不构造虚假的六指标。后续 `C-LOCAL-CAPABILITY-EVALUATOR` 对冻结的非零真实面板另建官方 baseline/bundle/anchors，并验证六指标和能力诊断；任何生成/评分条件变化另立协议。
