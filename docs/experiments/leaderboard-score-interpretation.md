# 2026 leaderboard 分数解释与比较证据

调查日期：2026-09-26。公开榜统计快照时间：`2026-09-26T12:45:43.409Z`。本文保留当时结果；实时榜单后续会变化。

## 公开榜并非全部为正

直接读取[官方 leaderboard](https://virtualcellchallenge.org/leaderboard)使用的[公开 JSON 接口](https://virtualcellchallenge.org/api/leaderboard?get_final=false)，`num_entries` 与返回数组长度均为 **1,174**：727 队为正，447 队为负，没有恰好为零的队伍；分数中位数为 **0.0697639613**。负分从第 728 名开始。以下数字是对完整响应的统计，并非对榜单首页的观察。

| 排名 | 队伍 / 模型 | Overall |
| --- | --- | ---: |
| 1 | Illumina AI / PerturbationAI | 0.3579208307 |
| 2 | cqawesome / m0941a | 0.3183475504 |
| 3 | Testh / test | 0.3014277418 |
| 831 | OpenProphetDB OpenProphetDB / ContextModuleCVAE-c4-20260925-lodo-response-s17 | -0.0359001763 |
| 1,174 | Oleh RCL / cpa-v1-string-context-transfer | -1.0321196552 |

我们的公开 entry 为 `wDKL5Vvgw5wf2LuqfdUG`，提交时间为 `2026-09-25T17:40:32.869412Z`。全部 1,174 个条目的比较身份一致：`partition=val`、`panel_id=vcc2026-val-1`、`anchor_version=vcc2026-valA-r4+vcc2026-valB-r4+vcc2026-valC-r4`。因此这次真实官方提交与榜首可以直接比较；不能因为另有本地验证结果就把它误写为“从未提交官方榜”。[官方公开接口](https://virtualcellchallenge.org/api/leaderboard?get_final=false)

榜单显示每队**最近一次提交**，不是历史最佳提交；六项列同时展示 scaled score 和 raw metric。动态页面正文可在其[公开页面脚本](https://virtualcellchallenge.org/_next/static/chunks/app/(public)/leaderboard/page-6aadeba8af47e905.js)核对。此次普通网页抓取只得到 `Loading...`，调查通过页面公开脚本定位上述公开 API，没有使用账户凭据。

## 零分表示什么

对 context \(c\) 的指标 \(m\)，原始聚合指标为 \(u_{c,m}\)，同背景平均扰动响应基线为 \(b_{c,m}\)，真实实验拆半重复参照为 \(r_{c,m}\)。先计算

\[
z_{c,m}=\frac{u_{c,m}-b_{c,m}}{r_{c,m}-b_{c,m}},\qquad
\mathrm{Overall}=\frac{1}{18}\sum_{c\in\{A,B,C\}}\sum_{m=1}^{6}C_m(z_{c,m}).
\]

这里每个 context、每个指标等权，不按扰动数或有效 DE 任务数加权。0 对应平均扰动响应基线，1 对应重复实验参照；Overall 不是正确率，也不是固定在 0–1 的百分比。[官方 Evaluation 页面](https://virtualcellchallenge.org/evaluation#scoring)、[该页公开正文脚本](https://virtualcellchallenge.org/_next/static/chunks/app/(public)/evaluation/page-83d97ad9fca21302.js)、[官方 `score.py::_replicate_entries`](https://github.com/ArcInstitute/cell-eval2/blob/5e64833518a6603a0301cbe28185d49c30f4a986/src/cell_eval2/score.py#L377)

逐项处理 \(C_m\) 为：

| 指标 | 处理 |
| --- | --- |
| `expr_mse_unbiased_capped_norm` / MSE | 截断到 `[0, 1]` |
| `de_wilcoxon_lfc_nmae` / NMAE | 下限 `-6`，无上限截断 |
| PDS、direction fidelity、direction reach、significant-set Jaccard | 不作上下限截断；可为负 |

这由固定源码的[评分策略 `ERROR_LINEAR` 与 `BOUNDED_UNFLOORED`](https://github.com/ArcInstitute/cell-eval2/blob/5e64833518a6603a0301cbe28185d49c30f4a986/src/cell_eval2/scoring.py#L187)和[MSE 注册项](https://github.com/ArcInstitute/cell-eval2/blob/5e64833518a6603a0301cbe28185d49c30f4a986/src/cell_eval2/catalog.py#L333)确认。不能给所有指标套一个统一的下限，也不能将 MSE 的 scaled `0` 解释为预测误差为零。

**零基线不是直接复制未扰动 NTC。** 它的平均扰动响应来自被评估 context 的真实扰动数据，按合格构建体的均值等权汇总。参赛者没有这些留出标签，所以这是用于衡量目标特异性能力的 oracle comparator，不是参赛者无需学习就能精确实现的输出。官方 `baseline.py` 开头也明确说明这种 transductive 性质。[官方指标定义 §0](https://github.com/ArcInstitute/cell-eval2/blob/5e64833518a6603a0301cbe28185d49c30f4a986/docs/vcc2026_metrics/vcc2026-metrics.md#0-overview)、[`baseline.py` 的基线语义](https://github.com/ArcInstitute/cell-eval2/blob/5e64833518a6603a0301cbe28185d49c30f4a986/src/cell_eval2/baseline.py#L1)

因此负 Overall 表示六项综合低于该参照，不能单独推出“所有指标都失败”“比 NTC 原样输出还差”“模型完全没学到信息”或“评分实现必然出错”。相反，正 Overall 也不保证六项均优于基线。这些是根据上述聚合规则得出的解释。

## 与榜首和 NTC 提交逐项比较

下表全部取自同一[官方榜单接口](https://virtualcellchallenge.org/api/leaderboard?get_final=false)，均为 scaled score；列顺序与接口的 `score_pds/mse/nmae/fid/reach/jac` 一致。

| 模型 | PDS | MSE | NMAE | Fidelity | Reach | Jaccard | Overall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Illumina AI | 0.865831 | 0.479760 | 0.267683 | 0.125842 | 0.387785 | 0.020623 | 0.357921 |
| cqawesome | 0.826810 | 0.401815 | 0.287323 | 0.084790 | 0.291146 | 0.018201 | 0.318348 |
| Testh | 0.800620 | 0.408202 | 0.248949 | 0.052692 | 0.282170 | 0.015934 | 0.301428 |
| 我们的 CVAE cycle 4 | 0.002145 | 0 | -0.156851 | -0.024458 | -0.033825 | -0.002412 | -0.035900 |
| Amgen Virtual Cell Club：control-resampling-baseline | -0.011876 | 0 | -0.003279 | -1.720475 | -0.006410 | -0.082575 | -0.304102 |

Amgen entry 为 `2ohYMNRX2rp39VJQHfeY`，其作者描述为按 guide 平衡抽取验证 NTC 的无效应基线；这是参赛者自述，本文没有复现其生成代码。同榜 Formosa Bear（`qaAOFXglWrRJqJeHC6jR`）自述直接复制每背景随机抽取的 400 个 NTC counts，Overall 为 `-0.3032965079`。这些实例清楚展示了“NTC 原样输出”与评分零基线的差别；它们不能替代我们自己按完全相同生成协议建立的配对基线。[官方公开接口](https://virtualcellchallenge.org/api/leaderboard?get_final=false)

需要同时看 raw metric，以免截断隐藏错误规模：

| 模型 | raw PDS ↑ | raw expression error ↓ | raw LFC-NMAE ↓ | raw fidelity ↑ | raw reach ↑ | raw Jaccard ↑ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Illumina AI | 0.889836 | 0.532355 | 0.839656 | 0.549430 | 0.421734 | 0.038192 |
| 我们的 CVAE cycle 4 | 0.501048 | 11.966576 | 1.100267 | 0.505945 | 0.049204 | 0.029537 |
| Amgen NTC 重采样 | 0.494608 | 1.031266 | 1.002802 | 0.001386 | 0.073249 | 0 |

PDS 衡量扰动身份的可区分性：随机配对或相同响应预测的无信息水平为约 0.5；NMAE 衡量真实显著变化基因的效应幅度，预测 LFC 处处为零时理论值为 1，这不保证有限细胞 NTC 重采样的结果恰好为 1。raw MSE 是按真实扰动相对 NTC 的变化幅度归一化后的误差，不是普通训练 MSE。[官方指标简表 §1、§2、§7](https://github.com/ArcInstitute/cell-eval2/blob/5e64833518a6603a0301cbe28185d49c30f4a986/docs/vcc2026_metrics/vcc2026-metrics-brief.md)

据此可作的有限判断：我们的扰动身份区分接近无信息水平，表达误差远大于 NTC 对照；这两点是具体弱项。但 NTC 的 fidelity 几乎为零，因而受到很大的负分，我们的 fidelity 接近平均响应基线，两者 Overall 差距不能解释成全面生物学准确度的等比例提升。

榜首三队的 `description` 均为 `null`，公开 entry 没有代码链接；不能凭模型名称断言其架构、训练数据或获胜机制。另有 `mw`（entry `XyO7jFujRSgUcj5F7MDn`，第 610 名）公开自述采用 K562 LFC 乘性迁移到重采样 NTC，得分 `+0.0590390216`。这是可用于设计本地对照的线索，不是已经复现的方案；其 raw MSE 仍为 `6.060004`，也再次说明正 Overall 不保证表达误差良好。[官方公开接口](https://virtualcellchallenge.org/api/leaderboard?get_final=false)

## 我们三次官方提交：改进发生在哪一项

从归档 `official-summary.json` 重新计算六项均值，三次均与官方 `score_avg` 在 `1e-12` 内一致，且 panel/anchors 相同。负分是实际发布结果，不是把训练 loss 当作 Overall 或均值计算错误。

| 提交 | PDS | MSE | NMAE | Fidelity | Reach | Jaccard | Overall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| exp002 SharedResponseShrink / 20260922-d | 0.232316 | 0 | 0.029302 | -1.439033 | 0.052035 | -0.061650 | -0.197838 |
| exp001 ContextRelationXGB / 20260923-a | 0.007589 | 0 | 0.001819 | -1.722954 | -0.022812 | -0.082562 | -0.303153 |
| exp003 ContextModuleCVAE / cycle 4 | 0.002145 | 0 | -0.156851 | -0.024458 | -0.033825 | -0.002412 | -0.035900 |

来源：[exp002 官方结果](../../experiments/exp002-response-transfer-validation/outputs/20260922-d/official-summary.json)、[exp001 官方结果](../../experiments/exp001-context-pair-xgb/outputs/20260923-a/official-summary.json)、[exp003 官方结果](../../experiments/exp003-context-module-cvae/outputs/20260925-lodo-response-s17/predictions/leaderboard-holdout-H1-cycle-0004-seed-101/official-summary.json)。

CVAE 相对 SharedResponse 的 Overall 增加 `0.161938`，其中 Fidelity 的单项改善对总分贡献 `1.414575/6 = 0.235762`，同时 PDS、NMAE 和 Reach 下降。故目前的进展主要是摆脱 Fidelity 的大额负分，不能说靶点特异预测全面提高。Fidelity 含覆盖因素，旧分数低本身也不能区分“没有预测出足够有置信度的变化”与“方向预测错误”；需要原始 DE 表进一步拆解。[归档逐项比较](../../experiments/exp003-context-module-cvae/outputs/20260925-lodo-response-s17/predictions/leaderboard-holdout-H1-cycle-0004-seed-101/official-comparison.json)

这些是完整流程的比较。旧模型使用五个训练背景，新提交使用排除 H1 的四背景模型；架构、目标、数据覆盖及生成方法均有变化，不能把差值单独归因于 CVAE 架构。

## 当前模型的本地证据

以下读取 `20260925-lodo-response-s17` 的已完成 H1 折，不借用历史 v1 模型统计。其余折尚未全部完成；没有把当前 run 描述为完整五折交付。

### 跨背景泛化比继续优化训练误差更紧迫

H1 留出验证第 4 周期最佳分为 `-0.067635835`，第 20 周期为 `-0.105008598`。该折在完整覆盖与稳定条件满足后正常早停。第 4→20 周期，四个训练背景的 response loss 均下降，response skill 相对零响应均提高；但 H1 验证变差。此证据支持泛化/目标适配问题，不能由此认定多训练一定会提高官方分数，也不能证明所有其他背景都会如此。[该折完成记录](../../experiments/exp003-context-module-cvae/outputs/20260925-lodo-response-s17/predictions/holdout-H1/complete.json)、[周期 4](../../experiments/exp003-context-module-cvae/outputs/20260925-lodo-response-s17/cache/holdout-H1/cycle-0004.json)、[周期 20](../../experiments/exp003-context-module-cvae/outputs/20260925-lodo-response-s17/cache/holdout-H1/cycle-0020.json)

最佳模型 297 个 H1 靶点的辅助研究指标取逐任务均值：

| 指标 | 值 |
| --- | ---: |
| 响应 Pearson 相关 | 0.044258 |
| 有效应 readout 的方向一致率（观测绝对变化 >0.05） | 0.532893 |
| 预测响应 RMS / 观测响应 RMS | 0.043879 / 0.044313 |
| 响应 MSE / 有限样本 NTC MSE | 0.004223 / 0.003060 |
| 预测/观测总计数均值比 | 1.179390 |
| 总计数 CV：预测 / 观测 | 0.105064 / 0.348275 |

来源：[逐任务指标](../../experiments/exp003-context-module-cvae/outputs/20260925-lodo-response-s17/predictions/holdout-H1/metrics.parquet)，列定义见 [evaluation.py](../../experiments/exp003-context-module-cvae/src/evaluation.py)。这些是本地共同基因轴等既定研究口径，不是官方 A/B/C 的 raw/scaled 分数，也不与官方方向指标混同。

新模型整体响应幅度已接近观测，但方向和相关性弱；不能继续用历史模型“幅度偏大”的诊断直接解释它。总计数分布明显偏窄是另一个校准信号，但仅凭 library-size CV 不能证明所有基因都欠离散，更不能宣称它已被证实为负分的唯一原因。

### 监督覆盖不足且主目标与评分仍有差异

此次官方提交有 **7,632/18,533（41.18%）readout 没有训练测量监督**，**42/300（14%）靶点不在合格训练任务中**。生成时保留原模型输出，没有对这些 readout 自动回退为 NTC。这是该 H1 留出 checkpoint 的具体限制，不是全项目所有数据都缺少这些基因。[导出身份](../../experiments/exp003-context-module-cvae/outputs/20260925-lodo-response-s17/predictions/leaderboard-holdout-H1-cycle-0004-seed-101/export-identity.json)、[比较中的限制记录](../../experiments/exp003-context-module-cvae/outputs/20260925-lodo-response-s17/predictions/leaderboard-holdout-H1-cycle-0004-seed-101/official-comparison.json)

主 delta loss 只在 **6,114** 个共同可测基因上计算 `log1p(CP10K(weighted mean counts))`，比较预测扰动与预测 NTC；官方表达比较使用 group-sum 的 `log1p(CP50K)`、DE 使用细胞归一化表达及 Wilcoxon，并以真实 NTC 为参照。NTC 辅助锚已经存在，但同样只作用于共同轴；原生轴的 NB 与总计数约束承担不同职责。因而“主训练误差下降”不等于“全部官方 readout 上的真实 NTC 相对效应准确”。这是一项待通过对照验证的目标适配问题，不是本次确认的实现 bug。[目标实现](../../experiments/exp003-context-module-cvae/src/objectives.py)、[固定训练统计的轴](../../experiments/exp003-context-module-cvae/outputs/20260925-lodo-response-s17/cache/holdout-H1/task-statistics/complete.json)、[官方归一化及参照定义](https://github.com/ArcInstitute/cell-eval2/blob/5e64833518a6603a0301cbe28185d49c30f4a986/docs/metrics.md#12-normalization)

## 建议的改进顺序与验证方式

以下是基于上述证据的研究建议，不是已经证明有效的改动。本次未启动新 run。

1. **先把响应预测与生成误差拆开。** 在同一留出背景、相同基因轴与细胞数上，配对比较 NTC、现有 SharedResponse、当前 CVAE；分别检查均值/效应与分布生成。SharedResponse 的官方 PDS 明显更好，是已有的有用对照。可以检验“共享靶点响应 + 背景条件残差”的方案，只有残差在留出背景改善才增加容量；不要先假设更复杂的联合生成就会更准确。
2. **优先处理监督覆盖。** 对已训练/未训练 readout、已见/未见靶点分别报 raw 指标与效应误差贡献；对新训练方案评估逐背景可测轴的 masked response 监督，补充测量覆盖及跨背景支持，不将缺测补成零。若最终提交采用包含 H1 的模型或多折集成，应在新的明确配置中验证；当前四背景单折结果不能代表全部数据训练的上限。不能未经验证就把未监督基因回退为 NTC 后宣称改进。
3. **校准 NTC 与计数分布，再判断 DE 改进。** 用独立真实 NTC 检查生成 NTC 的均值、零率、逐基因方差和总计数分布；跑相同 DE 流程测量伪变化。针对当前总计数均值偏高、CV 偏窄的证据，分别固定响应均值或分布参数做对照，验证哪一项影响 NMAE/Fidelity/Jaccard。主 delta 两个预测分支可能同时带有残余基线偏移，需检验辅助 NTC 锚是否足够；目前没有完成这一因果检验。
4. **让可优化目标覆盖官方真正关心的效应。** 在新 run 中对照不同归一化轴/尺度、真实 NTC 相对误差及可靠效应权重，同时分开 on-target 与下游 readout。目标是提高靶点间可区分性、效应方向与幅度；不能靠增加所有扰动的共同变化去换 Fidelity，或仅优化 loss 数字。低表达 LFC、生成方差与 DE 覆盖需联查，避免只调整幅度。
5. **用跨背景结果决定复杂度。** 当前 H1 折在训练拟合继续改善时留出变差，优先评估正则化、背景迁移与简单对照。现有历史消融没有证明真实先验的总体收益，后续先验证可迁移响应，再用控制容量的消融检验模块、状态及先验增益；不能据一次负结果否定生物先验。继续训练、扩大模型或增加知识模块都需要新的证据支撑。[历史消融与限制](../../experiments/exp003-context-module-cvae/REPORT.md)

每项涉及数据、模型或训练条件变化时另建 run，保持原结果及原评分口径。沿用正式六项评分、raw 指标与固定本地参照，先在留出背景验证，再用官方 validation 检查迁移；已用于选择的 validation 属于开发反馈。正分不是唯一验收条件，必须确认是否同时改善响应区分、幅度及分布，而不是仅减少一项惩罚。

## 年份、阶段与本地评估边界

当前是 **2026 验证榜 A/B/C**；官方计划于 2026-10-22 发布另外三个背景的 final test，最终排名由 final test 决定。2025 是同一 H1 背景的留出扰动任务，2026 是未见背景迁移，不能混合解释两年的成绩。[Arc 2026 官方任务与时间表](https://arcinstitute.org/news/virtual-cell-challenge-2026)

官方说明验证和最终测试背景、扰动面板不同，PDS 又是在各自面板内部排序，因此阶段间绝对分数不可直接比较。同理，使用官方 `cell-eval2` 在本地 H1/K562/RPE1/HepG2/Jurkat 上重建参照，也只获得本地验证尺度，不会自动获得官方榜单的数值可比性。应同时记录 raw metric、baseline、replicate、panel、样本量、测序深度、基因轴与评分版本。[官方 Evaluation](https://virtualcellchallenge.org/evaluation#scoring)

本调查只读取公开网页/API、已安装的固定官方源码和仓库文档，没有修改实验代码、环境、历史产物或启动训练。
