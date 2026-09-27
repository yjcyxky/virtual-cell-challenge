# exp007 本地与官方评分口径核查

调查日期：2026-09-27。只读取已有源码、配置、指标及官方文档；未修改训练代码、环境、历史 run 或重新提交。

## 结论

**跨实验补充核查：exp004 → exp007 的六项 raw 指标公式没有改变，但本地评分协议确实改变了，包括 mean-response baseline 的生成方式、baseline 是否排除自身靶基因、真实参考细胞以及由此重算的 baseline/replicate anchors。不能将“同一官方包、同一指标公式”表述为“本地评分方式完全未变”。** 下文先前关于核心实现未变的判断仅指 exp007 自身训练到导出，不适用于 exp004 到 exp007。

**官方部署证据边界（再次核查公开来源）：当前仅证实 exp007 使用固定官方包的默认 baseline 构造，未证实排行榜实际 r4 bundles 采用相同的构造参数、NTC 选择或 profile 参与集合。不得把此前的“对齐官方 baseline”理解为已经验证与 leaderboard 端到端等价。**

exp007 第 512 轮本地 H1 Overall 为 `0.020419414867561134`，官方 A/B/C Overall 为 `-0.12779834383804198`。这两个数使用不同 reference、panel 和 baseline/replicate anchors，不能把差值解释为同一数据上模型准确度下降同样幅度。

最明显证据是 Fidelity：本地 raw 为 `0.29169028079778847`，官方回执 raw 为 `0.30974049436509715`，后者反而较高；scaled 却从 `0.17928123492966228` 变成 `-0.6867549109343541`。这证明不能将 scaled 跌幅描述成“原始 Fidelity 崩溃”。不同数据的 raw 也不是同条件配对实验，故不能据 raw 较高断言官方泛化更好。

来源：[本地 raw](../experiments/exp007/outputs/20260927-exp007-h1-log2fc-s17/predictions/HepG2+Jurkat+K562+RPE1/H1/model-0512/aggregate.csv)、[本地 scaled](../experiments/exp007/outputs/20260927-exp007-h1-log2fc-s17/predictions/HepG2+Jurkat+K562+RPE1/H1/model-0512/scores.csv)、[官方完整回执](../experiments/exp007/outputs/20260927-exp007-h1-log2fc-s17/predictions/leaderboard-round-0512-seed-101/official-status.json)。

## 数值证据

| 指标 | 本地 H1 raw | 官方回执 raw | 本地 scaled | 官方 scaled |
|---|---:|---:|---:|---:|
| PDS ↑ | 0.507973883 | 0.499676700 | 0.025199748 | 0.000049858 |
| Expression error ↓ | 1.125906722 | 1.475296742 | 0 | 0 |
| LFC-NMAE ↓ | 1.001980631 | 1.000194261 | -0.081753651 | 0.000890958 |
| Fidelity ↑ | 0.291690281 | 0.309740494 | 0.179281235 | -0.686754911 |
| Reach ↑ | 0.037266110 | 0.067395057 | -0.056768198 | -0.013233789 |
| Jaccard ↑ | 0.030835470 | 0.005520943 | 0.056557355 | -0.067742179 |

表中所有数字直接来自上一节的三份文件。Expression error 的 scaled 均为零掩盖了 raw 的差别，不能解释为误差为零。

按两组 scaled 分数相减，再除以六，Overall 的净差为 `-0.1482177587`。Fidelity 项贡献 `-0.1443393576`，约占净差 `97.38%`；Jaccard 为 `-0.0207165891`、PDS 为 `-0.0041916483`，NMAE 与 Reach 分别抵消 `+0.0137741016`、`+0.0072557348`，MSE 为零。这只是跨面板分数的算术分解，不是“Fidelity 机制导致 97.38% 泛化损失”的因果归因。

本地 H1 Fidelity baseline `b=0.16500263308066102`，取自 [baseline_agg.csv](../experiments/exp007/outputs/20260927-exp007-h1-log2fc-s17/cache/official/H1/bundle/baseline_agg.csv)；replicate `r=0.8716446180426178`，取自 [anchor_agg.parquet](../experiments/exp007/outputs/20260927-exp007-h1-log2fc-s17/cache/official/H1/bundle/anchor_agg.parquet)，并通过保存的 raw/scaled 值复算确认。因而本地：

```text
(0.2916902808 - 0.1650026331) / (0.8716446180 - 0.1650026331)
= 0.1792812349
```

固定版本的官方文档给出的验证背景 Fidelity baseline 范围为 `0.505–0.522`、replicate 为 `0.795–0.832`，和 H1 的 baseline 显著不同。这是官方文档中的范围，不能冒充本条提交 r4 三个 context 的逐项 anchor 回执。官方每背景分别缩放后汇总；只有汇总 raw/scaled 不能解出三个独立背景的精确 anchors。[固定版本官方定义 §3](https://github.com/ArcInstitute/cell-eval2/blob/5e64833518a6603a0301cbe28185d49c30f4a986/docs/vcc2026_metrics/vcc2026-metrics.md#3-dge-direction-fidelity)、[官方 CLI 分数解释](https://vcc-cli-wiki.virtualcellchallenge.org/#5-submit)。

Fidelity 是方向正确的调用数除以预测调用数与真实显著基因数的较大者，因此同时受方向和覆盖影响；单个总分不能区分方向错误与调用覆盖不足。[固定官方源码 direction.py](https://github.com/ArcInstitute/cell-eval2/blob/5e64833518a6603a0301cbe28185d49c30f4a986/src/cell_eval2/metrics/direction.py#L776)。

## 本地实际配置和官方要求

本地从 `EvalConfig.from_preset('vcc2026')` 出发，只覆盖 device、threads、pert_chunk、cache 路径、DE backend；没有发现修改六项指标定义、阈值、归一化尺度或得分方向。本地保存配置为 counts、control_source=real、CPM=1e6、bulk_target_sum=50000、Wilcoxon、BH per_pert、p_adj<0.05、control CPM>5、epsilon=1e-9、PDS panel target exclusion。来源：[evaluation.py:169–175](../experiments/exp007/src/evaluation.py#L169)、[实际 bundle config](../experiments/exp007/outputs/20260927-exp007-h1-log2fc-s17/cache/official/H1/bundle/config.yaml)、[固定官方 preset](https://github.com/ArcInstitute/cell-eval2/blob/5e64833518a6603a0301cbe28185d49c30f4a986/src/cell_eval2/configs/vcc2026.yaml)。

本地 baseline 的 `exclude_target_gene=True`、`emit=dispersed` 与固定官方包默认值一致。这不是任意放宽评分。stream_baseline 使用官方 `_emission_scale` 和 `_emit_scaled_resample`，为了避免大矩阵驻留内存而分块写出。此核查只确认调用与配置；未重跑数值 parity。来源：[evaluation.py:73–110](../experiments/exp007/src/evaluation.py#L73)、[evaluation.py:203–212](../experiments/exp007/src/evaluation.py#L203)、[官方 profile 默认值](https://github.com/ArcInstitute/cell-eval2/blob/5e64833518a6603a0301cbe28185d49c30f4a986/src/cell_eval2/baseline.py#L104)、[官方 emission 默认值](https://github.com/ArcInstitute/cell-eval2/blob/5e64833518a6603a0301cbe28185d49c30f4a986/src/cell_eval2/baseline.py#L347)。

但使用同一个包不等于使用同一评估数据：

| 项目 | 本地 exp007 | 官方提交 |
|---|---|---|
| 背景 | H1 | A/B/C |
| 靶点 | 297 个可用靶点 | 每背景 300 个官方面板靶点 |
| 基因轴 | H1 原生 18,008 基因 | 官方 18,533 基因 |
| 真实细胞取样 | 每靶点最多 400、不放回；完整 H1 NTC pool | 官方固定 reference；提交每靶点恰好 400 |
| anchors | 本地 `exp007-H1`，五次拆半、base seed 0 | `vcc2026-valA-r4+vcc2026-valB-r4+vcc2026-valC-r4` |

本地来源：[reference.json](../experiments/exp007/outputs/20260927-exp007-h1-log2fc-s17/cache/official/H1/reference.json)、[bundle manifest](../experiments/exp007/outputs/20260927-exp007-h1-log2fc-s17/cache/official/H1/bundle/manifest.json)、[checkpoint metrics](../experiments/exp007/outputs/20260927-exp007-h1-log2fc-s17/predictions/HepG2+Jurkat+K562+RPE1/H1/model-0512/metrics.json)。官方 panel/anchor 身份来自回执；官方提交规格来自 [VCC CLI requirements](https://vcc-cli-wiki.virtualcellchallenge.org/#submission-requirements-2026)。

本地 bundle manifest 的 `control_source_effective=pred` 是拆半 replicate 路径的元数据，不能据此认定模型评分把 control_source 偷换为 pred；实际配置和模型 run metadata 应分开读取。官方 replicate 本来就使用各半自己的 control 以避免共享控制诱导相关性。[官方定义 §0](https://github.com/ArcInstitute/cell-eval2/blob/5e64833518a6603a0301cbe28185d49c30f4a986/docs/vcc2026_metrics/vcc2026-metrics.md#0-overview)。

## 可下的判断与未决问题

已证实：本地与官方相对基准显著不同；模型在 H1 上超过本地 Fidelity baseline，但官方综合落后于其 Fidelity baseline。Jaccard 的原始重叠也明显较低，PDS 接近 0.5，说明不能仅以“尺度不同”否认官方实际弱项。

尚未证实：究竟多少差距由未见背景、不同 panel/基因轴、响应均值、细胞分布或导出流程分别造成。本次没有官方 A/B/C 隐藏真值、逐背景 DE 表和精确 r4 anchors，不应把上述因素分配为定量因果贡献，也不应声称更改 baseline 就能修好模型。checkpoint 与导出数值一致性须由独立代码/产物核查回答，不能单凭本报告断言。

## 导出前后代码与数值重放核查

主代理随后完成了以下只读核查。诊断通过已锁定 exp007 环境运行，入口为
`/home/jy001/micromamba/envs/virtual-cell/bin/python src/runtime.py --base-run uv run --locked --no-sync --python /home/jy001/micromamba/envs/virtual-cell/bin/python --no-python-downloads python -`，
临时代码使用标准输入，解包使用自动清理的系统临时目录，没有改写原 run 产物。

1. **模型与配置相同。** 实际 checkpoint 为 512 轮，SHA-256
   `3921541074a4e061d73837c20cd6cf871066fe7b800f0b0077f6b0150b5ec6f3`，
   与训练状态和导出记录一致。导出配置与原 run 的 `configuration` 完全相同，
   `identity.runtime` 内记录的软件版本逐项匹配。
2. **核心实现没有改变。** 将当前文件逐字节对照训练 commit `49e2faf`，
   `features.py`、`generation.py`、`training.py`、`common.py`、`evaluation.py` 全部相同。
   导出 commit `138fb83` 新增了 submission 模块、入口参数、A/B/C NTC 的字段读取分支，
   并增加锁定依赖 `vcc-cli==0.2.1`；没有在提交时新增幅度缩放、重新训练、改写 LFC 或换生成器。
3. **响应重算全部一致。** 使用同一 checkpoint 与缓存特征，重新计算 H1 的 297×18,008
   和 A/B/C 各 300×18,533 个预测响应，与保存的 `predicted-response.npz` / `A/B/C-response.npz`
   逐元素精确一致。H1 重放先完成，随后 A/B/C 重放完成，均为 PASS。
4. **生成 counts 抽样一致。** H1 的 AASS、MLLT1、ZRSR2，以及 A/B/C 每背景的
   ABCD1、PKN3、ZRANB3，共 12 个 target 的 400 细胞块，用正式 Generator 重新生成后
   与各自保存预测逐元素精确一致。这是抽查，不能写成所有 900 个 target 均重新生成过。
5. **整个提交包逐元素一致。** 解压实际 `.vcc` 的 `pred.h5ad.zst`，和提交前
   `predictions.h5ad` 分块对比全部 2,058,908,909 个非零值、全部稀疏索引/行指针、
   context/target 标签和基因顺序，全部精确一致。CLI 将 data 从 int32 改存 float32、
   indptr 从 int64 改存 int32，但数值未变；官方 prep 回执为 `normalization=counts-preserved`。
   `.vcc` SHA-256 为 `4001093ad0047309a4381abbc860462553861a8d6aaf495a13925b31c6f5f562`。

来源：[训练配置](../experiments/exp007/outputs/20260927-exp007-h1-log2fc-s17/config.yaml)、
[导出身份](../experiments/exp007/outputs/20260927-exp007-h1-log2fc-s17/predictions/leaderboard-round-0512-seed-101/export-identity.json)、
[导出实现](../experiments/exp007/src/submission.py)、
[官方打包回执](../experiments/exp007/outputs/20260927-exp007-h1-log2fc-s17/predictions/leaderboard-round-0512-seed-101/prep.json)。

这些证据未发现换模型、核心生成逻辑变化、基因/标签错位或打包归一化造成的数值偏差。
本地与官方模型输入的背景、NTC、基因轴和靶点面板确实变化：H1 已见靶点为 269/297，
官方面板为 258/300；H1 NTC 38,176 个，A/B/C 输入 NTC 每背景 18,400 个。
这构成迁移条件变化，不能由本次检查定量分离各因素对评分的影响。

另一个已有结果是同口径 H1 的 shared_response 对照 Overall 为 `0.1867720084`，
高于第 512 轮模型的 `0.0204194149`。本地模型为正不代表它已优于简单迁移基线。
来源：[完整本地指标](../experiments/exp007/outputs/20260927-exp007-h1-log2fc-s17/metrics.json)。

## exp004 到 exp007：实际评分协议变更

两者均锁定 cell-eval2 commit `5e64833518a6603a0301cbe28185d49c30f4a986`。保存的 EvalConfig 逐字段比较仅 num_threads（8 → 6）、pert_chunk（128 → 64）、cache_real 路径不同；rule_digest 均为 `fb5aa56b74368a7bc5befc8ccca8ca02a2cf4c8c7ad59b156ab86c3ae0859762`。六项 raw 定义、Wilcoxon/BH 阈值、归一化尺度、control_source=real、PDS 排除靶基因设置及 Overall 六项算术平均均未变。replicate 均为五次拆半、base seed 0。

但 EvalConfig 不包含完整的 baseline 构造和 reference 选择逻辑：

| 项目 | exp004 H1 | exp007 H1 |
|---|---|---|
| mean-response baseline | 同一均值向量复制到所有扰动细胞，float32 | 官方 dispersed 重采样并逐基因缩放，float32，可非整数 |
| baseline profile 排除自身靶基因 | `exclude_target_gene=False` | `exclude_target_gene=True` |
| 真实扰动细胞 | 每靶点最多 128 | 每靶点最多 400，不放回 |
| 真实 NTC | 独立 reference-half，3,072 | 完整 pool，38,176 |
| 真实参考总细胞 | 40,756 | 149,639 |
| 靶点 / 基因数 | 297 / 18,005 | 297 / 18,008 |
| NMAE 真实侧筛选后任务数 | 207 | 255 |
| 预测采样 | 匹配 reference 数量；101/202/303 三 seed 汇总 | 每靶点 400；seed 101 |

这里的 exclude_target_gene 变化只指 **baseline profile 构造**，不能误写为六项指标中的 PDS 靶基因排除开关变化。平铺均值与 dispersed 的细胞间方差不同，会影响 baseline 的 Wilcoxon 显著性及 Fidelity、Reach、Jaccard；reference 数量变化也会影响真实显著基因集和 replicate anchors。尚未进行固定 reference 的配对消融，不能将以下变化全部归因于单独一个开关。

源码来源：exp004 复用 `experiments/exp003-context-module-cvae/src/official.py:120–133`；exp007 为 [evaluation.py](../experiments/exp007/src/evaluation.py)。baseline 变更可追溯至 exp006 commit `b013bbeca93a971894f3a91ad03edbf4e022d809`，2026-09-26 21:11:57 -0400，`fix(exp006): align official baseline and prevent THP stalls`。exp007 继承此协议；[exp006 REPORT](../experiments/exp006/REPORT.md) 已注明旧 tiled baseline 分数不能作为同口径成绩比较。exp004 所引用源码、reference 与 bundle 位于并列 worktree `/home/jy001/Downloads/virtual-cell-challenge-worktrees/exp004`。

以下直接读取两个历史 H1 bundle 的 `baseline_agg.csv`（mean 行）与 `anchor_agg.parquet`（replicate 列），未改写历史结果：

| 指标 | exp004 baseline | exp007 baseline | exp004 replicate | exp007 replicate |
|---|---:|---:|---:|---:|
| PDS | 0.500000 | 0.496041 | 0.935988 | 0.969554 |
| Expression error | 0.954841 | 0.961232 | 0.015525 | 0.004945 |
| LFC-NMAE | 0.939665 | 0.952051 | 0.394003 | 0.341319 |
| Fidelity | 0.561495 | 0.165003 | 0.835575 | 0.871645 |
| Reach | 0.146069 | 0.088324 | 0.982044 | 0.987732 |
| Jaccard | 0.060169 | 0.006344 | 0.420783 | 0.439376 |

exp004 bundle：`experiments/exp004-shared-response/outputs/20260925-exp004-conditional-s17/cache/holdout-H1/official/reference-bundle/`（exp004 worktree）；exp007 bundle：[bundle](../experiments/exp007/outputs/20260927-exp007-h1-log2fc-s17/cache/official/H1/bundle)。真实样本信息分别来自其相邻 `reference.json` 和参考 obs；NMAE 数量来自 baseline count 行。

Fidelity 使用 `(raw - baseline) / (replicate - baseline)`。仅作标尺敏感性的算术示例：固定 exp007 raw `0.2916902808`，使用 exp007 anchors 得 `+0.179281`，代入 exp004 anchors 则约 `-0.984398`。后者不是合法的 exp007 重评成绩，因为 reference、预测样本和基因轴没有同步对齐。它明确说明仅声称“公式相同”不足以证明分数可比。

因此，exp004 本地与官方更接近，不能推出沿用到 exp007 的本地评分仍保持相同校准；exp007 在更低的本地 Fidelity baseline 上取得正分，也不能据此预计官方为正。另一方面，模型也从 CVAE/NB 生成路线换为 XGBoost log2FC 回归加 NTC 模板生成；尚不能将官方模型表现的变化全部归因于评估协议。需要固定 reference、预测及基因轴，分别切换 baseline 构造，才能隔离评分变更的贡献。本次只核查历史证据，没有重评、重训或替换任何历史分数。

## 官方包默认值与排行榜实际配置的证据边界

2026-09-27 重新在线核对固定 commit 的公开源码、比赛指标文档和 CLI 说明。结论如下：

1. **38,176 个 NTC 不是已查证的官方评分数量要求。** 这是 exp007 H1 全部可用 control 的数量，本地选择由 `data.py` / `evaluation.py` 决定。官方 baseline.py 虽在浮点求均值的精度注释中提到 38,176-cell control pool，但函数从调用者传入的 template 取 control 行，没有强制这个数量；这个注释不能证明线上 A/B/C reference 的 NTC 数量。官方比赛文档说明平台从隐藏 reference 取 held-out control，而 phase 发布的 control 是模型输入。exp007 的 reference.json 则明确写入 entire context NTC pool、same source baseline is available to prediction。因此本地模型输入和评分 control 的隔离方式没有复现官方描述。[官方源码](https://github.com/ArcInstitute/cell-eval2/blob/5e64833518a6603a0301cbe28185d49c30f4a986/src/cell_eval2/baseline.py#L505)、[官方提交与评分桥接说明](https://github.com/ArcInstitute/cell-eval2/blob/5e64833518a6603a0301cbe28185d49c30f4a986/docs/vcc2026_metrics/vcc2026-metrics.md#7-submission-file-requirements)。
2. **baseline profile 的 `exclude_target_gene=True` 有包默认值的直接证据。** `generic_response_profile` 默认开启，计算基因 g 的均值时排除针对 g 的扰动贡献。该开关与六项指标本身的 target exclusion 是不同环节。公开评分回执没有记录 r4 baseline 构建时该开关的实际值，故不能仅由函数默认参数推断部署值。[官方 profile 源码](https://github.com/ArcInstitute/cell-eval2/blob/5e64833518a6603a0301cbe28185d49c30f4a986/src/cell_eval2/baseline.py#L104)。
3. **`emit="dispersed"` 同样有包默认值的直接证据。** `build_baseline_prediction` 默认使用它；内部从 template 的 control 池有放回抽样，再乘逐基因 scale，输出 float32。它使预测细胞具有源 control 的异质性，但不是保证保持原始方差、协方差或真实扰动分布不变。此前“生成整数 counts”的表述错误，现已纠正；模型提交生成器的整数约束与这个评分 baseline 构造不同。尚无 r4 bundle 的 manifest 或构建命令证明线上使用了同一 emission。[官方 emission 源码](https://github.com/ArcInstitute/cell-eval2/blob/5e64833518a6603a0301cbe28185d49c30f4a986/src/cell_eval2/baseline.py#L305)。

官方指标文档另说明，baseline profile 来自通过细胞数与 knockdown-efficiency 筛选的 constructs，所展示的测量结果来自 cell-eval2 0.15.0 / competition rule_version 3 的官方 bundles。这不能替代当前回执所标识 `vcc2026-valA-r4+vcc2026-valB-r4+vcc2026-valC-r4` 的构建证据；bundle 名称中的 r4 也不能自动解释为 competition rule_version 4。该文档中“均值响应分配给各扰动”的文字不足以单独判定实际用了 tile 还是 dispersed。[官方 baseline 定义与来源说明](https://github.com/ArcInstitute/cell-eval2/blob/5e64833518a6603a0301cbe28185d49c30f4a986/docs/vcc2026_metrics/vcc2026-metrics.md#0-overview)。

要确认端到端一致，需要取得对应 r4 bundles 的构建配置/manifest，包括 NTC 来源及划分、profile 筛选、exclude_target_gene、emission 和随机种子。现有 status 回执只有 bundle 身份及汇总分数，没有这些字段。当前可成立的说法仅为“使用官方包默认 baseline 的本地 H1 评估”，不是“已验证复现 leaderboard 评分流程”。
