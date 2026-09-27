# exp007 本地与官方评分口径核查

调查日期：2026-09-27。只读取已有源码、配置、指标及官方文档；未修改训练代码、环境、历史 run 或重新提交。

## 结论

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
