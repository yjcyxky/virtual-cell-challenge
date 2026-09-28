# 官方 r4 anchors：公开来源核查

核查日期：2026-09-28。范围是公开官方文档、固定源码和本轮已保存回执；未读取隐藏真值、联系官方、训练或重新评分。本文补充[历史协议审计](../experiments/exp007-official-scoring-protocol-audit.md)，不引入旧实验的方法效果证据。

**结论：当前可以核验公开评分规则与包的默认行为，仍不能核验线上 r4 baseline 的完整构建条件。** 本轮两个提交的回执均为 `partition=val`、`panel_id=vcc2026-val-1`、`anchor_version=vcc2026-valA-r4+vcc2026-valB-r4+vcc2026-valC-r4`；回执没有构建参数、逐背景 anchors、package commit 或 rule digest。[linear 回执](../../experiments/init-linear/outputs/init-linear-s01/predictions/official-abc/linear/official-status.json)、[shared 回执](../../experiments/init-linear/outputs/init-linear-s01/predictions/official-abc/shared/official-status.json)

## 已验证与尚未验证

下表源码均固定到官方公开 commit `5e64833518a6603a0301cbe28185d49c30f4a986`。核查当日 GitHub `main` 仍指向此提交。[官方当前提交 API](https://api.github.com/repos/ArcInstitute/cell-eval2/commits/main)

| 事项 | 直接依据 | 对当前 r4 的证据边界 |
|---|---|---|
| baseline emission | `build_baseline_prediction` 默认 `emit="dispersed"`, `seed=0`；重采样 template control 后逐基因缩放，输出可为小数。另一合法参数是 `tile`，即复制均值向量。[baseline.py](https://github.com/ArcInstitute/cell-eval2/blob/5e64833518a6603a0301cbe28185d49c30f4a986/src/cell_eval2/baseline.py#L347) | 包默认值不是部署记录；未找到 r4 使用哪一臂及其实际 seed 的公开记录。 |
| 排除自身靶基因 | `generic_response_profile` 默认 `exclude_target_gene=True`，按基因排除对应扰动对均值的贡献；匹配数量由 `n_excluded` 记录。[profile 函数](https://github.com/ArcInstitute/cell-eval2/blob/5e64833518a6603a0301cbe28185d49c30f4a986/src/cell_eval2/baseline.py#L104) | 未核验 r4 实际开关及匹配数量。这与评分指标的 target exclusion 是不同环节。 |
| baseline profile 来源与筛选 | 指标说明称：通过细胞数与 knockdown-efficiency 筛选的 constructs，其平均 count 向量按 construct 等权平均。公开 profile 函数只聚合调用者传入的数据，不执行这两项筛选。[官方定义](https://github.com/ArcInstitute/cell-eval2/blob/5e64833518a6603a0301cbe28185d49c30f4a986/docs/vcc2026_metrics/vcc2026-metrics.md#0-overview)、[函数实现](https://github.com/ArcInstitute/cell-eval2/blob/5e64833518a6603a0301cbe28185d49c30f4a986/src/cell_eval2/baseline.py#L145) | r4 的筛选阈值、入选 construct 集合及所用细胞未核验；不能将本地全部合格扰动均值说成已复现线上 profile。 |
| 输入与评分 NTC | 官方说明：发布的 control 用作模型输入；平台从隐藏 reference 提取 held-out control，附加到提交后评分，`control_source=real`。[上传与评分的桥接](https://github.com/ArcInstitute/cell-eval2/blob/5e64833518a6603a0301cbe28185d49c30f4a986/docs/vcc2026_metrics/vcc2026-metrics.md#7-submission-file-requirements) | 契约已公开；r4 评分 NTC 的数量、物理身份、划分 seed、baseline template 的 NTC 来源均没有对应部署回执。源码里的 38,176 是数值精度注释，不是数量要求。 |
| replicate 拆分与 seed | 公开规则固定 `base_seed=0`, `n_splits=5`。具体种子由 `SeedSequence(base_seed).generate_state(n_splits)` 导出；各 perturbation/control 拆为不相交两半，各半使用自己的 control。LFC-NMAE 使用 full-reference gate 的专门估计器。[competition.py](https://github.com/ArcInstitute/cell-eval2/blob/5e64833518a6603a0301cbe28185d49c30f4a986/src/cell_eval2/competition.py#L37)、[anchor.py](https://github.com/ArcInstitute/cell-eval2/blob/5e64833518a6603a0301cbe28185d49c30f4a986/src/cell_eval2/anchor.py#L135) | 这是可复现的公开规则；尚未取得 r4 的 `derived_seeds`、估计器和 source fingerprint 实际记录。不能将 `control_source_effective=pred` 的 replicate 元数据误作模型评分 control 被替换。 |
| package 与规则版本 | 固定公开包版本是 `0.16.0`；源码 `competition_payload` 的 `rule_version` 是 **3**，注释明确描述此前 `-r3` bundles 在 `0.15.0` 下构建。[包版本](https://github.com/ArcInstitute/cell-eval2/blob/5e64833518a6603a0301cbe28185d49c30f4a986/pyproject.toml#L1)、[规则版本](https://github.com/ArcInstitute/cell-eval2/blob/5e64833518a6603a0301cbe28185d49c30f4a986/src/cell_eval2/competition.py#L374) | **`r4` 不等于已证明 `rule_version=4`。** 当前回执缺少 package commit/version 和 rule digest，无法认证该映射。 |

## 为什么 bundle 名称与默认值不足以认证

`build_real_bundle(real, baseline_pred, ...)` 接收已生成的 baseline 矩阵；它不要求该矩阵由默认 baseline 生成器产生。manifest 的 `profile` 字段是指标集合 `config.metrics`，不是 baseline 均值所使用的 construct 集合。因此同一 `vcc2026` preset、有效 bundle 或相同包版本，都不能单独证明 baseline emission、排除开关和 profile 筛选一致。[bundle 入口](https://github.com/ArcInstitute/cell-eval2/blob/5e64833518a6603a0301cbe28185d49c30f4a986/src/cell_eval2/real_bundle.py#L248)、[manifest 字段](https://github.com/ArcInstitute/cell-eval2/blob/5e64833518a6603a0301cbe28185d49c30f4a986/src/cell_eval2/real_bundle.py#L395)

公开代码能够保存用于核验的记录：bundle manifest/config、anchor meta/splits、baseline 构建 sidecar 中的 `emit/seed/exclude_target_gene/n_excluded`。但 bundle 入口的 baseline metadata 并不自动补齐这些构建记录。后续若官方公开 r4 信息，应核对这几类记录及源数据 fingerprint、profile 筛选说明和 package commit，无需取得隐藏表达矩阵。[baseline 构建 sidecar](https://github.com/ArcInstitute/cell-eval2/blob/5e64833518a6603a0301cbe28185d49c30f4a986/src/cell_eval2/baseline.py#L1477)、[anchor metadata](https://github.com/ArcInstitute/cell-eval2/blob/5e64833518a6603a0301cbe28185d49c30f4a986/src/cell_eval2/anchor.py#L536)

官方 CLI 将 anchor set 定义为每背景 bundle 的身份字符串；每背景先按自己的 baseline/replicate 缩放，再汇总。仅有三背景汇总 raw 与 normalized 值不足以解出各背景的 anchors。CLI 示例仍展示 `r1`，不是当前 r4 配置说明。[官方 CLI 分数与 stamp 说明](https://vcc-cli-wiki.virtualcellchallenge.org/#5-submit)

## 本地 2×2 协议诊断的解释范围

拟登记的两因素是 `emit={dispersed,tile}` × baseline profile 的 `exclude_target_gene={true,false}`。这些是**评分 baseline** 的变化；模型的 shared 响应、预测生成器及已有预测保持冻结。

- 固定本地 reference、两臂预测、基因/靶点轴、NTC、scorer 和 replicate 配方后，可测量 baseline 变化造成的 normalized 分项、Overall 和两模型差距变化；同时检查两模型 raw 指标应保持不变。
- 同一 reference 与 replicate 配方的 replicate anchors 应一致；不得为了获得想要的分数，借用另一 reference 的 anchors。每个新 baseline 都须在匹配群体上计算自身分数，并记录有效指标、截断和退化状态。[官方 bundle 双端约束](https://github.com/ArcInstitute/cell-eval2/blob/5e64833518a6603a0301cbe28185d49c30f4a986/src/cell_eval2/real_bundle.py#L248)
- 这是本地标尺敏感性诊断，不能识别 r4 的实际配方，不能量化 H1→A/B/C 背景迁移的因果贡献，也不能通过选择“更接近官方”的一臂宣布协议认证。两模型同序仅支持这次有限比较的排序一致，不构成普遍校准证据。

## 检索记录与回执身份

核查了固定公开树中的 14 份 Markdown/preset 文件、`baseline.py`、`anchor.py`、`real_bundle.py`、`competition.py`、公开提交历史及当前 CLI 说明，未找到当前 r4 构建 manifest 或命令。此结论限于这些公开来源，并不声称相关记录不存在。公开仓库只有四个可见提交；最新提交明确说明它是内部源码的树拷贝，没有携带原始历史，不能据此追溯被省略的部署构建过程。[官方提交历史 API](https://api.github.com/repos/ArcInstitute/cell-eval2/commits?per_page=100)、[冻结提交说明](https://github.com/ArcInstitute/cell-eval2/commit/5e64833518a6603a0301cbe28185d49c30f4a986)

| 回执 | entry_id | `official-status.json` SHA-256 |
|---|---|---|
| linear | `BpAGRy6zY8YFxKlLvmua` | `f4656ae92fda3bfc45949b148083d6d81215c8ff17be29632794ecf6474c0be6` |
| shared | `vWd1Z4K9tJ2tjfns2gSj` | `024dac1a0d29efbf20adf0395b771bebf7fa8fd14c095db7b2bcfa73b6fa2c2e` |

本文是 `protocol/constraint` 的来源核查，不是新模型有效性结论；本地诊断的登记、执行与关闭结论仍以 DAG 和 Ledger 为准。
