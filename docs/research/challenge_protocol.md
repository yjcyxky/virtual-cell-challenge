# 2026 challenge 数据与评估契约

本轮直接扫描 H1、K562 GWPS、RPE1、HepG2、Jurkat 的七个原始 H5AD，并核对完整文件 SHA-256 与登记 SOURCE。没有读取历史实验结果。输入版本、计数核验、缺测掩码及划分结果已经生成；**实际 cell-eval2 指标有效数、reference/anchor 数值和模型效果仍需在对应 Experiment 中执行后取得**。

## 官方目标及测量覆盖

目标为已登记的 `vcc2026-val-1`：官方 `gene_names.csv` 中 **18,533** 个基因的完整标签与顺序，`pert_counts.csv` 中 **300** 个靶点，A/B/C 每背景每靶点生成 **400** 个非负整数计数细胞。当前文件身份见 [数据审计](challenge_2026/data_audit.json)；提交规格由 [官方 CLI 文档](https://vcc-cli-wiki.virtualcellchallenge.org/#submission-requirements-2026)规定。后续官方面板更新须另登记版本。

| 背景 | 官方轴内已测基因 | 官方 300 靶点中有合格来源细胞 | 输入 NTC | 评分 NTC | 至少 4 个真实细胞的辅助靶点任务 |
|---|---:|---:|---:|---:|---:|
| H1 | 18,074 | 25 | 18,400 | 19,104 | 300 |
| K562 GWPS | 7,680 | 272 | 18,400 | 37,737 | 9,862 |
| RPE1 | 8,260 | 0 | 5,726 | 5,759 | 2,389 |
| HepG2 | 9,024 | 0 | 2,474 | 2,502 | 2,388 |
| Jurkat | 8,283 | 0 | 5,992 | 6,021 | 2,383 |

官方完整轴是建模与输出目标。每个背景的本地真值仅覆盖其中一部分：使用该背景已测官方基因子轴、保持官方顺序，单列覆盖；不将未知真值补零，不默认裁为五背景共同交集。本地归一化分数基于本地 reference，不能当作线上完整面板分数。其他有效来源靶点作为辅助监督；官方面板交集单列，RPE1/HepG2/Jurkat 的辅助外推成绩不等于直接验证了官方靶点。

## 已执行的正确性核查

- 完整读取七个表达矩阵，检查有限、非负整数及零文库。所有文件通过计数检查；native 和已测官方子轴均无零文库细胞。实测全零基因仍属于已测列，保留其测量身份。
- H1 以研究、背景、batch、barcode 标识物理细胞；比较完整 native 表达向量、guide 与靶点标签后确认三文件的重复 NTC 一致，去除 **76,352** 条重复记录。验证和测试文件缺失的 Ensembl 注释仅在完整 native 基因顺序一致时引用同源 Training 文件，并逐列记录来源。
- 官方标签和顺序优先；HGNC 提供唯一别名桥接，Ensembl 冲突和多列映射到同一基因均显式排除。TIAF1/MYO18A 等官方独立槽位保持区分，TMEM104 等历史官方标签不被新名称替换。详细原因见 [逐列映射](challenge_2026/gene_mapping.csv.gz) 和 [靶点映射](challenge_2026/target_mapping.csv)。
- NTC 在物理去重之后，按 batch 内稳定哈希排序拆成互斥输入/评分池。输入池最多 18,400 个，按 batch 比例确定配额；被输入上限裁掉的细胞保留为 unused，不回流评分池。完整池身份哈希、每批次数量见数据审计。
- 不使用响应强度、敲低效果、线粒体比例或 guide 一致性筛选本轮任务。K562/RPE1/HepG2/Jurkat 分别有 222/105/43/151 个细胞因靶点身份不合格被排除，原因保留；其余弱响应留给后续 QC 归因实验。生物独立重复未确认，记录 unknown。

## 划分与结构可评分性

[tasks.csv](challenge_2026/tasks.csv) 保存每背景/靶点的合格细胞数、guide/batch 覆盖、选定 reference 的物理身份哈希和排除原因。[splits.json](challenge_2026/splits.json) 保存 **15 个划分**的实际训练靶点与评价靶点列表；这些是协议定义，尚未创建 15 个训练或 W&B run。

| 划分 | 冻结规则 |
|---|---|
| S0 | 已见条件管线诊断：每靶点按物理身份选一半、最多 400 个作 reference；训练排除这些评价身份 |
| S1 | 全背景按统一靶点哈希分桶，名义 60% 训练、20% 内层验证、20% 测试；测试和验证靶点从所有训练背景监督中删除 |
| S2 | 五背景分别留出全部扰动标签，仅评价训练背景见过的靶点；允许目标背景的输入 NTC |
| S3 | 同时执行背景留出与全局靶点分桶；其余背景也不得保留测试靶点的响应标签 |
| S4 | Arc H1、Replogle 两背景、Nadig 两背景按研究整体留出，已见/未见靶点分开报告 |

真实 reference 按固定身份排序每靶点最多取 400 个；不足时保留实际数量，不复制真实细胞。至少 4 个真实细胞是本地拆半所需的结构门槛，**不是官方规定的 QC 阈值，也不保证六指标都有效**。DE 显著性、NMAE gate、PDS 候选面板及 anchor 可用性只能由实际 cell-eval2 调用确定。所有未定义指标和有效数必须保留；不为了得到有限 Overall 更换评价人群。

S2 的官方靶点重叠在 H1/K562 各仅 25 个，其他背景为 0；S3 的官方重叠在 H1/K562 分别为 6/52。整体辅助面板上的 PDS 与官方重叠子面板上的 PDS 不同；后者如单独评分，必须另建相应 reference/bundle，不能从全辅助面板的排名分数直接抽取后宣称同口径。

## 官方 cell-eval2 接入与边界

已固定官方 [cell-eval2 commit 5e648335](https://github.com/ArcInstitute/cell-eval2/tree/5e64833518a6603a0301cbe28185d49c30f4a986)，安装版本为 0.16.0；2026-09-28 读取的上游 HEAD 与此一致。[scorer.json](challenge_2026/scorer.json) 由安装包导出完整 `vcc2026` preset、六指标成员和 Python/YAML 源码哈希。

正式流程复用 [challenge.py](../../scripts/dossier/challenge.py)：`scorer_config` 只允许计算资源和缓存路径覆盖；`build_reference_bundle` 调用官方 generic response、dispersed baseline 和 `build_real_bundle(base_seed=0, n_splits=5)`；`score_prediction` 调用官方 raw、aggregate 与 normalized score 接口，核对六分项和 Overall。anchor 的独立 control、full-gate NMAE 以及退化判定交由固定版本官方实现。

评分器的 oracle baseline 可读取 reference 响应，仅用于标定评分尺度；它不是可供模型训练或推断使用的共享响应方法。建模的 M-SHARED/M-SOURCE 只能使用各划分允许的训练标签。

数据契约冻结固定的是参考人群、面板、处理规则与生成配方。大型 reference 和 anchors 在实际 Experiment 的 cache 中构建，记录内容哈希后才能报告 normalized Overall；分数失败时保留官方失败原因。实际 CPU/GPU、DE backend 及版本也必须随实验冻结，不能将不同 backend 的数值默认为完全一致。合成数据接口验证只证明官方调用链可运行，不构成本地方法证据。

## 重建与管理接入

```bash
micromamba run -n virtual-cell uv sync --project scripts/dossier --locked \
  --python /home/jy001/micromamba/envs/virtual-cell/bin/python --no-python-downloads
micromamba run -n virtual-cell uv run --project scripts/dossier --locked \
  --python /home/jy001/micromamba/envs/virtual-cell/bin/python --no-python-downloads \
  python scripts/dossier/challenge.py --config docs/research/challenge_protocol.json \
  --output /tmp/vcc2026-contract-reproduction
```

输出必须是不存在的新目录，原始数据只读；正式训练内重建时放入所属 Experiment 的 cache。配置见 [challenge_protocol.json](challenge_protocol.json)，输出哈希见 [artifacts.json](challenge_2026/artifacts.json)。全部报告来源登记进现有 DAG/Ledger，模型有效性账目仍为 pending。

下一执行单元是 C-BASE：落实具体 `outer_split × target_partition × seed`、完整配置、模型代码/环境和预先约定的结束条件，再启动独立拟合。实际配置必须绑定 DAG 协议的 `binding`；`check/gate/bind` 会拒绝缺失或不同的 benchmark 身份。取得真实 reference、anchors、训练与评估结果之后，才关闭 E-BASE 并选择后续比较。
