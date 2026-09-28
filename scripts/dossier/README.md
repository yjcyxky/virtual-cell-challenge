# 数据评估工具

范围、方法决策、验收和固定结果索引由 [GitHub Issue #1](https://github.com/yjcyxky/virtual-cell-challenge/issues/1) 及其关联 Issues 管理。此处只说明可执行接口。

```bash
micromamba run -n virtual-cell uv sync --project scripts/dossier --locked \
  --python /home/jy001/micromamba/envs/virtual-cell/bin/python --no-python-downloads
micromamba run -n virtual-cell uv run --project scripts/dossier --locked \
  python scripts/dossier/profile_arc_vcc2025_h1.py \
  --inventory data/assessments/inventory-20260919/report.json \
  --output data/assessments/h1-structure-reproduction
```

`rna.py` 提供 H5AD 只读扫描、测量轴与身份诊断；`render.py` 从结果包直接生成可筛选 HTML，不把缺失值填成零。来源适配器负责方法、元数据语义和适用性，不能仅更换文件名后套用。

2026 研发使用 `challenge.py` 直接核验五背景原始计数，冻结官方轴映射、物理 NTC 池和 S0–S4 划分，并提供固定版本 cell-eval2 的 reference/score 接口。命令、产物和边界见 [challenge 协议](../../docs/research/challenge_protocol.md)；当前研究状态由 DAG/Ledger 管理。`minimum_reference_cells` 是结构门槛，实际六指标有效性仍由官方引擎决定。

输出目录必须是新目录且位于原始数据之外。JSON、Parquet sidecars、HTML 与校验和是分析产物，不承担任务状态管理。旧档案保留其历史身份；原有首块整数抽查、固定深度参照与残留表达缺失填零路径已替换。
