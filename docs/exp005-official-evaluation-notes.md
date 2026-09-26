# EXP005：官方评估 API 与内存约束

调查日期：2026-09-26。只读源码调查及小型 API 能力检查；没有安装依赖、修改训练代码或执行训练。源码固定为 `ArcInstitute/cell-eval2@5e64833518a6603a0301cbe28185d49c30f4a986`。下文链接均指向该提交。

## 结论

优先尝试官方 `build_real_bundle` / `compute_metrics`，传入以 C-contiguous 整数 memmap 为 X 的普通 AnnData，并显式使用 CUDA/gpudge。在 vcc2026 的 counts、target_sum=1e6 配置下，已核实它不构造整张细胞×基因 float CPM 矩阵；机器当前约 60 GiB 可用内存时有实现可行性，但 anchor 拆半与 DE 表的峰值尚需实测。不能把 memmap 当成硬内存上限，也不能用缺少 moments 的 `score_cellstream` 代替完整六指标评估。若官方直接路径实测不适合，才采用下述分组读取、官方统计/DE/指标函数及完整面板聚合的薄编排，并做数值一致性验证。

参考样本可预先固定为每真实扰动至多 400 个不放回细胞、NTC 全部保留、所有合格扰动均保留。少于 400 的真实细胞不 bootstrap 补足；训练聚合仍可用全部合格细胞。这个本地抽样方案不是 evaluator 的隐式行为，也不等同官方隐藏 reference；其实际样本数、深度、原生可测基因轴必须记录。400 上限本身不保证内存安全。

## 已核实的接口限制

| 路径 | 实际行为与限制 |
| --- | --- |
| `compute_metrics(pred, real, config=cfg)` | H5AD 路径先 backed，但冷缓存 DE 经 `_materialize(...).to_memory()`；路径输入减少同时驻留两侧，不能保证单侧足够小。[run.py](https://github.com/ArcInstitute/cell-eval2/blob/5e64833518a6603a0301cbe28185d49c30f4a986/src/cell_eval2/run.py#L233) |
| `build_real_bundle(real, baseline_pred, ...)` | baseline leg 后 `load_anndata(real, backed=False)`；anchor 每个 split 复制两半。直接传入 AnnData/memmap 虽可避开第一步读盘复制，后续随机行切片 `.copy()` 仍会分配。[real_bundle.py](https://github.com/ArcInstitute/cell-eval2/blob/5e64833518a6603a0301cbe28185d49c30f4a986/src/cell_eval2/real_bundle.py#L248)、[ceiling.py](https://github.com/ArcInstitute/cell-eval2/blob/5e64833518a6603a0301cbe28185d49c30f4a986/src/cell_eval2/ceiling.py#L138) |
| `score_cellstream` / `score_h5ad_manifest` / `score_piece` | 走 `partition_inmem` reference，持久化格式没有 moments；`score_piece` 明确拒绝需要 moments 的指标，故不能完整计算 vcc2026 MSE。[拒绝位置](https://github.com/ArcInstitute/cell-eval2/blob/5e64833518a6603a0301cbe28185d49c30f4a986/src/cell_eval2/partition_inmem.py#L1148) |
| `cell_eval2.scale.score_streaming_cell` | 另一条真正支持 moments 的完整 raw 指标路径，保持全 panel PDS 与 MSE correction budget；需要 cellstream，DE 需要 gpudge/CUDA；没有 `cache_real`，每次重算真实侧；无 `mem_budget` 参数，不能把另一个接口的预算参数传进来。[scale.py](https://github.com/ArcInstitute/cell-eval2/blob/5e64833518a6603a0301cbe28185d49c30f4a986/src/cell_eval2/scale.py#L521) |
| 同上，fractional baseline | 对 cell archive 强制检测 input scale；fractional 每细胞总数会被判为 lognorm 并拒绝，不能假设允许 vcc2026 tiled mean baseline。[检查](https://github.com/ArcInstitute/cell-eval2/blob/5e64833518a6603a0301cbe28185d49c30f4a986/src/cell_eval2/scale.py#L566) |

`build_real_bundle._baseline_leg` 只在内部将 `allow_fractional_counts=True`，因为平均响应 baseline 本来就是小数；预测提交仍要求整数。不要通过舍入 baseline、只复制一行或改成任意噪声生成来省内存，这会改变标尺。`build_baseline_prediction` 当前默认是 `emit='dispersed'`，复现官方已定义 context-mean tiled comparator 时必须明确选择相应 emission，而非依赖默认。[baseline leg](https://github.com/ArcInstitute/cell-eval2/blob/5e64833518a6603a0301cbe28185d49c30f4a986/src/cell_eval2/real_bundle.py#L201)、[baseline API](https://github.com/ArcInstitute/cell-eval2/blob/5e64833518a6603a0301cbe28185d49c30f4a986/src/cell_eval2/baseline.py#L347)

## 保持官方数值的分组编排接口

固定基础配置，不把 CPU/GPU 切换视为无影响工程选择：

```python
from dataclasses import replace
from cell_eval2 import EvalConfig

preset = EvalConfig.from_preset("vcc2026")
cfg = replace(preset, pert_col="target_gene", device="cuda", num_threads=6,
              de=replace(preset.de, backend="gpudge"))
```

真实和预测的全量 NTC 参照必须一致。常规 `control_source='real'` 使用真实 NTC；不能为每个 perturbation 随机抽 400 NTC。特征 baseline 也可以用全部 NTC，不必另切 feature/reference pool。官方 replicate anchors 的拆半是评分标尺所需的独立步骤，不能因此把全量 feature baseline 改成半池。[控制替换](https://github.com/ArcInstitute/cell-eval2/blob/5e64833518a6603a0301cbe28185d49c30f4a986/src/cell_eval2/run.py#L894)

可复用的官方底层函数：

1. `run._side_bulks(..., moment_norms=...)`：对一个完整扰动组或小批完整组计算官方 bulk 与 `GroupMoments`。若使用 CPU 的 `prep.pseudobulk_bulk_lognorm_with_moments`，必须与选择的基准设备路径核对差异。拼接时保留 label 对齐及控制组；输出统计量通常是 `P×G`，不是 `N_cells×G`。[统计接口](https://github.com/ArcInstitute/cell-eval2/blob/5e64833518a6603a0301cbe28185d49c30f4a986/src/cell_eval2/run.py#L676)、[moments 语义](https://github.com/ArcInstitute/cell-eval2/blob/5e64833518a6603a0301cbe28185d49c30f4a986/src/cell_eval2/moments.py#L45)
2. `de_compute.compute_de`：对小批扰动，以完整 NTC AnnData 作 `reference`，传入 preset 的 `mean_calc/epsilon/target_sum/filter/fdr_scope`。gpudge external-ref 避免连接 NTC 与大预测矩阵。CPU pdex/scanpy 不支持这个 AnnData external-reference 接口，需要连接数据，不应悄悄切换。[签名](https://github.com/ArcInstitute/cell-eval2/blob/5e64833518a6603a0301cbe28185d49c30f4a986/src/cell_eval2/de_compute.py#L594)
3. `de.prepare_de` → `run.dispatch_de_metrics`：继续使用官方 DE gate、排序、方向、LFC、空集合约定。target resolution 必须从完整背景 gene/target universe 解析，不按小批重新解释；可逐扰动计算后保存 raw 行，避免同时持有几十百万行完整 DE 表。[prepare_de](https://github.com/ArcInstitute/cell-eval2/blob/5e64833518a6603a0301cbe28185d49c30f4a986/src/cell_eval2/de.py#L591)
4. `run.dispatch_anndata_metrics`：传完整真实 panel 的 bulks、全组 moments，以及完整预测 `pred_bulks_full`。PDS 排名不能只比较当前批次；MSE 的 correction budget 也依赖整个预测 panel，不能先逐批求最终 MSE 再平均。[dispatcher](https://github.com/ArcInstitute/cell-eval2/blob/5e64833518a6603a0301cbe28185d49c30f4a986/src/cell_eval2/run.py#L445)、[官方 streaming 如何保留全 panel](https://github.com/ArcInstitute/cell-eval2/blob/5e64833518a6603a0301cbe28185d49c30f4a986/src/cell_eval2/scale.py#L651)
5. `aggregate_metrics_wide(raw)` 与 `score_metrics(...)` 保留官方聚合、标尺和 clamps，不重写得分公式。standalone `anchor=` 必须同时传 baseline aggregate 及由 run meta 派生的 `anchor_expect`；不是随意传一组浮点 anchor。`real_bundle=` 则验证完整 provenance。[score_metrics](https://github.com/ArcInstitute/cell-eval2/blob/5e64833518a6603a0301cbe28185d49c30f4a986/src/cell_eval2/score.py#L511)

anchors 不能只做五次普通 half-vs-half 指标平均：LFC NMAE 使用 full-reference 的 gate 与分母。必须沿用 `anchor._derive_seeds(base_seed=0, n_splits=5)`、按 `pert_col` 的不放回等大拆半、每半独立 NTC，并用 `lfc_nmae_ref._nmae_ref_from_tables(de_full, de_a, de_b, p_adj_threshold=0.05, min_gate_size=10, target_resolution=...)` 替换其 NMAE 参照；不要使用 sqrt(2) 修正版。小批处理三份 DE 表可以避免全表驻留，但其拼接统计与官方 anchor 输出仍需验证。[官方 anchor 逻辑](https://github.com/ArcInstitute/cell-eval2/blob/5e64833518a6603a0301cbe28185d49c30f4a986/src/cell_eval2/anchor.py#L237)、[full-gate 算子](https://github.com/ArcInstitute/cell-eval2/blob/5e64833518a6603a0301cbe28185d49c30f4a986/src/cell_eval2/lfc_nmae_ref.py#L103)

## 内存与机器事实

单是 2,000,000×7,669 float32 矩阵即约 61.4 GB（57.1 GiB）；4,000,000 个预测细胞则约 122.7 GB。验证每项为非负整数且 ≤65535 后，可无损保存为 uint16，分别约 30.7 GB、61.4 GB 磁盘映射文件；不能截断超界值或将 fractional mean baseline 转成整数。普通 AnnData 的 X 可以保留 memmap，路径 H5AD 的 backed 行为不同。只携带评估所需 X/obs/var，不带 raw/layers，避免 anchor 同时复制无关表达层。文件映射页可由操作系统回收，不代表任意后续匿名内存分配都安全。

直接路径的源码证据：

- `run._materialize` / `io.load_anndata` 对普通 AnnData 直接返回；`run._side_bulks` 的 counts+CUDA 路径使用 `inmem_pseudobulk` 分行块 accumulator，源码明确避免 full normalize transient。[run.py](https://github.com/ArcInstitute/cell-eval2/blob/5e64833518a6603a0301cbe28185d49c30f4a986/src/cell_eval2/run.py#L694)
- `de_compute.compute_de` 的 `native_cpm = is_counts and target_sum == 1e6` 让常规调用直接把原始 counts 送入 gpudge；不需要启用另一条 streaming 接口的 `native_gpu_normalize` 参数。其他 target_sum 或 lognorm 输入不能套用这个结论。[de_compute.py](https://github.com/ArcInstitute/cell-eval2/blob/5e64833518a6603a0301cbe28185d49c30f4a986/src/cell_eval2/de_compute.py#L730)
- 整数 dtype 在 `_is_all_integer` 直接通过整数性检查，float baseline 逐行块检查；严格 hash 以 buffer protocol 读 C-contiguous dense X，不构造 `.tobytes()` 全量副本。非连续矩阵仍可能被 `ascontiguousarray` 复制。[norm.py](https://github.com/ArcInstitute/cell-eval2/blob/5e64833518a6603a0301cbe28185d49c30f4a986/src/cell_eval2/norm.py#L279)、[cache.py](https://github.com/ArcInstitute/cell-eval2/blob/5e64833518a6603a0301cbe28185d49c30f4a986/src/cell_eval2/cache.py#L70)
- 本机已安装 `gpudge==0.9.1` 的 `_csr_dense.ensure_csr` 对 dense 输入原样返回；`csr_row_sums` 对 dense 使用 float64 reduction，仅输出每行一个总数。`__init__.py` 的 named-reference 分支按基因块上传 NTC，并逐扰动上传 cells；CPM 在 GPU tile 内完成。`_refpool.inmem_external_ref_de` 则驻留排序后的完整 NTC float32 表，逐 target tile 计算，不连接整张预测矩阵。这些为已安装版本的直接源码检查；正式 exp005 必须锁定并记录实际 gpudge 版本。[官方 gpudge 仓库](https://github.com/ArcInstitute/gpudge)

主要峰值来自 anchor：两半 `.copy()` 合计约一张 uint16 原矩阵；随机索引的临时数组可能再占半张（2m×7669 示例约 46 GB decimal）。同时保留 full-reference DE，以及正在计算的 half DE、moments 和 GPU buffer。若有 9,000×8,000=7200 万 DE 行，每份表仅 8 个数值列即约 4.6 GB，另有字符串列和处理临时量；实际 control CPM gate 会减少部分输出行，但不能预先假定足够小。因此约 60 GiB 的可用内存让直接路径值得优先测量，尚不能保证 K562 全流程必定通过。预测 memmap 自身没有 anchor 拆半，通常更易控制驻留。CUDA allocator 缓存与 CPU 分配都计入 GB10 的统一物理内存预算；不能当作两套独立容量。

官方 `MemBudget` planner 只是批量估算，且所在 partition 路径不能直接满足完整六指标。[planner](https://github.com/ArcInstitute/cell-eval2/blob/5e64833518a6603a0301cbe28185d49c30f4a986/src/cell_eval2/h5ad_manifest.py#L98)

只读检查现有 exp003 环境：NVIDIA GB10、驱动 580.95.05、compute capability 12.1；`gpudge` 的 in-memory external-reference capability 为 True；`cellstream` 未安装。直接调用官方 `_reject_moments_metrics` 于 vcc2026 成功复现 `NotImplementedError`。这些仅确认接口与硬件可见，不证明 exp005 环境已就绪，也不是大批次性能测试。CPU DE 路径会建立整张 normalized/log1p 矩阵，不能作为同等内存成本的透明 fallback。

## 必须完成的数值验证

薄编排应在一个具有足够 DE 信号的小型整数 counts 数据上，对照同 commit、同 backend 的 `compute_metrics` + `build_real_bundle`：逐 raw metric、各聚合值、五个 anchor、六个 scaled 分数与 Overall；包含 panel-target exclusion、对照池、MSE panel correction、LFC full gate、ties、少于 400 的真实组。再改变小批大小确认结果不依赖分批。浮点误差阈值应按官方同设备测试标准确定，而不是将 DE 集合/排序差异藏进宽容差。

资源验证应记录最大 NTC 背景、实际全 panel 及 anchor 的 host/GPU 峰值，以执行时可用资源留出余量。30 GB 不是用户指定的科学条件；不应人为限制可用约 60 GiB 的机器。若直接路径超出实际资源，再使用经过官方数值对照的薄编排，不能缩减面板、切掉 NTC 或替换官方指标来“通过”。
