# exp006 K562 官方评估长时间无输出：诊断记录

> 本文保留当时的诊断和恢复验证快照；下文 PID、运行状态和后续建议均属于记录时点。

日期：2026-09-26，美东时间约 19:52–20:00。

Run：`20260926-exp006-loco-residual-s17`；PID：`3615234`。
训练代码：`7eb742f59957ebbeb774ccddeba66dbe25487a7e`。
评分：固定 cell-eval2 `5e64833518a6603a0301cbe28185d49c30f4a986`，CUDA/gpudge。
本次只诊断并保存结论，没有修改活动 run 的代码、输入、虚拟环境或主机 THP 设置，没有重启训练。

## 结论

当前瓶颈是 dense → CSR 转换时的大数组分配触发透明大页（THP）同步内存整理。
在 GB10 的共享内存系统上，页迁移伴随设备地址映射失效处理，CPU 大量消耗于内核，GPU 没有持续计算。
它发生在官方 bundle 的 **mean-response baseline pseudobulk** 阶段，尚未完成 baseline leg，
并非已进入五次 replicate split 的评分。训练本身已经完成 K562 折第 1 轮。

确认调用链（只读 GDB 栈）：

```text
Evaluation.prepare (evaluation.py:107)
  build_real_bundle (real_bundle.py:295)
    _baseline_leg (real_bundle.py:228)
      compute_metrics → _run_metrics → _side_bulks
        inmem_pseudobulk (streaming_bulk.py:371)
          make_blocks (streaming_bulk.py:368)
            scipy.sparse.csr_matrix(Xb)
              scipy.sparse._coo.__init__ (line 95)
                idx.astype(index_dtype, copy=False)
```

原生栈先后采到 `_aligned_cast_long_to_int`、`mapiter_get`。
官方库每块 100,000 行；K562 参考矩阵为 1,915,223 × 8,135，约 29.0 GiB uint16；
mean-response baseline 为同形 float32，约 58.0 GiB。每个满块有 8.135 亿个位置，
在稠密输入转 CSR 时还会创建 COO 索引和转换副本，远大于普通小数组操作。
最后一次栈快照的 `start=1200000, stop=1300000`，即该次遍历的第 13/20 块。
这不是整个 bundle 或整个实验的完成百分比。

## 证据与排除

1. 5 秒 `/proc` 采样：用户态 CPU ticks 增量 0、内核态增量 500；11,264 次 minor faults，0 次 major faults；日志字节数未增加，距上次更新约 116 分钟。
2. 独立 `perf stat` 5 秒：约 5.026 秒 task-clock，22,528 次 minor faults，0 次 major faults。
3. `perf record -F 49 -g -p 3615234` 的主要 PMU 组约 254 个样本：约 98.99% 调用栈经过页错误处理，98.55% 经过直接内存整理；主要耗时包含 `migrate_pages`、`arm_smmu_cmdq_issue_cmdlist`、`ptep_clear_flush`。
4. 当时主机约 68 GiB MemAvailable，swap 占用约 12 GiB，但短窗口换入很低、重大缺页为 0；不支持“当前主要在磁盘读写/swap 换页”的假设。此前确实存在内存压力，不应把 swap 占用当成没有压力的证据。
5. 原生和 Python 栈没有停在 GPU 同步或互斥锁等待，而在 SciPy/NumPy 转换；因此 GPU 算力不是当前采样窗口的主要瓶颈。
6. 主机 THP `enabled=madvise`、`defrag=madvise`，PMD hugepage 为 2 MiB。NumPy 默认向大数组请求 hugepage，符合触发条件。

## 最小对照测试

使用 exp006 已锁定环境，经基础 micromamba 环境调用 `uv run --locked --no-sync`。
临时脚本仅构造 `np.arange(8*1024*1024, dtype=np.int64)` 并执行 `.astype(np.int32)`，
两数组合计 96 MiB；计时覆盖分配和转换。3 秒阈值可稳定区分本次默认策略和禁用策略。
只改变测试子进程的 `prctl(PR_SET_THP_DISABLE, 1, 0, 0, 0)`，不改变主机或活动训练进程。

| 测试进程策略 | 分配和转换时间 | 阈值结果 |
|---|---:|---|
| 默认，第一次 | 9.5065 秒 | FAIL |
| 默认，第二次 | 9.3330 秒 | FAIL |
| 进程内禁用 THP，第一次 | 0.02868 秒 | PASS |
| 进程内禁用 THP，第二次 | 0.02917 秒 | PASS |
| 进程内禁用 THP，第三次 | 0.02920 秒 | PASS |

默认第二次与禁用第三次的输出 SHA-256 相同：
`c4744935e8653e85eaee99253e7982fbf265d0673bd0303b3b3a11f30feb382f`。
约 320 倍是这个最小操作的实测差异，不是整套评分的预测加速比。
测试没有在完整 run 上实施缓解措施，因此原始端到端停滞尚不能标记为修复。

## 建议恢复方式及限制

以下是最初只针对 THP 的建议。随后发现基线语义偏离官方默认，已不适用原 run 续算；
以文末“基线修正及停止状态”为准。

优先在 exp006 启动进程内、导入 NumPy/CUDA 前禁用 THP；无需修改全机设置。
`PR_SET_THP_DISABLE` 会跨 fork/exec 继承，且不会改变模型公式、数据、精度或评分定义。
NumPy 的 `NUMPY_MADVISE_HUGEPAGE=0` 只覆盖 NumPy 的申请；进程级 prctl 覆盖范围更完整。
这些行为依据 [Linux THP 官方文档](https://docs.kernel.org/admin-guide/mm/transhuge.html#process-thp-controls)。

实施时先结束活动进程，再记录并提交运行时工程修复、相应验证以及新旧代码身份；
使用完整 checkpoint 状态恢复，保留 H1 既有评分、K562 第 1 轮模型及已有可用缓存。
现有 main.py 对源码/环境身份作严格核对，不应直接修改源码后绕过该校验；
若作为同条件恢复，需要显式记录并验证工程修复的兼容性。若改变采样、精度或官方指标，必须新 run。

关闭 THP 能缓解已证实的页整理问题，但 dense→CSR 临时内存开销仍在；
恢复后须重跑原始 K562 bundle 路径，确认新缓存/评分产出，不能只凭最小测试宣称完整评估修复。
更换评分后端、减少靶点/细胞、改小官方库分块数均不属于本次已验证的修复。

附带观察：metrics.json 的 active_context 仍为 H1，是每折开始未立即持久化导致的状态滞后；
日志及调用栈确认实际是 K562。这不改变训练或评分数值，后续工程修复宜一并改善。

临时文件的首次清理命令被安全检查拒绝。2026-09-26 修正时重新核对四个确切文件，
采用不带强制选项的单文件删除，已清理 perf 数据、GDB 辅助文件及测试脚本。

## 基线修正及停止状态

进一步核对发现，实验手写平铺的基线相当于官方旧 `emit="tile"` 方式，
并且 `exclude_target_gene=False` 偏离官方默认。固定版本的官方 baseline.py 明确说明，
counts 数据支持的默认是 `emit="dispersed"`，tile 有已知偏差，仅为历史复现保留。
因此问题不仅是 THP 性能，也影响缩放分数与模型选择，不能沿用原 run 改口径。

用户要求停止后先发 SIGINT；进程在长时间原生数组操作中未及时处理，继而 SIGTERM 终止。
PID 3615234 已退出，本地 metrics 与原 W&B run summary 已记录 stopped 和评分适用性。
原 checkpoint、9 次 H1 评分及缓存保留；K562 无已完成评分，未发生 leaderboard 提交。

正式实现已改用官方 profile 与 dispersed 构造，固定基线 seed 0、排除自身靶基因，
使用分块写出的内存映射 CSR 作为输入。未修改官方包、官方分块大小、指标、数据面板或采样。
新评分协议写入配置与产物，并拒绝旧协议的缓存；严格的源码身份检查继续阻止旧 run 续算。
进程级 THP 策略在基础环境启动器和训练入口应用，写入运行身份。

回归测试先证明旧基线的 64,000/96,000 个元素不同于官方默认，并证明旧启动器未禁用 THP。
修正后 15 项测试通过：官方基线逐元素一致、计数 CSR 数值和空行往返、旧协议拒绝复用、
进程级大页策略继承、CUDA/gpudge 六指标与官方完整锚点、完整训练状态恢复。
测试中的非整数 counts 提示来自官方 dispersed 浮点缩放基线，不是模型预测整数校验被取消。
用户要求保持停止；尚未用修正代码运行全尺寸 K562，不能据小测试宣称完整实验已恢复或给出加速比。

## 重新运行：CSR 指针边界修正

用户随后授权恢复。新 run `20260927-exp006-dispersed-s17` 从头启动，代码 `f81ed08`，
K562 折先执行。进程 `/proc/PID/status` 确认 `THP_enabled=0`；第 1 轮训练完成，
RMSE 0.13802862115792083。完整 reference 写入时，累计 nnz 超过 int32 上限触发
`OverflowError: Python integer 2148241612 out of bounds for int32`，进程已正常报错退出。
尚未完成 reference，也未进入官方基线或产生任何新评分。

原因：目标 indptr 虽为 int64，右侧 `nnz + csr.indptr` 仍先按块内 int32 指针计算。
修复仅把该加法显式指定为 int64，既避免越界前的回绕，也避免越界后的异常。
通过真实 streaming writer 的合成大块元数据测试复现，不实际分配数十亿元素；
修正后 16 项测试通过，包含 CUDA/gpudge 完整评分与 checkpoint 恢复。

这是不改变训练与评分定义的存储工程修复，按协议沿用同一 run。恢复前明确校验：
生产源码仅这一行改变；配置、其他源码树、锁文件、运行时与输入身份不变；
第 1 轮模型摘要匹配，完整 `latest.pkl` 保留，没有已完成 reference/bundle/评分可被混用。
将完整旧身份、新身份、修复理由、checkpoint 摘要和测试结果追加到该 run 的配置迁移记录，
再更新当前代码身份。原失败日志保留，后续 W&B 仍使用同一个 ID。
未完成 CSR 文件由原写入入口重新生成；不复用损坏指针，也不放宽通用恢复身份检查。

恢复后完整 K562 reference 成功写出，nnz 为 6,012,089,092，shape 为 1,915,223 × 8,135。
实际 CSR 约 56.0 GiB，原两份完整 reference 加 12 GiB 门槛要求 124.0 GiB，超过整机 119 GiB，
触发永久资源等待。已 SIGINT 正常退出后修正：两份不相交半矩阵合计一份 reference，
额外半份预算覆盖切片/复制瞬时空间；门槛改为 1.5 份加 12 GiB，超过整机容量则明确报错。

同时公开 `build_baseline_prediction` 会一次性分配多份全尺寸临时矩阵，无法靠文件映射 reference 解决。
现在调用固定官方包内部 `_emission_scale` / `_emit_scaled_resample` 执行相同运算，
完整 NTC 池、按排序靶点顺序、一个连续 RNG，按 256 行写文件映射 CSR。
不复制评分公式，不改官方包。实验 layout 固定为 NTC 在前、其后排序靶点连续分组，入口严格核对。
多个种子和含奇数块长的逐元素测试对照公开构造器，连同零元素消除、不可达 profile 质量和全部
baseline_emission 诊断字段一起比较；实际官方 CUDA 评分亦通过。该存储执行变化不改评分协议。
继续沿用同一 run，完整记录第二次代码身份迁移；reference 文件、训练配置和 checkpoint 保持原样。

## 完整 K562 恢复验证（记录时仍在运行）

代码 `3a702ef`，同一 run，进程 PID 3803515，于 2026-09-27 01:36:32 UTC 恢复。
22 项测试通过。01:36:35 开始官方 profile，01:39:17 完成全 9,319 个靶点的 dispersed
基线写入和校验并调用 `build_real_bundle`，约 2 分 42 秒。官方包和数据面板未修改。

只读调用栈先观察到 CUDA/torch、官方 jackknife 校正，随后在约 01:56 UTC 确认：
`build_real_bundle → _anchor_leg → compute_replicate_anchor → _score_one_split → _disjoint_halves`。
因此旧 run 未通过的完整 baseline leg 已在约 17 分钟内通过，当前是后续的五次拆半锚点，
不能再把这一状态描述为仍卡在基线均值构建。real-side 官方缓存已落盘约 3.2 GiB。
新旧基线协议不同，这些时间不能当作同口径评分的精确加速倍数。

进程 `THP_enabled=0` 持续生效。计算中确有内存压力，拆半阶段进程 swap 一度约 13 GiB，
后续有所回落；02:00 UTC 仍在运行，未产生首个 K562 checkpoint Overall。
原 THP 卡点、CSR int32 溢出和不可能满足的资源等待已分别验证越过；
完整五次锚点、首个模型评分和整个训练尚未完成，不能宣称端到端成功。
临时 GDB helper 仅用于读取栈，不改训练进程代码、依赖或主机设置，检查后删除。
