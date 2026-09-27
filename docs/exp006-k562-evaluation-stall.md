# exp006 K562 官方评估长时间无输出：诊断记录

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
