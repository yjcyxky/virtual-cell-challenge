# 响应迁移实验记录

## 原生计数迁移：2026-09-26

状态：实现完成，正式训练与五背景评价尚未完成。计划见 [PLAN](PLAN.md)。当前代码已通过 29 项行为检查，包括训练/留出隔离、缺测标签、NTC 恒等生成、均值与深度分布校准、模型独立加载、官方六指标及确定性恢复。

正式 run 为 `20260926-native-count-transfer-s20260926`（[W&B](https://wandb.ai/yjcyxky/virtual-cell-challenge/runs/20260926-native-count-transfer-s20260926)）。输入约 153 GB 内容哈希核验通过。准备阶段发现全轴逐细胞扩展造成明显开销，改为原生轴聚合后映射统计向量；真实 K562 的 1,024 行微基准中四项统计逐元素一致，暖缓存耗时由约 4.14 秒降至 0.009 秒。另增混合面板/缺测的等价性检查与准备修复约束检查；当前 30 项 CPU 检查通过。

GPU 检查曾通过，也曾在初始化时仅分配 6,144 字节即报 CUDA OOM；当时系统仍有几十 GB 空闲，但 `/proc/buddyinfo` 中 2 MB 及以上普通内存块均为零。此证据支持物理内存碎片/并发分配问题，尚不能断言唯一根因。NVIDIA 对此类碎片现象的说明见 [官方 FAQ](https://docs.nvidia.com/cudf-spark/latest/faq.html#why-am-i-getting-maximum-pool-size-exceeded-when-allocating-pinned-memory)。只对本进程 256 MiB 私有内存的合并探针返回 ENOMEM，内存已释放；未改变系统配置、驱动或其他活动 run。GPU 评分标准保持固定，资源失败不计为完成。

本轮把共享靶点响应、NTC 相似性条件残差、原生测量轴监督与经验计数模板放入同一可恢复 run。配置为 `configs/native-count-transfer.json`；计划五背景留出、每折四组生成对照与三个种子，最后用含 H1 的全部五背景重拟合并准备 A/B/C 官方预测。正式运行结果、W&B 与局限在有效训练评价完成后补充。

本轮局部评价使用固定最多 300 靶点面板和新的计数聚合响应定义。既有 exp002 run 的历史指标不改写；历史官方结果和跨模型口径差异见 [leaderboard 调查](../../docs/leaderboard-score-interpretation.md)。
