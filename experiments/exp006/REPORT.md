# exp006 结果与停止说明

当前状态：**已按用户要求停止，修正后尚未重新启动**。

历史 run：`20260926-exp006-loco-residual-s17`。
W&B：https://wandb.ai/yjcyxky/virtual-cell-challenge/runs/20260926-exp006-loco-residual-s17

## 停止原因与结果适用性

旧 run 使用自建平铺均值基线（相当于 tile）且未排除自身靶基因，偏离固定版本
cell-eval2 对 counts 数据支持的 dispersed 默认。K562 评分还触发 THP 内存整理瓶颈。
原始分数、checkpoint 与缓存保留，历史评估口径不改写；这些分数不能用于修正后的模型选择。

停止时间（UTC）：2026-09-27T01:03:20.307275+00:00。先请求 SIGINT，随后 SIGTERM 终止长时间原生计算；进程已退出。

## 已执行的历史工作

H1 留出折训练完成 512 轮，7 个 checkpoint 和两个对照已评分。
K562 留出折训练完成第 1 轮，基线构建未完成，未产生 K562 分数。
全五背景重训未开始，leaderboard 提交为 0 次。

以下均为 **旧 tile 基线口径下的 H1 开发验证 Overall**，不是修正协议或 leaderboard 分数。

| 模型/对照 | 轮数 | Overall |
|---|---:|---:|
| model | 1 | -0.377965477 |
| model | 16 | -0.368523585 |
| model | 64 | -0.270955592 |
| model | 128 | -0.248111726 |
| model | 256 | -0.229731753 |
| model | 384 | -0.224312820 |
| model | 512 | -0.221236761 |
| shared | 0 | 0.019260375 |
| zero | 0 | -0.367115740 |

## 已完成的修正

- 官方 `generic_response_profile(..., exclude_target_gene=True)` 与
  `build_baseline_prediction(..., emit="dispersed", seed=0)` 生成评分基线。
- reference 和本地预测以分块、数值无损的文件映射 CSR 提交给官方评分函数，删除旧平铺实现。
- 进程内禁用透明大页并记录实际策略，主机设置与锁定依赖保持不变。
- 新协议写入配置和评分产物；旧评分、reference、bundle 禁止跨协议复用。
- 每折开始即保存 active_context，消除旧状态文件滞后。

验证：15 项测试通过，包括先失败后通过的官方基线一致性与 THP 启动测试，以及真实 CUDA/gpudge
评分、锚点和缓存重放。一次浮点基线 counts 提示来自官方 dispersed 构造。
完整 K562 数据规模的性能尚未重测，正式训练保持停止。

下一次启动须创建新 run，重新完成五折选模和全五背景重训，不能恢复旧 run 混入新评分口径。
完整训练与评估尚未完成，不把此次修复测试视作实验完成。

详细原因与验证见 `../../docs/exp006-k562-evaluation-stall.md`；研究方案见 `PLAN.md`。
