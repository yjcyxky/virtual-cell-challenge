# 初始线性响应路线

`init-linear-s01`（S2-H1 / context-seen / seed=1）已完成解析式岭回归拟合；官方评估及 E-BASE 尚未完成。冻结训练代码为 `26a03062011fe902ee4a2c41cbdc287858ce3561`。

四个训练背景保留 17,040 个背景×靶点任务、2,541,598 个扰动细胞；H1 扰动参考不进入训练。训练 masked MSE 为 0.00125619，不能当作官方评分或泛化效果证据。完整解析拟合状态保存于 `outputs/init-linear-s01/checkpoints/ridge.npz`。

2026-09-28 会话切换期间终端沙箱初始化失败；恢复命令能力后确认原执行进程已退出、最终 metrics 和预测清单尚未生成。已按实际观察登记 interrupted，保留已有预测及恢复记录于该 run 的 cache/，不作为方法阴性证据。恢复沿用同一 run、原配置和完整 checkpoint，只继续生成/评估。

[W&B run](https://wandb.ai/yjcyxky/virtual-cell-challenge/runs/init-linear-s01)：group=init-linear、id=init-linear-s01。后续须完成两面板四臂的官方六指标、配对差异与限制分析、Artifacts 同步及证据决策，再交付结果。
