# exp006 结果

Run `20260926-exp006-loco-residual-s17`，状态：preparing。

W&B：https://wandb.ai/yjcyxky/virtual-cell-challenge/runs/20260926-exp006-loco-residual-s17

协议：五折按背景留一（每折四训一验），按五背景等权均值选一个统一轮数，全五背景从头重训，再提交 leaderboard。

本地分数用于模型选择，属于开发验证结果，不是独立测试成绩；本地原生面板分数不等同 A/B/C 榜单。

已完成 checkpoint 背景评分：0/35。

W&B 日志同步：online；Artifact 同步：not yet uploaded。

与 exp005 的差异：模型、数据、固定采样和官方指标相同；由嵌套三背景内层训练改成四背景留一选模，估计目标不同，不把两者验证分数当作同口径独立测试比较。未复用 exp005 权重或 run 缓存。

限制：仅五个独立背景且曾用于项目开发；NTC bag 不是新背景。均值 residual 加 NTC 模板不能完整恢复扰动后的新状态与分布。未测基因不作零标签。
