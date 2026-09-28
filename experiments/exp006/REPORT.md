# exp006 结果

Run `20260927-exp006-dispersed-s17`，状态：cross_validation。

W&B：https://wandb.ai/yjcyxky/virtual-cell-challenge/runs/20260927-exp006-dispersed-s17

协议：五折按背景留一（每折四训一验），按五背景等权均值选一个统一轮数，全五背景从头重训，再提交 leaderboard。

评分基线：官方 dispersed 构造、排除自身靶基因；协议 cell-eval2-dispersed-exclude-target-v1。

本地分数用于模型选择，属于开发验证结果，不是独立测试成绩；本地原生面板分数不等同 A/B/C 榜单。

已完成 checkpoint 背景评分：0/35。

W&B 日志同步：online；Artifact 同步：not yet uploaded。

与 exp005 的差异：模型、数据与固定采样沿用，四背景留一选模替代嵌套三背景训练；评分基线改为官方支持的 dispersed/排除靶基因。旧 tile 基线分数不能用于本协议选模或作为同口径成绩比较。未复用旧权重或 run 缓存。

限制：仅五个独立背景且曾用于项目开发；NTC bag 不是新背景。均值 residual 加 NTC 模板不能完整恢复扰动后的新状态与分布。未测基因不作零标签。

其他历史 runs：

- `20260926-exp006-loco-residual-s17`：stopped，来源结果 `outputs/20260926-exp006-loco-residual-s17/metrics.json`，W&B https://wandb.ai/yjcyxky/virtual-cell-challenge/runs/20260926-exp006-loco-residual-s17。评估有效性：historical tiled baseline only; not eligible for corrected model selection。
