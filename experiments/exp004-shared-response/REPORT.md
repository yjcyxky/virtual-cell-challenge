# exp004 结果与状态

2026-09-25：用户已授权完整实现并启动训练与消融。计划及运行状态在 [Issue #35](https://github.com/yjcyxky/virtual-cell-challenge/issues/35) 维护。当前尚无 exp004 完整五折结果，不宣称性能改善或机制得到验证。

矩阵：9 arms × 3 training seeds × 5 held-out datasets，共 27 runs / 135 fits。首 run 为 `20260925-exp004-conditional-s17`，见 [W&B](https://wandb.ai/yjcyxky/virtual-cell-challenge/runs/20260925-exp004-conditional-s17)。完整训练后由各 run 的 metrics.json、每折官方六分项、checkpoint/预测摘要和版本化 Artifacts 汇总比较；队列最终汇总保存在末 run 的 matrix-comparison.json。

exp003 原 run 继续执行，历史分数保持原口径。exp003 H1 c4 本地综合 −0.0676358 与其官方提交 −0.0359002 使用不同数据、基因支持及 real bundle，不作为可直接换算的尺度。新实验的主要基线是本矩阵 reference arm，避免把代码环境/划分变化混入架构归因。三个辅助比较方向分别是共享表示、条件关系、幅度/区分目标；不同 loss 的绝对数值不直接比较。

启动交付须确认代码已提交、锁定独立环境、CPU/CUDA 行为验证及恢复验证通过，并观察到正式 optimizer 更新。启动完成只表示完整流程已交给后台执行；正常早停、五折评估与全部消融尚须等待。异常应在 Issue 中记录，不用 smoke test 代替训练结果。

实现验证：新架构/目标的 17 项行为测试和共享流程的 54 项回归测试合计 **71 passed**，包括 CPU/CUDA 完整状态精确恢复、零响应辅助梯度、共享读出更新、非可分离算子、NTC/缺测行为、监测 RNG 隔离、先验缓存完整性、官方六项评分与预测恢复流程。官方合成测试的常量浮点 baseline 触发两条上游 count-filter 警告；测试输入含此基线，实际模型生成输出另验证为非负整数。独立 uv 环境由固定 virtual-cell Python 创建并按锁文件同步；没有复用 exp003 的虚拟环境。
