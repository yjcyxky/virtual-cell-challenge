# exp004 结果与状态

2026-09-25：用户已授权完整实现并启动训练与消融。计划及运行状态在 [Issue #35](https://github.com/yjcyxky/virtual-cell-challenge/issues/35) 维护。当前尚无 exp004 完整五折结果，不宣称性能改善或机制得到验证。

矩阵：9 arms × 3 training seeds × 5 held-out datasets，共 27 runs / 135 fits。首 run 为 `20260925-exp004-conditional-s17`，见 [W&B](https://wandb.ai/yjcyxky/virtual-cell-challenge/runs/20260925-exp004-conditional-s17)。完整训练后由各 run 的 metrics.json、每折官方六分项、checkpoint/预测摘要和版本化 Artifacts 汇总比较；队列最终汇总保存在末 run 的 matrix-comparison.json。

exp003 原 run 继续执行，历史分数保持原口径。exp003 H1 c4 本地综合 −0.0676358 与其官方提交 −0.0359002 使用不同数据、基因支持及 real bundle，不作为可直接换算的尺度。新实验的主要基线是本矩阵 reference arm，避免把代码环境/划分变化混入架构归因。三个辅助比较方向分别是共享表示、条件关系、幅度/区分目标；不同 loss 的绝对数值不直接比较。

启动交付须确认代码已提交、锁定独立环境、CPU/CUDA 行为验证及恢复验证通过，并观察到正式 optimizer 更新。启动完成只表示完整流程已交给后台执行；正常早停、五折评估与全部消融尚须等待。异常应在 Issue 中记录，不用 smoke test 代替训练结果。

实现验证：新架构/目标的 17 项行为测试和共享流程的 54 项回归测试合计 **71 passed**，包括 CPU/CUDA 完整状态精确恢复、零响应辅助梯度、共享读出更新、非可分离算子、NTC/缺测行为、监测 RNG 隔离、先验缓存完整性、官方六项评分与预测恢复流程。官方合成测试的常量浮点 baseline 触发两条上游 count-filter 警告；测试输入含此基线，实际模型生成输出另验证为非负整数。独立 uv 环境由固定 virtual-cell Python 创建并按锁文件同步；没有复用 exp003 的虚拟环境。

首次正式入口在优化前复检失败：清理旧常量导入后，三个历史测试仍从 training 模块读取 LOSS_TERMS；此前测试进程已导入旧版本，未覆盖这次导入清理。修复为从当前 objective.loss_terms 获取项列表，模型/训练配置不变。修复后重跑相关验证，并在原 run 保存代码版本修订和失败日志；不新建阶段 run，不拼接训练历史。矩阵支持显式 `--repair-validation-reason` 向未优化 run 传递修复原因，已有优化状态或配置变化仍由原入口拒绝。

## 第 11 周期 checkpoint 的官方评估（2026-09-26）

用户明确授权本次生成和官网提交。原 run `20260925-exp004-conditional-s17`，H1 留出 `holdout-H1-best.pt`，cycle 11、step 134261，checkpoint SHA256 `0f317accc7611528316c1cefc8d4fac3ebe2de9ccf544be29fcf2d7015b297ad`。训练提交 `be92538b80b67745d92878002966afa7b2157875`；导出提交 `40f68a1ef910d2351d4ac42a7684bc53c5073e67`。在隔离 checkout 中使用原 Experiment 已锁定环境，未修改活动训练代码、数据、环境或 checkpoint，没有新增训练 run。

导出沿用 checkpoint 的共享读出、条件先验、PCA 和 NB 生成器。官方三个背景的全部 NTC 分别合并，不把 ntc_id 当技术 batch。CPU、seed 101、每批 128 个细胞；3×300×400=360,000 个细胞，18,533 基因。独立全文件审计及官方 vcc prep 通过，未裁剪或重抽超限细胞。CPU 与历史 CUDA 导出不保证相同随机样本。52 项测试通过，1 项 CUDA 导出一致性测试因活动任务占用下无法创建 CUDA 上下文而跳过；CPU 完整恢复与整数计数测试通过。

官方 entry `payUWYmujElaWQXgnrgl`，模型名 `Exp004-Conditional-c11-20260925-s17`，状态 `published`。VCC SHA256 `2f9ef8a9e2594ee9a468072c4700dfbfbf4bdb760fc0b72c68b599259100e621`，大小 3391856640 bytes。文件、校验、回执和逐项比较均保留于原 run 的 `predictions/leaderboard-holdout-H1-cycle-0011-seed-101/`。

|评分项|exp003 cycle 4 历史官方|exp004 conditional cycle 11 官方|
|---|---:|---:|
|PDS|0.002144845|-0.001492573|
|表达 MSE|0.000000000|0.000000000|
|LFC 幅度|-0.156850828|0.005569875|
|方向 fidelity|-0.024458006|-0.157535451|
|方向 reach|-0.033825432|-0.004022242|
|DE Jaccard|-0.002411636|-0.029302188|
|六项均分|-0.035900176|-0.031130430|

partition/panel/anchor 与历史 exp003 回执一致：True。本地 H1 三生成种子均分 -0.084655892，与官网 A/B/C 的数据及 real bundle 不同，不能直接换算；不同架构、所选周期及生成设备的单次比较不构成架构因果归因。评分零点不是原样 NTC 输出。

本次综合分比历史 exp003 提高 0.004769746，主要表现为 LFC 幅度项改善、方向 fidelity 下降，PDS 和 DE Jaccard 也下降，不能称为各项全面改善。输出中仍有 7,632 个 readout 没有训练表达测量监督，保留共享映射的原始预测。完整计数审计中每细胞总计数为 2,617–37,741，非零计数数目为 2,097,033,529，均符合官方上限。

官方 validation 反馈后续用于模型选择时明确作为开发反馈。此次提交完成不等于五折或 27-run 矩阵完成。[原 W&B run](https://wandb.ai/yjcyxky/virtual-cell-challenge/runs/20260925-exp004-conditional-s17) 通过 Public API 附加官网回执与导出身份，不启动第二个 history writer；关键结果的版本化 Artifact 待原训练写入结束后归档。
