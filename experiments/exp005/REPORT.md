# exp005 结果

Run `20260926-exp005-nested-residual-s17`，状态：failed。

W&B：https://wandb.ai/yjcyxky/virtual-cell-challenge/runs/20260926-exp005-nested-residual-s17

预处理工程修复（未改变配置，原提交 `d072065f37cb2fabcfb567a8ebb6727a9a3ba7bb`）：No context cache or model had completed. Reuse HDF5, validation and grouped-count buffers and write counts sequentially to avoid measured unified-memory minor-page-fault stalls; retain identical counts, labels, sampling and training configuration. Full-state resume and count-equivalence tests passed. Wait for CUDA resources without changing the official backend.


阻塞/失败：KeyboardInterrupt:

限制：独立细胞背景仅五个且曾用于项目开发；NTC bag 不构成新背景。均值 residual 加 NTC 模板无法完整恢复扰动导致的新状态和分布。原始覆盖不等于合格监督覆盖；未测基因不作零标签。
