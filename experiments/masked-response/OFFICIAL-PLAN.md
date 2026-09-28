# C-SHARED-QC-OFFICIAL：已完成 shared + 低深度 QC 的官方评分

按用户要求，使用 masked-response-qc-s01 的原 response.npz，沿用原 run/W&B，不新增拟合或修改原本地结论。checkpoint、原配置、代码、环境、完整官方输入按哈希校验；新增官方阶段配置和代码提交后执行。

方法是 shared，候选接口 arm 名称 linear 只是历史评分接口。原模型 predict 对已见靶点返回保存的 effects=shared，对未见靶点及未测得输出基因返回零 Δ。保留原固定 NTC log-shift、library-preserving stochastic-round 生成器，seed=1+官方靶点索引；A/B/C 各使用完整18,400个NTC，不在官方输入追加QC。完整18,533基因×300靶点×3背景，每组合400细胞，共360,000细胞。

复用 init-linear 已冻结的输入验证、流式CSR生成、官方prep和提交/恢复函数；通过 masked-response/runtime.py 解析本路线原模型。小矩阵回放测试检查解析到正确 predictor、计数逐值一致和未知靶点零Δ。正式文件全量审计形状、顺序、非负整数、每组合细胞数、计数及非零上限；提交后另抽样回放实际输出。

比较为 protocol_audit：预定本地与官方Overall差值绝对值≤0.02仅作为接近线索，不能证明协议等价。报告H1主280/重叠25靶点与官方300靶点的六项raw、normalized和Overall；比较既有官方shared entry vWd1Z4K9tJ2tjfns2gSj仅作同panel/anchor下的描述性数据选择整包差异，不把其当匹配随机QC消融。无重复种子，官方反馈均为开发证据。

入口 `cd experiments/masked-response && ./submit.sh --submit`；不带--submit仅生成/校验。恢复时固定VCC哈希并使用同一entry，不重复提交。结束须published、六项有效分数及Overall、panel/anchor/完整回执、原W&B版本化VCC Artifact、REPORT和Ledger关闭；评分失败保留待办并恢复，不作方法阴性。
