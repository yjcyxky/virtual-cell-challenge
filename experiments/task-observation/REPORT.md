# 五背景观测与计数校准报告

## 首次执行故障

`task-observation-s01` 在 H1 观测扫描完成后因 pandas Series 直接索引 SciPy sparse 的接口错误失败，未产生生成预测或评估结果，不能作为方法证据。原绑定和代码保留在 `5a1132e`；替代 run `task-observation-s02` 使用 NumPy 布尔数组索引，完整重跑相同科学条件，不复用失败 run 的中间统计。对应独立比较 `C-TASK-DATA-GENERATION-R2`。最终结果待完整批次完成。
