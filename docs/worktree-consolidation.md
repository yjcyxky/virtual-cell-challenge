# 实验 worktree 合并记录（2026-09-27）

全部 8 个实验 worktree 已合并到本地 `master`；对应本地分支与 worktree 已删除。未推送或删除远端分支。不在 worktree 中的其他分支不属于本次清理范围。

## 合并来源

| 原 worktree | 原分支 | 纳入主分支的末次提交 |
| --- | --- | --- |
| exp003-prior-diffusion | exp003-prior-diffusion | `614ff6d12ecf` |
| exp004 | exp004-shared-response | `be92538b80b6` |
| exp004-export-c11 | exp004-export-c11 | `ccd7bc714c51` |
| exp005 | exp005-residual-xgb | `77818f37445a` |
| exp006 | exp006-loco-xgb | `2a2a7409c292` |
| exp007 | exp007-h1-log2fc | `b87445584e8a` |
| exp00701 | exp00701-response-signatures | `27d0abd71c65` |
| exp008 | exp008 | `0d9c85b62bc3` |

`exp005` 原有未提交的失败状态报告先保存为 `77818f37445a`，再合并。唯一的合并冲突位于 exp003 训练模块的导入：同时保留先验扩散的 `representation_config` 和 exp004 按 objective 实例读取损失分量的实现。exp008 保留其方案文档状态，本次没有启动或宣称完成任何训练。

## 本地产物与路径

原 `/home/jy001/Downloads/virtual-cell-challenge-worktrees/<worktree>/experiments/<experiment>/` 中的 `outputs/` 和 `.venv/` 已通过同一文件系统内的重命名迁入 `/home/jy001/Downloads/virtual-cell-challenge/experiments/<experiment>/`。实际迁移的实验为 exp004-shared-response、exp005、exp006、exp007、exp00701；其余 worktree 原先仅引用主目录中的共享数据、环境或产物。

迁移清单共 1,731 个文件或链接条目：1,722 个保持原 inode、大小、修改时间及链接内容，另 9 个 exp00701 跨实验缓存软链接改为指向迁移后同一产物的相对路径。历史 run 的配置、日志、模型、预测、指标和 W&B 文件内容保持不变；共享数据与模型目录未改动。

当前 exp00701 的 `configs/source-reference.json`、`configs/exp004-reference.json` 只更新位置字符串，保留所有输入大小、SHA-256 和模型/评估参数。119 个固定输入均存在且大小符合记录，其中 88 个小于 16 MiB 的输入另行核对 SHA-256。大型已迁移文件通过 inode、大小和修改时间核对，未全量重算哈希。

历史文档和 run 元数据中的绝对路径保留作为原始执行记录。查找这些路径时，将 `virtual-cell-challenge-worktrees/<worktree>/` 前缀映射为 `virtual-cell-challenge/`。当前引用文件的摘要因路径变化而改变；历史恢复或复现应使用记录的 Git/锁文件版本并恢复相应路径映射，不得重写历史 run 的配置或引用摘要来绕过校验。

五套迁移环境均使用原有 micromamba `virtual-cell` 基础环境执行 `uv sync --locked --offline --reinstall`，显式绑定其 Python 并禁用 Python 下载；依赖锁文件和版本未变。环境入口与激活脚本已适配新位置，并重新记录环境指纹。基础环境未重建。

## 验证

| 实验 | 通过测试数 |
| --- | ---: |
| exp003-context-module-cvae | 67 |
| exp004-shared-response | 19 |
| exp005 | 10 |
| exp006 | 22 |
| exp007 | 25 |
| exp00701 | 35 |
| 合计 | 178 |

exp003 首次测试未显式设置训练入口的线程参数，一项官方评分恢复测试因约 `2e-16` 的浮点差异失败；按入口设置线程参数后完整 67 项通过，未修改测试断言或评分逻辑。其余实验测试均通过。六个实验的 `./reproduce.sh --help` 入口、合并提交可达性和 `git diff --check` 均通过。

删除前逐一确认 worktree 无未提交改动，唯一剩余的 Git 忽略内容为可重建的 Python/pytest 缓存及共享目录软链接。主目录原有未跟踪的 `AGENTS.md` 和 `docs/ideas/variations.md` 保持原状；原有未跟踪的目标覆盖说明与合入版本逐字节一致，现由 Git 跟踪。一次性迁移代码未进入仓库。
