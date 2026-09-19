# 数据档案（dossier）

每个数据集一页可交互的 HTML：出处（生辰八字）+ 数据特征 + **每项特征对 VCC 2026 流程的影响**。

```bash
.venv/bin/python scripts/dossier/profile_arc_vcc2025_h1.py   # 扫 raw/ → data/profiles/<id>.json
.venv/bin/python scripts/dossier/render.py arc_vcc2025_h1    # + templates/<id>.html → docs/datasets/<id>.html
```

页面里的每个数字都来自 profiler 的单次流式扫描，没有手抄常量：改了判据重跑两条命令即可。
`raw/` 全程只读。

## 加一个数据集

1. 抄 `profile_arc_vcc2025_h1.py` 写一个 profiler，输出 `data/profiles/<id>.json`。
   通用部分（形状、深度分布、批次、NTC 噪声地板）可直接复用，数据集特有的度量自己加。
2. 抄 `templates/arc_vcc2025_h1.html` 写模板。它是 artifact 形态的页面正文
   （`<title>` + `<style>` + 标签 + `<script>`，不含 doctype/head/body），
   `{{name}}` 占位符由 `render.py` 的 `ph` 字典填，`/*__DATA__*/` 换成注入的 JSON。
3. 在 `render.py` 的 `HEADLINE` 里写这一页的论点，缺占位符会直接报错而不是渲染出空洞。

`render.py` 同时输出 `docs/datasets/.<id>.body.html`——发布成 artifact 用的正文，
`docs/datasets/<id>.html` 是可直接双击打开的单文件版本。

## 判断写在哪

和 `data/sources.json` 一个规矩：事实（细胞数、sha256、分位数）由 profiler 算，
判断（"敲低失败"判定线 0.4、"被推动基因"阈值 0.25、评测深度 20k）是脚本顶部的常量，
改一个字符即可重算全页。
