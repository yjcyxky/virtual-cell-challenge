#!/usr/bin/env python
"""Render an assessment JSON bundle without inventing values or conclusions."""
import argparse
import html
import json
from pathlib import Path


def render(report):
    data = json.dumps(report, ensure_ascii=False, allow_nan=False).replace("<", "\\u003c")
    title = html.escape(report.get("title", "Dataset assessment"))
    return '''<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>''' + title + '''</title><style>
body{font:16px system-ui;margin:2rem;line-height:1.5}table{border-collapse:collapse;min-width:75%}td,th{border:1px solid #ccc;padding:.5rem;text-align:left;max-width:36rem;overflow-wrap:anywhere}section{overflow:auto}pre{white-space:pre-wrap;overflow-wrap:anywhere}.unknown{color:#815900}input{padding:.7rem;width:65%}a{color:#1464a5}.notice{background:#fff2d0;padding:1rem}</style>
<h1>''' + title + '''</h1><p id="identity"></p><div class="notice" id="limitations"></div>
<h2>结果与输入</h2><ul id="downloads"></ul><input id="filter" aria-label="筛选诊断表" placeholder="按 split、target、状态或证据筛选">
<main id="tables"></main><h2>方法、证据与版本</h2><pre id="metadata"></pre>
<script type="application/json" id="payload">''' + data + '''</script><script>
const r=JSON.parse(document.getElementById('payload').textContent);
const esc=x=>String(x).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
function fmt(x){if(x===null||x===undefined)return '<span class="unknown">unknown / not estimable</span>';return esc(typeof x==='object'?JSON.stringify(x):x);}
document.getElementById('identity').textContent=`${r.bundle_id} | ${r.status} | ${r.completed_at}`;
document.getElementById('limitations').textContent=(r.limitations||[]).join(' ');
document.getElementById('downloads').innerHTML=(r.artifacts||[]).map(a=>`<li><a href="${esc(a.file)}">${esc(a.file)}</a> — ${esc(a.description||'')} (${esc(a.sha256||'see SHA256SUMS')})</li>`).join('');
document.getElementById('metadata').textContent=JSON.stringify({...r,tables:undefined},null,2);
function draw(){const q=document.getElementById('filter').value.toLowerCase();document.getElementById('tables').innerHTML=(r.tables||[]).map(t=>{
const rows=t.rows.filter(row=>JSON.stringify(row).toLowerCase().includes(q));const cols=t.columns||Object.keys(t.rows[0]||{});
return `<section><h2>${esc(t.title)}</h2><p>${esc(t.description||'')} (${rows.length} matching rows)</p><table><thead><tr>${cols.map(c=>`<th>${esc(c)}</th>`).join('')}</tr></thead><tbody>${rows.map(row=>`<tr>${cols.map(c=>`<td>${fmt(row[c])}</td>`).join('')}</tr>`).join('')}</tbody></table></section>`}).join('');}
document.getElementById('filter').oninput=draw;draw();</script></html>'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with args.output.open("x") as fh:
        fh.write(render(json.loads(args.report.read_text())))


if __name__ == "__main__":
    main()
