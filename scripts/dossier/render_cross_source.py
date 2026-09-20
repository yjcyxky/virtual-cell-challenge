"""Standalone, offline viewers for complete coverage and source relation graphs."""
import base64
import gzip
import json
import numpy as np


STYLE = '<style>body{font:16px system-ui;margin:2rem;line-height:1.5;color:#16324a}input,select,button{font:inherit;padding:.5rem;margin:.3rem}select{max-width:90vw}table{border-collapse:collapse;width:100%}td,th{padding:.45rem;border:1px solid #ccd7df;text-align:left;overflow-wrap:anywhere}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#edf3f7;padding:1rem}.notice{background:#fff3ce;padding:1rem}a{color:#12619a}svg{width:100%;min-width:850px}section{overflow:auto}</style>'
HELPERS = """const esc=x=>String(x??'unknown').replace(/[&<>\"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','\"':'&quot;',\"'\":'&#39;'}[c]));
const byid=x=>document.getElementById(x);
"""


def coverage_page(panels, official_genes, statuses, codes, targets, target_counts):
    # A compact byte matrix includes every panel × official-gene status. The
    # browser only paginates the view, never samples the underlying universe.
    payload={'panels':panels,'genes':list(official_genes),'statuses':list(statuses),'targets':list(targets),
             'target_counts':target_counts.tolist(),
             'encoded':base64.b64encode(gzip.compress(np.asarray(codes,dtype=np.uint8).tobytes(),mtime=0)).decode()}
    data=json.dumps(payload,ensure_ascii=False,allow_nan=False).replace('<','\\u003c')
    return '<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>完整跨来源基因覆盖</title>'+STYLE+'''
<h1>完整跨来源基因覆盖</h1><p><a href="report.html">返回总报告</a> · <a href="relationships.html">研究／样本／文库关系图</a></p>
<p class="notice">官方分母为 18,533 个原始基因标签、300 个目标。缺测、全零、映射不确定及物种／模态不适用分开；全零仅指所列 QC 范围。已完成构件数不是独立生物重复数。所有推断和响应均已被分析接触。</p>
<label>来源筛选 <input id="source-filter" placeholder="scBase、Jiang、K562…"></label><br><label>选择面板 <select id="panel"></select></label>
<p id="scope">正在解码全部覆盖结果…</p><pre id="panel-meta"></pre><p id="files"></p>
<label>基因名称 <input id="gene-filter" placeholder="搜索覆盖全部基因"></label><label>状态 <select id="status"><option value="">全部状态</option></select></label>
<label><input type="checkbox" id="only-targets">仅官方 300 个靶基因</label>
<p id="count"></p><button id="prev">上一页</button><button id="next">下一页</button>
<section><table><thead><tr><th>官方基因标签</th><th>测量覆盖状态</th><th>已完成 CRISPRi 构件数</th><th>排除已证实合集副本后构件数</th><th>已完成 DE 构件数</th></tr></thead><tbody id="rows"></tbody></table></section>
<script type="application/json" id="payload">'''+data+'''</script><script>'''+HELPERS+'''
const r=JSON.parse(byid('payload').textContent), targetIndex=new Map(r.targets.map((g,i)=>[g,i]));let matrix=null,page=0;
function options(){const query=byid('source-filter').value.toLowerCase(),old=byid('panel').value;byid('panel').innerHTML=r.panels.map((p,i)=>[p,i]).filter(([p,i])=>JSON.stringify(p).toLowerCase().includes(query)).map(([p,i])=>`<option value="${i}">${esc(p.panel_id)}</option>`).join('');if([...byid('panel').options].some(o=>o.value===old))byid('panel').value=old;page=0;draw();}
function draw(){if(!matrix)return;const pi=Number(byid('panel').value),p=r.panels[pi];if(!byid('panel').options.length){byid('rows').innerHTML='';byid('count').textContent='0 matching panels';return;}
byid('scope').textContent=`全部 ${r.panels.length} 个面板 × ${r.genes.length} 个官方基因；当前 ${p.panel_id}`;
byid('panel-meta').textContent=JSON.stringify(p,null,2);byid('files').innerHTML=`<a href="coverage/${esc(p.gene_coverage_file)}">本面板完整机器可读覆盖</a> · <a href="coverage/${esc(p.native_mapping_file)}">原生特征映射（含歧义）</a> · <a href="coverage/all-panel-target-coverage.parquet">全部靶基因任务覆盖</a>`;
const q=byid('gene-filter').value.toLowerCase(),s=byid('status').value,only=byid('only-targets').checked;
const selected=r.genes.map((g,i)=>[g,i,matrix[pi*r.genes.length+i]]).filter(([g,i,c])=>g.toLowerCase().includes(q)&&(!s||r.statuses[c]===s)&&(!only||targetIndex.has(g)));
page=Math.min(page,Math.max(0,Math.ceil(selected.length/100)-1));byid('count').textContent=`${selected.length} matching genes；第 ${page+1} / ${Math.max(1,Math.ceil(selected.length/100))} 页`;
byid('prev').disabled=page===0;byid('next').disabled=(page+1)*100>=selected.length;
byid('rows').innerHTML=selected.slice(page*100,(page+1)*100).map(([g,i,c])=>{const ti=targetIndex.get(g),v=ti===undefined?null:r.target_counts[pi][ti];return `<tr><td>${esc(g)}</td><td>${esc(r.statuses[c])}</td><td>${v?v[0]:'不属于官方300目标'}</td><td>${v?v[1]:'不适用'}</td><td>${v?v[2]:'不适用'}</td></tr>`}).join('');}
byid('status').innerHTML+=r.statuses.map(s=>`<option>${esc(s)}</option>`).join('');
byid('source-filter').oninput=options;for(const id of ['panel','status','only-targets'])byid(id).onchange=()=>{page=0;draw()};byid('gene-filter').oninput=()=>{page=0;draw()};byid('prev').onclick=()=>{page--;draw()};byid('next').onclick=()=>{page++;draw()};
(async()=>{const bytes=Uint8Array.from(atob(r.encoded),c=>c.charCodeAt(0));const stream=new Blob([bytes]).stream().pipeThrough(new DecompressionStream('gzip'));matrix=new Uint8Array(await new Response(stream).arrayBuffer());if(matrix.length!==r.panels.length*r.genes.length)throw Error('coverage matrix dimension mismatch');options();})().catch(e=>{byid('scope').textContent='显示失败：'+e.message+'；完整结果仍在Parquet sidecar中。';console.error(e)});
</script></html>'''


def relationship_page(graph):
    data=json.dumps(graph,ensure_ascii=False,allow_nan=False).replace('<','\\u003c')
    return '<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>跨来源关系证据图</title>'+STYLE+'''
<h1>跨来源研究／样本／文库关系</h1><p><a href="report.html">返回总报告</a> · <a href="coverage.html">完整基因覆盖</a> · <a href="relationships.json">完整机器可读图</a></p>
<p class="notice">来源关联、候选 barcode 重叠、已确认捕获关联与计数完全一致副本分别标记。共享 barcode、样本、GEM 或研究均不自动等于独立生物重复，也不证明单个物理细胞。点击邻居可继续下钻；所有边保留证据路径。</p>
<label>查找节点 <input id="query" placeholder="Replogle、SRX、研究 accession…"></label><label>节点 <select id="node"></select></label>
<label>关系 <select id="relation"><option value="">全部关系</option></select></label><p id="scope"></p><pre id="node-meta"></pre>
<button id="prev">上一页邻居</button><button id="next">下一页邻居</button><section id="graph"></section>
<section><table><thead><tr><th>邻居</th><th>关系</th><th>记录分母／解释</th><th>证据</th></tr></thead><tbody id="edges"></tbody></table></section>
<script type="application/json" id="payload">'''+data+'''</script><script>'''+HELPERS+'''
const r=JSON.parse(byid('payload').textContent),nodes=new Map(r.nodes.map(n=>[n.id,n]));let page=0;
function options(preferred){const q=byid('query').value.toLowerCase(),old=preferred||byid('node').value;byid('node').innerHTML=r.nodes.filter(n=>JSON.stringify(n).toLowerCase().includes(q)).map(n=>`<option value="${esc(n.id)}">${esc(n.label)}</option>`).join('');if([...byid('node').options].some(o=>o.value===old))byid('node').value=old;page=0;draw()}
window.choose=id=>{byid('query').value='';options(id)};
function draw(){const id=byid('node').value,n=nodes.get(id);if(!n){byid('graph').innerHTML='';byid('edges').innerHTML='';byid('scope').textContent='0 matching nodes';return;}
const kind=byid('relation').value,all=r.edges.filter(e=>(e.a===id||e.b===id)&&(!kind||e.relation===kind));page=Math.min(page,Math.max(0,Math.ceil(all.length/30)-1));const edges=all.slice(page*30,(page+1)*30);
byid('scope').textContent=`完整图 ${r.nodes.length} 节点、${r.edges.length} 条边；当前节点 ${all.length} 条匹配关系，第 ${page+1} / ${Math.max(1,Math.ceil(all.length/30))} 页`;
byid('node-meta').textContent=JSON.stringify(n,null,2);byid('prev').disabled=page===0;byid('next').disabled=(page+1)*30>=all.length;
const height=Math.max(160,edges.length*70+30),cy=height/2;
byid('graph').innerHTML=`<svg viewBox="0 0 1050 ${height}" role="img" aria-label="所选节点及全部分页关系"><rect width="1050" height="${height}" fill="#f5f8fa"/><circle cx="140" cy="${cy}" r="15" fill="#12619a"/><text x="10" y="${cy-30}" font-size="12">${esc(n.label).slice(0,90)}</text>${edges.map((e,i)=>{const other=nodes.get(e.a===id?e.b:e.a),y=i*70+40,color=e.relation.includes('exact')?'#138143':e.relation.includes('candidate')?'#ad6716':'#366895';return `<path d="M155 ${cy} L620 ${y}" stroke="${color}" fill="none"/><g data-node="${esc(other.id)}" style="cursor:pointer"><circle cx="630" cy="${y}" r="9" fill="${color}"/><text x="650" y="${y+4}" font-size="12">${esc(other.label).slice(0,70)}</text><text x="650" y="${y+23}" font-size="10">${esc(e.relation)}</text></g>`}).join('')}</svg>`;
byid('graph').querySelectorAll('[data-node]').forEach(el=>el.onclick=()=>choose(el.dataset.node));
byid('edges').innerHTML=edges.map(e=>{const other=nodes.get(e.a===id?e.b:e.a);return `<tr><td><button data-node="${esc(other.id)}">${esc(other.label)}</button></td><td>${esc(e.relation)}</td><td>${esc(JSON.stringify(e.details||{}))}</td><td>${(e.evidence||[]).map(p=>`<a href="${esc(p)}">${esc(p)}</a>`).join('<br>')}</td></tr>`}).join('');byid('edges').querySelectorAll('[data-node]').forEach(el=>el.onclick=()=>choose(el.dataset.node));}
byid('relation').innerHTML+=[...new Set(r.edges.map(e=>e.relation))].sort().map(k=>`<option>${esc(k)}</option>`).join('');byid('query').oninput=()=>options();byid('node').onchange=()=>{page=0;draw()};byid('relation').onchange=()=>{page=0;draw()};byid('prev').onclick=()=>{page--;draw()};byid('next').onclick=()=>{page++;draw()};options('replogle:K562_gwps');
</script></html>'''
