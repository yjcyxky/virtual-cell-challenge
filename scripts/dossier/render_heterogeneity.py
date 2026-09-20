"""Offline, full-denominator plots and paginated drill-downs for frozen evidence."""
import base64
import gzip
import html
import json


def explorer_page(title,frame,display,metrics,description,download,links=None):
    # Column-oriented schema plus row arrays keeps the complete denominator
    # practical in a standalone file. No network or result sampling is used.
    encoded=base64.b64encode(gzip.compress(frame.to_json(orient='split',index=False,double_precision=12,force_ascii=False).encode(),compresslevel=6)).decode()
    config=json.dumps({'display':display,'metrics':metrics,'download':download,'links':links or {}},ensure_ascii=False).replace('<','\\u003c')
    return '''<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>'''+html.escape(title)+'''</title><style>
body{font:15px system-ui;margin:24px;color:#173042;background:#f7f9fb;line-height:1.6}h1{font-size:25px}a{color:#145b91}label{display:inline-block;margin:6px}input,select,button{font:inherit;padding:7px;border:1px solid #aac0cc;border-radius:5px;background:white}input{min-width:240px}.notice{padding:14px;background:#fff4d7;border-left:4px solid #b48620}.controls,section{background:white;padding:14px;margin-top:14px;border:1px solid #d3e0e7;border-radius:8px}canvas{width:100%;max-width:1100px;height:auto;display:block;cursor:crosshair}table{border-collapse:collapse;white-space:normal}td,th{border:1px solid #cbd8e0;padding:8px;max-width:320px;overflow-wrap:anywhere;text-align:left}th{background:#eaf1f5}tr:hover{background:#edf6ff}.scroll{overflow:auto}pre{white-space:pre-wrap;overflow-wrap:anywhere}#detail{max-height:600px;overflow:auto}.muted{color:#667c8b}button:disabled{opacity:.4}
</style><a href="report.html">← 评估结论与方法</a><h1>'''+html.escape(title)+'''</h1><p>'''+html.escape(description)+'''</p>
<div class="notice">筛选覆盖全部登记记录，分页只改变显示。空值保持不可估计，图中不补零；点击点或表格查看记录。相关性与 RMS 是描述性统计，细胞、guide、GEM 及共享对照不构成独立生物重复。终点类型／周期为未校准 RNA 推定；组成与状态内分量的 RMS 不可相加，也不是因果方差占比。</div>
<p><a id="download">完整机器可读结果</a></p><div class="controls">
<label>检索来源、目标或证据 <input id="filter" placeholder="搜索全部结果" aria-label="检索全部结果"></label>
<label id="target-label">目标精确筛选 <input id="target" list="targets" placeholder="全部目标" aria-label="目标精确筛选"></label><datalist id="targets"></datalist>
<label>状态 <select id="status" aria-label="结果状态"><option value="">全部状态</option></select></label>
<label>X <select id="x" aria-label="横轴指标"></select></label><label>Y <select id="y" aria-label="纵轴指标"></select></label>
</div><p id="counts">正在解压全部结果…</p><section><canvas id="plot" width="1100" height="540" aria-label="全部有效记录散点图"></canvas><p id="plot-summary"></p></section>
<section><button id="prev">上一页</button> <span id="page"></span> <button id="next">下一页</button><div class="scroll"><table><thead id="head"></thead><tbody id="body"></tbody></table></div></section>
<section><h2>选中记录与证据</h2><div id="evidence"></div><pre id="detail">点击点或表格行，查看完整记录。</pre></section>
<script id="config" type="application/json">'''+config+'''</script><script id="packed" type="text/plain">'''+encoded+'''</script><script>
(async()=>{const C=JSON.parse(document.getElementById('config').textContent),el=id=>document.getElementById(id);
const raw=Uint8Array.from(atob(el('packed').textContent),c=>c.charCodeAt(0));
const D=JSON.parse(await new Response(new Blob([raw]).stream().pipeThrough(new DecompressionStream('gzip'))).text());
const names=D.columns,rows=D.data,ix=Object.fromEntries(names.map((n,i)=>[n,i]));
const esc=x=>String(x).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const fmt=v=>v===null||v===undefined?'unknown / not estimable':typeof v==='number'?(Number.isInteger(v)?String(v):v.toPrecision(5)):typeof v==='object'?JSON.stringify(v):String(v);
el('download').href=C.download;
for(const m of C.metrics){for(const a of ['x','y']){const o=document.createElement('option');o.value=m;o.textContent=m;el(a).append(o)}}
el('x').value=C.metrics[0];el('y').value=C.metrics[1]||C.metrics[0];
const ti=ix.canonical_target,si=ix.status??ix.effect_status;
if(ti===undefined)el('target-label').hidden=true;else{const targets=[...new Set(rows.map(r=>r[ti]).filter(Boolean))].sort();el('targets').innerHTML=targets.map(t=>`<option value="${esc(t)}">`).join('')}
if(si!==undefined)for(const s of [...new Set(rows.map(r=>r[si]).filter(Boolean))].sort()){const o=document.createElement('option');o.value=s;o.textContent=s;el('status').append(o)}
const searchable=rows.map(r=>JSON.stringify(r).toLowerCase());let filtered=[],page=0,points=[];const pageSize=100;
function detail(i){const r=rows[i],record=Object.fromEntries(names.map((n,k)=>[n,r[k]]));el('detail').textContent=JSON.stringify(record,null,2);
el('evidence').innerHTML=Object.entries(C.links).filter(([k])=>r[ix[k]]).map(([k,prefix])=>`<p><a href="${esc(prefix+String(r[ix[k]]))}">${esc(k)}: ${esc(r[ix[k]])}</a></p>`).join('')}
function table(){const pages=Math.max(1,Math.ceil(filtered.length/pageSize));page=Math.max(0,Math.min(page,pages-1));el('page').textContent=`第 ${page+1} / ${pages} 页；每页至多 ${pageSize} 条`;
el('prev').disabled=page===0;el('next').disabled=page+1>=pages;el('head').innerHTML='<tr>'+C.display.map(k=>`<th>${esc(k)}</th>`).join('')+'</tr>';
el('body').innerHTML=filtered.slice(page*pageSize,(page+1)*pageSize).map(i=>`<tr data-row="${i}">${C.display.map(k=>`<td>${esc(fmt(rows[i][ix[k]]))}</td>`).join('')}</tr>`).join('');
for(const tr of el('body').querySelectorAll('tr'))tr.onclick=()=>detail(Number(tr.dataset.row))}
function plot(){const canvas=el('plot'),ctx=canvas.getContext('2d'),xn=el('x').value,yn=el('y').value,xi=ix[xn],yi=ix[yn];ctx.clearRect(0,0,1100,540);points=[];
let xmin=Infinity,xmax=-Infinity,ymin=Infinity,ymax=-Infinity;const valid=[];
for(const i of filtered){const x=rows[i][xi],y=rows[i][yi];if(!Number.isFinite(x)||!Number.isFinite(y))continue;valid.push(i);xmin=Math.min(xmin,x);xmax=Math.max(xmax,x);ymin=Math.min(ymin,y);ymax=Math.max(ymax,y)}
ctx.fillStyle='#173042';ctx.font='14px system-ui';ctx.fillText(xn,70,525);ctx.fillText(yn,70,22);
el('plot-summary').textContent=`${valid.length.toLocaleString()} / ${filtered.length.toLocaleString()} 条在两个坐标上同时有效；${(filtered.length-valid.length).toLocaleString()} 条缺少至少一个坐标。图中点不代表独立重复。`;
if(!valid.length){ctx.fillText('当前筛选没有可绘制的双有效坐标。不可估计记录仍在下表。',100,220);return}
if(xmin===xmax){xmin-=.5;xmax+=.5}if(ymin===ymax){ymin-=.5;ymax+=.5}const dx=(xmax-xmin)*.025,dy=(ymax-ymin)*.025;xmin-=dx;xmax+=dx;ymin-=dy;ymax+=dy;
ctx.strokeStyle='#dae4ea';ctx.fillStyle='#4b6372';for(let t=0;t<=5;t++){const x=75+t*196,y=480-t*86;ctx.beginPath();ctx.moveTo(x,50);ctx.lineTo(x,480);ctx.moveTo(75,y);ctx.lineTo(1055,y);ctx.stroke();ctx.fillText((xmin+(xmax-xmin)*t/5).toPrecision(3),x-18,501);ctx.fillText((ymin+(ymax-ymin)*t/5).toPrecision(3),8,y+4)}
ctx.fillStyle=valid.length>10000?'rgba(21,102,153,.13)':'rgba(21,102,153,.5)';for(const i of valid){const x=75+(rows[i][xi]-xmin)/(xmax-xmin)*980,y=480-(rows[i][yi]-ymin)/(ymax-ymin)*430;points.push([x,y,i]);ctx.fillRect(x-1.5,y-1.5,3,3)}
}
function draw(reset=true){if(reset)page=0;const q=el('filter').value.trim().toLowerCase(),target=el('target').value.trim(),status=el('status').value;
filtered=[];for(let i=0;i<rows.length;i++){const r=rows[i];if(q&&!searchable[i].includes(q))continue;if(target&&ti!==undefined&&r[ti]!==target)continue;if(status&&si!==undefined&&r[si]!==status)continue;filtered.push(i)}
el('counts').textContent=`筛选 ${filtered.length.toLocaleString()} / 登记 ${rows.length.toLocaleString()} 条；全量筛选、无结果抽样。`;table();plot()}
let timer;for(const id of ['filter','target'])el(id).oninput=()=>{clearTimeout(timer);timer=setTimeout(()=>draw(),180)};
for(const id of ['status','x','y'])el(id).onchange=()=>draw();el('prev').onclick=()=>{page--;table()};el('next').onclick=()=>{page++;table()};
el('plot').onclick=e=>{const b=el('plot').getBoundingClientRect(),x=(e.clientX-b.left)*1100/b.width,y=(e.clientY-b.top)*540/b.height;let nearest=null,best=100;
for(const [px,py,i]of points){const d=(px-x)**2+(py-y)**2;if(d<best){best=d;nearest=i}}if(nearest!==null)detail(nearest)};
draw();window.heterogeneityViewer={registered:rows.length,getFiltered:()=>filtered.length,getPlotted:()=>points.length,columns:names};
})().catch(e=>{document.getElementById('counts').textContent='视图加载失败：'+e.message;console.error(e)});
</script></html>'''
