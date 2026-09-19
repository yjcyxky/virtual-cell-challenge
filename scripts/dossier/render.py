#!/usr/bin/env python
"""把 data/profiles/<id>.json 渲染成 docs/datasets/<id>.html。

模板放在 templates/<id>.html：它是 artifact 形态的「页面正文」（<title> + <style>
+ 标签 + <script>，不含 doctype/html/head/body），渲染时补上外壳写成可直接双击
打开的单文件页面，同时另存一份正文供发布用。

占位符两种：{{name}} 由 PLACEHOLDERS 填，/*__DATA__*/ 换成注入页面的 JSON。

  .venv/bin/python scripts/dossier/render.py arc_vcc2025_h1
"""
import json, math, pathlib, re, sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
TPL_DIR = pathlib.Path(__file__).resolve().parent / "templates"
SPLITS = ["Training", "Validation", "Test"]

SHELL = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
  :root {{ color-scheme: light dark; }}
  body {{ margin: 0; font: 14px system-ui, sans-serif; }}
  img {{ max-width: 100%; }}
  [hidden] {{ display: none !important; }}
</style>
{body}
</html>
"""

# 这一行是页面的论点，不是标题——每个数据集自己写
HEADLINE = {
    "arc_vcc2025_h1": "唯一与评测同源的公开语料，但深度、对照臂和基因轴三处都对不上",
}


def power_fit(ks, vs):
    """log-log 最小二乘，返回 (系数 a, 指数 b)，使 v ≈ a * k**b。"""
    lk = [math.log(k) for k in ks]
    lv = [math.log(v) for v in vs]
    n = len(lk)
    mk, mv = sum(lk) / n, sum(lv) / n
    b = sum((x - mk) * (y - mv) for x, y in zip(lk, lv)) / sum((x - mk) ** 2 for x in lk)
    return math.exp(mv - b * mk), b


def pct(x, nd=1):
    return f"{x * 100:.{nd}f}".rstrip("0").rstrip(".") if nd else f"{round(x * 100)}"


def build(pid):
    prof = json.loads((ROOT / f"data/profiles/{pid}.json").read_text())
    src, cross, splits = prof["source"], prof["cross"], prof["splits"]
    nf = prof["null_floor"]

    # ── 噪声地板：对 p90 做幂律拟合，外推到散点的整个 x 轴 ──
    ks = [float(k) for k in nf["curve"]]
    p90 = [nf["curve"][k]["l2_p90"] for k in nf["curve"]]
    a, b = power_fit(ks, p90)
    floor_curve = []
    x = 25.0
    while x <= 6000:
        floor_curve.append({"k": round(x, 1), "l2": round(a * x ** b, 4)})
        x *= 1.18
    floor_at = lambda n: a * max(n, 1.0) ** b

    # ── 扰动表：摊平三个 split，补上地板归一化 ──
    perts = []
    for s in SPLITS:
        for r in splits[s]["perturbations"]:
            r = dict(r)
            r["residual"] = r["residual"] if r["residual"] is not None else 0.0
            r["ratio"] = round(r["l2"] / floor_at(r["n_cells"]), 3)
            perts.append(r)
    perts.sort(key=lambda r: -r["ratio"])

    res = sorted(x["residual"] for x in perts)
    med = lambda v: v[len(v) // 2] if len(v) % 2 else (v[len(v) // 2 - 1] + v[len(v) // 2]) / 2
    kd_median = med(res)
    weak = sorted([r for r in perts if r["residual"] > 0.4], key=lambda r: -r["residual"])
    worst = weak[0]

    ncells = [r["n_cells"] for r in perts]
    l2 = [r["l2"] for r in perts]
    ratio = [r["ratio"] for r in perts]
    logn = [math.log(n) for n in ncells]

    def corr(u, v):
        n = len(u)
        mu, mv = sum(u) / n, sum(v) / n
        num = sum((x - mu) * (y - mv) for x, y in zip(u, v))
        den = math.sqrt(sum((x - mu) ** 2 for x in u) * sum((y - mv) ** 2 for y in v))
        return num / den

    bat_umi = [x["median_umi"] for s in SPLITS for x in splits[s]["batches"] if x["median_umi"]]
    tshare = sorted(r["top_batch_share"] for r in perts)
    umi = splits["Training"]["umi"]
    n_genes = splits["Training"]["n_genes"]
    dup_rows = cross["rows_if_concatenated"] - cross["unique_cells"]
    ntc = cross["ntc_cells"]
    eval_depth = prof["eval_depth"]

    ph = {
        "id": pid,
        "headline": HEADLINE[pid],
        "tier_upper": src["tier"].upper(),
        "license": src["license"],
        "citation": src["citation"],
        "why": src["why_this_project"],
        "retrieved": src["retrieved_at"],
        "retrieved_by": src["retrieved_by"],
        "bucket": "arc-institute-virtual-cell-atlas",
        "n_files": src["n_files"],
        "bytes_gib": f"{src['bytes'] / 2 ** 30:.1f}",
        "generated_at": prof["generated_at"][:19].replace("T", " "),
        "guide_example": splits["Training"]["guide_example"],
        "ntc_guides": splits["Training"]["ntc_guides"],

        "unique_cells_fmt": f"{cross['unique_cells']:,}",
        "concat_rows_fmt": f"{cross['rows_if_concatenated']:,}",
        "dup_rows_fmt": f"{dup_rows:,}",
        "dup_pct": f"{dup_rows / cross['rows_if_concatenated'] * 100:.0f}",
        "ntc_cells_fmt": f"{ntc:,}",
        "ntc_share_pct": f"{ntc / cross['unique_cells'] * 100:.0f}",
        "ntc_share_naive_pct": f"{ntc * 3 / cross['rows_if_concatenated'] * 100:.0f}",

        "n_perts": len(perts),
        "n_genes_fmt": f"{n_genes:,}",
        "gene_gap": 18533 - n_genes,
        "axis_sha": splits["Training"]["gene_axis_sha"],
        "cells_min": min(ncells),
        "cells_max_fmt": f"{max(ncells):,}",

        "median_umi_fmt": f"{round(umi['median']):,}",
        "median_umi_k": f"{umi['median'] / 1000:.0f}",
        "min_umi_fmt": f"{round(umi['min']):,}",
        "eval_depth_k": eval_depth // 1000,
        "eval_depth_fmt": f"{eval_depth:,}",
        "depth_ratio": f"{umi['median'] / eval_depth:.1f}",
        "frac_above_eval_pct": f"{math.floor(umi['frac_above_eval'] * 10000) / 100:.2f}",
        "cells_below_eval": sum(round((1 - splits[s]["umi"]["frac_above_eval"]) * splits[s]["n_cells"])
                                for s in SPLITS),

        "kd_median_pct": f"{kd_median * 100:.1f}",
        "kd_weak_n": len(weak),
        "kd_worst_name": worst["target"],
        "kd_worst_pct": f"{worst['residual'] * 100:.0f}",
        "kd_worst_cells_fmt": f"{worst['n_cells']:,}",

        "corr_l2": f"{corr(logn, l2):+.2f}",
        "corr_ratio": f"{corr(logn, ratio):+.2f}",
        "null_reps": nf["reps"],
        "null_slope": f"{b:.2f}",
        "ratio_gt2_pct": f"{sum(r > 2 for r in ratio) / len(ratio) * 100:.0f}",
        "shift_thr": prof["shift_threshold"],
        "nshift_ge3_pct": f"{sum(r['n_shift'] >= 3 for r in perts) / len(perts) * 100:.0f}",
        "nshift_ge10_pct": f"{sum(r['n_shift'] >= 10 for r in perts) / len(perts) * 100:.0f}",

        "batch_umi_min_k": f"{min(bat_umi) / 1000:.0f}",
        "batch_umi_max_k": f"{max(bat_umi) / 1000:.0f}",
        "batch_umi_ratio": f"{max(bat_umi) / min(bat_umi):.1f}",
        "batch_corr_min": min(splits[s]["batch_corr"]["min"] for s in SPLITS),
        "top_share_max_pct": f"{max(tshare) * 100:.0f}",
        "top_share_med_pct": f"{med(tshare) * 100:.1f}",
    }

    payload = {
        "id": pid,
        "generated_at": prof["generated_at"],
        "eval_depth": eval_depth,
        "shift_threshold": prof["shift_threshold"],
        "source": {k: src[k] for k in ("title", "citation", "license", "retrieved_at", "files")},
        "cross": cross,
        "perts": perts,
        "splits": {s: {k: v for k, v in splits[s].items() if k != "perturbations"} for s in SPLITS},
        "derived": {"floor_curve": floor_curve, "kd_median": round(kd_median, 4),
                    "floor_a": round(a, 4), "floor_b": round(b, 4)},
    }

    body = (TPL_DIR / f"{pid}.html").read_text()
    missing = {m for m in re.findall(r"\{\{(\w+)\}\}", body)} - set(ph)
    if missing:
        raise SystemExit(f"模板里有未提供的占位符: {sorted(missing)}")
    body = re.sub(r"\{\{(\w+)\}\}", lambda m: str(ph[m.group(1)]), body)
    body = body.replace("/*__DATA__*/", json.dumps(payload, ensure_ascii=False, separators=(",", ":")))

    out = ROOT / f"docs/datasets/{pid}.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(SHELL.format(body=body), encoding="utf-8")
    body_out = ROOT / f"docs/datasets/.{pid}.body.html"      # 发布用：artifact 正文
    body_out.write_text(body, encoding="utf-8")
    print(f"{out}  ({out.stat().st_size / 1024:.0f} KB)")
    print(f"{body_out}  (artifact 正文)")


if __name__ == "__main__":
    build(sys.argv[1] if len(sys.argv) > 1 else "arc_vcc2025_h1")
