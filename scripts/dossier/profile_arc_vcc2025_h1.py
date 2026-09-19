#!/usr/bin/env python
"""arc_vcc2025_h1 的数据档案：一次流式扫描算完页面要用的全部统计量。

输出 data/profiles/arc_vcc2025_h1.json，由 scripts/dossier/render.py 渲染成
docs/datasets/arc_vcc2025_h1.html。raw/ 只读，这里只读不写。

  .venv/bin/python scripts/dossier/profile_arc_vcc2025_h1.py
"""
import hashlib, json, pathlib, sys, time
import numpy as np, pandas as pd, h5py
import scipy.sparse as sp

ROOT = pathlib.Path(__file__).resolve().parents[2]
SRC = ROOT / "data/raw/arc_vcc2025_h1"
OUT = ROOT / "data/profiles/arc_vcc2025_h1.json"
SPLITS = ["Training", "Validation", "Test"]
CHUNK = 20000
EVAL_DEPTH = 20000          # 2026 评测的中位深度，页面里的参照线
SHIFT_THR = 0.25            # |Δlog1p CP10K| 阈值，用于数“被推动的基因数”
HIST_RANGE = (0.0, 150000.0)  # UMI 直方图的固定区间
HIST_BINS = 60


def read_cat(g):
    return g["categories"][:].astype(str), g["codes"][:]


def profile(split):
    t0 = time.time()
    h = h5py.File(SRC / f"adata_{split}.h5ad", "r")
    n_cells, n_genes = (int(x) for x in h["X"].attrs["shape"])
    genes = h["var/_index"][:].astype(str)
    barcodes = h["obs/_index"][:].astype(str)
    tcats, tcodes = read_cat(h["obs/target_gene"])
    gcats, gcodes = read_cat(h["obs/guide_id"])
    bcats, bcodes = read_cat(h["obs/batch"])
    indptr = h["X/indptr"][:].astype(np.int64)
    data_ds, idx_ds = h["X/data"], h["X/indices"]
    n_t, n_b = len(tcats), len(bcats)
    tcodes, bcodes = tcodes.astype(np.int64), bcodes.astype(np.int64)

    umi = np.zeros(n_cells)
    ngene = np.diff(indptr)
    pb_t = np.zeros((n_t, n_genes))                 # 按扰动聚合的 pseudobulk
    pb_b = np.zeros((n_b, n_genes))                 # 按批次聚合的 pseudobulk
    cell_tb = np.zeros((n_t, n_b), dtype=np.int64)
    # 每个 target 自身基因的 CP10K，按扰动分组累加 -> 敲低效率
    pos = {g: i for i, g in enumerate(genes)}
    own_col = np.array([pos.get(t, -1) for t in tcats])
    own_idx = own_col[own_col >= 0]
    lut = np.full(n_genes, -1, dtype=np.int64)
    lut[own_idx] = np.arange(len(own_idx))
    frac = np.zeros((n_t, len(own_idx)))
    integer_ok = True

    for start in range(0, n_cells, CHUNK):
        stop = min(start + CHUNK, n_cells)
        lo, hi = indptr[start], indptr[stop]
        d = data_ds[lo:hi].astype(np.float64)
        ix = idx_ds[lo:hi].astype(np.int64)
        ip = (indptr[start:stop + 1] - lo).astype(np.int64)
        if start == 0:
            integer_ok = bool(np.all(d[:100000] == np.rint(d[:100000])))
        m = stop - start
        rows = np.repeat(np.arange(m), np.diff(ip))
        tot = np.bincount(rows, weights=d, minlength=m)
        umi[start:stop] = tot
        M = sp.csr_matrix((d, ix, ip), shape=(m, n_genes))
        tc, bc = tcodes[start:stop], bcodes[start:stop]
        one = np.ones(m)
        pb_t += (sp.csr_matrix((one, (tc, np.arange(m))), shape=(n_t, m)) @ M).toarray()
        pb_b += (sp.csr_matrix((one, (bc, np.arange(m))), shape=(n_b, m)) @ M).toarray()
        np.add.at(cell_tb, (tc, bc), 1)
        keep = lut[ix] >= 0
        if keep.any():
            safe = np.where(tot > 0, tot, 1.0)
            np.add.at(frac, (tc[rows[keep]], lut[ix[keep]]), d[keep] / safe[rows[keep]] * 1e4)
        del d, ix, M
    h.close()

    cells_t = np.bincount(tcodes, minlength=n_t)
    cells_b = np.bincount(bcodes, minlength=n_b)
    ntc = "non-targeting"
    ntc_i = int(np.where(tcats == ntc)[0][0])

    def logcp10k(mat):
        s = mat.sum(axis=1, keepdims=True)
        return np.log1p(mat / np.where(s == 0, 1, s) * 1e4)

    lp = logcp10k(pb_t)
    delta = lp - lp[ntc_i]
    own_names = list(tcats[own_col >= 0])
    guides = pd.Series(gcats[gcodes]).groupby(pd.Series(tcats[tcodes])).nunique()

    rows_out = []
    for i, t in enumerate(tcats):
        if i == ntc_i:
            continue
        d = delta[i]
        j = own_names.index(t) if t in own_names else None
        own = frac[i, j] / max(cells_t[i], 1) if j is not None else None
        ctl = frac[ntc_i, j] / max(cells_t[ntc_i], 1) if j is not None else None
        rows_out.append({
            "target": str(t), "split": split, "n_cells": int(cells_t[i]),
            "n_guides": int(guides.get(t, 0)),
            "cp10k_pert": round(float(own), 3) if own is not None else None,
            "cp10k_ctrl": round(float(ctl), 3) if ctl is not None else None,
            "residual": round(float(own / ctl), 4) if ctl else None,
            "l2": round(float(np.linalg.norm(d)), 3),
            "n_shift": int((np.abs(d) > SHIFT_THR).sum()),
            "max_abs": round(float(np.abs(d).max()), 3),
            "n_batches": int((cell_tb[i] > 0).sum()),
            "top_batch_share": round(float(cell_tb[i].max() / max(cell_tb[i].sum(), 1)), 4),
        })

    hist, edges = np.histogram(umi, bins=HIST_BINS, range=HIST_RANGE)   # 三个 split 共用分箱，可叠图
    q = lambda p: float(np.percentile(umi, p))
    lb = logcp10k(pb_b)
    top = np.argsort(lb.var(axis=0))[::-1][:2000]
    corr = np.corrcoef(lb[:, top])

    return {
        "split": split, "n_cells": n_cells, "n_genes": n_genes, "nnz": int(indptr[-1]),
        "density": round(float(indptr[-1] / (n_cells * n_genes)), 4),
        "counts_are_integer": integer_ok,
        "n_targets": int(n_t - 1), "n_guide_ids": int(len(gcats)), "n_batches": int(n_b),
        "ntc_cells": int(cells_t[ntc_i]),
        "ntc_guides": int(guides.get(ntc, 0)),
        "guide_example": str(gcats[0]),
        "umi": {"median": q(50), "p01": q(1), "p05": q(5), "p25": q(25), "p75": q(75),
                "p95": q(95), "p99": q(99), "mean": float(umi.mean()),
                "min": float(umi.min()), "max": float(umi.max()),
                "frac_above_eval": float((umi >= EVAL_DEPTH).mean()),
                "frac_above_hist_range": float((umi > HIST_RANGE[1]).mean()),
                "hist": hist.tolist(), "edges": [float(x) for x in edges]},
        "genes_detected": {"median": float(np.median(ngene)), "p05": float(np.percentile(ngene, 5)),
                           "p95": float(np.percentile(ngene, 95))},
        "batches": [{"batch": str(bcats[i]), "n_cells": int(cells_b[i]),
                     "median_umi": float(np.median(umi[bcodes == i])) if cells_b[i] else None}
                    for i in range(n_b)],
        "batch_corr": {"min": round(float(corr[np.triu_indices(n_b, 1)].min()), 4),
                       "median": round(float(np.median(corr[np.triu_indices(n_b, 1)])), 4)},
        "perturbations": rows_out,
        "barcodes_sha": hashlib.sha256("".join(sorted(barcodes)).encode()).hexdigest()[:16],
        "ntc_barcodes": sorted(barcodes[tcodes == ntc_i].tolist()),
        "gene_axis_sha": hashlib.sha256("|".join(genes).encode()).hexdigest()[:16],
        "elapsed_s": round(time.time() - t0, 1),
    }


def null_floor(split="Validation", ks=(40, 80, 160, 320, 640, 1280, 2560), reps=12):
    """NTC 自抽样噪声地板：抽 k 个对照细胞做 pseudobulk，与完整 NTC 比 L2。

    这是“什么都没发生”时该有的距离。没有它，细胞数少的扰动会被裸 L2 误判成强效应。
    """
    rng = np.random.default_rng(0)
    h = h5py.File(SRC / f"adata_{split}.h5ad", "r")
    n_cells, n_genes = (int(x) for x in h["X"].attrs["shape"])
    tcats, tcodes = read_cat(h["obs/target_gene"])
    is_ntc = tcodes.astype(np.int64) == int(np.where(tcats == "non-targeting")[0][0])
    pool = np.flatnonzero(is_ntc)

    assign = np.full((len(ks), n_cells), -1, dtype=np.int32)   # 每个 k 一套互不重叠的子集
    meta, gid = [], 0
    for a, k in enumerate(ks):
        perm = rng.permutation(pool)
        for r in range(min(reps, len(pool) // k)):
            assign[a, perm[r * k:(r + 1) * k]] = gid
            meta.append((k, gid)); gid += 1

    pb = np.zeros((gid, n_genes))
    pb_all = np.zeros(n_genes)
    indptr = h["X/indptr"][:].astype(np.int64)
    for start in range(0, n_cells, CHUNK):
        stop = min(start + CHUNK, n_cells)
        lo, hi = indptr[start], indptr[stop]
        M = sp.csr_matrix((h["X/data"][lo:hi].astype(np.float64),
                           h["X/indices"][lo:hi].astype(np.int64),
                           (indptr[start:stop + 1] - lo).astype(np.int64)),
                          shape=(stop - start, n_genes))
        sub = is_ntc[start:stop]
        if sub.any():
            pb_all += np.asarray(M[sub].sum(axis=0)).ravel()
        for a in range(len(ks)):
            g = assign[a, start:stop]
            keep = g >= 0
            if keep.any():
                ind = sp.csr_matrix((np.ones(int(keep.sum())), (g[keep], np.flatnonzero(keep))),
                                    shape=(gid, stop - start))
                pb += (ind @ M).toarray()
        del M
    h.close()

    def logcp(v):
        s = v.sum(axis=-1, keepdims=True)
        return np.log1p(v / np.where(s == 0, 1, s) * 1e4)

    ref, lp = logcp(pb_all), logcp(pb)
    curve = {}
    for k in ks:
        gs = [g for kk, g in meta if kk == k]
        if not gs:
            continue
        d = lp[gs] - ref
        l2 = np.linalg.norm(d, axis=1)
        ns = (np.abs(d) > SHIFT_THR).sum(axis=1)
        curve[str(k)] = {"n_rep": len(gs),
                         "l2_median": round(float(np.median(l2)), 3),
                         "l2_p90": round(float(np.percentile(l2, 90)), 3),
                         "n_shift_median": float(np.median(ns)),
                         "n_shift_p90": float(np.percentile(ns, 90))}
    return {"ks": list(ks), "reps": reps, "source_split": split,
            "ntc_cells": int(len(pool)), "curve": curve}


def main():
    res = {s: profile(s) for s in SPLITS}
    for s in SPLITS:
        print(f"{s}: {res[s]['n_cells']} cells in {res[s]['elapsed_s']}s", flush=True)

    tset = {s: {p["target"] for p in res[s]["perturbations"]} for s in SPLITS}
    ntc_sets = {s: set(res[s].pop("ntc_barcodes")) for s in SPLITS}
    floor = null_floor()
    print("null floor:", floor["curve"]["40"]["l2_p90"], "(k=40, p90)", flush=True)
    src = json.loads((SRC / "SOURCE.json").read_text())
    counts = {s: pd.read_csv(SRC / f"pert_counts_{s}.csv").to_dict("records") for s in SPLITS}

    payload = {
        "id": "arc_vcc2025_h1",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "eval_depth": EVAL_DEPTH,
        "shift_threshold": SHIFT_THR,
        "source": src,
        "splits": res,
        "pert_counts_csv": counts,
        "null_floor": floor,
        "cross": {
            "target_overlap": {f"{a}|{b}": sorted(tset[a] & tset[b])
                               for a in SPLITS for b in SPLITS if a < b},
            "ntc_identical": {f"{a}|{b}": ntc_sets[a] == ntc_sets[b]
                              for a in SPLITS for b in SPLITS if a < b},
            "ntc_cells": len(ntc_sets["Training"]),
            "rows_if_concatenated": sum(res[s]["n_cells"] for s in SPLITS),
            "unique_cells": len(set().union(*ntc_sets)) +
                            sum(res[s]["n_cells"] - res[s]["ntc_cells"] for s in SPLITS),
            "gene_axis_identical": len({res[s]["gene_axis_sha"] for s in SPLITS}) == 1,
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, ensure_ascii=False))
    print("written", OUT, f"({OUT.stat().st_size/1024:.0f} KB)")


if __name__ == "__main__":
    sys.exit(main())
