#!/usr/bin/env python
"""Read-only official A/B/C baseline and NTC-guide assessment; no hidden targets."""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import uuid
import numpy as np
import pandas as pd
from annotation import annotate, marker_model, state_model, RULES
from rna import RNAFile, hash_file, value_hash, scan, mapping_audit, quantiles
from response import grouped_moments, conditional_de, correlation, resampling
from profile_responses import build_cache, write_json
from render import render

ROOT = Path(__file__).resolve().parents[2]


def validate_context(context, source, genes, manifest):
    if source.var.index.astype(str).tolist() != genes or source.shape[1] != manifest["n_genes"]:
        raise ValueError("official_gene_axis_or_order_mismatch")
    if set(source.obs.context) != {context}:
        raise ValueError("official_context_identity_mismatch")
    if set(source.obs.target_gene) != {manifest["control_label"]}:
        raise ValueError("official_noncontrol_records_present")
    n = manifest["per_context"][context]
    guide_sizes = source.obs.ntc_id.value_counts()
    if source.shape[0] != n["control_cells"] or len(guide_sizes) != n["n_ntc_ids"] or not (guide_sizes == 400).all():
        raise ValueError("official_NTC_counts_mismatch")


def pooled_without(groups, omitted):
    chosen = [g for i, g in enumerate(groups) if i != omitted]
    n = sum(g["n"] for g in chosen)
    mean = sum(g["n"] * g["mean"] for g in chosen) / n
    variance = sum((g["n"] - 1) * g["variance"] + g["n"] * (g["mean"] - mean) ** 2 for g in chosen) / (n - 1)
    return {"n": n, "mean": mean, "variance": variance,
            "detected": sum(g["detected"] for g in chosen), "batch": "context_only_no_batch_observed"}


def guide_halves(log, guides, names, seed):
    rng = np.random.default_rng(seed)
    means = np.empty((len(names), 20, 2, log.shape[1]), dtype=np.float32)
    sizes = np.empty((len(names), 20, 2), dtype=np.int32)
    for j, name in enumerate(names):
        ids = np.flatnonzero(guides == name)
        for r in range(20):
            for h, part in enumerate(np.array_split(rng.permutation(ids), 2)):
                means[j, r, h] = log[part].mean(axis=0, dtype=np.float64)
                sizes[j, r, h] = len(part)
    return means, sizes


def assess(inventory, references, output):
    if output.exists() or output.resolve().is_relative_to((ROOT / "data/raw").resolve()):
        raise ValueError("Fresh non-raw output required")
    inv = json.loads(inventory.read_text())
    if inv["status"] != "completed":
        raise ValueError("Completed inventory required")
    source_dir = ROOT / "data/raw/arc_vcc2026_controls"
    registry = json.loads((source_dir / "SOURCE.json").read_text())
    if hash_file(source_dir / "SOURCE.json") != inv["input_sha256"]["data/raw/arc_vcc2026_controls/SOURCE.json"]:
        raise ValueError("Official source provenance changed")
    manifest = json.loads((source_dir / "manifest.json").read_text())
    official_genes = pd.read_csv(source_dir / "gene_names.csv").gene_name.astype(str).tolist()
    targets = pd.read_csv(source_dir / "pert_counts.csv").target_gene.astype(str).tolist()
    if len(set(targets)) != manifest["n_constructs"] or len(set(official_genes)) != len(official_genes):
        raise ValueError("Official target/gene panel identity mismatch")
    for line in (references / "SHA256SUMS").read_text().splitlines():
        digest, name = line.split("  ", 1)
        if hash_file(references / name) != digest:
            raise ValueError("Reference snapshot changed")
    ref = json.loads((references / "gene_sets.json").read_text())
    for name, digest in ref["local_inputs"].items():
        if hash_file(ROOT / name) != digest:
            raise ValueError("Local annotation reference changed")
    registered = {f["file"]: f for f in inv["file_results"] if f["source_id"] == "arc_vcc2026_controls"}
    hashes, before = {}, {}
    h1_axis_path = ROOT / "data/raw/vcc_gene_axis/gene_names.csv"
    axis_record = next(x for x in inv["file_results"] if x["source_id"] == "vcc_gene_axis" and Path(x["file"]).name == "gene_names.csv")
    if hash_file(h1_axis_path) != axis_record["sha256"]:
        raise ValueError("H1 reference axis changed")
    hashes[str(h1_axis_path.relative_to(ROOT))] = axis_record["sha256"]
    for name in ["manifest.json", "gene_names.csv", "pert_counts.csv"] + [f"context_{c}.h5ad" for c in manifest["contexts"]]:
        path = source_dir / name
        digest = hash_file(path)
        # Inventory file keys are basenames within each source.
        expected = registered.get(name) or registered.get(str(path.relative_to(ROOT)))
        if expected is None or digest != expected["sha256"]:
            raise ValueError(f"Official input differs from inventory: {name}")
        hashes[str(path.relative_to(ROOT))] = digest
        before[name] = [getattr(path.stat(), k) for k in ["st_size", "st_mtime_ns", "st_ctime_ns", "st_ino"]]
    output.mkdir(parents=True)
    shutil.copytree(references, output / "references")
    start = time.monotonic()
    hgnc = pd.read_csv(ROOT / "data/raw/networks/hgnc_complete_set.txt", sep="\t", low_memory=False)
    mapping = mapping_audit(official_genes, hgnc, official_genes)
    mapping.to_parquet(output / "gene_mapping.parquet", index=False)
    symbols = [row.mapped_symbol if pd.notna(row.mapped_symbol) and not row.many_to_one_mapping else None for row in mapping.itertuples()]
    type_model = marker_model(symbols, ref["profiles"])
    views, cell_frames, structure_rows = {}, {}, []
    pooled_mean = np.zeros(len(official_genes))
    for i, context in enumerate(manifest["contexts"]):
        path = source_dir / f"context_{context}.h5ad"
        with RNAFile(path) as source:
            validate_context(context, source, official_genes, manifest)
        digest = hashes[str(path.relative_to(ROOT))]
        summary, cells, gene_qc = scan(path, digest, "official-vcc2026-" + context)
        if not summary["all_counts_finite_nonnegative_integer"]:
            raise ValueError("Invalid official counts; count methods not run")
        cells["context"] = context
        cell_frames[context] = cells
        gene_qc.to_parquet(output / f"gene_qc_{context}.parquet", index=False)
        summary["context"] = context
        structure_rows.append(summary)
        log, thin = build_cache(path, output / ("cache_" + context), digest, RULES["seed"] + i)
        views[context] = (log, thin)
        pooled_mean += log.sum(axis=0, dtype=np.float64)
    pooled_mean /= sum(len(x[0]) for x in views.values())
    state_weights, state_coverage = state_model(symbols, ref["states"], pooled_mean)
    write_json(output / "type_coverage.json", type_model["coverage"])
    write_json(output / "state_coverage.json", state_coverage)
    np.savez_compressed(output / "scoring_weights.npz", type_weights=type_model["weights"], state_weights=state_weights, baseline_mean=pooled_mean)
    guide_rows, resample_rows, profiles, annotated = [], [], {}, []
    for context, (log, thin) in views.items():
        cells = cell_frames[context]
        guides = cells.source_ntc_id.astype(str).to_numpy()
        names = sorted(set(guides))
        moments = grouped_moments(log, guides, names)
        thin_moments = grouped_moments(thin, guides, names)
        halves, half_sizes = guide_halves(log, guides, names, RULES["seed"] + ord(context))
        total_half_sums = (halves * half_sizes[:, :, :, None]).sum(axis=0, dtype=np.float64)
        total_half_sizes = half_sizes.sum(axis=0)
        annotations = []
        for a in range(0, len(log), 1024):
            block = np.asarray(log[a:a + 1024])
            frame, scores = annotate(block, ["non-targeting"] * len(block), type_model)
            for name, score in scores.items():
                frame[name] = score
            state_scores = block @ state_weights
            for j, name in enumerate(ref["states"]):
                frame["state__" + name] = state_scores[:, j] if state_coverage[j]["status"] == "completed" else np.nan
            annotations.append(frame)
        ann = pd.concat([cells, pd.concat(annotations, ignore_index=True)], axis=1)
        ann["inference_exposure"] = "official_NTC_baseline_RNA"
        ann["future_prediction_availability"] = "only_if_this_context_baseline_measured"
        ann["source_model_identity"] = "anonymous context " + context
        ann["inference_reference_match"] = "human broad markers; unknown cell line/protocol transfer uncalibrated"
        annotated.append(ann)
        profiles[context] = log.mean(axis=0, dtype=np.float64)
        baseline = pd.DataFrame({"source_gene": official_genes, "mean_logCP10K": profiles[context],
                                  "variance_logCP10K": log.var(axis=0, ddof=1, dtype=np.float64),
                                  "detection_fraction": (log > 0).mean(axis=0)})
        baseline.to_parquet(output / f"baseline_{context}.parquet", index=False)
        for j, guide in enumerate(names):
            own, other = np.flatnonzero(guides == guide), np.flatnonzero(guides != guide)
            control = pooled_without(moments, j)
            thin_control = pooled_without(thin_moments, j)
            effect = moments[j]["mean"] - control["mean"]
            thin_effect = thin_moments[j]["mean"] - thin_control["mean"]
            de, de_meta = conditional_de([moments[j]], [control])
            reference_halves = ((total_half_sums - halves[j] * half_sizes[j, :, :, None]) /
                                (total_half_sizes - half_sizes[j])[:, :, None])[:, :, None, :]
            seed = RULES["seed"] ^ int(value_hash([context, guide])[:8], 16)
            samples, stability = resampling(log[own], np.zeros(len(own)), log[other], np.zeros(len(other)), [0],
                                           reference_halves, np.ones(len(official_genes), dtype=bool), seed)
            resample_rows.extend({"context": context, "ntc_id": guide, **x} for x in samples)
            gene_result = pd.DataFrame({"source_gene": official_genes, "mean_logCP10K_guide": moments[j]["mean"],
                "mean_logCP10K_other_guides": control["mean"], "guide_minus_others": effect, "thinned_difference": thin_effect,
                "guide_detection_fraction": moments[j]["detected"] / moments[j]["n"], "guide_variance": moments[j]["variance"], **(de or {})})
            filename = f"guide_{context}_{value_hash(guide)[:12]}.parquet"
            gene_result.to_parquet(output / filename, index=False)
            group = ann.iloc[own]
            guide_rows.append({"context": context, "ntc_id": guide, "status": "completed", "n_cells": len(own),
                "reference_cells": len(other), "self_reference_overlap": len(np.intersect1d(own, other)),
                "library_size": quantiles(group.computed_total_counts), "detected_genes": quantiles(group.computed_detected_genes),
                "guide_RMS": float(np.sqrt(np.mean(effect ** 2))), "depth_correlation": correlation(effect, thin_effect),
                "half_effect_correlations": quantiles([x["half_effect_correlation"] for x in samples]),
                "NTC_null_RMS": quantiles([x["null_rms"] for x in samples]), "stability": stability,
                "DE": de_meta, "BH_differing_genes": int((de["q_bh"] <= .05).sum()) if de else None,
                "inferred_types": group.inferred_type.value_counts().to_dict(), "confidence_calibration": "uncalibrated",
                "gene_results": filename, "confounding": "guide/batch/culture cannot be separated; no independent replicate labels",
                "independent_biological_repeats": None})
            print(f"official guide {context} {j+1}/{len(names)}", flush=True)
    all_cells = pd.concat(annotated, ignore_index=True)
    all_cells.to_parquet(output / "cells.parquet", index=False, compression="zstd")
    pd.DataFrame(resample_rows).to_parquet(output / "resampling.parquet", index=False)
    state_cols = ["state__" + name for name in ref["states"]]
    state_rows = all_cells.groupby(["context", "source_ntc_id"], observed=True)[state_cols].agg(["mean", "std", "median"])
    state_rows.columns = ["__".join(x) for x in state_rows.columns]
    state_rows.reset_index().to_parquet(output / "states.parquet", index=False)
    context_rows = []
    for context, group in all_cells.groupby("context"):
        context_rows.append({"context": context, "cells": len(group), "guides": group.source_ntc_id.nunique(),
            "inferred_types": group.inferred_type.value_counts().to_dict(), "inferred_lineages": group.inferred_lineage.value_counts().to_dict(),
            "unknown_fraction": float((group.inferred_type == "unknown").mean()), "uncalibrated": True,
            "state_means": group[state_cols].mean().to_dict(), "status": "completed"})
    contrasts = [{"context_1": a, "context_2": b, "baseline_RMS": float(np.sqrt(np.mean((profiles[a] - profiles[b]) ** 2))),
                  "baseline_profile_correlation": correlation(profiles[a], profiles[b]), "interpretation": "baseline difference, not perturbation response"}
                 for a in profiles for b in profiles if a < b]
    h1 = set(pd.read_csv(ROOT / "data/raw/vcc_gene_axis/gene_names.csv", header=None).iloc[:, 0])
    coverage = [{"target_gene": t, "measured_in_official_axis": t in official_genes, "measured_in_H1_native_axis": t in h1,
                 "official_response_available": False} for t in targets]
    pd.DataFrame(coverage).to_parquet(output / "target_panel_coverage.parquet", index=False)
    changed = [name for name, s in before.items() if s != [getattr((source_dir / name).stat(), k) for k in ["st_size", "st_mtime_ns", "st_ctime_ns", "st_ino"]]]
    report = {"schema_version": 2, "bundle_id": "official-controls-" + uuid.uuid4().hex, "title": "官方匿名 A/B/C 全 NTC 基线与 guide 档案",
        "status": "completed" if not changed and len(guide_rows) == 138 else "failed", "completed_at": datetime.now(timezone.utc).isoformat(),
        "duration_seconds": time.monotonic() - start, "input_sha256": hashes, "changed_inputs": changed,
        "code_commit": subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
        "code": {n: hash_file(Path(__file__).with_name(n)) for n in ["profile_official_controls.py", "annotation.py", "rna.py", "response.py", "profile_responses.py", "render.py"]},
        "runtime": {"python": sys.version, "packages": {p: importlib.metadata.version(p) for p in ["numpy", "scipy", "pandas", "h5py", "pyarrow"]}, "uv_lock_sha256": hash_file(Path(__file__).with_name("uv.lock"))},
        "manifest_source": manifest, "reference_sha256": hash_file(references / "gene_sets.json"),
        "methods": {"protocol": "https://github.com/yjcyxky/virtual-cell-challenge/issues/6#issuecomment-5744480178",
                    "annotation_rules": RULES, "guide_comparison": "Guide versus all other guides in same context; no batch labels. Twenty guide-stratified disjoint control halves; own guide entirely excluded. Per-guide cell-conditional Welch/BH/BY diagnostics, not experiment-level evidence.",
                    "state_background": "Fixed expression bins from pooled A/B/C NTC baseline mean; same gene weights for all contexts.",
                    "target_panel": "Literal measured feature coverage; measured target does not establish available perturbation-response training task. Full source/task assessment in #19."},
        "exposure": "All provided official validation NTC baselines; no perturbation endpoints or hidden truth",
        "limitations": ["A/B/C 匿名身份原样保留。推断只是宽类型/表达状态代理，未校准，不识别真实细胞系。",
            "所有细胞都是 NTC；guide 差异不是官方扰动响应，也不是纯技术噪声。缺少批次/培养标签，guide 与这些来源不可分离。",
            "46 个 guide 不等于 46 个独立培养重复；DE、区间和重采样仅限条件性细胞抽样。",
            "基线细胞状态可在相应 NTC 已测量时使用；manifest 中 ground_truth_cells 只是源声明，并非本地可用隐藏数据。"],
        "tables": [{"title": "完整计数检查", "rows": structure_rows}, {"title": "匿名背景组成与状态", "rows": context_rows},
                   {"title": "背景基线差异", "rows": contrasts}, {"title": "全部 NTC guide", "rows": guide_rows},
                   {"title": "目标面板测量覆盖", "rows": coverage}], "reproduce": sys.argv}
    report["artifacts"] = [{"file": str(p.relative_to(output)), "sha256": hash_file(p)} for p in sorted(output.rglob("*"))
                           if p.is_file() and not any(part.startswith("cache_") for part in p.relative_to(output).parts)]
    write_json(output / "report.json", report)
    (output / "report.html").write_text(render(report))
    print(json.dumps({"status": report["status"], "contexts": context_rows}, default=str))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ["inventory", "references", "output"]:
        p.add_argument("--" + name, type=Path, required=True)
    args = p.parse_args()
    existed = args.output.exists()
    try:
        assess(args.inventory, args.references, args.output)
    except Exception as exc:
        if not existed and args.output.exists() and not (args.output / "report.json").exists():
            write_json(args.output / "failure.json", {"status": "failed", "error": f"{type(exc).__name__}: {exc}"})
        raise


if __name__ == "__main__":
    main()
