#!/usr/bin/env python
"""Reproducible H1 response assessment on all tasks, with resumable analysis caches."""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys
import time
import uuid

import numpy as np
import pandas as pd

from rna import RNAFile, hash_file, value_hash, quantiles
from response import (grouped_moments, matched_effect, conditional_de, control_half_means,
                      resampling, consistency, correlation)
from render import render

ROOT = Path(__file__).resolve().parents[2]
PARAMETERS = {"seed": 20260919, "normalization_total": 10000, "thinning_expected_cap": 10000,
              "resampling_repetitions": 20, "de_min_target_cells": 20, "de_min_cells_per_stratum_arm": 2,
              "de_min_detected_cells_either_arm": 10, "fdr": 0.05, "consistency_min_cells": 10}


def serial(value):
    if isinstance(value, dict):
        return {k: serial(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [serial(v) for v in value]
    if isinstance(value, np.generic):
        return serial(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(serial(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def build_cache(path, directory, source_hash, seed, chunk=2048):
    directory.mkdir(exist_ok=True)
    stamp = directory / "identity.json"
    key = {"input_sha256": source_hash, "parameters": PARAMETERS, "seed": seed,
           "code_sha256": hash_file(Path(__file__)), "reader_sha256": hash_file(Path(__file__).with_name("rna.py"))}
    if stamp.exists():
        recorded = json.loads(stamp.read_text())
        if recorded["identity"] != key:
            raise ValueError("Analysis cache identity mismatch")
        for name, digest in recorded["hashes"].items():
            if hash_file(directory / name) != digest:
                raise ValueError(f"Analysis cache changed: {name}")
        return np.load(directory / "logcp.npy", mmap_mode="r"), np.load(directory / "thin_logcp.npy", mmap_mode="r")
    rng = np.random.default_rng(seed)
    with RNAFile(path) as source:
        log = np.lib.format.open_memmap(directory / "logcp.npy", mode="w+", dtype="float32", shape=source.shape)
        thin_log = np.lib.format.open_memmap(directory / "thin_logcp.npy", mode="w+", dtype="float32", shape=source.shape)
        for start, matrix in source.blocks(chunk):
            stop = start + matrix.shape[0]
            totals = np.asarray(matrix.sum(axis=1)).ravel()
            if not np.isfinite(matrix.data).all() or (matrix.data < 0).any() or (matrix.data != np.floor(matrix.data)).any():
                raise ValueError("Invalid raw count encountered in response cache")
            scale = np.divide(10000., totals, out=np.zeros_like(totals), where=totals > 0)
            thin = matrix.copy()
            thin.data = rng.binomial(matrix.data.astype(np.int64), np.repeat(np.minimum(1, scale), np.diff(matrix.indptr))).astype(float)
            thin.eliminate_zeros()
            thin_totals = np.asarray(thin.sum(axis=1)).ravel()
            thin_scale = np.divide(10000., thin_totals, out=np.zeros_like(thin_totals), where=thin_totals > 0)
            thin.data = np.log1p(thin.data * np.repeat(thin_scale, np.diff(thin.indptr)))
            matrix.data = np.log1p(matrix.data * np.repeat(scale, np.diff(matrix.indptr)))
            log[start:stop], thin_log[start:stop] = matrix.toarray().astype(np.float32), thin.toarray().astype(np.float32)
            if start % (chunk * 20) == 0:
                print(f"cache {path.name}: {stop}/{source.shape[0]} cells", flush=True)
        log.flush()
        thin_log.flush()
        del log, thin_log
    write_json(stamp, {"identity": key, "hashes": {name: hash_file(directory / name) for name in ["logcp.npy", "thin_logcp.npy"]}})
    return np.load(directory / "logcp.npy", mmap_mode="r"), np.load(directory / "thin_logcp.npy", mmap_mode="r")


def task_result(target_name, target, thin_target, cell_rows, control, control_batches,
                batch_names, controls, thin_controls, half_controls, genes, mapping_conflicts, seed):
    batches = cell_rows.source_batch.astype(str).to_numpy()
    groups = grouped_moments(target, batches, batch_names)
    thin_groups = grouped_moments(thin_target, batches, batch_names)
    effect, match = matched_effect(groups, controls)
    if effect is None:
        return {"target": target_name, "status": "not_estimable", "matching": match}, None, []
    gene_positions = np.flatnonzero(np.asarray(genes) == target_name)
    own = int(gene_positions[0]) if len(gene_positions) == 1 else None
    downstream = np.ones(len(genes), dtype=bool)
    if own is not None:
        downstream[own] = False
    thin_effect, thin_match = matched_effect(thin_groups, thin_controls)
    de, de_meta = conditional_de(groups, controls)
    stability, stability_meta = resampling(target, batches, control, control_batches, batch_names,
                                           half_controls, downstream, seed, PARAMETERS["resampling_repetitions"])
    agreement = consistency(target, batches, cell_rows.source_guide_id.astype(str).to_numpy(), batch_names,
                            controls, downstream, PARAMETERS["consistency_min_cells"])
    total_n = sum(t["n"] for t, c in zip(groups, controls) if t["n"] and c["n"])
    weights = np.array([t["n"] / total_n if c["n"] else 0 for t, c in zip(groups, controls)])
    control_mean = sum(w * c["mean"] for w, c in zip(weights, controls))
    control_detection = sum(w * c["detected"] / c["n"] for w, c in zip(weights, controls) if c["n"])
    target_mean = control_mean + effect
    target_detection = sum(t["detected"] for t in groups) / len(target)
    target_rna = {"status": "not_estimable", "reason": "target_missing_or_ambiguous"}
    if own is not None and target_name not in mapping_conflicts:
        pert_cp = float(np.expm1(target[:, own].astype(float)).mean())
        ctrl_cp = sum(w * np.expm1(control[c["indices"], own].astype(float)).mean() for w, c in zip(weights, controls) if c["n"])
        target_rna = {"status": "completed" if ctrl_cp > 0 else "not_estimable", "reason": None if ctrl_cp > 0 else "zero_control_target_RNA",
                      "mean_CP10K_target": pert_cp, "mean_CP10K_control": float(ctrl_cp),
                      "RNA_ratio": pert_cp / ctrl_cp if ctrl_cp > 0 else None,
                      "interpretation": "RNA association only; not measured intervention dose, protein suppression, or ground truth efficiency"}
    elif target_name in mapping_conflicts:
        target_rna["reason"] = "source_symbol_Ensembl_conflict"
    gene_results = pd.DataFrame({"source_gene": genes, "is_target": np.logical_not(downstream),
                                 "effect_all_matched_cells": effect, "mean_logCP10K_target": target_mean,
                                 "mean_logCP10K_matched_control": control_mean,
                                 "detection_fraction_target": target_detection, "detection_fraction_matched_control": control_detection,
                                 "variance_logCP10K_target": target.var(axis=0, ddof=1, dtype=np.float64) if len(target) > 1 else np.nan,
                                 "effect_thinned": thin_effect})
    if de is not None:
        for name, values in de.items():
            gene_results[name] = values
    else:
        for name in ("effect_de_subset", "se", "df", "p", "q_bh", "q_by", "ci95_low", "ci95_high"):
            gene_results[name] = np.nan
        gene_results["tested"] = False
    correlations = [x["half_effect_correlation"] for x in stability if x["half_effect_correlation"] is not None]
    null_rms = [x["null_rms"] for x in stability if x["null_rms"] is not None]
    summary = {"target": target_name, "status": "completed", "n_cells": len(target), "matching": match,
               "effect_scale": "batch-weighted difference of mean cell log1p(CP10K)",
               "downstream_RMS": float(np.sqrt(np.mean(effect[downstream] ** 2))),
               "downstream_L2": float(np.linalg.norm(effect[downstream])),
               "DE": de_meta, "DEG_BH_excluding_target": int((de["q_bh"][downstream] <= .05).sum()) if de is not None else None,
               "DEG_BY_excluding_target": int((de["q_by"][downstream] <= .05).sum()) if de is not None else None,
               "target_RNA": target_rna, "consistency": agreement,
               "stability": dict(stability_meta, correlations=quantiles(correlations), null_RMS=quantiles(null_rms)),
               "depth_sensitivity": {"status": thin_match["status"], "correlation": correlation(effect[downstream], thin_effect[downstream]),
                                     "downstream_RMS": float(np.sqrt(np.mean(thin_effect[downstream] ** 2))),
                                     "interpretation": "One seeded binomial-thinning analysis, not an official required depth"},
               "library_size": quantiles(cell_rows.computed_total_counts),
               "detected_genes": quantiles(cell_rows.computed_detected_genes),
               "independent_biological_repeats": None}
    return serial(summary), gene_results, serial(stability)


def assess(structure_dir, output, resume=False):
    source_report = json.loads((structure_dir / "report.json").read_text())
    if source_report["status"] != "completed":
        raise ValueError("Completed H1 structure assessment required")
    if any(not s["all_counts_finite_nonnegative_integer"] for s in source_report["tables"][0]["rows"]):
        raise ValueError("Response methods require valid counts")
    code = {name: hash_file(Path(__file__).with_name(name)) for name in ("rna.py", "response.py", "render.py", "profile_responses.py")}
    identity = {"structure_sha256": hash_file(structure_dir / "report.json"), "code": code, "parameters": PARAMETERS,
                "uv_lock_sha256": hash_file(Path(__file__).with_name("uv.lock"))}
    if output.exists():
        if not resume or json.loads((output / "identity.json").read_text()) != identity:
            raise ValueError("Existing output requires matching --resume identity")
        if (output / "report.json").exists():
            raise ValueError("Completed results are immutable")
    else:
        output.mkdir(parents=True)
        write_json(output / "identity.json", identity)
    cells = pd.read_parquet(structure_dir / "cells.parquet")
    mappings = pd.read_parquet(structure_dir / "gene_mapping.parquet")
    source_genes = pd.read_parquet(structure_dir / "genes.parquet")
    for artifact in source_report["artifacts"]:
        if hash_file(structure_dir / artifact["file"]) != artifact["sha256"]:
            raise ValueError(f"Structure artifact changed: {artifact['file']}")
    summaries, resamples = [], []
    control, control_thin, control_batches, controls, thin_controls, halves = (None,) * 6
    t0 = time.monotonic()
    inputs, input_stats = {}, {}
    for split_i, split in enumerate(("Training", "Validation", "Test")):
        relative = f"data/raw/arc_vcc2025_h1/adata_{split}.h5ad"
        path = ROOT / relative
        s = path.stat()
        input_stats[relative] = (s.st_size, s.st_mtime_ns, s.st_ctime_ns, s.st_ino)
        actual = hash_file(path)
        if actual != source_report["input_sha256"][relative]:
            raise ValueError(f"Source changed: {relative}")
        inputs[relative] = actual
        cache_dir = output / ("cache_" + split)
        log, thin = build_cache(path, cache_dir, actual, PARAMETERS["seed"] + split_i)
        subset = cells[cells.split == split].sort_values("row_index").reset_index(drop=True)
        genes = source_genes[source_genes.split == split].source_gene.tolist()
        batch_names = sorted(subset.source_batch.unique())
        if control is None:
            ids = np.flatnonzero(subset.source_target_gene == "non-targeting")
            control, control_thin = log[ids], thin[ids]
            control_batches = subset.source_batch.to_numpy()[ids]
            controls = grouped_moments(control, control_batches, batch_names)
            thin_controls = grouped_moments(control_thin, control_batches, batch_names)
            halves = control_half_means(control, control_batches, batch_names, PARAMETERS["resampling_repetitions"], PARAMETERS["seed"] + 10)
            control_axis = genes
            control_batch_names = batch_names
        if genes != control_axis or batch_names != control_batch_names:
            raise ValueError("H1 control-reference axis/batch compatibility changed")
        conflicts = set(mappings.loc[mappings.symbol_vs_ensembl == "conflict", "source_gene"])
        for target in sorted(set(subset.source_target_gene) - {"non-targeting"}):
            task_id = split + "-" + value_hash(target)[:16]
            directory = output / task_id
            directory.mkdir(exist_ok=True)
            if (directory / "result.json").exists():
                saved = json.loads((directory / "result.json").read_text())
                if saved["identity"] != identity:
                    raise ValueError("Task cache identity mismatch")
                if saved["genes_sha256"] != hash_file(directory / "genes.parquet"):
                    raise ValueError("Task gene result changed")
                if saved["resampling_sha256"] != hash_file(directory / "resampling.json"):
                    raise ValueError("Task resampling result changed")
                summary = saved["summary"]
                samples = json.loads((directory / "resampling.json").read_text())
            else:
                ids = np.flatnonzero(subset.source_target_gene == target)
                target_log, target_thin = log[ids], thin[ids]
                seed = PARAMETERS["seed"] ^ int(value_hash([split, target])[:8], 16)
                summary, per_gene, samples = task_result(target, target_log, target_thin, subset.iloc[ids], control,
                    control_batches, batch_names, controls, thin_controls, halves, genes, conflicts, seed)
                summary["split"] = split
                if per_gene is None:
                    per_gene = pd.DataFrame({"source_gene": genes, "status": summary["status"]})
                per_gene.to_parquet(directory / "genes.parquet", index=False)
                summary["gene_results"] = {"file": task_id + "/genes.parquet", "sha256": hash_file(directory / "genes.parquet")}
                write_json(directory / "resampling.json", samples)
                write_json(directory / "result.json", {"identity": identity, "summary": summary,
                    "genes_sha256": hash_file(directory / "genes.parquet"), "resampling_sha256": hash_file(directory / "resampling.json")})
            summaries.append(summary)
            resamples.extend({"split": split, "target": target, **s} for s in samples)
            if len(summaries) % 10 == 0:
                print(f"response tasks {len(summaries)}/300", flush=True)
        del log, thin
    pd.DataFrame(resamples).to_parquet(output / "resampling.parquet", index=False)
    write_json(output / "tasks.json", summaries)
    compact_rows = [{"split": s["split"], "target": s["target"], "n_cells": s.get("n_cells"), "status": s["status"],
                     "downstream_RMS": s.get("downstream_RMS"), "DE_status": s.get("DE", {}).get("status"),
                     "DEG_BH_excluding_target": s.get("DEG_BH_excluding_target"), "target_RNA": s.get("target_RNA"),
                     "guide_consistency": s.get("consistency", {}).get("guide"),
                     "stability": s.get("stability"), "depth_sensitivity": s.get("depth_sensitivity")} for s in summaries]
    changed = []
    for relative, before in input_stats.items():
        s = (ROOT / relative).stat()
        if (s.st_size, s.st_mtime_ns, s.st_ctime_ns, s.st_ino) != before:
            changed.append(relative)
    report = {"schema_version": 2, "bundle_id": "h1-response-" + uuid.uuid4().hex, "title": "H1 全扰动响应与样本内稳定性",
              "status": "completed" if len(summaries) == 300 and not changed else "failed", "identity": identity,
              "inputs_unchanged": not changed, "changed_inputs": changed,
              "completed_at": datetime.now(timezone.utc).isoformat(), "duration_seconds": time.monotonic() - t0,
              "input_sha256": inputs, "code_commit": subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
              "runtime": {"python": sys.version, "packages": {p: importlib.metadata.version(p) for p in ("numpy", "scipy", "pandas", "h5py", "pyarrow")}},
              "methods": {"protocol": "https://github.com/yjcyxky/virtual-cell-challenge/issues/4#issuecomment-5744199366",
                          "parameters": PARAMETERS, "control_reference": "Training NTC counts and one shared thinning realization for all splits; #3 proved identical controls and gene axes",
                          "DE": "Weighted stratum-wise cell mean differences; variance sum(w^2*(s_t^2/n_t+s_c^2/n_c)); Welch-Satterthwaite df. Per-task BH and BY sensitivity over detection-qualified native genes. Target excluded only from downstream summary, retained in per-gene DE family.",
                          "cache": "Read-only input → float32 log1p(CP10K) and seeded binomial-thinned analysis views. No source filtering/relabeling. Cached input/code/config and content hashes checked on resume."},
              "exposure": "All 300 public H1 responses and all measured genes; not untouched future validation",
              "limitations": ["细胞为统计与重采样单位，独立培养重复未知；检验和区间只描述当前样本，不能证明跨实验复现。",
                              "NTC 变异不等于纯技术噪声；基因间相关及模型近似影响名义 FDR/区间，提供 BY 敏感性。",
                              "RNA 比值不等于真实干预剂量或蛋白抑制；单 guide 不伪造一致性。",
                              "半样本分位数不是置信区间；深度敏感性是一次固定种子的分析变换，非官方深度要求。"],
              "diagnostic_status_counts": {key: dict(Counter(s[key]["status"] for s in summaries if key in s)) for key in ("DE", "stability", "target_RNA")},
              "artifacts": [{"file": "tasks.json", "sha256": hash_file(output / "tasks.json"), "description": "All 300 detailed task summaries"},
                            {"file": "resampling.parquet", "sha256": hash_file(output / "resampling.parquet"), "description": "All half-split and disjoint-control resamples"}] +
                           [{**s["gene_results"], "description": s["split"] + ": " + s["target"] + " — full gene statistics"} for s in summaries if "gene_results" in s],
              "tables": [{"title": "全部任务", "rows": compact_rows}], "reproduce": {"argv": sys.argv}}
    write_json(output / "report.json", report)
    (output / "report.html").write_text(render(report))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--structure", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    if any(output.is_relative_to((ROOT / x).resolve()) for x in ("data/raw", "models")):
        parser.error("Outputs must be outside raw inputs")
    existed = output.exists()
    try:
        report = assess(args.structure.resolve(), output, args.resume)
    except Exception as exc:
        if not existed and output.is_dir() and not (output / "report.json").exists():
            write_json(output / "failure.json", {"status": "failed", "error": f"{type(exc).__name__}: {exc}"})
        raise
    return 0 if report["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
