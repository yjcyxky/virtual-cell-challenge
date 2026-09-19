#!/usr/bin/env python
"""Full H1 read-only per-cell inference, composition and state expression proxies."""
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
from scipy.stats import spearmanr
from annotation import RULES, marker_model, state_model, annotate, validate_records
from rna import hash_file, value_hash
from render import render

ROOT = Path(__file__).resolve().parents[2]


def json_write(path, value):
    def convert(x):
        if isinstance(x, np.generic):
            return x.item()
        raise TypeError(type(x).__name__)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False, default=convert) + "\n")


def as_records(frame):
    return json.loads(frame.to_json(orient="records"))


def composition_and_states(frame, state_names):
    controls = frame[(frame.split == "Training") & (frame.source_target_gene == "non-targeting")]
    baseline_composition = controls.groupby(["source_batch", "inferred_type"], observed=True).size().unstack(fill_value=0)
    baseline_composition = baseline_composition.div(baseline_composition.sum(axis=1), axis=0)
    baseline_states = controls.groupby("source_batch", observed=True)[state_names].mean()
    composition, states, tasks = [], [], []
    for (split, target), group in frame.groupby(["split", "source_target_gene"], observed=True):
        proportions = group.inferred_type.value_counts(normalize=True)
        batch_weights = group.source_batch.value_counts(normalize=True)
        matched = batch_weights.index.intersection(baseline_composition.index)
        if len(matched) != len(batch_weights):
            raise ValueError("Unmatched control batches in annotation summary")
        expected = baseline_composition.loc[matched].mul(batch_weights.loc[matched], axis=0).sum()
        for label in sorted(set(proportions.index) | set(expected.index)):
            composition.append({"split": split, "target": target, "inferred_type": label,
                "n_cells": len(group), "fraction": float(proportions.get(label, 0)),
                "batch_matched_NTC_fraction": float(expected.get(label, 0)),
                "difference": float(proportions.get(label, 0) - expected.get(label, 0)),
                "status": "completed", "interpretation": "descriptive inferred composition; no independent biological replication"})
        for state in state_names:
            mean = group[state].mean()
            baseline = baseline_states.loc[matched, state].mul(batch_weights.loc[matched]).sum(min_count=1)
            values = group[state].to_numpy(dtype=float)
            depth = group.computed_total_counts.to_numpy(dtype=float)
            rho = float(spearmanr(values, depth).statistic) if np.isfinite(values).all() and np.ptp(values) > 0 and np.ptp(depth) > 0 else np.nan
            states.append({"split": split, "target": target, "state": state, "mean": mean,
                "batch_matched_NTC_mean": baseline, "difference": mean - baseline,
                "q10": group[state].quantile(.1), "median": group[state].median(), "q90": group[state].quantile(.9),
                "spearman_with_library_size": rho, "status": "completed" if pd.notna(mean) and pd.notna(baseline) else "not_estimable"})
        tasks.append({"split": split, "target": target, "n_cells": len(group),
            "unknown_type_fraction": float((group.inferred_type == "unknown").mean()),
            "lineage_counts": group.inferred_lineage.value_counts().to_dict(), "type_counts": group.inferred_type.value_counts().to_dict(),
            "method_conflict_fraction": float(group.method_conflict.mean()), "mixed_marker_fraction": float(group.mixed_marker_signal.mean()),
            "target_marker_label_changed": int(group.target_marker_label_changed.sum()),
            "source_context_mismatch_count": int(group.source_context_mismatch.sum()),
            "confidence_calibration": "uncalibrated", "probability_correct": None, "status": "completed"})
    return pd.DataFrame(composition), pd.DataFrame(states), tasks


def assess(structure, response_cache, references, output, chunk=1024):
    start = time.monotonic()
    if output.exists() or any(output.resolve().is_relative_to((ROOT / p).resolve()) for p in ["data/raw", "models"]):
        raise ValueError("Fresh result directory outside raw/model inputs required")
    report = json.loads((structure / "report.json").read_text())
    if report["status"] != "completed":
        raise ValueError("Completed structure required")
    for a in report["artifacts"]:
        if hash_file(structure / a["file"]) != a["sha256"]:
            raise ValueError("Changed structure sidecar")
    for line in (references / "SHA256SUMS").read_text().splitlines():
        digest, name = line.split("  ", 1)
        if hash_file(references / name) != digest:
            raise ValueError("Changed annotation reference")
    reference = json.loads((references / "gene_sets.json").read_text())
    for name, digest in reference["local_inputs"].items():
        if hash_file(ROOT / name) != digest:
            raise ValueError("Changed local reference input")
    output.mkdir(parents=True)
    shutil.copytree(references, output / "references")
    cells = pd.read_parquet(structure / "cells.parquet")
    mappings = pd.read_parquet(structure / "gene_mapping.parquet")
    genes = pd.read_parquet(structure / "genes.parquet")
    code = {n: hash_file(Path(__file__).with_name(n)) for n in ["annotation.py", "profile_annotations.py", "rna.py", "render.py"]}
    axis = None
    frames, cache_identities, inputs, before_stats = [], {}, {}, {}
    for split in ["Training", "Validation", "Test"]:
        relative = f"data/raw/arc_vcc2025_h1/adata_{split}.h5ad"
        raw = ROOT / relative
        before_stats[relative] = [getattr(raw.stat(), k) for k in ["st_size", "st_mtime_ns", "st_ctime_ns", "st_ino"]]
        digest = hash_file(raw)
        if digest != report["input_sha256"][relative]:
            raise ValueError("Changed source H5AD")
        inputs[relative] = digest
        cache = response_cache / ("cache_" + split)
        identity = json.loads((cache / "identity.json").read_text())
        if identity["identity"]["input_sha256"] != digest or hash_file(cache / "logcp.npy") != identity["hashes"]["logcp.npy"]:
            raise ValueError("Response normalized view identity mismatch")
        cache_identities[split] = identity
        log = np.load(cache / "logcp.npy", mmap_mode="r")
        subset = cells[cells.split == split].sort_values("row_index").reset_index(drop=True)
        validate_records(subset, digest, len(log))
        native = genes[genes.split == split].source_gene.tolist()
        if log.shape[1] != len(native) or subset.gene_axis_sha256.nunique() != 1 or subset.gene_axis_sha256.iloc[0] != value_hash(native):
            raise ValueError("Normalized view gene-axis mismatch")
        if axis is None:
            axis = native
            current = mappings[mappings.split == split].copy()
            if current.source_gene.tolist() != native:
                raise ValueError("Gene mapping order mismatch")
            safe = current.mapped_symbol.where(~current.many_to_one_mapping & (current.symbol_vs_ensembl != "conflict"))
            resolved = [x if pd.notna(x) else None for x in safe]
            target_symbols = {g: m for g, m in zip(native, resolved) if m is not None}
            model = marker_model(resolved, reference["profiles"])
            ids = np.flatnonzero(subset.source_target_gene == "non-targeting")
            mean = np.zeros(log.shape[1])
            for a in range(0, len(ids), chunk):
                mean += log[ids[a:a + chunk]].sum(axis=0, dtype=np.float64)
            mean /= len(ids)
            state_weights, state_coverage = state_model(resolved, reference["states"], mean)
            json_write(output / "type_coverage.json", model["coverage"])
            json_write(output / "state_coverage.json", state_coverage)
            np.savez_compressed(output / "scoring_weights.npz", type_weights=model["weights"], state_weights=state_weights,
                                baseline_mean=mean, valid_gene_axis=model["valid_axis"])
        elif native != axis:
            raise ValueError("Split gene axes differ")
        parts = []
        for a in range(0, len(log), chunk):
            b = min(a + chunk, len(log))
            block = np.asarray(log[a:b])
            targets = [target_symbols.get(str(x), str(x)) for x in subset.source_target_gene.iloc[a:b]]
            labels, scores = annotate(block, targets, model)
            for name, values in scores.items():
                labels[name] = values
            states = block @ state_weights
            for j, (name, details) in enumerate(zip(reference["states"], state_coverage)):
                labels["state__" + name] = states[:, j] if details["status"] == "completed" else np.nan
            parts.append(labels)
            if a % (chunk * 40) == 0:
                print(f"annotation {split}: {b}/{len(log)}", flush=True)
        annotation = pd.concat(parts, ignore_index=True)
        combined = pd.concat([subset, annotation], axis=1)
        combined["source_model_description"] = "H1 hESC (study-level source description; not per-cell truth)"
        combined["inference_reference_match"] = "human broad markers; protocol/perturbation shift uncalibrated"
        combined["source_context_mismatch"] = ~combined.inferred_lineage.isin(["pluripotent", "unknown"])
        combined["inference_exposure"] = np.where(combined.source_target_gene == "non-targeting", "NTC_baseline_RNA", "perturbed_endpoint_RNA")
        combined["future_prediction_availability"] = np.where(combined.source_target_gene == "non-targeting", "only_if_matched_baseline_measured", "unavailable_before_target_endpoint_measurement")
        combined["inference_method"] = "broad-marker-rank-and-expression-v1"
        frames.append(combined)
        del log
    complete = pd.concat(frames, ignore_index=True)
    complete.to_parquet(output / "cells.parquet", index=False, compression="zstd")
    state_names = ["state__" + x for x in reference["states"]]
    composition, states, tasks = composition_and_states(complete, state_names)
    composition.to_parquet(output / "composition.parquet", index=False)
    states.to_parquet(output / "states.parquet", index=False)
    json_write(output / "tasks.json", tasks)
    # #3 verified these duplicated NTC records have identical counts and source labels.
    unique = complete.drop_duplicates(["study_id", "source_batch", "source_barcode"])
    duplicate_groups = complete[complete.duplicated(["study_id", "source_batch", "source_barcode"], keep=False)]
    grouping = duplicate_groups.groupby(["study_id", "source_batch", "source_barcode"], observed=True)
    duplicate_label_conflicts = int((grouping.inferred_type.nunique() > 1).sum())
    if duplicate_label_conflicts:
        raise ValueError("Identical NTC records received inconsistent labels")
    changed = [r for r, before in before_stats.items() if before != [getattr((ROOT / r).stat(), k) for k in ["st_size", "st_mtime_ns", "st_ctime_ns", "st_ino"]]]
    result = {"schema_version": 2, "bundle_id": "h1-annotation-" + uuid.uuid4().hex, "title": "H1 逐细胞宽类型推断与状态表达代理",
        "status": "completed" if not changed else "failed", "completed_at": datetime.now(timezone.utc).isoformat(),
        "duration_seconds": time.monotonic() - start, "input_sha256": inputs, "changed_inputs": changed,
        "structure_sha256": hash_file(structure / "report.json"), "reference_sha256": hash_file(references / "gene_sets.json"),
        "code": code, "code_commit": subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
        "runtime": {"python": sys.version, "packages": {p: importlib.metadata.version(p) for p in ["numpy", "scipy", "pandas", "pyarrow"]}, "uv_lock_sha256": hash_file(Path(__file__).with_name("uv.lock"))},
        "methods": {"protocol": "https://github.com/yjcyxky/virtual-cell-challenge/issues/5#issuecomment-5744329833", "parameters": RULES,
            "type": "Two correlated marker scores, conservative agreement/margin gate, lineage fallback. Native counts preserved. Fixed broad hypotheses are not exhaustive.",
            "state": "Mean marker log1p(CP10K) minus expression-bin-matched control gene mean; fixed Training NTC mean bins and seed across all splits. Signed expression proxies, not pathway activation or verified phase labels.",
            "uncertainty": "No independently labeled calibration cohort. Probability correct is null for every record; multi-score agreement is not ground truth.",
            "composition": "All source records retained. Pooled unique count deduplicates only #3-confirmed barcode/batch/count-identical NTC copies; task-level split views preserve source membership."},
        "cache_identities": cache_identities, "record_count": len(complete), "unique_content_identity_records": len(unique),
        "duplicate_label_conflicts": duplicate_label_conflicts,
        "pooled_unique_types": unique.inferred_type.value_counts().to_dict(), "pooled_unique_lineages": unique.inferred_lineage.value_counts().to_dict(),
        "diagnostic_status_counts": complete.inference_status.value_counts().to_dict(),
        "uncalibrated_records": int((complete.confidence_calibration == "uncalibrated").sum()),
        "limitations": ["所有细胞类型、状态分数均为 RNA 推断；置信分数未校准，probability_correct 为空，不能作为 ground truth。",
            "候选参考只覆盖宽类型；亚型一律 unknown。粗谱系可回退，方法冲突、混合 marker 与源背景不符均保留。",
            "扰动改变 marker 或增殖状态可能改变类型分数；靶 marker 排除敏感性不消除间接响应混杂。",
            "状态是表达代理，不是通路活性、蛋白水平、真实周期阶段或分化轨迹。独立生物重复未知。",
            "扰动终点推断在预测该终点前不可获得；基线代理仅在相应 NTC 已测量时可用。"],
        "tables": [{"title": "类型参考覆盖", "rows": [{k: v for k, v in x.items() if k not in ["measured_symbols", "unmeasured_or_ambiguous"]} for x in model["coverage"]]},
                   {"title": "状态参考覆盖", "rows": [{k: v for k, v in x.items() if k != "background_source_rows"} for x in state_coverage]},
                   {"title": "全部任务的类型与不确定性", "rows": tasks},
                   {"title": "状态变化", "rows": as_records(states)}], "reproduce": sys.argv}
    result["artifacts"] = [{"file": str(p.relative_to(output)), "sha256": hash_file(p)} for p in sorted(output.rglob("*")) if p.is_file()]
    json_write(output / "report.json", result)
    (output / "report.html").write_text(render(result))
    (output / "SHA256SUMS").write_text("".join(f"{hash_file(p)}  {p.relative_to(output)}\n" for p in sorted(output.rglob("*")) if p.is_file() and p.name != "SHA256SUMS"))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["structure", "response-cache", "references", "output"]:
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    existed = args.output.exists()
    try:
        result = assess(args.structure, args.response_cache, args.references, args.output)
    except Exception as exc:
        if not existed and args.output.exists() and not (args.output / "report.json").exists():
            json_write(args.output / "failure.json", {"status": "failed", "error": f"{type(exc).__name__}: {exc}"})
        raise
    print(json.dumps({"status": result["status"], "records": result["record_count"], "types": result["pooled_unique_types"]}))


if __name__ == "__main__":
    main()
