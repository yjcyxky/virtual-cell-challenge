#!/usr/bin/env python
"""Full H1 structure assessment; read-only inputs and independent result sidecars.

Run with the locked scripts/dossier environment. See --help for required inputs.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import sys
import time
import uuid

import pandas as pd
import numpy as np

from rna import scan, mapping_audit, duplicate_audit, hash_file, value_hash
from render import render

ROOT = Path(__file__).resolve().parents[2]
SPLITS = ("Training", "Validation", "Test")
EVIDENCE_URL = "https://arcinstitute.org/news/behind-the-data-virtual-cell-challenge"
METHODS = {
    "counts-v1": "Scan every stored value for finiteness, negativity and integer compatibility. Dense implicit/sparse implicit zeros count as measured zeros. Compute per-cell total counts and positive-gene detection; preserve invalid flags.",
    "identity-v1": "Record ID = SHA256(JSON(input file SHA256, original row index)). Candidate duplicate = same study, batch and full source barcode. Confirm identical ordered gene-axis hash, canonical sparse count hash and source target/guide labels. No rows removed.",
    "mapping-v1": "Preserve original identifiers; HGNC Approved symbol exact match has priority, then unique alias/previous symbol/Ensembl. Ambiguity and many-to-one mappings flagged, never collapsed. Literal and mapped current-axis coverage reported separately.",
    "control-v1": "Task = split × source target; list same-batch NTC counts for each target batch. This supports descriptive matching conditional on observed metadata, not verified same-culture replication. No response estimation in this stage.",
}
OBSERVABILITY = [
    {"factor": "genetics_lineage", "status": "source_reported", "value": "H1 hESC; subclone/genotype not in supplied obs", "evidence": EVIDENCE_URL},
    {"factor": "molecular_state", "status": "pending", "value": "RNA proxies and per-cell inference in Issue #5", "evidence": "https://github.com/yjcyxky/virtual-cell-challenge/issues/5"},
    {"factor": "target_system", "status": "partly_observed", "value": "target_gene; baseline RNA measurable; protein activity not measured", "evidence": "H5AD obs and X"},
    {"factor": "intervention", "status": "source_reported", "value": "dCas9-KRAB CRISPRi; paired guides in one construct; not two independent repeats", "evidence": EVIDENCE_URL},
    {"factor": "time_history", "status": "unknown", "value": None, "evidence": "No sampling-time field; reviewed primary description does not map per-batch timing"},
    {"factor": "culture_environment", "status": "unknown", "value": None, "evidence": "No culture/medium/density field in supplied obs"},
    {"factor": "population_selection", "status": "partly_observed", "value": "Endpoint count distributions; target panel selected using pilot signal/quality; no longitudinal lineage tracking", "evidence": EVIDENCE_URL},
    {"factor": "measurement_analysis", "status": "source_reported_and_observed", "value": "Probe-based 10x Flex; batch labels observed; independent culture-replicate identity unknown", "evidence": EVIDENCE_URL},
]


def task_table(cells):
    rows = []
    for (split, target), group in cells.groupby(["split", "source_target_gene"], sort=True, dropna=False):
        if target == "non-targeting":
            continue
        control = cells[(cells.split == split) & (cells.source_target_gene == "non-targeting") & cells.source_batch.notna()]
        control_counts = control.groupby("source_batch", dropna=False).size().to_dict()
        matched = group.source_batch.map(control_counts).fillna(0)
        rows.append({"split": split, "target": target, "n_cells": len(group),
                     "guide_constructs": group.source_guide_id.nunique(), "batches": group.source_batch.nunique(),
                     "cells_with_same_batch_NTC": int((matched > 0).sum()),
                     "matched_control_cells": int(control[control.source_batch.isin(group.source_batch)].shape[0]),
                     "min_NTC_per_target_batch": int(matched.min()),
                     "independent_biological_repeats": None,
                     "control_status": "observed_batch_match_only" if (matched > 0).all() else "unmatched_batches",
                     "status": "completed"})
    return pd.DataFrame(rows)


def stat_key(path):
    s = path.stat()
    return [s.st_size, s.st_mtime_ns, s.st_ctime_ns, s.st_ino]


def records(frame):
    return json.loads(frame.to_json(orient="records"))


def assess(root, output, inventory, chunk=2048):
    t0 = time.monotonic()
    if output.exists():
        raise FileExistsError("Use a fresh assessment directory; existing results are immutable")
    if output.resolve().is_relative_to((root / "data/raw").resolve()):
        raise ValueError("Result directory must be outside raw inputs")
    audit = json.loads(inventory.read_text())
    if audit["status"] != "completed":
        raise ValueError("A completed inventory assessment is required")
    evidence = {f["file"]: f for f in audit["file_results"]}
    relative_inputs = [f"data/raw/arc_vcc2025_h1/adata_{s}.h5ad" for s in SPLITS]
    relative_inputs += [f"data/raw/arc_vcc2025_h1/pert_counts_{s}.csv" for s in SPLITS]
    relative_inputs += ["data/raw/networks/hgnc_complete_set.txt", "data/raw/arc_vcc2026_controls/gene_names.csv"]
    identities, before = {}, {}
    for relative in relative_inputs:
        path = root / relative
        before[relative] = stat_key(path)
        actual = hash_file(path)
        if actual != evidence[relative]["sha256"]:
            raise ValueError(f"Input differs from inventory: {relative}")
        identities[relative] = actual
    output.mkdir(parents=True)
    hgnc = pd.read_csv(root / "data/raw/networks/hgnc_complete_set.txt", sep="\t", dtype=str, keep_default_na=False)
    official = pd.read_csv(root / "data/raw/arc_vcc2026_controls/gene_names.csv").iloc[:, 0].astype(str).tolist()
    summaries, cells_all, mappings, genes_all, axes = [], [], [], [], []
    for split in SPLITS:
        relative = f"data/raw/arc_vcc2025_h1/adata_{split}.h5ad"
        summary, cells, genes = scan(root / relative, identities[relative], "VCC2025-H1", chunk)
        cells["split"] = split
        genes["split"] = split
        mapping = mapping_audit(genes.source_gene.tolist(), hgnc, official)
        mapping["split"] = split
        unique_mapped = set(mapping.loc[~mapping.many_to_one_mapping, "mapped_symbol"].dropna())
        observed = set(genes.source_gene)
        literal_sums = genes.groupby("source_gene").computed_sum.sum().to_dict()
        resolved_indices, ambiguous_symbols = defaultdict(list), set()
        for index, row in mapping.iterrows():
            if pd.notna(row.mapped_symbol):
                resolved_indices[row.mapped_symbol].append(index)
            ambiguous_symbols.update(row.candidates.split("|"))
        for gene in official:
            resolved = resolved_indices.get(gene, [])
            status = "measured_literal" if gene in literal_sums else "mapped_unique" if len(resolved) == 1 else "mapping_ambiguous" if gene in ambiguous_symbols else "unmeasured_in_supplied_axis"
            count_sum = float(literal_sums[gene]) if gene in literal_sums else float(genes.iloc[resolved[0]].computed_sum) if status == "mapped_unique" else None
            axes.append({"split": split, "official_gene": gene, "measurement_status": status,
                         "observed_count_sum": count_sum,
                         "measured_all_zero": count_sum == 0 if count_sum is not None else None})
        summary.update(split=split, n_targets=int(cells.source_target_gene.nunique() - 1),
                       NTC_cells=int((cells.source_target_gene == "non-targeting").sum()),
                       batches=int(cells.source_batch.nunique()), guide_constructs=int(cells.source_guide_id.nunique()),
                       official_axis_genes=len(official), literal_axis_overlap=len(observed & set(official)),
                       mapped_unique_axis_overlap=len(unique_mapped & set(official)),
                       measured_all_zero_genes=int((genes.computed_sum == 0).sum()),
                       official_no_identified_measurement=sum(a["measurement_status"] == "unmeasured_in_supplied_axis" for a in axes if a["split"] == split),
                       mapping_status_counts=dict(Counter(mapping.mapping_status)),
                       many_to_one_features=int(mapping.many_to_one_mapping.sum()))
        summaries.append(summary)
        cells_all.append(cells)
        mappings.append(mapping)
        genes_all.append(genes)
        print(f"{split}: {len(cells)} cells; all {summary['stored_values_checked']} stored values checked", flush=True)
    cells = pd.concat(cells_all, ignore_index=True)
    duplicates = duplicate_audit(cells)
    tasks = task_table(cells)
    guide_counts = cells.groupby(["split", "source_target_gene", "source_guide_id", "source_batch"], dropna=False).size().reset_index(name="cells")
    csv_checks = []
    for split in SPLITS:
        declared = pd.read_csv(root / f"data/raw/arc_vcc2025_h1/pert_counts_{split}.csv")
        csv_checks.append({"split": split, "columns": list(declared), "rows": len(declared),
                           "interpretation": "Preserved challenge requested counts, not assumed actual observed counts",
                           "declared_rows": records(declared)})
    files = {"cells.parquet": (cells, "All original record identities/labels plus computed QC; no expression matrix"),
             "genes.parquet": (pd.concat(genes_all, ignore_index=True), "Original features/annotations and full-scan gene sums/detection"),
             "gene_mapping.parquet": (pd.concat(mappings, ignore_index=True), "Mapping ambiguity and official-axis coverage"),
             "official_axis_coverage.parquet": (pd.DataFrame(axes), "Every official gene: measured, mapped, ambiguous or unmeasured; missing is not zero"),
             "duplicate_groups.parquet": (duplicates, "All repeated identity groups and count/label equality classification"),
             "tasks.parquet": (tasks, "All target tasks and control matching qualification"),
             "guide_batch_counts.parquet": (guide_counts, "Full observed guide-construct × batch coverage")}
    artifacts = []
    for filename, (frame, description) in files.items():
        frame.to_parquet(output / filename, index=False)
        artifacts.append({"file": filename, "description": description, "sha256": hash_file(output / filename), "rows": len(frame)})
    changed = [p for p in before if stat_key(root / p) != before[p]]
    identical = duplicates[duplicates.classification == "content_and_label_identical"]
    excess = int((identical.records - 1).sum())
    report = {
        "schema_version": 2, "bundle_id": "h1-structure-" + uuid.uuid4().hex,
        "title": "H1 全量计数、身份、测量轴与对照结构", "status": "failed" if changed else "completed",
        "completed_at": datetime.now(timezone.utc).isoformat(), "duration_seconds": time.monotonic() - t0,
        "input_sha256": identities, "inventory": {"bundle_id": audit["bundle_id"], "sha256": hash_file(inventory)},
        "inputs_unchanged": not changed, "changed_inputs": changed,
        "code": {"commit": subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip(),
                 "files": {p.name: hash_file(p) for p in Path(__file__).parent.glob("*.py")},
                 "uv_lock_sha256": hash_file(Path(__file__).with_name("uv.lock"))},
        "runtime": {"python": sys.version, "platform": platform.platform(),
                    "packages": {p: importlib.metadata.version(p) for p in ("numpy", "scipy", "h5py", "pandas", "pyarrow")}},
        "parameters": {"chunk_cells": chunk}, "methods": METHODS,
        "evidence": [{"url": EVIDENCE_URL, "accessed": "2026-09-19", "supports": "H1, dual-guide CRISPRi, Flex, pilot target selection and split design; no per-batch independent culture map"}],
        "exposure": {"scope": "All Training/Validation/Test counts and target labels", "response_summaries": "not computed; QC, target counts, identities and coverage exposed", "future_holdout": "These public records have been explored; not untouched validation"},
        "limitations": ["来源身份、观测标签、计算 QC 和未来推断必须分别使用。",
                        "batch 不等于独立培养重复；时间和培养环境未知，匹配仅限已观察的 batch。",
                        "测量轴外基因不填零；测量零不是生物学绝对不表达。",
                        "重复仅报告，源文件无删除；本阶段不估计响应效应、DE 或细胞类型。"],
        "duplicates": {"identity_groups": len(duplicates), "identical_groups": len(identical),
                       "conflicting_groups": int((duplicates.classification == "identity_conflict").sum()),
                       "all_rows": len(cells), "excess_identical_rows": excess,
                       "unique_records_after_confirmed_copy_accounting": len(cells) - excess,
                       "NTC_rows_as_stored": int((cells.source_target_gene == "non-targeting").sum()),
                       "NTC_excess_identical_rows": int((identical.loc[identical.target == "non-targeting", "records"] - 1).sum()),
                       "impact": "Naive concatenation reuses control evidence; no increase in independent replication"},
        "requested_count_csv": csv_checks, "artifacts": artifacts,
        "tables": [{"title": "各 split 全量扫描", "rows": summaries},
                   {"title": "重复与统计分母", "rows": [{"classification": k, "groups": int(v)} for k, v in duplicates.classification.value_counts().items()]},
                   {"title": "异质性可观测性", "rows": OBSERVABILITY},
                   {"title": "全部扰动任务与对照资格", "rows": records(tasks)}],
        "reproduce": {"argv": sys.argv},
    }
    with (output / "report.json").open("x") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2, allow_nan=False)
        fh.write("\n")
    (output / "report.html").write_text(render(report))
    (output / "SHA256SUMS").write_text("".join(f"{hash_file(p)}  {p.name}\n" for p in sorted(output.iterdir()) if p.is_file() and p.name != "SHA256SUMS"))
    print(json.dumps({"status": report["status"], "duplicates": report["duplicates"]}), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--chunk-cells", type=int, default=2048)
    args = parser.parse_args()
    if args.chunk_cells < 1:
        parser.error("chunk size must be positive")
    result = assess(args.root.resolve(), args.output.resolve(), args.inventory.resolve(), args.chunk_cells)
    return 0 if result["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
