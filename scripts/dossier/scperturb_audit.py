#!/usr/bin/env python
"""All-file scPerturb modality/provenance/numeric audit, without editing inputs."""
from __future__ import annotations
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
import uuid
import xml.etree.ElementTree as ET
import zipfile
import h5py
import numpy as np
import pandas as pd
from rna import read_array, hash_file, value_hash
from render import render

ROOT = Path(__file__).resolve().parents[2]
PAPER = "https://pmc.ncbi.nlm.nih.gov/articles/PMC12220817/"
NS = {"s": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
MISSING = {"", "none", "nan", "na", "n/a", "unknown", "null", "<na>"}
# Explicitly reviewed RNA import paths in the fixed upstream source snapshot.
RNA_IMPORTS = {
    "CuiHacohen2023": "RNA assay counts exported from Seurat; source names nFeature_RNA/nCount_RNA",
    "JoungZhang2023": "TFAtlas gene matrix; combinatorial import explicitly reconstructs tdata.raw.X",
    "LaraAstiasoHuntly2023": "read_10x_h5 gene expression libraries",
    "LiangWang2023": "OSCAR_DM/EM gene expression matrices; exact numerical scale checked separately",
    "LotfollahiTheis2023": "GSE206741_count_matrix.mtx and gene_metadata",
    "NadigOConner2024": "GSE264667_*_raw_singlecell_01.h5ad with source gene_name/ensembl axes",
    "SantinhaPlatt2023": "Seurat RNA count assay export; original nCount_RNA/nFeature_RNA",
    "SunshineHein2023": "10x RNA gene features; viral and lentiviral expression moved to separate obsm",
    "WesselsSatija2023": "THP1-CaRPool-seq gene expression; ADT and HTO features separately stored",
    "XuCao2023": "GSM6752591 whole_tx and nascent_tx_count_matrix.mtx; guide counts in obsm",
}


def xlsx_rows(path):
    with zipfile.ZipFile(path) as archive:
        shared = ["".join(x.itertext()) for x in ET.fromstring(archive.read("xl/sharedStrings.xml")).findall("s:si", NS)]
        rows = []
        for row in ET.fromstring(archive.read("xl/worksheets/sheet1.xml")).findall(".//s:sheetData/s:row", NS):
            values = {}
            for cell in row.findall("s:c", NS):
                node = cell.find("s:v", NS)
                value = node.text if node is not None else ""
                if cell.attrib.get("t") == "s":
                    value = shared[int(value)]
                values[re.sub(r"\d", "", cell.attrib["r"])] = value
            if values.get("A"):
                rows.append(values)
        headers = rows[0]
        return [{headers[k]: v for k, v in row.items() if k in headers and headers[k]} for row in rows[1:]]


def metadata_for(stem, metadata):
    key = "Index (=FirstauthorLastauthorYear)"
    exact = [r for r in metadata if r[key] + ("_" + r["dataset_index"] if r.get("dataset_index") else "") == stem]
    family = [r for r in metadata if r[key] == stem.split("_")[0]]
    return exact or family, "exact_file" if exact else "study_only" if family else "not_in_legacy_table"


def modality(labels, feature_names):
    modes = set()
    for label in labels:
        x = label.lower()
        if "(protein)" in x or x == "protein":
            modes.add("protein")
        elif "(rna)" in x or x == "rna":
            modes.add("RNA")
        elif "rna" in x and "protein" in x:
            modes.add("mixed_or_unspecified")
    if len(modes) == 1:
        return next(iter(modes))
    if len(modes) > 1:
        return "mixed_or_ambiguous"
    return "unknown"


def numeric_audit(node):
    dense = isinstance(node, h5py.Dataset)
    encoding = "dense" if dense else node.attrs.get("encoding-type", "unknown")
    shape = tuple(map(int, node.shape if dense else node.attrs["shape"]))
    if len(shape) != 2 or (not dense and encoding not in ["csr_matrix", "csc_matrix"]):
        raise ValueError("Unsupported matrix shape/encoding")
    data = node if dense else node["data"]
    count = bad = negative = noninteger = positive = 0
    minimum, maximum = np.inf, -np.inf
    chunk = max(1, (4 << 20) // max(shape[1], 1)) if dense else 4 << 20
    for start in range(0, len(data), chunk):
        x = np.asarray(data[start:start + chunk], dtype=np.float64).ravel()
        finite = np.isfinite(x)
        count += len(x)
        bad += int((~finite).sum())
        negative += int((x < 0).sum())
        positive += int((x > 0).sum())
        noninteger += int((finite & (x != np.floor(x))).sum())
        if finite.any():
            minimum, maximum = min(minimum, float(x[finite].min())), max(maximum, float(x[finite].max()))
    if not dense:
        pointer = node["indptr"][:]
        outer, inner = shape if encoding == "csr_matrix" else shape[::-1]
        if len(pointer) != outer + 1 or pointer[0] != 0 or pointer[-1] != count or (np.diff(pointer) < 0).any() or len(node["indices"]) != count:
            raise ValueError("Invalid sparse structure")
        for a in range(0, count, 4 << 20):
            indices = node["indices"][a:a + (4 << 20)]
            if (indices < 0).any() or (indices >= inner).any():
                raise ValueError("Out-of-bounds sparse index")
    if count < shape[0] * shape[1]:
        minimum, maximum = min(minimum, 0), max(maximum, 0)
    return {"shape": list(shape), "encoding": encoding, "dtype": str(data.dtype), "stored_values_checked": count,
            "nonfinite": bad, "negative": negative, "noninteger": noninteger, "positive_stored_values": positive,
            "minimum": minimum if np.isfinite(minimum) else None, "maximum": maximum if np.isfinite(maximum) else None,
            "nonnegative_integer_compatible": not (bad or negative or noninteger), "status": "completed"}


def layer_semantics(mode, numeric, declared=None):
    if numeric["nonfinite"]:
        return {"value": "invalid_numeric", "count_methods": "not_applicable", "reason": "nonfinite_values"}
    if declared == "normalized" or numeric["negative"] or numeric["noninteger"]:
        return {"value": "continuous_or_transformed_measurement", "count_methods": "not_applicable",
                "reason": "noninteger_or_negative_values_do_not_satisfy_raw_count_method; exact transform requires source evidence"}
    if mode == "protein":
        return {"value": "antibody_measurement_count_compatible", "count_methods": "not_applicable",
                "reason": "protein measurement; RNA library/marker methods do not apply"}
    if mode != "RNA":
        return {"value": "unresolved_measurement", "count_methods": "not_estimable", "reason": "resolve_modality_first"}
    return {"value": "source_reported_RNA_count_matrix_numeric_compatible", "count_methods": "conditional",
            "reason": "resource paper states unnormalized matrices where available; per-study preprocessing evidence retained; this is not raw sequencing or guaranteed identical UMI semantics",
            "source_evidence": PAPER}


def label_summary(array):
    missing = np.array([x is None or str(x).lower() in MISSING for x in array])
    c = Counter(str(x) for x in array if x is not None)
    return {"nunique_including_literal_missing": len(c), "missing_or_sentinel_rows": int(missing.sum()),
            "top_values": c.most_common(12), "values": sorted(c) if len(c) <= 100 else None}


def auxiliary_audit(node):
    if isinstance(node, h5py.Dataset) or "data" in node and "indptr" in node:
        result = numeric_audit(node)
        result["source_axis"] = "unnamed auxiliary columns"
        return result
    if node.attrs.get("encoding-type") == "dataframe":
        columns = {}
        index = node.attrs.get("_index", "_index")
        for name in node:
            if name == index:
                continue
            values = read_array(node[name])
            if values.dtype.kind not in "biuf":
                columns[name] = {"status": "not_applicable", "reason": "nonnumeric annotation column"}
            else:
                finite = np.isfinite(values)
                columns[name] = {"values_checked": len(values), "nonfinite": int((~finite).sum()),
                                 "negative": int((values < 0).sum()), "noninteger": int((finite & (values != np.floor(values))).sum()), "status": "completed"}
        return {"status": "completed", "source_axis": list(columns), "columns": columns}
    return {"status": "not_estimable", "reason": "unsupported auxiliary encoding", "encoding": str(node.attrs.get("encoding-type"))}


def source_evidence(study, references, evidence):
    selected = [x for x in evidence["files"] if study in x["path"]]
    accessions, urls = set(), set()
    for item in selected:
        path = references / item["path"]
        if path.suffix == ".ipynb":
            value = json.loads(path.read_text())
            text = "\n".join("".join(c.get("source", [])) for c in value.get("cells", []))
        else:
            text = path.read_text()
        accessions.update(re.findall(r"\b(?:GSE\d{5,}|GSM\d{6,}|E-MTAB-\d+|SCP\d+)\b", text))
        urls.update(re.findall(r"https?://[^\s\"'<>]+", text))
    return {"processing_sources": selected, "accession_candidates_from_processing": sorted(accessions),
            "upstream_URL_candidates_from_processing": sorted(urls),
            "caution": "Current fixed processing code is provenance evidence, not proof that its exact commit generated Zenodo v1.4 bytes."}


def audit_file(path_string, expected_hash, metadata, references_string, evidence):
    path, references = Path(path_string), Path(references_string)
    before = [getattr(path.stat(), k) for k in ["st_size", "st_mtime_ns", "st_ctime_ns", "st_ino"]]
    if hash_file(path) != expected_hash:
        raise ValueError("Source hash differs from completed inventory")
    meta, match = metadata_for(path.stem, metadata)
    study = path.stem.split("_")[0]
    provenance = source_evidence(study, references, evidence)
    with h5py.File(path, "r") as handle:
        obs, var = handle["obs"], handle["var"]
        index_key = obs.attrs.get("_index", "_index")
        barcodes = read_array(obs[index_key])
        genes = read_array(var[var.attrs.get("_index", "_index")])
        labels = {name: label_summary(read_array(obs[name])) for name in obs if name != index_key}
        reference_modalities = sorted({x["Modality = Data type"] for x in meta if x.get("Modality = Data type")})
        mode = modality(reference_modalities, genes)
        modality_evidence = "fixed_scPerturb_metadata_table"
        if mode == "unknown" and study in RNA_IMPORTS and provenance["processing_sources"]:
            mode = "RNA"
            modality_evidence = RNA_IMPORTS[study]
        matrices = {"X": numeric_audit(handle["X"])}
        for name in handle.get("layers", {}):
            matrices["layers/" + name] = numeric_audit(handle["layers"][name])
        if "raw" in handle and "X" in handle["raw"]:
            matrices["raw/X"] = numeric_audit(handle["raw/X"])
        auxiliary = {name: auxiliary_audit(handle["obsm"][name]) for name in handle.get("obsm", {})}
        for name, matrix in matrices.items():
            if matrix["shape"][0] != len(barcodes) or (name != "raw/X" and matrix["shape"][1] != len(genes)):
                raise ValueError("Matrix and metadata axes differ")
            matrix["semantics"] = layer_semantics(mode, matrix)
            if name == "layers/nascent_counts":
                matrix["measurement_scope"] = "nascent RNA source layer; distinct from whole-transcriptome X; not exchangeable"
        organism = read_array(obs["organism"]) if "organism" in obs else np.array([None] * len(barcodes))
        species = sorted(set("human" if str(x).lower() in ["human", "homo sapiens"] else "mouse" if str(x).lower() in ["mouse", "mus musculus"] else "unknown" for x in organism))
        perturbation = read_array(obs["perturbation"]) if "perturbation" in obs else np.array([None] * len(barcodes))
        perts = Counter(str(x) for x in perturbation if x is not None)
        control_candidates = {p: n for p, n in perts.items() if any(k in p.lower() for k in ["control", "ctrl", "non-target", "dmso", "vehicle", "gfp", "neg_"])}
        intervention_source = sorted({x["Perturbation"] for x in meta if x.get("Perturbation")})
        intervention_observed = labels.get("perturbation_type", {}).get("values")
        paired = path.stem.replace("_protein", "_RNA") if mode == "protein" else path.stem.replace("_RNA", "_protein")
        paired_path = path.with_name(paired + ".h5ad")
        relationships = []
        if paired != path.stem and paired_path.exists():
            with h5py.File(paired_path, "r") as pair:
                other = read_array(pair["obs"][pair["obs"].attrs.get("_index", "_index")])
            shared = len(set(map(str, barcodes)) & set(map(str, other)))
            relationships.append({"file": paired_path.name, "relationship": "paired_modalities_candidate", "shared_source_barcode_labels": shared,
                                  "same_ordered_barcode_axis": list(barcodes) == list(other), "independent_cells_or_counts": "not established by barcode alone"})
        overlap = "replogle2022" if study == "ReplogleWeissman2022" else "nadig2025" if study == "NadigOConner2024" else None
        if overlap:
            relationships.append({"source": overlap, "relationship": "same_original_study_processed_copy_candidate",
                "evidence": provenance["processing_sources"], "cell_level_duplicate": "not_established; compare original identities, feature axes and counts in #19"})
        result = {"file": path.name, "path": str(path.relative_to(ROOT)), "input_sha256": expected_hash, "study_id": study,
            "status": "completed", "cells": len(barcodes), "features": len(genes), "species": species,
            "modality": mode, "modality_evidence": modality_evidence, "source_metadata_match": match,
            "source_metadata_rows": meta, "provenance": provenance, "intervention_source_table": intervention_source,
            "intervention_obs": intervention_observed, "intervention_resolution": "source_reported; retain disagreements and generic CRISPR separately",
            "matrices": matrices, "auxiliary_obsm": auxiliary,
            "auxiliary_interpretation": "Separate measured/derived axes; RNA normalization and cell type inference never automatically applied to obsm",
            "source_obs_fields": labels, "source_var_columns": list(var),
            "source_feature_examples": [str(g) for g in genes[:20]], "source_feature_axis_sha256": value_hash([str(g) for g in genes]),
            "duplicate_feature_identifiers": len(genes) - len(set(map(str, genes))), "duplicate_source_barcodes": len(barcodes) - len(set(map(str, barcodes))),
            "control": {"explicit_source_control_rows": perts.get("control", 0), "candidate_labels_not_automatically_accepted": control_candidates,
                        "unknown_perturbation_rows": labels.get("perturbation", {}).get("missing_or_sentinel_rows"),
                        "reason": "unknown/multiplet/None not relabeled as controls; combination/stimulation background requires per-study matching"},
            "relationships": relationships,
            "RNA_deep_assessment": {"status": "pending" if mode == "RNA" else "not_applicable" if mode == "protein" else "blocked",
                "issue": 14, "reason": "RNA count/continuous methods selected per matrix semantics" if mode == "RNA" else "protein axis cannot use RNA library/marker methods" if mode == "protein" else "resolve modality with primary evidence",
                "count_method_prerequisite": matrices["X"]["semantics"], "type_reference_prerequisite": "species-matched markers; do not apply human reference to mouse",
                "response_prerequisite": "observed perturbation + defensible matched control; cell-level descriptions separate from independent repeats"}}
    after = [getattr(path.stat(), k) for k in ["st_size", "st_mtime_ns", "st_ctime_ns", "st_ino"]]
    if before != after:
        raise ValueError("Input changed during full audit")
    result["input_unchanged"] = True
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--inventory", type=Path, required=True)
    p.add_argument("--references", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args()
    if args.output.exists() or args.output.resolve().is_relative_to((ROOT / "data/raw").resolve()):
        raise ValueError("Fresh non-raw output directory required")
    inv = json.loads(args.inventory.read_text())
    if inv["status"] != "completed":
        raise ValueError("Completed inventory required")
    evidence = json.loads((args.references / "evidence.json").read_text())
    for item in evidence["files"]:
        if hash_file(args.references / item["path"]) != item["sha256"]:
            raise ValueError("Reference snapshot changed")
    metadata = xlsx_rows(args.references / "metadata/scperturb_dataset_info.xlsx")
    source = json.loads((ROOT / "data/raw/scperturb/SOURCE.json").read_text())
    if hash_file(ROOT / "data/raw/scperturb/SOURCE.json") != inv["input_sha256"]["data/raw/scperturb/SOURCE.json"]:
        raise ValueError("Source manifest changed")
    args.output.mkdir(parents=True)
    shutil.copytree(args.references, args.output / "references")
    start, results, failures = time.monotonic(), [], []
    with ProcessPoolExecutor(args.workers) as pool:
        futures = {pool.submit(audit_file, str(ROOT / "data/raw/scperturb" / f["name"]), f["sha256"], metadata, str(args.references), evidence): f for f in source["files"]}
        for future in as_completed(futures):
            item = futures[future]
            try:
                result = future.result()
                results.append(result)
                (args.output / (item["name"] + ".json")).write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
            except Exception as exc:
                failures.append({"file": item["name"], "status": "failed", "error": f"{type(exc).__name__}: {exc}"})
            print(f"modality audit {len(results) + len(failures)}/{len(source['files'])}; failures={len(failures)}", flush=True)
    results.sort(key=lambda x: x["file"])
    rows = [{"file": r["file"], "status": r["status"], "cells": r["cells"], "features": r["features"], "species": r["species"],
             "modality": r["modality"], "intervention_source": r["intervention_source_table"], "intervention_obs": r["intervention_obs"],
             "X_semantics": r["matrices"]["X"]["semantics"], "control_rows": r["control"]["explicit_source_control_rows"],
             "unknown_perturbation_rows": r["control"]["unknown_perturbation_rows"], "RNA_next": r["RNA_deep_assessment"]} for r in results]
    checklist = [{"file": r["file"], "input_sha256": r["input_sha256"], "species": r["species"], "matrices": r["matrices"],
                  "analysis": r["RNA_deep_assessment"]} for r in results]
    (args.output / "analysis_manifest.json").write_text(json.dumps(checklist, indent=2) + "\n")
    report = {"bundle_id": "scperturb-modalities-" + uuid.uuid4().hex, "title": "scPerturb 全文件模态、来源与数值层审计",
              "status": "completed" if not failures and len(results) == 54 else "failed", "completed_at": datetime.now(timezone.utc).isoformat(),
              "duration_seconds": time.monotonic() - start, "files_completed": len(results), "failures": failures,
              "inventory_sha256": hash_file(args.inventory), "code_commit": subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
              "code_sha256": hash_file(Path(__file__)), "uv_lock_sha256": hash_file(Path(__file__).with_name("uv.lock")),
              "runtime": {"python": sys.version, "h5py": h5py.__version__, "numpy": np.__version__, "pandas": pd.__version__},
              "modality_counts": dict(Counter(r["modality"] for r in results)), "exposure": "All stored X/layer/raw numeric values and all source metadata; no perturbation effects calculated in this audit",
              "methods": {"numeric": "Every dense value or sparse stored value, full index bounds and pointer checks; integer compatibility does not establish raw count semantics.",
                          "references": evidence, "source_card": "Fixed resource metadata plus study processing source; ambiguous/generic labels retained.", "primary_paper": PAPER},
              "limitations": ["元数据统一不保证相同测量语义。整数外观不等于原始 UMI，连续值也不自动证明具体归一化方法。",
                              "来源 celltype/周期/guide 标签可能已包含推断，保留来源身份，不作为独立真值。",
                              "研究级重叠、同 barcode 和多模态配对不自动代表独立重复或逐细胞计数相同。",
                              "RNA 深度诊断列为后续 #14，pending 不表示本票已计算响应或细胞类型。"],
              "tables": [{"title": "全部文件", "rows": rows}, {"title": "失败", "rows": failures}], "reproduce": sys.argv}
    report["artifacts"] = [{"file": x.name, "sha256": hash_file(x)} for x in sorted(args.output.glob("*.json"))]
    (args.output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    (args.output / "report.html").write_text(render(report))
    return 0 if report["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
