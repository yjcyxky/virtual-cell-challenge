#!/usr/bin/env python
"""Read-only scBaseCount provenance assessment from frozen ENA XML evidence.

Run in the locked scripts/dossier environment. --resume only reuses identical
requests with validated response hashes and identical selection/code identities.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET

import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/dossier"))
from render import render

SOURCE = "scbasecount_2026_01_12_human"
ENDPOINT = "https://www.ebi.ac.uk/ena/browser/api/xml/"


def digest(data):
    return hashlib.sha256(data).hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def txt(node, path):
    return node.findtext(path, default="") if node is not None else ""


def attributes(node, kind):
    values = defaultdict(list)
    if node is not None:
        for item in node.findall(f".//{kind}_ATTRIBUTE"):
            values[txt(item, "TAG")].append(txt(item, "VALUE"))
    return dict(values)


def identifiers(node):
    if node is None:
        return []
    values = [node.attrib.get("accession", "")]
    values.extend(x.text for x in node.findall("./IDENTIFIERS/*") if x.text)
    return sorted(set(values) - {""})


def fetch_batch(kind, ids, cache):
    url = ENDPOINT + ",".join(ids)
    key = kind.lower() + "-" + digest(url.encode())[:20]
    response, metadata = cache / (key + ".xml"), cache / (key + ".json")
    if metadata.exists():
        entry = json.loads(metadata.read_text())
        if entry["url"] != url or digest(response.read_bytes()) != entry["sha256"]:
            raise ValueError(f"Frozen evidence changed: {key}")
        return entry
    error = None
    for attempt in range(3):
        try:
            time.sleep(0.1)
            with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "VCC-readonly-assessment/1.0"}), timeout=60) as handle:
                data = handle.read()
            tree = ET.fromstring(data)
            nodes = tree.findall(".//" + kind)
            if tree.tag == kind:
                nodes = [tree]
            entry = {"kind": kind, "url": url, "requested": ids, "retrieved_at": now(),
                     "sha256": digest(data), "file": str(response.name),
                     "returned": [i for n in nodes for i in identifiers(n)], "status": "completed"}
            temporary = response.with_suffix(".partial")
            temporary.write_bytes(data)
            temporary.replace(response)
            metadata.write_text(json.dumps(entry, ensure_ascii=False, indent=2))
            return entry
        except (OSError, ET.ParseError) as exc:
            error = f"{type(exc).__name__}: {exc}"
            time.sleep(2 ** attempt)
    return {"kind": kind, "url": url, "requested": ids, "retrieved_at": now(), "status": "failed", "error": error}


def retrieve(kind, ids, cache, workers):
    ids = sorted(set(ids) - {""})
    batches = [ids[i:i + 50] for i in range(0, len(ids), 50)]
    with ThreadPoolExecutor(max_workers=workers) as executor:
        entries = list(executor.map(lambda chunk: fetch_batch(kind, chunk, cache), batches))
    returned = set(x for e in entries for x in e.get("returned", []))
    missing = sorted(set(ids) - returned)
    if missing:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            entries.extend(executor.map(lambda i: fetch_batch(kind, [i], cache), missing))
    records, evidence = {}, {}
    for entry in entries:
        if entry["status"] != "completed":
            continue
        tree = ET.fromstring((cache / entry["file"]).read_bytes())
        nodes = tree.findall(".//" + kind)
        if tree.tag == kind:
            nodes = [tree]
        for node in nodes:
            for accession in identifiers(node):
                records[accession], evidence[accession] = node, entry["file"]
    print(f"{kind}: {sum(i in records for i in ids)}/{len(ids)} requested accessions resolved", flush=True)
    return records, evidence, entries


def fetch_document(url, cache, extension):
    """Freeze supplementary primary-source evidence without accession guessing."""
    key = "supplement-" + digest(url.encode())[:20]
    path, meta = cache / (key + extension), cache / (key + ".json")
    if meta.exists():
        entry = json.loads(meta.read_text())
        if digest(path.read_bytes()) != entry["sha256"]:
            raise ValueError("Supplementary evidence hash mismatch")
        return path.read_bytes(), entry
    with urllib.request.urlopen(url, timeout=60) as response:
        data = response.read()
    entry = {"url": url, "file": path.name, "retrieved_at": now(), "sha256": digest(data), "status": "completed", "kind": "supplementary"}
    path.write_bytes(data)
    meta.write_text(json.dumps(entry, indent=2))
    return data, entry


def ncbi_package(data, expected_accession):
    """UID lookup is not identity evidence: accept only the requested experiment."""
    tree = ET.fromstring(data)
    for package in tree.findall(".//EXPERIMENT_PACKAGE"):
        experiment = package.find("EXPERIMENT")
        if experiment is not None and experiment.get("accession") == expected_accession:
            return experiment, package.find("SAMPLE"), package.find("STUDY"), None
    return None, None, None, " | ".join((x.text or "") for x in tree.findall(".//ERROR")) or "requested_accession_not_returned"


def classify(local, experiment, sample, study):
    """Evidence flags, not inferred ground-truth labels or automatic filters."""
    sample_attributes = attributes(sample, "SAMPLE")
    experiment_title = txt(experiment, "TITLE")
    library = txt(experiment, ".//LIBRARY_NAME")
    study_title = txt(study, ".//STUDY_TITLE")
    sample_title = txt(sample, "TITLE")
    species = txt(sample, ".//SCIENTIFIC_NAME")
    tax_id = txt(sample, ".//TAXON_ID")
    direct = " ".join([experiment_title, library])
    # Protocol text often names all library types: only direct library/title
    # evidence is used to flag a particular experiment as non-GEX.
    non_gex = re.findall(r"(?i)\b(?:ADT|HTO|CITE[- ]?seq|ATAC[- ]?seq|VDJ|V\(D\)J|TCR|BCR|sgRNA|gRNA|guide[- _]?RNA|CRISPR[- _]?guide)\b", direct)
    gex = re.findall(r"(?i)\b(?:GEX|mRNA|RNA[- ]?seq|gene[- _]?expression|transcriptome)\b", direct)
    text_all = " ".join([direct, sample_title, study_title, json.dumps(sample_attributes)])
    perturb = re.findall(r"(?i)\b(?:perturb[- ]?seq|CRISPRi|CRISPRa|knockdown|knockout|sgRNA|stimulated|stimulation|treated|treatment|drug)\b", text_all)
    study_xml = ET.tostring(study, encoding="unicode") if study is not None else ""
    overlap = []
    if "SRP376262" in identifiers(study) or "PRJNA831566" in identifiers(study):
        overlap.append("replogle2022")
    if "SRP501831" in identifiers(study) or "PRJNA1100571" in identifiers(study):
        overlap.append("nadig2025")
    for geo, source in [("GSE264667", "nadig2025"), ("GSE225775", "mcfaline_figueroa2024")]:
        if geo in study_xml and source not in overlap:
            overlap.append(source)
    missing = [kind for kind, node in (("experiment", experiment), ("sample", sample), ("study", study)) if node is None]
    flags = []
    if species and (tax_id != "9606" or species != "Homo sapiens"):
        flags.append("species_conflict")
    if non_gex:
        flags.append("non_GEX_title_signal")
    if perturb:
        flags.append("perturbation_or_condition_signal")
    if overlap:
        flags.append("supervised_study_overlap")
    if missing:
        flags.append("incomplete_evidence")
    if not tax_id:
        flags.append("species_unknown")
    return {
        "experiment_accession": local["srx_accession"],
        "source_organism": local.get("organism"), "source_perturbation": local.get("perturbation"),
        "source_cell_line": local.get("cell_line"), "source_cells_from_metadata": local.get("obs_count"),
        "source_cell_count_is_unique": False,
        "upstream_species": species or None, "upstream_tax_id": tax_id or None,
        "experiment_title": experiment_title or None, "library_name": library or None,
        "library_strategy": txt(experiment, ".//LIBRARY_STRATEGY") or None,
        "library_source": txt(experiment, ".//LIBRARY_SOURCE") or None,
        "library_selection": txt(experiment, ".//LIBRARY_SELECTION") or None,
        "experiment_design": txt(experiment, ".//DESIGN_DESCRIPTION") or None,
        "sample_accessions": identifiers(sample), "study_accessions": identifiers(study),
        "sample_title": sample_title or None, "study_title": study_title or None,
        "sample_attributes": sample_attributes,
        "library_assessment": "conflicting_or_mixed_title_signals" if non_gex and gex else "non_GEX_title_signal" if non_gex else "GEX_title_signal" if gex else "unknown",
        "library_evidence_terms": sorted(set(non_gex + gex)),
        "condition_evidence_terms": sorted(set(perturb)), "supervised_overlap": overlap,
        "baseline_eligibility": "not_certified_from_archive_metadata",
        "per_cell_perturbation_labels": "not_available_from_sample_metadata",
        "independent_biological_repeats": None,
        "flags": flags, "status": "blocked" if missing else "completed",
        "missing_records": missing,
        "recovery": "Recover missing archive records / resolve submission contradictions and original sample design before baseline use" if missing or flags else "Validate per-cell labels and experimental design before claiming untreated baseline",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.is_relative_to((ROOT / "data/raw").resolve()) or output.is_relative_to((ROOT / "models").resolve()):
        parser.error("Results must be outside raw inputs")
    selection_path = ROOT / f"data/selections/{SOURCE}.json"
    identity = {"selection_sha256": digest(selection_path.read_bytes()), "code_sha256": digest(Path(__file__).read_bytes())}
    if output.exists():
        if not args.resume or json.loads((output / "identity.json").read_text()) != identity:
            parser.error("Existing output requires --resume with identical code and selection")
        if (output / "report.json").exists():
            parser.error("Completed results are immutable; use a new output directory")
    else:
        output.mkdir(parents=True)
        (output / "identity.json").write_text(json.dumps(identity, indent=2))
    cache = output / "upstream"
    cache.mkdir(exist_ok=True)
    selection = json.loads(selection_path.read_text())["samples"]
    local_files = {p.stem for p in (ROOT / f"data/raw/{SOURCE}/h5ad").glob("*.h5ad")}
    selected_ids = {s["srx_accession"] for s in selection}
    if local_files != selected_ids:
        raise ValueError("Local H5AD selection and frozen manifest differ")
    archive_ids = {x for x in selected_ids if re.fullmatch(r"[SED]RX\d+", x)}
    exps, ee, evidence = retrieve("EXPERIMENT", archive_ids, cache, args.workers)
    sample_ids, study_ids = {}, {}
    for accession in selected_ids:
        if accession not in exps:
            continue
        sample_node = exps[accession].find(".//SAMPLE_DESCRIPTOR")
        study_node = exps[accession].find("STUDY_REF")
        sample_ids[accession] = sample_node.get("accession", "") if sample_node is not None else ""
        study_ids[accession] = study_node.get("accession", "") if study_node is not None else ""
    samples, se, sample_evidence = retrieve("SAMPLE", sample_ids.values(), cache, args.workers)
    studies, ste, study_evidence = retrieve("STUDY", study_ids.values(), cache, args.workers)
    evidence += sample_evidence + study_evidence
    supplements, external_gaps = {}, {}
    for local in selection:
        accession = local["srx_accession"]
        if accession.startswith("NRX"):
            collection = local.get("czi_collection_id")
            if collection:
                url = "https://api.cellxgene.cziscience.com/curation/v1/collections/" + collection
                data, entry = fetch_document(url, cache, ".jsondata")
                evidence.append(entry)
                supplements[accession] = (json.loads(data), entry["file"])
            continue
        if accession in exps and sample_ids.get(accession) in samples:
            continue
        time.sleep(0.4)  # Respect unauthenticated NCBI E-utilities request rate.
        url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=sra&id=" + str(local["entrez_id"])
        data, entry = fetch_document(url, cache, ".xml")
        evidence.append(entry)
        exp, sample, study, error = ncbi_package(data, accession)
        if exp is None:
            external_gaps[accession] = {"reason": error, "evidence_file": entry["file"]}
            continue
        exps[accession], ee[accession] = exp, entry["file"]
        if sample is not None:
            sid = sample.get("accession")
            sample_ids[accession], samples[sid], se[sid] = sid, sample, entry["file"]
        if study is not None:
            stid = study.get("accession")
            study_ids[accession], studies[stid], ste[stid] = stid, study, entry["file"]
    documentation_url = "https://raw.githubusercontent.com/ArcInstitute/arc-virtual-cell-atlas/main/scBaseCount/README.md"
    _, documentation_evidence = fetch_document(documentation_url, cache, ".txt")
    evidence.append(documentation_evidence)
    _, nadig_evidence = fetch_document("https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE264667&targ=self&form=text&view=full", cache, ".txt")
    evidence.append(nadig_evidence)
    failed_ids = {accession for e in evidence if e["status"] == "failed" for accession in e.get("requested", [])}
    rows = []
    for local in selection:
        accession = local["srx_accession"]
        sid, stid = sample_ids.get(accession), study_ids.get(accession)
        row = classify(local, exps.get(accession), samples.get(sid), studies.get(stid))
        if row["missing_records"] and failed_ids.intersection({accession, sid, stid}):
            row["status"] = "failed"
        row.update(sample_ref=sid, study_ref=stid, evidence_files=[x for x in [ee.get(accession), se.get(sid), ste.get(stid)] if x])
        row["upstream_resolution_status"] = row["status"]
        if accession in external_gaps:
            gap = external_gaps[accession]
            row.update(status="completed", upstream_resolution_status="blocked", recovery=gap["reason"] + "; recover original public study metadata or await restored archive availability")
            row["evidence_files"].append(gap["evidence_file"])
        if accession in supplements:
            collection, filename = supplements[accession]
            row.update(status="completed", upstream_resolution_status="collection_only", study_title=collection.get("name"),
                       archive_kind="non_SRA_NRX", czi_collection_id=collection.get("collection_id"),
                       czi_collection_version_id=collection.get("collection_version_id"),
                       czi_dataset_candidates=[d["dataset_id"] for d in collection.get("datasets", [])],
                       specific_dataset_assignment="not_estimable_from_supplied_NRX_mapping",
                       recovery="NRX is a non-SRA identifier. Collection recovered; obtain NRX-to-original-dataset/sample mapping before assigning a specific dataset or donor.",
                       missing_records=["specific_original_dataset_sample_mapping"],
                       flags=["collection_only_provenance", "specific_sample_mapping_unknown"])
            row["evidence_files"] += [filename, documentation_evidence["file"]]
            row["study_accessions"] = [collection.get("collection_id")]
            row["sample_ref"] = None
            row["study_ref"] = collection.get("collection_id")
        rows.append(row)
    shared = Counter(r["sample_ref"] for r in rows if r["sample_ref"])
    for row in rows:
        row["local_experiments_sharing_sample"] = shared.get(row["sample_ref"], 0)
        if row["local_experiments_sharing_sample"] > 1:
            row["flags"].append("shared_sample_multi_library_or_run")
    (output / "samples.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2))
    (output / "evidence.json").write_text(json.dumps(evidence, ensure_ascii=False, indent=2))
    parquet_rows = [{**row, "sample_attributes": json.dumps(row["sample_attributes"], ensure_ascii=False)} for row in rows]
    pq.write_table(pa.Table.from_pylist(parquet_rows), output / "samples.parquet")
    flags = Counter(f for row in rows for f in row["flags"])
    artifacts = [{"file": p.name, "sha256": digest(p.read_bytes()), "description": "Fixed archive evidence / sample assessment"} for p in sorted(output.iterdir()) if p.is_file()]
    report = {"schema_version": 2, "bundle_id": "scbase-provenance-" + identity["selection_sha256"][:16],
              "title": "scBaseCount 全样本上游溯源与可观测性", "completed_at": now(),
              "status": "failed" if any(r["status"] == "failed" for r in rows) else "blocked" if any(r["status"] == "blocked" for r in rows) else "completed",
              "identity": identity, "code_commit": subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
              "scope": {"local_H5AD": len(local_files), "assessed_experiments": len(rows), "unique_sample_refs": len(shared),
                        "flag_counts": dict(flags), "metadata_cell_total_not_unique": sum(s["obs_count"] for s in selection)},
              "methods": {"archive-v2": "ENA experiment XML → sample and study XML; missing SRX uses NCBI UID with strict returned-accession identity check; NRX uses source-linked CELLxGENE collection, never a synthetic UID as an SRA identifier. Store full response/hash/time/request evidence; exact selection and code identity required to resume. Completed assessment can retain blocked metadata fields supported by explicit archive unavailability evidence.",
                          "flag-v1": "Species by tax_id/scientific_name; library/condition title regexes are evidence signals, not truth labels; exact study accessions/GEO links establish supervised study overlap; shared sample IDs do not establish independence"},
              "limitations": ["归档元数据可能缺失或错误；标记为文本线索，不是实验真值。",
                              "nominal untreated 不等于逐细胞未处理；多文库与重复细胞须在表达评估中核查。",
                              "细胞数量来自选择元数据，不代表去重后的表达细胞数；原始标签没有改写。"],
              "exposure": {"source_expression": "not_read_in_this_provenance_stage", "archive_metadata": "all selected experiments and linked sample/study records"},
              "artifacts": artifacts, "reproduce": {"argv": sys.argv, "environment": "scripts/dossier/uv.lock"},
              "tables": [{"title": "完整样本台账", "rows": rows, "columns": ["experiment_accession", "sample_ref", "study_ref", "upstream_species", "experiment_title", "library_assessment", "flags", "status", "upstream_resolution_status", "recovery"]}]}
    (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    (output / "report.html").write_text(render(report))
    (output / "SHA256SUMS").write_text("".join(digest(p.read_bytes()) + "  " + str(p.relative_to(output)) + "\n" for p in sorted(output.rglob("*")) if p.is_file() and p.name != "SHA256SUMS"))
    print(json.dumps(report["scope"]), flush=True)
    return 0 if report["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
