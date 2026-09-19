"""Freeze a metadata-selected human scBaseCount download; never refresh in place."""
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import urllib.parse

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
SOURCE_ID = "scbasecount_2026_01_12_human"
SEED = "scbasecount-vcc-20260918"
BUDGET = 200_000_000_000
ALLOWED = {
    "none", "none reported", "none specified", "none indicated", "untreated",
    "unperturbed", "control", "no perturbation", "no perturbations",
    "not applicable", "not_applicable",
}
PRIORITY_PATTERN = re.compile(
    r"\b(k562|rpe1|jurkat|hepg2|a549|mcf7|mcf-7|ht29|ht-29|hap1|bxpc3|bxpc-3|"
    r"h1|h9|hesc|ipsc|ipscs|hela|hek293|hek293t|u2os)\b", re.I
)


def stable_key(row):
    return hashlib.sha256(f"{SEED}|{row['srx_accession']}".encode()).hexdigest()


def main():
    raw = ROOT / "data/raw" / SOURCE_ID
    logs = ROOT / "data/logs/scbasecount_2026_01_12"
    catalog_path = logs / "human_h5ad_catalog.json"
    metadata_path = raw / "metadata/sample_metadata.parquet"
    catalog = json.loads(catalog_path.read_text())
    by_id = {Path(o["name"]).stem: o for o in catalog["objects"]}
    rows = pq.read_table(metadata_path).to_pylist()
    assert len(rows) == len(by_id) == 35263
    assert len({r["srx_accession"] for r in rows}) == len(rows)
    eligible = [r for r in rows if r["organism"] == "Homo sapiens"
                and r["cell_prep"] == "single_cell" and r["obs_count"] >= 500
                and str(r["perturbation"]).strip().lower() in ALLOWED
                and r["srx_accession"] in by_id]
    priority = sorted([r for r in eligible if PRIORITY_PATTERN.search(r["cell_line"] or "")], key=stable_key)
    priority_ids = {r["srx_accession"] for r in priority}
    strata = defaultdict(list)
    for row in eligible:
        if row["srx_accession"] in priority_ids:
            continue
        ontology = (row["tissue_ontology_term_id"] or "").strip()
        key = ontology or "unmapped:" + (row["tissue"] or "unknown").strip().lower()
        strata[key].append(row)
    for group in strata.values():
        group.sort(key=stable_key)
    selected = []
    used = 0

    def take(row):
        nonlocal used
        size = int(by_id[row["srx_accession"]]["size"])
        if used + size <= BUDGET:
            selected.append(row)
            used += size

    for row in priority:
        take(row)
    for index in range(max(map(len, strata.values()), default=0)):
        for key in sorted(strata):
            if index < len(strata[key]):
                take(strata[key][index])
    assert len(selected) == len({r["srx_accession"] for r in selected})
    metadata_downloads = json.loads((logs / "metadata_downloads.json").read_text())
    files = [{k: entry[k] for k in ["name", "url", "bytes", "checksum", "generation"]}
             for entry in metadata_downloads]
    for row in selected:
        obj = by_id[row["srx_accession"]]
        files.append({"name": "h5ad/" + Path(obj["name"]).name,
                      "url": "https://storage.googleapis.com/arc-institute-virtual-cell-atlas/"
                             + obj["name"] + "?generation=" + obj["generation"],
                      "bytes": int(obj["size"]), "checksum": "md5-b64:" + obj["md5Hash"],
                      "generation": obj["generation"]})
    selection = {
        "seed": SEED, "expression_budget_bytes": BUDGET,
        "release": "2026-01-12", "organism": "Homo sapiens", "count_feature": "GeneFull_Ex50pAS",
        "full_human_files": len(by_id), "full_human_bytes": sum(int(o["size"]) for o in by_id.values()),
        "eligible_files": len(eligible), "selected_files": len(selected), "selected_bytes": used,
        "selected_cells_from_metadata": sum(r["obs_count"] for r in selected),
        "priority_cell_line_files": sum(r["srx_accession"] in priority_ids for r in selected),
        "tissue_strata_available": len(strata), "allowed_perturbation_labels": sorted(ALLOWED),
        "min_cells": 500, "cell_prep": "single_cell",
        "selection_order": "priority cell lines first; then round-robin tissue ontology strata; SHA-256 ordering within strata",
        "priority_cell_line_regex": PRIORITY_PATTERN.pattern,
        "caveat": "Metadata-based candidates, not verified per-cell untreated controls. Disease/control labels require downstream validation.",
        "metadata_sha256": hashlib.sha256(metadata_path.read_bytes()).hexdigest(),
        "catalog_sha256": hashlib.sha256(catalog_path.read_bytes()).hexdigest(),
    }
    selection_dir = ROOT / "data/selections"
    selection_dir.mkdir(exist_ok=True)
    selection_path = selection_dir / f"{SOURCE_ID}.json"
    selection_payload = {"selection": selection, "samples": selected}
    with selection_path.open("x") as out:
        json.dump(selection_payload, out, indent=2, ensure_ascii=False)
        out.write("\n")
    source = {
        "id": SOURCE_ID, "tier": "d", "kind": "gcs_snapshot",
        "title": "scBaseCount 2026-01-12 human metadata-selected subset (GeneFull_Ex50pAS)",
        "citation": "Youngblut et al., scBaseCount; publication release 2026-01-12.",
        "license": "CC0 1.0 (Arc Virtual Cell Atlas); retain original study provenance.",
        "why": "Human cell-state representation data for VCC; complements the public CRISPRi training corpus.",
        "caveat": "A <=200 GB expression subset, not full scBaseCount. Metadata-reported no-perturbation labels are not proof of homogeneous untreated cells. X is the Unique count matrix; other mapping layers must not be mixed into raw-count training.",
        "release": selection["release"], "organism": selection["organism"],
        "count_feature": selection["count_feature"],
        "snapshot_version": "2026-01-12-human-" + hashlib.sha256(selection_path.read_bytes()).hexdigest()[:16],
        "selection_manifest": str(selection_path.relative_to(ROOT)),
        "selection_manifest_sha256": hashlib.sha256(selection_path.read_bytes()).hexdigest(),
        "selection": selection,
        "documentation_url": "https://github.com/ArcInstitute/arc-virtual-cell-atlas/tree/main/scBaseCount",
    }
    sources_path = ROOT / "data/sources.json"
    lock_path = ROOT / "data/registry.lock.json"
    sources = json.loads(sources_path.read_text())
    lock = json.loads(lock_path.read_text())
    assert SOURCE_ID not in {s["id"] for s in sources["sources"]}
    assert SOURCE_ID not in {s["id"] for s in lock["sources"]}
    sources["sources"].append(source)
    lock["sources"].append({**source, "files": files})
    lock["locked_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    lock["total_bytes"] = sum(f["bytes"] or 0 for s in lock["sources"] for f in s["files"])
    for path, payload in [(sources_path, sources), (lock_path, lock)]:
        temporary = path.with_name(path.name + ".scbasecount.tmp")
        with temporary.open("x") as out:
            json.dump(payload, out, indent=2, ensure_ascii=False)
            out.write("\n")
        temporary.replace(path)
    selection_path.chmod(0o444)
    print(json.dumps(selection, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
