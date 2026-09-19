#!/usr/bin/env python
"""Attach primary-source semantic review to an immutable completed numeric audit."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess
import sys
import uuid
from rna import hash_file
from render import render

ROOT = Path(__file__).resolve().parents[2]


def review(card):
    """Preserve every source label, adding separately attributable interpretations."""
    stem = Path(card["file"]).stem
    reviewed = {"status": "completed", "source_fields_preserved": True, "findings": []}
    if stem.startswith("XieHon2017"):
        reviewed.update(intervention="CRISPRi", primary_evidence="GSE81884.soft.gz",
                        interpretation="Original dCas9-KRAB experiment; legacy resource CRISPR-cas9 wording does not establish nuclease knockout")
    elif stem.startswith("SunshineHein2023"):
        reviewed.update(intervention="CRISPRi", primary_evidence="GSE208240.soft.gz",
                        interpretation="Calu-3 CRISPRi, SARS-CoV-2 infection, 24-hour post-infection endpoint; source CRISPR-cas9 label retained",
                        experimental_background="SARS-CoV-2 infection; NTC is not infection-free baseline")
    elif stem.startswith("SchraivogelSteinmetz2020"):
        reviewed.update(intervention="CRISPRi", primary_evidence="GSE135497.soft.gz",
                        interpretation="K562 dCas9-BFP-KRAB enhancer/promoter targeting; TAP-seq selected transcript readout, not unbiased whole transcriptome")
    elif stem.startswith("NadigOConner2024"):
        reviewed.update(intervention="CRISPRi", primary_evidence="existing independent Nadig source registration and GSE264667",
                        interpretation="Same original study as local nadig2025; observed generic CRISPR is not a separate modality")
    semantics = card["matrices"]["X"]["semantics"]
    if stem == "WeinrebKlein2020":
        semantics.update(value="source_confirmed_normalized_mRNA_counts", count_methods="not_applicable",
                         reason="Processing code reads *_normed_counts.mtx; GEO describes normalized mRNA counts. Do not thin, round, or present sums as raw UMI.",
                         primary_evidence="GSE140802.soft.gz")
        reviewed["findings"].append("clone_matrix is a separate lineage-tracing binary membership axis; library_names and clones are not automatically biological replicates")
    if stem.startswith("JoungZhang2023"):
        semantics.update(value="source_processed_continuous_RNA_expression", count_methods="not_applicable",
                         reason="GEO distinguishes raw CSV from processed H5AD; scPerturb imports the processed H5AD or its .raw.X. Full X is noninteger; AnnData .raw does not mean raw UMI.",
                         exact_normalization="not_verified", primary_evidence="GSE217460.soft.gz" if "atlas" in stem else "GSE217066.soft.gz")
        if stem.endswith("atlas"):
            reviewed["findings"].append("Perturbation is a numeric TF/ORF code; upstream script explicitly leaves decoding as TODO. Do not treat code as target gene or infer control from code frequency.")
            card["RNA_deep_assessment"]["target_name_mapping"] = {"status": "blocked", "reason": "source numeric TF/ORF code dictionary required; expression diagnostics still apply",
                "recovery": "Recover original GSE217460 cell-to-ORF CSV / TFAtlas reference mapping and prove barcode/code joins"}
    if stem == "GehringPachter2019":
        semantics.update(value="source_log1p_normalized_RNA_expression", count_methods="not_applicable",
                         reason="Source GPCTP_2019 notebook normalizes per lane to 10000 per cell then log1p before cDNAdata construction; full X is noninteger.",
                         primary_evidence="Gehring_original_processing.ipynb")
        reviewed["findings"].append("scPerturb Gehring processing source uses comparison == instead of assignment in BMP high-dose branch; dose=0 cannot be assumed unexposed without original experimental mapping")
        card["RNA_deep_assessment"]["dose_mapping"] = {"status": "blocked", "reason": "possible source high-BMP-dose coding error and dropped original BMP code",
            "recovery": "Join original ClickTag experimental design to source barcodes; keep current labels and recovered annotations separate"}
    card["RNA_deep_assessment"]["count_method_prerequisite"] = semantics
    card["primary_source_review"] = reviewed
    return card


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ["audit", "evidence", "output"]:
        p.add_argument("--" + name, type=Path, required=True)
    args = p.parse_args()
    if args.output.exists() or args.output.resolve().is_relative_to((ROOT / "data/raw").resolve()):
        raise ValueError("Fresh non-raw output required")
    base = json.loads((args.audit / "report.json").read_text())
    if base["status"] != "completed" or base["files_completed"] != 54:
        raise ValueError("Completed 54-file numeric audit required")
    for artifact in base["artifacts"]:
        if hash_file(args.audit / artifact["file"]) != artifact["sha256"]:
            raise ValueError("Prior audit artifact changed")
    evidence = json.loads((args.evidence / "evidence.json").read_text())
    for item in evidence:
        if "sha256" not in item or hash_file(args.evidence / item["file"]) != item["sha256"]:
            raise ValueError("Primary review evidence incomplete or changed")
    shutil.copytree(args.audit, args.output)
    shutil.copytree(args.evidence, args.output / "primary_review_evidence")
    cards = {}
    for path in sorted(args.output.glob("*.h5ad.json")):
        card = review(json.loads(path.read_text()))
        cards[card["file"]] = card
        path.write_text(json.dumps(card, ensure_ascii=False, indent=2) + "\n")
    checklist_path = args.output / "analysis_manifest.json"
    checklist = json.loads(checklist_path.read_text())
    for row in checklist:
        row["analysis"] = cards[row["file"]]["RNA_deep_assessment"]
        row["matrices"] = cards[row["file"]]["matrices"]
        row["primary_source_review"] = cards[row["file"]]["primary_source_review"]
    checklist_path.write_text(json.dumps(checklist, ensure_ascii=False, indent=2) + "\n")
    for row in base["tables"][0]["rows"]:
        card = cards[row["file"]]
        row["X_semantics"], row["RNA_next"] = card["matrices"]["X"]["semantics"], card["RNA_deep_assessment"]
        row["primary_source_review"] = card["primary_source_review"]
    base["prior_numeric_assessment"] = {"bundle_id": base["bundle_id"], "report_sha256": hash_file(args.audit / "report.json"),
                                        "producer_commit": base["code_commit"], "producer_code_sha256": base["code_sha256"]}
    base["bundle_id"] = "scperturb-reviewed-" + uuid.uuid4().hex
    base["completed_at"] = datetime.now(timezone.utc).isoformat()
    base["semantic_review"] = {"primary_evidence": evidence, "code_sha256": hash_file(Path(__file__)),
                               "code_commit": subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
                               "reproduce": sys.argv}
    base["limitations"].append("原研究支持的干预与测量语义另列 primary_source_review，来源字段不覆盖。Joung atlas 数字靶编码及 Gehring 剂量映射仍需恢复，相关诊断不能凭猜测执行。")
    base["artifacts"] = [{"file": str(x.relative_to(args.output)), "sha256": hash_file(x)} for x in sorted(args.output.rglob("*"))
                         if x.is_file() and x.name not in ["report.json", "report.html", "SHA256SUMS"]]
    (args.output / "report.json").write_text(json.dumps(base, ensure_ascii=False, indent=2) + "\n")
    (args.output / "report.html").write_text(render(base))
    print(json.dumps({"status": base["status"], "files": len(cards), "bundle_id": base["bundle_id"]}))


if __name__ == "__main__":
    main()
