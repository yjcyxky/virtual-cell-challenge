#!/usr/bin/env python
"""Freeze annotation reference bytes and derived gene sets; no raw data edits."""
import argparse
from datetime import datetime, timezone
import gzip
import io
import json
import os
from pathlib import Path
import subprocess
import urllib.request
import zipfile
import pandas as pd
from annotation import PROFILES
from rna import hash_file, mapping_audit

ROOT = Path(__file__).resolve().parents[2]
SEURAT_COMMIT = "9db57c26f93c588989c80744b0e5bb27c4d8c116"
PATHWAYS = {"hypoxia": "R-HSA-1234174", "heat_shock": "R-HSA-3371453", "UPR": "R-HSA-381119",
            "oxidative_stress_senescence": "R-HSA-2559580", "interferon_alpha_beta": "R-HSA-909733",
            "interferon_gamma": "R-HSA-877300", "WNT_noncanonical": "R-HSA-3858494",
            "TGF_beta": "R-HSA-170834", "EMT_related": "R-HSA-2173791"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rscript", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.resolve().is_relative_to((ROOT / "data/raw").resolve()):
        raise ValueError("Fresh reference snapshot outside raw required")
    args.output.mkdir(parents=True)
    evidence = []
    urls = {"PanglaoDB_markers_27_Mar_2020.tsv.gz": "https://panglaodb.se/markers/PanglaoDB_markers_27_Mar_2020.tsv.gz",
            "cc.genes.updated.2019.rda": f"https://raw.githubusercontent.com/satijalab/seurat/{SEURAT_COMMIT}/data/cc.genes.updated.2019.rda"}
    for name, url in urls.items():
        with urllib.request.urlopen(url, timeout=120) as response:
            data = response.read()
        path = args.output / name
        path.write_bytes(data)
        evidence.append({"file": name, "url": url, "sha256": hash_file(path), "retrieved_at": datetime.now(timezone.utc).isoformat()})
    marker = pd.read_csv(io.BytesIO(gzip.decompress((args.output / "PanglaoDB_markers_27_Mar_2020.tsv.gz").read_bytes())), sep="\t")
    marker = marker[marker.species.str.contains("Hs", na=False)]
    hgnc = pd.read_csv(ROOT / "data/raw/networks/hgnc_complete_set.txt", sep="\t", low_memory=False)

    def resolved(genes):
        genes = sorted(set(genes))
        mapped = mapping_audit(genes, hgnc, [])
        return sorted(set(m if pd.notna(m) else g for g, m in zip(genes, mapped.mapped_symbol)))

    profiles = {}
    for name, (lineage, members) in PROFILES.items():
        chosen = marker[marker["cell type"].isin(members)]
        profiles[name] = {"lineage": lineage, "reference_types": members,
                          "reference_types_found": sorted(chosen["cell type"].unique()),
                          "genes": resolved(chosen["official gene symbol"].dropna()),
                          "reference": urls["PanglaoDB_markers_27_Mar_2020.tsv.gz"]}
    r_environment = os.environ.copy()
    r_root = args.rscript.resolve().parents[1] / "lib/R"
    r_exec = r_root / "bin/exec/R"
    r_command = [str(args.rscript)]
    # Some preinstalled conda ELF files lack owner execute permission. Launch the
    # readable ELF through the system loader without changing that environment.
    if r_exec.exists() and not os.access(r_exec, os.X_OK):
        r_command = ["/lib/ld-linux-aarch64.so.1", str(r_exec), "--vanilla", "--slave"]
        r_environment["R_HOME"] = str(r_root)
        r_environment["LD_LIBRARY_PATH"] = str(r_root / "lib") + ":" + str(r_root.parent)
    subprocess.run(r_command + ["-e",
        'a<-commandArgs(TRUE);load(a[1]);x<-cc.genes.updated.2019;write.table(data.frame(state=rep(names(x),lengths(x)),gene=unlist(x)),a[2],sep="\\t",row.names=FALSE,quote=FALSE)',
        "--args", str(args.output / "cc.genes.updated.2019.rda"), str(args.output / "cycle.tsv")], check=True, env=r_environment)
    cycles = pd.read_csv(args.output / "cycle.tsv", sep="\t")
    states = {"cycle_" + phase.replace(".genes", ""): {"genes": resolved(group.gene), "reference": urls["cc.genes.updated.2019.rda"]}
              for phase, group in cycles.groupby("state")}
    reactome = ROOT / "data/raw/networks/ReactomePathways.gmt.zip"
    with zipfile.ZipFile(reactome) as archive:
        for filename in archive.namelist():
            for line in archive.read(filename).decode().splitlines():
                title, identifier, *genes = line.split("\t")
                for name, requested in PATHWAYS.items():
                    if requested == identifier:
                        states[name] = {"genes": resolved(genes), "reference": f"https://reactome.org/content/detail/{identifier}", "source_title": title}
    if not all(name in states for name in PATHWAYS):
        raise ValueError("Missing requested Reactome set")
    states["pluripotency_marker"] = {"genes": profiles["pluripotent_like"]["genes"], "reference": profiles["pluripotent_like"]["reference"]}
    result = {"profiles": profiles, "states": states, "evidence": evidence,
              "seurat_commit": SEURAT_COMMIT, "human_markers_only": True,
              "local_inputs": {str(p.relative_to(ROOT)): hash_file(p) for p in [reactome, ROOT / "data/raw/networks/hgnc_complete_set.txt"]},
              "code_sha256": hash_file(Path(__file__)), "profile_code_sha256": hash_file(Path(__file__).with_name("annotation.py")),
              "r_version": subprocess.check_output(r_command + ["-e", "cat(R.version.string)"], text=True, env=r_environment).strip(),
              "r_command": r_command,
              "limitations": ["Analyst-defined broad candidate universe is not exhaustive.", "No independent labeled calibration cohort; scores are not correctness probabilities.", "Healthy marker references do not establish malignant identity or equivalence across protocols.", "Reference gene renaming is an analysis-only HGNC mapping; original bytes preserved."]}
    (args.output / "gene_sets.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    manifests = {p.name: hash_file(p) for p in args.output.iterdir() if p.is_file()}
    (args.output / "SHA256SUMS").write_text("".join(f"{digest}  {name}\n" for name, digest in sorted(manifests.items())))
    print(json.dumps({"profiles": len(profiles), "states": len(states), "output": str(args.output)}))


if __name__ == "__main__":
    main()
