#!/usr/bin/env python
"""Index native Orion/CD4 counts after acquisition; no filtering, normalization or fitting.

Outputs are assessment artifacts. Fit-specific matrices and splits stay in Experiment outputs.
Each completed file is resumable only with identical source, code and axis identities.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from importlib.metadata import version
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy import sparse

from challenge import ChallengeIdentity, NTC
from rna import RNAFile, hash_file, quantiles, value_hash

ROOT = Path(__file__).resolve().parents[2]
SOURCES = ("xaira_orion", "zhu2026_cd4")


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def parquet_counts(batch, tokens):
    """Decode paired sparse lists, including sliced Arrow arrays and non-dense tokens."""
    columns = [batch.column(batch.schema.get_field_index(k)) for k in ("gene_token_id", "gene_expression")]
    if any(a.null_count for a in columns):
        raise ValueError("null_sparse_count_list")
    offsets = [a.offsets.to_numpy() for a in columns]
    if not np.array_equal(np.diff(offsets[0]), np.diff(offsets[1])):
        raise ValueError("unaligned_sparse_count_lists")
    values = [a.values.slice(int(o[0]), int(o[-1] - o[0])) for a, o in zip(columns, offsets)]
    if any(a.null_count for a in values):
        raise ValueError("null_sparse_count_value")
    arrays = [a.to_numpy(zero_copy_only=False) for a in values]
    indices = tokens.get_indexer(arrays[0])
    if (indices < 0).any():
        raise ValueError("unknown_gene_token")
    return sparse.csr_matrix((arrays[1].astype(np.float64, copy=True), indices, offsets[0] - offsets[0][0]),
                             shape=(len(batch), len(tokens)))


def count_qc(block):
    values = block.data
    if not np.isfinite(values).all() or (values < 0).any() or not np.equal(values, np.floor(values)).all():
        raise ValueError("invalid_raw_counts")
    block.sum_duplicates()
    block.eliminate_zeros()
    block.sort_indices()
    return np.asarray(block.sum(axis=1)).ravel(), np.diff(block.indptr)


def resolve_targets(names, ids, controls, identity):
    names, ids = names.fillna("").astype(str), ids.fillna("").astype(str)
    mapping = {(a, b): identity.resolve(a, b) for a, b in set(zip(names[~controls], ids[~controls]))}
    result = [(NTC, "explicit_source_control") if ctrl else mapping[(a, b)]
              for a, b, ctrl in zip(names, ids, controls)]
    return [x[0] for x in result], [x[1] for x in result]


def standardize(obs, source, context, identity, library):
    """Preserve source columns, expose explicit controls and guide-curated identities."""
    obs = obs.reset_index(drop=True)
    frame = obs.add_prefix("native__")
    frame["source_id"], frame["context"] = source, context
    if source == "xaira_orion":
        names = obs.gene_target.copy()
        ids = names.map(library)
        controls = names.eq("Non-Targeting")
        frame["donor"], frame["condition"] = "unknown", "baseline_culture"
        frame["batch"], frame["barcode"], frame["guide"] = obs["sample"], obs.cell_barcode, obs.guide_target
        frame["source_quality_pass"] = obs.pass_guide_filter.eq(1)
        frame["source_assignment_pass"] = obs.pass_guide_filter.eq(1)
        frame["target_annotation"] = "source_gene_target_with_design_library_ensembl"
    else:
        donor, condition = context.split("_", 1)
        frame["donor"], frame["condition"] = donor, condition
        frame["batch"], frame["barcode"], frame["guide"] = obs.lane_id, obs.source_barcode, obs.guide_id
        controls = obs.guide_type.eq("non-targeting")
        curated = obs.guide_id.map(library.target_gene_name)
        curated_ids = obs.guide_id.map(library.target_gene_id)
        names = curated.fillna(obs.perturbed_gene_name)
        ids = curated_ids.fillna(obs.perturbed_gene_id)
        frame["source_quality_pass"] = obs.low_quality.eq(False)
        frame["target_annotation"] = np.where(curated.notna(), "author_curated_guide", "original_cell_annotation")
        frame["source_assignment_pass"] = obs.guide_group.eq("targeting single sgRNA") & obs.guide_id.notna()
    frame["is_control"] = controls
    frame["target"], frame["target_mapping_reason"] = resolve_targets(names, ids, controls, identity)
    frame["physical_id"] = [value_hash([source, context, str(b), str(c)])
                             for b, c in zip(frame.batch, frame.barcode)]
    return frame


def summarize_file(path, source, relative, directory, identity, library, gene_metadata, source_sha):
    before = path.stat()
    directory.mkdir(parents=True, exist_ok=True)
    if source == "xaira_orion":
        context = path.name.split("_Batch")[0]
        native = pq.ParquetFile(path)
        names = [c for c in native.schema_arrow.names if c not in ("gene_token_id", "gene_expression")]
        obs = native.read(columns=names).to_pandas()
        genes = gene_metadata.copy()
        tokens = pd.Index(genes.gene_token_id)
        if tokens.has_duplicates:
            raise ValueError("duplicate_gene_token")
        def blocks():
            start = 0
            for batch in native.iter_batches(batch_size=1024, columns=["gene_token_id", "gene_expression"]):
                yield start, parquet_counts(batch, tokens)
                start += len(batch)
        reader = None
    else:
        context = path.name.removesuffix(".assigned_guide.h5ad")
        if not re.fullmatch(r"D[1-4]_(Rest|Stim8hr|Stim48hr)", context):
            raise ValueError("unknown_donor_state_file")
        reader = RNAFile(path)
        obs = reader.obs.copy()
        obs["source_barcode"] = obs.index.astype(str)
        genes = reader.var.rename(columns={"gene_ids": "ensembl_id"}).reset_index(drop=True)
        blocks = lambda: reader.blocks(2048)
    try:
        frame = standardize(obs, source, context, identity, library)
        if frame.physical_id.duplicated().any():
            raise ValueError("duplicate_physical_cell_within_file")
        mapping = identity.features(genes.gene_name.astype(str), genes.ensembl_id.astype(str))
        measured = mapping.loc[mapping.measured, "source_position"].to_numpy()
        n, g = len(frame), len(genes)
        totals, detected, matched = np.zeros(n), np.zeros(n, dtype=np.int32), np.zeros(n)
        gene_sums, gene_detected, checked = np.zeros(g), np.zeros(g, dtype=np.int64), 0
        for start, block in blocks():
            stop = start + block.shape[0]
            totals[start:stop], detected[start:stop] = count_qc(block)
            matched[start:stop] = np.asarray(block[:, measured].sum(axis=1)).ravel()
            gene_sums += np.asarray(block.sum(axis=0)).ravel()
            gene_detected += np.bincount(block.indices, minlength=g)
            checked += block.nnz
        frame["source_row"] = np.arange(n)
        frame["source_file"], frame["source_sha256"] = relative, source_sha
        frame["native_total_counts"], frame["official_measured_total_counts"] = totals, matched
        frame["detected_genes"] = detected
        frame["structural_count_support"] = (totals > 0) & (matched > 0)
        frame.to_parquet(directory / "cells.parquet", index=False, compression="zstd")
        genes["sum_counts"], genes["detected_cells"] = gene_sums, gene_detected
        genes.to_parquet(directory / "genes.parquet", index=False)
        mapping.to_parquet(directory / "gene_mapping.parquet", index=False)
        official = pd.DataFrame({"official_position": range(len(identity.official)), "gene_name": identity.official})
        official = official.merge(mapping.loc[mapping.measured, ["official_position", "source_position"]],
                                  on="official_position", how="left", validate="one_to_one")
        official["measured"] = official.source_position.notna()
        official.to_parquet(directory / "official_axis.parquet", index=False)
        tasks = frame.groupby(["context", "target", "is_control"], dropna=False, observed=True).agg(
            cells=("physical_id", "size"), guides=("guide", "nunique"), batches=("batch", "nunique"),
            quality_pass_cells=("source_quality_pass", "sum"), nonempty_cells=("structural_count_support", "sum"))
        tasks.reset_index().to_parquet(directory / "tasks.parquet", index=False)
        controls_by_batch = frame.groupby("batch", observed=True).is_control.sum()
        result = {"context": context, "file": relative, "source_sha256": source_sha,
                  "cells": n, "native_genes": g, "official_measured_genes": int(mapping.measured.sum()),
                  "ntc_cells": int(frame.is_control.sum()), "target_labels": int(frame.loc[~frame.is_control, "target"].nunique()),
                  "unresolved_target_cells": int(frame.target.isna().sum()), "nonzero_values_scanned": int(checked),
                  "finite_nonnegative_integer_counts": True, "zero_native_libraries": int((totals == 0).sum()),
                  "source_quality_pass_cells": int(frame.source_quality_pass.sum()),
                  "library_size": quantiles(totals), "detected_genes": quantiles(detected),
                  "batches_without_ntc": [str(x) for x in controls_by_batch[controls_by_batch == 0].index],
                  "source_total_counts_mismatches": int((~np.isclose(totals, obs.total_counts.to_numpy(), rtol=0, atol=0)).sum()),
                  "selection": "All acquired rows retained; quality, assignment and support are annotations, not training eligibility."}
    finally:
        if reader is not None:
            reader.handle.close()
    after = path.stat()
    if (before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns):
        raise ValueError("source_changed_during_scan")
    result["artifacts"] = {p.name: hash_file(p) for p in directory.glob("*.parquet")}
    return result


def ingest(source, output, inventory):
    report = json.loads(inventory.read_text())
    if report["status"] != "completed" or not report["registry_inputs_unchanged"]:
        raise ValueError("completed_immutable_inventory_required")
    records = {r["file"]: r for r in report["file_results"] if r["source_id"] == source}
    registered = next(s for s in json.loads((ROOT / "data/registry.lock.json").read_text())["sources"] if s["id"] == source)
    if len(records) != len(registered["files"]) or any(r["status"] != "completed" for r in records.values()):
        raise ValueError("incomplete_source_inventory")
    for entry in registered["files"]:
        relative = str(Path(registered.get("root", "data/raw")) / source / entry["name"])
        record = records.get(relative)
        if record is None or record["expected_bytes"] != entry["bytes"]:
            raise ValueError("inventory_does_not_match_current_source")
        current = (ROOT / relative).stat()
        if current.st_size != entry["bytes"] or current.st_mode & 0o222:
            raise ValueError("complete_frozen_source_required")
    if output.resolve().is_relative_to((ROOT / "data/raw" / source).resolve()):
        raise ValueError("assessment_must_not_modify_raw_inputs")
    output.mkdir(parents=True, exist_ok=True)
    base = ROOT / "data/raw" / source
    axis_path = ROOT / "data/raw/arc_vcc2026_controls/gene_names.csv"
    hgnc_path = ROOT / "data/raw/networks/hgnc_complete_set.txt"
    axis = pd.read_csv(axis_path).gene_name.tolist()
    identity = ChallengeIdentity(axis, pd.read_csv(hgnc_path, sep="\t", low_memory=False))
    if source == "xaira_orion":
        genes = pd.read_parquet(base / "metadata/gene_metadata.parquet")
        table = pd.read_csv(base / "metadata/guide_library.csv")
        id_column = next(c for c in ("target_gene_id", "ensembl_gene_id") if c in table)
        library = table.groupby("target_gene_name")[id_column].agg(lambda x: next(iter(set(x.dropna()))) if x.nunique() == 1 else "")
    else:
        genes = None
        library = pd.read_csv(base / "suppl_tables/sgrna_library_metadata.suppl_table.csv", low_memory=False).set_index("sgRNA")
        if library.index.has_duplicates:
            raise ValueError("ambiguous_curated_guide_mapping")
    code = {p.name: hash_file(p) for p in [Path(__file__), Path(__file__).with_name("rna.py"), Path(__file__).with_name("challenge.py")]}
    provenance = base / "SOURCE.json"
    if hash_file(provenance) != report["input_sha256"][str(provenance.relative_to(ROOT))]:
        raise ValueError("source_provenance_changed_since_inventory")
    binding = {"code": code, "axis_sha256": hash_file(axis_path), "hgnc_sha256": hash_file(hgnc_path),
               "inventory_sha256": hash_file(inventory), "source_id": source}
    complete = []
    for relative, record in sorted(records.items()):
        if not ((source == "xaira_orion" and "/data/" in relative and relative.endswith(".parquet")) or relative.endswith(".assigned_guide.h5ad")):
            continue
        path = ROOT / relative
        directory = output / "files" / path.stem
        marker = directory / "completed.json"
        expected = {**binding, "source_sha256": record["sha256"]}
        if marker.exists():
            result = json.loads(marker.read_text())
            if result["binding"] != expected or any(hash_file(directory / p) != h for p, h in result["artifacts"].items()):
                raise ValueError("completed_index_identity_changed")
        else:
            print(f"Index and scan {source}: {path.name}", flush=True)
            result = summarize_file(path, source, relative, directory, identity, library, genes, record["sha256"])
            result["binding"] = expected
            write_json(marker, result)
        complete.append(result)
    contexts = []
    targets = pd.read_csv(ROOT / "data/raw/arc_vcc2026_controls/pert_counts.csv").target_gene.tolist()
    for context in sorted({r["context"] for r in complete}):
        rows = [r for r in complete if r["context"] == context]
        frames = [pd.read_parquet(output / "files" / Path(r["file"]).stem / "cells.parquet",
                                  columns=["target", "is_control", "physical_id", "source_quality_pass", "source_assignment_pass", "structural_count_support"]) for r in rows]
        cells = pd.concat(frames, ignore_index=True)
        duplicates = int(cells.physical_id.duplicated().sum())
        if duplicates:
            raise ValueError(f"cross_file_physical_duplicates:{context}:{duplicates}")
        counts = cells.target.value_counts()
        usable = cells.loc[cells.source_quality_pass & cells.source_assignment_pass & cells.structural_count_support, "target"].value_counts()
        coverage = pd.DataFrame({"target_gene": targets})
        coverage["all_cells"] = coverage.target_gene.map(counts).fillna(0).astype(int)
        coverage["source_supported_cells"] = coverage.target_gene.map(usable).fillna(0).astype(int)
        coverage.to_csv(output / f"{context}-official-target-coverage.csv", index=False)
        contexts.append({"context": context, "files": len(rows), "cells": len(cells), "ntc_cells": int(cells.is_control.sum()),
                         "official_targets_observed": int((coverage.all_cells > 0).sum()),
                         "official_targets_50_source_supported_cells": int((coverage.source_supported_cells >= 50).sum()),
                         "official_targets_400_source_supported_cells": int((coverage.source_supported_cells >= 400).sum()),
                         "official_measured_genes": sorted({r["official_measured_genes"] for r in rows})})
    result = {"status": "completed", "binding": binding, "contexts": contexts, "files": len(complete),
              "completed_at": datetime.now(timezone.utc).isoformat(),
              "versions": {p: version(p) for p in ("numpy", "pandas", "scipy", "h5py", "pyarrow")},
              "support_definition": "Source quality and guide assignment pass, positive native and measured-axis counts. Diagnostic thresholds 50/400 do not define training eligibility or perturbation efficacy.",
              "scope": "Full raw-count validity and metadata/axis indices; no DE selection, filtering, normalization or train/test split; pooled DE artifacts are not used as cell-level labels."}
    write_json(output / "report.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-id", choices=SOURCES, required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        print(json.dumps(ingest(args.source_id, args.output, args.inventory), ensure_ascii=False))
    except Exception as error:
        if args.output.exists():
            write_json(args.output / "failure.json", {"status": "failed", "type": type(error).__name__, "error": str(error)})
        raise
