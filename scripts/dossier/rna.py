"""Read-only RNA matrix diagnostics, gene identity and record identity.

No normalization, filtering, or source mutation occurs here. Sparse canonicalization
is used only for diagnostic equality and never written back to an input file.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy import sparse


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while block := fh.read(16 << 20):
            digest.update(block)
    return digest.hexdigest()


def value_hash(value) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def decode_categories(categories, codes, name):
    if codes.dtype.kind not in "iu" or np.any(codes < -1) or np.any(codes >= len(categories)):
        raise ValueError(f"invalid_category_codes: {name}")
    return np.array([categories[c] if c >= 0 else None for c in codes], dtype=object)


def read_array(node):
    if isinstance(node, h5py.Dataset):
        a = node[:]
        if "categories" in node.attrs:
            return decode_categories(read_array(node.file[node.attrs["categories"]]), a, node.name)
        return node.asstr()[:] if a.dtype.kind in "OSU" else a
    encoding = node.attrs.get("encoding-type", "")
    if encoding == "categorical" or {"categories", "codes"} <= set(node):
        categories, codes = read_array(node["categories"]), node["codes"][:]
        return decode_categories(categories, codes, node.name)
    if "values" in node and "mask" in node:
        a = read_array(node["values"]).astype(object)
        a[node["mask"][:]] = None
        return a
    raise ValueError(f"Unsupported annotation encoding {encoding!r}: {node.name}")


def read_frame(group) -> pd.DataFrame:
    index_key = group.attrs.get("_index", "_index")
    frame = pd.DataFrame({k: read_array(group[k]) for k in group if k not in (index_key, "__categories")})
    frame.index = read_array(group[index_key])
    return frame


class RNAFile:
    """Bounded row reads for dense/CSR H5AD; CSC native column reads."""
    def __init__(self, path):
        self.path = Path(path)
        self.handle = h5py.File(path, "r")
        try:
            self.x = self.handle["X"]
            self.encoding = "dense" if isinstance(self.x, h5py.Dataset) else self.x.attrs.get("encoding-type")
            self.shape = tuple(int(x) for x in (self.x.shape if self.encoding == "dense" else self.x.attrs["shape"]))
            self.obs, self.var = read_frame(self.handle["obs"]), read_frame(self.handle["var"])
            if self.shape != (len(self.obs), len(self.var)):
                raise ValueError(f"axis_shape_mismatch: X={self.shape}, obs={len(self.obs)}, var={len(self.var)}")
            if self.encoding not in ("dense", "csr_matrix", "csc_matrix"):
                raise ValueError(f"Unsupported matrix encoding: {self.encoding}")
            self.indptr = None if self.encoding == "dense" else self.x["indptr"][:]
            if self.indptr is not None:
                expected = self.shape[0 if self.encoding == "csr_matrix" else 1] + 1
                if len(self.indptr) != expected or self.indptr[0] != 0 or np.any(np.diff(self.indptr) < 0):
                    raise ValueError("invalid_sparse_indptr")
                if self.indptr[-1] != len(self.x["data"]) or len(self.x["indices"]) != len(self.x["data"]):
                    raise ValueError("invalid_sparse_lengths")
        except Exception:
            self.handle.close()
            raise

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.handle.close()

    def blocks(self, chunk=2048):
        if self.encoding == "csc_matrix":
            raise ValueError("CSC requires native column diagnostics or an explicit analysis cache; row rescans prohibited")
        for start in range(0, self.shape[0], chunk):
            stop = min(start + chunk, self.shape[0])
            if self.encoding == "dense":
                matrix = sparse.csr_matrix(self.x[start:stop].astype(np.float64))
            else:
                lo, hi = self.indptr[start], self.indptr[stop]
                data = self.x["data"][lo:hi].astype(np.float64)
                indices = self.x["indices"][lo:hi]
                if np.any(indices < 0) or np.any(indices >= self.shape[1]):
                    raise ValueError("sparse_gene_index_out_of_bounds")
                matrix = sparse.csr_matrix((data, indices, self.indptr[start:stop + 1] - lo),
                                           shape=(stop - start, self.shape[1]))
            yield start, matrix


def quantiles(values):
    values = np.asarray(values)
    values = values[np.isfinite(values)]
    if not len(values):
        return None
    return dict(zip(("min", "p01", "p05", "p25", "median", "p75", "p95", "p99", "max"),
                    [float(v) for v in np.percentile(values, [0, 1, 5, 25, 50, 75, 95, 99, 100])]))


def scan(path, input_sha256, study_id, chunk=2048):
    """Return full numeric audit, independent per-record QC, and per-gene sums."""
    with RNAFile(path) as source:
        n, g = source.shape
        genes = [str(x) for x in source.var.index]
        axis_hash = value_hash(genes)
        qc = source.obs.copy().add_prefix("source_")
        qc.insert(0, "source_barcode", list(source.obs.index))
        qc.insert(0, "row_index", np.arange(n))
        qc.insert(0, "input_sha256", input_sha256)
        qc.insert(0, "study_id", study_id)
        qc.insert(0, "record_id", [value_hash([input_sha256, i]) for i in range(n)])
        qc["gene_axis_sha256"] = axis_hash
        totals, detected = np.empty(n), np.empty(n, dtype=np.int32)
        numeric_valid = np.empty(n, dtype=bool)
        row_hashes, issues = [], Counter()
        gene_sums, gene_detected = np.zeros(g), np.zeros(g, dtype=np.int64)
        stored_values = 0
        for start, matrix in source.blocks(chunk):
            d = matrix.data
            finite = np.isfinite(d)
            bad = ~finite | (d < 0) | (finite & (d != np.floor(d)))
            issues.update(nonfinite=int((~finite).sum()), negative=int((d < 0).sum()),
                          noninteger=int((finite & (d != np.floor(d))).sum()))
            stored_values += len(d)
            row_numbers = np.repeat(np.arange(matrix.shape[0]), np.diff(matrix.indptr))
            valid = np.bincount(row_numbers, weights=bad.astype(int), minlength=matrix.shape[0]) == 0
            numeric_valid[start:start + matrix.shape[0]] = valid
            matrix.sum_duplicates()
            matrix.eliminate_zeros()
            matrix.sort_indices()
            stop = start + matrix.shape[0]
            totals[start:stop] = np.asarray(matrix.sum(axis=1)).ravel()
            detected[start:stop] = np.asarray((matrix > 0).sum(axis=1)).ravel()
            gene_sums += np.asarray(matrix.sum(axis=0)).ravel()
            gene_detected += np.asarray((matrix > 0).sum(axis=0)).ravel()
            for i in range(matrix.shape[0]):
                lo, hi = matrix.indptr[i:i + 2]
                digest = hashlib.sha256()
                digest.update(axis_hash.encode())
                digest.update(matrix.indices[lo:hi].astype("<i8").tobytes())
                digest.update(matrix.data[lo:hi].astype("<f8").tobytes())
                row_hashes.append(digest.hexdigest())
        qc["computed_total_counts"], qc["computed_detected_genes"] = totals, detected
        qc["computed_numeric_valid"], qc["computed_count_sha256"] = numeric_valid, row_hashes
        summary = {"n_cells": n, "n_genes": g, "encoding": source.encoding,
                   "stored_values_checked": stored_values, "numeric_violations": dict(issues),
                   "all_counts_finite_nonnegative_integer": not any(issues.values()),
                   "zero_libraries": int((totals == 0).sum()), "invalid_cells": int((~numeric_valid).sum()),
                   "library_size": quantiles(totals), "detected_genes": quantiles(detected),
                   "gene_axis_sha256": axis_hash, "duplicate_gene_identifiers": len(genes) - len(set(genes)),
                   "source_obs_columns": list(source.obs), "source_var_columns": list(source.var),
                   "status": "completed"}
        var = source.var.copy()
        var.insert(0, "source_gene", genes)
        var["computed_sum"], var["computed_detected_cells"] = gene_sums, gene_detected
        return summary, qc.reset_index(drop=True), var.reset_index(drop=True)


def mapping_audit(genes, hgnc: pd.DataFrame, official_genes, source_ensembl=None):
    approved = hgnc[hgnc.status == "Approved"]
    direct = set(approved.symbol)
    aliases = defaultdict(set)
    for row in approved.itertuples(index=False):
        for field in ("alias_symbol", "prev_symbol", "ensembl_gene_id"):
            for value in str(getattr(row, field, "")).split("|"):
                if value and value != "nan":
                    aliases[value].add(row.symbol)
    rows = []
    official = set(official_genes)
    for index, gene in enumerate(genes):
        possibilities = {gene} if gene in direct else aliases.get(gene.split(".")[0], set())
        resolved = next(iter(possibilities)) if len(possibilities) == 1 else None
        original_id = str(source_ensembl[index]) if source_ensembl is not None else None
        id_candidates = aliases.get(original_id.split(".")[0], set()) if original_id else set()
        agreement = "source_ensembl_unavailable" if original_id is None else "id_unresolved" if not id_candidates else "symbol_unresolved" if not resolved else "consistent" if resolved in id_candidates else "conflict"
        rows.append({"source_gene": gene, "mapped_symbol": resolved,
                     "source_ensembl": original_id, "ensembl_candidates": "|".join(sorted(id_candidates)),
                     "symbol_vs_ensembl": agreement,
                     "mapping_status": "approved_symbol" if gene in direct else "unique_alias_or_ensembl" if resolved else "ambiguous" if possibilities else "unmapped",
                     "candidates": "|".join(sorted(possibilities)), "literal_in_official_axis": gene in official,
                     "mapped_in_official_axis": resolved in official if resolved else None})
    result = pd.DataFrame(rows)
    counts = Counter(x for x in result.mapped_symbol if pd.notna(x))
    result["many_to_one_mapping"] = [counts[x] > 1 if pd.notna(x) else False for x in result.mapped_symbol]
    return result


def duplicate_audit(cells):
    """Same barcode in different studies/batches is not duplicate evidence."""
    keys = ["study_id", "source_batch", "source_barcode"]
    repeated = cells[cells.duplicated(keys, keep=False)].copy()
    if repeated.empty:
        return pd.DataFrame(columns=keys + ["records", "classification"])
    rows = []
    for key, group in repeated.groupby(keys, dropna=False, sort=False):
        count_same = group.computed_count_sha256.nunique() == 1 and group.gene_axis_sha256.nunique() == 1
        labels_same = all(group[k].nunique(dropna=False) == 1 for k in ("source_target_gene", "source_guide_id") if k in group)
        rows.append(dict(zip(keys, key), records=len(group),
                         record_ids="|".join(group.record_id),
                         splits="|".join(group.split) if "split" in group else "unknown",
                         target=group.source_target_gene.iloc[0] if "source_target_gene" in group else None,
                         classification="content_and_label_identical" if count_same and labels_same else "identity_conflict"))
    return pd.DataFrame(rows)
