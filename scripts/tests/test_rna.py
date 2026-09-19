"""Behavioral tests on actual dense and sparse H5AD fixtures."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import h5py
import numpy as np
import pandas as pd
from scipy import sparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "dossier"))
from rna import scan, hash_file, mapping_audit, duplicate_audit, RNAFile
from render import render
from profile_arc_vcc2025_h1 import assess as h1_assess


def h5ad(path, values, encoding="csr", genes=("TP53", "GAPDH"), barcodes=None):
    values = np.asarray(values, dtype=float)
    barcodes = barcodes or [f"cell{i}" for i in range(len(values))]
    with h5py.File(path, "w") as f:
        f.attrs.update({"encoding-type": "anndata", "encoding-version": "0.1.0"})
        if encoding == "dense":
            f.create_dataset("X", data=values)
        else:
            m = sparse.csr_matrix(values)
            x = f.create_group("X")
            x.attrs.update({"encoding-type": "csr_matrix", "encoding-version": "0.1.0", "shape": values.shape})
            for key in ("data", "indices", "indptr"):
                x.create_dataset(key, data=getattr(m, key))
        for name, index in [("obs", barcodes), ("var", genes)]:
            g = f.create_group(name)
            g.attrs.update({"_index": "_index", "encoding-type": "dataframe", "encoding-version": "0.2.0"})
            g.create_dataset("_index", data=list(index), dtype=h5py.string_dtype())
        for name, values in {"batch": ["batch1"] * len(barcodes), "target_gene": ["non-targeting"] * len(barcodes),
                             "guide_id": ["NTC_A|NTC_B"] * len(barcodes)}.items():
            f["obs"].create_dataset(name, data=values, dtype=h5py.string_dtype())
        f["var"].create_dataset("gene_id", data=list(genes), dtype=h5py.string_dtype())


class RNATests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def assess(self, name, values, **kwargs):
        path = self.root / name
        h5ad(path, values, **kwargs)
        digest = hash_file(path)
        result = scan(path, digest, "study1", chunk=1)
        self.assertEqual(hash_file(path), digest)
        return result

    def test_dense_sparse_equal_diagnostics_and_hashes(self):
        dense, dc, dg = self.assess("dense.h5ad", [[0, 3], [2, 1], [0, 0]], encoding="dense")
        csr, cc, cg = self.assess("sparse.h5ad", [[0, 3], [2, 1], [0, 0]])
        self.assertEqual(dense["zero_libraries"], 1)
        self.assertTrue(dense["all_counts_finite_nonnegative_integer"])
        self.assertEqual(dc.computed_total_counts.tolist(), [3, 3, 0])
        self.assertEqual(dc.computed_detected_genes.tolist(), [1, 2, 0])
        self.assertEqual(dc.computed_count_sha256.tolist(), cc.computed_count_sha256.tolist())
        self.assertEqual(dg.computed_sum.tolist(), [2, 4])

    def test_bad_counts_after_first_chunk(self):
        report, cells, _ = self.assess("late.h5ad", [[1, 2], [-1, 0.5], [np.nan, np.inf]])
        self.assertFalse(report["all_counts_finite_nonnegative_integer"])
        self.assertEqual(report["numeric_violations"], {"nonfinite": 2, "negative": 1, "noninteger": 1})
        self.assertEqual(cells.computed_numeric_valid.tolist(), [True, False, False])

    def test_axis_size_mismatch_is_failure(self):
        path = self.root / "bad.h5ad"
        h5ad(path, [[1, 2]], genes=["TP53"])
        with self.assertRaisesRegex(ValueError, "axis_shape_mismatch"):
            scan(path, hash_file(path), "study1")

    def test_gene_order_participates_in_equality(self):
        _, a, _ = self.assess("a.h5ad", [[1, 2]])
        _, b, _ = self.assess("b.h5ad", [[1, 2]], genes=["GAPDH", "TP53"])
        self.assertNotEqual(a.computed_count_sha256[0], b.computed_count_sha256[0])
        a["split"], b["split"] = "a", "b"
        d = duplicate_audit(pd.concat([a, b]))
        self.assertEqual(d.classification.tolist(), ["identity_conflict"])

    def test_duplicate_evidence_distinguishes_identity_and_content(self):
        _, a, _ = self.assess("a.h5ad", [[1, 2]])
        _, b, _ = self.assess("b.h5ad", [[1, 2]])
        a["split"], b["split"] = "a", "b"
        self.assertEqual(duplicate_audit(pd.concat([a, b])).classification.tolist(), ["content_and_label_identical"])
        b["study_id"] = "independent_study"
        self.assertTrue(duplicate_audit(pd.concat([a, b])).empty)
        b["study_id"] = "study1"
        b["source_batch"] = "independent_batch"
        self.assertTrue(duplicate_audit(pd.concat([a, b])).empty)
        _, c, _ = self.assess("c.h5ad", [[1, 3]])
        c["split"] = "c"
        self.assertEqual(duplicate_audit(pd.concat([a, c])).classification.tolist(), ["identity_conflict"])

    def test_mapping_ambiguity_and_missing_not_zero(self):
        reference = pd.DataFrame({"status": ["Approved"] * 2, "symbol": ["TP53", "GAPDH"],
                                  "alias_symbol": ["P53|SHARED", "SHARED"], "prev_symbol": ["", ""],
                                  "ensembl_gene_id": ["ENSG1", "ENSG2"]})
        mapped = mapping_audit(["TP53", "P53", "SHARED", "ABSENT"], reference, ["TP53", "GAPDH"])
        self.assertEqual(mapped.mapping_status.tolist(), ["approved_symbol", "unique_alias_or_ensembl", "ambiguous", "unmapped"])
        self.assertEqual(mapped.many_to_one_mapping.tolist(), [True, True, False, False])
        self.assertTrue(pd.isna(mapped.mapped_symbol.iloc[3]))
        page = render({"tables": [{"title": "mapping", "rows": [{"mapped": None}]}]})
        self.assertIn('"mapped": null', page)
        self.assertIn("unknown / not estimable", page)

    def test_category_missing_is_not_last_category(self):
        path = self.root / "category.h5ad"
        h5ad(path, [[1, 2], [3, 4]])
        with h5py.File(path, "a") as f:
            del f["obs/batch"]
            g = f["obs"].create_group("batch")
            g.create_dataset("categories", data=["b0", "b1"], dtype=h5py.string_dtype())
            g.create_dataset("codes", data=[0, -1])
        with RNAFile(path) as r:
            self.assertEqual(r.obs.batch.iloc[0], "b0")
            self.assertTrue(pd.isna(r.obs.batch.iloc[1]))

    def test_full_h1_bundle_serializes_and_preserves_inputs(self):
        paths = []
        for split in ("Training", "Validation", "Test"):
            source = self.root / "data/raw/arc_vcc2025_h1"
            source.mkdir(parents=True, exist_ok=True)
            path = source / f"adata_{split}.h5ad"
            h5ad(path, [[1, 2], [3, 0]])
            paths.append(path)
            count_path = source / f"pert_counts_{split}.csv"
            count_path.write_text("target_gene,n_cells\nTP53,2\n")
            paths.append(count_path)
        hgnc = self.root / "data/raw/networks/hgnc_complete_set.txt"
        hgnc.parent.mkdir(parents=True)
        hgnc.write_text("status\tsymbol\talias_symbol\tprev_symbol\tensembl_gene_id\nApproved\tTP53\tP53\t\tENSG1\nApproved\tGAPDH\t\t\tENSG2\n")
        official = self.root / "data/raw/arc_vcc2026_controls/gene_names.csv"
        official.parent.mkdir(parents=True)
        official.write_text("gene_name\nTP53\nGAPDH\nUNMEASURED\n")
        paths.extend([hgnc, official])
        hashes = {str(p.relative_to(self.root)): hash_file(p) for p in paths}
        inventory = self.root / "inventory.json"
        inventory.write_text(json.dumps({"status": "completed", "bundle_id": "fixture", "file_results": [{"file": p, "sha256": h} for p, h in hashes.items()]}))
        with patch("profile_arc_vcc2025_h1.subprocess.check_output", return_value="fixture-commit"):
            result = h1_assess(self.root, self.root / "result", inventory, chunk=1)
        self.assertEqual(result["status"], "completed")
        loaded = json.loads((self.root / "result/report.json").read_text())
        self.assertEqual(loaded["duplicates"]["excess_identical_rows"], 4)
        self.assertEqual(loaded["duplicates"]["unique_records_after_confirmed_copy_accounting"], 2)
        axes = pd.read_parquet(self.root / "result/official_axis_coverage.parquet")
        self.assertTrue(axes.loc[axes.official_gene == "UNMEASURED", "observed_count_sum"].isna().all())
        self.assertEqual(hashes, {str(p.relative_to(self.root)): hash_file(p) for p in paths})
        self.assertTrue((self.root / "result/report.html").exists())


if __name__ == "__main__":
    unittest.main()
