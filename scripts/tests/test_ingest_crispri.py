"""Acquisition indices must retain cells, counts and unmeasured official slots."""
import sys
import tempfile
import unittest
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from scipy import sparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "dossier"))
from challenge import ChallengeIdentity
from ingest_crispri import count_qc, parquet_counts, summarize_file


class IngestTests(unittest.TestCase):
    def identity(self):
        hgnc = pd.DataFrame({"status": ["Approved"] * 3, "symbol": ["A", "B", "C"],
                             "prev_symbol": ["OLD_A", "", ""], "ensembl_gene_id": ["ENSG1", "ENSG2", "ENSG3"]})
        return ChallengeIdentity(["A", "B", "C"], hgnc)

    def test_sparse_arrow_slices_noncontiguous_tokens_and_invalid_values(self):
        batch = pa.RecordBatch.from_pydict({"gene_token_id": [[40], [90, 40, 40], []],
                                          "gene_expression": [[7.0], [3.0, 1.0, 2.0], []]})
        block = parquet_counts(batch.slice(1), pd.Index([40, 90]))
        totals, detected = count_qc(block)
        np.testing.assert_array_equal(block.toarray(), [[3, 3], [0, 0]])
        np.testing.assert_array_equal(totals, [6, 0])
        np.testing.assert_array_equal(detected, [2, 0])
        for value in [-1.0, 0.5, np.nan, np.inf]:
            with self.assertRaisesRegex(ValueError, "invalid_raw_counts"):
                count_qc(sparse.csr_matrix([[value]]))
        for ids, counts, message in [([[40]], [[None]], "null_sparse_count_value"),
                                      ([[8]], [[1.0]], "unknown_gene_token"),
                                      ([[40, 90]], [[1.0]], "unaligned_sparse_count_lists")]:
            bad = pa.RecordBatch.from_pydict({"gene_token_id": ids, "gene_expression": counts})
            with self.assertRaisesRegex(ValueError, message):
                parquet_counts(bad, pd.Index([40, 90]))

    def test_orion_full_axis_mask_and_no_cell_filtering(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "HCT116_Batch1.parquet"
            pq.write_table(pa.table({"gene_token_id": [[40], [90], []], "gene_expression": [[4.0], [8.0], []],
                                    "sample": ["HCT116_Batch1"] * 3, "cell_barcode": ["c1", "c2", "c3"],
                                    "gene_target": ["Non-Targeting", "OLD_A", "A"],
                                    "guide_target": ["NTC1|NTC2", "A1|A2", "A3|A4"],
                                    "pass_guide_filter": [1, 1, 0], "total_counts": [4.0, 8.0, 0.0]}), path)
            genes = pd.DataFrame({"gene_token_id": [40, 90], "gene_name": ["OLD_A", "B"], "ensembl_id": ["ENSG1", "ENSG2"]})
            output = root / "index"
            result = summarize_file(path, "xaira_orion", "data/raw/example.parquet", output,
                                    self.identity(), pd.Series({"OLD_A": "ENSG1", "A": "ENSG1"}), genes, "test")
            cells, axis = pd.read_parquet(output / "cells.parquet"), pd.read_parquet(output / "official_axis.parquet")
            self.assertEqual(cells.target.tolist(), ["non-targeting", "A", "A"])
            self.assertEqual(cells.source_quality_pass.tolist(), [True, True, False])
            self.assertEqual(cells.native_total_counts.tolist(), [4, 8, 0])
            self.assertEqual(axis.gene_name.tolist(), ["A", "B", "C"])
            self.assertEqual(axis.measured.tolist(), [True, True, False])
            self.assertTrue(pd.isna(axis.source_position.iloc[2]))
            self.assertEqual(result["source_total_counts_mismatches"], 0)
            self.assertEqual(result["zero_native_libraries"], 1)

    def test_cd4_curated_guides_and_multi_assignments_remain_distinct(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "D2_Stim8hr.assigned_guide.h5ad"
            obs = pd.DataFrame({"lane_id": ["lane1"] * 3, "guide_id": ["ntc-1", "old-guide", "multiple"],
                                "guide_type": ["non-targeting", "targeting", "targeting"],
                                "guide_group": ["targeting single sgRNA", "targeting single sgRNA", "multi sgRNA"],
                                "perturbed_gene_name": ["NTC", "B", "B"],
                                "perturbed_gene_id": ["", "ENSG2", "ENSG2"], "low_quality": [False, False, True],
                                "total_counts": [5.0, 9.0, 2.0]}, index=["c1", "c2", "c3"])
            genes = pd.DataFrame({"gene_name": ["A", "B"], "gene_ids": ["ENSG1", "ENSG2"]}, index=["g1", "g2"])
            ad.AnnData(X=sparse.csr_matrix([[5.0, 0.0], [1.0, 8.0], [0.0, 2.0]]), obs=obs, var=genes).write_h5ad(path)
            library = pd.DataFrame({"target_gene_name": ["A"], "target_gene_id": ["ENSG1"]}, index=["old-guide"])
            output = root / "index"
            result = summarize_file(path, "zhu2026_cd4", str(path), output, self.identity(), library, None, "test")
            cells = pd.read_parquet(output / "cells.parquet")
            self.assertEqual(result["cells"], 3)
            self.assertEqual(cells.target.tolist(), ["non-targeting", "A", "B"])
            self.assertEqual(cells.native__perturbed_gene_name.tolist(), ["NTC", "B", "B"])
            self.assertEqual(cells.source_assignment_pass.tolist(), [True, True, False])
            self.assertEqual(set(cells.donor), {"D2"})
            self.assertEqual(set(cells.condition), {"Stim8hr"})
            self.assertEqual(cells.native_total_counts.tolist(), [5, 9, 2])


if __name__ == "__main__":
    unittest.main()
