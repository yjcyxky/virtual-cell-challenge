"""Official control adaptation preserves the trained feature and gene semantics."""
import sys
from pathlib import Path
from types import SimpleNamespace

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from data import inspect_panel, prepare_context, NTC
from submission import helpers


def test_official_ntc_preparation_preserves_literal_slots_and_cpm(tmp_path):
    genes = ['Z', 'TIAF1', 'MYO18A']
    values = np.array([[2, 7, 1], [9, 2, 1], [2, 3, 9], [4, 8, 2]], dtype=np.uint16)
    obs = pd.DataFrame({'context': ['A'] * 4, 'target_gene': [NTC] * 4,
                        'ntc_id': ['g1', 'g1', 'g2', 'g2']}, index=list('abcd'))
    source = tmp_path / 'official.h5ad'
    ad.AnnData(sparse.csr_matrix(values), obs=obs, var=pd.DataFrame(index=genes)).write_h5ad(source)
    identity = SimpleNamespace(canonical=lambda name: name if name in genes[1:] else None)
    frame, mapping = inspect_panel(source, 'A', identity)
    assert mapping.valid.all()
    assert mapping.gene.tolist() == genes
    assert set(frame.target) == {NTC}
    assert frame.guide.tolist() == obs.ntc_id.tolist()
    assert set(frame.batch) == {'official'}
    config = {'seed': 17, 'minimum_target_cells': 50, 'lfc_epsilon': 1e-9,
              'ntc_augmentation_bags': 1, 'ntc_bag_cells': 4, 'reference_cells_per_target': 4}
    prepare_context('A', [(source, frame, mapping)], genes,
                    {g: i for i, g in enumerate(genes)}, config, tmp_path)
    with np.load(tmp_path / 'A/statistics.npz') as stats:
        axis = stats['gene_indices']
        np.testing.assert_allclose(stats['baseline_cpm'],
                                   (values / values.sum(1, keepdims=True) * 1e6).mean(0)[axis])
        assert stats['response'].shape == (0, 3)
        assert stats['targets'].size == 0
    np.testing.assert_array_equal(np.load(tmp_path / 'A/counts.npy'), values[:, axis])


def test_submission_audit_rejects_mislabelled_or_fractional_output(tmp_path):
    legacy = helpers()
    genes = ['G1', 'G2']
    targets = ['G1']
    obs = pd.DataFrame({'context': ['A', 'A'], 'target_gene': ['G1', 'G1']}, index=['a', 'b'])
    path = tmp_path / 'prediction.h5ad'
    data = ad.AnnData(sparse.csr_matrix(np.array([[1, 2], [3, 4]], dtype=np.int32)),
                     obs=obs, var=pd.DataFrame(index=genes))
    data.write_h5ad(path)
    assert legacy.audit_prediction(path, genes, targets, ['A'], 2, 10, 100)['cells'] == 2
    with pytest.raises(AssertionError):
        legacy.audit_prediction(path, genes[::-1], targets, ['A'], 2, 10, 100)
    data.X = sparse.csr_matrix(np.array([[1.5, 2], [3, 4]]))
    data.write_h5ad(path)
    with pytest.raises(AssertionError):
        legacy.audit_prediction(path, genes, targets, ['A'], 2, 10, 100)
