"""Checkpoint identity, NTC-only deployment and strict submission count contracts."""
import numpy as np
import pandas as pd
import pytest
import torch
import anndata as ad
from scipy import sparse

from test_protocol import setup_model, configuration
from data import logcp
from state import context_features
from submission import restore_model, deployment_state, model_inputs, generate, audit_prediction


def controls_fixture(tmp_path):
    genes = [f'G{i}' for i in range(24)]
    official = tmp_path / 'controls'; official.mkdir()
    rng = np.random.default_rng(901)
    arrays = {}
    for context in ['A', 'B']:
        counts = rng.poisson(4, size=(16, 24)).astype(np.float32)
        arrays[context] = counts
        obs = pd.DataFrame({'context': context, 'target_gene': 'non-targeting',
                            'ntc_id': ['guide0', 'guide1'] * 8}, index=[str(i) for i in range(16)])
        ad.AnnData(sparse.csr_matrix(counts), obs=obs, var=pd.DataFrame(index=genes)).write_h5ad(official / f'context_{context}.h5ad')
    manifest = {'contexts': ['A', 'B'], 'cells_per_pert': 8,
                'per_context': {c: {'control_cells': 16} for c in ['A', 'B']}}
    return official, manifest, genes, arrays


def test_checkpoint_restore_is_exact_and_rejects_wrong_cycle(tmp_path):
    model, batch = setup_model()
    path = tmp_path / 'checkpoint.pt'
    torch.save({'model': model.state_dict(), 'config': configuration(), 'cycle': 4}, path)
    restored, _ = restore_model(path, 4, 'cpu')
    torch.manual_seed(21); expected = model.generate(batch)
    torch.manual_seed(21); actual = restored.generate(batch)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    with pytest.raises(AssertionError, match='cycle_mismatch'):
        restore_model(path, 7, 'cpu')


def test_deployment_preserves_trained_pca_and_matches_pooled_training_features():
    model, _ = setup_model()
    counts = np.random.default_rng(33).poisson(4, size=(600, 24)).astype(np.float32)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    state, _ = deployment_state(model, counts, configuration())
    coordinates = logcp(counts, model.common.numpy().astype(bool)) @ model.projection.numpy() - model.origin.numpy()
    expected, _ = context_features([counts], [coordinates], np.ones(24, bool), configuration())
    for key in state:
        np.testing.assert_allclose(state[key], expected[0][key], rtol=1e-6, atol=1e-6)
    assert all(torch.equal(value, model.state_dict()[key]) for key, value in before.items())
    batch = model_inputs(state, 3, 8, 'cpu')
    assert batch['target'].tolist() == [3] * 8 and batch['mask'].all()


def test_full_generation_has_exact_identity_counts_and_no_model_mutation(tmp_path):
    official, manifest, genes, _ = controls_fixture(tmp_path)
    model, _ = setup_model()
    before = {k: v.clone() for k, v in model.state_dict().items()}
    output = tmp_path / 'export'; output.mkdir()
    record = generate(model, configuration(), official, manifest, genes, genes[:2], output, 101, 4)
    audit = audit_prediction(output / 'predictions.h5ad', genes, genes[:2], ['A', 'B'], 8, 100000, 1000000)
    assert audit['cells'] == 32 and audit['target_contexts'] == 4
    assert audit['all_values_finite_nonnegative_integers']
    assert all(torch.equal(value, model.state_dict()[key]) for key, value in before.items())
    assert generate(model, configuration(), official, manifest, genes, genes[:2], output, 101, 4) == record


def test_over_limit_cells_are_rejected_without_clipping(tmp_path, monkeypatch):
    official, manifest, genes, _ = controls_fixture(tmp_path)
    model, _ = setup_model()
    monkeypatch.setattr(model, 'generate', lambda batch: torch.full_like(batch['base'], 1000000))
    output = tmp_path / 'export'; output.mkdir()
    with pytest.raises(AssertionError, match='official_count_limit'):
        generate(model, configuration(), official, manifest, genes, genes[:1], output, 101, 4)
    assert not (output / 'generation.json').exists()
    assert not (output / 'predictions.h5ad').exists()
