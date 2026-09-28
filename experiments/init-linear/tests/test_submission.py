"""Export correctness boundaries; no official submission during tests."""
import json
import hashlib
from pathlib import Path
import sys
import subprocess
import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import submission
from data import normalized
from evaluation import generate_counts
from model import predict


def test_gate_blocks_changed_checkpoint_and_uncommitted_export(tmp_path):
    def ref(path):
        return {'path': path, 'sha256': hashlib.sha256((tmp_path / path).read_bytes()).hexdigest()}
    def write(path, value):
        p = tmp_path / path; p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(value))
    (tmp_path / 'checkpoint').write_bytes(b'complete fit')
    (tmp_path / 'export.py').write_text('original export')
    config = {'run_id': 'run', 'comparison_id': 'comparison', 'arms': ['linear', 'shared'],
              'seed': 1, 'data': {}, 'generation': {}, 'checkpoint_ref': ref('checkpoint')}
    write('config.json', config)
    write('training.json', config)
    write('metrics.json', {'checkpoint_ref': ref('checkpoint'), 'evaluation_completed': True})
    node = {'status': 'completed', 'config_ref': ref('training.json'), 'code_refs': [],
            'metrics_ref': ref('metrics.json'), 'expected_config': config,
            'official_evaluation': {'comparison_id': 'comparison', 'config_ref': ref('config.json'),
                                    'code_refs': [ref('export.py')]}}
    write('docs/research/experiment_dag.json', {'nodes': {'run': node}})
    for args in [('init', '-q'), ('add', '.'), ('-c', 'user.name=Test', '-c', 'user.email=test@example.invalid', 'commit', '-qm', 'fixture')]:
        subprocess.run(['git', '-C', str(tmp_path), *args], check=True)
    assert submission.gate(tmp_path, tmp_path / 'config.json')[0] == config
    (tmp_path / 'checkpoint').write_bytes(b'changed fit')
    with pytest.raises(ValueError, match='checkpoint'):
        submission.gate(tmp_path, tmp_path / 'config.json')
    (tmp_path / 'checkpoint').write_bytes(b'complete fit')
    (tmp_path / 'export.py').write_text('changed export')
    node['official_evaluation']['code_refs'] = [ref('export.py')]
    write('docs/research/experiment_dag.json', {'nodes': {'run': node}})
    with pytest.raises(ValueError, match='committed'):
        submission.gate(tmp_path, tmp_path / 'config.json')


def test_official_ntc_adapter_preserves_normalization_and_rejects_identity():
    controls = ad.AnnData(sparse.csr_matrix([[2, 1], [1, 7], [4, 5]], dtype=np.float32),
                          obs=pd.DataFrame({'context': ['A']*3, 'target_gene': ['non-targeting']*3}),
                          var=pd.DataFrame(index=['g1', 'g2']))
    cfg = {'chunk_rows': 2, 'target_sum': 10000}
    stats = submission.official_statistics(controls, ['g1', 'g2'], 'A', 3, cfg)
    np.testing.assert_allclose(stats['mean'][0], np.asarray(normalized(controls.X, 10000).mean(0)).ravel())
    for genes, context, cells in [(['g2', 'g1'], 'A', 3), (['g1', 'g2'], 'B', 3), (['g1', 'g2'], 'A', 4)]:
        with pytest.raises(ValueError):
            submission.official_statistics(controls, genes, context, cells, cfg)


def test_export_replays_generator_and_keeps_unseen_and_unmeasured_semantics(tmp_path, monkeypatch):
    controls = ad.AnnData(sparse.csr_matrix([[2, 1, 3], [1, 7, 2]], dtype=np.float32),
                          obs=pd.DataFrame({'context': ['A']*2, 'target_gene': ['non-targeting']*2}),
                          var=pd.DataFrame(index=['g1', 'g2', 'g3']))
    official = tmp_path / 'official'; official.mkdir()
    controls.write_h5ad(official / 'context_A.h5ad')
    config = {'run_id': 'unit', 'seed': 1, 'generation': {'cells_per_target': 4},
              'data': {'chunk_rows': 2, 'target_sum': 10000}, 'max_nnz': 1000, 'max_counts_per_cell': 1000000}
    model = {'genes': np.array(['g1', 'g2', 'g3']), 'targets': np.array(['g1']),
             'shared': np.array([[.2, -.1, 10]], np.float32), 'trained_mask': np.array([True, True, False])}
    from data import ROOT
    monkeypatch.setattr(submission, 'ROOT', tmp_path)
    import data
    monkeypatch.setattr(data, 'ROOT', tmp_path)
    # Load the existing streaming writer/audit from their real repository location.
    monkeypatch.setattr(submission, 'ROOT', ROOT)
    helper = submission.helpers()
    directory = tmp_path / 'generated'; directory.mkdir()
    manifest = {'contexts': ['A'], 'per_context': {'A': {'control_cells': 2}}}
    targets = ['g1', 'g2']
    result = submission.generate(config, model, directory, official, manifest, model['genes'].tolist(), targets, 'shared', helper)
    actual = ad.read_h5ad(directory / 'predictions.h5ad')
    stats = submission.official_statistics(controls, model['genes'], 'A', 2, config['data'])
    delta = predict(model, stats, targets, 'shared')
    assert np.all(delta[1] == 0) and np.all(delta[:, 2] == 0)
    expected = sparse.vstack([generate_counts(controls.X, delta[i], 1+i, 4, 10000) for i in range(2)])
    np.testing.assert_array_equal(actual.X.toarray(), expected.toarray())
    assert actual.obs.target_gene.tolist() == ['g1']*4 + ['g2']*4
    assert result['unseen_targets'] == ['g2'] and result['audit']['cells'] == 8
    assert submission.generate(config, model, directory, official, manifest, model['genes'].tolist(), targets, 'shared', helper) == result


def test_submit_rejects_mismatched_receipt_before_network(tmp_path, monkeypatch):
    import data
    monkeypatch.setattr(data, 'ROOT', tmp_path)
    file = tmp_path / 'predictions.vcc'; file.write_bytes(b'frozen')
    (tmp_path / 'submission-file.json').write_text(json.dumps({'file_ref': data.ref(file)}))
    (tmp_path / 'submission-entry.json').write_text(json.dumps({'entry_id': 'existing', 'model_name': 'model', 'submission_sha256': 'wrong'}))
    monkeypatch.setattr(submission, 'cli_json', lambda *a: pytest.fail('must not contact official server'))
    with pytest.raises(ValueError, match='receipt'):
        submission.submit({'model_names': {'linear': 'model'}}, tmp_path, 'linear', None)


def test_published_entry_is_reused_without_new_submission(tmp_path, monkeypatch):
    import data
    monkeypatch.setattr(data, 'ROOT', tmp_path)
    file = tmp_path / 'predictions.vcc'; file.write_bytes(b'frozen')
    seal = data.ref(file)
    (tmp_path / 'submission-file.json').write_text(json.dumps({'file_ref': seal}))
    (tmp_path / 'submission-entry.json').write_text(json.dumps({'entry_id': 'existing', 'model_name': 'model', 'submission_sha256': seal['sha256']}))
    status = {k: .2 for k in ['score_pds', 'score_mse', 'score_nmae', 'score_fid', 'score_reach', 'score_jac', 'score_avg']}
    status.update(status='published', partition='val', panel_id='panel', anchor_version='anchor')
    calls = []
    def cli(config, args):
        calls.append(args)
        return status
    monkeypatch.setattr(submission, 'cli_json', cli)
    class Run:
        summary = {}
    result = submission.submit({'model_names': {'linear': 'model'}, 'panel_id': 'panel'}, tmp_path, 'linear', Run())
    assert calls == [['status', 'existing'], ['status', 'existing']]
    assert result['entry_id'] == 'existing'
