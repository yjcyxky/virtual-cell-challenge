"""Frozen QC predictor export, provenance and same-entry recovery boundaries."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy import sparse
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import submission
import model
from evaluation import generate_counts


def test_gate_rejects_changed_checkpoint_or_uncommitted_export(tmp_path):
    def write(path, value):
        p = tmp_path / path
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(value))
    def ref(path):
        return {'path': path, 'sha256': hashlib.sha256((tmp_path / path).read_bytes()).hexdigest()}
    (tmp_path / 'checkpoint').write_bytes(b'frozen shared qc')
    (tmp_path / 'export.py').write_text('frozen export')
    config = {'run_id': 'run', 'comparison_id': 'comparison', 'arms': ['linear'], 'seed': 1,
              'data': {}, 'generation': {}, 'checkpoint_ref': ref('checkpoint')}
    training = dict(config, representation={'kind': 'shared'}, training_selection={'kind': 'low_depth'})
    write('config.json', config)
    write('training.json', training)
    write('metrics.json', {'checkpoint_ref': ref('checkpoint'), 'evaluation_completed': True})
    node = {'status': 'completed', 'config_ref': ref('training.json'), 'code_refs': [],
            'metrics_ref': ref('metrics.json'), 'expected_config': training,
            'official_evaluation': {'comparison_id': 'comparison', 'config_ref': ref('config.json'),
                                    'code_refs': [ref('export.py')]}}
    write('docs/research/experiment_dag.json', {'nodes': {'run': node}})
    for args in [('init', '-q'), ('add', '.'), ('-c', 'user.name=Test', '-c', 'user.email=test@example.invalid', 'commit', '-qm', 'fixture')]:
        subprocess.run(['git', '-C', str(tmp_path), *args], check=True)
    assert submission.gate(tmp_path, tmp_path / 'config.json')[0] == config
    (tmp_path / 'checkpoint').write_bytes(b'changed')
    with pytest.raises(ValueError, match='checkpoint'):
        submission.gate(tmp_path, tmp_path / 'config.json')
    (tmp_path / 'checkpoint').write_bytes(b'frozen shared qc')
    (tmp_path / 'export.py').write_text('changed')
    node['official_evaluation']['code_refs'] = [ref('export.py')]
    write('docs/research/experiment_dag.json', {'nodes': {'run': node}})
    with pytest.raises(ValueError, match='committed'):
        submission.gate(tmp_path, tmp_path / 'config.json')


def test_export_uses_masked_predictor_and_replays_counts(tmp_path, monkeypatch):
    import data
    monkeypatch.setattr(data, 'ROOT', tmp_path)
    assert Path(model.__file__).resolve().parent == Path(submission.__file__).resolve().parent
    controls = ad.AnnData(sparse.csr_matrix([[2, 1, 3], [1, 7, 2]], dtype=np.float32),
                          obs=pd.DataFrame({'context': ['A']*2, 'target_gene': ['non-targeting']*2}, index=['c1', 'c2']),
                          var=pd.DataFrame(index=['g1', 'g2', 'g3']))
    official = tmp_path / 'official'; official.mkdir()
    controls.write_h5ad(official / 'context_A.h5ad')
    config = {'run_id': 'unit', 'seed': 1, 'generation': {'cells_per_target': 4},
              'data': {'chunk_rows': 2, 'target_sum': 10000}, 'max_nnz': 1000, 'max_counts_per_cell': 1000000}
    saved = {'genes': np.array(['g1', 'g2', 'g3']), 'targets': np.array(['g1']),
             'effects': np.array([[.2, -.1, 0]], np.float32), 'trained_mask': np.array([True, True, False])}
    directory = tmp_path / 'generated'; directory.mkdir()
    manifest = {'contexts': ['A'], 'per_context': {'A': {'control_cells': 2}}}
    targets = ['g1', 'g2']
    helper = submission.io.helpers()
    result = submission.generate(config, saved, directory, official, manifest, saved['genes'].tolist(), targets, 'linear', helper)
    delta = model.predict(saved, {}, targets, 'linear')
    assert np.all(delta[1] == 0) and np.all(delta[:, 2] == 0)
    expected = sparse.vstack([generate_counts(controls.X, delta[i], 1+i, 4, 10000) for i in range(2)])
    actual = ad.read_h5ad(directory / 'predictions.h5ad')
    np.testing.assert_array_equal(actual.X.toarray(), expected.toarray())
    assert actual.obs.target_gene.tolist() == ['g1']*4 + ['g2']*4
    assert result['model_kind'] == 'shared_low_depth_qc' and result['unseen_targets'] == ['g2']
    assert result['unseen_policy'].startswith('zero delta')
    assert submission.generate(config, saved, directory, official, manifest, saved['genes'].tolist(), targets, 'linear', helper) == result
    config['checkpoint_ref'] = {'path': 'test-checkpoint', 'sha256': '0'*64}
    submission.replay(config, saved, directory, official, manifest, targets)
    replay = json.loads((directory / 'export-replay.json').read_text())
    assert len(replay['checks']) == 2 and all(x['exact'] for x in replay['checks'])


def test_resume_reuses_published_entry_without_submission(tmp_path, monkeypatch):
    import data
    monkeypatch.setattr(data, 'ROOT', tmp_path)
    file = tmp_path / 'predictions.vcc'; file.write_bytes(b'frozen')
    seal = data.ref(file)
    (tmp_path / 'submission-file.json').write_text(json.dumps({'file_ref': seal}))
    (tmp_path / 'submission-entry.json').write_text(json.dumps({'entry_id': 'existing', 'model_name': 'qc', 'submission_sha256': seal['sha256']}))
    status = {k: .2 for k in ['score_pds', 'score_mse', 'score_nmae', 'score_fid', 'score_reach', 'score_jac', 'score_avg']}
    status.update(status='published', partition='val', panel_id='panel', anchor_version='anchor')
    calls = []
    def cli(config, args):
        calls.append(args)
        return status
    monkeypatch.setattr(submission.io, 'cli_json', cli)
    class Run:
        summary = {}
    result = submission.io.submit({'model_names': {'linear': 'qc'}, 'panel_id': 'panel'}, tmp_path, 'linear', Run())
    assert calls == [['status', 'existing'], ['status', 'existing']]
    assert result['entry_id'] == 'existing'
