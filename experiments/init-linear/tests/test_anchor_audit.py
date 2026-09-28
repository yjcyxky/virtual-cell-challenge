"""Protocol audit refuses changed identities and reuses the official scorer exactly."""
import json
from pathlib import Path
import sys
import numpy as np
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import anchor_audit


def test_resume_refuses_a_different_baseline_recipe(tmp_path):
    path = tmp_path / 'identity.json'
    original = {'config_ref': {'sha256': 'first'}}
    path.write_text(json.dumps({'binding': original}))
    anchor_audit.verify_resume(path, original)
    with pytest.raises(ValueError, match='identity changed'):
        anchor_audit.verify_resume(path, {'config_ref': {'sha256': 'changed'}})


def test_official_rescaling_replays_saved_scores_and_rejects_other_reference():
    output = anchor_audit.EXPERIMENT / 'outputs/init-linear-s01'
    if not (output / 'metrics.json').exists():
        pytest.skip('requires preserved completed-run results')
    original = json.loads((output / 'metrics.json').read_text())['evaluation']['official_overlap']['linear']
    source = anchor_audit.ROOT / original['result_directory']
    meta = json.loads((source / 'run_meta.json').read_text())
    values, _ = anchor_audit.scores(source / 'aggregate.csv', meta, output / 'cache/bundle-official_overlap')
    np.testing.assert_allclose(values['Overall'], original['Overall'], rtol=0, atol=1e-12)
    for key in values['normalized']:
        np.testing.assert_allclose(values['normalized'][key], original['normalized'][key], rtol=0, atol=1e-12)
    with pytest.raises(ValueError):
        anchor_audit.scores(source / 'aggregate.csv', meta, output / 'cache/bundle-all')


def test_gate_requires_unchanged_sources_and_committed_stage(tmp_path):
    import hashlib
    import subprocess
    def write(path, value):
        p = tmp_path / path; p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(value))
        return {'path': path, 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}
    source = write('source.json', {'frozen': 1})
    config = write('config.json', {'run_id': 'run', 'comparison_id': 'compare', 'source_refs': [source], 'conditions': [{}]})
    code = write('code.py', 'original code')
    stage = {'status': 'registered', 'config_ref': config, 'code_refs': [code], 'comparison_id': 'compare', 'requires_evidence': ['prerequisite']}
    node = {'status': 'completed', 'anchor_audit': stage, 'config_ref': config, 'code_refs': [], 'metrics_ref': source}
    dag = {'nodes': {'run': node}, 'comparisons': {'compare': {'status': 'draft', 'decision_rule': {'primary_metric': 'Overall'}}}}
    write('docs/research/experiment_dag.json', dag)
    write('docs/research/evidence_ledger.json', {'evidence': {'prerequisite': {'state': 'not_supported'}}})
    for args in [('init', '-q'), ('add', '.'), ('-c', 'user.name=Test', '-c', 'user.email=test@example.invalid', 'commit', '-qm', 'fixture')]:
        subprocess.run(['git', '-C', str(tmp_path), *args], check=True)
    anchor_audit.gate(tmp_path, tmp_path / 'config.json')
    write('source.json', {'frozen': 2})
    with pytest.raises(ValueError, match='source.json'):
        anchor_audit.gate(tmp_path, tmp_path / 'config.json')
    write('source.json', {'frozen': 1})
    stage['code_refs'] = [write('code.py', 'changed code')]
    write('docs/research/experiment_dag.json', dag)
    with pytest.raises(ValueError, match='committed'):
        anchor_audit.gate(tmp_path, tmp_path / 'config.json')
