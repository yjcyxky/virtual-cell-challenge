"""Zero-response decoder check on disjoint NTC; secondary, never an official score."""
import anndata as ad
import numpy as np
import pandas as pd
from scipy import stats
from data import NTC, normalized, ref, write_json
from evaluation import generate_counts
from vcc_mechanism.population import draw, dispersion


def contrast(x, y, alpha):
    difference = np.asarray(x.mean(0)-y.mean(0)).ravel()
    with np.errstate(invalid='ignore', divide='ignore'):
        p = stats.ttest_ind(x, y, axis=0, equal_var=False).pvalue
    p = np.nan_to_num(p, nan=1.)
    q = stats.false_discovery_control(p)
    return difference, p, q, {'mean_log_rms': float(np.sqrt(np.mean(difference**2))),
                            'welch_bh_discoveries': int((q < alpha).sum())}


def ntc_audit(output, config, decoder=None):
    spec = config['diagnostics']
    path = output/'cache/ntc-diagnostics.json'
    if path.exists():
        return
    ntc = ad.read_h5ad(output/'cache/H1-input.h5ad')
    real = ad.read_h5ad(output/'cache/H1-reference.h5ad', backed='r')
    indices = np.flatnonzero(real.obs.target_gene.eq(NTC))
    rng = np.random.default_rng(spec['seed'])
    chosen = np.sort(rng.choice(indices, config['generation']['cells_per_target'], replace=False))
    reference = real[chosen].to_memory().X; real.file.close()
    rows = np.random.default_rng(spec['seed']).integers(0, ntc.n_obs, config['generation']['cells_per_target'])
    templates = ntc.X[rows]
    if decoder is not None:
        generated = decoder(ntc)
    elif 'transport' in config:
        generated = generate_counts(ntc.X, np.zeros(ntc.n_vars), spec['seed'], config['generation']['cells_per_target'], config['data']['target_sum'])
    else:
        s = dict(np.load(output/'cache/moments/H1.npz'))
        profile = s['mean'][0]/s['mean'][0].sum()
        phi = dispersion(s, config['population'])[0] if config['population']['kind'] == 'negative_binomial' else None
        generated, _ = draw(ntc.X, profile, phi, spec['seed'], config)
    y = normalized(reference, config['data']['target_sum']).toarray()
    records, table = {}, {'gene': ntc.var_names}
    for name, matrix in [('resampled_input', templates), ('zero_response_decoder', generated)]:
        x = normalized(matrix, config['data']['target_sum']).toarray()
        delta, p, q, summary = contrast(x, y, spec['alpha'])
        summary.update(mean_library=float(np.asarray(matrix.sum(1)).mean()),
                       mean_detected=float(np.asarray((matrix > 0).sum(1)).mean()))
        records[name] = summary
        table.update({name+'_mean_log_difference': delta, name+'_p': p, name+'_q': q})
    csv = output/'cache/ntc-diagnostics.csv'; pd.DataFrame(table).to_csv(csv, index=False)
    write_json(path, {'diagnostic_only': True, 'official_scoring_formula_changed': False,
                     'test': 'Welch on cell log1p CPM10k; BH across all measured genes; not cell-eval2 DE score',
                     'scope': '400 generated/input and 400 disjoint scoring NTC; decoder drift, no biological replication claim',
                     'seed': spec['seed'], 'alpha': spec['alpha'], 'measured_genes': ntc.n_vars,
                     'results': records, 'per_gene_ref': ref(csv)})
