"""Use the frozen official DE implementation; never substitute a local test."""
import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from cell_eval2.de_compute import compute_de, _resolve_backend
from challenge import scorer_config

from .common import NTC


def annotated(matrix, genes, labels, prefix):
    matrix = sparse.csr_matrix(matrix, dtype=np.float32)
    if len(labels) != matrix.shape[0] or len(genes) != matrix.shape[1]:
        raise ValueError('Matrix labels do not match axes')
    return ad.AnnData(matrix, obs=pd.DataFrame({'target_gene': labels},
        index=[f'{prefix}-{i}' for i in range(matrix.shape[0])]), var=pd.DataFrame(index=genes))


def de_table(matrix, control, genes, labels, threads=8):
    cfg = scorer_config(device='cuda', num_threads=threads)
    backend = _resolve_backend(cfg.de.backend)
    if backend != 'gpudge':
        raise ValueError('This registered execution requires the gpudge backend')
    predicted = annotated(matrix, genes, labels, 'target')
    reference = annotated(control, genes, [NTC] * control.shape[0], 'reference')
    result = compute_de(predicted, backend=cfg.de.backend, groupby=cfg.pert_col,
        reference=reference, control_group=cfg.control, mean_calc=cfg.de.mean_calc,
        epsilon=cfg.de.epsilon, input_type=cfg.input_type, target_sum=cfg.target_sum,
        clip_value=cfg.de.clip_value, fdr_scope=cfg.de.fdr_scope,
        filter_gene_min_cpm_cell=cfg.filter.filter_gene_min_cpm_cell,
        threads=cfg.num_threads, device=cfg.device)
    return result, backend


def de_summary(table):
    frame = table.to_pandas()
    out = {}
    for target, group in frame.groupby('target', sort=True):
        group = group.loc[group.feature.ne(target)]
        significant = group.p_adj.lt(.05)
        out[str(target)] = {'tested_genes': len(group), 'significant_genes': int(significant.sum()),
            'up': int((significant & group.log2_fold_change.gt(0)).sum()),
            'down': int((significant & group.log2_fold_change.lt(0)).sum()),
            'median_absolute_lfc': float(group.log2_fold_change.abs().median()) if len(group) else None}
    return out
