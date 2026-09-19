"""Source contracts for read-only CRISPRi assessment; constructs are interventions.

The Replogle protocol is registered in Issue #7. No source fields are overwritten.
"""
from collections import Counter
import numpy as np
import pandas as pd
from rna import RNAFile, quantiles

REPLOGLE = {
    'K562_essential': {'cell_line': 'K562', 'days_post_transduction': 6, 'library': 'essential',
                       'effector': 'dCas9-KRAB', 'expected_lineages': ['hematopoietic', 'immune']},
    'K562_gwps': {'cell_line': 'K562', 'days_post_transduction': 8, 'library': 'genome_wide',
                 'effector': 'dCas9-KRAB', 'expected_lineages': ['hematopoietic', 'immune']},
    'rpe1': {'cell_line': 'RPE1', 'days_post_transduction': 7, 'library': 'essential',
             'effector': 'ZIM3-KRAB-dCas9', 'expected_lineages': ['epithelial']},
}


def task_metadata(obs):
    required = ['gene', 'gene_id', 'transcript', 'gene_transcript', 'sgID_AB', 'gem_group']
    if not set(required) <= set(obs):
        raise ValueError('missing_construct_or_GEM_metadata')
    if obs[required].isna().any().any():
        raise ValueError('missing_task_identity')
    if obs.index.duplicated().any():
        raise ValueError('duplicate_source_barcode_within_experiment')
    rows = []
    for task, group in obs.groupby('gene_transcript', observed=True, sort=True):
        if any(group[k].nunique() != 1 for k in ['gene', 'gene_id', 'transcript', 'sgID_AB']):
            raise ValueError(f'construct_identity_conflict: {task}')
        row = group.iloc[0]
        components = str(row.sgID_AB).split('|')
        rows.append({'task': task, 'source_target_gene': row.gene, 'source_target_ensembl': row.gene_id,
                     'source_transcript': row.transcript, 'source_guide_id': row.sgID_AB,
                     'guide_components': len(components), 'independent_constructs': 1,
                     'is_control': row.gene == 'non-targeting', 'n_cells': len(group),
                     'GEM_groups': group.gem_group.nunique(), 'biological_replicates': None})
    return pd.DataFrame(rows)


def control_coverage(obs):
    control = obs.gene == 'non-targeting'
    baseline = obs.loc[control].gem_group.value_counts()
    rows = []
    for (task, gem), group in obs.groupby(['gene_transcript', 'gem_group'], observed=True, sort=True):
        n = int(baseline.get(gem, 0))
        rows.append({'task': task, 'source_batch': str(gem), 'target_cells': len(group),
                     'NTC_cells': n, 'matched': n > 0,
                     'DE_stratum_eligible': len(group) >= 2 and n >= 2,
                     'is_control': bool(control.loc[group.index[0]])})
    return pd.DataFrame(rows)


def bulk_comparison(singlecell_path, bulk_path, chunk=2048):
    """Compare every supplied pseudobulk row against local single-cell group means.

    Finite tolerance covers source float32 means; disagreement is evidence, not
    silently repaired. Rows absent from the single-cell artifact remain explicit.
    """
    with RNAFile(singlecell_path) as sc, RNAFile(bulk_path) as bulk:
        if list(sc.var.index) != list(bulk.var.index):
            raise ValueError('bulk_singlecell_gene_axis_mismatch')
        names = list(bulk.obs.index)
        if len(names) != len(set(names)):
            raise ValueError('duplicate_bulk_identity')
        lookup = {name: i for i, name in enumerate(names)}
        absent = set(sc.obs.gene_transcript) - lookup.keys()
        if absent:
            raise ValueError('singlecell_construct_missing_from_bulk')
        codes = np.array([lookup[x] for x in sc.obs.gene_transcript])
        sums = np.zeros(bulk.shape, dtype=np.float64)
        counts = np.bincount(codes, minlength=len(names))
        for start, matrix in sc.blocks(chunk):
            local = codes[start:start + matrix.shape[0]]
            # Sparse assignment aggregates all rows without expensive per-task raw scans.
            from scipy import sparse
            assignment = sparse.csr_matrix((np.ones(len(local)), (local, np.arange(len(local)))),
                                            shape=(len(names), len(local)))
            aggregate = (assignment @ matrix).tocoo()
            sums[aggregate.row, aggregate.col] += aggregate.data
        means = np.divide(sums, counts[:, None], out=np.full_like(sums, np.nan), where=counts[:, None] > 0)
        result = []
        violations = Counter(nonfinite=0, negative=0, noninteger=0)
        for start, matrix in bulk.blocks(chunk):
            actual = matrix.toarray()
            finite = np.isfinite(actual)
            violations.update(nonfinite=int((~finite).sum()), negative=int((actual < 0).sum()),
                              noninteger=int((finite & (actual != np.floor(actual))).sum()))
            for offset in range(len(actual)):
                i = start + offset
                expected = means[i]
                present = counts[i] > 0
                err = np.abs(actual[offset] - expected)
                result.append({'task': names[i], 'singlecell_records': int(counts[i]),
                    'source_num_cells_filtered': float(bulk.obs.num_cells_filtered.iloc[i]),
                    'source_num_cells_unfiltered': int(bulk.obs.num_cells_unfiltered.iloc[i]),
                    'count_agrees_filtered': bool(counts[i] == bulk.obs.num_cells_filtered.iloc[i]),
                    'comparison_status': 'completed' if present else 'not_estimable',
                    'reason': None if present else 'no_singlecell_rows_in_local_release',
                    'max_absolute_error_to_singlecell_mean': float(err.max()) if present else None,
                    'agrees_singlecell_mean_float32_tolerance': bool(np.allclose(actual[offset], expected, rtol=2e-5, atol=2e-5)) if present else None,
                    'bulk_row_total': float(actual[offset].sum()), 'independent_observation': False})
        frame = pd.DataFrame(result)
        return frame, {'rows': bulk.shape[0], 'genes': bulk.shape[1], 'values_checked': int(np.prod(bulk.shape)),
            'numeric_violations': dict(violations), 'semantic_class': 'source_reported_pseudobulk_continuous_expression',
            'local_singlecell_groups': int((counts > 0).sum()), 'groups_without_local_singlecell_records': int((counts == 0).sum()),
            'rows_matching_local_singlecell_mean': int(frame.agrees_singlecell_mean_float32_tolerance.fillna(False).sum()),
            'rows_matching_source_filtered_count': int(frame.count_agrees_filtered.sum()),
            'tolerance': {'rtol': 2e-5, 'atol': 2e-5},
            'mean_error_distribution': quantiles(frame.max_absolute_error_to_singlecell_mean),
            'status': 'completed', 'independent_biological_replicates': False}
