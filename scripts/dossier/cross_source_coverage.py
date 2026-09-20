"""Coverage is measured feature presence, never an imputed expression value."""
from collections import Counter
import numpy as np
import pandas as pd

STATUS = [
    'measured_nonzero', 'measured_all_zero', 'measured_activity_unavailable',
    'outside_native_panel', 'official_identifier_unresolved',
    'ambiguous_or_conflicting_native_mapping', 'not_applicable_species_or_modality',
    'indeterminate_species_or_modality',
]


def gene_coverage(mapping, official, detected=None, applicable=True):
    """Require one conflict-free native feature; preserve literal coverage separately.

    ``detected`` is a file-wide count, not a per-task absence assertion. A shared
    native panel alone does not make two normalization denominators equivalent.
    """
    mapping = mapping.reset_index(drop=True)
    if detected is not None and len(detected) != len(mapping):
        raise ValueError('native_gene_order_length_mismatch')
    canonical = mapping.mapped_symbol
    conflict = mapping.get('symbol_vs_ensembl', pd.Series('', index=mapping.index)).eq('conflict')
    multiplicity = canonical.map(Counter(canonical.dropna())).fillna(0)
    safe = canonical.notna() & ~conflict & multiplicity.eq(1)
    # A conflicting Ensembl/symbol pair contaminates both candidate identities.
    uncertain = set(canonical[~safe].dropna())
    for field in ['candidates', 'ensembl_candidates']:
        if field in mapping:
            for value in mapping.loc[~safe, field].dropna():
                uncertain.update(str(value).split('|'))
    safe_map = {gene: i for i, gene in canonical[safe].items() if gene not in uncertain}
    literal = set(mapping.source_gene)
    if 'source_symbol' in mapping:
        literal.update(mapping.source_symbol.dropna())
    indices = official.mapped_symbol.map(safe_map).to_numpy(dtype=float,copy=True)
    observed = np.full(len(official), np.nan)
    present = np.isfinite(indices)
    if detected is not None:observed[present] = np.asarray(detected,dtype=float)[indices[present].astype(int)]
    states = np.full(len(official), STATUS[3], dtype=object)
    states[present] = STATUS[2]
    states[present & (observed>0)] = STATUS[0]
    states[present & (observed==0)] = STATUS[1]
    states[official.mapped_symbol.isna().to_numpy()] = STATUS[4]
    states[official.mapped_symbol.isin(uncertain).to_numpy()] = STATUS[5]
    if not applicable:states[:]=STATUS[7] if applicable is None else STATUS[6];indices[:]=np.nan;observed[:]=np.nan
    return pd.DataFrame({'official_gene':official.source_gene.to_numpy(),'canonical_symbol':official.mapped_symbol.to_numpy(),
                         'official_mapping_status':official.mapping_status.to_numpy(),'literal_native_presence':official.source_gene.isin(literal).to_numpy(),
                         'native_feature_index':indices,'status':states,'detected_records_in_declared_QC_scope':observed})


def target_coverage(panel, targets, tasks, applicable=True):
    """Keep source construct task multiplicity; no counting copies as new evidence."""
    rows = []
    all_counts = tasks.canonical_target.value_counts()
    qualified = tasks.loc[tasks.modality.eq('CRISPRi') & tasks.effect_status.eq('completed')]
    qualified_counts = qualified.canonical_target.value_counts()
    no_copy = qualified.loc[~qualified.confirmed_collection_copy].canonical_target.value_counts()
    de = qualified.loc[qualified.DE_status.eq('completed')].canonical_target.value_counts()
    for target in targets.itertuples(index=False):
        count = int(all_counts.get(target.mapped_symbol,0));n_qualified=int(qualified_counts.get(target.mapped_symbol,0))
        rows.append({'panel_id': panel, 'official_target': target.source_gene,
                     'canonical_target': target.mapped_symbol, 'candidate_construct_tasks': count,
                     'completed_CRISPRi_construct_tasks': n_qualified,
                     'completed_CRISPRi_tasks_after_confirmed_copy_exclusion': int(no_copy.get(target.mapped_symbol,0)),
                     'completed_DE_tasks': int(de.get(target.mapped_symbol,0)),
                     'status': 'indeterminate_species_or_modality' if applicable is None else 'not_applicable_species_or_modality' if not applicable else 'observed_response' if n_qualified else 'not_estimable' if count else 'no_observed_single_target_CRISPRi_response',
                     'independent_biological_replicates': None})
    return pd.DataFrame(rows)


def resolve_target(source_name, source_ensembl, approved, aliases):
    """Use intervention Ensembl identity only when actually provided by source.

    An RNA feature with a matching ambiguous label is not intervention identity.
    """
    name=str(source_name) if source_name is not None else ''
    candidates={name} if name in approved else aliases.get(name.split('.')[0],set())
    ids=aliases.get(str(source_ensembl).split('.')[0],set()) if source_ensembl else set()
    if ids and candidates and not ids.intersection(candidates):
        return None,'source_symbol_Ensembl_conflict'
    if len(ids)==1 and (not candidates or ids<=candidates):
        return next(iter(ids)),'unique_source_intervention_Ensembl_corroboration'
    if len(ids)>1:return None,'ambiguous_source_intervention_Ensembl'
    if len(candidates)==1:return next(iter(candidates)),'unique_source_symbol_or_alias'
    return None,'ambiguous_source_target_name' if candidates else 'source_target_unresolved'
