"""Coverage is measured feature presence, never an imputed expression value."""
from collections import Counter
import numpy as np
import pandas as pd

STATUS = [
    'measured_nonzero', 'measured_all_zero', 'measured_activity_unavailable',
    'outside_native_panel', 'official_identifier_unresolved',
    'ambiguous_or_conflicting_native_mapping', 'not_applicable_species_or_modality',
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
    rows = []
    for row in official.itertuples(index=False):
        gene = row.mapped_symbol
        index = safe_map.get(gene)
        if not applicable:
            status = STATUS[6]
        elif pd.isna(gene):
            status = STATUS[4]
        elif gene in uncertain:
            status = STATUS[5]
        elif index is None:
            status = STATUS[3]
        elif detected is None or pd.isna(detected[index]):
            status = STATUS[2]
        else:
            status = STATUS[1] if detected[index] == 0 else STATUS[0]
        rows.append({'official_gene': row.source_gene, 'canonical_symbol': gene,
                     'official_mapping_status': row.mapping_status,
                     'literal_native_presence': row.source_gene in literal,
                     'native_feature_index': index if applicable else None,
                     'status': status,
                     'detected_records_in_declared_QC_scope': float(detected[index]) if applicable and index is not None and detected is not None else None})
    return pd.DataFrame(rows)


def target_coverage(panel, targets, tasks, applicable=True):
    """Keep source construct task multiplicity; no counting copies as new evidence."""
    rows = []
    for target in targets.itertuples(index=False):
        selected = tasks.loc[tasks.canonical_target.eq(target.mapped_symbol)]
        qualified = selected.loc[selected.modality.eq('CRISPRi') & selected.effect_status.eq('completed')]
        independent_source = qualified.loc[~qualified.confirmed_collection_copy]
        rows.append({'panel_id': panel, 'official_target': target.source_gene,
                     'canonical_target': target.mapped_symbol, 'candidate_construct_tasks': len(selected),
                     'completed_CRISPRi_construct_tasks': len(qualified),
                     'completed_CRISPRi_tasks_after_confirmed_copy_exclusion': len(independent_source),
                     'completed_DE_tasks': int(qualified.DE_status.eq('completed').sum()),
                     'status': 'not_applicable_species_or_modality' if not applicable else 'observed_response' if len(qualified) else 'not_estimable' if len(selected) else 'no_observed_single_target_CRISPRi_response',
                     'independent_biological_replicates': None})
    return pd.DataFrame(rows)
