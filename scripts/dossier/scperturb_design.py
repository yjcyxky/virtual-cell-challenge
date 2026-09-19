"""Explicit per-study control roles and read-only response applicability.

Source labels stay in obs. These fields describe analysis decisions, never edits
to the source and never ground-truth guide assignment or biological replication.
"""
import json
import re
import numpy as np
import pandas as pd
from scperturb_cells import biological_background

MISSING={'','nan','none','na','unassigned','multiplet','no_reads_found','*'}
DEEP_PREFIXES=('Adamson','Gasperini','Schraivogel','Sunshine','TianKampmann2019','TianKampmann2021_CRISPRi','Xie','Xu','Replogle','Nadig')


def text(value):
    return '' if pd.isna(value) else str(value)


def tokens_are(value,pattern,separator=';'):
    parts=text(value).split(separator)
    return bool(parts) and all(re.fullmatch(pattern,p) for p in parts)


def gasperini_targets(value):
    value=text(value)
    pattern=r'chr(?:[0-9]+|X|Y)(?::[0-9]+-[0-9]+|\.[0-9]+_(?:top|second)_two)|[A-Za-z0-9.-]+_TSS|scrambled_[0-9]+'
    parts=re.findall(pattern,value)
    if not parts or '_'.join(parts)!=value:return None
    return sorted(set(p for p in parts if not p.startswith('scrambled_')))


def control_rule(file,row):
    p=text(row.get('perturbation'));g=text(row.get('guide_id'))
    if file.startswith('Adamson'):
        return (p in ['3x_neg_ctrl_pMJ144-1','3x_neg_ctrl_pMJ144-2'],
                'author_demo_explicit_triple_negative_construct' if '10X005' in file else 'negative_construct_crosswalk_not_verified')
    if file.startswith('Gasperini'):
        t=gasperini_targets(row.get('all_gene'));return t==[], 'explicit_source_scrambled_guides_only'
    if file.startswith('Schraivogel'):return p.startswith('non-targeting_'),'explicit_non_targeting_guide_not_zero_guide_control'
    if file.startswith('Xie'):return False,'literal_control_is_unassigned_GFP_reference_role_unverified'
    if file.startswith('Frangieh'):return tokens_are(g,r'.*_SITE_[0-9]+'),'all_explicit_intergenic_SITE_guides_only'
    if file.startswith('Norman'):return tokens_are(g,r'NegCtrl[0-9]+_NegCtrl[0-9]+'),'explicit_negative_guide_pairs_only'
    if file.startswith('TianKampmann2019'):return tokens_are(g,'control'),'explicit_source_control_guide_combinations'
    if file.startswith('TianKampmann2021'):return tokens_are(g,r'non-targeting[^;]*'),'explicit_non_targeting_guides'
    if file.startswith('Xu'):return tokens_are(g,r'NO-TARGET_[0-9]+'),'explicit_NO_TARGET_guides'
    if file.startswith('Sunshine'):
        good=row.get('good_coverage') in [True,'True','true'] and row.get('match_type') in ['exact_match','likely_match']
        return good and tokens_are(g,r'non-targeting_[^;]+'),'non_targeting_source_good_coverage_exact_or_likely_match'
    if file.startswith('Dixit'):return tokens_are(g,r'p_INTERGENIC[0-9]+'),'intergenic_cutting_control_not_non_targeting'
    if file.startswith('Papalexi'):return tokens_are(g,r'NTg[0-9]+'),'explicit_NTg_guide'
    if file.startswith('Santinha'):return g.startswith('Safe_H') and p=='control','safe_harbor_cutting_control_not_non_targeting'
    if file.startswith('Wessels'):return p=='control' and row.get('Guide.Class')=='NT','source_Cas13_non_targeting_guide_class'
    if file.startswith('Cui'):return p=='control' and row.get('cytokine_family')=='Control','source_PBS_reference'
    if file.startswith('Srivatsan') and 'sciplex2' in file:
        return p not in MISSING|{'control'} and text(row.get('dose_value')) in ['0','0.0'] and text(row.get('top_oligo'))!='','source_same_drug_nominal_zero_dose_not_unassigned_control'
    if file.startswith('Srivatsan') and 'sciplex4' in file:return p=='control' and row.get('perturbation_2')=='control','both_components_vehicle'
    if file.startswith('McFarland') and row.get('perturbation_type')=='CRISPR':return p=='sgLACZ','source_foreign_gene_LACZ_reference_not_untransduced'
    explicit={
        'Aissa':'source_GSM_control_xenograft','Chang':'source_untreated_sample','Datlinger':'source_CTRL_guide_mapping',
        'Lara':'source_NTC_guide_mapping','Liang':'source_Non_Targeting_guide_mapping','Lotfollahi':'source_DMSO_mapping',
        'McFarland':'source_vehicle_mapping','Nadig':'source_non_targeting_construct_mapping','Replogle':'source_non_targeting_construct_mapping',
        'Schiebinger':'source_serum_medium_reference_not_unperturbed_initial_state','Srivatsan':'source_Vehicle_mapping','Shifrut':'source_NonTarget_guide_mapping',
        'Zhao':'source_DMSO_or_none_within_same_sample_tissue'}
    for prefix,reason in explicit.items():
        if file.startswith(prefix):return p=='control',reason
    if file.startswith(('Gehring','Joung','Weinreb')):return False,'no_verified_matched_negative_reference_in_local_file'
    raise ValueError('unreviewed_scPerturb_control_rule:'+file)


def single_target(file,row,safe_symbols):
    p=text(row.get('perturbation'))
    if p.lower() in MISSING or p=='control':return None,'source_unassigned_or_control'
    if file.startswith('Gasperini'):
        parts=gasperini_targets(row.get('all_gene'))
        if parts is None:return None,'source_target_composition_unparsed'
        if len(parts)!=1:return None,'multiple_distinct_source_targets_or_no_target'
        p=parts[0]
        if p.endswith('_TSS'):p=p[:-4]
        else:return p,'single_source_regulatory_locus'
    elif file.startswith('Schraivogel'):
        m=re.match(r'(chr(?:[0-9]+|X|Y):[0-9]+-[0-9]+)_',p)
        if m:return m[1],'single_source_regulatory_locus'
        p=p.split('_')[0]
    elif file.startswith('Adamson'):
        known={'ATF6_only_pMJ145':'ATF6','PERK_only_pMJ146':'EIF2AK3','IRE1_only_pMJ148':'ERN1'}
        if '10X005' in file:
            if p not in known:return None,'combined_or_unverified_source_construct'
            p=known[p]
        else:p=p.split('_p')[0]
    elif file.startswith('Xie'):
        if re.fullmatch(r'chr(?:[0-9]+|X|Y);[0-9]+;[0-9]+;[+-]',p):return p,'single_source_regulatory_locus'
        return None,'multiple_or_unresolved_regulatory_targets'
    elif file.startswith('Sunshine'):
        if row.get('good_coverage') not in [True,'True','true'] or row.get('match_type') not in ['exact_match','likely_match']:
            return None,'source_guide_match_or_coverage_not_supported'
    # A single source symbol is required; inferred composite nperts is not used.
    if p in safe_symbols:return (safe_symbols[p] if isinstance(safe_symbols,dict) else p),'single_gene_source_target'
    if file.startswith(('Tian','Sunshine','Xu','Adamson','Replogle','Nadig','Gasperini')) and re.fullmatch(r'[A-Za-z][A-Za-z0-9.-]*',p):
        return p,'source_single_gene_label_not_resolved_in_HGNC'
    return None,'unresolved_or_multiple_source_target_genes'


def build_design(file,obs,safe_symbols,recovered=None):
    obs=obs.copy().reset_index(drop=True);n=len(obs)
    bio,bgs=biological_background(file,obs)
    technical=[]
    for prefix,fields in [
        ('Gasperini',['sample']),('Schraivogel',['replicate']),('Sunshine',['gem_group']),('Tian',['batch']),
        ('Xie',['batch','sample','replicate','filename']),('Replogle',['batch']),('Nadig',['batch']),('DatlingerBock2017',['replicate']),('Cui',['bio_replicate']),
        ('Lara',['sample']),('Liang',['sample']),('SrivatsanTrapnell2020_sciplex3',['plate','replicate']),
        ('SrivatsanTrapnell2020_sciplex4',['plate_id']),('Schiebinger',['replicate']),('Wessels',['10X_lane'])]:
        if file.startswith(prefix):technical=[c for c in fields if c in obs];break
    if file.startswith('Adamson'):
        if recovered is None:raise ValueError('Adamson_original_barcode_scope_required')
        if not np.array_equal(recovered.row_index.to_numpy(),np.arange(n)):raise ValueError('Adamson_recovery_row_order')
        # Use the published collection label, with a separately verified identity
        # concordance gate. Never replace it with a reconstructed guide call.
        obs['original_GEM_group']=recovered.original_GEM_group.to_numpy();technical=['original_GEM_group']
        if '10X005' in file:
            chemical=recovered.original_GEM_group.map({1:'tunicamycin',2:'thapsigargin',3:'DMSO'})
            if chemical.isna().any():raise ValueError('unreviewed_original_Adamson_chemical_GEM')
            keys=[json.dumps([bgs[int(b)]['background_key'],drug],separators=(',',':')) for b,drug in zip(bio,chemical)]
            bio,unique=pd.factorize(keys,sort=True)
            bgs=[{'background_index':i,'background_key':key,'source_fields':['collection_biological_background','original_GEO_GEM_author_chemical_mapping'],
                  'source_values':json.loads(key),'interpretation':'Recovered author chemical condition; per-cell inference retains its independently documented pooled endpoint background'} for i,key in enumerate(unique)]
    else:recovered=None
    results=[];deep=file.startswith(DEEP_PREFIXES)
    for i,row in enumerate(obs.to_dict('records')):
        control,rule=control_rule(file,row);target,target_kind=single_target(file,row,safe_symbols)
        reason=None
        if recovered is not None and not bool(recovered.source_guide_label_concordant.iloc[i]):
            control=False;target=None;reason='source_guide_label_differs_from_original_GEO_or_missing'
        p=text(row.get('perturbation'));known=p.lower() not in MISSING
        if p=='control' and not control:known=False;reason=reason or 'literal_control_not_verified_negative_reference'
        background=[bgs[int(bio[i])]['background_key'],[[c,text(row.get(c))] for c in technical]]
        if file.startswith('Srivatsan') and 'sciplex2' in file:background.append(['same_source_drug',p])
        if file.startswith('McFarland'):background.append(['source_intervention_mode',text(row.get('perturbation_type'))])
        condition_fields=[c for c in ['perturbation','dose_value','dose_unit','perturbation_2','dose_value_2'] if c in obs]
        condition=json.dumps([[c,text(row.get(c))] for c in condition_fields],ensure_ascii=False,separators=(',',':'))
        results.append({'row_index':i,'biological_background_index':int(bio[i]),'response_background':json.dumps(background,ensure_ascii=False,separators=(',',':')),
            'analysis_condition':condition,'source_perturbation_literal':p,'control_eligible':bool(control),'control_rule':rule,
            'condition_identity_supported':known,'response_scope_reason':reason,
            'single_target':target,'target_kind':target_kind,'deep_response_candidate':deep and target is not None and known,
            'deep_role':'CRISPRi_single_target' if deep else 'other_RNA_descriptive_role',
            'source_guide_for_consistency':text(row.get('guide_id',row.get('barcode',p))),
            'independent_biological_replicate':None,'truth_label':False})
    return pd.DataFrame(results),{'file':file,'biological_backgrounds':bgs,'technical_matching_fields':technical,
        'control_rules':sorted(set(r['control_rule'] for r in results)),'all_rows_retained':True,
        'control_status_counts':pd.Series([r['control_eligible'] for r in results]).value_counts().to_dict(),
        'limitations':['Technical matching does not establish independent cultures.','Source annotation and reference compatibility remain uncalibrated.']}
