"""Explicit source hash interpretation for the two non-genetic McFaline screens."""
import json
import gzip
import re
import numpy as np
import pandas as pd


def read_hash_capture(path):
    """Decode exact five-field records, including the source's split final field.

    A four-field physical line may only be followed by a tab + integer UMI line.
    This documented read view retains source bytes and rejects all other wrapping.
    """
    rows=[];pending=None;wrapped=0;physical=0
    with gzip.open(path,'rt',newline='') as handle:
        for physical,line in enumerate(handle,1):
            fields=line.rstrip('\r\n').split('\t')
            if pending is not None:
                if len(fields)!=2 or fields[0]!='':raise ValueError('ambiguous_hash_count_continuation')
                fields=pending+[fields[1]];pending=None;wrapped+=1
            elif len(fields)==4:pending=fields;continue
            if len(fields)!=5 or not all(fields):raise ValueError('invalid_hash_capture_field_count')
            try:fields[3]=int(fields[3]);fields[4]=int(fields[4])
            except ValueError:raise ValueError('hash_axis_and_count_must_be_integer')
            if fields[4]<0:raise ValueError('negative_hash_UMI')
            rows.append(fields)
    if pending is not None:raise ValueError('truncated_hash_count_continuation')
    return pd.DataFrame(rows,columns=['sample','barcode','hash','axis','umi']),{'physical_lines':physical,'logical_records':len(rows),'split_final_field_records':wrapped,
        'interpretation':'lossless five-field read view; a four-field line followed by TAB+integer continues its UMI field; source bytes unchanged'}


def rt_cell_line(barcode):
    match=re.fullmatch(r'.+_RT_BC_(\d+)_Lig_BC_\d+',str(barcode))
    if not match:raise ValueError('unrecognized_combinatorial_barcode')
    well=int(match.group(1))
    return ['A172','T98G','U87MG'][(well-1)%12//4] if 1<=well<=96 else None


def parse_hash(label,group,line=None):
    fields=str(label).split('_')
    if group=='chemical3':
        if len(fields)!=7 or fields[5]!='rep':raise ValueError('invalid_GSC_hash')
        plate,well,line,drug,dose,_,rep=fields;dose=float(dose);tramet=0.;rep='rep_'+rep
        components=[] if dose==0 else [[drug,dose]]
    elif group=='chemical4':
        if len(fields)!=5 or line not in ['A172','T98G','U87MG']:raise ValueError('invalid_combination_hash_or_line')
        plate,tramet,drug,dose,rep=fields;dose=float(dose);tramet=float(tramet);well=None
        if drug=='vehicle' and dose!=0:raise ValueError('nonzero_vehicle_code')
        if drug.lower()=='trametinib' and tramet>0:raise ValueError('ambiguous_duplicate_trametinib_component')
        components=([[drug,dose]] if drug!='vehicle' and dose>0 else [])+([['Trametinib',tramet]] if tramet>0 else [])
    else:raise ValueError('unsupported_chemical_group')
    if not np.isfinite([dose,tramet]).all() or min(dose,tramet)<0:raise ValueError('invalid_chemical_dose')
    components=sorted(components)
    result={'source_hash_label':label,'hash_plate':plate,'hash_well':well,'cell_line':line,'replicate':rep,
        'source_hash_compound':drug,'source_hash_compound_dose':dose,'source_hash_trametinib_dose':tramet,
        'components':json.dumps(components,separators=(',',':')),'is_vehicle':not components,'dose_unit_from_primary_methods':'uM',
        'exposure_hours_from_primary_methods':72,'vehicle_percent_vv_from_primary_methods':.1 if group=='chemical3' else .2,
        'genetic_supervision_role':'not_applicable_no_genetic_intervention','source_labels_are_inferred':True}
    result['stratum']=json.dumps([line,plate,rep],separators=(',',':'))
    result['condition_key']=json.dumps([line,plate,rep,components],separators=(',',':'))
    return result


def design(metadata,sheet,group):
    labels=set(sheet['hash']);rows=[]
    for r in metadata.to_dict('records'):
        line=r.get('cell_type') if group=='chemical4' else r.get('GSC_line')
        parsed=parse_hash(r['top_oligo_W'],group,line)
        treatment='vehicle' if parsed['is_vehicle'] else 'Trametinib' if group=='chemical4' and parsed['source_hash_compound']=='vehicle' else parsed['source_hash_compound']
        dose=parsed['source_hash_trametinib_dose'] if group=='chemical4' and parsed['source_hash_compound']=='vehicle' and not parsed['is_vehicle'] else parsed['source_hash_compound_dose']
        consistent=(line==parsed['cell_line'] and str(r['source_list_key'])==line and r['hash_plate']==parsed['hash_plate'] and r['replicate']==parsed['replicate'] and r['treatment']==treatment and float(r['dose'])==dose)
        if group=='chemical4':consistent=consistent and rt_cell_line(r['source_barcode'])==line and float(r['trametinib_dose'])==parsed['source_hash_trametinib_dose']
        supported=pd.notna(r['hash_umis_W']) and r['hash_umis_W']>=5 and pd.notna(r['top_to_second_best_ratio_W']) and r['top_to_second_best_ratio_W']>=2.5
        reason='source_hash_not_in_sheet' if r['top_oligo_W'] not in labels else 'source_condition_disagreement' if not consistent else 'insufficient_source_hash_support' if not supported else None
        rows.append({'source_barcode':r['source_barcode'],**parsed,'condition_eligible':reason is None,'assignment_limitation':reason})
    return pd.DataFrame(rows)
