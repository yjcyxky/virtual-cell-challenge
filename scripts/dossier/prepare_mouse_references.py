#!/usr/bin/env python
"""Freeze species-matched mouse markers and explicitly orthology-derived RNA proxies."""
import argparse
from collections import defaultdict
from datetime import datetime,timezone
import json
from pathlib import Path
import shutil
import subprocess
import pandas as pd
from annotation import PROFILES
from rna import hash_file
from profile_responses import write_json


def one_to_one_orthologs(table):
    candidates=defaultdict(set);rows=[]
    for key,group in table.groupby('DB Class Key'):
        human=set(group.loc[group['NCBI Taxon ID']=='9606','Symbol']);mouse=set(group.loc[group['NCBI Taxon ID']=='10090','Symbol'])
        status='one_human_one_mouse' if len(human)==len(mouse)==1 else 'non_unique_or_missing_species_member'
        rows.append({'class_key':key,'human_symbols':sorted(human),'mouse_symbols':sorted(mouse),'status':status})
        if status=='one_human_one_mouse':candidates[next(iter(human))].add(next(iter(mouse)))
    mapping={h:next(iter(m)) for h,m in candidates.items() if len(m)==1}
    reverse=defaultdict(set)
    for h,m in mapping.items():reverse[m].add(h)
    return {h:m for h,m in mapping.items() if len(reverse[m])==1},rows


def mouse_symbol_index(table):
    symbols=set(table.loc[table['NCBI Taxon ID']=='10090','Symbol']);case=defaultdict(set)
    for symbol in symbols:case[symbol.upper()].add(symbol)
    return symbols,{key:next(iter(values)) for key,values in case.items() if len(values)==1}


def run(human,evidence,output):
    output.mkdir(parents=True,exist_ok=False)
    for line in (human/'SHA256SUMS').read_text().splitlines():
        digest,name=line.split('  ',1)
        if hash_file(human/name)!=digest:raise ValueError('human_reference_changed')
    source=evidence/'HOM_MouseHumanSequence.rpt';retrieval=json.loads((evidence/'retrieval.json').read_text())
    if hash_file(source)!=retrieval['sha256']:raise ValueError('MGI_evidence_changed')
    shutil.copytree(evidence,output/'MGI');shutil.copy2(human/'PanglaoDB_markers_27_Mar_2020.tsv.gz',output/'PanglaoDB_markers_27_Mar_2020.tsv.gz')
    shutil.copy2(human/'gene_sets.json',output/'human-source-gene-sets.json')
    table=pd.read_csv(source,sep='\t',dtype=str);orthologs,groups=one_to_one_orthologs(table);symbols,case=mouse_symbol_index(table)
    pd.DataFrame(groups).to_parquet(output/'all-homology-group-roles.parquet',index=False)
    pd.DataFrame(sorted(orthologs.items()),columns=['human_symbol','mouse_symbol']).to_parquet(output/'one-to-one-orthologs.parquet',index=False)
    pd.DataFrame({'symbol':sorted(symbols)}).to_parquet(output/'mouse-source-symbols.parquet',index=False)
    markers=pd.read_csv(human/'PanglaoDB_markers_27_Mar_2020.tsv.gz',sep='\t');markers=markers.loc[markers.species.str.contains('Mm',na=False)]
    profiles={};mapping=[]
    for name,(lineage,members) in PROFILES.items():
        chosen=markers.loc[markers['cell type'].isin(members)];genes=[]
        for symbol in sorted(set(chosen['official gene symbol'].dropna())):
            resolved=symbol if symbol in symbols else case.get(symbol.upper())
            genes.append(resolved if resolved else '__UNRESOLVED_MOUSE_MARKER__:'+symbol)
            mapping.append({'profile':name,'source_symbol':symbol,'mouse_symbol':resolved,'status':'exact_mouse_symbol' if symbol in symbols else 'unique_mouse_symbol_case_alignment' if resolved else 'unresolved_or_ambiguous','source':'PanglaoDB Mm rows'})
        profiles[name]={'lineage':lineage,'reference_types':members,'reference_types_found':sorted(chosen['cell type'].unique()),'genes':sorted(set(genes)),
            'reference':'https://panglaodb.se/markers/PanglaoDB_markers_27_Mar_2020.tsv.gz','species':'Mus musculus','unmapped_genes_remain_in_coverage_denominator':True}
    states={};source_states=json.loads((human/'gene_sets.json').read_text())['states']
    for name,state in source_states.items():
        if name=='pluripotency_marker':
            states[name]={'genes':profiles['pluripotent_like']['genes'],'reference':profiles['pluripotent_like']['reference'],'derivation':'mouse_PanglaoDB_markers'}
        else:
            genes=[orthologs.get(g,'__UNRESOLVED_MOUSE_ORTHOLOG__:'+g) for g in state['genes']]
            states[name]={'genes':sorted(set(genes)),'reference':state['reference'],'orthology_reference':retrieval['url'],'derivation':'strict_one_human_one_mouse_MGI_group','source_human_genes':state['genes'],
                'one_to_one_mapped_genes':sum(g in orthologs for g in state['genes']),'reference_total_genes':len(state['genes'])}
    pd.DataFrame(mapping).to_parquet(output/'marker-symbol-mapping.parquet',index=False)
    result={'profiles':profiles,'states':states,'species':'Mus musculus','human_markers_only':False,'code_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'code_sha256':hash_file(Path(__file__)),'human_reference_sha256':hash_file(human/'gene_sets.json'),'MGI_evidence':retrieval,'mouse_marker_source_rows':len(markers),
        'one_to_one_ortholog_pairs':len(orthologs),'created_at':datetime.now(timezone.utc).isoformat(),
        'limitations':['Healthy reference and analyst-defined types are not exhaustive or independently calibrated.',
            'RNA state proxies except mouse pluripotency markers are derived through frozen one-to-one orthology; pathway function is not established per cell.',
            'Unresolved or ambiguous marker and ortholog entries remain explicit reference coverage denominator members.',
            'MGI report is an orthology report, not an exhaustive mouse nomenclature or Ensembl identifier authority.']}
    write_json(output/'gene_sets.json',result)
    (output/'SHA256SUMS').write_text(''.join(f'{hash_file(p)}  {p.relative_to(output)}\n' for p in sorted(output.rglob('*')) if p.is_file()))
    return {'profiles':{k:len(v['genes']) for k,v in profiles.items()},'states':{k:(v.get('one_to_one_mapped_genes'),len(v['genes'])) for k,v in states.items()},'one_to_one_pairs':len(orthologs)}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ['human','evidence','output']:p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args();print(json.dumps(run(a.human,a.evidence,a.output)))
