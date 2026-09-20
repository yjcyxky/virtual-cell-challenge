#!/usr/bin/env python
"""Build all-source feature and task coverage from frozen assessment artifacts."""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import numpy as np
import pandas as pd
from cross_source_coverage import gene_coverage, target_coverage, STATUS
from profile_responses import write_json
from rna import hash_file, mapping_audit

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'data/assessments'


class Frozen:
    def __init__(self):
        self.indices = {}; self.used = {}

    def path(self, folder, name):
        folder = Path(folder)
        if folder not in self.indices:
            if (folder/'SHA256SUMS').exists():
                self.indices[folder] = {n: d for d, n in [line.split('  ', 1) for line in (folder/'SHA256SUMS').read_text().splitlines()]}
            else:
                report = json.loads((folder/'report.json').read_text())
                if report['status']!='completed':raise ValueError('unfinished_component')
                artifacts = report['artifacts']
                self.indices[folder] = dict(artifacts) if isinstance(artifacts,dict) else {r['file']:r['sha256'] for r in artifacts}
                self.indices[folder]['report.json'] = hash_file(folder/'report.json')
                self.used[str((folder/'report.json').relative_to(ROOT))] = self.indices[folder]['report.json']
        path = folder/name; expected = self.indices[folder].get(name)
        if expected is None or hash_file(path) != expected:
            raise ValueError('frozen_artifact_changed:' + str(path))
        self.used[str(path.relative_to(ROOT))] = expected
        return path

    def json(self, folder, name):
        return json.loads(self.path(folder, name).read_text())

    def parquet(self, folder, name):
        return pd.read_parquet(self.path(folder, name))


def build(output):
    started = time.monotonic(); output.mkdir(parents=True, exist_ok=False)
    frozen = Frozen(); panels = []; tasks = []; conditions = []; compact_codes = []
    official_path = ROOT/'data/raw/arc_vcc2026_controls/gene_names.csv'
    target_path = ROOT/'data/raw/arc_vcc2026_controls/pert_counts.csv'
    hgnc_path = ROOT/'data/raw/networks/hgnc_complete_set.txt'
    raw_refs = {str(p.relative_to(ROOT)): hash_file(p) for p in [official_path, target_path, hgnc_path]}
    hgnc = pd.read_csv(hgnc_path, sep='\t', low_memory=False)
    names = pd.read_csv(official_path).gene_name.tolist()
    official = mapping_audit(names, hgnc, names)
    targets = mapping_audit(sorted(set(pd.read_csv(target_path).target_gene)), hgnc, names)
    if len(official) != 18533 or len(targets) != 300 or targets.mapped_symbol.isna().any():
        raise ValueError('official_scope_changed')
    official.to_parquet(output/'official-identifiers.parquet', index=False)
    targets.to_parquet(output/'target-identifiers.parquet', index=False)
    (output/'gene-coverage').mkdir(); (output/'native-mappings').mkdir(); (output/'conditions').mkdir()
    mapping_cache = {}; target_cache = {}
    approved = hgnc.loc[hgnc.status.eq('Approved')]
    approved_symbols = set(approved.symbol); aliases = defaultdict(set)
    for row in approved.itertuples(index=False):
        for field in ['alias_symbol','prev_symbol','ensembl_gene_id']:
            for value in str(getattr(row,field,'')).split('|'):
                if value and value!='nan': aliases[value].add(row.symbol)

    def canonical(value):
        if value is None: return None
        value = str(value)
        if value not in target_cache:
            candidates = {value} if value in approved_symbols else aliases.get(value.split('.')[0],set())
            target_cache[value] = next(iter(candidates)) if len(candidates)==1 else None
        return target_cache[value]

    def add_panel(panel_id, family, context, folder, mapping_name, genes_name=None,
                  split=None, species='human', modality='RNA', applicable=True,
                  records=None, qc_scope=None, copy=False, metadata=None):
        mapping = frozen.parquet(folder, mapping_name)
        if split is not None and 'split' in mapping: mapping = mapping.loc[mapping.split.eq(split)].reset_index(drop=True)
        detected = None
        if genes_name:
            genes = frozen.parquet(folder, genes_name)
            if split is not None and 'split' in genes: genes = genes.loc[genes.split.eq(split)].reset_index(drop=True)
            if len(genes) != len(mapping): raise ValueError('native_QC_axis_length:' + panel_id)
            # Each mapping was generated in source feature order. Verify the
            # native key as well as length before attaching zero/nonzero QC.
            compatible = [('source_gene_id', 'source_gene_id'), ('source_gene_id', 'source_gene'), ('source_feature_id', 'source_gene'), ('source_ensembl_id', 'ensembl_id'), ('source_gene', 'source_gene')]
            aligned = False
            for a, b in compatible:
                if a in mapping and b in genes and mapping[a].astype(str).tolist() == genes[b].astype(str).tolist(): aligned = True; break
            if not aligned and 'source_symbol' in genes and mapping.source_gene.astype(str).tolist() == genes.source_symbol.astype(str).tolist(): aligned = True
            if not aligned: raise ValueError('native_QC_axis_order:' + panel_id)
            detected_column = next((x for x in ['computed_detected_cells', 'computed_detected_records'] if x in genes), None)
            if detected_column: detected = genes[detected_column].to_numpy()
        coverage = gene_coverage(mapping, official, detected, applicable and species == 'human')
        key = hashlib.sha256(panel_id.encode()).hexdigest()[:20]
        coverage.to_parquet(output/'gene-coverage'/(key+'.parquet'), index=False, compression='zstd')
        encoded = np.array([STATUS.index(x) for x in coverage.status], dtype=np.uint8)
        compact_codes.append(encoded)
        # Full feature mappings are content-addressed to avoid duplicating the
        # same 36,601-gene scBase axis in 1,808 presentation files.
        mapping_key = hashlib.sha256(pd.util.hash_pandas_object(mapping.astype(str), index=False).values.tobytes()).hexdigest()
        if mapping_key not in mapping_cache:
            mapping.to_parquet(output/'native-mappings'/(mapping_key+'.parquet'), index=False, compression='zstd'); mapping_cache[mapping_key] = True
        row = {'panel_id': panel_id, 'family': family, 'source_context': context,
               'species': species, 'modality': modality, 'human_RNA_applicable': applicable and species == 'human',
               'source_records': records, 'native_features': len(mapping),
               'official_literal_present': int(coverage.literal_native_presence.sum()),
               'official_safely_measured': int(coverage.status.isin(STATUS[:3]).sum()),
               'official_measured_all_zero': int(coverage.status.eq(STATUS[1]).sum()),
               'official_missing': int(coverage.status.eq(STATUS[3]).sum()),
               'official_unresolved': int(coverage.status.eq(STATUS[4]).sum()),
               'official_ambiguous_mapping': int(coverage.status.eq(STATUS[5]).sum()),
               'official_denominator': 18533, 'canonical_official_resolved': int(official.mapped_symbol.notna().sum()),
               'zero_QC_scope': qc_scope or panel_id if detected is not None else 'unavailable',
               'confirmed_collection_copy': copy, 'independent_biological_replicates': None,
               'gene_coverage_file': 'gene-coverage/'+key+'.parquet', 'native_mapping_file': 'native-mappings/'+mapping_key+'.parquet',
               'source_folder': str(folder.relative_to(ROOT)), 'source_mapping_file': mapping_name,
               'source_mapping_sha256': frozen.indices[folder][mapping_name],
               'source_metadata': metadata or {}}
        panels.append(row)
        return row

    def add_tasks(panel_id, folder, name, modality='CRISPRi', copy=False, transform=None):
        data = frozen.json(folder, name)
        for index, task in enumerate(data):
            if transform:
                extra = transform(task)
                if extra is None: continue
            else: extra = {}
            target = extra.get('target', task.get('target', task.get('target_gene')))
            result = task.get('gene_results')
            if isinstance(result, dict): result = result['file']
            result = str((Path(name).parent/result).as_posix()) if result else None
            effect_status = task['status']; matched = task.get('matching', {})
            tasks.append({'panel_id': extra.get('panel_id', panel_id),
                'source_task': task.get('task', task.get('task_id', str(index))),
                'source_target': target, 'canonical_target': canonical(target),
                'source_construct': task.get('source_transcript', task.get('task')),
                'source_guide': task.get('source_guide_id'), 'source_background_index': task.get('biological_background_index'),
                'source_condition': extra.get('condition', task.get('condition', task.get('context', task.get('split')))),
                'modality': extra.get('modality', modality), 'target_kind': task.get('target_kind', 'source_single_target_construct'),
                'effect_status': effect_status, 'reason': task.get('reason'),
                'DE_status': task.get('DE', {}).get('status', 'not_estimable'),
                'source_target_cells': task.get('n_source_candidate_cells', task.get('observed_primary_target_cells', task.get('n_cells'))),
                'matched_target_cells': task.get('n_matched_target_cells', task.get('matched_primary_target_cells', matched.get('target_cells_used', task.get('n_cells')))),
                'matched_control_cells': matched.get('control_cells', task.get('control_cells')),
                'confirmed_collection_copy': copy, 'independent_biological_replicates': None,
                'source_folder': str(folder.relative_to(ROOT)), 'source_task_file': name,
                'source_task_file_sha256': frozen.indices[folder][name], 'source_gene_result_file': result,
                'source_gene_result_sha256': frozen.indices[folder].get(result),
                'downstream_RMS': task.get('downstream_RMS'),
                'target_RNA_ratio': task.get('target_RNA', {}).get('RNA_ratio')})

    def add_conditions(family, folder, name, interpretation):
        path = frozen.path(folder, name); key = str(len(conditions)).zfill(3)+'-'+path.name
        shutil.copy2(path, output/'conditions'/key)
        count = len(pd.read_parquet(path)) if path.suffix == '.parquet' else len(json.loads(path.read_text()))
        conditions.append({'family': family, 'source_folder': str(folder.relative_to(ROOT)), 'source_file': name,
                           'file': 'conditions/'+key, 'sha256': hash_file(path), 'rows': count, 'interpretation': interpretation})

    h1 = BASE/'h1-structure-20260919-v3'; h1r = BASE/'h1-response-20260919'
    for split, records in [('Training',221273), ('Validation',98927), ('Test',170846)]:
        add_panel('H1:'+split, 'H1', split, h1, 'gene_mapping.parquet', 'genes.parquet', split=split, records=records,
                  metadata={'cell_line':'H1','source_split':split,'shared_NTC_records':38176,'unique_global_source_records':414694,'source_issue':3})
    add_tasks(None, h1r, 'tasks.json', transform=lambda t:{'panel_id':'H1:'+t['split']})
    official_folder = BASE/'official-controls-20260919'
    for c in ['A','B','C']:
        add_panel('official:'+c, 'official_controls', c, official_folder, 'gene_mapping.parquet', 'gene_qc_'+c+'.parquet', records=18400,
                  metadata={'anonymous_context':c,'actual_perturbation_responses':False,'source_issue':6})
    for family, contexts in [('replogle',['K562_essential','K562_gwps','rpe1']), ('nadig',['hepg2','jurkat'])]:
        structure = BASE/(family+'-structure-20260919')
        for context in contexts:
            add_panel(family+':'+context, family, context, structure, context+'/gene_mapping.parquet', context+'/genes.parquet',
                      metadata={'source_issue':7 if family=='replogle' else 8,'protocol_contract':'scripts/dossier/crispri.py; exact effector/time/culture retained in source dossier','subline_or_passage':None})
            add_tasks(family+':'+context, BASE/(family+'-response-'+context+'-20260919'), 'tasks.json')
    jiang = BASE/'jiang-dossier-20260919-v2'
    contexts = frozen.parquet(jiang, 'context-coverage.parquet')
    for row in contexts.itertuples(index=False):
        _, line, stimulus = row.context.split('__'); pid = row.context
        source_stimulus = {'TGFB1':'TGFB'}.get(stimulus,stimulus)
        add_panel(pid, 'Jiang', pid, jiang, pid+'/gene-mapping.parquet', 'source-'+source_stimulus+'/genes.parquet', records=row.n_cells,
                  qc_scope='entire_source_stimulus:'+stimulus,
                  metadata={'cell_line':line,'stimulation':stimulus,'stimulation_hours':24,'dose':None,'source_issue':9,'evidence_issue':25,'subline_or_passage':None})
        add_tasks(pid, jiang, pid+'/tasks.json')
    add_conditions('Jiang', jiang, 'all-task-stratum-design.parquet', 'Every source context × target × technical matching stratum; not independent culture count')
    gxe1 = BASE/'mcfaline-gxe1-dossier-20260919-v2'
    add_panel('GxE1', 'McFaline_GxE1', 'A172 × source effector/library/drug/dose', gxe1, 'gene-mapping.parquet', 'genes.parquet', records=18588,
              metadata={'hours':96,'source_issue':10,'evidence_issue':24,'source_labels_and_raw_counts_unchanged':True})
    add_tasks('GxE1', gxe1, 'tasks.json', transform=lambda t:{'target':t.get('intervention_target'), 'modality':t.get('effector') if t.get('role')=='genetic_response' else 'chemical_fixed_source_genotype'})
    add_conditions('GxE1', gxe1, 'condition-coverage.json', 'All declared genetic contexts including absent combinations; source drug doses in uM')
    gxe2 = BASE/'mcfaline-gxe2-dossier-20260919'
    gr = frozen.json(gxe2, 'genetic/report.json')
    for row in gr['contexts']:
        pid='GxE2:'+row['context']; line,drug,dose=json.loads(row['context'])
        add_panel(pid, 'McFaline_GxE2', row['context'], gxe2, 'genetic/gene-mapping.parquet', 'genetic/genes.parquet', records=row['n_cells'], qc_scope='entire_GxE2_source_CDS',
                  metadata={'cell_line':line,'drug':drug,'dose_uM':dose,'hours':72,'NTC_cells':row['n_NTC'],'effector':'dCas9-BFP-KRAB','source_issue':11,'evidence_issue':26,'subline_or_passage':None})
        add_tasks(pid, gxe2, 'genetic/'+row['directory']+'/tasks.json')
    add_conditions('GxE2', gxe2, 'all-chemical-tasks.parquet', 'All fixed-source-genotype drug contrasts; distinct from matched-NTC genetic response')
    for group,records in [('chemical3',179576),('chemical4',266662)]:
        folder = BASE/('mcfaline-'+group+'-dossier-20260919-v2')
        add_panel('McFaline:'+group, 'McFaline_'+group, 'all source cell lines × compound/dose/replicate', folder, 'gene-mapping.parquet', 'genes.parquet', records=records,
                  metadata={'hours':72,'genetic_supervision':False,'source_issue':12,'cell_lines': ['GBM4','GBM8','GSC0131','GSC0827'] if group=='chemical3' else ['A172','T98G','U87MG']})
        add_conditions(group, folder, 'condition-diagnostics.parquet', 'All declared chemical comparisons; missing combinations and vehicle references kept')
    sc = BASE/'scperturb-dossier-20260919'
    cr = frozen.json(sc, 'cells/report.json'); index = {r['file']:r for r in frozen.json(sc,'file-index.json')}
    for row in cr['files']:
        file = row['file']; name=file.removesuffix('.h5ad'); copied=index[file]['exact_source_copy_verified']
        audit = frozen.json(sc, 'source-audit/'+file+'.json')
        methods = frozen.json(sc, 'design/'+name+'/methods.json')
        add_panel('scPerturb:'+name, 'scPerturb', name, sc, 'cells/'+name+'/gene-mapping.parquet', 'cells/'+name+'/genes.parquet',
                  records=row['cells'], species=row['species'], copy=copied,
                  metadata={'study_id':audit['study_id'],'provenance':audit['provenance'],'intervention':audit['intervention_source_table'],
                            'biological_backgrounds':methods['biological_backgrounds'],'control_rules':methods['control_rules'],
                            'technical_matching_fields':methods['technical_matching_fields'],'expression_scale':row['expression_scale'],
                            'source_issue':14,'evidence_issues':[23,27]})
        name_task='response/'+name+'/deep-response/tasks.json'
        if name_task in frozen.indices[sc]: add_tasks('scPerturb:'+name, sc, name_task, copy=copied)
        add_conditions('scPerturb:'+name, sc, 'response/'+name+'/descriptions/condition-contrasts.parquet', 'Every observed source condition × matching stratum; includes non-CRISPRi and unassigned records with eligibility reasons')
    provenance = BASE/'scbase-provenance-20260919-v4'; expression = BASE/'scbase-expression-20260919-v2'
    catalog = frozen.json(BASE/'cross-source-identities-20260919-v2','scbase-source-catalog.json')
    for i,row in enumerate(catalog):
        accession=row['experiment_accession']; pid='scBase:'+accession
        add_panel(pid, 'scBaseCount', row['source_sample'], expression, accession+'/gene_mapping.parquet', accession+'/genes.parquet',
                  species='human' if row['RNA_eligible'] else str(row['source_species']), applicable=row['RNA_eligible'], records=row['stored_records'],
                  metadata={'source_study':row['source_study'],'source_sample':row['source_sample'],'source_species':row['source_species'],
                            'source_cell_line':row['source_cell_line'],'source_library_name':row['source_library_name'],
                            'author_library_relation':row['author_library_relation'],'per_cell_perturbation_labels':'unavailable','source_issue':16,'evidence_issue':22})
        if (i+1)%100==0: print('scBase coverage '+str(i+1)+'/1808',flush=True)
    tahoe = BASE/'tahoe-dossier-20260919-v2'
    add_panel('Tahoe', 'Tahoe', '14 source plates × 50 cell lines × sample/compound/dose', tahoe, 'counts/gene-mapping.parquet', 'counts/gene-coverage.parquet', records=8467330,
              metadata={'hours':24,'plate14_author_biological_repeat_of':6,'genetic_supervision':False,'source_issue':17})
    add_conditions('Tahoe', tahoe, 'all-condition-diagnostics.parquet', 'All 65,576 observed sample × cell line combinations; genetic CRISPRi supervision not applicable')
    # Auxiliary evidence remains in its source-specific measurement class.
    priors=BASE/'priors-dossier-20260919-v4'; prior_report=frozen.json(priors,'report.json')
    prior_rows=prior_report['tables'][0]['rows']; prior_coverage=[]
    (output/'auxiliary-coverage').mkdir()
    for row in prior_rows:
        name=row['resource']+'-official-coverage.parquet'
        if name in frozen.indices[priors]:
            path=frozen.path(priors,name);shutil.copy2(path,output/'auxiliary-coverage'/name)
            prior_coverage.append({**row,'coverage_file':'auxiliary-coverage/'+name,'single_cell_CRISPRi_RNA_supervision':False})
        else: prior_coverage.append({**row,'coverage_file':None,'single_cell_CRISPRi_RNA_supervision':False})
    write_json(output/'auxiliary-resources.json',prior_coverage)
    frame=pd.DataFrame(tasks)
    if frame[['panel_id','source_task','source_task_file']].duplicated().any(): raise ValueError('task_key_collision')
    frame.to_parquet(output/'all-source-tasks.parquet', index=False, compression='zstd')
    # Unique gene targets and source construct tasks are different denominators.
    target_frames=[]
    for row in panels:
        selected=frame.loc[frame.panel_id.eq(row['panel_id'])]
        part=target_coverage(row['panel_id'],targets,selected,row['human_RNA_applicable']); target_frames.append(part)
        row['source_construct_tasks']=len(selected)
        row['completed_CRISPRi_construct_tasks']=int((selected.modality.eq('CRISPRi') & selected.effect_status.eq('completed')).sum())
        row['official_targets_with_CRISPRi_response']=int(part.completed_CRISPRi_construct_tasks.gt(0).sum())
        row['official_targets_after_copy_exclusion']=int(part.completed_CRISPRi_tasks_after_confirmed_copy_exclusion.gt(0).sum())
    target_frame=pd.concat(target_frames,ignore_index=True);target_frame.to_parquet(output/'all-panel-target-coverage.parquet',index=False,compression='zstd')
    write_json(output/'panels.json',panels);write_json(output/'condition-index.json',conditions)
    pd.DataFrame([{**r,'source_metadata':json.dumps(r['source_metadata'],ensure_ascii=False)} for r in panels]).to_parquet(output/'panels.parquet',index=False)
    np.savez_compressed(output/'official-coverage-codes.npz',codes=np.stack(compact_codes),panels=np.array([r['panel_id'] for r in panels]),official_genes=np.array(names),statuses=np.array(STATUS))
    write_json(output/'used-inputs.json',{'frozen_artifacts':frozen.used,'raw_reference_hashes':raw_refs})
    for p,d in raw_refs.items():
        if hash_file(ROOT/p)!=d:raise ValueError('reference_changed_during_coverage')
    summary={'status':'completed','phase':'all_source_gene_and_task_coverage','panels':len(panels),'native_mapping_axes':len(mapping_cache),
             'official_gene_denominator':18533,'official_identifier_resolved':int(official.mapped_symbol.notna().sum()),'official_target_denominator':300,
             'source_construct_tasks':len(frame),'construct_task_status_counts':frame.effect_status.value_counts().to_dict(),
             'construct_task_modality_counts':frame.modality.value_counts(dropna=False).to_dict(),
             'confirmed_collection_copy_tasks':int(frame.confirmed_collection_copy.sum()),'condition_files':len(conditions),
             'condition_rows_by_family':{r['family']:r['rows'] for r in conditions},'human_RNA_applicable_panels':sum(r['human_RNA_applicable'] for r in panels),
             'independent_biological_replicates':None,'all_inputs_unchanged':True,
             'coverage_rules':'Unique conflict-free human canonical mapping only; absent/unresolved/ambiguous/N/A separate; zero QC limited to declared file or stimulus scope; no missing-feature zero fill.',
             'exposure':'All frozen source result panels and tasks; individual outcome diagnostics already examined. No untouched evaluation split claimed.',
             'code_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'code_sha256':hash_file(Path(__file__)),
             'core_sha256':hash_file(Path(__file__).with_name('cross_source_coverage.py')),'completed_at':datetime.now(timezone.utc).isoformat(),'duration_seconds':time.monotonic()-started,'reproduce':sys.argv}
    summary['artifacts']={str(p.relative_to(output)):hash_file(p) for p in sorted(output.rglob('*')) if p.is_file()}
    write_json(output/'report.json',summary)
    (output/'SHA256SUMS').write_text(''.join(hash_file(p)+'  '+str(p.relative_to(output))+'\n' for p in sorted(output.rglob('*')) if p.is_file() and p.name!='SHA256SUMS'))
    print(json.dumps({k:v for k,v in summary.items() if k not in ['artifacts','condition_rows_by_family']}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    try:build(args.output)
    except Exception as error:
        if args.output.exists():write_json(args.output/'failure.json',{'status':'failed','error':repr(error),'code_sha256':hash_file(Path(__file__))})
        raise
