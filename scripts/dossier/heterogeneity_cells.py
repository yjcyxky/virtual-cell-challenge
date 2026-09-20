"""Verified source-specific cell/control adapters for descriptive endpoint analysis."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy import sparse
from profile_cross_source_coverage import Frozen,ROOT,BASE
from profile_response_heterogeneity import task_eligibility
from profile_background import matrix_in_memory
from rna import RNAFile,hash_file,value_hash

COMMON=['record_id','input_sha256','row_index','source_barcode','computed_total_counts','computed_total_expression','computed_detected_genes','computed_numeric_valid',
        'source_batch','source_task','source_target_gene','source_guide_id','inferred_type','inferred_lineage','probability_correct','confidence_calibration','state__cycle_s','state__cycle_g2m']


class CellContexts:
    def __init__(self,source):
        self.source=source.resolve();self.frozen=Frozen();self.raw_inputs={};self.raw_stats={};self.cache_identity={}
        report=self.frozen.json(self.source,'report.json')
        if report['bundle_id']!='cross-source-dossier-cba1296d48884511b1477b5f9417314b':raise ValueError('unregistered_parent_identity')
        self.panels={p['panel_id']:p for p in self.frozen.json(self.source,'coverage/panels.json')}
        tasks=self.frozen.parquet(self.source,'coverage/all-source-tasks.parquet');rows=[]
        for row in tasks.to_dict('records'):
            if task_eligibility(row,self.panels[row['panel_id']])[0]:
                row['task_uid']=value_hash([row['panel_id'],row['source_task'],row['source_task_file']]);rows.append(row)
        self.tasks=pd.DataFrame(rows)
        if len(self.tasks)!=37845:raise ValueError('registered_task_count_changed')

    def cells(self,folder,name,extra=()):
        path=self.frozen.path(folder,name);columns=pq.read_schema(path).names
        return pd.read_parquet(path,columns=[c for c in dict.fromkeys(COMMON+list(extra)) if c in columns])

    def verified_path(self,path,digest):
        path=Path(path);path=path if path.is_absolute() else ROOT/path;key=str(path.relative_to(ROOT))
        if key not in self.raw_inputs:
            if hash_file(path)!=digest:raise ValueError('count_or_view_input_changed:'+key)
            self.raw_inputs[key]=digest;st=path.stat();self.raw_stats[key]=(st.st_size,st.st_mtime_ns,st.st_ctime_ns)
        elif self.raw_inputs[key]!=digest:raise ValueError('one_count_input_two_identities')
        return path

    def cached_log(self,folder,expected_input,n,g):
        identity=json.loads((folder/'identity.json').read_text())
        if identity['identity']['input_sha256']!=expected_input or identity['identity']['parameters']['normalization_total']!=10000:raise ValueError('cached_log_input_or_scale_mismatch')
        path=self.verified_path(folder/'logcp.npy',identity['hashes']['logcp.npy']);log=np.load(path,mmap_mode='r')
        if log.shape!=(n,g):raise ValueError('cached_log_shape_mismatch')
        self.cache_identity[str(folder.relative_to(ROOT))]=identity
        return log

    def raw_matrix(self,path,digest):
        path=self.verified_path(path,digest)
        with RNAFile(path) as source:
            genes=source.var.index.astype(str).tolist();barcodes=source.obs.index.astype(str).to_numpy()
            matrix=matrix_in_memory(source)
        if not np.isfinite(matrix.data).all() or (matrix.data<0).any() or (matrix.data!=np.floor(matrix.data)).any():raise ValueError('source_not_nonnegative_integer_counts')
        return matrix,genes,barcodes

    def normalized_selection(self,matrix,rows,cells,barcodes,barcode_column='source_barcode'):
        rows=np.asarray(rows,dtype=np.int64)
        if len(np.unique(rows))!=len(rows) or (rows<0).any() or (rows>=matrix.shape[0]).any():raise ValueError('context_source_row_identity_invalid')
        if not np.array_equal(barcodes[rows],cells[barcode_column].astype(str).to_numpy()):raise ValueError('context_count_barcode_identity_mismatch')
        x=matrix[rows].astype(np.float64)
        totals=np.asarray(x.sum(axis=1)).ravel();expected=cells['computed_total_counts' if 'computed_total_counts' in cells else 'computed_total_expression'].to_numpy()
        if not np.array_equal(totals,expected):raise ValueError('context_count_totals_mismatch')
        scale=np.divide(10000.,totals,out=np.zeros_like(totals),where=totals>0)
        x.data=np.log1p(x.data*np.repeat(scale,np.diff(x.indptr))).astype(np.float32)
        return x.astype(np.float32)

    def make(self,pid,context,cells,log,genes,tasks,control,task_labels,baseline_id=None):
        cells=cells.reset_index(drop=True).copy();control=np.asarray(control,dtype=bool);task_labels=np.asarray(task_labels,dtype=object)
        if len(cells)!=log.shape[0] or len(genes)!=log.shape[1] or len(control)!=len(cells):raise ValueError('context_axis_mismatch')
        if cells.record_id.duplicated().any() or cells.probability_correct.notna().any() or not cells.confidence_calibration.eq('uncalibrated').all():raise ValueError('source_cell_inference_contract_mismatch')
        if not pd.Series(task_labels[control]).isna().all():raise ValueError('target_reference_self_overlap')
        mapping=self.frozen.parquet(self.source,'coverage/'+self.panels[pid]['native_mapping_file'])
        return {'context_id':value_hash([pid,context]),'panel_id':pid,'source_context':context,'cells':cells,'log':log,'genes':genes,'mapping':mapping,
                'tasks':tasks,'control':control,'task_labels':task_labels,'baseline_id':baseline_id or value_hash([pid,context]),'source_metadata':self.panels[pid]['source_metadata']}

    def iter_contexts(self):
        # Existing normalized caches have original source and full cache hashes.
        h1=BASE/'h1-annotation-20260919';all_h1=self.cells(h1,'cells.parquet',['split'])
        for split in ['Training','Validation','Test']:
            pid='H1:'+split;tasks=self.tasks.loc[self.tasks.panel_id.eq(pid)].copy();cells=all_h1.loc[all_h1.split.eq(split)].sort_values('row_index').reset_index(drop=True)
            mapping=self.frozen.parquet(self.source,'coverage/'+self.panels[pid]['native_mapping_file']);genes=mapping.source_gene.tolist()
            log=self.cached_log(BASE/'h1-response-20260919'/('cache_'+split),cells.input_sha256.iloc[0],len(cells),len(genes))
            labels=cells.source_target_gene.map(dict(zip(tasks.source_target,tasks.task_uid)));control=cells.source_target_gene.eq('non-targeting')
            yield self.make(pid,split,cells,log,genes,tasks,control,labels,baseline_id='H1_shared_exact_NTC')
            del cells,log
        del all_h1
        for family,contexts in [('replogle',['K562_essential','K562_gwps','rpe1']),('nadig',['hepg2','jurkat'])]:
            for context in contexts:
                pid=family+':'+context;tasks=self.tasks.loc[self.tasks.panel_id.eq(pid)].copy();folder=BASE/(family+'-annotation-'+context+'-20260919')
                cells=self.cells(folder,'cells.parquet').sort_values('row_index').reset_index(drop=True)
                mapping=self.frozen.parquet(self.source,'coverage/'+self.panels[pid]['native_mapping_file']);genes=mapping.source_gene_id.tolist()
                log=self.cached_log(BASE/(family+'-response-'+context+'-20260919')/'cache',cells.input_sha256.iloc[0],len(cells),len(genes))
                labels=cells.source_task.map(dict(zip(tasks.source_task,tasks.task_uid)));control=cells.source_target_gene.eq('non-targeting')
                yield self.make(pid,context,cells,log,genes,tasks,control,labels)
                del cells,log
        # Each conversion is loaded once per source stimulus and reused read-only.
        jiang=BASE/'jiang-dossier-20260919-v2';report=self.frozen.json(jiang,'report.json');args=report['reproduce'];cache=ROOT/args[args.index('--cache')+1]
        for stimulus in ['IFNB','IFNG','INS','TGFB','TNFA']:
            identity=json.loads((cache/stimulus/'identity.json').read_text());self.cache_identity[str((cache/stimulus).relative_to(ROOT))]=identity
            matrix,genes,barcodes=self.raw_matrix(cache/stimulus/'counts.h5ad',identity['output_sha256'])
            for pid in sorted(p for p in self.tasks.panel_id.unique() if p.startswith('Jiang__') and p.endswith('__'+('TGFB1' if stimulus=='TGFB' else stimulus))):
                tasks=self.tasks.loc[self.tasks.panel_id.eq(pid)].copy();cells=self.cells(jiang,pid+'/cells.parquet').sort_values('row_index').reset_index(drop=True)
                log=self.normalized_selection(matrix,cells.row_index,cells,barcodes);labels=cells.source_task.map(dict(zip(tasks.source_task,tasks.task_uid)))
                yield self.make(pid,pid,cells,log,genes,tasks,cells.source_target_gene.eq('non-targeting'),labels)
                del cells,log
            del matrix
        gxe1=BASE/'mcfaline-gxe1-dossier-20260919-v2';report=self.frozen.json(gxe1,'report.json');args=report['reproduce'];cache=ROOT/args[args.index('--cache')+1]
        identity=self.frozen.json(gxe1,'adapter/identity.json');matrix,genes,barcodes=self.raw_matrix(cache/'coordinate-counts.h5ad',identity['artifacts']['coordinate-counts.h5ad'])
        all_cells=self.cells(gxe1,'cells.parquet',['genetic_response_eligible','analysis_target','source_context'])
        for context,tasks in self.tasks.loc[self.tasks.panel_id.eq('GxE1')].groupby('source_condition',sort=True):
            cells=all_cells.loc[all_cells.source_context.eq(context)&all_cells.genetic_response_eligible].sort_values('row_index').reset_index(drop=True)
            log=self.normalized_selection(matrix,cells.row_index,cells,barcodes);labels=cells.analysis_target.map(dict(zip(tasks.source_target,tasks.task_uid)))
            yield self.make('GxE1',context,cells,log,genes,tasks,cells.analysis_target.eq('NTC'),labels)
            del cells,log
        del matrix,all_cells
        gxe2=BASE/'mcfaline-gxe2-dossier-20260919';report=self.frozen.json(gxe2,'genetic/report.json');identity=self.frozen.json(gxe2,'adapter/identity.json')
        grouped={}
        for row in report['contexts']:
            name='genetic/'+row['directory']+'/cells.parquet';cells=self.cells(gxe2,name,['count_view_file','CDS_view_row','source_CDS_source_barcode','primary_genetic_response_eligible','analysis_target'])
            paths=cells.count_view_file.unique()
            if len(paths)!=1:raise ValueError('GxE2_context_multiple_source_matrices')
            grouped.setdefault(paths[0],[]).append((row,name))
        for path,items in grouped.items():
            path=Path(path);path=path if path.is_absolute() else ROOT/path;relative=str(path.relative_to(path.parent.parent))
            matrix,genes,barcodes=self.raw_matrix(path,identity['artifacts'][relative])
            for row,name in items:
                pid='GxE2:'+row['context'];tasks=self.tasks.loc[self.tasks.panel_id.eq(pid)].copy()
                cells=self.cells(gxe2,name,['CDS_view_row','source_CDS_source_barcode','primary_genetic_response_eligible','analysis_target'])
                cells=cells.loc[cells.primary_genetic_response_eligible].sort_values('CDS_view_row').reset_index(drop=True)
                log=self.normalized_selection(matrix,cells.CDS_view_row,cells,barcodes,'source_CDS_source_barcode');labels=cells.analysis_target.map(dict(zip(tasks.source_task,tasks.task_uid)))
                yield self.make(pid,row['context'],cells,log,genes,tasks,cells.analysis_target.eq('NTC'),labels)
                del cells,log
            del matrix
        sc=BASE/'scperturb-dossier-20260919'
        for pid,all_tasks in self.tasks.loc[self.tasks.panel_id.str.startswith('scPerturb:')].groupby('panel_id',sort=True):
            name=pid.removeprefix('scPerturb:');cells=self.cells(sc,'cells/'+name+'/cells.parquet');design=self.frozen.parquet(sc,'design/'+name+'/row-applicability.parquet')
            if not np.array_equal(cells.row_index,design.row_index):raise ValueError('scPerturb_design_row_identity_mismatch')
            audit=self.frozen.json(sc,'source-audit/'+name+'.h5ad.json')
            matrix,genes,barcodes=self.raw_matrix(ROOT/audit['path'],audit['input_sha256'])
            positive=cells.computed_numeric_valid&cells.computed_total_expression.gt(0)
            for bg,tasks in all_tasks.groupby('source_background_index',sort=True):
                selected=positive&design.biological_background_index.eq(bg)&(design.control_eligible|design.deep_response_candidate)
                part=cells.loc[selected].copy().reset_index(drop=True);d=design.loc[selected].reset_index(drop=True)
                part['source_batch']=d.response_background;part['source_guide_id']=d.source_guide_for_consistency
                labels=d.single_target.map(dict(zip(tasks.source_target,tasks.task_uid))).where(d.deep_response_candidate)
                log=self.normalized_selection(matrix,part.row_index,part,barcodes)
                yield self.make(pid,'source_biological_background_'+str(int(bg)),part,log,genes,tasks,d.control_eligible,labels)
                del part,log
            del matrix,cells,design

    def verify_unchanged(self):
        for key,expected in self.raw_stats.items():
            s=(ROOT/key).stat()
            if (s.st_size,s.st_mtime_ns,s.st_ctime_ns)!=expected:raise ValueError('raw_or_count_view_changed_during_analysis:'+key)
        return {'frozen_artifacts':self.frozen.used,'raw_or_normalized_view_sha256':self.raw_inputs,'cache_identities':self.cache_identity,
                'all_consumed_count_views_unchanged':True,'original_input_mutations':0}
