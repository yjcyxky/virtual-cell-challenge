"""Source-faithful sparse coordinate adapter for sci-Plex-GxE."""
from pathlib import Path
import h5py
import numpy as np
import pandas as pd
from convert_seurat_cache import frame
from rna import RNAFile,hash_file


def coordinate_cache(coordinates,cell_table,gene_table,destination,chunk=1000000):
    if destination.exists():raise ValueError('fresh_coordinate_cache_required')
    cells=pd.read_csv(cell_table,sep='\t',header=None,names=['barcode','sample'],dtype=str,keep_default_na=False)
    genes=pd.read_csv(gene_table,sep='\t',header=None,names=['gene_id','gene_name'],dtype=str,keep_default_na=False)
    if cells.barcode.duplicated().any() or genes.gene_id.duplicated().any():raise ValueError('duplicate_coordinate_axis_identity')
    n,g=len(cells),len(genes);sizes=np.zeros(n,np.int64);last_cell=-1;nnz=0;numeric_sum=0
    destination.parent.mkdir(parents=True,exist_ok=True)
    with h5py.File(destination,'w') as h:
        h.attrs['encoding-type']='anndata';h.attrs['encoding-version']='0.1.0'
        frame(h.create_group('obs'),cells.set_index('barcode'));frame(h.create_group('var'),genes.set_index('gene_id'))
        x=h.create_group('X');x.attrs['encoding-type']='csr_matrix';x.attrs['encoding-version']='0.1.0';x.attrs['shape']=(n,g)
        values=x.create_dataset('data',shape=(0,),maxshape=(None,),dtype='<f8',chunks=(1000000,))
        indices=x.create_dataset('indices',shape=(0,),maxshape=(None,),dtype='<i4',chunks=(1000000,))
        with pd.read_csv(coordinates,sep='\t',header=None,names=['gene','cell','count'],dtype=np.int64,chunksize=chunk) as reader:
            for batch in reader:
                a=batch.to_numpy();gi,ci,d=a[:,0]-1,a[:,1]-1,a[:,2]
                if (gi<0).any() or (gi>=g).any() or (ci<0).any() or (ci>=n).any():raise ValueError('one_based_coordinate_out_of_bounds')
                if (d<0).any():raise ValueError('negative_coordinate_count')
                if ci[0]<last_cell or (np.diff(ci)<0).any():raise ValueError('coordinate_cell_order_not_monotonic')
                last_cell=int(ci[-1]);stop=nnz+len(batch);values.resize((stop,));indices.resize((stop,));values[nnz:stop]=d;indices[nnz:stop]=gi
                sizes+=np.bincount(ci,minlength=n);nnz=stop;numeric_sum+=int(d.sum())
        x.create_dataset('indptr',data=np.r_[0,np.cumsum(sizes)])
    return {'shape':[n,g],'coordinate_rows':nnz,'coordinate_count_sum':numeric_sum,'index_base':1,
        'orientation':'gene index, cell index, UMI count; independent cell-by-gene CSR view',
        'zero_coordinate_cells':int((sizes==0).sum()),'all_source_features_preserved':True,
        'coordinate_sha256':hash_file(coordinates),'cell_table_sha256':hash_file(cell_table),'gene_table_sha256':hash_file(gene_table),'cache_sha256':hash_file(destination)}


def compare_CDS(raw,processed):
    """Compare all CDS rows to source coordinates on its declared feature axis."""
    rows=[]
    with RNAFile(raw) as source,RNAFile(processed) as cds:
        source_cells=pd.Index(source.obs.index);source_genes=pd.Index(source.var.index)
        cell_indices=source_cells.get_indexer(cds.obs.index);gene_indices=source_genes.get_indexer(cds.var.index)
        if (cell_indices<0).any() or (gene_indices<0).any():raise ValueError('CDS_cells_or_genes_absent_from_coordinate_axes')
        desired={int(original):i for i,original in enumerate(cell_indices)}
        # Small GxE1 CDS fits in memory. Larger sources are read by bounded row blocks.
        ptr=cds.indptr
        for start,matrix in source.blocks(256):
            for local in range(matrix.shape[0]):
                original=start+local
                if original not in desired:continue
                j=desired[original];lo,hi=ptr[j:j+2]
                from scipy import sparse
                expected=sparse.csr_matrix((cds.x['data'][lo:hi],cds.x['indices'][lo:hi],np.array([0,hi-lo])),shape=(1,cds.shape[1]))
                delta=matrix[local,gene_indices]-expected;delta.eliminate_zeros()
                rows.append({'source_row':original,'CDS_row':j,'source_barcode':str(source.obs.index[original]),'common_features':len(gene_indices),
                    'differing_values':delta.nnz,'max_absolute_difference':float(np.max(np.abs(delta.data))) if delta.nnz else 0,
                    'classification':'counts_identical_on_CDS_feature_axis' if not delta.nnz else 'count_mismatch',
                    'independent_observation':False})
    return pd.DataFrame(rows)


def parse_gxe1_hash(label):
    fields=str(label).split('_')
    if len(fields)!=8 or fields[2]!='A172' or fields[3] not in ['CRISPRi','CRISPRa'] or fields[4] not in ['MMR','HPRT1']:
        raise ValueError('unrecognized_GxE1_hash_condition')
    dose=float(fields[5])
    if not np.isfinite(dose) or dose<0:raise ValueError('invalid_source_dose')
    return {'hash_plate':fields[0],'hash_well':fields[1],'cell_line':fields[2],'effector':fields[3],'guide_library':fields[4],
        'dose_value':dose,'drug':fields[6],'source_last_hash_field':fields[7],
        'exposure_hours_from_primary_methods':96,'dose_unit_from_primary_methods':None if fields[6]=='dmso' else 'uM',
        'vehicle_percent_vv_from_primary_methods':0.1,'vehicle_source_dose_is_placeholder':fields[6]=='dmso'}


def assigned_design(cds_metadata):
    rows=[]
    collisions=set(cds_metadata.loc[cds_metadata.new_cell.duplicated(keep=False)].index) if 'new_cell' in cds_metadata else set()
    for barcode,r in cds_metadata.iterrows():
        facts=parse_gxe1_hash(r.top_oligo_W)
        consistent=(str(r.CRISPR_hash)==facts['effector'] and str(r.gRNA_library)==facts['guide_library'] and float(r.dose)==facts['dose_value'] and str(r.treatment)==facts['drug'])
        target_set=sorted(set(str(r.gene_id).split(','))) if pd.notna(r.gene_id) else []
        effectors=set(str(r.CRISPR_gRNA).split(',')) if pd.notna(r.CRISPR_gRNA) else set()
        guide_conflict=bool(effectors-{'control',facts['effector']})
        library_targets={'HPRT1':{'HPRT1','NTC'},'MMR':{'MGMT','MLH1','MSH2','MSH3','MSH6','PMS2','NTC'}}
        library_conflict=bool(set(target_set)-library_targets[facts['guide_library']])
        hash_supported=pd.notna(r.hash_umis_W) and r.hash_umis_W>=5 and pd.notna(r.top_to_second_best_ratio_W) and r.top_to_second_best_ratio_W>=2.5
        reason='metadata_vs_hash_condition_conflict' if not consistent else 'insufficient_or_ambiguous_hash' if not hash_supported else 'unassigned_guide' if not target_set else 'guide_join_key_collision' if barcode in collisions else 'hash_vs_guide_effector_conflict' if guide_conflict else 'multiple_source_target_genes' if len(target_set)>1 else 'hash_library_vs_guide_target_conflict' if library_conflict else None
        rows.append({'source_barcode':barcode,**facts,'hash_condition_consistent':consistent,'hash_supported':hash_supported,
            'source_targets':target_set,'guide_effector_conflict':guide_conflict,'guide_library_conflict':library_conflict,'analysis_target':target_set[0] if len(target_set)==1 else None,
            'guide_join_key_collision':barcode in collisions,
            'genetic_response_assignment_status':'eligible' if reason is None else 'not_estimable','assignment_limitation':reason,
            'labels_are_source_inferences':True})
    return pd.DataFrame(rows)
