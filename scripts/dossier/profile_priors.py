#!/usr/bin/env python
"""Full numeric/identity and coverage assessment of auxiliary resources, without training."""
import argparse
from collections import Counter,defaultdict
from datetime import datetime,timezone
import gzip
import json
from pathlib import Path
import re
import shutil
import struct
import subprocess
import sys
import time
import uuid
import zipfile
import h5py
import numpy as np
import pandas as pd
from rna import hash_file,mapping_audit,quantiles
from profile_responses import write_json,serial
from render import render
ROOT=Path(__file__).resolve().parents[2]
SOURCES={'lincs_l1000','depmap_24q4','networks','vcc_gene_axis','arc_se600m','esm2_650m'}


def depmap_features(headers,namespace):
    symbols=[];ids=[]
    for header in headers:
        match=re.fullmatch(r'(.+) \(([^()]+)\)',header)
        if match:symbol,identifier=match.groups()
        elif namespace=='ensembl_gene_id' and re.fullmatch(r'ENSG\d+(?:\.\d+)?',header):symbol=identifier=header
        else:raise ValueError('DepMap_unrecognized_feature_header: '+header)
        symbols.append(symbol);ids.append(identifier)
    return symbols,ids


def safe_mapping(symbols,ids,kind,hgnc,official):
    result=mapping_audit(list(symbols),hgnc,official)
    lookup=defaultdict(set)
    if kind:
        for row in hgnc[hgnc.status=='Approved'].itertuples(index=False):
            for value in str(getattr(row,kind)).split('|'):
                if value and value!='nan':lookup[value.split('.')[0]].add(row.symbol)
    candidates=[lookup.get(str(x).split('.')[0],set()) if kind else set() for x in ids]
    result['source_identifier']=list(ids);result['identifier_namespace']=kind
    result['identifier_candidates']=['|'.join(sorted(x)) for x in candidates]
    result['identifier_conflict']=[bool(c) and pd.notna(s) and s not in c for s,c in zip(result.mapped_symbol,candidates)]
    result['safe_symbol']=[None if conflict or len(c)>1 else s if pd.notna(s) else next(iter(c)) if len(c)==1 else None for s,c,conflict in zip(result.mapped_symbol,candidates,result.identifier_conflict)]
    result['safe_mapping_ambiguous']=[len(c)>1 or conflict or state=='ambiguous' for c,conflict,state in zip(candidates,result.identifier_conflict,result.mapping_status)]
    counts=Counter(result.safe_symbol.dropna());result['safe_many_to_one']=[counts[x]>1 if pd.notna(x) else False for x in result.safe_symbol]
    return result


def coverage(name,mapping,official,targets,out,semantics,qualifier=None):
    mapping.to_parquet(out/(name+'-gene-mapping.parquet'),index=False)
    rows=[];groups={gene:group for gene,group in mapping.dropna(subset=['safe_symbol']).groupby('safe_symbol')}
    for gene in official:
        selected=groups.get(gene,mapping.iloc[:0])
        rows.append({'resource':name,'official_gene':gene,'official_target':gene in targets,'mapped_source_features':len(selected),
            'covered':bool(len(selected)),'measurement_semantics':semantics,'missing_meaning':'outside_unambiguous_resource_coverage_not_zero',
            **({'qualifiers':sorted(set(selected[qualifier].astype(str)))} if qualifier else {})})
    pd.DataFrame(rows).to_parquet(out/(name+'-official-coverage.parquet'),index=False)
    covered={x for x in mapping.safe_symbol if pd.notna(x)}
    return {'resource':name,'source_features':len(mapping),'unambiguous_symbols':len(covered),'official_axis_covered':len(covered&set(official)),
        'official_targets_covered':len(covered&targets),'unmapped':int(mapping.safe_symbol.isna().sum()),'ambiguous_or_conflicting':int(mapping.safe_mapping_ambiguous.sum()),
        'many_to_one_features':int(mapping.safe_many_to_one.sum()),'semantics':semantics,'status':'completed'}


class Numeric:
    def __init__(self,g):
        self.n=0;self.missing=np.zeros(g,np.int64);self.infinity=np.zeros(g,np.int64);self.finite=np.zeros(g,np.int64)
        self.negative=np.zeros(g,np.int64);self.fractional=np.zeros(g,np.int64);self.sums=np.zeros(g);self.squares=np.zeros(g)
        self.minimum=np.full(g,np.inf);self.maximum=np.full(g,-np.inf);self.outside_probability=0
    def add(self,a):
        a=np.asarray(a,dtype=np.float64);valid=np.isfinite(a);v=np.where(valid,a,0)
        self.n+=len(a);self.missing+=np.isnan(a).sum(0);self.infinity+=np.isinf(a).sum(0);self.finite+=valid.sum(0)
        self.negative+=((a<0)&valid).sum(0);self.fractional+=(valid&(a!=np.floor(a))).sum(0)
        self.sums+=v.sum(0);self.squares+=(v*v).sum(0)
        self.minimum=np.minimum(self.minimum,np.where(valid,a,np.inf).min(0));self.maximum=np.maximum(self.maximum,np.where(valid,a,-np.inf).max(0))
        self.outside_probability+=int((valid&((a<0)|(a>1))).sum())
    def frame(self,genes):
        mean=np.divide(self.sums,self.finite,out=np.full(len(genes),np.nan),where=self.finite>0)
        variance=np.divide(self.squares,self.finite,out=np.full(len(genes),np.nan),where=self.finite>0)-mean*mean
        return pd.DataFrame({'source_gene':genes,'finite':self.finite,'missing_NaN':self.missing,'infinity':self.infinity,
            'negative':self.negative,'noninteger':self.fractional,'mean':mean,'variance':np.maximum(0,variance),'minimum':self.minimum,'maximum':self.maximum})
    def summary(self):
        return {'rows':self.n,'features':len(self.finite),'values_checked':self.n*len(self.finite),'missing_NaN':int(self.missing.sum()),
            'infinity':int(self.infinity.sum()),'negative_values':int(self.negative.sum()),'noninteger_values':int(self.fractional.sum()),
            'minimum':float(self.minimum.min()),'maximum':float(self.maximum.max())}


def lincs(cache,out,hgnc,official,targets):
    base=ROOT/'data/raw/lincs_l1000';identity=json.loads((cache/'identity.json').read_text())
    # The gzip source and decompressed file are separately content locked.
    if hash_file(cache/'expression.gctx')!=identity['decompressed_sha256']:raise ValueError('LINCS_decompression_cache_changed')
    meta={name:pd.read_csv(base/f'GSE92742_Broad_LINCS_{name}_info.txt.gz',sep='\t',dtype=str,keep_default_na=False) for name in ['gene','cell','sig','pert']}
    with h5py.File(cache/'expression.gctx','r') as h:
        values=h['0/DATA/0/matrix'];genes=h['0/META/ROW/id'].asstr()[:];signatures=h['0/META/COL/id'].asstr()[:]
        if values.shape!=(len(signatures),len(genes)) or len(set(signatures))!=len(signatures) or len(set(genes))!=len(genes):raise ValueError('LINCS_axis_identity_error')
        gm=meta['gene'].set_index('pr_gene_id').reindex(genes)
        if gm.pr_gene_symbol.isna().any():raise ValueError('LINCS_missing_gene_metadata')
        sm=meta['sig'].set_index('sig_id').reindex(signatures)
        if sm.cell_id.isna().any():raise ValueError('LINCS_missing_signature_metadata')
        mapping=safe_mapping(gm.pr_gene_symbol,genes,'entrez_id',hgnc,official)
        mapping['measurement_class']=np.where(gm.pr_is_lm.to_numpy()=='1','measured_landmark','inferred_gene')
        mapping['source_is_BING']=gm.pr_is_bing.to_numpy();mapping['mean_MODZ_semantics']='aggregate_standardized_response_not_expression_counts'
        numeric=Numeric(len(genes));sig_stats=[]
        for start in range(0,len(signatures),1024):
            a=values[start:start+1024];numeric.add(a)
            sig_stats.append(pd.DataFrame({'sig_id':signatures[start:start+len(a)],'missing_values':(~np.isfinite(a)).sum(1),
                'mean_MODZ':np.mean(a,axis=1),'RMS_MODZ':np.sqrt(np.mean(a.astype(float)**2,axis=1))}))
        numeric.frame(genes).to_parquet(out/'lincs-gene-numeric.parquet',index=False)
        sig_frame=sm.reset_index().rename(columns={'index':'sig_id'}).merge(pd.concat(sig_stats),on='sig_id',validate='one_to_one')
        sig_frame['source_missing_fields']=sig_frame.apply(lambda row:'|'.join(k for k,v in row.items() if str(v)=='-666'),axis=1)
        sig_frame['source_distilled_instance_count']=sig_frame.distil_id.str.split('|').str.len()
        sig_frame['independent_biological_replicates']=None
        sig_frame.to_parquet(out/'lincs-signatures.parquet',index=False)
    row=coverage('lincs',mapping,official,targets,out,'Level5 MODZ response; landmark measurements and inferred genes distinct','measurement_class')
    row.update(numeric.summary());row.update(signatures=len(signatures),cell_contexts=sm.cell_id.nunique(),perturbagen_types=sm.pert_type.value_counts().to_dict(),
        landmark_features=int((mapping.measurement_class=='measured_landmark').sum()),inferred_features=int((mapping.measurement_class=='inferred_gene').sum()),
        missing_cell_metadata=int((~sm.cell_id.isin(meta['cell'].cell_id)).sum()),missing_perturbagen_metadata=int((~sm.pert_id.isin(meta['pert'].pert_id)).sum()),
        interpretation='Not raw counts or single cells; distilled instance labels do not prove independent biological repeats')
    for key in ['cell','pert']:meta[key].to_parquet(out/f'lincs-{key}-metadata.parquet',index=False)
    return [row]


def depmap(out,hgnc,official,targets):
    base=ROOT/'data/raw/depmap_24q4';models=pd.read_csv(base/'Model.csv');profiles=pd.read_csv(base/'OmicsProfiles.csv')
    if models.ModelID.duplicated().any() or profiles.ProfileID.duplicated().any():raise ValueError('DepMap_duplicate_model_or_profile_identity')
    models.to_parquet(out/'depmap-models.parquet',index=False);profiles.to_parquet(out/'depmap-profiles.parquet',index=False)
    rows=[]
    for filename,semantics,namespace in [
        ('CRISPRGeneEffect.csv','Chronos integrated copy-number/screen-quality corrected fitness gene effect','entrez_id'),
        ('CRISPRGeneDependency.csv','Gene dependency probability, not transcript response','entrez_id'),
        ('OmicsExpressionAllGenesTPMLogp1Profile.csv','Bulk RNA-seq RSEM log2(TPM+1), not single-cell UMI','ensembl_gene_id')]:
        name='depmap-'+filename[:-4];header=pd.read_csv(base/filename,nrows=0).columns[1:].tolist()
        symbols,ids=depmap_features(header,namespace)
        mapping=safe_mapping(symbols,ids,namespace,hgnc,official);mapping['source_header']=header
        numeric=Numeric(len(header));identities=[];record_stats=[]
        for chunk in pd.read_csv(base/filename,index_col=0,chunksize=32):
            a=chunk.to_numpy(dtype=float);numeric.add(a);identities+=chunk.index.astype(str).tolist()
            record_stats.append(pd.DataFrame({'source_record_id':chunk.index.astype(str),'finite_features':np.isfinite(a).sum(1),'missing_features':np.isnan(a).sum(1)}))
        if len(set(identities))!=len(identities):raise ValueError('DepMap_duplicate_matrix_record_identity')
        numeric.frame(header).to_parquet(out/(name+'-numeric.parquet'),index=False)
        records=pd.concat(record_stats,ignore_index=True)
        if namespace=='ensembl_gene_id':
            records=records.merge(profiles[['ProfileID','ModelID','ModelCondition','Datatype']],left_on='source_record_id',right_on='ProfileID',how='left',validate='one_to_one')
        else:records['ModelID']=records.source_record_id
        records=records.merge(models[['ModelID','CellLineName','OncotreeLineage']],on='ModelID',how='left',validate='many_to_one')
        records.to_parquet(out/(name+'-records.parquet'),index=False)
        row=coverage(name,mapping,official,targets,out,semantics);row.update(numeric.summary());row['records_missing_model_metadata']=int(records.CellLineName.isna().sum())
        if filename=='CRISPRGeneDependency.csv':row['probability_range_violations']=numeric.outside_probability
        rows.append(row)
    return rows


def networks(out,hgnc,official,targets):
    base=ROOT/'data/raw/networks';rows=[]
    proteins=pd.read_csv(base/'9606.protein.info.v12.0.txt.gz',sep='\t',dtype=str)
    idcol=proteins.columns[0];protein_ids=set(proteins[idcol]);mapping=safe_mapping(proteins.preferred_name,proteins[idcol],None,hgnc,official)
    for filename in ['9606.protein.links.v12.0.txt.gz','9606.protein.physical.links.v12.0.txt.gz']:
        endpoints=set();degree=Counter();score_bins=Counter();count=bad=unknown=selflinks=0
        for frame_ in pd.read_csv(base/filename,sep=r'\s+',chunksize=250000):
            a,b=frame_.protein1,frame_.protein2;scores=frame_.combined_score.to_numpy()
            count+=len(frame_);bad+=int((~np.isfinite(scores)|(scores<0)|(scores>1000)).sum());unknown+=int((~a.isin(protein_ids)|~b.isin(protein_ids)).sum());selflinks+=int((a==b).sum())
            endpoints.update(a);endpoints.update(b);degree.update(a);degree.update(b);score_bins.update((scores//100).astype(int))
        name='string-physical' if 'physical' in filename else 'string-functional'
        selected=mapping[mapping.source_identifier.isin(endpoints)].copy();selected['directed_endpoint_occurrences']=[degree[x] for x in selected.source_identifier]
        row=coverage(name,selected,official,targets,out,'Static protein association edges; confidence score 0–1000, not regulatory sign, causality, or perturbation response')
        row.update(edge_rows=count,score_range_violations=bad,unknown_endpoint_rows=unknown,self_link_rows=selflinks,score_bins_100={str(k):v for k,v in score_bins.items()},edge_orientation='Source directed rows retained in counts; do not double as independent interaction evidence')
        rows.append(row)
    proteins.to_parquet(out/'string-protein-metadata.parquet',index=False)
    with zipfile.ZipFile(base/'ReactomePathways.gmt.zip') as z:
        lines=z.read('ReactomePathways.gmt').decode().splitlines()
    records=[]
    for line in lines:
        fields=line.split('\t')
        if len(fields)<3:raise ValueError('invalid_Reactome_GMT_row')
        records.append({'pathway_name':fields[0],'pathway_id':fields[1],'source_genes':fields[2:],'species_by_stable_id':'human' if fields[1].startswith('R-HSA-') else 'other_or_unknown'})
    genes=sorted({g for r in records if r['species_by_stable_id']=='human' for g in r['source_genes']})
    mapping=safe_mapping(genes,genes,None,hgnc,official);row=coverage('reactome-human',mapping,official,targets,out,'Curated pathway membership; includes viral protein labels, not measured response')
    row.update(pathways=len(records),release_number=None,version_status='mutable_current_URL; local content SHA256 and retrieval date frozen, named upstream release unproven')
    rows.append(row);pd.DataFrame(records).to_parquet(out/'reactome-pathways.parquet',index=False)
    header=[];annotations=[]
    with gzip.open(base/'goa_human.gaf.gz','rt') as f:
        for line in f:
            if line.startswith('!'):header.append(line.rstrip());continue
            fields=line.rstrip('\n').split('\t')
            if len(fields)!=17:raise ValueError('invalid_GAF_column_count')
            annotations.append({'database':fields[0],'object_id':fields[1],'symbol':fields[2],'qualifier':fields[3],'GO_ID':fields[4],
                'reference':fields[5],'evidence':fields[6],'with_from':fields[7],'aspect':fields[8],'object_type':fields[11],'taxon':fields[12],
                'annotation_date':fields[13],'assigned_by':fields[14],'annotation_extension':fields[15],'gene_product_form':fields[16]})
    gaf=pd.DataFrame(annotations);gaf.to_parquet(out/'GOA-annotations.parquet',index=False);write_json(out/'GOA-headers.json',header)
    usable=gaf[(gaf.taxon.str.split('|').str[0]=='taxon:9606')&~gaf.qualifier.str.split('|').apply(lambda x:'NOT' in x)]
    genes=sorted(set(usable.symbol));mapping=safe_mapping(genes,genes,None,hgnc,official)
    row=coverage('GOA-human-positive',mapping,official,targets,out,'Ontology annotation, evidence codes and NOT qualifiers retained; no absence-as-negative assumption')
    row.update(annotation_rows=len(gaf),positive_human_rows=len(usable),evidence_codes=gaf.evidence.value_counts().to_dict(),source_header_dates=[x for x in header if 'date' in x.lower()])
    rows.append(row)
    approved=hgnc[hgnc.status=='Approved'];mapping=safe_mapping(approved.symbol,approved.hgnc_id,None,hgnc,official)
    row=coverage('HGNC-approved',mapping,official,targets,out,'Gene identity and nomenclature mapping; not measured expression or pathway activity')
    row.update(release_number=None,version_status='mutable URL; frozen input SHA256, per-entry dates are not a release identifier')
    rows.append(row)
    return rows


def weights(path,out):
    with path.open('rb') as f:
        header_size=struct.unpack('<Q',f.read(8))[0]
        if header_size>100000000:raise ValueError('invalid_safetensors_header_size')
        header=json.loads(f.read(header_size));start=8+header_size;entries=[];ranges=[]
        for name,entry in header.items():
            if name=='__metadata__':continue
            dtype=entry['dtype'];shape=entry['shape'];lo,hi=entry['data_offsets'];length=int(np.prod(shape,dtype=np.int64))
            types={'F32':'<f4','F16':'<f2','F64':'<f8','BF16':'<u2','I64':'<i8','I32':'<i4','I16':'<i2','I8':'i1','U8':'u1','BOOL':'?'}
            if dtype not in types:raise ValueError('unsupported_tensor_dtype: '+dtype)
            dt=np.dtype(types[dtype])
            if hi-lo!=length*dt.itemsize or start+hi>path.stat().st_size:raise ValueError('invalid_tensor_shape_or_offsets')
            ranges.append((lo,hi));a=np.memmap(path,offset=start+lo,dtype=dt,mode='r',shape=(length,));nonfinite=0
            for i in range(0,length,2000000):
                chunk=a[i:i+2000000]
                nonfinite+=int((((chunk&0x7f80)==0x7f80) if dtype=='BF16' else ~np.isfinite(chunk)).sum())
            entries.append({'tensor':name,'dtype':dtype,'shape':shape,'elements':length,'nonfinite':nonfinite,'data_start':lo,'data_end':hi})
    ordered=sorted(ranges)
    if ordered and (ordered[0][0]!=0 or ordered[-1][1]+start!=path.stat().st_size or any(a[1]!=b[0] for a,b in zip(ordered,ordered[1:]))):raise ValueError('tensor_ranges_overlap_or_gap')
    name=path.parent.name+'-'+path.stem
    pd.DataFrame(entries).to_parquet(out/(name+'-tensors.parquet'),index=False)
    return {'resource':name,'status':'completed','semantics':'Pretrained model parameters; not aligned gene embeddings or observed biological labels',
        'tensor_count':len(entries),'elements':sum(x['elements'] for x in entries),'nonfinite':sum(x['nonfinite'] for x in entries),
        'official_axis_covered':None,'official_targets_covered':None,'coverage_status':'not_applicable_without_gene_sequence_embedding_mapping',
        'gene_features_computed':False,'inference_or_training_run':False,'header_metadata':header.get('__metadata__',{})}


def assess(inventory,cache,evidence,output):
    if output.exists():raise ValueError('fresh_output_required')
    start=time.monotonic();commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    inv=json.loads((inventory/'report.json').read_text());inputs={r['file']:r['sha256'] for r in inv['file_results'] if r['source_id'] in SOURCES}
    axis=ROOT/'data/raw/arc_vcc2026_controls/gene_names.csv';target_path=ROOT/'data/raw/arc_vcc2026_controls/pert_counts.csv'
    inputs.update({str(p.relative_to(ROOT)):hash_file(p) for p in [axis,target_path]})
    for file,digest in inputs.items():
        if hash_file(ROOT/file)!=digest:raise ValueError('inventory_input_changed: '+file)
    before={file:(ROOT/file).stat() for file in inputs};output.mkdir(parents=True);shutil.copytree(evidence,output/'evidence')
    for source in SOURCES:
        source_path=ROOT/('models' if source in ['arc_se600m','esm2_650m'] else 'data/raw')/source/'SOURCE.json'
        shutil.copy2(source_path,output/(source+'-SOURCE.json'))
    hgnc=pd.read_csv(ROOT/'data/raw/networks/hgnc_complete_set.txt',sep='\t',low_memory=False)
    official=pd.read_csv(axis).gene_name.tolist();targets=set(pd.read_csv(target_path).target_gene)
    rows=[]
    for step,call in [('lincs',lambda:lincs(cache,output,hgnc,official,targets)),('depmap',lambda:depmap(output,hgnc,official,targets)),('networks',lambda:networks(output,hgnc,official,targets))]:
        rows.extend(call());print(step+' completed',flush=True);write_json(output/'partial-coverage.json',rows)
    old=pd.read_csv(ROOT/'data/raw/vcc_gene_axis/gene_names.csv',header=None).iloc[:,0].astype(str).tolist()
    mapping=safe_mapping(old,old,None,hgnc,official);row=coverage('old-VCC-axis',mapping,official,targets,output,'Legacy gene-name axis only; no new expression observations')
    row.update(literal_shared=len(set(old)&set(official)),old_only=len(set(old)-set(official)),current_only=len(set(official)-set(old)),
        ordered_identical=old==official,duplicate_names=len(old)-len(set(old)))
    rows.append(row);pd.DataFrame({'gene':sorted(set(old)|set(official))}).assign(in_old=lambda x:x.gene.isin(old),in_current=lambda x:x.gene.isin(official)).to_parquet(output/'old-vs-current-axis.parquet',index=False)
    for source in ['esm2_650m','arc_se600m']:
        for p in sorted((ROOT/'models'/source).glob('*.safetensors')):rows.append(weights(p,output))
        for p in (ROOT/'models'/source).iterdir():
            if p.is_file() and p.suffix!='.safetensors':shutil.copy2(p,output/(source+'-'+p.name))
    # Required external assets are explicit in the downloaded SE config and absent locally.
    config=(ROOT/'models/arc_se600m/config.yaml').read_text();external_paths=sorted(set(re.findall(r'/[^\s#]+',config)))
    pd.DataFrame({'source_config_path':external_paths,'local_path_exists':[Path(x).exists() for x in external_paths],
        'interpretation':'Source training/configuration path; not a local input or proof of exact training membership'}).to_parquet(output/'SE-external-assets.parquet',index=False)
    decisions=[
        {'resource':'LINCS','candidate_use':'Response signature/perturbagen prior with cell, dose, duration and modality matching','limitation':'MODZ and inferred genes cannot be used as single-cell counts; source -666 metadata means unavailable'},
        {'resource':'DepMap','candidate_use':'Cell-line background and knockout fitness context','limitation':'Fitness dependency differs from CRISPRi RNA response; bulk profile similarity does not establish anonymous A/B/C identity'},
        {'resource':'STRING/Reactome/GO/HGNC','candidate_use':'Static relation, pathway and identifier priors with evidence/qualifier preservation','limitation':'No causal direction, activation, treatment specificity or measurement completeness implied'},
        {'resource':'ESM2','candidate_use':'Potential protein-sequence representation after isoform/sequence/gene mapping is specified','limitation':'Amino-acid tokenizer and model weights provide no ready-made 18,533-gene feature matrix'},
        {'resource':'State SE600M','candidate_use':'Potential expression representation after missing gene embeddings/masks/mappings are supplied','limitation':'Config mentions scBaseCount, cellxgene and Tahoe training sources; exact cell membership unknown, independence cannot be presumed'},
        {'resource':'Legacy VCC axis','candidate_use':'Historical identity comparison','limitation':'Unmeasured, absent, ambiguous and renamed genes are different states; do not zero-fill current-only genes'}]
    for file,digest in inputs.items():
        after=(ROOT/file).stat();prior=before[file]
        if (after.st_size,after.st_mtime_ns,after.st_ctime_ns)!=(prior.st_size,prior.st_mtime_ns,prior.st_ctime_ns) or hash_file(ROOT/file)!=digest:raise ValueError('source_changed_during_assessment')
    report={'schema_version':2,'bundle_id':'priors-'+uuid.uuid4().hex,'title':'辅助证据、命名与参考权重全量评估','status':'completed',
        'completed_at':datetime.now(timezone.utc).isoformat(),'duration_seconds':time.monotonic()-start,'input_sha256':inputs,'inputs_unchanged':True,
        'code_commit':commit,'code_sha256':hash_file(Path(__file__)),'runtime':{'python':sys.version,'numpy':np.__version__,'pandas':pd.__version__,'h5py':h5py.__version__},
        'uv_lock_sha256':hash_file(Path(__file__).with_name('uv.lock')),'official_axis_size':len(official),'official_targets':len(targets),
        'references':['https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE92742','https://clue.io/connectopedia/replicate_collapse',
            'https://doi.org/10.25452/figshare.plus.27993248.v1','https://string-db.org/help/faq/','https://reactome.org/download-data',
            'https://geneontology.org/docs/go-annotation-file-gaf-format-2.2/','https://www.genenames.org/help/hgnc-data/'],
        'methods':{'numeric':'Every matrix value and weight tensor examined in bounded chunks; missing, infinity, negative and fractional counts separately reported',
            'mapping':'Frozen HGNC symbol/alias/previous-name and Entrez/Ensembl corroboration; conflicting or ambiguous IDs excluded from safe coverage without source mutation',
            'version':'DepMap 24Q4 figshare v1; STRING v12; GAF header date; mutable HGNC/Reactome URLs content locked but named release not invented',
            'unit':'LINCS signature, DepMap model/profile, protein edge, pathway membership, ontology annotation, model tensor; none relabeled as independent single cells'},
        'exposure':'All local auxiliary matrices/metadata/weights inspected. No training, model inference, anonymous context identification or aligned feature construction.',
        'limitations':['覆盖表示对应资源中可无歧义映射的特征或关系，不代表同种测量或实测表达。',
            'LINCS 推定基因、NTC/剂量/时间缺失、DepMap 的 bulk 与适应度语义分别保留。',
            '当前链接不是版本号；HGNC 与 Reactome 的内容哈希固定，无法从本地证据确认的发行版本保持 unknown。',
            'SE 配置包含 scBaseCount/cellxgene/Tahoe 训练来源线索；未提供精确训练成员表，不能保证独立验证。',
            '预训练权重尚不是与本届基因轴对齐的特征；缺少的外部输入保持缺少。'],
        'tables':[{'title':'全部资源的语义、覆盖与数值核验','rows':rows},{'title':'候选用途与限制','rows':decisions}],
        'reproduce':sys.argv}
    files=sorted(p for p in output.rglob('*') if p.is_file());report['artifacts']=[{'file':str(p.relative_to(output)),'sha256':hash_file(p)} for p in files]
    write_json(output/'report.json',report);(output/'report.html').write_text(render(serial(report)))
    (output/'SHA256SUMS').write_text(''.join(f'{hash_file(p)}  {p.relative_to(output)}\n' for p in sorted(output.rglob('*')) if p.is_file() and p.name!='SHA256SUMS'))
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['inventory','cache','evidence','output']:p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    try:result=assess(a.inventory,a.cache,a.evidence,a.output)
    except Exception as error:
        if a.output.exists() and not (a.output/'report.json').exists():write_json(a.output/'failure.json',{'status':'failed','error':str(error),'type':type(error).__name__})
        raise
    print(json.dumps({'status':result['status'],'bundle_id':result['bundle_id']}))
