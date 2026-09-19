"""Source-level behavior: intervention identity, control matching and aggregation."""
from pathlib import Path
import sys
import tempfile
import unittest
import h5py
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'dossier'))
from crispri import task_metadata, control_coverage, bulk_comparison
from rna import hash_file
from test_rna import h5ad


class CrispriTests(unittest.TestCase):
    def test_promoters_are_distinct_and_dual_guides_not_replicates(self):
        obs=pd.DataFrame({'gene':['A','A','non-targeting','KNTC1'],
            'gene_id':['ENSG1','ENSG1','NTC','ENSG2'],'transcript':['P1','P2','NTC','P1'],
            'gene_transcript':['A_P1','A_P2','control','KNTC1_P1'],
            'sgID_AB':['a|b','c|d','e|f','g|h'],'gem_group':[1,2,1,1]},index=['a','b','c','d'])
        result=task_metadata(obs).set_index('task')
        self.assertEqual(len(result),4)
        self.assertEqual(result.is_control.sum(),1)
        self.assertTrue((result.guide_components==2).all())
        self.assertTrue((result.independent_constructs==1).all())
        self.assertTrue(result.biological_replicates.isna().all())
        coverage=control_coverage(obs).set_index('task')
        self.assertFalse(coverage.loc['A_P2','matched'])
        self.assertTrue(coverage.loc['KNTC1_P1','matched'])
        conflicting=pd.concat([obs,obs.iloc[:1].rename(index={'a':'different_cell'})])
        conflicting.iloc[-1,conflicting.columns.get_loc('gene')]='WRONG'
        with self.assertRaisesRegex(ValueError,'construct_identity_conflict'):
            task_metadata(conflicting)

    def test_bulk_means_are_not_new_cells_and_unavailable_groups_not_zero(self):
        with tempfile.TemporaryDirectory() as root:
            sc,bulk=Path(root)/'sc.h5ad',Path(root)/'bulk.h5ad'
            h5ad(sc,[[1,2],[2,4],[8,6]])
            h5ad(bulk,[[1.5,3],[7,6],[2,2]],barcodes=['A','B','C'])
            with h5py.File(sc,'a') as f:
                f['obs'].create_dataset('gene_transcript',data=['A','A','B'],dtype=h5py.string_dtype())
            with h5py.File(bulk,'a') as f:
                f['obs'].create_dataset('num_cells_filtered',data=[2,1,0])
                f['obs'].create_dataset('num_cells_unfiltered',data=[3,1,2])
            before=[hash_file(sc),hash_file(bulk)]
            frame,summary=bulk_comparison(sc,bulk,chunk=1)
            self.assertEqual(summary['numeric_violations']['noninteger'],1)
            self.assertEqual(summary['rows_matching_local_singlecell_mean'],1)
            self.assertEqual(summary['groups_without_local_singlecell_records'],1)
            self.assertFalse(frame.independent_observation.any())
            self.assertEqual(frame.loc[2,'reason'],'no_singlecell_rows_in_local_release')
            self.assertTrue(pd.isna(frame.loc[2,'max_absolute_error_to_singlecell_mean']))
            self.assertEqual(before,[hash_file(sc),hash_file(bulk)])

class StateCompositionTests(unittest.TestCase):
    def test_no_cross_GEM_or_type_borrowing_and_uncalibrated_output(self):
        from profile_crispri import state_composition
        records=[]
        for task,target,batch,label,n,value in [
            ('NTC','non-targeting','b1','A',20,1),
            ('MATCH','GENE1','b1','A',12,3),
            ('WRONG_BATCH','GENE2','b2','A',12,3),
            ('WRONG_TYPE','GENE3','b1','B',12,3),
            ('LOW','GENE4','b1','A',3,3)]:
            for i in range(n):
                records.append({'source_task':task,'source_target_gene':target,'source_batch':batch,
                    'inferred_type':label,'inferred_lineage':'lineage','state__cycle':value,
                    'computed_total_counts':100+i,'method_conflict':False,'mixed_marker_signal':False,
                    'target_marker_label_changed':False,'source_context_mismatch':False})
        composition,states,within,tasks=state_composition(pd.DataFrame(records),['cycle'])
        self.assertEqual(states.set_index('task').loc['MATCH','difference'],2)
        self.assertEqual(states.set_index('task').loc['WRONG_BATCH','status'],'not_estimable')
        self.assertEqual(within.set_index('task').loc['MATCH','status'],'completed')
        for task in ['WRONG_BATCH','WRONG_TYPE','LOW']:
            self.assertEqual(within.set_index('task').loc[task,'status'],'not_estimable')
        self.assertTrue(all(t['probability_correct'] is None and t['confidence_calibration']=='uncalibrated' for t in tasks))

class CrispriPipelineTests(unittest.TestCase):
    def test_full_response_and_annotation_bundle_preserves_source_and_task_scope(self):
        import json
        from unittest.mock import patch
        from rna import scan,value_hash
        from profile_crispri import responses,annotate_context
        with tempfile.TemporaryDirectory() as root:
            root=Path(root);raw=root/'data/raw/fixture.h5ad';raw.parent.mkdir(parents=True)
            rng=np.random.default_rng(42)
            values=rng.poisson(2,size=(64,20)).astype(float);values[40:,0]=0
            genes=[f'ENSG{i}' for i in range(20)]
            h5ad(raw,values,genes=genes)
            with h5py.File(raw,'a') as f:
                for name,labels in {
                    'gene':['non-targeting']*40+['G0']*24,
                    'gene_id':['NTC']*40+['ENSG0']*24,
                    'gene_transcript':['NTC_A']*20+['NTC_B']*20+['P1']*12+['P2']*12,
                    'transcript':['NTC']*40+['P1']*12+['P2']*12,
                    'sgID_AB':['a|b']*20+['c|d']*20+['e|f']*12+['g|h']*12,
                    'gem_group':['1']*64}.items():
                    f['obs'].create_dataset(name,data=labels,dtype=h5py.string_dtype())
            digest=hash_file(raw)
            summary,cells,var=scan(raw,digest,'fixture')
            cells['source_batch']=cells.source_gem_group
            cells['source_task']=cells.source_gene_transcript
            cells['source_target_gene']=cells.source_gene
            cells['source_guide_id']=cells.source_sgID_AB
            structure=root/'structure';directory=structure/'experiment';directory.mkdir(parents=True)
            cells.to_parquet(directory/'cells.parquet',index=False);var.to_parquet(directory/'genes.parquet',index=False)
            pd.DataFrame({'source_gene_id':genes,'mapped_symbol':[f'G{i}' for i in range(20)],
                'symbol_vs_ensembl':['consistent']*20,'many_to_one_mapping':[False]*20}).to_parquet(directory/'gene_mapping.parquet',index=False)
            pd.DataFrame({'task':['NTC_A','NTC_B','P1','P2'],'is_control':[True,True,False,False],
                'source_target_gene':['non-targeting','non-targeting','G0','G0'],
                'source_target_ensembl':['NTC','NTC','ENSG0','ENSG0'],
                'source_transcript':['NTC','NTC','P1','P2'],'source_guide_id':['a|b','c|d','e|f','g|h']}).to_parquet(directory/'tasks.parquet',index=False)
            (structure/'report.json').write_text(json.dumps({'status':'completed',
                'input_sha256':{'data/raw/fixture.h5ad':digest},
                'tables':[{'rows':[{'context':'experiment','expected_lineages':['epithelial']}]}],
                'artifacts':[{'file':str(p.relative_to(structure)),'sha256':hash_file(p)} for p in directory.iterdir()]}))
            references=root/'references';references.mkdir()
            (references/'gene_sets.json').write_text(json.dumps({'profiles':{
                'unsupported':{'lineage':'epithelial','genes':['ABSENT']*12}},
                'states':{'cycle':{'genes':[f'G{i}' for i in range(6)],'reference':'fixture'}},'local_inputs':{}}))
            (references/'SHA256SUMS').write_text(hash_file(references/'gene_sets.json')+'  gene_sets.json\n')
            with patch('profile_crispri.ROOT',root):
                result=responses(structure,'experiment',root/'response',workers=1)
                self.assertEqual(result['expected_tasks'],2);self.assertEqual(result['actual_tasks'],2)
                tasks=json.loads((root/'response/tasks.json').read_text())
                self.assertEqual({t['task'] for t in tasks},{'P1','P2'})
                self.assertTrue(all(t['DE']['status']=='not_estimable' for t in tasks))
                self.assertTrue(all(t['target_RNA']['RNA_ratio']==0 for t in tasks))
                annotation=annotate_context(structure,'experiment',root/'response',references,root/'annotation',chunk=16)
                self.assertEqual(annotation['record_count'],64)
                inferred=pd.read_parquet(root/'annotation/cells.parquet')
                self.assertTrue((inferred.inferred_type=='unknown').all())
                self.assertTrue(inferred.probability_correct.isna().all())
                self.assertEqual(inferred.source_gene_transcript.tolist(),cells.source_gene_transcript.tolist())
                self.assertTrue((root/'annotation/report.html').exists())
                with self.assertRaisesRegex(ValueError,'completed_same_context_response_required'):
                    parent=json.loads((root/'response/report.json').read_text());parent['context']='wrong'
                    (root/'response/report.json').write_text(json.dumps(parent))
                    annotate_context(structure,'experiment',root/'response',references,root/'wrong')
            self.assertEqual(hash_file(raw),digest)


if __name__=='__main__':unittest.main()
