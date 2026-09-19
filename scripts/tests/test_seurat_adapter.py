"""Real RDS serialization tests for count-layer choice and cell-axis validation."""
from pathlib import Path
import os
import subprocess
import sys
import tempfile
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'dossier'))
from convert_seurat_cache import RROOT,make_h5ad
from rna import RNAFile,hash_file

class SeuratAdapterTests(unittest.TestCase):
    def test_counts_preserved_and_axis_or_label_errors_rejected(self):
        env=os.environ.copy();env['R_HOME']=str(RROOT);env['LD_LIBRARY_PATH']=str(RROOT/'lib')+':'+str(RROOT.parent)
        r=['/lib/ld-linux-aarch64.so.1',str(RROOT/'bin/exec/R'),'--vanilla','--slave']
        script=Path(__file__).resolve().parents[1]/'dossier/export_seurat_readonly.R'
        with tempfile.TemporaryDirectory() as root:
            root=Path(root)
            fixture=r'''setClass('dgCMatrix',slots=c(x='numeric',i='integer',p='integer',Dim='integer',Dimnames='list'))
setClass('Assay',slots=c(counts='dgCMatrix',data='dgCMatrix',scale.data='matrix'))
setClass('Seurat',slots=c(assays='list',meta.data='data.frame'))
x<-new('dgCMatrix',x=c(2,7,1),i=as.integer(c(0,1,1)),p=as.integer(c(0,2,3)),Dim=as.integer(c(2,2)),Dimnames=list(c('G1','G2'),c('01','02')))
y<-x;y@x<-log1p(y@x)
m<-data.frame(cell_type=c('A','A'),pathway=c('IFNB','IFNB'),Batch_info=c('Rep1','Rep1'),guide=c('001','002'),gene=c('NT','G1'),row.names=c('01','02'))
z<-new('Seurat',assays=list(RNA=new('Assay',counts=x,data=y,scale.data=matrix(numeric(),0,0))),meta.data=m)
saveRDS(z,'good.rds')
rownames(z@meta.data)<-c('02','01');saveRDS(z,'wrong-axis.rds')
z@meta.data<-m;z@meta.data$guide[[1]]<-NA_character_;saveRDS(z,'missing-label.rds')'''
            subprocess.run(r+['-e',fixture],env=env,cwd=root,check=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
            original=hash_file(root/'good.rds')
            for name in ['good','wrong-axis','missing-label']:
                result=subprocess.run(r+['-f',str(script),'--args',str(root/(name+'.rds')),str(root/name)],env=env,text=True,capture_output=True)
                if name=='good':self.assertEqual(result.returncode,0,result.stderr)
                else:
                    self.assertNotEqual(result.returncode,0)
                    self.assertIn('cell_axis_metadata_mismatch' if name=='wrong-axis' else 'missing_source_label',result.stderr)
            report=make_h5ad(root/'good',root/'counts.h5ad')
            self.assertTrue(report['all_values_exactly_preserved'])
            self.assertGreater(report['layer_audit'][1]['noninteger'],0)
            with RNAFile(root/'counts.h5ad') as source:
                np.testing.assert_array_equal(next(source.blocks())[1].toarray(),[[2,7],[0,1]])
                self.assertEqual(source.obs.index.tolist(),['01','02'])
                self.assertEqual(source.obs.guide.tolist(),['001','002'])
            self.assertEqual(original,hash_file(root/'good.rds'))

if __name__=='__main__':unittest.main()
