import sys
from pathlib import Path
import unittest
import numpy as np
from scipy import sparse
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'dossier'))
from annotation import annotate,marker_model
from sparse_annotation import annotate_sparse


class SparseAnnotationTests(unittest.TestCase):
    def test_dense_equivalence_including_zeros_ties_unmapped_and_target_removal(self):
        rng=np.random.default_rng(5);genes=['G'+str(i) for i in range(80)]+[None,'duplicate','duplicate']
        profiles={str(i):{'lineage':str(i//2),'genes':['G'+str(j) for j in range(i*12,i*12+25)]} for i in range(4)}
        model=marker_model(genes,profiles);x=rng.integers(0,4,size=(101,len(genes))).astype(np.float32);x[rng.random(x.shape)<.8]=0
        x[0]=0;x=np.log1p(x);targets=['G'+str(i%80) for i in range(len(x))]
        a,sa=annotate(x,targets,model);b,sb=annotate_sparse(sparse.csr_matrix(x),targets,model)
        for name in sa:np.testing.assert_allclose(sa[name],sb[name],atol=2e-6,rtol=2e-6)
        for name in ['inferred_type','inferred_lineage','inference_reason','target_excluded_type','target_marker_label_changed']:
            self.assertEqual(a[name].tolist(),b[name].tolist())
        self.assertTrue(b.probability_correct.isna().all())

    def test_constant_vector_has_zero_expression_score(self):
        model=marker_model(['G'+str(i) for i in range(80)],{'p':{'lineage':'p','genes':['G'+str(i) for i in range(25)]}})
        _,scores=annotate_sparse(sparse.csr_matrix(np.full((1,80),np.log(2),np.float32)),['none'],model)
        self.assertEqual(scores['expression__p'][0],0)

    def test_negative_expression_is_not_treated_as_implicit_zero(self):
        model=marker_model(['g'],{'p':{'lineage':'p','genes':['g']}})
        with self.assertRaises(ValueError):annotate_sparse(sparse.csr_matrix([[-1]]),['g'],model)


if __name__=='__main__':unittest.main()
