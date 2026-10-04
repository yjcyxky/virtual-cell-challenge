"""Check frozen kNN inference and the full unsubmitted prediction diagnostic path."""
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'src'),str(ROOT/'scripts'),str(ROOT/'scripts/dossier')]
from vcc_task import common
from vcc_task.frozen_knn import predict_knn
from vcc_task.depmap_submission import generate_context, assemble, legacy_io
from vcc_task.official_diagnostics import diagnose_official


def checkpoint():
    support = np.asarray([[0,1,1,0,0,0],[1,0,1,1,0,0]],bool)
    return {'mode':'knn','model_state':{'complete':True},
        'response_gauge':'zero_mean_over_source_observed_readouts_excluding_own',
        'axis':np.asarray(['A','B','C','D','E','F']), 'shared_labels':['A','B'],
        'shared_values':np.asarray([[0,.2,.4,0,0,0],[.1,0,.3,.2,0,0]],np.float32),
        'shared_support_bits':np.packbits(support),'shared_support_size':support.size,
        'source_template':np.asarray([-.2,.1,0,.1,0,0],np.float32),
        'observed_readouts':support.any(0),'source_contexts':{'A':['source1'],'B':['source2']},
        'decoder':{'own_residual':.2,'total':50000},
        'inference_descriptors':{'symbols':np.asarray(['A','B','C']),
                                 'raw':np.asarray([[1,0],[0,1],[.99,.01]],np.float32)}}


class FrozenSubmissionTests(unittest.TestCase):
    def test_three_context_packaging_preserves_contexts_targets_and_counts(self):
        _, helper = legacy_io()
        genes = checkpoint()['axis'].tolist()
        cfg = {'prediction_cells':8,'max_nnz':10000,'max_counts_per_cell':1000000,
               'checkpoint_ref':{'path':'model.pt','sha256':'0'*64},'fit_scope':{'outer_split':'S2-H1'}}
        with tempfile.TemporaryDirectory() as temp, patch.object(common,'ROOT',Path(temp)):
            directory = Path(temp)
            for i, context in enumerate('ABC'):
                (directory/context).mkdir()
                obs = pd.DataFrame({'context':context,'target_gene':np.repeat(['A','B'],8)},
                                   index=[f'{context}-{j}' for j in range(16)])
                ad.AnnData(sparse.csr_matrix(np.full((16,6),i+1,dtype='i4')),obs=obs,
                           var=pd.DataFrame(index=genes)).write_h5ad(directory/context/'counts.h5ad')
            result = assemble(cfg,directory,None,genes,['A','B'],helper)
            self.assertEqual(result['audit']['cells'],48)
            data = ad.read_h5ad(directory/'predictions.h5ad')
            self.assertEqual(data.obs.context.astype(str).tolist(),list(np.repeat(list('ABC'),16)))
            np.testing.assert_array_equal(np.asarray(data.X.sum(1)).ravel(),np.repeat([6,12,18],16))
            self.assertEqual(assemble(cfg,directory,None,genes,['A','B'],helper),result)

    def test_nearest_response_masks_and_missing_prior_template(self):
        values, coverage = predict_knn(checkpoint(),['A','C','F'],1)
        self.assertEqual(coverage['C']['neighbor_targets'],['A'])
        self.assertFalse(coverage['C']['seen_target'])
        self.assertTrue(coverage['C']['functional_extrapolation'])
        self.assertTrue(coverage['F']['missing_prior_template_fallback'])
        np.testing.assert_array_equal(values[:,4:],0)
        self.assertEqual(values[0,0],0)
        self.assertEqual(values[1,2],0)
        # An absent gene descriptor uses the already centered template.
        np.testing.assert_allclose(values[2,:4],[-.2,.1,0,.1],atol=1e-7)
        self.assertAlmostEqual(float(values[0,:4].sum()),0,places=6)

    def test_context_generation_diagnostics_and_unchanged_resume(self):
        cfg={'run_id':'fixture','seed':930,'model':{'neighbors':1},'fit_scope':{'outer_split':'S2-H1'},
             'prediction_cells':8,'max_nnz':100000,'decoder':{'own_residual':.2,'total':50000},
             'emitter':{'iterations':128,'smoothing':1e-6,'tolerance':1e-7},
             'runtime':{'num_threads':2},
             'diagnostics':{'repeat_targets':2,'generation_repeats':3,'null_repeats':2,
                            'ntc_sampling_repeats':2,'covariance_genes':3}}
        rng=np.random.default_rng(88)
        ntc=ad.AnnData(sparse.csr_matrix(rng.poisson(5,size=(64,6)).astype('i4')),
            obs=pd.DataFrame({'context':'A','target_gene':'non-targeting'},index=[f'n{i}' for i in range(64)]),
            var=pd.DataFrame(index=checkpoint()['axis']))
        with tempfile.TemporaryDirectory() as temp, patch.object(common,'ROOT',Path(temp)):
            directory=Path(temp)/'A'; directory.mkdir()
            ref={'path':'frozen.pt','sha256':'0'*64}
            generated=generate_context(cfg,checkpoint(),['A','B'],ntc,'A',directory,ref)
            result=diagnose_official(directory/'counts.h5ad',ntc,directory/'diagnostics',
                ['A','B'],cfg,ref,generated['coverage'],checkpoint()['observed_readouts'])
            self.assertTrue(result['hard_constraints_passed'])
            self.assertEqual(result['null']['added_zero_de'],[0,0])
            self.assertEqual(len(generated['independent_repeats']),2)
            repeated=diagnose_official(directory/'counts.h5ad',ntc,directory/'diagnostics',
                ['A','B'],cfg,ref,generated['coverage'],checkpoint()['observed_readouts'])
            self.assertEqual(result,repeated)
            actual=ad.read_h5ad(directory/'counts.h5ad')
            self.assertEqual(actual.shape,(16,6))
            self.assertEqual(set(actual.obs.context),{'A'})
            self.assertEqual(actual.obs.target_gene.value_counts().to_dict(),{'A':8,'B':8})


if __name__=='__main__':
    unittest.main()
