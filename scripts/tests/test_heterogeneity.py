import unittest
import numpy as np
from dossier.heterogeneity import compare_vectors,endpoint_cycle_partition,mixture_identity


class HeterogeneityTests(unittest.TestCase):
    def test_missing_genes_not_zero_and_constant_correlation_not_zero(self):
        result=compare_vectors([1,np.nan,3],[1,2,5],minimum=3)
        self.assertEqual(result['status'],'not_estimable');self.assertEqual(result['genes'],2)
        result=compare_vectors([2,2,2],[1,2,3],minimum=3)
        self.assertIsNone(result['correlation']);self.assertGreater(result['difference_RMS'],0)

    def test_pure_composition_and_pure_within_are_separate(self):
        cm=np.array([[0.,2.],[10.,12.]])
        result,meta=mixture_identity(cm,cm,[25,75],[50,50],['a','a'])
        np.testing.assert_allclose(result['total'],[2.5,2.5]);np.testing.assert_allclose(result['composition'],[2.5,2.5]);np.testing.assert_allclose(result['within'],0)
        result,meta=mixture_identity(cm+2,cm,[25,75],[50,50],['a','a'])
        np.testing.assert_allclose(result['total'],[4.5,4.5]);np.testing.assert_allclose(result['composition'],[2.5,2.5]);np.testing.assert_allclose(result['within'],2)

    def test_unobserved_control_state_is_excluded_with_support_loss(self):
        result,meta=mixture_identity([[4.,5.],[9.,10.]],[[3.,4.],[np.nan,np.nan]],[12,88],[5,0],['a','a'])
        self.assertEqual(meta['supported_target_cells'],12);self.assertEqual(meta['supported_target_fraction'],.12)
        np.testing.assert_allclose(result['total'],[1,1]);np.testing.assert_allclose(result['composition'],0)
        result,meta=mixture_identity([[4.],[9.]],[[3.],[np.nan]],[2,98],[5,0],['a','a'])
        self.assertIsNone(result);self.assertEqual(meta['status'],'not_estimable')

    def test_cycle_boundaries_use_controls_and_do_not_invent_equal_quantiles(self):
        labels,meta=endpoint_cycle_partition([0,1,2,100],[0]*4,[True,True,True,False])
        np.testing.assert_allclose(meta['boundaries'],[2/3,4/3]);self.assertEqual(labels[-1],'high_cycle_RNA_proxy')
        labels,meta=endpoint_cycle_partition([1,1,1],[0,0,0],[True]*3)
        self.assertEqual(meta['status'],'not_estimable');self.assertTrue((labels=='unknown_cycle_proxy').all())
