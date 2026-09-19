from pathlib import Path
import sys
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'dossier'))
from chemical_diagnostics import distribution_contrast


class ChemicalDiagnosticTests(unittest.TestCase):
    def test_missing_controls_and_disjoint_small_control_draws(self):
        rng=np.random.default_rng(8);target=rng.normal(size=(25,3));control=rng.normal(size=(12,3));target[:,2]=np.nan;control[:,2]=np.nan
        a,states,comp,within,draws=distribution_contrast(target,control,['unknown']*25,['unknown']*12,['a','b','missing'],19)
        self.assertEqual(a['complete_dimensions'],2);self.assertEqual(len(draws),20)
        self.assertTrue(all(r['null_reference_intersection']==0 and r['null_arm_cells']==6 and r['null_target_cells_not_matched']==19 for r in draws))
        self.assertEqual(states[2]['status'],'not_estimable');self.assertEqual(comp[0]['fraction_difference'],0)
        self.assertIsNone(within[2]['mean_difference'])
        b,*_=distribution_contrast(target,np.empty((0,3)),['unknown']*25,[],['a','b','missing'],19)
        self.assertEqual(b['status'],'not_estimable')


if __name__=='__main__':unittest.main()
