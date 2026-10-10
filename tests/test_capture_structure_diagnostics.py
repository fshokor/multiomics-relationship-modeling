import unittest
import numpy as np
import pandas as pd
from src.capture_structure_diagnostics import (
    matched_groups, balanced_indices, conditional_capture_effects,
    learn_capture_basis, project_out, equal_stratum_weights, classifier_metrics)


class CaptureTests(unittest.TestCase):
    def setUp(self):
        self.obs=pd.DataFrame({'DonorID':np.repeat(['a','b'],200), 'Site':['s']*400,
                              'cell_type_harmonized':['T']*400, 'assay':(['CITE']*100+['Multiome']*100)*2})
        rng=np.random.default_rng(42)
        self.z=rng.normal(size=(400,5))
        self.z[:,0]+=self.obs.assay.eq('CITE').to_numpy()*5

    def test_matching_and_balance(self):
        mask,_=matched_groups(self.obs)
        self.assertTrue(mask.all())
        ids=balanced_indices(self.obs,42,cap=70)
        self.assertEqual(len(ids),280)
        self.assertTrue(self.obs.iloc[ids].groupby(['DonorID','assay']).size().eq(70).all())
        changed=self.obs.copy()
        changed.loc[:99,'Site']='unmatched'
        mask,_=matched_groups(changed)
        self.assertFalse(mask[:200].any())
        self.assertTrue(mask[200:].all())

    def test_stratum_adjustment_recovers_capture_effect(self):
        shifted=self.z.copy()
        shifted[200:]+=100
        a,_=conditional_capture_effects(self.z,self.obs)
        b,_=conditional_capture_effects(shifted,self.obs)
        np.testing.assert_allclose(a.common_capture_beta,b.common_capture_beta,atol=1e-10)
        np.testing.assert_allclose(a.common_capture_partial_r2,b.common_capture_partial_r2,atol=1e-10)
        self.assertGreater(a.common_capture_partial_r2.iloc[0],.8)

    def test_projection_basis_is_orthonormal_and_uses_training_only(self):
        basis=learn_capture_basis(self.z[:200],self.obs.iloc[:200],3)
        np.testing.assert_allclose(basis@basis.T,np.eye(3),atol=1e-10)
        result=project_out(self.z[200:],basis)
        np.testing.assert_allclose(result@basis.T,0,atol=1e-10)
        np.testing.assert_allclose(project_out(result,basis),result,atol=1e-10)
        original=self.z.copy()
        project_out(self.z,basis)
        np.testing.assert_array_equal(original,self.z)

    def test_equal_stratum_weight(self):
        part=self.obs.drop(index=range(50))
        sums=part.assign(w=equal_stratum_weights(part)).groupby(['DonorID','assay']).w.sum()
        np.testing.assert_allclose(sums,sums.iloc[0])

    def test_reversed_capture_scores_are_not_information_removal(self):
        class Identity:
            def transform(self,x): return x
        class Reversed:
            def predict_proba(self,x):
                p=x[:,0]
                return np.column_stack([1-p,p])
        z=self.obs.assay.eq('Multiome').to_numpy(float)[:,None]
        metrics=classifier_metrics(Identity(),Reversed(),z,self.obs)
        self.assertEqual(metrics['auroc'],0)
        self.assertEqual(metrics['orientation_free_auroc_descriptive'],1)


if __name__=='__main__':
    unittest.main()
