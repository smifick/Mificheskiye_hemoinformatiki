"""Checks for validation-label isolation and portable v7 inference."""
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import experiment_v7 as E
import train_v7 as V


class TestV7(unittest.TestCase):
    def test_code_does_not_read_validation_labels(self):
        rng=np.random.default_rng(19)
        x=rng.normal(size=(12,5))
        k=x@x.T+np.eye(12)
        w=SimpleNamespace(y=np.array([0,10000,2,10000,1,3,10000,4,1,2,3,4.]))
        flag=np.array([1,1,0,0,0,0,0,0,1,0,0,0],bool)
        tr=np.arange(8);va=np.arange(8,12)
        p=E.code(w,flag,{}, {'v6':k},tr,va,E.p_specs()[0])
        w.y[va]=[10000,10000,0,36000]
        q=E.code(w,flag,{}, {'v6':k},tr,va,E.p_specs()[0])
        np.testing.assert_array_equal(p,q)
        self.assertTrue(np.all((p>=0)&(p<=1)))

    def test_portable_formula_and_lookup(self):
        ref=V.matrices(['C','CC'])
        model={'feature_weight':.5,'A_reference':ref,'A_mean':.2,
               'A_legacy_dual':np.zeros(2),'A_feature_dual':np.zeros(2),
               'A_flag_probability':.1,'B_reference':ref,'B_mean':np.log(11),
               'B_dual':np.zeros(2),'B_smear':1.,'zero_rate':0.,
               'source_presence':{V.source_key('C')},'lookup':{V.source_key('C'):17.}}
        p,audit=V.infer(model,['C','CC'],details=True)
        np.testing.assert_allclose(p,[17.,2008.])
        np.testing.assert_array_equal(audit['source_present'],[True,False])
        np.testing.assert_array_equal(audit['external_lookup'],[True,False])
        # Averaging probabilities before the v6 clipping/zero-rate step is a
        # different estimator; preserve the complete-prediction ensemble.
        model['A_feature_dual']=np.array([0.,.8])
        model['B_mean']=np.log(11001)
        model['zero_rate']=.1
        q=V.infer(model,['CC'])
        np.testing.assert_allclose(q,[9850.])


if __name__=='__main__':unittest.main()
