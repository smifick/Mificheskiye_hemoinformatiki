import sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from train_v2 import kernel,stage_fit,stage_predict

class V2Tests(unittest.TestCase):
    def test_count_tanimoto_identity_and_bounds(self):
        X=np.array([[1.,2.,0.],[1.,1.,0.],[0.,0.,1.]])
        K=kernel(X,X)
        np.testing.assert_allclose(K.diagonal(),1)
        np.testing.assert_allclose(K,K.T)
        self.assertTrue(((K>=0)&(K<=1)).all())
    def test_mixture_predictions_do_not_require_inference_labels(self):
        # Synthetic encoded labels; the held-out row has no code or target at inference.
        X=np.array([[1.,0.],[1.,1.],[0.,1.],[1.,2.],[2.,1.],[1.,3.]])
        K=kernel(X,X);F={'tabular':X};ks={'multi':K,'m2':K}
        codes=np.array([0,1,0,1,2,0]);y=np.array([5.,10000.,2.,10000.,0.,3.])
        spec=dict(family='mixture',classifier='krr',regressor='krr_raw')
        fit=stage_fit(spec,np.arange(5),y,codes,F,ks)
        pred=stage_predict(fit,np.array([5]),F,ks)
        self.assertEqual(pred.shape,(1,));self.assertTrue(np.isfinite(pred).all());self.assertGreaterEqual(pred[0],0)
        changed=y.copy();changed[5]=999999.
        fit2=stage_fit(spec,np.arange(5),changed,codes,F,ks)
        np.testing.assert_allclose(pred,stage_predict(fit2,np.array([5]),F,ks))


    def test_centered_constant_target_is_preserved(self):
        X=np.array([[1.,0.],[1.,1.],[0.,1.],[2.,1.]])
        K=kernel(X,X);y=np.array([10.,10.,10.,999.]);codes=np.zeros(4,dtype=int)
        spec=dict(family='centered',key='m2',alpha=1.)
        b=stage_fit(spec,np.arange(3),y,codes,{'tabular':X},{'m2':K})
        np.testing.assert_allclose(stage_predict(b,np.array([3]),{'tabular':X},{'m2':K}),[10.])

    def test_weighted_neighbors_use_training_targets_only(self):
        K=np.array([[1.,.2,.8],[.2,1.,.2],[.8,.2,1.]])
        spec=dict(family='knn',key='m2',neighbors=2,power=2)
        b=stage_fit(spec,np.array([0,1]),np.array([10.,20.,999999.]),np.zeros(3,dtype=int),{}, {'m2':K})
        p=stage_predict(b,np.array([2]),{}, {'m2':K})
        np.testing.assert_allclose(p,[(10*.8**2+20*.2**2)/(.8**2+.2**2)])

if __name__=='__main__':unittest.main()
