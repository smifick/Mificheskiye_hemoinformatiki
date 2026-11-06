import sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from two_component_v4 import fit_A,predict_A,fit_B,predict_B,combine

class TwoComponentTests(unittest.TestCase):
 def setUp(self):
  self.X=np.random.default_rng(17).normal(size=(18,8));self.groups=np.arange(18);self.tr=np.arange(12)
  self.y=np.where(np.arange(18)%2,10000.,np.arange(18)+1.)
 def test_soft_mixture_endpoints(self):
  np.testing.assert_allclose(combine(np.array([0.,1.,.5]),np.array([3.,7.,100.])),[3.,10000.,5050.])
 def test_A_ignores_validation_responses(self):
  b=fit_A(self.X,self.y,self.groups,self.tr);p=predict_A(b,self.X[12:])
  changed=self.y.copy();changed[12:]=10000
  p2=predict_A(fit_A(self.X,changed,self.groups,self.tr),self.X[12:])
  np.testing.assert_allclose(p,p2);self.assertTrue(((p>=0)&(p<=1)).all())
 def test_log_calibration_ignores_validation_responses(self):
  X=np.abs(self.X);dot=X@X.T;K=dot/((X*X).sum(1)[:,None]+(X*X).sum(1)[None,:]-dot)
  targets=np.arange(18,dtype=float)+1
  a=fit_B(K,targets,self.groups,self.tr,np.array([],dtype=int),0.,'log_calibrated')
  changed=targets.copy();changed[12:]=100000
  b=fit_B(K,changed,self.groups,self.tr,np.array([],dtype=int),0.,'log_calibrated')
  np.testing.assert_allclose(a['correction'],b['correction']);np.testing.assert_allclose(predict_B(a,K[12:]),predict_B(b,K[12:]))
if __name__=='__main__':unittest.main()
