import sys, unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from train_model import standardize,featurize,tanimoto
from download_external import finite_number,duration

class PipelineTests(unittest.TestCase):
    def test_salt_parent_and_charged_equivalent(self):
        self.assertEqual(standardize('CC[NH3+].[Cl-]')[0],standardize('CCN')[0])
    def test_stereoisomers_share_validation_group(self):
        self.assertEqual(standardize('C[C@H]1CCCCO1')[1],standardize('C[C@@H]1CCCCO1')[1])
    def test_invalid_smiles_stops_prediction(self):
        with self.assertRaises(ValueError):standardize('invalid')
    def test_tanimoto_kernel_invariants(self):
        X,_=featurize(['CCO','CCN','c1ccccc1']);K=tanimoto(X,X)
        np.testing.assert_allclose(K,K.T);np.testing.assert_allclose(K.diagonal(),1)
        self.assertTrue(((K>=0)&(K<=1)).all())
    def test_endpoint_and_context_validation(self):
        self.assertIsNone(finite_number('>100'));self.assertIsNone(finite_number('nan'))
        self.assertIsNone(finite_number('-1'));self.assertEqual(finite_number('12.3'),12.3)
        self.assertEqual(duration('after 48 hours'),'48');self.assertEqual(duration('24 h and 48 h'),'')

if __name__=='__main__':unittest.main()
