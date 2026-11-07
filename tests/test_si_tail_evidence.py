import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import joblib
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import si_tail_evidence as evidence
from curate_si_tail_evidence import number, table_structure
from rdkit import Chem


def synthetic_features(n=30):
    rng=np.random.default_rng(87)
    x=rng.integers(0,2,size=(n,8)).astype(float)
    return {'fingerprints':x, 'torsion':x.copy(), 'feature':x.copy(),
            'descriptors':rng.normal(size=(n,4))}


class EvidenceTests(unittest.TestCase):
    def test_support_projection_only_removes_negative_values(self):
        class Fixed:
            def predict(self,features):
                return np.array([-10.,5.,100000.])
        model=evidence.NonnegativeProjection(Fixed())
        pred=model.predict({})
        np.testing.assert_array_equal(pred,[0.,5.,100000.])
        y=np.array([0.,10.,120000.])
        self.assertLessEqual(np.mean((pred-y)**2),np.mean((Fixed().predict({})-y)**2))

    def test_log1p_retransformation_uses_shift_and_has_no_upper_cap(self):
        x=synthetic_features(3)
        model=evidence.RegimeSmearing('global')
        model.gate_reference_=x
        model.code_mean_=model.feature_mean_=0.
        model.code_dual_=model.feature_dual_=np.zeros(3)
        model.zero_conditional_rate_=0.
        model.physical_reference_=x
        model.physical_mean_=np.log(11.)
        model.physical_dual_=np.zeros(3)
        model.global_factor_=3.
        prediction,detail=model.predict_details(x)
        np.testing.assert_allclose(prediction,32.)  # 3*(10+1)-1, not 3*10
        model.physical_mean_=np.log(200001.)
        self.assertGreater(model.predict(x).min(),500000.)

    def test_glm_can_extrapolate_without_an_arbitrary_multiplier(self):
        x=np.linspace(0,1,40)[:,None]
        y=np.exp(2*x[:,0])-1
        model=evidence.MeanGLM(power=2.,alpha=.001).fit({'descriptors':x},y)
        prediction=model.predict({'descriptors':np.array([[2.]])})[0]
        self.assertGreater(prediction, y.max()*2)

    def test_glm_preprocessing_and_response_scale_are_training_only(self):
        features=synthetic_features()
        y=np.exp(features['descriptors'][:,0])+1
        model=evidence.MeanGLM(alpha=10).fit(evidence.subset(features,np.arange(20)),y[:20])
        np.testing.assert_allclose(model.pipeline_[1].mean_,features['descriptors'][:20].mean(axis=0))
        self.assertAlmostEqual(model.response_scale_,(y[:20]+1).mean())
        poisoned=y.copy();poisoned[20:]=1e15
        repeat=evidence.MeanGLM(alpha=10).fit(evidence.subset(features,np.arange(20)),poisoned[:20])
        np.testing.assert_allclose(model.predict(features),repeat.predict(features))

    def test_conditional_smearing_is_positive_and_can_reject_a_sparse_tail_gate(self):
        features=synthetic_features()
        y=np.exp(features['descriptors'][:,0]+2)
        y[::5]=10000
        groups=np.repeat(np.arange(10),3)
        model=evidence.RegimeSmearing('conditional',prior_count=5).fit(features,y,groups)
        pred,detail=model.predict_details(features)
        self.assertTrue(np.isfinite(pred).all())
        self.assertTrue((detail['smearing_factor']>0).all())
        self.assertEqual(model.diagnostics_['physical_n'],24)
        self.assertTrue(((detail['p_code']>=0)&(detail['p_code']<=1)).all())

    def test_external_alias_of_inner_validation_identity_is_excluded(self):
        features=synthetic_features(24)
        features['identity']=np.array([f'test-identity-{i}' for i in range(24)],object)
        y=np.arange(24,dtype=float)+1
        groups=np.repeat(np.arange(8),3)
        ext=evidence.subset(features,np.array([0]))
        # The identity is shared but the external scaffold alias is deliberately wrong.
        first=evidence.RegimeSmearing('global').fit(features,y,groups,
            external=(ext,np.array([100.]),np.array(['different-alias']),.1))
        second=evidence.RegimeSmearing('global').fit(features,y,groups,
            external=(ext,np.array([1e8]),np.array(['different-alias']),.1))
        self.assertAlmostEqual(first.ratios_[0],second.ratios_[0],places=10)

    def test_cached_similarities_and_reloaded_model_have_no_fitted_global_state(self):
        smiles=['CCO','CCN','c1ccccc1','c1ccccc1O','C1CCC1','C1CCCC1',
                'C1CCCCC1','C1CCNCC1','C1CCOCC1','c1ccncc1','C1CC1','C1CCCCCC1']
        features,_,groups=evidence.structures(smiles)
        np.testing.assert_allclose(evidence.kernel(features,features),evidence._direct_kernel(features,features))
        model=evidence.RegimeSmearing('none').fit(features,np.arange(len(smiles))+1.,groups)
        before=model.predict(features)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'model.joblib';joblib.dump(model,path)
            evidence._STRUCTURE_REGISTRY.clear();evidence._cached_kernel.cache_clear()
            after=joblib.load(path).predict(features)
        np.testing.assert_allclose(before,after,rtol=0,atol=1e-10)

    def test_table_bounds_are_not_exact_and_cage_positions_are_distinct(self):
        self.assertEqual(number('>100'),(100.,None,'>'))
        self.assertEqual(number('NA d'),(None,None,None))
        self.assertEqual(number('0.03 ± 0.01'),(.03,.01,'='))
        graphs=[Chem.MolFromSmiles(table_structure(name)) for name in ['12a','12b']]
        degrees=[]
        for mol in graphs:
            n=next(atom for atom in mol.GetAtoms() if atom.GetAtomicNum()==7)
            # Carbonyl carbon has a double-bonded oxygen, leaving one cage neighbor.
            attached=[atom for atom in n.GetNeighbors() if not any(b.GetBondTypeAsDouble()==2 for b in atom.GetBonds())]
            degrees.append(attached[0].GetDegree())
        self.assertEqual(degrees,[4,3])
        self.assertNotEqual(table_structure('18a'),table_structure('18b'))


if __name__=='__main__':
    unittest.main()
