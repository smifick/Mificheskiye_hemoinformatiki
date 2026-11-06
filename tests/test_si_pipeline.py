"""Data pairing, feature consistency, and leakage isolation checks."""
import tempfile
from pathlib import Path
import unittest
import joblib
import numpy as np
import pandas as pd

from si_pipeline.chemistry import canonicalize,structure_keys
from si_pipeline.collect import ChEMBL,to_uM,clean_activity,collect_pairs,validate_pair
from si_pipeline.features import MolecularFeatures
from si_pipeline.modeling import fit_model,predict_model
from si_pipeline.predict import predict_frame
from si_pipeline.train import select_external,grouped_folds


def activity(endpoint,value,unit='nM',relation='=',smiles='CCO',aid=1):
    return dict(standard_type=endpoint,standard_value=value,standard_units=unit,
                standard_relation=relation,canonical_smiles=smiles,activity_id=aid,
                document_chembl_id='CHEMBL_DOC',data_validity_comment=None)


class FakeChEMBL:
    def get(self,key,params=None):
        if key.startswith('document/'):
            return {'doi':None}
        return {'target_chembl_id':'CHEMBL_TARGET','document_chembl_id':'CHEMBL_DOC',
                'assay_cell_type':'Example cells','cell_chembl_id':'CHEMBL_CELL'}

    def pages(self,resource,key,**params):
        if params['standard_type']=='IC50':
            return iter([activity('IC50',1000,aid=1),activity('IC50',4000,aid=2),
                         activity('IC50',1,relation='>',aid=3),activity('IC50',2,smiles='invalid!',aid=4)])
        return iter([activity('CC50',100000,aid=5)])


class TestCollection(unittest.TestCase):
    def test_unit_conversion_preserves_case(self):
        self.assertEqual(to_uM(1000,'nM'),1.)
        self.assertEqual(to_uM(1,'mM'),1000.)
        self.assertEqual(to_uM(1,'M'),1000000.)
        self.assertEqual(to_uM(1,'µM'),1.)
        with self.assertRaises(ValueError):to_uM(1,'ug/mL')
        with self.assertRaises(ValueError):to_uM(float('inf'),'uM')

    def test_censoring_and_wrong_endpoint_are_rejected(self):
        with self.assertRaises(ValueError):clean_activity(activity('IC50',3,relation='>'),'IC50')
        with self.assertRaises(ValueError):clean_activity(activity('EC50',3),'IC50')

    def test_reviewed_pair_and_replicates(self):
        manifest=pd.DataFrame([dict(context_id='ctx',target_chembl_id='CHEMBL_TARGET',
            ic50_assay_id='CHEMBL_IC',cc50_assay_id='CHEMBL_CC',cell_line='Example',
            ic50_hours=48,cc50_hours=48,evidence_url='https://example.org/source')])
        dataset,provenance,rejected=collect_pairs(FakeChEMBL(),manifest)
        self.assertEqual(list(dataset),['smiles','IC50_uM','CC50_uM','SI','doi'])
        self.assertEqual(len(dataset),1)
        self.assertAlmostEqual(dataset.IC50_uM.iloc[0],2.)
        self.assertAlmostEqual(dataset.SI.iloc[0],50.)
        self.assertEqual(dataset.doi.iloc[0],'')
        self.assertEqual(len(rejected),2)
        self.assertIn('geometric mean',provenance.replicate_aggregation.iloc[0])
        manifest.loc[0,'cc50_hours']=72
        data,_,rejected=collect_pairs(FakeChEMBL(),manifest)
        self.assertTrue(data.empty)
        self.assertIn('incompatible',rejected.reason.iloc[0])
        manifest['pairing_policy']='author_defined_si'
        manifest['reported_si_verified']=True
        data,_,_=collect_pairs(FakeChEMBL(),manifest)
        self.assertEqual(len(data),1)

    def test_pair_different_documents_is_rejected(self):
        row=dict(context_id='ctx',target_chembl_id='T',ic50_assay_id='A',cc50_assay_id='B',
                 cell_line='Example',ic50_hours=48,cc50_hours=48,evidence_url='doi:10.1/x')
        a=dict(target_chembl_id='T',document_chembl_id='DOC_A')
        b=dict(document_chembl_id='DOC_B')
        with self.assertRaisesRegex(ValueError,'documents'):validate_pair(row,a,b)

    def test_pagination_and_cache(self):
        import json
        from types import SimpleNamespace
        pages=[{'activities':[{'id':1}],'page_meta':{'next':'/chembl/api/data/activity.json?offset=1'}},
               {'activities':[{'id':2}],'page_meta':{'next':None}}]
        calls=[]
        with tempfile.TemporaryDirectory() as d:
            client=ChEMBL(d)
            def request(url,params=None,timeout=None):
                i=len(calls);calls.append(url);payload=pages[i]
                return SimpleNamespace(content=json.dumps(payload).encode(),raise_for_status=lambda:None,json=lambda:payload)
            client.session.get=request
            self.assertEqual(list(client.pages('activity.json','activities')),[{'id':1},{'id':2}])
            self.assertEqual(list(client.pages('activity.json','activities')),[{'id':1},{'id':2}])
            self.assertEqual(len(calls),2)
            self.assertTrue(client.log[-1]['cache_hit'])


class TestModeling(unittest.TestCase):
    def test_canonical_equivalent_smiles_features(self):
        f=MolecularFeatures(morgan_bits=128).fit(['CCO'])
        x=f.transform(['CCO','OCC'])
        np.testing.assert_allclose(x[0],x[1],equal_nan=True)
        self.assertEqual(x.shape[1],len(f.get_feature_names_out()))
        self.assertEqual(canonicalize('CCO.Cl'),canonicalize('CCO'))

    def test_mordred_roundtrip(self):
        try:import mordred
        except ImportError:self.skipTest('Optional mordredcommunity not installed')
        f=MolecularFeatures(morgan_bits=16,mordred=True).fit(['CCO'])
        x=f.transform(['CCO'])
        self.assertGreater(len(f.mordred_names_),1000)
        self.assertEqual(x.shape[1],len(f.feature_names_))

    def test_validation_labels_do_not_change_fit(self):
        x=np.arange(36,dtype=float).reshape(12,3)
        y=np.array([10000,3,7,0,10000,2,5,10,9,7,8,6.])
        spec=dict(family='ridge',external_weight=0.)
        tr=np.arange(8);va=np.arange(8,12)
        ex=np.empty((0,3));ey=np.array([]);keep=np.zeros(0,bool)
        a=fit_model(spec,x,y,tr,ex,ey,keep)
        changed=y.copy();changed[va]=[10000,10000,0,50000]
        b=fit_model(spec,x,changed,tr,ex,ey,keep)
        np.testing.assert_allclose(predict_model(a,x[va]),predict_model(b,x[va]))

    def test_external_holdout_scaffolds_and_parents_blocked(self):
        keep=select_external(np.array(['g0','g1','g9']),np.array(['x','y','p0']),
                             np.array(['g0','g1']),np.array(['p0','p1']),np.array([0]))
        np.testing.assert_array_equal(keep,[False,True,False])

    def test_external_si_never_labels_inactivity_classifier(self):
        x=np.arange(24,dtype=float).reshape(8,3)
        y=np.array([10000,3,7,0,10000,2,5,10.])
        ex=np.array([[10.,20.,30.]])
        spec=dict(family='ridge',external_weight=.25)
        a=fit_model(spec,x,y,np.arange(8),ex,np.array([2.]),np.ones(1,bool))
        b=fit_model(spec,x,y,np.arange(8),ex,np.array([10000.]),np.ones(1,bool))
        np.testing.assert_allclose(a['classifier'].predict_proba(x),b['classifier'].predict_proba(x))

    def test_physical_mode_preserves_real_si_10000(self):
        x=np.arange(12,dtype=float).reshape(4,3)
        y=np.array([2.,4.,10000.,8.])
        model=fit_model(dict(family='ridge',external_weight=0.),x,y,np.arange(4),
                        np.empty((0,3)),np.array([]),np.zeros(0,bool),mode='ratio')
        self.assertIsNone(model['classifier'])
        self.assertGreater(model['target_scale'],1000.)

    def test_scaffold_folds_are_disjoint(self):
        smiles=['c1ccccc1','Cc1ccccc1','C1CCCCC1','CC1CCCCC1','c1ccc2ccccc2c1','Cc1ccc2ccccc2c1',
                'C1CC1','CC1CC1','C1CCC1','CC1CCC1']
        groups=np.array([structure_keys(s)[2] for s in smiles])
        for tr,va in grouped_folds(np.arange(len(smiles)),groups,3,42):
            self.assertTrue(set(groups[tr]).isdisjoint(groups[va]))

    def test_inference_ignores_all_forbidden_columns_and_reloads(self):
        smiles=pd.Series(['CCO','CCC','CCN','CCCl','CCBr','CCF','CCCO','CCCN'])
        f=MolecularFeatures(morgan_bits=32).fit(smiles)
        x=f.transform(smiles)
        y=np.array([1,2,3,4,5,6,7,8.])
        model=fit_model(dict(family='ridge',external_weight=0.),x,y,np.arange(len(y)),
                        np.empty((0,x.shape[1])),np.array([]),np.zeros(0,bool),mode='ratio')
        bundle=dict(format_version=1,featurizer=f,model=model,smiles_col='SMILES',target_col='SI VACV')
        clean=pd.DataFrame({'index':[4,3],'SMILES':['CCO','CCN']})
        poisoned=clean.assign(CC50_uM=[1e99,0],IC50_uM_VACV=[0,1e-99],strain=['changed','changed'],Culture=['x','y'])
        a=predict_frame(bundle,clean)
        b=predict_frame(bundle,poisoned)
        pd.testing.assert_frame_equal(a,b)
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'model.joblib';joblib.dump(bundle,path)
            pd.testing.assert_frame_equal(a,predict_frame(joblib.load(path),clean))
        sample=pd.DataFrame({'index':[3,4],'SI VACV':[999,999]})
        self.assertEqual(predict_frame(bundle,clean,sample=sample)['index'].tolist(),[3,4])


if __name__=='__main__':unittest.main()
