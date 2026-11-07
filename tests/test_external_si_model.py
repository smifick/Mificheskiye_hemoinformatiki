"""Checks for label isolation and the explicit multi-output regression."""
import copy,sys,unittest
import tempfile
import joblib
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import external_si_model as M
import train_v5 as T
from prepare_external_training import select_auxiliary_rows

class ExternalModelTests(unittest.TestCase):
    def test_auxiliary_sampling_does_not_depend_on_measurement_values(self):
        data=pd.DataFrame({'activity_id':list(range(12)),'endpoint':['a']*6+['b']*6,'value_uM':np.arange(12)+1.,'measurement_key':[str(x) for x in range(12)]})
        expected=select_auxiliary_rows(data,2).activity_id.tolist()
        data['value_uM']=1e90;data['measurement_key']='changed_values'
        self.assertEqual(expected,select_auxiliary_rows(data,2).activity_id.tolist())
    def test_stale_organizer_cache_is_rejected_before_training(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'data/raw').mkdir(parents=True);(root/'results/audit_20261107').mkdir(parents=True)
            old=pd.DataFrame({'SMILES':['C','CC'],'SI VACV':[2.,4.]})
            test=pd.DataFrame({'SMILES':['CCC'],'index':[0]})
            joblib.dump({'train':old,'test':test,'y':old['SI VACV'].to_numpy()},root/'results/audit_20261107/state_cache.joblib')
            changed=old.copy();changed.loc[0,'SMILES']='N'
            changed.to_csv(root/'data/raw/train.csv',index=False);test.to_csv(root/'data/raw/test.csv',index=False)
            with self.assertRaisesRegex(ValueError,'Stale organizer state'):
                M.load_context(root,root/'unread_bank.csv')
    def context(self):
        # A known PSD kernel makes this test independent of chemistry parsing.
        x=np.arange(9,dtype=float)[:,None]
        k=np.exp(-.1*(x-x.T)**2)
        bank=pd.DataFrame({'SMILES':['C','CC','CCC'],'value':[9.,20.,3.],
            'source':['physical','aux','aux'],'task':['SI','aux','aux'],'sign':[1.,-1.,-1.],
            'tautomer_parent':['ext','blocked','novel'],'scaffold':['other','g2','auxnew']})
        return dict(n=6,y=np.array([2.,4.,8.,16.,10000.,0.]),K=k,bank=bank,
            org_keys=np.array(['a','b','c','d','e','f']),org_scaffolds=np.array(['g0','g1','g2','g3','g4','g5']))
    def test_single_output_matches_weighted_kernel_ridge(self):
        ctx=self.context();tr=np.array([0,1,2,3,4,5]);cfg={'source_weights':{'physical':.5},'rho':0.,'alpha':3.}
        core=M.fit_core(ctx,tr,cfg,[]);ids=np.array([0,1,2,3,6]);values=np.array([2.,4.,8.,16.,9.]);weights=np.array([1.,1.,1.,1.,.5])
        expected=T.wkrr(ctx['K'],ids,np.log1p(values),weights,np.array([0,2]),3.)
        np.testing.assert_allclose(M.predict_log(ctx,core,[0,2]),expected,rtol=1e-12,atol=1e-12)
    def test_heldout_organizer_labels_do_not_enter_core(self):
        ctx=self.context();cfg={'source_weights':{'physical':1.,'aux':.25},'rho':.5}
        before=M.predict_log(ctx,M.fit_core(ctx,[0,1,3,4,5],cfg,[2]),[2])
        altered=copy.deepcopy(ctx);altered['y'][2]=1e100
        after=M.predict_log(altered,M.fit_core(altered,[0,1,3,4,5],cfg,[2]),[2])
        np.testing.assert_array_equal(before,after)
    def test_blocked_external_values_do_not_enter_core(self):
        ctx=self.context();cfg={'source_weights':{'physical':1.,'aux':.25},'rho':.5}
        before=M.predict_log(ctx,M.fit_core(ctx,[0,1,3,4,5],cfg,[2]),[2])
        ctx['bank'].loc[1,'value']=1e100
        after=M.predict_log(ctx,M.fit_core(ctx,[0,1,3,4,5],cfg,[2]),[2])
        np.testing.assert_array_equal(before,after)
    def test_blocked_scaffold_excludes_different_structure(self):
        ctx=self.context();self.assertFalse(M.allowed_bank(ctx,[2])[1])
        self.assertTrue(M.allowed_bank(ctx,[2])[2])
    def test_zero_coupling_cannot_turn_auxiliary_into_SI(self):
        ctx=self.context();cfg={'source_weights':{'physical':1.,'aux':1.},'rho':0.}
        core=M.fit_core(ctx,[0,1,2,3],cfg,[])
        self.assertEqual(core['auxiliary_rows'],0);self.assertEqual(core['physical_external_rows'],1)
    def test_signed_task_kernel_is_positive_semidefinite(self):
        tasks=np.array(['main','a','a','b']);sign=np.array([1.,-1.,-1.,1.])
        for rho in [.25,.5]:
            gram=(1-rho)*(tasks[:,None]==tasks[None,:])+rho*sign[:,None]*sign[None,:]
            self.assertGreater(np.linalg.eigvalsh(gram).min(),-1e-12)

if __name__=='__main__':unittest.main()
